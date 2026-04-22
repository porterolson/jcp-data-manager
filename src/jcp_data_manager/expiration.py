"""WordPress job expiration workflow."""

from __future__ import annotations

from time import sleep
from typing import Any

import polars as pl
import requests
from bs4 import BeautifulSoup
from requests.auth import HTTPBasicAuth

from .config import GeminiSettings, WordPressSettings
from .io import write_dataframe


SOFT_404_SYSTEM_PROMPT = """
You are a web content classifier.

Your task is to analyze raw HTML content and estimate the probability (from 0 to 1)
that the page represents a soft 404 error page.

Definition:
A soft 404 page is a page that returns an HTTP 200 (or other non-404 status) but whose
content indicates the requested resource does not exist, is unavailable, or should be
considered missing.

Instructions:
- Base your judgment ONLY on the provided HTML content.
- Do NOT assume access to HTTP headers, status codes, or external context.
- Do NOT explain your reasoning.
- Do NOT include text, labels, or formatting.

Output:
- Return a single floating-point number between 0 and 1 (inclusive).

Return ONLY the number.
""".strip()


def fetch_wordpress_posts(
    settings: WordPressSettings,
    *,
    status: str = "draft",
    per_page: int = 100,
) -> list[dict[str, Any]]:
    response = requests.get(
        settings.posts_endpoint,
        params={"per_page": per_page, "page": 1, "status": status},
        auth=HTTPBasicAuth(settings.username, settings.app_password),
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def posts_to_dataframe(posts: list[dict[str, Any]]) -> pl.DataFrame:
    rows = []
    for post in posts:
        meta = post.get("meta") or {}
        footnote = meta.get("footnotes") or None
        rows.append(
            {
                "post_id": post.get("id"),
                "footnote": footnote,
            }
        )

    return pl.DataFrame(rows) if rows else pl.DataFrame(schema={"post_id": pl.Int64, "footnote": pl.Utf8})


def fetch_footnote_metadata(df: pl.DataFrame) -> pl.DataFrame:
    response_codes: list[int | None] = []
    direct_url_html: list[str | None] = []
    request_errors: list[str | None] = []

    for footnote in df["footnote"].to_list():
        if not footnote:
            response_codes.append(None)
            direct_url_html.append(None)
            request_errors.append(None)
            continue

        try:
            response = requests.get(footnote, timeout=30)
            response_codes.append(response.status_code)
            direct_url_html.append(str(BeautifulSoup(response.text, "html.parser")))
            request_errors.append(None)
        except Exception as exc:
            response_codes.append(None)
            direct_url_html.append(None)
            request_errors.append(str(exc))

    return df.with_columns(
        pl.Series("response_code", response_codes),
        pl.Series("direct_url_html", direct_url_html),
        pl.Series("request_error", request_errors),
    )


def score_soft_404_probabilities(html_pages: list[str | None], settings: GeminiSettings) -> list[float | None]:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=settings.api_key)
    probabilities: list[float | None] = []

    for html in html_pages:
        if html is None:
            probabilities.append(None)
            continue

        response = client.models.generate_content(
            model=settings.model,
            contents=(
                "What is the probability the provided html is a soft 404 error page?\n\n"
                f"{html}\n\n"
                "Return ONLY a single number between 0 and 1."
            ),
            config=types.GenerateContentConfig(
                system_instruction=SOFT_404_SYSTEM_PROMPT,
                response_mime_type="text/plain",
            ),
        )

        try:
            probabilities.append(float(response.text.strip()))
        except Exception:
            probabilities.append(None)

    return probabilities


def privatize_invalid_posts(
    post_ids: list[int],
    settings: WordPressSettings,
    *,
    sleep_seconds: float = 0.2,
) -> dict[int, dict[str, Any]]:
    auth = HTTPBasicAuth(settings.username, settings.app_password)
    results: dict[int, dict[str, Any]] = {}

    for post_id in post_ids:
        response = requests.post(
            f"{settings.posts_endpoint}/{post_id}",
            json={"status": "private"},
            auth=auth,
            timeout=30,
        )

        try:
            payload = response.json()
        except ValueError:
            payload = {}

        results[post_id] = {
            "private_status_code": response.status_code,
            "private_error": None if response.ok else payload.get("message", response.text[:200]),
        }
        sleep(sleep_seconds)

    return results


def run_expiration_check(
    *,
    wordpress_settings: WordPressSettings,
    gemini_settings: GeminiSettings,
    status: str = "draft",
    per_page: int = 100,
    threshold: float = 0.5,
    privatize_invalid: bool = True,
    sleep_seconds: float = 0.2,
    output_path: str | None = None,
) -> pl.DataFrame:
    posts = fetch_wordpress_posts(wordpress_settings, status=status, per_page=per_page)
    report = posts_to_dataframe(posts)

    if report.is_empty():
        if output_path:
            write_dataframe(report, output_path)
        return report

    report = fetch_footnote_metadata(report)
    probabilities = score_soft_404_probabilities(report["direct_url_html"].to_list(), gemini_settings)
    report = report.with_columns(
        pl.Series("prob_soft_404", probabilities),
    )
    report = report.with_columns(
        ((pl.col("response_code") != 200) | (pl.col("prob_soft_404") > threshold)).cast(pl.Int8).alias("is_invalid"),
    )

    if privatize_invalid:
        invalid_post_ids = report.filter(pl.col("is_invalid") == 1)["post_id"].to_list()
        privatize_results = privatize_invalid_posts(
            invalid_post_ids,
            wordpress_settings,
            sleep_seconds=sleep_seconds,
        )
        report = report.with_columns(
            pl.Series(
                "private_status_code",
                [privatize_results.get(post_id, {}).get("private_status_code") for post_id in report["post_id"].to_list()],
            ),
            pl.Series(
                "private_error",
                [privatize_results.get(post_id, {}).get("private_error") for post_id in report["post_id"].to_list()],
            ),
        )

    if output_path:
        write_dataframe(report, output_path)

    return report
