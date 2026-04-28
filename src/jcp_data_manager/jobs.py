"""Job scraping and WordPress posting workflows."""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable
from urllib.parse import quote

import polars as pl
import requests
from bs4 import BeautifulSoup
from requests.auth import HTTPBasicAuth

from .config import GithubModelsSettings, WordPressSettings
from .io import write_dataframe
from .job_templates import build_job_description_system_prompt, build_post_content


DEFAULT_JOB_SITES = ["indeed", "linkedin", "zip_recruiter", "google", "glassdoor"]
DEFAULT_KEYWORDS = [
    "what you'll need",
    "qualifications",
    "what you'll bring",
    "experience and attributes",
    "basic qualifications",
    "years of",
    "what you have",
    "what you'll bring",
]


def _slugify(value: str) -> str:
    return value.strip().lower().replace(" ", "_")


def parse_date_posted(value: str) -> datetime:
    try:
        return datetime.strptime(value, "%m/%d/%Y")
    except ValueError as exc:
        raise ValueError("Invalid date format. Use MM/DD/YYYY, for example 04/21/2026.") from exc


def calculate_hours_old(value: str, *, now: datetime | None = None) -> int:
    today = now or datetime.today()
    date_posted = parse_date_posted(value)
    delta = today - date_posted
    return max(0, int(delta.total_seconds() / 3600))


def page_contains_keywords(url: str, keywords: Iterable[str], timeout: int = 10) -> tuple[int, str | None]:
    try:
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=timeout)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        page_text = soup.get_text(separator=" ").lower()
        for keyword in keywords:
            if keyword.lower() in page_text:
                return 1, str(soup)
        return 0, None
    except Exception:
        return 0, None


def scrape_and_filter_jobs(
    *,
    occupation_title: str,
    date_posted: str,
    location: str,
    results_wanted: int = 20,
    timeout: int = 10,
    keywords: Iterable[str] = DEFAULT_KEYWORDS,
    now: datetime | None = None,
) -> pl.DataFrame:
    from jobspy import scrape_jobs

    hours_old = calculate_hours_old(date_posted, now=now)

    jobs = scrape_jobs(
        site_name=DEFAULT_JOB_SITES,
        search_term=occupation_title,
        google_search_term=f"{occupation_title} jobs near {location} since {date_posted}",
        location=location,
        results_wanted=results_wanted,
        hours_old=hours_old,
        country_indeed="USA",
        linkedin_fetch_description=True,
    )

    jobs_df = pl.DataFrame(jobs)
    if jobs_df.is_empty():
        return jobs_df

    if "job_url_direct" not in jobs_df.columns:
        raise ValueError("Expected JobSpy output to include 'job_url_direct'.")

    jobs_with_direct_links = jobs_df.filter(pl.col("job_url_direct").is_not_null())
    if jobs_with_direct_links.is_empty():
        return jobs_with_direct_links

    qualification_flags: list[int] = []
    html_snapshots: list[str | None] = []
    for url in jobs_with_direct_links["job_url_direct"].to_list():
        if not url:
            qualification_flags.append(0)
            html_snapshots.append(None)
            continue

        flag, html = page_contains_keywords(str(url), keywords, timeout=timeout)
        qualification_flags.append(flag)
        html_snapshots.append(html)

    enriched = jobs_with_direct_links.with_columns(
        pl.Series("qualifications", qualification_flags),
        pl.Series("original_html", html_snapshots),
    )
    return enriched.filter(pl.col("qualifications") == 1)


def _safe_text(value: object) -> str:
    if value is None:
        return ""
    return str(value)


def build_survey_url(original_ad_url: str) -> str:
    return f"https://jobconnectionsproject.org/survey/?ad_url={quote(original_ad_url, safe='')}"


def insert_survey_url_into_post_html(full_html: str, survey_url: str) -> str:
    footer_text = (
        "The Job Connections Project is a non-profit company that advertises open positions for other companies. "
        "Please read the hiring company's job ad below, then click 'Continue'."
    )
    survey_cta_html = (
        f"<p><a class=\"button-link\" href=\"{survey_url}\">Continue</a></p>"
    )

    updated_html = full_html.replace('href="https://jobconnectionsproject.org/survey/"', f'href="{survey_url}"')
    updated_html = updated_html.replace("href='https://jobconnectionsproject.org/survey/'", f"href='{survey_url}'")
    updated_html = updated_html.replace('href="/survey/"', f'href="{survey_url}"')
    updated_html = updated_html.replace("href='/survey/'", f"href='{survey_url}'")
    updated_html = updated_html.replace('action="https://jobconnectionsproject.org/survey/"', f'action="{survey_url}"')
    updated_html = updated_html.replace("action='https://jobconnectionsproject.org/survey/'", f"action='{survey_url}'")
    updated_html = updated_html.replace('action="/survey/"', f'action="{survey_url}"')
    updated_html = updated_html.replace("action='/survey/'", f"action='{survey_url}'")

    if survey_url not in updated_html:
        updated_html = updated_html.replace(footer_text, f"{footer_text}\n\n{survey_cta_html}", 1)

    return updated_html


def generate_job_post_html(df: pl.DataFrame, settings: GithubModelsSettings, *, experiment: int) -> list[str]:
    if df.is_empty():
        return []

    from azure.ai.inference import ChatCompletionsClient
    from azure.ai.inference.models import SystemMessage, UserMessage
    from azure.core.credentials import AzureKeyCredential

    client = ChatCompletionsClient(
        endpoint=settings.endpoint,
        credential=AzureKeyCredential(settings.token),
    )
    system_prompt = build_job_description_system_prompt(experiment)

    html_responses: list[str] = []
    for row in df.select(["description", "title", "location"]).iter_rows(named=True):
        response = client.complete(
            messages=[
                SystemMessage(system_prompt),
                UserMessage(
                    f"Here is the job title: {_safe_text(row['title'])},"
                    f"the location is {_safe_text(row['location'])},"
                    f"and the job description is: {_safe_text(row['description'])}"
                ),
            ],
            model=settings.model,
        )
        html_responses.append(response.choices[0].message.content)

    return html_responses


def clean_description_hard(text: str | None) -> str:
    if not text:
        return ""
    cleaned = re.sub(r"-{3,}", " ", text)
    cleaned = re.sub(r"[\r\n\t]+", " ", cleaned)
    cleaned = re.sub(r"[^\w\s.,;:()\-&/]", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def build_google_job_scripts(df: pl.DataFrame, *, now: datetime | None = None) -> list[str]:
    today = now or datetime.today()
    valid_through = today + timedelta(days=60)
    scripts: list[str] = []

    for row in df.select(["title", "description", "location", "company", "id"]).iter_rows(named=True):
        raw_location = _safe_text(row["location"])
        parts = [part.strip() for part in raw_location.split(",") if part and part.strip()]
        city = parts[0] if len(parts) >= 1 else None
        region = parts[1] if len(parts) >= 2 else None

        address: dict[str, str] = {
            "@type": "PostalAddress",
            "addressCountry": "US",
        }
        if city:
            address["addressLocality"] = city
        if region:
            address["addressRegion"] = region

        clean_desc = clean_description_hard(_safe_text(row["description"]))
        identifier_value = _safe_text(row["id"])
        company = _safe_text(row["company"])
        title = _safe_text(row["title"])

        script = f"""
<script type="application/ld+json">
{{
  "@context": "https://schema.org/",
  "@type": "JobPosting",
  "title": {json.dumps(title)},
  "description": {json.dumps(clean_desc)},
  "identifier": {{
    "@type": "PropertyValue",
    "name": "Job Connections Project",
    "value": {json.dumps(identifier_value)}
  }},
  "datePosted": "{today.date()}",
  "validThrough": "{valid_through.date()}",
  "employmentType": "FULL_TIME",
  "hiringOrganization": {{
    "@type": "Organization",
    "name": {json.dumps(company)}
  }},
  "jobLocation": {{
    "@type": "Place",
    "address": {json.dumps(address)}
  }}
}}
</script>
""".strip()
        scripts.append(script)

    return scripts


def default_jobs_output_path(
    *,
    occupation_title: str,
    location: str,
    now: datetime | None = None,
) -> Path:
    today = now or datetime.today()
    filename = f"{today.month}-{today.day}-{today.year}_{_slugify(occupation_title)}_{_slugify(location)}_jobs.csv"
    return Path(filename)


def _posting_result_from_response(response: requests.Response, *, include_linkedin_popup: bool) -> dict[str, object]:
    try:
        payload = response.json()
    except ValueError:
        payload = {}

    error_message = None
    if not response.ok:
        error_message = payload.get("message") if isinstance(payload, dict) else response.text[:300]

    return {
        "wordpress_post_id": payload.get("id") if isinstance(payload, dict) else None,
        "wordpress_status_code": response.status_code,
        "wordpress_error": error_message,
        "posted_with_linkedin_popup": include_linkedin_popup,
    }


def post_jobs_to_wordpress(
    df: pl.DataFrame,
    *,
    settings: WordPressSettings,
    include_linkedin_popup: bool,
    experiment: int,
    status: str = "draft",
) -> pl.DataFrame:
    if df.is_empty():
        return df.with_columns(
            pl.Series("wordpress_post_id", [], dtype=pl.Int64),
            pl.Series("wordpress_status_code", [], dtype=pl.Int64),
            pl.Series("wordpress_error", [], dtype=pl.Utf8),
            pl.Series("posted_with_linkedin_popup", [], dtype=pl.Boolean),
        )

    results: list[dict[str, object]] = []
    auth = HTTPBasicAuth(settings.username, settings.app_password)

    rows = df.select(["title", "location", "google_ad_scripts", "jcp_job_html", "job_url_direct"]).iter_rows(named=True)
    for row in rows:
        original_ad_url = _safe_text(row["job_url_direct"])
        survey_url = build_survey_url(original_ad_url)
        full_html = build_post_content(
            google_script=_safe_text(row["google_ad_scripts"]),
            generated_html=_safe_text(row["jcp_job_html"]),
            include_linkedin_popup=include_linkedin_popup,
            experiment=experiment,
        )
        full_html = insert_survey_url_into_post_html(full_html, survey_url)

        payload = {
            "title": f"{_safe_text(row['title'])} - {_safe_text(row['location'])}",
            "content": full_html,
            "status": status,
            "featured_media": settings.featured_media_id,
            "meta": {
                "footnotes": original_ad_url,
            },
        }

        response = requests.post(
            settings.posts_endpoint,
            json=payload,
            auth=auth,
            timeout=30,
        )
        results.append(_posting_result_from_response(response, include_linkedin_popup=include_linkedin_popup))

    return df.with_columns(
        pl.Series("wordpress_post_id", [result["wordpress_post_id"] for result in results]),
        pl.Series("wordpress_status_code", [result["wordpress_status_code"] for result in results]),
        pl.Series("wordpress_error", [result["wordpress_error"] for result in results]),
        pl.Series("posted_with_linkedin_popup", [result["posted_with_linkedin_popup"] for result in results]),
    )


def run_job_posting_pipeline(
    *,
    occupation_title: str,
    date_posted: str,
    location: str,
    github_settings: GithubModelsSettings,
    wordpress_settings: WordPressSettings | None = None,
    output_path: str | Path | None = None,
    results_wanted: int = 20,
    include_linkedin_popup: bool = True,
    experiment: int,
    skip_post: bool = False,
    keyword_timeout: int = 10,
    now: datetime | None = None,
) -> tuple[pl.DataFrame, Path]:
    build_job_description_system_prompt(experiment)

    filtered_jobs = scrape_and_filter_jobs(
        occupation_title=occupation_title,
        date_posted=date_posted,
        location=location,
        results_wanted=results_wanted,
        timeout=keyword_timeout,
        now=now,
    )

    if filtered_jobs.is_empty():
        final_df = filtered_jobs
    else:
        html_responses = generate_job_post_html(filtered_jobs, github_settings, experiment=experiment)
        google_scripts = build_google_job_scripts(filtered_jobs, now=now)
        final_df = filtered_jobs.with_columns(
            pl.Series("jcp_job_html", html_responses),
            pl.Series("google_ad_scripts", google_scripts),
            pl.lit(experiment).alias("experiment"),
        )

        if not skip_post:
            if wordpress_settings is None:
                raise ValueError("WordPress settings are required unless --skip-post is used.")
            final_df = post_jobs_to_wordpress(
                final_df,
                settings=wordpress_settings,
                include_linkedin_popup=include_linkedin_popup,
                experiment=experiment,
            )

    resolved_output = Path(output_path) if output_path else default_jobs_output_path(
        occupation_title=occupation_title,
        location=location,
        now=now,
    )
    write_dataframe(final_df, resolved_output)
    return final_df, resolved_output
