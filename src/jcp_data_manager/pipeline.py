"""High-level pipeline orchestration."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from .enrichment import enrich_with_image_analysis, enrich_with_name_analysis
from .merge import merge_user_data


def run_pipeline(
    *,
    sessions_path: str | Path,
    linkedin_path: str | Path,
    with_image_analysis: bool = True,
    with_name_analysis: bool = True,
    image_timeout: int = 20,
    enforce_detection: bool = True,
) -> pl.DataFrame:
    merged = merge_user_data(linkedin=linkedin_path, sessions=sessions_path)

    if with_image_analysis:
        merged = enrich_with_image_analysis(
            merged,
            timeout=image_timeout,
            enforce_detection=enforce_detection,
        )

    if with_name_analysis:
        merged = enrich_with_name_analysis(merged)

    return merged
