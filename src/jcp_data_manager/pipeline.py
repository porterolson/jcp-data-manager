"""High-level pipeline orchestration."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from .enrichment import enrich_with_image_analysis, enrich_with_name_analysis
from .merge import prepare_sessions_data


def run_pipeline(
    *,
    sessions_path: str | Path,
    with_image_analysis: bool = True,
    with_name_analysis: bool = True,
    image_timeout: int = 20,
    enforce_detection: bool = True,
) -> pl.DataFrame:
    merged = prepare_sessions_data(sessions_path)

    if with_image_analysis:
        merged = enrich_with_image_analysis(
            merged,
            timeout=image_timeout,
            enforce_detection=enforce_detection,
        )

    if with_name_analysis:
        merged = enrich_with_name_analysis(merged)

    return merged
