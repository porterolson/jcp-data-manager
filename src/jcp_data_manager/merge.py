"""Core normalization logic extracted from the notebook."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from .io import load_sessions_data


NORMALIZED_SESSION_REQUIRED_COLUMNS = {"session_id", "user_id"}


def expand_struct(df: pl.DataFrame, column_name: str) -> pl.DataFrame:
    if column_name not in df.columns:
        return df
    if not isinstance(df.schema[column_name], pl.Struct):
        return df
    return df.with_columns(pl.col(column_name).struct.unnest())


def expand_list_struct_drop_inner(df: pl.DataFrame, column_name: str) -> pl.DataFrame:
    if column_name not in df.columns:
        return df

    dtype = df.schema[column_name]
    if not isinstance(dtype, pl.List) or not isinstance(dtype.inner, pl.Struct):
        return df

    exploded = df.explode(column_name)
    inner_names = [field.name for field in exploded.schema[column_name].fields]
    outer_names = set(exploded.columns) - {column_name}
    keep_inner = [name for name in inner_names if name not in outer_names]

    return (
        exploded.with_columns(
            pl.struct(
                [pl.col(column_name).struct.field(name).alias(name) for name in keep_inner]
            ).alias(column_name)
        ).unnest(column_name)
    )


def drop_columns_if_present(df: pl.DataFrame, columns: list[str]) -> pl.DataFrame:
    present = [column for column in columns if column in df.columns]
    if not present:
        return df
    return df.select(pl.exclude(present))


def validate_raw_sessions_data(df: pl.DataFrame) -> None:
    if "session" not in df.columns and not NORMALIZED_SESSION_REQUIRED_COLUMNS.issubset(df.columns):
        raise ValueError(
            "Session data must contain either a nested 'session' struct or top-level "
            f"{sorted(NORMALIZED_SESSION_REQUIRED_COLUMNS)} columns."
        )


def validate_normalized_sessions_data(df: pl.DataFrame) -> None:
    missing = sorted(NORMALIZED_SESSION_REQUIRED_COLUMNS - set(df.columns))
    if missing:
        raise ValueError(
            "Normalized session data is missing required columns: "
            f"{missing}."
        )


def build_session_survey_id(df: pl.DataFrame) -> pl.DataFrame:
    if "session_id" not in df.columns:
        return df

    if "survey_id" not in df.columns:
        return df.with_columns(pl.col("session_id").alias("session_survey_id"))

    return df.with_columns(
        (pl.col("session_id") + "-" + pl.col("survey_id").fill_null("")).alias("session_survey_id")
    )


def assert_unique_key(df: pl.DataFrame, key: str) -> None:
    if key not in df.columns:
        raise ValueError(f"Expected unique key column '{key}' to exist.")
    is_unique = df.select(pl.col(key).is_unique().all()).item()
    if not is_unique:
        duplicates = df.filter(pl.col(key).is_duplicated()).select(key).unique().to_series().to_list()
        preview = duplicates[:10]
        raise ValueError(f"Column '{key}' is not unique. Example duplicate keys: {preview}")


def normalize_sessions_export(df: pl.DataFrame) -> pl.DataFrame:
    validate_raw_sessions_data(df)

    normalized = expand_struct(df, "session")
    normalized = expand_list_struct_drop_inner(normalized, "linkedin_rows")
    normalized = expand_struct(normalized, "profile_data")
    normalized = expand_list_struct_drop_inner(normalized, "job_survey_rows")
    normalized = drop_columns_if_present(normalized, ["session", "profile_data"])

    cast_columns: list[pl.Expr] = []
    if "user_id" in normalized.columns:
        cast_columns.append(pl.col("user_id").cast(pl.Utf8))
    if "session_id" in normalized.columns:
        cast_columns.append(pl.col("session_id").cast(pl.Utf8))
    if "survey_id" in normalized.columns:
        cast_columns.append(pl.col("survey_id").cast(pl.Utf8))

    if cast_columns:
        normalized = normalized.with_columns(cast_columns)

    validate_normalized_sessions_data(normalized)
    normalized = build_session_survey_id(normalized)
    assert_unique_key(normalized, "session_survey_id")

    return normalized


def prepare_sessions_data(sessions: str | Path) -> pl.DataFrame:
    sessions_df = load_sessions_data(sessions)
    return normalize_sessions_export(sessions_df)


def merge_user_data(sessions: str | Path, linkedin: str | Path | None = None) -> pl.DataFrame:
    """Backward-compatible alias for the old API name."""

    return prepare_sessions_data(sessions)
