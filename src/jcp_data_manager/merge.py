"""Core merge logic extracted from the notebook."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from .io import load_linkedin_data, load_sessions_data


LINKEDIN_RENAME_MAP = {
    "wordpress_user_id": "user_id",
    "last_synced_at": "account_last_synced_at",
    "token_expires_at": "linkedin_token_expires_at",
    "ip_history": "linkedin_ip_history",
    "profile_data": "linkedin_profile_data",
}

LINKEDIN_DROP_COLUMNS = ["wordpress_display_name"]
SESSIONS_REQUIRED_COLUMNS = {"user_id", "session_id"}
LINKEDIN_USER_ID_CANDIDATES = ("wordpress_user_id", "user_id")


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


def normalize_linkedin_data(df: pl.DataFrame) -> pl.DataFrame:
    rename_map = {old: new for old, new in LINKEDIN_RENAME_MAP.items() if old in df.columns}
    normalized = df.rename(rename_map)

    if "user_id" in normalized.columns:
        normalized = normalized.with_columns(pl.col("user_id").cast(pl.Utf8))

    drop_columns = [column for column in LINKEDIN_DROP_COLUMNS if column in normalized.columns]
    if drop_columns:
        normalized = normalized.select(pl.exclude(drop_columns))

    return normalized


def validate_sessions_data(df: pl.DataFrame) -> None:
    missing = sorted(SESSIONS_REQUIRED_COLUMNS - set(df.columns))
    if missing:
        raise ValueError(
            "Session data is missing required columns: "
            f"{missing}. Expected the JSON under 'sessions' to include at least {sorted(SESSIONS_REQUIRED_COLUMNS)}."
        )


def validate_linkedin_data(df: pl.DataFrame) -> None:
    if not any(column in df.columns for column in LINKEDIN_USER_ID_CANDIDATES):
        raise ValueError(
            "LinkedIn data is missing a user id column. "
            "Expected either 'wordpress_user_id' or 'user_id'."
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


def merge_user_data(linkedin: str | Path, sessions: str | Path) -> pl.DataFrame:
    sessions_df = load_sessions_data(sessions)
    linkedin_raw_df = load_linkedin_data(linkedin)

    validate_sessions_data(sessions_df)
    validate_linkedin_data(linkedin_raw_df)

    linkedin_df = normalize_linkedin_data(linkedin_raw_df)

    if "user_id" in sessions_df.columns:
        sessions_df = sessions_df.with_columns(pl.col("user_id").cast(pl.Utf8))

    merged = sessions_df.join(linkedin_df, on=["user_id"], how="left")
    merged = expand_struct(merged, "linkedin_profile_data")
    merged = expand_list_struct_drop_inner(merged, "treatment_snapshot")
    merged = build_session_survey_id(merged)
    assert_unique_key(merged, "session_survey_id")

    return merged
