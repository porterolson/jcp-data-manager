"""Optional enrichment steps for merged data."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import polars as pl
import requests


IMAGE_ANALYSIS_SCHEMA: dict[str, pl.DataType] = {
    "picture": pl.Utf8,
    "df_age": pl.Float64,
    "df_face_confidence": pl.Float64,
    "df_dominant_gender": pl.Utf8,
    "df_dominant_race": pl.Utf8,
    "df_prob_male": pl.Float64,
    "df_prob_female": pl.Float64,
    "df_prob_white": pl.Float64,
    "df_prob_black": pl.Float64,
    "df_prob_asian": pl.Float64,
    "df_prob_indian": pl.Float64,
    "df_prob_middle_eastern": pl.Float64,
    "df_prob_latino": pl.Float64,
    "df_status": pl.Utf8,
    "df_error": pl.Utf8,
}

NAME_ANALYSIS_SCHEMA: dict[str, pl.DataType] = {
    "given_name": pl.Utf8,
    "family_name": pl.Utf8,
    "gg_gender_label": pl.Utf8,
    "eth_asian": pl.Float64,
    "eth_hispanic": pl.Float64,
    "eth_nh_black": pl.Float64,
    "eth_nh_white": pl.Float64,
    "eth_race_label": pl.Utf8,
    "eth_processing_status": pl.Utf8,
    "ethnicolr_status": pl.Utf8,
}


def _require_dependency(import_name: str, package_hint: str) -> Any:
    try:
        module = __import__(import_name, fromlist=["*"])
    except ImportError as exc:
        raise ImportError(
            f"Optional dependency '{import_name}' is required. Install with `pip install -e \".[{package_hint}]\"`."
        ) from exc
    return module


def _load_deepface_class() -> Any:
    try:
        from deepface import DeepFace as deepface_class

        return deepface_class
    except (ImportError, AttributeError):
        try:
            from deepface.DeepFace import DeepFace as deepface_class

            return deepface_class
        except ImportError as exc:
            raise ImportError(
                "DeepFace is installed but its DeepFace class could not be imported. "
                "Try reinstalling with `pip install -e \".[image]\"`."
            ) from exc


def _build_picture_row_template(link: str | None) -> dict[str, Any]:
    return {
        "picture": link,
        "df_age": None,
        "df_face_confidence": None,
        "df_dominant_gender": None,
        "df_dominant_race": None,
        "df_prob_male": None,
        "df_prob_female": None,
        "df_prob_white": None,
        "df_prob_black": None,
        "df_prob_asian": None,
        "df_prob_indian": None,
        "df_prob_middle_eastern": None,
        "df_prob_latino": None,
        "df_status": None,
        "df_error": None,
    }


def image_analysis_dataframe(
    df: pl.DataFrame,
    picture_column: str = "picture",
    timeout: int = 20,
    enforce_detection: bool = True,
) -> pl.DataFrame:
    _require_dependency("deepface", "image")
    deepface = _load_deepface_class()

    if picture_column not in df.columns:
        raise ValueError(f"Column '{picture_column}' does not exist.")

    picture_links = (
        df.select(picture_column)
        .filter(pl.col(picture_column).is_not_null())
        .unique()
        .to_series()
        .to_list()
    )

    if not picture_links:
        return pl.DataFrame(schema=IMAGE_ANALYSIS_SCHEMA)

    rows: list[dict[str, Any]] = []
    with TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir) / "image.jpg"

        for link in picture_links:
            row = _build_picture_row_template(link)
            try:
                response = requests.get(link, timeout=timeout)
                response.raise_for_status()
                temp_path.write_bytes(response.content)

                result = deepface.analyze(
                    img_path=str(temp_path),
                    actions=["age", "gender", "race"],
                    enforce_detection=enforce_detection,
                )

                face = result[0] if isinstance(result, list) else result
                gender_dict = face.get("gender", {}) or {}
                race_dict = face.get("race", {}) or {}

                row["df_age"] = face.get("age")
                row["df_face_confidence"] = face.get("face_confidence")
                row["df_dominant_gender"] = face.get("dominant_gender")
                row["df_dominant_race"] = face.get("dominant_race")
                row["df_prob_male"] = _safe_float(gender_dict.get("Man"))
                row["df_prob_female"] = _safe_float(gender_dict.get("Woman"))
                row["df_prob_white"] = _safe_float(race_dict.get("white"))
                row["df_prob_black"] = _safe_float(race_dict.get("black"))
                row["df_prob_asian"] = _safe_float(race_dict.get("asian"))
                row["df_prob_indian"] = _safe_float(race_dict.get("indian"))
                row["df_prob_middle_eastern"] = _safe_float(race_dict.get("middle eastern"))
                row["df_prob_latino"] = _safe_float(race_dict.get("latino hispanic"))
                row["df_status"] = "ok"
            except Exception as exc:
                message = str(exc)
                row["df_status"] = "no_face_detected" if "Face could not be detected" in message else "error"
                row["df_error"] = message

            rows.append(row)

    return pl.DataFrame(rows, schema=IMAGE_ANALYSIS_SCHEMA)


def enrich_with_image_analysis(
    df: pl.DataFrame,
    picture_column: str = "picture",
    timeout: int = 20,
    enforce_detection: bool = True,
) -> pl.DataFrame:
    picture_df = image_analysis_dataframe(
        df,
        picture_column=picture_column,
        timeout=timeout,
        enforce_detection=enforce_detection,
    )
    return df.join(picture_df, on=[picture_column], how="left")


def name_analysis_dataframe(
    df: pl.DataFrame,
    given_name_column: str = "given_name",
    family_name_column: str = "family_name",
) -> pl.DataFrame:
    pandas = _require_dependency("pandas", "names")
    gender_module = _require_dependency("gender_guesser.detector", "names")
    ethnicolr_module = _require_dependency("ethnicolr", "names")

    detector = gender_module.Detector(case_sensitive=False)
    pred_fl_reg_name = ethnicolr_module.pred_fl_reg_name

    required = [given_name_column, family_name_column]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for name analysis: {missing}")

    unique_names = (
        df.select(required)
        .filter(pl.col(given_name_column).is_not_null())
        .unique()
    )

    if unique_names.is_empty():
        schema = {
            given_name_column: pl.Utf8,
            family_name_column: pl.Utf8,
            **{key: value for key, value in NAME_ANALYSIS_SCHEMA.items() if key not in {"given_name", "family_name"}},
        }
        return pl.DataFrame(schema=schema)

    rows: list[dict[str, Any]] = []
    for row in unique_names.iter_rows(named=True):
        first_name = row[given_name_column]
        last_name = row[family_name_column]
        gender_guess = detector.get_gender(first_name) if first_name is not None else None

        temp_df = pandas.DataFrame({"first_name": [first_name], "last_name": [last_name]})

        try:
            result = pred_fl_reg_name(temp_df, "last_name", "first_name")
            record = result.iloc[0]
            rows.append(
                {
                    given_name_column: first_name,
                    family_name_column: last_name,
                    "gg_gender_label": gender_guess,
                    "eth_asian": record.get("asian"),
                    "eth_hispanic": record.get("hispanic"),
                    "eth_nh_black": record.get("nh_black"),
                    "eth_nh_white": record.get("nh_white"),
                    "eth_race_label": record.get("race"),
                    "eth_processing_status": record.get("processing_status"),
                    "ethnicolr_status": "ok",
                }
            )
        except Exception as exc:
            rows.append(
                {
                    given_name_column: first_name,
                    family_name_column: last_name,
                    "gg_gender_label": gender_guess,
                    "eth_asian": None,
                    "eth_hispanic": None,
                    "eth_nh_black": None,
                    "eth_nh_white": None,
                    "eth_race_label": None,
                    "eth_processing_status": None,
                    "ethnicolr_status": f"error: {exc}",
                }
            )

    schema = {
        given_name_column: pl.Utf8,
        family_name_column: pl.Utf8,
        **{key: value for key, value in NAME_ANALYSIS_SCHEMA.items() if key not in {"given_name", "family_name"}},
    }
    return pl.DataFrame(rows, schema=schema)


def enrich_with_name_analysis(
    df: pl.DataFrame,
    given_name_column: str = "given_name",
    family_name_column: str = "family_name",
) -> pl.DataFrame:
    name_df = name_analysis_dataframe(
        df,
        given_name_column=given_name_column,
        family_name_column=family_name_column,
    )
    return df.join(name_df, on=[given_name_column, family_name_column], how="left")


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)
