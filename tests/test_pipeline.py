from __future__ import annotations

import json

import polars as pl
import pytest

from jcp_data_manager.io import load_sessions_data, write_dataframe
from jcp_data_manager.merge import prepare_sessions_data
from jcp_data_manager.pipeline import run_pipeline


def _write_json(path, payload) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_prepare_sessions_data_handles_server_merged_export(tmp_path) -> None:
    sessions_path = tmp_path / "sessions.json"

    _write_json(
        sessions_path,
        {
            "sessions": [
                {
                    "session": {
                        "user_id": 101,
                        "session_id": "s-1",
                        "treatment_snapshot": [{"plan": "A", "cohort": "north"}],
                    },
                    "linkedin_rows": [
                        {
                            "user_id": 101,
                            "profile_data": {
                                "given_name": "Ada",
                                "family_name": "Lovelace",
                                "picture": "https://example.com/ada.jpg",
                            },
                        }
                    ],
                    "job_survey_rows": [
                        {
                            "survey_id": "survey-a",
                            "post_url": "https://example.com/job-a",
                        }
                    ],
                },
                {
                    "session": {
                        "user_id": 202,
                        "session_id": "s-2",
                        "treatment_snapshot": [{"plan": "B", "cohort": "south"}],
                    },
                    "linkedin_rows": [
                        {
                            "user_id": 202,
                            "profile_data": {
                                "given_name": "Grace",
                                "family_name": "Hopper",
                                "picture": "https://example.com/grace.jpg",
                            },
                        }
                    ],
                    "job_survey_rows": [
                        {
                            "survey_id": "survey-b",
                            "post_url": "https://example.com/job-b",
                        }
                    ],
                },
            ]
        },
    )

    merged = prepare_sessions_data(sessions_path)

    assert merged.height == 2
    assert "session_survey_id" in merged.columns
    assert merged["session_survey_id"].to_list() == ["s-1-survey-a", "s-2-survey-b"]
    assert "given_name" in merged.columns
    assert "plan" in merged.columns
    assert merged.filter(pl.col("user_id") == "101").select("given_name").item() == "Ada"


def test_run_pipeline_without_optional_enrichment(tmp_path) -> None:
    sessions_path = tmp_path / "sessions.json"

    _write_json(
        sessions_path,
        {
            "sessions": [
                {
                    "session": {"user_id": "9", "session_id": "session-9"},
                    "linkedin_rows": [
                        {
                            "user_id": 9,
                            "profile_data": {"given_name": "Test", "family_name": "User"},
                        }
                    ],
                    "job_survey_rows": [{"survey_id": "survey-9"}],
                }
            ]
        },
    )

    merged = run_pipeline(sessions_path=sessions_path, with_image_analysis=False, with_name_analysis=False)

    assert merged.height == 1
    assert merged.select("given_name").item() == "Test"


def test_prepare_sessions_data_rejects_invalid_session_shape(tmp_path) -> None:
    sessions_path = tmp_path / "sessions.json"

    _write_json(sessions_path, {"sessions": [{"job_survey_rows": []}]})

    with pytest.raises(ValueError, match="nested 'session' struct"):
        prepare_sessions_data(sessions_path)


def test_loaders_accept_expected_top_level_shape(tmp_path) -> None:
    sessions_path = tmp_path / "sessions.json"

    _write_json(sessions_path, {"sessions": []})

    assert load_sessions_data(sessions_path).is_empty()


def test_write_dataframe_json_is_row_oriented(tmp_path) -> None:
    output_path = tmp_path / "output.json"
    df = pl.DataFrame([{"user_id": "1", "session_id": "abc"}])

    write_dataframe(df, output_path)

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload == [{"user_id": "1", "session_id": "abc"}]
