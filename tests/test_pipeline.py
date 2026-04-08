from __future__ import annotations

import json

import polars as pl
import pytest

from jcp_data_manager.io import load_linkedin_data, load_sessions_data, write_dataframe
from jcp_data_manager.merge import merge_user_data
from jcp_data_manager.pipeline import run_pipeline


def _write_json(path, payload) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_merge_user_data_handles_arbitrary_same_shape_json(tmp_path) -> None:
    sessions_path = tmp_path / "sessions.json"
    linkedin_path = tmp_path / "linkedin.json"

    _write_json(
        sessions_path,
        {
            "sessions": [
                {
                    "user_id": 101,
                    "session_id": "s-1",
                    "survey_id": "survey-a",
                    "treatment_snapshot": [{"plan": "A", "cohort": "north"}],
                },
                {
                    "user_id": 202,
                    "session_id": "s-2",
                    "survey_id": None,
                    "treatment_snapshot": [{"plan": "B", "cohort": "south"}],
                },
            ]
        },
    )
    _write_json(
        linkedin_path,
        [
            {
                "wordpress_user_id": 101,
                "wordpress_display_name": "Person One",
                "profile_data": {
                    "given_name": "Ada",
                    "family_name": "Lovelace",
                    "picture": "https://example.com/ada.jpg",
                },
            },
            {
                "wordpress_user_id": 202,
                "profile_data": {
                    "given_name": "Grace",
                    "family_name": "Hopper",
                    "picture": "https://example.com/grace.jpg",
                },
            },
        ],
    )

    merged = merge_user_data(linkedin=linkedin_path, sessions=sessions_path)

    assert merged.height == 2
    assert "session_survey_id" in merged.columns
    assert merged["session_survey_id"].to_list() == ["s-1-survey-a", "s-2-"]
    assert "given_name" in merged.columns
    assert "plan" in merged.columns
    assert merged.filter(pl.col("user_id") == "101").select("given_name").item() == "Ada"


def test_run_pipeline_without_optional_enrichment(tmp_path) -> None:
    sessions_path = tmp_path / "sessions.json"
    linkedin_path = tmp_path / "linkedin.json"

    _write_json(
        sessions_path,
        {"sessions": [{"user_id": "9", "session_id": "session-9", "survey_id": "survey-9"}]},
    )
    _write_json(
        linkedin_path,
        [{"wordpress_user_id": 9, "profile_data": {"given_name": "Test", "family_name": "User"}}],
    )

    merged = run_pipeline(sessions_path=sessions_path, linkedin_path=linkedin_path)

    assert merged.height == 1
    assert merged.select("given_name").item() == "Test"


def test_merge_user_data_rejects_invalid_session_shape(tmp_path) -> None:
    sessions_path = tmp_path / "sessions.json"
    linkedin_path = tmp_path / "linkedin.json"

    _write_json(sessions_path, {"sessions": [{"user_id": "1"}]})
    _write_json(linkedin_path, [{"wordpress_user_id": 1, "profile_data": {"given_name": "A"}}])

    with pytest.raises(ValueError, match="missing required columns"):
        merge_user_data(linkedin=linkedin_path, sessions=sessions_path)


def test_loaders_accept_expected_top_level_shapes(tmp_path) -> None:
    sessions_path = tmp_path / "sessions.json"
    linkedin_path = tmp_path / "linkedin.json"

    _write_json(sessions_path, {"sessions": []})
    _write_json(linkedin_path, [])

    assert load_sessions_data(sessions_path).is_empty()
    assert load_linkedin_data(linkedin_path).is_empty()


def test_write_dataframe_json_is_row_oriented(tmp_path) -> None:
    output_path = tmp_path / "output.json"
    df = pl.DataFrame([{"user_id": "1", "session_id": "abc"}])

    write_dataframe(df, output_path)

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload == [{"user_id": "1", "session_id": "abc"}]
