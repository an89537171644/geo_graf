from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from soilstamp.data import (
    failure_analysis_contract,
    failure_analysis_summary,
    failure_summary,
    prepare_measurements,
)


def _three_test_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "test_id": ["B", "B", "A", "A", "C", "C"],
            "sequence_no": [0, 1, 0, 1, 0, 1],
            "F_kN": [100.0, 140.0, 80.0, 120.0, 90.0, 150.0],
            "p_kPa": [np.nan] * 6,
            "settlement_mm": [0.5, np.nan, 0.4, np.nan, 0.6, 1.2],
            "status": ["stable", "failure", "stable", "failure", "stable", "stable"],
        }
    )


def test_failure_analysis_counts_two_observed_and_one_right_censored() -> None:
    failures = failure_summary(_three_test_frame())
    result = failure_analysis_summary(failures)

    assert failures["test_id"].tolist() == ["A", "B", "C"]
    assert failures["censoring_type"].tolist() == [
        "interval_censored",
        "interval_censored",
        "right_censored",
    ]
    assert result == {
        "contract_version": "failure-analysis/1.0",
        "summary_method": "none",
        "capacity_axis": "auto",
        "capacity_unit": None,
        "analysis_status": "descriptive_only_no_point_estimate",
        "point_estimate": None,
        "point_estimate_unit": None,
        "n_tests": 3,
        "n_failure_observed": 2,
        "n_interval_censored": 2,
        "n_right_censored": 1,
        "n_indeterminate": 0,
        "n_review_required": 0,
    }


def test_missing_valid_lower_bound_is_indeterminate_not_right_censored() -> None:
    frame = pd.DataFrame(
        {
            "test_id": ["EMPTY"],
            "sequence_no": [0],
            "F_kN": [np.nan],
            "p_kPa": [np.nan],
            "settlement_mm": [np.nan],
            "status": ["invalid"],
        }
    )

    row = failure_summary(frame).iloc[0]

    assert row["censoring_type"] == "indeterminate"
    assert not bool(row["right_censored"])
    assert row["classification_status"] == "review_required"
    assert row["classification_warning"] == "missing_valid_lower_bound"
    assert pd.isna(row["lower_bound"])


def test_failure_without_settlement_column_keeps_interval_and_sequence_bounds() -> None:
    frame = pd.DataFrame(
        {
            "test_id": ["NO-S", "NO-S"],
            "sequence_no": [10, 20],
            "F_kN": [100.0, 125.0],
            "p_kPa": [np.nan, np.nan],
            "status": ["stable", "failure"],
        }
    )

    row = failure_summary(frame).iloc[0]

    assert bool(row["failure_observed"])
    assert bool(row["interval_censored"])
    assert row["censoring_type"] == "interval_censored"
    assert row["capacity_kind"] == "force"
    assert row["lower_bound"] == 100.0
    assert row["upper_bound"] == 125.0
    assert row["capacity_unit"] == "kN"
    assert not bool(row["lower_inclusive"])
    assert bool(row["upper_inclusive"])
    assert row["lower_bound_sequence_no"] == 10
    assert row["upper_bound_sequence_no"] == 20
    assert pd.isna(row["s_failure"])


def test_multiple_failure_rows_are_visible_and_require_review() -> None:
    frame = pd.DataFrame(
        {
            "test_id": ["MULTI"] * 3,
            "sequence_no": [0, 1, 2],
            "F_kN": [100.0, 130.0, 150.0],
            "p_kPa": [np.nan] * 3,
            "settlement_mm": [0.5, np.nan, np.nan],
            "status": ["stable", "failure", "failure"],
        }
    )

    row = failure_summary(frame).iloc[0]

    assert row["failure_event_count"] == 2
    assert row["failure_sequence_no"] == 1
    assert row["censoring_type"] == "interval_censored"
    assert row["classification_status"] == "review_required"
    assert "multiple_failure_events:2" in row["classification_warning"]


def test_explicit_pressure_protocol_uses_pressure_capacity_even_when_force_is_available() -> None:
    raw = pd.DataFrame(
        {
            "test_id": ["PRESSURE", "PRESSURE"],
            "stage": [0, 1],
            "load": [100.0, 120.0],
            "settlement": [0.5, np.nan],
            "status": ["stable", "failure"],
        }
    )
    frame, _ = prepare_measurements(
        raw,
        {
            "load_kind": "pressure",
            "load_unit": "kPa",
            "stamp_area_m2": 0.1,
        },
        strict_metadata=False,
    )
    assert frame["F_kN"].notna().all()

    row = failure_summary(frame).iloc[0]

    assert row["capacity_kind"] == "pressure"
    assert row["lower_bound"] == 100.0
    assert row["upper_bound"] == 120.0
    assert row["capacity_unit"] == "kPa"
    assert row["display"] == "100 < pu ≤ 120 кПа"


def test_failure_analysis_rejects_unapproved_summary_method() -> None:
    failures = failure_summary(_three_test_frame())

    with pytest.raises(ValueError, match="доступен только 'none'"):
        failure_analysis_summary(failures, summary_method="midpoint_mean")

    contract = failure_analysis_contract()
    assert contract["contract_version"] == "failure-analysis/1.0"
    assert contract["supported_summary_methods"] == ["none"]


def test_failure_summary_is_invariant_to_test_block_order() -> None:
    frame = _three_test_frame()
    blocks = [part for _, part in frame.groupby("test_id", sort=False)]
    reordered = pd.concat(list(reversed(blocks)), ignore_index=True)

    expected = failure_summary(frame).reset_index(drop=True)
    actual = failure_summary(reordered).reset_index(drop=True)

    pd.testing.assert_frame_equal(actual, expected)
