from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from soilstamp.analysis import center_and_tilt
from soilstamp.data import prepare_measurements
from soilstamp.indicators import (
    indicator_audit_frame,
    indicator_event_frame,
    indicator_passport_frame,
    process_indicator_frame,
)


def _passport(mode: str, **overrides) -> dict:
    result = {
        "type": "ИЧ-10",
        "serial_number": "IND-001",
        "range_mm": 10.0,
        "division_mm": 0.01,
        "correction_factor": 1.0,
        "verification_date": "2026-01-15",
        "verification_valid_until": "2030-01-15",
        "mode": mode,
        "initial_reading": 0.0,
        "zero_correction_mm": 0.0,
        "max_increment_mm": 2.0,
        "reverse_tolerance_mm": 0.02,
        "travel_range_mm": 50.0,
    }
    result.update(overrides)
    return result


def _frame(values, *, branches=None, turns=None) -> pd.DataFrame:
    count = len(values)
    result = pd.DataFrame(
        {
            "test_id": ["T1"] * count,
            "stage": list(range(count)),
            "load": list(range(count)),
            "branch": branches or ["loading"] * count,
            "indicator_1": values,
        }
    )
    if turns is not None:
        result["indicator_1_turn_number"] = turns
    return result


def _process(values, passport, *, branches=None, turns=None):
    frame = _frame(values, branches=branches, turns=turns)
    metadata = {"indicator_passports": {"indicator_1": passport}}
    return process_indicator_frame(frame, metadata)


def test_decreasing_scale_is_converted_from_initial_reading() -> None:
    result = _process(
        [9.80, 9.50, 9.10],
        _passport("decreasing", initial_reading=9.80, max_increment_mm=1.0),
    )

    audit = indicator_audit_frame(result)
    assert not [issue for issue in result.issues if issue.level == "error"]
    assert audit["turn_number"].tolist() == [0, 0, 0]
    assert audit["computed_increment_mm"].tolist() == pytest.approx([0.0, 0.30, 0.40])
    assert audit["cumulative_settlement_mm"].tolist() == pytest.approx([0.0, 0.30, 0.70])


def test_increasing_scale_is_converted_from_initial_reading() -> None:
    result = _process(
        [0.10, 0.40, 0.90],
        _passport("increasing", initial_reading=0.10, max_increment_mm=1.0),
    )

    audit = indicator_audit_frame(result)
    assert audit["computed_increment_mm"].tolist() == pytest.approx([0.0, 0.30, 0.50])
    assert audit["cumulative_settlement_mm"].tolist() == pytest.approx([0.0, 0.30, 0.80])


def test_one_zero_crossing_is_unwrapped_and_logged() -> None:
    result = _process(
        [9.80, 0.20, 0.70],
        _passport("increasing_wrapped", initial_reading=9.80, max_increment_mm=1.0),
    )

    audit = indicator_audit_frame(result)
    events = indicator_event_frame(result)
    assert audit["turn_number"].tolist() == [0, 1, 1]
    assert audit["computed_increment_mm"].tolist() == pytest.approx([0.0, 0.40, 0.50])
    crossing = events[events["event_type"] == "zero_crossing"]
    assert len(crossing) == 1
    assert crossing.iloc[0]["turn_before"] == 0
    assert crossing.iloc[0]["turn_after"] == 1
    assert "zero_crossing" in audit.loc[1, "warning"]


def test_multiple_full_revolutions_use_explicit_turn_counter() -> None:
    result = _process(
        [0.00, 0.20, 0.40],
        _passport(
            "increasing_wrapped",
            initial_reading=0.0,
            max_increment_mm=25.0,
            travel_range_mm=50.0,
        ),
        turns=[0, 2, 3],
    )

    audit = indicator_audit_frame(result)
    assert not [issue for issue in result.issues if issue.level == "error"]
    assert audit["turn_number"].tolist() == [0, 2, 3]
    assert audit["computed_increment_mm"].tolist() == pytest.approx([0.0, 20.20, 10.20])
    assert audit["cumulative_settlement_mm"].tolist() == pytest.approx([0.0, 20.20, 30.40])
    assert (indicator_event_frame(result)["event_type"] == "zero_crossing").sum() == 2


def test_multiple_possible_revolutions_without_counter_are_blocked() -> None:
    result = _process(
        [0.00, 0.20],
        _passport(
            "increasing_wrapped",
            initial_reading=0.0,
            max_increment_mm=25.0,
            travel_range_mm=50.0,
        ),
        turns=[0, np.nan],
    )

    audit = indicator_audit_frame(result)
    assert any(issue.code == "ambiguous_indicator_turn" for issue in result.issues)
    assert audit.loc[1, "processing_status"] == "error"
    assert np.isnan(audit.loc[1, "cumulative_settlement_mm"])


def test_rejected_turn_and_zero_correction_are_reported_honestly() -> None:
    result = _process(
        [0.20, 0.30],
        _passport(
            "increasing_wrapped",
            initial_reading=0.0,
            max_increment_mm=1.0,
            zero_correction_mm=0.1,
        ),
        turns=[2, 0],
    )

    audit = indicator_audit_frame(result)
    events = indicator_event_frame(result)
    rejected_crossing = events[
        (events["event_type"] == "zero_crossing") & (events["row_index"] == 0)
    ].iloc[0]
    correction = events[events["event_type"] == "zero_correction_applied"]
    assert "заблокирована" in rejected_crossing["reason"]
    assert np.isnan(audit.loc[0, "applied_correction_mm"])
    assert correction["row_index"].tolist() == [1]


def test_small_reverse_motion_is_preserved_and_logged() -> None:
    result = _process(
        [1.00, 1.20, 1.19, 1.30],
        _passport(
            "increasing",
            initial_reading=1.0,
            max_increment_mm=0.5,
            reverse_tolerance_mm=0.02,
        ),
    )

    audit = indicator_audit_frame(result)
    assert not [issue for issue in result.issues if issue.level == "error"]
    assert audit.loc[2, "computed_increment_mm"] == pytest.approx(-0.01)
    assert audit.loc[2, "cumulative_settlement_mm"] == pytest.approx(0.19)
    assert "small_reverse_motion" in audit.loc[2, "warning"]
    assert "small_reverse_motion" in indicator_event_frame(result)["event_type"].tolist()


def test_large_reverse_motion_is_blocked_and_logged() -> None:
    result = _process(
        [1.00, 0.50],
        _passport(
            "increasing",
            initial_reading=1.0,
            max_increment_mm=1.0,
            reverse_tolerance_mm=0.02,
        ),
    )

    audit = indicator_audit_frame(result)
    assert any(
        issue.code == "unexpected_indicator_reverse_motion"
        for issue in result.issues
    )
    assert audit.loc[1, "processing_status"] == "error"
    assert "unexpected_reverse_motion" in indicator_event_frame(result)["event_type"].tolist()


def test_invalid_jump_is_not_applied() -> None:
    result = _process(
        [1.00, 4.00],
        _passport("increasing", initial_reading=1.0, max_increment_mm=1.0),
    )

    audit = indicator_audit_frame(result)
    assert any(issue.code == "invalid_indicator_jump" for issue in result.issues)
    assert audit.loc[1, "processing_status"] == "error"
    assert np.isnan(audit.loc[1, "cumulative_settlement_mm"])
    assert result.settlement_by_row[1] is None


def test_initial_reading_must_belong_to_declared_scale() -> None:
    result = _process(
        [0.20],
        _passport("increasing", initial_reading=10.50, max_increment_mm=1.0),
    )

    assert any(
        issue.code == "initial_indicator_reading_out_of_range"
        for issue in result.issues
    )
    audit = indicator_audit_frame(result)
    assert audit.loc[0, "processing_status"] == "unprocessed"
    assert audit.loc[0, "original_reading"] == pytest.approx(0.20)


def test_missing_reading_remains_nan_without_forward_fill() -> None:
    result = _process(
        [1.00, np.nan, 1.50],
        _passport("increasing", initial_reading=1.0, max_increment_mm=1.0),
    )

    audit = indicator_audit_frame(result)
    assert audit.loc[1, "processing_status"] == "missing"
    assert np.isnan(audit.loc[1, "cumulative_settlement_mm"])
    assert result.settlement_by_row[1] is None
    assert audit.loc[2, "computed_increment_mm"] == pytest.approx(0.50)
    assert audit.loc[2, "cumulative_settlement_mm"] == pytest.approx(0.50)


def test_ready_cumulative_settlement_mode_applies_declared_factor_and_zero() -> None:
    passport = _passport(
        "cumulative_settlement",
        initial_reading=None,
        correction_factor=1.02,
        zero_correction_mm=0.10,
        max_increment_mm=None,
    )
    result = _process([0.00, 0.25, 0.50], passport)

    audit = indicator_audit_frame(result)
    assert audit["turn_number"].tolist() == [0, 0, 0]
    assert audit["computed_increment_mm"].tolist() == pytest.approx([0.0, 0.255, 0.255])
    assert audit["applied_correction_mm"].tolist() == pytest.approx([0.10, 0.10, 0.10])
    assert audit["cumulative_settlement_mm"].tolist() == pytest.approx([0.10, 0.355, 0.61])
    assert (indicator_event_frame(result)["event_type"] == "zero_correction_applied").sum() == 1


def test_ready_settlement_reverse_motion_is_also_logged() -> None:
    passport = _passport(
        "cumulative_settlement",
        initial_reading=None,
        zero_correction_mm=0.0,
        max_increment_mm=None,
    )
    result = _process([0.00, 0.50, 0.20], passport)

    audit = indicator_audit_frame(result)
    assert audit.loc[2, "processing_status"] == "error"
    assert any(
        issue.code == "unexpected_indicator_reverse_motion"
        for issue in result.issues
    )
    assert "unexpected_reverse_motion" in indicator_event_frame(result)["event_type"].tolist()


def test_prepare_measurements_keeps_point_audit_and_passport_tables() -> None:
    raw = _frame([0.20, 9.55, 8.90])
    metadata = {
        "stamp_shape": "custom",
        "stamp_area_m2": 0.1,
        "indicator_passports": {
            "indicator_1": _passport(
                "decreasing_wrapped",
                initial_reading=0.20,
                max_increment_mm=1.0,
                zero_correction_mm=0.05,
            )
        },
    }

    prepared, issues = prepare_measurements(raw, metadata, strict_metadata=False)

    assert not [issue for issue in issues if issue.level == "error"]
    assert prepared["settlement_raw_mm"].tolist() == pytest.approx([0.05, 0.70, 1.35])
    audit = indicator_audit_frame(prepared)
    passports = indicator_passport_frame(prepared)
    assert len(audit) == len(raw)
    assert {
        "original_reading",
        "turn_number",
        "computed_increment_mm",
        "cumulative_settlement_mm",
        "applied_correction_mm",
        "warning",
        "conversion_method",
    }.issubset(audit.columns)
    assert passports.loc[0, "serial_number"] == "IND-001"
    assert passports.loc[0, "division_mm"] == pytest.approx(0.01)


def test_channels_are_calibrated_before_tilt_plane_fit() -> None:
    raw = pd.DataFrame(
        {
            "test_id": ["T1"],
            "stage": [0],
            "load": [0.0],
            "branch": ["loading"],
            "indicator_1": [1.0],
            "indicator_2": [10.0],
            "indicator_3": [100.0],
        }
    )
    metadata = {
        "stamp_shape": "custom",
        "stamp_area_m2": 0.1,
        "indicator_passports": {
            "indicator_1": _passport(
                "cumulative_settlement",
                initial_reading=None,
                max_increment_mm=None,
                correction_factor=1.0,
                serial_number="IND-1",
            ),
            "indicator_2": _passport(
                "cumulative_settlement",
                initial_reading=None,
                max_increment_mm=None,
                correction_factor=0.1,
                serial_number="IND-2",
            ),
            "indicator_3": _passport(
                "cumulative_settlement",
                initial_reading=None,
                max_increment_mm=None,
                correction_factor=0.01,
                serial_number="IND-3",
            ),
        },
    }

    prepared, issues = prepare_measurements(raw, metadata, strict_metadata=False)
    tilt = center_and_tilt(
        prepared,
        {
            "indicator_1": (0.0, 0.0),
            "indicator_2": (100.0, 0.0),
            "indicator_3": (0.0, 100.0),
        },
    )

    assert not [issue for issue in issues if issue.level == "error"]
    assert np.isnan(prepared.loc[0, "indicator_calibration_factor"])
    assert tilt.loc[0, "center_settlement_mm"] == pytest.approx(1.0)
    assert tilt.loc[0, "tilt_magnitude_mm_per_mm"] == pytest.approx(0.0, abs=1e-12)
