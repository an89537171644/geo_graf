from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from soilstamp.analysis import center_and_tilt
from soilstamp.data import prepare_measurements
from soilstamp.indicators import (
    indicator_aggregation_frame,
    indicator_audit_frame,
    indicator_event_frame,
    indicator_passport_frame,
    process_indicator_frame,
    resolve_settlement_aggregation,
)


def _passport(mode: str, **overrides) -> dict:
    result = {
        "type": "ИЧ-10",
        "serial_number": "IND-001",
        "instrument_id": "INST-001",
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
        "initial_turn": 0,
        "cumulative_sign": 1.0,
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
    metadata = {
        "experiment_date": "2026-02-01",
        "metrology_status": "confirmed",
        "settlement_aggregation": "primary_channel",
        "settlement_aggregation_channels": ["indicator_1"],
        "settlement_primary_channel": "indicator_1",
        "settlement_missing_channel_policy": "block",
        "indicator_passports": {"indicator_1": passport},
    }
    return process_indicator_frame(frame, metadata)


def _aggregation_metadata(
    passports: dict[str, dict],
    *,
    method: str,
    channels: list[str],
    primary: str | None = None,
    missing_policy: str = "block",
    experiment_date: str | None = "2026-02-01",
) -> dict:
    metadata = {
        "metrology_status": "confirmed",
        "settlement_aggregation": method,
        "settlement_aggregation_channels": channels,
        "settlement_primary_channel": primary,
        "settlement_missing_channel_policy": missing_policy,
        "indicator_passports": passports,
    }
    if experiment_date is not None:
        metadata["experiment_date"] = experiment_date
    return metadata


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
        "experiment_date": "2026-02-01",
        "metrology_status": "confirmed",
        "settlement_aggregation": "primary_channel",
        "settlement_aggregation_channels": ["indicator_1"],
        "settlement_primary_channel": "indicator_1",
        "settlement_missing_channel_policy": "block",
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
        "experiment_date": "2026-02-01",
        "metrology_status": "confirmed",
        "settlement_aggregation": "plane_center",
        "settlement_aggregation_channels": [
            "indicator_1",
            "indicator_2",
            "indicator_3",
        ],
        "settlement_primary_channel": None,
        "settlement_missing_channel_policy": "block",
        "indicator_passports": {
            "indicator_1": _passport(
                "cumulative_settlement",
                initial_reading=None,
                max_increment_mm=None,
                correction_factor=1.0,
                serial_number="IND-1",
                x_mm=0.0,
                y_mm=0.0,
            ),
            "indicator_2": _passport(
                "cumulative_settlement",
                initial_reading=None,
                max_increment_mm=None,
                correction_factor=0.1,
                serial_number="IND-2",
                x_mm=100.0,
                y_mm=0.0,
            ),
            "indicator_3": _passport(
                "cumulative_settlement",
                initial_reading=None,
                max_increment_mm=None,
                correction_factor=0.01,
                serial_number="IND-3",
                x_mm=0.0,
                y_mm=100.0,
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


def test_all_channels_mean_blocks_missing_channel_without_changing_denominator() -> None:
    frame = pd.DataFrame(
        {
            "test_id": ["T1", "T1"],
            "stage": [0, 1],
            "load": [0.0, 1.0],
            "indicator_1": [1.0, 2.0],
            "indicator_2": [3.0, np.nan],
        }
    )
    passports = {
        "indicator_1": _passport("cumulative_settlement", initial_reading=None),
        "indicator_2": _passport(
            "cumulative_settlement",
            initial_reading=None,
            serial_number="IND-002",
            instrument_id="INST-002",
        ),
    }
    result = process_indicator_frame(
        frame,
        _aggregation_metadata(
            passports,
            method="all_channels_mean",
            channels=["indicator_1", "indicator_2"],
        ),
    )
    aggregation = indicator_aggregation_frame(result)

    assert result.settlement_by_row[0] == pytest.approx(2.0)
    assert result.settlement_by_row[1] is None
    assert aggregation["aggregation_status"].tolist() == [
        "ok",
        "blocked_missing_channels",
    ]
    assert aggregation.loc[1, "channels_required"] == '["indicator_1","indicator_2"]'
    assert aggregation.loc[1, "channels_used"] == '["indicator_1"]'
    assert aggregation.loc[1, "missing_channels"] == '["indicator_2"]'


def test_selected_channels_are_fixed_and_never_replaced_by_available_channel() -> None:
    frame = pd.DataFrame(
        {
            "test_id": ["T1", "T1"],
            "stage": [0, 1],
            "load": [0.0, 1.0],
            "indicator_1": [1.0, 1.0],
            "indicator_2": [3.0, np.nan],
            "indicator_3": [100.0, 100.0],
        }
    )
    passports = {
        f"indicator_{number}": _passport(
            "cumulative_settlement",
            initial_reading=None,
            serial_number=f"IND-00{number}",
            instrument_id=f"INST-00{number}",
            assignment_status=(
                "migration_review_required" if number == 3 else "confirmed"
            ),
        )
        for number in (1, 2, 3)
    }
    result = process_indicator_frame(
        frame,
        _aggregation_metadata(
            passports,
            method="selected_channels_mean",
            channels=["indicator_1", "indicator_2"],
        ),
    )

    assert result.settlement_by_row[0] == pytest.approx(2.0)
    assert result.settlement_by_row[1] is None
    second = indicator_aggregation_frame(result).iloc[1]
    assert json.loads(second["channels_used"]) == ["indicator_1"]
    assert "indicator_3" not in json.loads(second["channels_used"])


def test_primary_and_no_aggregation_are_explicit() -> None:
    frame = pd.DataFrame(
        {
            "test_id": ["T1"],
            "stage": [0],
            "load": [0.0],
            "indicator_1": [1.0],
            "indicator_2": [9.0],
        }
    )
    passports = {
        "indicator_1": _passport("cumulative_settlement", initial_reading=None),
        "indicator_2": _passport(
            "cumulative_settlement",
            initial_reading=None,
            serial_number="IND-002",
            instrument_id="INST-002",
        ),
    }
    primary = process_indicator_frame(
        frame,
        _aggregation_metadata(
            passports,
            method="primary_channel",
            channels=["indicator_1"],
            primary="indicator_2",
        ),
    )
    disabled = process_indicator_frame(
        frame,
        _aggregation_metadata(
            passports,
            method="no_aggregation",
            channels=[],
        ),
    )

    assert primary.settlement_by_row[0] == pytest.approx(9.0)
    assert disabled.settlement_by_row[0] is None
    assert indicator_aggregation_frame(disabled).loc[0, "aggregation_status"] == "no_aggregation"


def test_plane_center_differs_from_arithmetic_mean_for_asymmetric_layout() -> None:
    frame = pd.DataFrame(
        {
            "test_id": ["T1"],
            "stage": [0],
            "load": [0.0],
            "indicator_1": [1.0],
            "indicator_2": [2.0],
            "indicator_3": [3.0],
        }
    )
    positions = ((0.0, 0.0), (100.0, 0.0), (0.0, 100.0))
    passports = {
        f"indicator_{number}": _passport(
            "cumulative_settlement",
            initial_reading=None,
            serial_number=f"IND-00{number}",
            instrument_id=f"INST-00{number}",
            x_mm=positions[number - 1][0],
            y_mm=positions[number - 1][1],
        )
        for number in (1, 2, 3)
    }
    result = process_indicator_frame(
        frame,
        _aggregation_metadata(
            passports,
            method="plane_center",
            channels=["indicator_1", "indicator_2", "indicator_3"],
        ),
    )
    row = indicator_aggregation_frame(result).iloc[0]

    assert result.settlement_by_row[0] == pytest.approx(1.0)
    assert np.mean([1.0, 2.0, 3.0]) != pytest.approx(result.settlement_by_row[0])
    assert row["plane_rank"] == 3
    assert row["plane_residual_rms_mm"] == pytest.approx(0.0, abs=1e-12)


def test_plane_center_symmetric_layout_and_missing_policy_are_auditable() -> None:
    positions = {
        "indicator_1": (-100.0, 0.0),
        "indicator_2": (0.0, 100.0),
        "indicator_3": (100.0, 0.0),
        "indicator_4": (0.0, -100.0),
    }
    passports = {
        channel: _passport(
            "cumulative_settlement",
            initial_reading=None,
            serial_number=channel,
            instrument_id=f"INST-{channel}",
            x_mm=position[0],
            y_mm=position[1],
        )
        for channel, position in positions.items()
    }
    full_frame = pd.DataFrame(
        {
            "test_id": ["T1"],
            "stage": [0],
            "load": [0.0],
            "indicator_1": [1.0],
            "indicator_2": [4.0],
            "indicator_3": [3.0],
            "indicator_4": [0.0],
        }
    )
    frame = pd.DataFrame(
        {
            "test_id": ["T1"],
            "stage": [0],
            "load": [0.0],
            "indicator_1": [0.0],
            "indicator_2": [1.0],
            "indicator_3": [2.0],
            "indicator_4": [np.nan],
        }
    )
    channels = list(positions)
    symmetric = process_indicator_frame(
        full_frame,
        _aggregation_metadata(
            passports,
            method="plane_center",
            channels=channels,
            missing_policy="block",
        ),
    )
    blocked = process_indicator_frame(
        frame,
        _aggregation_metadata(
            passports,
            method="plane_center",
            channels=channels,
            missing_policy="block",
        ),
    )
    allowed = process_indicator_frame(
        frame,
        _aggregation_metadata(
            passports,
            method="plane_center",
            channels=channels,
            missing_policy="allow_if_solvable",
        ),
    )

    assert symmetric.settlement_by_row[0] == pytest.approx(2.0)
    assert indicator_aggregation_frame(symmetric).loc[0, "plane_rank"] == 3
    assert blocked.settlement_by_row[0] is None
    assert indicator_aggregation_frame(blocked).loc[0, "aggregation_status"] == "blocked_missing_channels"
    allowed_row = indicator_aggregation_frame(allowed).iloc[0]
    assert allowed_row["aggregation_status"] == "ok"
    assert allowed_row["plane_rank"] == 3
    assert json.loads(allowed_row["missing_channels"]) == ["indicator_4"]


def test_plane_center_blocks_collinear_coordinates_with_rank() -> None:
    frame = pd.DataFrame(
        {
            "test_id": ["T1"],
            "stage": [0],
            "load": [0.0],
            "indicator_1": [1.0],
            "indicator_2": [2.0],
            "indicator_3": [3.0],
        }
    )
    passports = {
        f"indicator_{number}": _passport(
            "cumulative_settlement",
            initial_reading=None,
            serial_number=f"IND-{number}",
            instrument_id=f"INST-{number}",
            x_mm=float(number),
            y_mm=0.0,
        )
        for number in (1, 2, 3)
    }
    result = process_indicator_frame(
        frame,
        _aggregation_metadata(
            passports,
            method="plane_center",
            channels=["indicator_1", "indicator_2", "indicator_3"],
        ),
    )
    row = indicator_aggregation_frame(result).iloc[0]

    assert result.settlement_by_row[0] is None
    assert row["aggregation_status"] == "blocked_collinear_geometry"
    assert row["plane_rank"] == 2


def test_per_channel_factors_and_initial_readings_are_applied_before_mean() -> None:
    frame = pd.DataFrame(
        {
            "test_id": ["T1", "T1"],
            "stage": [0, 1],
            "load": [0.0, 1.0],
            "indicator_1": [1.0, 2.0],
            "indicator_2": [5.0, 6.0],
        }
    )
    passports = {
        "indicator_1": _passport(
            "increasing", initial_reading=1.0, correction_factor=1.0
        ),
        "indicator_2": _passport(
            "increasing",
            initial_reading=5.0,
            correction_factor=2.0,
            serial_number="IND-002",
            instrument_id="INST-002",
            max_increment_mm=3.0,
        ),
    }
    result = process_indicator_frame(
        frame,
        _aggregation_metadata(
            passports,
            method="selected_channels_mean",
            channels=["indicator_1", "indicator_2"],
        ),
    )

    assert result.settlement_by_row[0] == pytest.approx(0.0)
    assert result.settlement_by_row[1] == pytest.approx(1.5)


@pytest.mark.parametrize(
    ("experiment_date", "valid_from", "valid_until", "expected_status", "aggregate_ok"),
    [
        ("2026-02-01", "2020-01-01", "2025-12-31", "expired_at_experiment", False),
        ("2020-06-01", "2019-01-01", "2021-01-01", "valid_at_experiment", True),
        ("2018-06-01", "2019-01-01", "2021-01-01", "not_yet_valid_at_experiment", False),
        (None, "2019-01-01", "2021-01-01", "review_required", False),
    ],
)
def test_verification_is_evaluated_only_at_experiment_date(
    experiment_date: str | None,
    valid_from: str,
    valid_until: str,
    expected_status: str,
    aggregate_ok: bool,
) -> None:
    passport = _passport(
        "cumulative_settlement",
        initial_reading=None,
        verification_date=valid_from,
        verification_valid_until=valid_until,
    )
    result = process_indicator_frame(
        _frame([1.0]),
        _aggregation_metadata(
            {"indicator_1": passport},
            method="primary_channel",
            channels=["indicator_1"],
            primary="indicator_1",
            experiment_date=experiment_date,
        ),
    )
    passport_row = indicator_passport_frame(result).iloc[0]

    assert passport_row["verification_status"] == expected_status
    assert passport_row["verification_evaluation_date"] == experiment_date
    assert bool(result.settlement_by_row[0] is not None) is aggregate_ok


def test_system_date_cannot_change_verification_or_aggregation(monkeypatch) -> None:
    import soilstamp.indicators as indicator_module

    real_date = indicator_module.date

    class PastDate(real_date):
        @classmethod
        def today(cls):
            return cls(2000, 1, 1)

    class FutureDate(real_date):
        @classmethod
        def today(cls):
            return cls(2099, 1, 1)

    metadata = _aggregation_metadata(
        {
            "indicator_1": _passport(
                "cumulative_settlement",
                initial_reading=None,
                verification_date="2019-01-01",
                verification_valid_until="2021-01-01",
            )
        },
        method="primary_channel",
        channels=["indicator_1"],
        primary="indicator_1",
        experiment_date="2020-06-01",
    )
    monkeypatch.setattr(indicator_module, "date", PastDate)
    past = process_indicator_frame(_frame([1.0]), metadata).to_dict()
    monkeypatch.setattr(indicator_module, "date", FutureDate)
    future = process_indicator_frame(_frame([1.0]), metadata).to_dict()

    assert past == future


def test_reference_is_added_after_vertical_mean_and_never_averaged() -> None:
    frame = pd.DataFrame(
        {
            "test_id": ["T1"],
            "stage": [0],
            "load": [0.0],
            "indicator_1": [10.0],
            "indicator_2": [20.0],
            "reference_indicator": [2.0],
        }
    )
    passports = {
        "indicator_1": _passport("cumulative_settlement", initial_reading=None),
        "indicator_2": _passport(
            "cumulative_settlement",
            initial_reading=None,
            serial_number="IND-002",
            instrument_id="INST-002",
        ),
        "reference_indicator": _passport(
            "cumulative_settlement",
            initial_reading=None,
            serial_number="REF-001",
            instrument_id="REF-INST-001",
            cumulative_sign=-1.0,
        ),
    }
    result = process_indicator_frame(
        frame,
        _aggregation_metadata(
            passports,
            method="selected_channels_mean",
            channels=["indicator_1", "indicator_2"],
        ),
    )
    row = indicator_aggregation_frame(result).iloc[0]

    assert result.settlement_by_row[0] == pytest.approx(13.0)
    assert json.loads(row["channels_required"]) == [
        "indicator_1",
        "indicator_2",
        "reference_indicator",
    ]
    assert row["reference_correction_mm"] == pytest.approx(-2.0)


def test_raw_reference_without_passport_blocks_aggregation_status() -> None:
    frame = pd.DataFrame(
        {
            "test_id": ["T1"],
            "stage": [0],
            "load": [0.0],
            "indicator_1": [10.0],
            "reference_indicator": [1.0],
        }
    )
    result = process_indicator_frame(
        frame,
        _aggregation_metadata(
            {"indicator_1": _passport("cumulative_settlement", initial_reading=None)},
            method="primary_channel",
            channels=["indicator_1"],
            primary="indicator_1",
        ),
    )

    assert result.settlement_by_row[0] is None
    assert any(
        issue.code == "missing_reference_indicator_passport"
        for issue in result.issues
    )
    row = indicator_aggregation_frame(result).iloc[0]
    assert row["aggregation_status"] == "blocked_missing_channels"
    assert json.loads(row["channels_required"])[-1] == "reference_indicator"


def test_direct_settlement_remains_authoritative_with_expired_indicator() -> None:
    frame = _frame([100.0]).assign(settlement=0.25)
    result = process_indicator_frame(
        frame,
        _aggregation_metadata(
            {
                "indicator_1": _passport(
                    "cumulative_settlement",
                    initial_reading=None,
                    verification_valid_until="2025-01-01",
                )
            },
            method="primary_channel",
            channels=["indicator_1"],
            primary="indicator_1",
            experiment_date="2026-01-01",
        ),
    )

    assert result.settlement_by_row[0] is None
    assert (
        indicator_aggregation_frame(result).loc[0, "aggregation_status"]
        == "not_applied_direct_settlement"
    )


def test_direct_settlement_clears_unapplied_plane_diagnostics() -> None:
    frame = pd.DataFrame(
        {
            "test_id": ["T1"],
            "stage": [0],
            "load": [0.0],
            "settlement": [0.25],
            "indicator_1": [1.0],
            "indicator_2": [2.0],
            "indicator_3": [3.0],
        }
    )
    positions = ((0.0, 0.0), (100.0, 0.0), (0.0, 100.0))
    passports = {
        f"indicator_{number}": _passport(
            "cumulative_settlement",
            initial_reading=None,
            serial_number=f"IND-{number}",
            instrument_id=f"INST-{number}",
            x_mm=positions[number - 1][0],
            y_mm=positions[number - 1][1],
        )
        for number in (1, 2, 3)
    }
    result = process_indicator_frame(
        frame,
        _aggregation_metadata(
            passports,
            method="plane_center",
            channels=["indicator_1", "indicator_2", "indicator_3"],
        ),
    )
    row = indicator_aggregation_frame(result).iloc[0]

    assert row["aggregation_status"] == "not_applied_direct_settlement"
    assert row["channels_used"] == "[]"
    assert pd.isna(row["plane_rank"])
    assert pd.isna(row["plane_residual_rms_mm"])
    assert pd.isna(row["tilt_magnitude_mm_per_mm"])
    assert not bool(row["tilt_direction_resolved"])


def test_common_passport_is_migration_review_required_and_not_aggregated() -> None:
    passport = _passport("cumulative_settlement", initial_reading=None)
    metadata = _aggregation_metadata(
        {},
        method="all_channels_mean",
        channels=["indicator_1"],
    )
    metadata.pop("indicator_passports")
    metadata["indicator_passport"] = passport
    result = process_indicator_frame(_frame([1.0]), metadata)

    assert result.settlement_by_row[0] is None
    assert (
        indicator_aggregation_frame(result).loc[0, "aggregation_status"]
        == "migration_review_required"
    )
    assert indicator_passport_frame(result).loc[0, "assignment_status"] == "migration_review_required"


def test_aggregation_resolver_does_not_replace_explicit_empty_active_set() -> None:
    metadata = _aggregation_metadata(
        {"indicator_1": _passport("cumulative_settlement", initial_reading=None)},
        method="all_channels_mean",
        channels=["indicator_1"],
    )

    resolution, issues = resolve_settlement_aggregation(
        metadata, "T1", active_channels=[]
    )

    assert resolution.status == "blocked_invalid_policy"
    assert resolution.channels_required == ()
    assert any(issue.code == "all_channels_basis_mismatch" for issue in issues)


def test_global_draft_blocks_individually_confirmed_channel() -> None:
    passport = _passport(
        "cumulative_settlement",
        initial_reading=None,
        assignment_status="confirmed",
    )
    metadata = _aggregation_metadata(
        {"indicator_1": passport},
        method="primary_channel",
        channels=["indicator_1"],
        primary="indicator_1",
    )
    metadata["metrology_status"] = "draft"

    result = process_indicator_frame(_frame([1.0]), metadata)

    assert result.settlement_by_row[0] is None
    assert (
        indicator_aggregation_frame(result).loc[0, "aggregation_status"]
        == "blocked_metrology_status"
    )
    assert (
        indicator_passport_frame(result).loc[0, "assignment_status"]
        == "review_required"
    )
    assert any(
        issue.code == "metrology_assignment_review_required"
        for issue in result.issues
    )


@pytest.mark.parametrize(
    "bad_index",
    [pd.Index(["row-a"]), pd.Index([0, 0])],
)
def test_process_indicator_frame_rejects_non_unique_or_non_integer_index(
    bad_index: pd.Index,
) -> None:
    frame = _frame([1.0]) if len(bad_index) == 1 else _frame([1.0, 2.0])
    frame.index = bad_index

    result = process_indicator_frame(
        frame,
        _aggregation_metadata(
            {"indicator_1": _passport("cumulative_settlement", initial_reading=None)},
            method="primary_channel",
            channels=["indicator_1"],
            primary="indicator_1",
        ),
    )

    assert not result.audit_rows
    assert not result.aggregation_rows
    assert [issue.code for issue in result.issues] == [
        "invalid_indicator_frame_index"
    ]
