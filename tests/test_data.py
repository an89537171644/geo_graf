from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from soilstamp.analysis import center_and_tilt
from soilstamp.data import (
    AuditTrail,
    apply_manual_point_correction,
    apply_settlement_correction,
    failure_summary,
    prepare_measurements as _prepare_measurements,
)


META = {"stamp_diameter_mm": 300.0, "lever_ratio": 1.0}


def prepare_measurements(*args, **kwargs):
    """Most unit fixtures intentionally exercise the compatibility layer."""

    kwargs.setdefault("strict_metadata", False)
    return _prepare_measurements(*args, **kwargs)


def test_public_prepare_is_default_safe_for_missing_physical_metadata() -> None:
    raw = pd.DataFrame(
        {"test_id": ["T1"], "stage": [1], "load": [1.0], "settlement": [0.2]}
    )
    prepared, issues = _prepare_measurements(raw, {})

    assert any(item.code == "missing_explicit_metadata" for item in issues)
    assert any(item.code == "missing_explicit_geometry" for item in issues)
    assert "settlement_raw_mm" not in prepared.columns


def test_missing_zero_is_not_fabricated() -> None:
    raw = pd.DataFrame(
        {
            "test_id": ["T1", "T1"],
            "stage": [1, 2],
            "load": [5.0, 10.0],
            "settlement": [0.6, 1.0],
            "status": ["stable", "stable"],
        }
    )
    prepared, issues = prepare_measurements(raw, META)
    shifted, correction_issues = apply_settlement_correction(prepared, "zero_shifted")
    assert len(shifted) == len(raw)
    assert not np.isclose(shifted["F_kN"], 0.0).any()
    assert shifted["settlement_mm"].tolist() == [0.6, 1.0]
    assert any(item.code == "missing_measured_zero" for item in correction_issues)
    assert not [item for item in issues if item.level == "error"]


def test_measured_zero_shift_preserves_raw_layer() -> None:
    raw = pd.DataFrame(
        {
            "test_id": ["T1", "T1"],
            "stage": [1, 2],
            "load": [0.0, 10.0],
            "settlement": [0.25, 0.75],
        }
    )
    prepared, _ = prepare_measurements(raw, META)
    shifted, _ = apply_settlement_correction(prepared, "zero_shifted")
    assert shifted["settlement_raw_mm"].tolist() == [0.25, 0.75]
    assert shifted["settlement_mm"].tolist() == [0.0, 0.5]


def test_failure_without_settlement_is_interval_censored() -> None:
    raw = pd.DataFrame(
        {
            "test_id": ["T1", "T1", "T1"],
            "stage": [1, 2, 3],
            "load": [100.0, 200.0, 250.0],
            "settlement": [1.0, 2.0, np.nan],
            "status": ["stable", "stable", "ушла"],
        }
    )
    prepared, _ = prepare_measurements(raw, META)
    result = failure_summary(prepared).iloc[0]
    assert bool(result["failure_reached"])
    assert not bool(result["right_censored"])
    assert result["F_last_stable"] == 200.0
    assert result["F_failure_step"] == 250.0
    assert result["Fu_lower"] == 200.0
    assert result["Fu_upper"] == 250.0
    assert pd.isna(result["s_failure"])
    assert prepared.loc[2, "settlement_mm"] != prepared.loc[2, "settlement_mm"]


def test_right_censoring_stores_lower_bound() -> None:
    raw = pd.DataFrame(
        {
            "test_id": ["T1", "T1"],
            "stage": [1, 2],
            "load": [100.0, 300.0],
            "settlement": [1.0, 4.0],
            "status": ["stable", "stable"],
        }
    )
    prepared, _ = prepare_measurements(raw, META)
    result = failure_summary(prepared).iloc[0]
    assert bool(result["right_censored"])
    assert result["Fu_lower"] == 300.0
    assert pd.isna(result["Fu_upper"])
    assert result["display"] == "Fu > 300 кН"


def test_unloading_order_and_reverse_indicator_are_preserved() -> None:
    raw = pd.DataFrame(
        {
            "test_id": ["T1"] * 5,
            "stage": [1, 2, 3, 4, 5],
            "load": [0.0, 100.0, 200.0, 100.0, 0.0],
            "settlement": [0.2, 1.0, 2.4, 1.9, 1.3],
        }
    )
    prepared, _ = prepare_measurements(raw, META)
    assert prepared["F_kN"].tolist() == raw["load"].tolist()
    assert prepared["branch"].tolist() == ["loading", "loading", "loading", "unloading", "unloading"]
    assert prepared["settlement_mm"].tolist()[-2:] == [1.9, 1.3]


def test_manual_correction_requires_reason_and_appends_audit() -> None:
    raw = pd.DataFrame(
        {"test_id": ["T1"], "stage": [1], "load": [1.0], "settlement": [0.5]}
    )
    prepared, _ = prepare_measurements(raw, META)
    audit = AuditTrail()
    with pytest.raises(ValueError):
        apply_manual_point_correction(
            prepared,
            test_id="T1",
            sequence_no=0,
            corrected_settlement_mm=0.4,
            reason="",
            audit=audit,
        )
    corrected = apply_manual_point_correction(
        prepared,
        test_id="T1",
        sequence_no=0,
        corrected_settlement_mm=0.4,
        reason="Поверочная поправка",
        audit=audit,
    )
    assert corrected.loc[0, "settlement_mm"] == 0.4
    assert prepared.loc[0, "settlement_mm"] == 0.5
    assert len(audit.events) == 1
    assert audit.events[0]["action"] == "manual_point_correction"


def test_missing_geometry_is_not_fabricated() -> None:
    raw = pd.DataFrame(
        {"test_id": ["T1"], "stage": [1], "load": [1.0], "settlement": [0.5]}
    )
    prepared, issues = prepare_measurements(raw, {})
    assert prepared["D_mm"].isna().all()
    assert prepared["p_kPa"].isna().all()
    assert any(issue.code == "missing_stamp_area" for issue in issues)


def test_force_pressure_and_settlement_units_are_converted() -> None:
    raw = pd.DataFrame(
        {"test_id": ["T1"], "stage": [1], "load": [1000.0], "settlement": [0.1]}
    )
    prepared, issues = prepare_measurements(
        raw,
        {
            "load_kind": "force",
            "load_unit": "N",
            "settlement_unit": "cm",
            "stamp_shape": "custom",
            "stamp_area_m2": 0.1,
        },
    )
    assert not [issue for issue in issues if issue.level == "error"]
    assert np.isclose(prepared.loc[0, "F_kN"], 1.0)
    assert np.isclose(prepared.loc[0, "p_kPa"], 10.0)
    assert np.isclose(prepared.loc[0, "settlement_mm"], 1.0)

    pressure_raw = raw.assign(load=0.1, settlement=0.001)
    pressure, pressure_issues = prepare_measurements(
        pressure_raw,
        {
            "load_kind": "pressure",
            "load_unit": "MPa",
            "settlement_unit": "m",
            "stamp_shape": "custom",
            "stamp_area_m2": 0.1,
        },
    )
    assert not [issue for issue in pressure_issues if issue.level == "error"]
    assert np.isclose(pressure.loc[0, "p_kPa"], 100.0)
    assert np.isclose(pressure.loc[0, "F_kN"], 10.0)
    assert np.isclose(pressure.loc[0, "settlement_mm"], 1.0)


def test_load_zero_factor_lever_and_kgf_are_applied_explicitly() -> None:
    raw = pd.DataFrame(
        {"test_id": ["T1"], "stage": [1], "load": [102.0], "settlement": [0.1]}
    )
    prepared, issues = prepare_measurements(
        raw,
        {
            "load_kind": "force",
            "load_unit": "kgf",
            "load_zero": 2.0,
            "load_factor": 0.5,
            "lever_ratio": 10.0,
            "stamp_area_m2": 0.1,
        },
    )

    assert not [item for item in issues if item.level == "error"]
    assert np.isclose(prepared.loc[0, "F_kN"], 100.0 * 0.5 * 10.0 * 0.00980665)


def test_zero_shift_does_not_use_final_unloading_zero() -> None:
    raw = pd.DataFrame(
        {
            "test_id": ["T1", "T1", "T1"],
            "stage": [1, 2, 3],
            "load": [10.0, 20.0, 0.0],
            "settlement": [0.5, 1.5, 0.9],
            "branch": ["loading", "loading", "unloading"],
        }
    )
    prepared, _ = prepare_measurements(raw, META)
    shifted, issues = apply_settlement_correction(prepared, "zero_shifted")
    assert shifted["settlement_mm"].tolist() == [0.5, 1.5, 0.9]
    assert any(issue.code == "missing_measured_zero" for issue in issues)


def test_branch_suggestion_respects_load_resolution() -> None:
    raw = pd.DataFrame(
        {
            "test_id": ["T1"] * 4,
            "stage": [1, 2, 3, 4],
            "load": [0.0, 10.0, 9.9999, 10.0],
            "settlement": [0.0, 1.0, 1.1, 1.2],
        }
    )
    prepared, _ = prepare_measurements(raw, {**META, "load_resolution_kN": 0.01})
    assert prepared["branch_suggested"].tolist() == ["loading", "loading", "hold", "hold"]


def test_failure_negation_and_unstable_point_are_not_misclassified() -> None:
    raw = pd.DataFrame(
        {
            "test_id": ["T1"] * 4,
            "stage": [1, 2, 3, 4],
            "load": [100.0, 200.0, 250.0, 300.0],
            "settlement": [1.0, 2.0, 3.0, np.nan],
            "status": ["stable", "unstable", "no failure observed", "failure"],
        }
    )
    prepared, _ = prepare_measurements(raw, META)
    assert prepared["is_failure"].tolist() == [False, False, False, True]
    summary = failure_summary(prepared).iloc[0]
    assert summary["F_last_stable"] == 250.0


def test_test_specific_diameter_recomputes_inherited_area() -> None:
    area_300 = np.pi * 0.3**2 / 4.0
    raw = pd.DataFrame(
        {
            "test_id": ["T1", "T2"],
            "stage": [1, 1],
            "load": [10.0, 10.0],
            "settlement": [1.0, 1.0],
        }
    )
    prepared, issues = prepare_measurements(
        raw,
        {
            "stamp_diameter_mm": 300.0,
            "stamp_area_m2": area_300,
            "tests": {"T2": {"stamp_diameter_mm": 600.0}},
        },
    )
    assert not [issue for issue in issues if issue.level == "error"]
    p1, p2 = prepared.set_index("test_id")["p_kPa"].loc[["T1", "T2"]]
    assert np.isclose(p1 / p2, 4.0)


def test_audit_events_are_deep_copies() -> None:
    audit = AuditTrail()
    parameters = {"nested": {"value": 1}}
    returned = audit.record(
        "test",
        scope="T1",
        reason="Проверка",
        parameters=parameters,
    )
    parameters["nested"]["value"] = 50
    returned["parameters"]["nested"]["value"] = 75
    exported = audit.events
    exported[0]["parameters"]["nested"]["value"] = 99
    assert audit.events[0]["parameters"]["nested"]["value"] == 1


def test_pressure_input_without_area_keeps_pressure_branches_and_zero_shift() -> None:
    raw = pd.DataFrame(
        {
            "test_id": ["T1"] * 4,
            "stage": range(4),
            "load": [0.0, 100.0, 50.0, 0.0],
            "settlement": [0.2, 1.2, 0.9, 0.6],
        }
    )
    prepared, issues = prepare_measurements(
        raw,
        {
            "load_kind": "pressure",
            "load_unit": "kPa",
            "pressure_resolution_kPa": 0.1,
        },
    )
    assert prepared["p_kPa"].tolist() == [0.0, 100.0, 50.0, 0.0]
    assert prepared["F_kN"].isna().all()
    assert prepared["branch_suggested"].tolist() == ["loading", "loading", "unloading", "unloading"]
    assert any(issue.code == "missing_stamp_area_for_force" for issue in issues)
    shifted, _ = apply_settlement_correction(prepared, "zero_shifted")
    assert np.allclose(shifted["settlement_mm"], [0.0, 1.0, 0.7, 0.4])

    failure_raw = pd.DataFrame(
        {
            "test_id": ["P1"] * 3,
            "stage": range(3),
            "load": [100.0, 200.0, 250.0],
            "settlement": [1.0, 2.0, np.nan],
            "status": ["stable", "stable", "failure"],
        }
    )
    pressure_failure, _ = prepare_measurements(
        failure_raw, {"load_kind": "pressure", "load_unit": "kPa"}
    )
    bounds = failure_summary(pressure_failure).iloc[0]
    assert pd.isna(bounds["Fu_lower"])
    assert bounds["pu_lower"] == 200.0
    assert bounds["pu_upper"] == 250.0
    assert bounds["display"] == "200 < pu ≤ 250 кПа"


def test_failure_bounds_use_stable_load_steps_even_without_settlement() -> None:
    raw = pd.DataFrame(
        {
            "test_id": ["T1", "T1", "T1"],
            "stage": [1, 2, 3],
            "load": [100.0, 200.0, 250.0],
            "settlement": [1.0, np.nan, np.nan],
            "status": ["stable", "stable", "failure"],
        }
    )
    prepared, _ = prepare_measurements(raw, META)
    summary = failure_summary(prepared).iloc[0]
    assert summary["F_last_stable"] == 200.0
    assert pd.isna(summary["s_last_stable"])

    censored_raw = raw.iloc[:2].assign(status="stable")
    censored, _ = prepare_measurements(censored_raw, META)
    censored_summary = failure_summary(censored).iloc[0]
    assert censored_summary["Fu_lower"] == 200.0


def test_invalid_numeric_metadata_and_manual_nonfinite_values_are_rejected() -> None:
    raw = pd.DataFrame(
        {"test_id": ["T1"], "stage": [1], "load": [1.0], "settlement": [0.5]}
    )
    _, issues = prepare_measurements(raw, {**META, "gamma_kN_m3": "bad"})
    assert any(issue.level == "error" and "gamma" in issue.message for issue in issues)
    prepared, _ = prepare_measurements(raw, META)
    audit = AuditTrail()
    with pytest.raises(ValueError, match="конечным"):
        apply_manual_point_correction(
            prepared,
            test_id="T1",
            sequence_no=0,
            corrected_settlement_mm=np.inf,
            reason="bad",
            audit=audit,
        )
    with pytest.raises(ValueError, match="конечн"):
        apply_settlement_correction(
            prepared,
            "seating_corrected",
            seating_offsets_mm={"T1": np.nan},
        )


def test_nonfinite_and_nonnumeric_measurements_are_blocking_errors() -> None:
    raw = pd.DataFrame(
        {
            "test_id": ["T1", "T1"],
            "stage": [1, 2],
            "load": [1.0, np.inf],
            "settlement": ["bad", np.inf],
        }
    )
    _, issues = prepare_measurements(raw, META)
    codes = {issue.code for issue in issues if issue.level == "error"}
    assert "non_numeric_load" in codes
    assert "invalid_measurement" in codes


def test_not_failed_status_is_not_a_failure() -> None:
    raw = pd.DataFrame(
        {
            "test_id": ["T1"],
            "stage": [1],
            "load": [10.0],
            "settlement": [1.0],
            "status": ["not failed"],
        }
    )
    prepared, _ = prepare_measurements(raw, META)
    assert not bool(prepared.loc[0, "is_failure"])


@pytest.mark.parametrize(
    "status",
    ["failure was not observed", "no signs of failure", "failed: no"],
)
def test_free_text_failure_phrases_are_not_automatic_events(status: str) -> None:
    raw = pd.DataFrame(
        {
            "test_id": ["T1"],
            "stage": [1],
            "load": [10.0],
            "settlement": [1.0],
            "status": [status],
        }
    )
    prepared, issues = prepare_measurements(raw, META)
    assert not bool(prepared.loc[0, "is_failure"])
    assert any(issue.code == "unaccepted_status" for issue in issues)


def _channel_passport(
    *,
    instrument_id: str = "IND-T1",
    correction_factor: float = 1.0,
    cumulative_sign: float = 1.0,
    x_mm: float | None = None,
    y_mm: float | None = None,
) -> dict:
    result = {
        "type": "ИЧ-10",
        "serial_number": instrument_id,
        "instrument_id": instrument_id,
        "range_mm": 10.0,
        "division_mm": 0.01,
        "correction_factor": correction_factor,
        "verification_date": "2026-01-01",
        "verification_valid_until": "2030-01-01",
        "mode": "cumulative_settlement",
        "initial_reading": None,
        "initial_turn": 0,
        "zero_correction_mm": 0.0,
        "max_increment_mm": None,
        "reverse_tolerance_mm": 0.02,
        "travel_range_mm": 50.0,
        "cumulative_sign": cumulative_sign,
    }
    if x_mm is not None and y_mm is not None:
        result.update({"x_mm": x_mm, "y_mm": y_mm})
    return result


def _indicator_metadata(**overrides) -> dict:
    factor = float(overrides.pop("indicator_calibration_factor", 1.0))
    sign = float(overrides.pop("indicator_sign", 1.0))
    instrument = str(overrides.pop("indicator_instrument_id", "IND-T1"))
    overrides.pop("indicator_unit", None)
    result = {
        **META,
        "settlement_unit": "mm",
        "indicator_resolution_mm": 0.01,
        "experiment_date": "2026-02-01",
        "metrology_status": "confirmed",
        "settlement_aggregation": "primary_channel",
        "settlement_aggregation_channels": ["indicator_1"],
        "settlement_primary_channel": "indicator_1",
        "settlement_missing_channel_policy": "block",
        "indicator_passports": {
            "indicator_1": _channel_passport(
                instrument_id=instrument,
                correction_factor=factor,
                cumulative_sign=sign,
            )
        },
    }
    result.update(overrides)
    return result


def test_indicator_only_without_explicit_mode_is_blocked() -> None:
    raw = pd.DataFrame(
        {"test_id": ["T1"], "stage": [1], "load": [1.0], "indicator_1": [10.0]}
    )
    _, issues = prepare_measurements(raw, META)
    issue = next(item for item in issues if item.code == "indicator_mode_not_confirmed")
    assert issue.rows == [0]
    assert issue.blocks_processing


def test_indicator_fallback_is_blocked_only_where_direct_settlement_is_missing() -> None:
    raw = pd.DataFrame(
        {
            "test_id": ["T1", "T1"],
            "stage": [1, 2],
            "load": [1.0, 2.0],
            "settlement": [0.25, np.nan],
            "indicator_1": [900.0, 10.0],
        }
    )
    _, issues = prepare_measurements(raw, META)
    issue = next(item for item in issues if item.code == "indicator_mode_not_confirmed")
    assert issue.rows == [1]


def test_direct_settlement_does_not_require_indicator_calibration() -> None:
    raw = pd.DataFrame(
        {
            "test_id": ["T1"],
            "stage": [1],
            "load": [1.0],
            "settlement": [0.25],
            "indicator_1": [900.0],
        }
    )
    prepared, issues = prepare_measurements(raw, META)
    assert not any(item.level == "error" for item in issues)
    assert prepared.loc[0, "settlement_raw_mm"] == pytest.approx(0.25)
    assert not bool(prepared.loc[0, "indicator_calibration_confirmed"])
    assert any(item.code == "uncalibrated_indicators_ignored" for item in issues)


def test_explicit_indicator_unit_and_factor_are_applied_independently() -> None:
    raw = pd.DataFrame(
        {
            "test_id": ["T1", "T1"],
            "stage": [1, 2],
            "load": [1.0, 2.0],
            "indicator_1": [10.0, 20.0],
        }
    )
    metadata = _indicator_metadata(
        settlement_unit="cm", indicator_unit="mm", indicator_calibration_factor=0.1
    )
    prepared, issues = prepare_measurements(raw, metadata)
    assert not any(item.level == "error" for item in issues)
    assert prepared["settlement_raw_mm"].tolist() == pytest.approx([1.0, 2.0])
    assert prepared["indicator_scale_to_mm"].tolist() == [1.0, 1.0]
    assert prepared["indicator_calibration_confirmed"].all()


def test_reference_indicator_requires_explicit_reference_passport() -> None:
    raw = pd.DataFrame(
        {
            "test_id": ["T1"],
            "stage": [1],
            "load": [1.0],
            "indicator_1": [10.0],
            "reference_indicator": [1.0],
        }
    )
    _, issues = prepare_measurements(raw, _indicator_metadata())
    assert any(
        item.code == "missing_reference_indicator_passport"
        for item in issues
    )


def test_empty_optional_reference_column_does_not_require_reference_sign() -> None:
    raw = pd.DataFrame(
        {
            "test_id": ["T1"],
            "stage": [1],
            "load": [1.0],
            "indicator_1": [10.0],
            "reference_indicator": [np.nan],
        }
    )
    prepared, issues = prepare_measurements(raw, _indicator_metadata())

    assert not any(item.level == "error" for item in issues)
    assert prepared.loc[0, "settlement_raw_mm"] == pytest.approx(10.0)
    assert not bool(prepared.loc[0, "reference_channel_used"])


def test_uncalibrated_auxiliary_indicators_are_ignored_for_tilt() -> None:
    raw = pd.DataFrame(
        {
            "test_id": ["T1"],
            "stage": [1],
            "load": [1.0],
            "settlement": [0.25],
            "indicator_1": [1.0],
            "indicator_2": [1.0],
            "indicator_3": [1.0],
        }
    )
    prepared, _ = prepare_measurements(raw, META)
    tilt = center_and_tilt(
        prepared,
        {
            "indicator_1": (0.0, 0.0),
            "indicator_2": (100.0, 0.0),
            "indicator_3": (0.0, 100.0),
        },
    )
    assert tilt.empty


def test_center_and_tilt_uses_indicator_scale_and_calibration_factor() -> None:
    raw = pd.DataFrame(
        {
            "test_id": ["T1"],
            "stage": [1],
            "load": [1.0],
            "indicator_1": [10.0],
            "indicator_2": [10.0],
            "indicator_3": [10.0],
        }
    )
    metadata = _indicator_metadata(
        settlement_unit="cm",
        indicator_calibration_factor=0.1,
    )
    metadata["settlement_aggregation"] = "plane_center"
    metadata["settlement_aggregation_channels"] = [
        "indicator_1",
        "indicator_2",
        "indicator_3",
    ]
    metadata["indicator_passports"] = {
        "indicator_1": _channel_passport(
            instrument_id="IND-1", correction_factor=0.1, x_mm=0.0, y_mm=0.0
        ),
        "indicator_2": _channel_passport(
            instrument_id="IND-2", correction_factor=0.1, x_mm=100.0, y_mm=0.0
        ),
        "indicator_3": _channel_passport(
            instrument_id="IND-3", correction_factor=0.1, x_mm=0.0, y_mm=100.0
        ),
    }
    prepared, issues = prepare_measurements(raw, metadata)
    tilt = center_and_tilt(
        prepared,
        {
            "indicator_1": (0.0, 0.0),
            "indicator_2": (100.0, 0.0),
            "indicator_3": (0.0, 100.0),
        },
    )

    assert not any(item.level == "error" for item in issues)
    assert tilt.loc[0, "center_settlement_mm"] == pytest.approx(1.0)
    assert tilt.loc[0, "indicator_calibration_factor"] == pytest.approx(0.1)


def test_test_specific_indicator_calibration_is_resolved_per_test() -> None:
    raw = pd.DataFrame(
        {
            "test_id": ["T1", "T2"],
            "stage": [1, 1],
            "load": [1.0, 1.0],
            "indicator_1": [10.0, 10.0],
        }
    )
    metadata = _indicator_metadata(
        tests={
            "T1": {
                "indicator_passports": {
                    "indicator_1": _channel_passport(
                        instrument_id="IND-T1", correction_factor=0.1
                    )
                },
            },
            "T2": {
                "indicator_passports": {
                    "indicator_1": _channel_passport(
                        instrument_id="IND-T2",
                        correction_factor=2.0,
                        cumulative_sign=-1.0,
                    )
                },
            },
        }
    )
    prepared, issues = prepare_measurements(raw, metadata)

    assert not any(item.level == "error" for item in issues)
    assert prepared["settlement_raw_mm"].tolist() == pytest.approx([1.0, -20.0])
    assert prepared["indicator_instrument_id"].tolist() == ["IND-T1", "IND-T2"]


def test_reference_correction_is_numeric_and_missing_reference_stays_nan() -> None:
    raw = pd.DataFrame(
        {
            "test_id": ["T1", "T1"],
            "stage": [1, 2],
            "load": [1.0, 2.0],
            "indicator_1": [10.0, 10.0],
            "reference_indicator": [1.0, np.nan],
        }
    )
    metadata = _indicator_metadata()
    metadata["indicator_passports"]["reference_indicator"] = _channel_passport(
        instrument_id="REF-T1", cumulative_sign=-1.0
    )
    prepared, issues = prepare_measurements(raw, metadata)

    assert not any(item.level == "error" for item in issues)
    assert prepared.loc[0, "settlement_raw_mm"] == pytest.approx(9.0)
    assert pd.isna(prepared.loc[1, "settlement_raw_mm"])
    assert any(item.code == "missing_reference_indicator" for item in issues)


def test_prepared_layer_persists_aggregation_and_metrology_records() -> None:
    raw = pd.DataFrame(
        {
            "test_id": ["T1"],
            "stage": [1],
            "load": [1.0],
            "indicator_1": [2.0],
        }
    )
    prepared, issues = prepare_measurements(raw, _indicator_metadata())

    assert not [issue for issue in issues if bool(issue.blocks_processing)]
    assert prepared.loc[0, "aggregation_method"] == "primary_channel"
    assert prepared.loc[0, "aggregation_status"] == "ok"
    assert prepared.loc[0, "channels_required"] == '["indicator_1"]'
    assert prepared.attrs["indicator_processing_schema"] == "indicator-processing/2.0"
    assert len(prepared.attrs["indicator_aggregation_results"]) == 1
    evaluation = prepared.attrs["metrology_evaluations"][0]
    assert evaluation["verification_status"] == "valid_at_experiment"
    assert evaluation["verification_evaluation_date"] == "2026-02-01"
    assert evaluation["verification_evaluation_rule"]


def test_expired_channel_is_not_marked_calibration_confirmed_or_authoritative() -> None:
    raw = pd.DataFrame(
        {
            "test_id": ["T1"],
            "stage": [1],
            "load": [1.0],
            "indicator_1": [2.0],
        }
    )
    metadata = _indicator_metadata(experiment_date="2031-01-01")
    prepared, issues = prepare_measurements(raw, metadata)

    assert any(
        issue.code == "indicator_verification_expired_at_experiment"
        for issue in issues
    )
    assert not bool(prepared.loc[0, "indicator_calibration_confirmed"])
    assert prepared.loc[0, "aggregation_status"] == "blocked_metrology_status"
    assert pd.isna(prepared.loc[0, "settlement_raw_mm"])


def test_reference_used_flag_describes_actual_aggregation_not_raw_presence() -> None:
    raw = pd.DataFrame(
        {
            "test_id": ["T1"],
            "stage": [1],
            "load": [1.0],
            "settlement": [0.25],
            "indicator_1": [10.0],
            "reference_indicator": [1.0],
        }
    )
    metadata = _indicator_metadata()
    metadata["indicator_passports"]["reference_indicator"] = _channel_passport(
        instrument_id="REF-T1", cumulative_sign=-1.0
    )
    prepared, issues = prepare_measurements(raw, metadata)

    assert not [issue for issue in issues if bool(issue.blocks_processing)]
    assert prepared.loc[0, "aggregation_status"] == "not_applied_direct_settlement"
    assert not bool(prepared.loc[0, "reference_channel_used"])


def test_invalid_auxiliary_indicator_is_nonblocking_for_direct_settlement() -> None:
    raw = pd.DataFrame(
        {
            "test_id": ["T1"],
            "stage": [1],
            "load": [1.0],
            "settlement": [0.25],
            "indicator_1": ["not-a-number"],
        }
    )

    prepared, issues = prepare_measurements(raw, META)

    assert not [issue for issue in issues if bool(issue.blocks_processing)]
    assert any(
        issue.code == "invalid_auxiliary_indicator_ignored" for issue in issues
    )
    assert prepared.loc[0, "settlement_raw_mm"] == pytest.approx(0.25)


def test_invalid_indicator_is_blocking_when_same_test_needs_fallback() -> None:
    raw = pd.DataFrame(
        {
            "test_id": ["T1", "T1"],
            "stage": [1, 2],
            "load": [1.0, 2.0],
            "settlement": [0.25, np.nan],
            "indicator_1": ["not-a-number", 1.0],
        }
    )

    _, issues = prepare_measurements(raw, _indicator_metadata())

    issue = next(issue for issue in issues if issue.code == "invalid_measurement")
    assert issue.rows == [0]
    assert issue.column == "indicator_1"
    assert bool(issue.blocks_processing)


def test_no_aggregation_never_falls_back_to_indicator_value() -> None:
    raw = pd.DataFrame(
        {
            "test_id": ["T1"],
            "stage": [1],
            "load": [1.0],
            "indicator_1": [2.0],
        }
    )
    metadata = _indicator_metadata()
    metadata.update(
        {
            "settlement_aggregation": "no_aggregation",
            "settlement_aggregation_channels": [],
            "settlement_primary_channel": None,
        }
    )

    prepared, issues = prepare_measurements(raw, metadata)

    assert not [issue for issue in issues if bool(issue.blocks_processing)]
    assert pd.isna(prepared.loc[0, "settlement_raw_mm"])
    assert prepared.loc[0, "aggregation_status"] == "no_aggregation"


def test_verification_issues_are_retained_for_each_channel() -> None:
    raw = pd.DataFrame(
        {
            "test_id": ["T1"],
            "stage": [1],
            "load": [1.0],
            "indicator_1": [1.0],
            "indicator_2": [1.0],
        }
    )
    metadata = _indicator_metadata(experiment_date="2031-01-01")
    second = _channel_passport(instrument_id="IND-T2")
    metadata["indicator_passports"]["indicator_2"] = second
    metadata.update(
        {
            "settlement_aggregation": "all_channels_mean",
            "settlement_aggregation_channels": ["indicator_1", "indicator_2"],
            "settlement_primary_channel": None,
        }
    )

    _, issues = prepare_measurements(raw, metadata)
    expired = [
        issue
        for issue in issues
        if issue.code == "indicator_verification_expired_at_experiment"
    ]

    assert {issue.column for issue in expired} == {"indicator_1", "indicator_2"}
