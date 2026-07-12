from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from soilstamp.analysis import calculate_moduli_for_test, estimate_moduli
from soilstamp.methodology import ModulusOverrides, resolve_modulus_method


APPROVAL = {
    "status": "approved",
    "author": "engineer@example.test",
    "timestamp_utc": "2026-07-12T06:00:00+00:00",
    "reason": "Confirmed for the test calculation range.",
}


def _curve(
    pressures: list[float] | np.ndarray,
    settlements: list[float] | np.ndarray,
    *,
    branches: list[str] | None = None,
) -> pd.DataFrame:
    p = np.asarray(pressures, dtype=float)
    s = np.asarray(settlements, dtype=float)
    return pd.DataFrame(
        {
            "test_id": "T1",
            "sequence_no": np.arange(len(p)),
            "p_kPa": p,
            "settlement_mm": s,
            "D_mm": 300.0,
            "branch": branches or ["loading"] * len(p),
            "status": "stable",
        }
    )


def _linear_curve() -> pd.DataFrame:
    p = np.arange(0.0, 251.0, 50.0)
    return _curve(p, 0.1 + 0.01 * p)


def _antonov_metadata(
    *,
    p_range: tuple[float, float] | None = (0.0, 200.0),
    p_range_source: str = "explicit",
) -> dict[str, object]:
    method: dict[str, object] = {
        "profile_id": "antonov_round_stamp_v1",
    }
    if p_range is not None:
        method.update(
            {
                "p_range_kpa": p_range,
                "p_range_source": p_range_source,
                "approval": APPROVAL,
            }
        )
    return {"stamp_shape": "circle", "modulus_method": method}


def _regression(table: pd.DataFrame) -> pd.Series:
    return table.loc[table["method"].eq("E_regression")].iloc[0]


def _custom_overrides(shape_factor: float) -> ModulusOverrides:
    return ModulusOverrides(
        profile_id="custom_v1",
        p_range_kpa=(0.0, 200.0),
        p_range_source="explicit",
        nu=0.30,
        shape_factor=shape_factor,
        approval_status="approved",
        author="engineer@example.test",
        timestamp_utc="2026-07-12T06:00:00+00:00",
        reason="Approved project-specific coefficient comparison.",
    )


def test_antonov_profile_uses_fixed_coefficients_and_approved_explicit_range() -> None:
    result = calculate_moduli_for_test(
        _linear_curve(),
        _antonov_metadata(),
        "T1",
        bootstrap=0,
    )
    regression = _regression(result)

    assert regression["profile_id"] == "antonov_round_stamp_v1"
    assert regression["profile_version"] == "1.0"
    assert regression["nu"] == 0.30
    assert regression["shape_factor"] == 0.80
    assert regression["p_range_source"] == "explicit"
    assert regression["review_status"] == "approved"
    assert bool(regression["is_primary"])


def test_custom_shape_factor_changes_modulus_by_exact_ratio() -> None:
    frame = _linear_curve()
    table_k08 = calculate_moduli_for_test(
        frame,
        {},
        "T1",
        overrides=_custom_overrides(0.80),
        bootstrap=0,
    )
    table_k10 = calculate_moduli_for_test(
        frame,
        {},
        "T1",
        overrides=_custom_overrides(1.00),
        bootstrap=0,
    )

    e_k08 = float(_regression(table_k08)["E_stamp_app_kPa"])
    e_k10 = float(_regression(table_k10)["E_stamp_app_kPa"])
    assert np.isclose(e_k10 / e_k08, 1.25, rtol=1e-12, atol=0.0)
    assert bool(_regression(table_k08)["is_primary"])
    assert bool(_regression(table_k10)["is_primary"])


def test_missing_range_uses_observed_curve_only_as_nonprimary_diagnostic() -> None:
    resolution = resolve_modulus_method(
        _antonov_metadata(p_range=None),
        "T1",
        available_p_range=(0.0, 250.0),
    )

    assert resolution.p_min_kpa == 0.0
    assert resolution.p_max_kpa == 250.0
    assert resolution.p_range_source == "diagnostic_full_curve"
    assert resolution.p_range_origin == "observed_data"
    assert resolution.review_status == "review_required"
    assert not resolution.is_primary


def test_nonlinear_whole_curve_is_never_silently_promoted_to_primary() -> None:
    p = np.arange(0.0, 251.0, 50.0)
    nonlinear = _curve(p, 0.1 + 0.002 * p + 0.00004 * p**2)
    result = calculate_moduli_for_test(
        nonlinear,
        _antonov_metadata(p_range=None),
        "T1",
        bootstrap=0,
    )
    regression = _regression(result)

    assert regression["p_min_kPa"] == 0.0
    assert regression["p_max_kPa"] == 250.0
    assert regression["n"] == len(p)
    assert regression["p_range_source"] == "diagnostic_full_curve"
    assert regression["review_status"] == "review_required"
    assert not bool(regression["is_primary"])


def test_accepted_pcr_requires_a_manual_audited_value_not_auto_result() -> None:
    metadata = _antonov_metadata(p_range_source="accepted_pcr")
    auto_only = SimpleNamespace(pcr_auto=200.0, pcr_manual=None)
    auto_resolution = resolve_modulus_method(
        metadata,
        "T1",
        pcr_result=auto_only,
        available_p_range=(0.0, 250.0),
    )
    assert auto_resolution.accepted_pcr_kpa is None
    assert auto_resolution.review_status == "review_required"
    assert not auto_resolution.is_primary

    manually_accepted = SimpleNamespace(
        pcr_auto=180.0,
        pcr_manual=200.0,
        manual_reason="Accepted after curve review.",
        manual_author="engineer@example.test",
        manual_confirmed_at_utc="2026-07-12T06:05:00+00:00",
    )
    manual_resolution = resolve_modulus_method(
        metadata,
        "T1",
        pcr_result=manually_accepted,
        available_p_range=(0.0, 250.0),
    )
    assert manual_resolution.accepted_pcr_kpa == 200.0
    assert manual_resolution.review_status == "approved"
    assert manual_resolution.is_primary


def test_auto_pcr_cannot_raise_precedence_of_older_persisted_acceptance() -> None:
    metadata = _antonov_metadata(p_range_source="accepted_pcr")
    metadata["modulus_method"]["accepted_pcr"] = {
        "value_kPa": 200.0,
        "accepted_by": "engineer@example.test",
        "accepted_at": "2026-07-12T06:00:00+00:00",
        "reason": "Accepted for the original global range.",
    }
    metadata["tests"] = {
        "T1": {"modulus_method": {"p_range_kpa": (50.0, 200.0)}}
    }

    auto_only = SimpleNamespace(pcr_auto=180.0, pcr_manual=None)
    resolution = resolve_modulus_method(
        metadata,
        "T1",
        pcr_result=auto_only,
        available_p_range=(0.0, 250.0),
    )

    assert resolution.accepted_pcr_kpa == 200.0
    assert resolution.review_status == "review_required"
    assert not resolution.is_primary


def test_per_test_metadata_and_decision_layers_have_explicit_precedence_sources() -> None:
    metadata = {
        "modulus_method": {
            "profile_id": "custom_v1",
            "p_range_kpa": (0.0, 200.0),
            "p_range_source": "explicit",
            "nu": 0.20,
            "shape_factor": 0.70,
            "approval": APPROVAL,
        },
        "tests": {
            "T1": {
                "modulus_method": {
                    "nu": 0.25,
                    "shape_factor": 0.80,
                }
            }
        },
    }
    per_test = resolve_modulus_method(metadata, "T1", available_p_range=(0.0, 250.0))
    assert per_test.nu == 0.25
    assert per_test.shape_factor == 0.80
    assert per_test.nu_source == "metadata.tests.T1.modulus_method.nu"
    assert per_test.shape_factor_source == "metadata.tests.T1.modulus_method.shape_factor"

    resolved = resolve_modulus_method(
        metadata,
        "T1",
        manual_confirmation=ModulusOverrides(nu=0.30, shape_factor=0.90),
        overrides=ModulusOverrides(nu=0.35, shape_factor=1.00),
        available_p_range=(0.0, 250.0),
    )
    assert resolved.nu == 0.35
    assert resolved.shape_factor == 1.00
    assert resolved.nu_source == "cli_override.modulus_method.nu"
    assert resolved.shape_factor_source == "cli_override.modulus_method.shape_factor"


def test_conflicting_antonov_coefficient_is_recorded_and_downgrades_result() -> None:
    resolution = resolve_modulus_method(
        _antonov_metadata(),
        "T1",
        overrides=ModulusOverrides(shape_factor=1.0),
        available_p_range=(0.0, 250.0),
    )

    assert resolution.shape_factor == 1.0
    assert resolution.shape_factor_source == "cli_override.modulus_method.shape_factor"
    assert resolution.review_status == "review_required"
    assert not resolution.is_primary
    assert "shape_factor" in resolution.methodology_note


def test_newer_range_cannot_inherit_older_approval_record() -> None:
    resolution = resolve_modulus_method(
        _antonov_metadata(p_range=(0.0, 200.0)),
        "T1",
        overrides=ModulusOverrides(
            p_range_kpa=(0.0, 150.0),
            p_range_source="explicit",
        ),
        available_p_range=(0.0, 250.0),
    )

    assert resolution.p_max_kpa == 150.0
    assert resolution.p_range_origin == "cli_override.modulus_method.p_range_kPa"
    assert resolution.review_status == "review_required"
    assert not resolution.is_primary


def test_approval_timestamp_must_be_parseable_and_timezone_aware() -> None:
    metadata = _antonov_metadata()
    metadata["modulus_method"]["approval"] = {
        **APPROVAL,
        "timestamp_utc": "yesterday",
    }

    resolution = resolve_modulus_method(
        metadata, "T1", available_p_range=(0.0, 250.0)
    )

    assert resolution.review_status == "review_required"
    assert not resolution.is_primary


def test_antonov_profile_requires_confirmed_round_stamp_shape() -> None:
    metadata = _antonov_metadata()
    metadata.pop("stamp_shape")

    resolution = resolve_modulus_method(
        metadata, "T1", available_p_range=(0.0, 250.0)
    )

    assert resolution.review_status == "review_required"
    assert not resolution.is_primary
    assert "форма штампа не подтверждена" in resolution.methodology_note


def test_approved_requested_range_must_be_covered_by_observations() -> None:
    result = calculate_moduli_for_test(
        _linear_curve(),
        _antonov_metadata(p_range=(0.0, 1000.0)),
        "T1",
        bootstrap=0,
    )
    regression = _regression(result)

    assert regression["p_max_kPa"] == 250.0
    assert regression["requested_p_max_kPa"] == 1000.0
    assert regression["review_status"] == "review_required"
    assert not bool(regression["is_primary"])
    assert "выходит за наблюдённые" in regression["methodology_note"]


@pytest.mark.parametrize(
    "settlements",
    [
        [2.0, 1.5, 1.0, 0.5, 0.0, -0.5],
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    ],
)
def test_invalid_regression_cannot_be_primary(settlements: list[float]) -> None:
    frame = _curve(np.arange(0.0, 251.0, 50.0), settlements)
    result = calculate_moduli_for_test(
        frame, _antonov_metadata(p_range=(0.0, 250.0)), "T1", bootstrap=0
    )
    regression = _regression(result)

    assert not bool(regression["is_primary"])
    assert regression["review_status"] == "review_required"
    assert "конечного положительного" in regression["methodology_note"]


@pytest.mark.parametrize("diameter", [0.0, -300.0, float("inf")])
def test_invalid_diameter_is_rejected(diameter: float) -> None:
    frame = _linear_curve()
    frame["D_mm"] = diameter

    with pytest.raises(ValueError, match="конечным и положительным"):
        calculate_moduli_for_test(
            frame, _antonov_metadata(), "T1", bootstrap=0
        )


def test_used_indices_keep_last_stable_hold_at_duplicate_pressure() -> None:
    frame = _curve(
        [0.0, 50.0, 50.0, 100.0, 150.0],
        [0.0, 0.4, 0.5, 1.0, 1.5],
        branches=["loading", "loading", "hold", "loading", "loading"],
    )
    metadata = _antonov_metadata(p_range=(0.0, 150.0))
    regression = _regression(
        calculate_moduli_for_test(frame, metadata, "T1", bootstrap=0)
    )

    assert regression["used_indices"] == [0, 2, 3, 4]
    assert regression["n"] == 4


def test_direct_per_test_api_filters_a_mixed_frame_before_resolution() -> None:
    t1 = _linear_curve()
    t2 = _linear_curve().copy()
    t2["test_id"] = "T2"
    t2["settlement_mm"] = 0.1 + 0.10 * t2["p_kPa"]
    t2.index = np.arange(100, 100 + len(t2))
    mixed = pd.concat([t1, t2])

    from_mixed = calculate_moduli_for_test(
        mixed, _antonov_metadata(), "T1", bootstrap=0
    )
    from_t1 = calculate_moduli_for_test(
        t1, _antonov_metadata(), "T1", bootstrap=0
    )

    mixed_row = _regression(from_mixed)
    t1_row = _regression(from_t1)
    assert mixed_row["E_stamp_app_kPa"] == t1_row["E_stamp_app_kPa"]
    assert mixed_row["used_indices"] == t1_row["used_indices"]


def test_legacy_estimate_api_is_explicitly_diagnostic_and_unapproved() -> None:
    result = estimate_moduli(
        _linear_curve(),
        p_min_kpa=0.0,
        p_max_kpa=200.0,
        nu=0.30,
        shape_factor=1.00,
        bootstrap=0,
    )
    regression = _regression(result)

    assert regression["profile_id"] == "diagnostic_unapproved_v1"
    assert regression["profile_source"] == "legacy_api"
    assert regression["nu_source"] == "legacy_argument"
    assert regression["shape_factor_source"] == "legacy_argument"
    assert regression["review_status"] == "review_required"
    assert not bool(regression["is_primary"])
    assert "diagnostic/unapproved" in regression["methodology_note"]
