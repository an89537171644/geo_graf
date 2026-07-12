from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from soilstamp.analysis import (
    aggregate_group_curve,
    compare_groups,
    confirm_manual_pcr,
    center_and_tilt,
    deformation_work,
    derivative_diagnostics,
    estimate_moduli,
    fit_segmented_pcr,
    group_mean_curve,
    hysteresis_metrics,
    resolve_pairing_design,
)
from soilstamp.data import AuditTrail, prepare_measurements as _prepare_measurements


def prepare_measurements(*args, **kwargs):
    """Legacy-scientific fixtures opt into documented compatibility defaults."""

    kwargs.setdefault("strict_metadata", False)
    return _prepare_measurements(*args, **kwargs)


def _prepared_curve(test_id: str, p: np.ndarray, s: np.ndarray, group: str = "g") -> pd.DataFrame:
    area = np.pi * 0.3**2 / 4.0
    raw = pd.DataFrame(
        {
            "test_id": test_id,
            "stage": np.arange(len(p)),
            "load": p * area,
            "settlement": s,
            "branch": "loading",
            "group": group,
            "status": "stable",
        }
    )
    prepared, issues = prepare_measurements(
        raw,
        {"stamp_area_m2": area, "stamp_diameter_mm": 300.0, "lever_ratio": 1.0},
    )
    assert not [issue for issue in issues if issue.level == "error"]
    return prepared


def test_segmented_regression_recovers_known_breakpoint() -> None:
    p = np.arange(0.0, 401.0, 50.0)
    s = 0.2 + 0.003 * p + 0.010 * np.maximum(0.0, p - 150.0)
    frame = _prepared_curve("T1", p, s)
    result = fit_segmented_pcr(frame, bootstrap=80, seed=17)
    assert abs(result.pcr_auto - 150.0) < 0.5
    assert abs(result.slope_before - 0.003) < 1e-6
    assert abs(result.slope_after - 0.013) < 1e-6
    assert result.r2 > 0.999999
    assert result.n == len(p)
    assert result.alternative["method"] == "independent_two_line_bic"


def test_apparent_modulus_regression_and_secant() -> None:
    p = np.arange(0.0, 251.0, 50.0)
    s = 0.1 + 0.01 * p
    frame = _prepared_curve("T1", p, s)
    results = estimate_moduli(frame, nu=0.30, shape_factor=1.0, bootstrap=50, seed=2)
    expected_kpa = (1 - 0.3**2) * 0.3 / (0.01 / 1000.0)
    regression = results[results["method"] == "E_regression"].iloc[0]
    secant = results[results["method"] == "E_secant"].iloc[0]
    assert np.isclose(regression["E_stamp_app_kPa"], expected_kpa, rtol=1e-10)
    assert np.isclose(secant["E_stamp_app_kPa"], expected_kpa, rtol=1e-10)
    assert np.isclose(regression["r2"], 1.0)
    assert pd.isna(secant["r2"])
    assert results[results["method"].str.startswith("E_incremental")]["note"].str.contains("диагностический").all()


def test_linear_curve_does_not_produce_arbitrary_pcr() -> None:
    p = np.arange(0.0, 401.0, 50.0)
    frame = _prepared_curve("T1", p, 0.2 + 0.01 * p)
    with pytest.raises(ValueError, match="не идентифицируется"):
        fit_segmented_pcr(frame, bootstrap=20, seed=1)


def test_decreasing_compliance_break_is_not_called_pcr() -> None:
    p = np.arange(0.0, 401.0, 50.0)
    s = 0.2 + 0.013 * p - 0.010 * np.maximum(0.0, p - 150.0)
    frame = _prepared_curve("T1", p, s)
    with pytest.raises(ValueError, match="не возрастает"):
        fit_segmented_pcr(frame, bootstrap=20, seed=1)


def test_two_point_regression_does_not_claim_degenerate_ci() -> None:
    frame = _prepared_curve("T1", np.array([0.0, 100.0]), np.array([0.1, 1.1]))
    result = estimate_moduli(frame, bootstrap=50)
    regression = result[result["method"] == "E_regression"].iloc[0]
    assert pd.isna(regression["ci_low_kPa"])
    assert pd.isna(regression["ci_high_kPa"])


def test_group_mean_uses_exact_t_interval_and_real_union_levels() -> None:
    frames = []
    for index, offset in enumerate([0.0, 1.0, 2.0], 1):
        frames.append(_prepared_curve(f"T{index}", np.array([0.0, 100.0]), np.array([1.0 + offset, 2.0 + offset])))
    frame = pd.concat(frames, ignore_index=True)
    mean = group_mean_curve(frame, bootstrap=100, seed=4)
    assert mean["p_kPa"].tolist() == [0.0, 100.0]
    row = mean.iloc[0]
    exact_half_width = stats.t.ppf(0.975, 2) * 1.0 / np.sqrt(3)
    assert np.isclose(row["mean_settlement_mm"], 2.0)
    assert np.isclose(row["t_ci_low_mm"], 2.0 - exact_half_width)
    assert np.isclose(row["t_ci_high_mm"], 2.0 + exact_half_width)
    assert row["measured_n"] == 3
    assert row["interpolated_n"] == 0


def test_group_interpolation_is_inside_range_and_flagged() -> None:
    first = _prepared_curve("T1", np.array([0.0, 100.0, 200.0]), np.array([0.0, 1.0, 2.0]))
    second = _prepared_curve("T2", np.array([0.0, 200.0]), np.array([0.0, 2.0]))
    mean = group_mean_curve(pd.concat([first, second], ignore_index=True), bootstrap=50, seed=1)
    middle = mean[np.isclose(mean["p_kPa"], 100.0)].iloc[0]
    assert middle["n"] == 2
    assert middle["measured_n"] == 1
    assert middle["interpolated_n"] == 1
    assert not bool(middle["all_measured"])
    assert mean["p_kPa"].min() >= 0.0 and mean["p_kPa"].max() <= 200.0


def test_group_mean_uses_only_common_pressure_support() -> None:
    long_curve = _prepared_curve(
        "T1", np.array([0.0, 100.0, 200.0]), np.array([0.0, 1.0, 2.0])
    )
    short_curve = _prepared_curve("T2", np.array([100.0]), np.array([1.2]))
    mean = group_mean_curve(pd.concat([long_curve, short_curve], ignore_index=True), bootstrap=30)
    assert mean["p_kPa"].tolist() == [100.0]
    assert mean["n"].tolist() == [2]
    assert mean["measured_n"].tolist() == [2]


def test_coordinate_aggregation_is_test_id_order_invariant_including_bootstrap() -> None:
    frame = pd.concat(
        [
            _prepared_curve("T2", np.array([0.0, 100.0, 200.0]), np.array([0.0, 3.0, 4.0])),
            _prepared_curve("T1", np.array([0.0, 100.0, 200.0]), np.array([0.0, 1.0, 4.0])),
            _prepared_curve("T3", np.array([0.0, 100.0, 200.0]), np.array([0.0, 6.0, 10.0])),
        ],
        ignore_index=True,
    )
    reordered = pd.concat(
        [
            frame[frame["test_id"].eq("T3")],
            frame[frame["test_id"].eq("T2")],
            frame[frame["test_id"].eq("T1")],
        ],
        ignore_index=True,
    )

    first = aggregate_group_curve(frame, bootstrap=80, seed=7)
    second = aggregate_group_curve(reordered, bootstrap=80, seed=7)

    pd.testing.assert_frame_equal(first, second)
    assert first["source_test_ids"].unique().tolist() == ["T1,T2,T3"]
    assert first["bootstrap_ci_low"].notna().all()


def test_f_s_aggregation_accepts_repeats_with_identical_geometry() -> None:
    first = _prepared_curve(
        "T1", np.array([0.0, 100.0, 200.0]), np.array([0.0, 1.0, 2.0])
    )
    second = _prepared_curve(
        "T2", np.array([100.0, 200.0, 300.0]), np.array([1.4, 2.4, 3.4])
    )

    result = aggregate_group_curve(
        pd.concat([first, second], ignore_index=True),
        axis_mode="F-s",
        statistic="mean",
        bootstrap=30,
    )

    area = np.pi * 0.3**2 / 4.0
    assert result["F_kN"].tolist() == pytest.approx([100.0 * area, 200.0 * area])
    assert result["n"].tolist() == [2, 2]
    assert result["y"].tolist() == pytest.approx([1.2, 2.2])


def test_f_s_aggregation_rejects_different_stamp_diameters() -> None:
    first = _prepared_curve("T1", np.array([0.0, 100.0]), np.array([0.0, 1.0]))
    second = _prepared_curve("T2", np.array([0.0, 100.0]), np.array([0.0, 1.5]))
    second = second.assign(D_mm=500.0)

    with pytest.raises(ValueError, match="одинаковых диаметра и площади"):
        aggregate_group_curve(
            pd.concat([first, second], ignore_index=True),
            axis_mode="F-s",
            bootstrap=20,
        )


def test_p_s_over_d_normalizes_each_curve_before_aggregation_inside_support() -> None:
    first = _prepared_curve(
        "T1", np.array([0.0, 100.0, 200.0]), np.array([0.0, 10.0, 20.0])
    ).assign(D_mm=100.0)
    second = _prepared_curve(
        "T2", np.array([100.0, 200.0]), np.array([30.0, 50.0])
    ).assign(D_mm=200.0)

    result = aggregate_group_curve(
        pd.concat([first, second], ignore_index=True),
        axis_mode="p-s/D",
        statistic="mean",
        bootstrap=30,
    )

    assert result["p_kPa"].tolist() == [0.0, 100.0, 200.0]
    at_zero = result.loc[np.isclose(result["p_kPa"], 0.0)].iloc[0]
    at_100 = result.loc[np.isclose(result["p_kPa"], 100.0)].iloc[0]
    assert at_zero["n"] == 1
    assert at_zero["y"] == pytest.approx(0.0)
    assert at_100["n"] == 2
    assert at_100["y"] == pytest.approx(np.mean([10.0 / 100.0, 30.0 / 200.0]))
    assert result["y_quantity"].unique().tolist() == ["settlement_over_d"]
    assert result["mean_settlement_mm"].isna().all()
    assert result["t_ci_low_mm"].isna().all()
    assert result["bootstrap_ci_low_mm"].isna().all()


def test_coordinate_aggregation_marks_only_fully_measured_levels() -> None:
    detailed = _prepared_curve(
        "T1", np.array([0.0, 100.0, 200.0]), np.array([0.0, 1.0, 2.0])
    )
    sparse = _prepared_curve("T2", np.array([0.0, 200.0]), np.array([0.0, 2.0]))

    result = aggregate_group_curve(
        pd.concat([detailed, sparse], ignore_index=True),
        axis_mode="p-s",
        bootstrap=30,
    )

    middle = result.loc[np.isclose(result["p_kPa"], 100.0)].iloc[0]
    assert middle["n"] == 2
    assert middle["measured_n"] == 1
    assert middle["interpolated_n"] == 1
    assert not bool(middle["draw_marker"])
    assert result.loc[~np.isclose(result["p_kPa"], 100.0), "draw_marker"].all()


def test_coordinate_aggregation_supports_median() -> None:
    frame = pd.concat(
        [
            _prepared_curve("T1", np.array([0.0, 100.0]), np.array([0.0, 1.0])),
            _prepared_curve("T2", np.array([0.0, 100.0]), np.array([10.0, 11.0])),
            _prepared_curve("T3", np.array([0.0, 100.0]), np.array([100.0, 101.0])),
        ],
        ignore_index=True,
    )

    result = aggregate_group_curve(frame, statistic="median", bootstrap=50, seed=4)

    assert result["statistic"].unique().tolist() == ["median"]
    assert result["y"].tolist() == pytest.approx([10.0, 11.0])
    assert result["median_settlement"].tolist() == pytest.approx([10.0, 11.0])
    assert result["mean_settlement_mm"].isna().all()
    assert result["t_ci_low"].isna().all()


def test_group_comparison_preserves_pairs_for_bootstrap_and_permutation() -> None:
    frames = [
        _prepared_curve("B1", np.array([0.0, 100.0]), np.array([0.2, 2.0]), "baseline").assign(pair_id="P1"),
        _prepared_curve("R1", np.array([0.0, 100.0]), np.array([0.1, 1.2]), "reinforced").assign(pair_id="P1"),
        _prepared_curve("B2", np.array([0.0, 100.0]), np.array([0.3, 2.4]), "baseline").assign(pair_id="P2"),
        _prepared_curve("R2", np.array([0.0, 100.0]), np.array([0.2, 1.4]), "reinforced").assign(pair_id="P2"),
    ]
    result = compare_groups(
        pd.concat(frames, ignore_index=True), "baseline", "reinforced", bootstrap=100, seed=9
    )
    assert set(result["analysis_design"]) == {"paired"}
    assert set(result["pairing_status"]) == {"paired_validated"}
    assert set(result["pairing_reason"]) == {"complete_pairing"}
    assert set(result["pairing_warning"]) == {""}
    assert set(result["pair_ids_used"]) == {"P1,P2"}
    assert set(result["n_pairs"]) == {2}
    assert result["permutation_p"].between(0, 1).all()
    assert result["permutation_p_fdr_bh"].between(0, 1).all()
    assert (result["delta_s_mm"] > 0).all()


def test_paired_comparison_uses_the_same_pairs_at_each_pressure_level() -> None:
    frames = [
        _prepared_curve(
            "B1", np.array([0.0, 100.0]), np.array([0.0, 2.0]), "baseline"
        ).assign(pair_id="P1"),
        _prepared_curve(
            "R1", np.array([0.0, 100.0]), np.array([0.0, 1.0]), "reinforced"
        ).assign(pair_id="P1"),
        _prepared_curve(
            "B2", np.array([0.0, 50.0]), np.array([0.0, 0.8]), "baseline"
        ).assign(pair_id="P2"),
        _prepared_curve(
            "R2", np.array([0.0, 100.0]), np.array([0.0, 1.2]), "reinforced"
        ).assign(pair_id="P2"),
    ]

    result = compare_groups(
        pd.concat(frames, ignore_index=True),
        "baseline",
        "reinforced",
        bootstrap=30,
    )
    at_100 = result.loc[np.isclose(result["p_kPa"], 100.0)].iloc[0]

    assert at_100["analysis_design"] == "paired"
    assert at_100["n_baseline"] == at_100["n_reinforced"] == at_100["n_pairs"] == 1
    assert at_100["s_baseline_mm"] == pytest.approx(2.0)
    assert at_100["s_reinforced_mm"] == pytest.approx(1.0)
    assert at_100["delta_s_mm"] == pytest.approx(1.0)


def test_partial_pairing_does_not_drop_unpaired_tests() -> None:
    frames = [
        _prepared_curve("B1", np.array([0.0, 100.0]), np.array([0.2, 2.0]), "baseline").assign(pair_id="P1"),
        _prepared_curve("B2", np.array([0.0, 100.0]), np.array([0.3, 2.3]), "baseline").assign(pair_id=pd.NA),
        _prepared_curve("R1", np.array([0.0, 100.0]), np.array([0.1, 1.2]), "reinforced").assign(pair_id="P1"),
        _prepared_curve("R2", np.array([0.0, 100.0]), np.array([0.2, 1.4]), "reinforced").assign(pair_id=pd.NA),
    ]
    result = compare_groups(pd.concat(frames, ignore_index=True), "baseline", "reinforced", bootstrap=30)
    assert set(result["analysis_design"]) == {"independent"}
    assert set(result["pairing_status"]) == {"independent_fallback"}
    assert result["pairing_reason"].str.contains("missing_pair_id").all()
    assert result["pairing_warning"].str.contains("independent analysis").all()
    assert set(result["n_baseline"]) == {2}
    assert set(result["n_reinforced"]) == {2}


def test_baseline_group_name_never_implies_pairing_without_pair_id() -> None:
    frames = [
        _prepared_curve("B1", np.array([0.0, 100.0]), np.array([0.2, 2.0]), "baseline"),
        _prepared_curve("B2", np.array([0.0, 100.0]), np.array([0.3, 2.3]), "baseline"),
        _prepared_curve("R1", np.array([0.0, 100.0]), np.array([0.1, 1.2]), "reinforced"),
        _prepared_curve("R2", np.array([0.0, 100.0]), np.array([0.2, 1.4]), "reinforced"),
    ]
    frame = pd.concat(frames, ignore_index=True).assign(baseline_group="baseline")

    result = compare_groups(frame, "baseline", "reinforced", bootstrap=30)

    assert set(result["analysis_design"]) == {"independent"}
    assert result["pairing_reason"].str.contains("missing_pair_id").all()
    assert set(result["n_baseline"]) == {2}
    assert set(result["n_reinforced"]) == {2}


def test_whitespace_pair_id_is_missing_and_forces_independent_fallback() -> None:
    frames = [
        _prepared_curve("B1", np.array([0.0, 100.0]), np.array([0.2, 2.0]), "baseline").assign(pair_id=" P1 "),
        _prepared_curve("B2", np.array([0.0, 100.0]), np.array([0.3, 2.3]), "baseline").assign(pair_id="   "),
        _prepared_curve("R1", np.array([0.0, 100.0]), np.array([0.1, 1.2]), "reinforced").assign(pair_id="P1"),
        _prepared_curve("R2", np.array([0.0, 100.0]), np.array([0.2, 1.4]), "reinforced").assign(pair_id="P2"),
    ]

    result = compare_groups(pd.concat(frames, ignore_index=True), "baseline", "reinforced", bootstrap=30)

    assert set(result["analysis_design"]) == {"independent"}
    assert result["pairing_reason"].str.contains("noncanonical_pair_id:baseline:B1").all()
    assert result["pairing_reason"].str.contains("missing_pair_id:baseline:B2").all()
    assert result["pairing_reason"].str.contains("incomplete_pair_set").all()


def test_pair_id_edge_whitespace_is_not_silently_normalized_into_a_pair() -> None:
    frames = [
        _prepared_curve(
            "B1", np.array([0.0, 100.0]), np.array([0.2, 2.0]), "baseline"
        ).assign(pair_id=" P1 "),
        _prepared_curve(
            "R1", np.array([0.0, 100.0]), np.array([0.1, 1.2]), "reinforced"
        ).assign(pair_id="P1"),
    ]

    result = compare_groups(
        pd.concat(frames, ignore_index=True),
        "baseline",
        "reinforced",
        bootstrap=30,
    )

    assert set(result["analysis_design"]) == {"independent"}
    assert result["pairing_reason"].str.contains(
        "noncanonical_pair_id:baseline:B1"
    ).all()
    assert result["pairing_warning"].str.len().gt(0).all()


def test_duplicate_pair_id_within_group_forces_independent_fallback() -> None:
    frames = [
        _prepared_curve("B1", np.array([0.0, 100.0]), np.array([0.2, 2.0]), "baseline").assign(pair_id="P1"),
        _prepared_curve("B2", np.array([0.0, 100.0]), np.array([0.3, 2.3]), "baseline").assign(pair_id="P1"),
        _prepared_curve("R1", np.array([0.0, 100.0]), np.array([0.1, 1.2]), "reinforced").assign(pair_id="P1"),
        _prepared_curve("R2", np.array([0.0, 100.0]), np.array([0.2, 1.4]), "reinforced").assign(pair_id="P2"),
    ]

    result = compare_groups(pd.concat(frames, ignore_index=True), "baseline", "reinforced", bootstrap=30)

    assert set(result["analysis_design"]) == {"independent"}
    assert result["pairing_reason"].str.contains(
        "duplicate_pair_id_within_group:baseline:P1"
    ).all()
    assert set(result["n_baseline"]) == {2}
    assert set(result["n_reinforced"]) == {2}


def test_nonanalyzable_duplicate_pair_cannot_disappear_before_pairing_check() -> None:
    valid_baseline = _prepared_curve(
        "B1", np.array([0.0, 100.0]), np.array([0.2, 2.0]), "baseline"
    ).assign(pair_id="P1")
    invalid_baseline = _prepared_curve(
        "B2", np.array([0.0, 100.0]), np.array([0.3, 2.3]), "baseline"
    ).assign(pair_id="P1", settlement_mm=np.nan)
    reinforced = _prepared_curve(
        "R1", np.array([0.0, 100.0]), np.array([0.1, 1.2]), "reinforced"
    ).assign(pair_id="P1")

    result = compare_groups(
        pd.concat([valid_baseline, invalid_baseline, reinforced], ignore_index=True),
        "baseline",
        "reinforced",
        bootstrap=30,
    )

    assert set(result["analysis_design"]) == {"independent"}
    assert result["pairing_reason"].str.contains(
        "duplicate_pair_id_within_group:baseline:P1"
    ).all()
    assert result["pairing_reason"].str.contains(
        "missing_analyzable_curve:baseline:B2"
    ).all()
    assert set(result["n_baseline"]) == {1}
    assert set(result["n_reinforced"]) == {1}


def test_same_test_id_cannot_be_used_in_both_groups() -> None:
    baseline = _prepared_curve(
        "T-SAME", np.array([0.0, 100.0]), np.array([0.2, 2.0]), "baseline"
    ).assign(pair_id="P1")
    reinforced = _prepared_curve(
        "T-SAME", np.array([0.0, 100.0]), np.array([0.1, 1.2]), "reinforced"
    ).assign(pair_id="P1")

    resolution = resolve_pairing_design(baseline, reinforced)
    assert resolution.analysis_design == "independent"
    assert "overlapping_test_id:T-SAME" in resolution.pairing_reason

    with pytest.raises(ValueError, match="test_id.*обе"):
        compare_groups(
            pd.concat([baseline, reinforced], ignore_index=True),
            "baseline",
            "reinforced",
            bootstrap=30,
        )


def test_group_comparison_rejects_self_comparison() -> None:
    frame = _prepared_curve(
        "T1", np.array([0.0, 100.0]), np.array([0.2, 2.0]), "baseline"
    ).assign(pair_id="P1")

    with pytest.raises(ValueError, match="две разные группы"):
        compare_groups(frame, "baseline", "baseline", bootstrap=30)


def test_incomplete_pair_set_forces_independent_fallback() -> None:
    frames = [
        _prepared_curve("B1", np.array([0.0, 100.0]), np.array([0.2, 2.0]), "baseline").assign(pair_id="P1"),
        _prepared_curve("B2", np.array([0.0, 100.0]), np.array([0.3, 2.3]), "baseline").assign(pair_id="P2"),
        _prepared_curve("R1", np.array([0.0, 100.0]), np.array([0.1, 1.2]), "reinforced").assign(pair_id="P1"),
        _prepared_curve("R2", np.array([0.0, 100.0]), np.array([0.2, 1.4]), "reinforced").assign(pair_id="P3"),
    ]

    result = compare_groups(pd.concat(frames, ignore_index=True), "baseline", "reinforced", bootstrap=30)

    assert set(result["analysis_design"]) == {"independent"}
    assert result["pairing_reason"].str.contains("incomplete_pair_set").all()
    assert result["pairing_warning"].str.len().gt(0).all()


def test_derivative_does_not_bridge_unloading_to_reloading() -> None:
    raw = pd.DataFrame(
        {
            "test_id": ["T1"] * 6,
            "stage": range(6),
            "load": [0, 10, 20, 10, 0, 10],
            "settlement": [0.0, 1.0, 2.0, 1.7, 1.2, 1.6],
            "branch": ["loading", "loading", "loading", "unloading", "unloading", "reloading"],
            "status": "stable",
        }
    )
    frame, _ = prepare_measurements(raw, {"stamp_shape": "custom", "stamp_area_m2": 0.1})
    result = derivative_diagnostics(frame)
    assert not ((result["from_sequence"] == 2) & (result["to_sequence"] == 5)).any()
    assert set(zip(result["from_sequence"], result["to_sequence"])) == {(0, 1), (1, 2)}


def test_deformation_work_does_not_integrate_across_nan_gap() -> None:
    raw = pd.DataFrame(
        {
            "test_id": ["T1"] * 4,
            "stage": range(4),
            "load": [0, 10, 20, 30],
            "settlement": [0.0, 1.0, np.nan, 3.0],
            "branch": "loading",
        }
    )
    frame, _ = prepare_measurements(raw, {"stamp_shape": "custom", "stamp_area_m2": 0.1})
    result = deformation_work(frame).iloc[0]
    assert result["integrated_segments"] == 1
    assert result["skipped_gaps"] == 2


def test_pcr_excludes_pending_status_and_uses_last_stable_hold() -> None:
    area = np.pi * 0.3**2 / 4.0
    p = np.array([0.0, 50.0, 50.0, 100.0, 150.0, 200.0, 250.0, 300.0])
    true_s = 0.2 + 0.003 * p + 0.01 * np.maximum(0.0, p - 150.0)
    true_s[1] = 0.30  # transitional reading before stabilization
    raw = pd.DataFrame(
        {
            "test_id": "T1",
            "stage": range(len(p)),
            "load": p * area,
            "settlement": true_s,
            "branch": ["loading", "loading", "hold", "loading", "loading", "loading", "loading", "loading"],
            "status": ["stable", "stable", "stable", "stable", "stable", "pending", "stable", "stable"],
        }
    )
    frame, issues = prepare_measurements(
        raw,
        {"stamp_diameter_mm": 300.0, "stamp_area_m2": area},
    )
    assert any(issue.code == "unaccepted_status" for issue in issues)
    result = fit_segmented_pcr(frame, min_side=2, bootstrap=30)
    assert 2 in result.used_indices  # hold at p=50
    assert 1 not in result.used_indices
    assert 5 not in result.used_indices  # pending p=200


def test_group_comparison_tail_reports_actual_n_and_no_inference_at_n1() -> None:
    frames = [
        _prepared_curve("B1", np.array([0.0, 100.0, 200.0]), np.array([0.2, 2.0, 4.0]), "baseline"),
        _prepared_curve("B2", np.array([0.0, 100.0]), np.array([0.3, 2.2]), "baseline"),
        _prepared_curve("R1", np.array([0.0, 100.0, 200.0]), np.array([0.1, 1.2, 2.5]), "reinforced"),
        _prepared_curve("R2", np.array([0.0, 100.0]), np.array([0.2, 1.4]), "reinforced"),
    ]
    result = compare_groups(pd.concat(frames, ignore_index=True), "baseline", "reinforced", bootstrap=30)
    tail = result[np.isclose(result["p_kPa"], 200.0)].iloc[0]
    assert tail["n_baseline"] == 1
    assert tail["n_reinforced"] == 1
    assert pd.isna(tail["delta_s_ci_low_mm"])
    assert pd.isna(tail["permutation_p"])
    assert pd.isna(tail["effect_size"])


def test_hysteresis_energy_uses_only_first_complete_cycle_and_includes_hold() -> None:
    raw = pd.DataFrame(
        {
            "test_id": ["T1"] * 5,
            "stage": range(5),
            "load": [0.0, 100.0, 0.0, 100.0, 0.0],
            "settlement": [0.0, 2.0, 1.0, 3.0, 1.5],
            "branch": ["loading", "loading", "unloading", "reloading", "unloading"],
            "status": "stable",
        }
    )
    frame, _ = prepare_measurements(raw, {"stamp_shape": "custom", "stamp_area_m2": 1.0})
    metric = hysteresis_metrics(frame).iloc[0]
    assert np.isclose(metric["hysteresis_energy_kJ_m2"], 0.05)

    hold_raw = pd.DataFrame(
        {
            "test_id": ["T2"] * 4,
            "stage": range(4),
            "load": [0.0, 100.0, 100.0, 0.0],
            "settlement": [0.0, 1.0, 2.0, 1.0],
            "branch": ["loading", "loading", "hold", "unloading"],
            "status": "stable",
        }
    )
    hold_frame, _ = prepare_measurements(
        hold_raw, {"stamp_shape": "custom", "stamp_area_m2": 1.0}
    )
    hold_metric = hysteresis_metrics(hold_frame).iloc[0]
    assert np.isclose(hold_metric["hysteresis_energy_kJ_m2"], 0.10)


def test_manual_pcr_must_stay_inside_tested_pressure_range() -> None:
    p = np.arange(0.0, 401.0, 50.0)
    s = 0.2 + 0.003 * p + 0.010 * np.maximum(0.0, p - 150.0)
    result = fit_segmented_pcr(_prepared_curve("T1", p, s), bootstrap=20)
    with pytest.raises(ValueError, match="испытанном диапазоне"):
        confirm_manual_pcr(
            result,
            -1.0,
            reason="bad",
            audit=AuditTrail(),
            scope="T1",
        )
    with pytest.raises(ValueError, match="испытанном диапазоне"):
        confirm_manual_pcr(
            result,
            500.0,
            reason="bad",
            audit=AuditTrail(),
            scope="T1",
        )


def test_tilt_direction_is_undefined_below_indicator_resolution() -> None:
    raw = pd.DataFrame(
        {
            "test_id": ["T1"],
            "stage": [1],
            "load": [1.0],
            "indicator_1": [1.0],
            "indicator_2": [1.0],
            "indicator_3": [1.0],
            "reference_indicator": [0.0],
        }
    )
    metadata = {
        "stamp_shape": "custom",
        "stamp_area_m2": 0.1,
        "indicator_resolution_mm": 0.01,
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
            name: {
                "type": "ИЧ-10",
                "serial_number": name,
                "instrument_id": name,
                "range_mm": 10.0,
                "division_mm": 0.01,
                "correction_factor": 1.0,
                "verification_date": "2026-01-01",
                "verification_valid_until": "2030-01-01",
                "mode": "cumulative_settlement",
                "initial_reading": None,
                "initial_turn": 0,
                "zero_correction_mm": 0.0,
                "max_increment_mm": None,
                "reverse_tolerance_mm": 0.02,
                "travel_range_mm": 50.0,
                "cumulative_sign": -1.0
                if name == "reference_indicator"
                else 1.0,
                **(
                    {
                        "x_mm": {
                            "indicator_1": 0.0,
                            "indicator_2": 100.0,
                            "indicator_3": 0.0,
                        }[name],
                        "y_mm": {
                            "indicator_1": 0.0,
                            "indicator_2": 0.0,
                            "indicator_3": 100.0,
                        }[name],
                    }
                    if name != "reference_indicator"
                    else {}
                ),
            }
            for name in (
                "indicator_1",
                "indicator_2",
                "indicator_3",
                "reference_indicator",
            )
        },
    }
    frame, _ = prepare_measurements(raw, metadata)
    tilt = center_and_tilt(
        frame,
        {
            "indicator_1": [0.0, 0.0],
            "indicator_2": [100.0, 0.0],
            "indicator_3": [0.0, 100.0],
        },
    ).iloc[0]
    assert not bool(tilt["tilt_direction_resolved"])
    assert pd.isna(tilt["tilt_direction_deg"])


def test_center_and_tilt_uses_fixed_channels_and_explicit_missing_policy() -> None:
    frame = pd.DataFrame(
        {
            "test_id": ["T1"],
            "stage": [1],
            "indicator_1_settlement_mm": [0.0],
            "indicator_2_settlement_mm": [1.0],
            "indicator_3_settlement_mm": [2.0],
            "indicator_4_settlement_mm": [np.nan],
            "indicator_resolution_mm": [0.01],
            "indicator_calibration_confirmed": [True],
        }
    )
    positions = {
        "indicator_1": (-100.0, 0.0),
        "indicator_2": (0.0, 100.0),
        "indicator_3": (100.0, 0.0),
        "indicator_4": (0.0, -100.0),
    }

    blocked = center_and_tilt(frame, positions, channels=list(positions)).iloc[0]
    allowed = center_and_tilt(
        frame,
        positions,
        channels=list(positions),
        missing_channel_policy="allow_if_solvable",
    ).iloc[0]

    assert blocked["aggregation_status"] == "blocked_missing_channels"
    assert pd.isna(blocked["center_settlement_mm"])
    assert allowed["aggregation_status"] == "ok"
    assert allowed["plane_rank"] == 3
    assert allowed["missing_channels"] == '["indicator_4"]'

    unconfirmed = frame.copy()
    unconfirmed["indicator_calibration_confirmed"] = False
    assert center_and_tilt(
        unconfirmed, positions, channels=list(positions)
    ).empty


def test_center_and_tilt_does_not_infer_basis_from_explicit_empty_channels() -> None:
    frame = pd.DataFrame(
        {
            "test_id": ["T1"],
            "stage": [1],
            "indicator_1_settlement_mm": [0.0],
            "indicator_2_settlement_mm": [1.0],
            "indicator_3_settlement_mm": [2.0],
        }
    )
    positions = {
        "indicator_1": (0.0, 0.0),
        "indicator_2": (100.0, 0.0),
        "indicator_3": (0.0, 100.0),
    }

    assert center_and_tilt(frame, positions, channels=[]).empty
