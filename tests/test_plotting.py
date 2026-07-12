from __future__ import annotations

from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from PIL import Image

from soilstamp.data import failure_summary, prepare_measurements as _prepare_measurements
from soilstamp.plotting import (
    CurveSelectionDecision,
    export_figure,
    plot_curves,
    plot_failure_intervals,
    resolve_curve_selections,
)


def prepare_measurements(*args, **kwargs):
    kwargs.setdefault("strict_metadata", False)
    return _prepare_measurements(*args, **kwargs)


def _failure_frame() -> pd.DataFrame:
    raw = pd.DataFrame(
        {
            "test_id": ["T1", "T1", "T1"],
            "stage": [1, 2, 3],
            "load": [100.0, 200.0, 250.0],
            "settlement": [1.0, 2.0, np.nan],
            "status": ["stable", "stable", "failure"],
            "group": ["g", "g", "g"],
            "branch": ["loading", "loading", "loading"],
        }
    )
    frame, _ = prepare_measurements(
        raw,
        {
            "stamp_shape": "custom",
            "stamp_diameter_mm": 300.0,
            "stamp_area_m2": 1.0,
            "lever_ratio": 1.0,
        },
    )
    return frame


def _repeated_frame(
    *,
    test_order: tuple[str, ...] = ("T2", "T1"),
    three_curves: bool = False,
) -> pd.DataFrame:
    settlements = {
        "T1": [0.0, 1.0, 2.0],
        "T2": [0.0, 2.0, 4.0],
        "T3": [0.0, 9.0, 10.0],
    }
    if three_curves:
        test_order = (*test_order, "T3")
    rows: list[dict[str, object]] = []
    for test_id in test_order:
        levels = [0.0, 100.0, 200.0] if test_id != "T2" else [0.0, 200.0]
        values = settlements[test_id] if test_id != "T2" else [0.0, 4.0]
        for sequence_no, (load, settlement) in enumerate(zip(levels, values, strict=True), 1):
            rows.append(
                {
                    "test_id": test_id,
                    "stage": sequence_no,
                    "load": load,
                    "settlement": settlement,
                    "status": "stable",
                    "group": "g",
                    "branch": "loading",
                }
            )
    frame, _ = prepare_measurements(
        pd.DataFrame(rows),
        {
            "stamp_shape": "custom",
            "stamp_diameter_mm": 300.0,
            "stamp_area_m2": 1.0,
            "lever_ratio": 1.0,
        },
    )
    return frame


def test_antonov_orientation_frame_grid_and_polyline_vertices() -> None:
    frame = _failure_frame()
    output = plot_curves(frame, mode="raw_protocol", axis_mode="F-s")
    ax = output.figure.axes[0]
    assert ax.xaxis.get_label_position() == "top"
    assert not bool(ax.xaxis_inverted())
    assert bool(ax.yaxis_inverted())
    assert all(ax.spines[name].get_visible() for name in ["top", "right", "bottom", "left"])
    assert any(line.get_visible() for line in ax.get_xgridlines())
    raw_line = next(line for line in ax.lines if line.get_gid() == "raw-T1")
    assert np.allclose(raw_line.get_xdata(), [100.0, 200.0])
    assert np.allclose(raw_line.get_ydata(), [1.0, 2.0])
    assert raw_line.get_path().codes is None  # ordinary straight polyline, no spline path codes
    assert ax.get_xlim()[1] > 250.0  # failure load without settlement still defines the axis
    plt.close(output.figure)


def test_mixed_diameters_warn_on_force_axis() -> None:
    frame = pd.concat([_failure_frame(), _failure_frame().assign(test_id="T2", D_mm=500.0)], ignore_index=True)
    output = plot_curves(frame, mode="raw_protocol", axis_mode="F-s")
    assert any("разные диаметры" in warning for warning in output.warnings)
    plt.close(output.figure)


def test_p_over_pu_normalization_requires_confirmed_capacity() -> None:
    frame = _failure_frame().assign(pu_kPa_confirmed=250.0)
    output = plot_curves(frame, mode="raw_protocol", axis_mode="p/pu-s/D")
    raw_line = next(line for line in output.figure.axes[0].lines if line.get_gid() == "raw-T1")
    assert np.allclose(raw_line.get_xdata(), [0.4, 0.8])
    assert output.figure.axes[0].get_xlim()[1] > 1.0
    plt.close(output.figure)


def test_publication_exports_are_valid_and_png_is_600_dpi() -> None:
    output = plot_curves(_failure_frame(), mode="antonov_publication", axis_mode="F-s")
    svg = export_figure(output.figure, "svg")
    pdf = export_figure(output.figure, "pdf")
    png = export_figure(output.figure, "png")
    assert b"<svg" in svg[:1000]
    assert pdf.startswith(b"%PDF")
    assert png.startswith(b"\x89PNG\r\n\x1a\n")
    image = Image.open(BytesIO(png))
    dpi = image.info.get("dpi")
    assert dpi is not None and abs(dpi[0] - 600) < 1
    plt.close(output.figure)


def test_antonov_caption_decodes_numbered_curves() -> None:
    output = plot_curves(_failure_frame(), mode="antonov_publication", axis_mode="F-s")
    assert output.curve_map == {1: "g, индивидуальная T1"}
    assert "1 — g" in output.caption
    assert "без spline" in output.caption
    plt.close(output.figure)


def test_repeated_publication_series_requires_explicit_selection() -> None:
    frame = _repeated_frame()
    with pytest.raises(ValueError, match="явно выберите"):
        plot_curves(frame, mode="antonov_publication", axis_mode="p-s", bootstrap=20)


@pytest.mark.parametrize("method", ["mean_curve", "median_curve", "individual_curves"])
def test_explicit_publication_strategies_are_recorded(method: str) -> None:
    output = plot_curves(
        _repeated_frame(three_curves=True),
        mode="antonov_publication",
        axis_mode="p-s",
        selections=[CurveSelectionDecision(group="g", method=method)],  # type: ignore[arg-type]
        bootstrap=20,
        seed=4,
    )
    assert output.selection_records[0]["method"] == method
    assert set(output.plotted_points["selection_method"]) == {method}
    if method == "individual_curves":
        assert len(output.curve_map) == 3
    else:
        middle = output.plotted_points[np.isclose(output.plotted_points["x"], 100.0)].iloc[0]
        expected = 4.0 if method == "mean_curve" else 2.0
        assert middle["y"] == pytest.approx(expected)
    plt.close(output.figure)


def test_manual_representative_requires_membership_author_utc_timestamp_and_reason() -> None:
    frame = _repeated_frame()
    valid = {
        "group": "g",
        "method": "manual_representative",
        "test_id": "T1",
        "author": "Engineer",
        "timestamp_utc": "2026-07-12T05:30:00Z",
        "reason": "Confirmed representative laboratory run.",
    }
    for field, value, match in (
        ("test_id", "OTHER", "не входит"),
        ("author", "", "author"),
        ("timestamp_utc", "", "timestamp_utc"),
        ("timestamp_utc", "2026-07-12T08:30:00+03:00", "UTC"),
        ("reason", "", "reason"),
    ):
        invalid = {**valid, field: value}
        with pytest.raises(ValueError, match=match):
            resolve_curve_selections(frame, [invalid])

    output = plot_curves(
        frame,
        mode="antonov_publication",
        axis_mode="F-s",
        selections=[valid],
        bootstrap=20,
    )
    assert output.curve_map == {1: "g, репрезентативная T1"}
    assert output.selection_records == [
        {
            **valid,
            "timestamp_utc": "2026-07-12T05:30:00+00:00",
        }
    ]
    plt.close(output.figure)


def test_publication_result_and_numbering_are_invariant_to_test_block_order() -> None:
    first = plot_curves(
        _repeated_frame(test_order=("T2", "T1")),
        mode="antonov_publication",
        axis_mode="p-s",
        selections={"g": "mean_curve"},
        bootstrap=30,
        seed=17,
    )
    second = plot_curves(
        _repeated_frame(test_order=("T1", "T2")),
        mode="antonov_publication",
        axis_mode="p-s",
        selections={"g": "mean_curve"},
        bootstrap=30,
        seed=17,
    )
    assert first.curve_map == second.curve_map
    pd.testing.assert_frame_equal(
        first.plotted_points.reset_index(drop=True),
        second.plotted_points.reset_index(drop=True),
        check_like=True,
    )
    plt.close(first.figure)
    plt.close(second.figure)


def test_group_ci_stores_counts_and_marks_only_all_measured_levels() -> None:
    output = plot_curves(
        _repeated_frame(),
        mode="group_mean_ci",
        axis_mode="F-s",
        bootstrap=30,
        seed=8,
    )
    middle = output.plotted_points[np.isclose(output.plotted_points["x"], 100.0)].iloc[0]
    assert middle["n"] == 2
    assert middle["measured_n"] == 1
    assert middle["interpolated_n"] == 1
    assert not bool(middle["draw_marker"])
    marker_line = next(
        line for line in output.figure.axes[0].lines if line.get_gid() == "group-mean-markers-g"
    )
    assert list(marker_line.get_xdata()) == [0.0, 200.0]
    assert not any(line.get_linestyle() == ":" for line in output.figure.axes[0].lines)
    plt.close(output.figure)


def test_aggregate_publication_has_antonov_structure_and_no_pooled_failure_point() -> None:
    first = _failure_frame().assign(test_id="T1")
    second = _failure_frame().assign(test_id="T2")
    second.loc[second["is_failure"], ["F_kN", "p_kPa"]] = [300.0, 300.0]
    frame = pd.concat([second, first], ignore_index=True)
    output = plot_curves(
        frame,
        mode="antonov_publication",
        axis_mode="F-s",
        selections={"g": "mean_curve"},
        bootstrap=20,
    )
    ax = output.figure.axes[0]
    aggregate = next(line for line in ax.lines if line.get_gid() == "aggregate-g-mean_curve")
    assert ax.xaxis.get_label_position() == "top"
    assert ax.yaxis_inverted()
    assert aggregate.get_color() == "black"
    assert aggregate.get_path().codes is None
    assert not any(line.get_linestyle() == ":" for line in ax.lines)
    assert all("250" not in text.get_text() for text in ax.texts)
    plt.close(output.figure)


def test_failure_interval_plot_is_individual_deterministic_and_has_no_mean() -> None:
    failures = pd.DataFrame(
        {
            "test_id": ["F2", "C1", "F1"],
            "failure_reached": [True, False, True],
            "right_censored": [False, True, False],
            "Fu_lower": [120.0, 180.0, 100.0],
            "Fu_upper": [170.0, np.nan, 150.0],
            "pu_lower": [12.0, 18.0, 10.0],
            "pu_upper": [17.0, np.nan, 15.0],
            "s_failure": [np.nan, np.nan, np.nan],
        }
    )
    output = plot_failure_intervals(failures, capacity_axis="force")
    assert output.curve_map == {1: "C1", 2: "F1", 3: "F2"}
    assert output.plotted_points["censoring_type"].tolist() == [
        "right_censored",
        "interval_censored",
        "interval_censored",
    ]
    assert "наблюдалось разрушение — 2" in output.caption
    assert "правоцензурировано — 1" in output.caption
    assert "точечная оценка не рассчитывалась" in output.caption
    gids = {artist.get_gid() for artist in output.figure.axes[0].get_children()}
    assert "failure-open-lower-F1" in gids
    assert "failure-closed-upper-F1" in gids
    assert "failure-right-arrow-C1" in gids
    right_lower = next(
        line for line in output.figure.axes[0].lines if line.get_gid() == "failure-right-lower-C1"
    )
    assert right_lower.get_markerfacecolor() == "white"
    assert not any(
        np.allclose(line.get_xdata(), [150.0, 150.0])
        for line in output.figure.axes[0].lines
        if len(line.get_xdata()) == 2
    )
    plt.close(output.figure)


def test_failure_interval_plot_warns_for_indeterminate_bounds() -> None:
    failures = failure_summary(_failure_frame()).assign(
        Fu_lower=np.nan,
        Fu_upper=np.nan,
        p_last_stable=np.nan,
        p_failure_step=np.nan,
        pu_lower=np.nan,
        pu_upper=np.nan,
        lower_bound=np.nan,
        upper_bound=np.nan,
    )
    output = plot_failure_intervals(failures, capacity_axis="force")
    assert output.warnings and "недостаточно границ" in output.warnings[0]
    assert output.plotted_points.iloc[0]["censoring_type"] == "indeterminate"
    assert any(text.get_gid() == "failure-indeterminate-T1" for text in output.figure.axes[0].texts)
    plt.close(output.figure)


def test_diagnostic_rejects_unfiltered_multiple_tests() -> None:
    with pytest.raises(ValueError, match="ровно одно"):
        plot_curves(_repeated_frame(), mode="diagnostic", axis_mode="p-s", bootstrap=20)
