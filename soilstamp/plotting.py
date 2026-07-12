"""Matplotlib publication and diagnostic figures."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from io import BytesIO
from collections.abc import Iterable, Mapping
from typing import Any, Literal

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from matplotlib.ticker import AutoMinorLocator, MultipleLocator

from .analysis import aggregate_group_curve, fit_segmented_pcr
from .data import _stable_status_mask
from .schema import PCRResult, VERSION


LINE_STYLES = ["-", "--", "-.", ":"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "<", ">"]


@dataclass(slots=True)
class PlotOutput:
    figure: Figure
    caption: str
    warnings: list[str]
    curve_map: dict[int, str]
    selection_records: list[dict[str, Any]] = field(default_factory=list)
    plotted_points: pd.DataFrame = field(default_factory=pd.DataFrame)


CurveSelectionMethod = Literal[
    "mean_curve",
    "median_curve",
    "manual_representative",
    "individual_curves",
]
CURVE_SELECTION_METHODS = {
    "mean_curve",
    "median_curve",
    "manual_representative",
    "individual_curves",
}


@dataclass(frozen=True, slots=True)
class CurveSelectionDecision:
    """Explicit publication decision for one experimental series."""

    group: str
    method: CurveSelectionMethod
    test_id: str | None = None
    author: str | None = None
    timestamp_utc: str | None = None
    reason: str | None = None


CurveSelectionInput = (
    Mapping[str, CurveSelectionDecision | Mapping[str, Any] | str]
    | Iterable[CurveSelectionDecision | Mapping[str, Any]]
    | None
)


def _parse_utc_timestamp(value: str | None) -> str:
    rendered = str(value or "").strip()
    if not rendered:
        raise ValueError("Для manual_representative требуется timestamp_utc.")
    candidate = rendered[:-1] + "+00:00" if rendered.endswith("Z") else rendered
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError as exc:
        raise ValueError(
            "timestamp_utc для manual_representative должен быть корректным ISO 8601."
        ) from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise ValueError("timestamp_utc для manual_representative должен содержать часовой пояс UTC.")
    return parsed.isoformat()


def _coerce_curve_selection(
    value: CurveSelectionDecision | Mapping[str, Any] | str,
    *,
    group_hint: str | None = None,
) -> CurveSelectionDecision:
    if isinstance(value, CurveSelectionDecision):
        decision = value
    elif isinstance(value, str):
        if group_hint is None:
            raise ValueError("Строковая стратегия требует имени группы.")
        decision = CurveSelectionDecision(group=group_hint, method=value)  # type: ignore[arg-type]
    elif isinstance(value, Mapping):
        payload = dict(value)
        if group_hint is not None:
            supplied_group = payload.get("group")
            if supplied_group is not None and str(supplied_group) != group_hint:
                raise ValueError(
                    f"Решение для группы {group_hint!r} содержит другую группу {supplied_group!r}."
                )
            payload["group"] = group_hint
        try:
            decision = CurveSelectionDecision(**payload)
        except TypeError as exc:
            raise ValueError(f"Некорректная структура выбора кривой: {exc}") from exc
    else:
        raise ValueError(f"Неподдерживаемый выбор кривой: {type(value).__name__}.")
    if group_hint is not None and decision.group != group_hint:
        raise ValueError(
            f"Решение для группы {group_hint!r} содержит другую группу {decision.group!r}."
        )
    if decision.method not in CURVE_SELECTION_METHODS:
        raise ValueError(f"Неизвестный способ выбора кривой: {decision.method!r}.")
    return decision


def resolve_curve_selections(
    frame: pd.DataFrame,
    selections: CurveSelectionInput = None,
) -> dict[str, CurveSelectionDecision]:
    """Resolve explicit, order-independent publication selections.

    A single-test group is unambiguous and therefore defaults to
    ``individual_curves``. Every repeated group requires a supplied decision.
    """

    if "group" not in frame or "test_id" not in frame:
        raise ValueError("Для выбора публикационных кривых нужны group и test_id.")
    groups = sorted(frame["group"].dropna().astype(str).unique().tolist())
    supplied: dict[str, CurveSelectionDecision] = {}
    if selections is None:
        pass
    elif isinstance(selections, Mapping):
        for raw_group, value in selections.items():
            group = str(raw_group)
            if group in supplied:
                raise ValueError(f"Выбор для группы {group!r} задан повторно.")
            supplied[group] = _coerce_curve_selection(value, group_hint=group)
    else:
        for value in selections:
            decision = _coerce_curve_selection(value)
            if decision.group in supplied:
                raise ValueError(f"Выбор для группы {decision.group!r} задан повторно.")
            supplied[decision.group] = decision
    unknown = sorted(set(supplied) - set(groups))
    if unknown:
        raise ValueError(f"Выбор задан для отсутствующих групп: {', '.join(unknown)}.")

    resolved: dict[str, CurveSelectionDecision] = {}
    for group in groups:
        group_frame = frame[frame["group"].astype(str) == group]
        test_ids = sorted(group_frame["test_id"].dropna().astype(str).unique().tolist())
        decision = supplied.get(group)
        if decision is None:
            if len(test_ids) != 1:
                raise ValueError(
                    f"Группа {group!r} содержит {len(test_ids)} повторностей; явно выберите "
                    "mean_curve, median_curve, manual_representative или individual_curves."
                )
            decision = CurveSelectionDecision(group=group, method="individual_curves")
        if decision.method == "manual_representative":
            test_id = str(decision.test_id or "")
            author = str(decision.author or "").strip()
            reason = str(decision.reason or "").strip()
            if test_id not in test_ids:
                raise ValueError(
                    f"manual_representative для {group!r}: test_id {test_id!r} не входит в группу."
                )
            if not author:
                raise ValueError("Для manual_representative требуется author.")
            if not reason:
                raise ValueError("Для manual_representative требуется непустой reason.")
            timestamp = _parse_utc_timestamp(decision.timestamp_utc)
            decision = CurveSelectionDecision(
                group=group,
                method=decision.method,
                test_id=test_id,
                author=author,
                timestamp_utc=timestamp,
                reason=reason,
            )
        resolved[group] = decision
    return resolved


def _axis_spec(frame: pd.DataFrame, axis_mode: str) -> tuple[pd.Series, pd.Series, str, str, list[str]]:
    warnings: list[str] = []
    if axis_mode == "F-s":
        diameter = pd.to_numeric(frame.get("D_mm"), errors="coerce").dropna().unique()
        if len(diameter) > 1:
            warnings.append(
                "Общий F–s содержит разные диаметры D. Для сравнения основным выбран p–s."
            )
        return frame["F_kN"], frame["settlement_mm"], "F, кН", "s, мм", warnings
    if axis_mode == "p-s":
        return frame["p_kPa"], frame["settlement_mm"], "p, кПа", "s, мм", warnings
    if axis_mode == "p-s/D":
        y = frame["settlement_mm"] / frame["D_mm"]
        return frame["p_kPa"], y, "p, кПа", "s/D", warnings
    if axis_mode == "p/pu-s/D":
        x = pd.Series(np.nan, index=frame.index, name="p_over_pu")
        for test_id, part in frame.groupby("test_id", sort=False):
            confirmed = pd.to_numeric(part.get("pu_kPa_confirmed"), errors="coerce").dropna()
            if confirmed.empty:
                warnings.append(
                    f"{test_id}: нет явно подтверждённого pu; интервальная ступень разрушения не использована для p/pu."
                )
                continue
            pu = float(confirmed.iloc[0])
            if pu <= 0:
                warnings.append(f"{test_id}: pu должно быть положительным; кривая исключена.")
                continue
            x.loc[part.index] = part["p_kPa"] / pu
        if x.notna().sum() == 0:
            raise ValueError("Для p/pu нет ни одного испытания с достигнутым разрушением.")
        y = frame["settlement_mm"] / frame["D_mm"]
        return x, y, "p/pu", "s/D", warnings
    if axis_mode == "F/(gammaD3)-s/D":
        if "gamma_kN_m3" not in frame or frame["gamma_kN_m3"].isna().all():
            raise ValueError("Для F/(γD³) требуется gamma_kN_m3 в metadata.")
        diameter_m = frame["D_mm"] / 1000.0
        x = frame["F_kN"] / (frame["gamma_kN_m3"] * diameter_m**3)
        y = frame["settlement_mm"] / frame["D_mm"]
        return x, y, "F/(γD³)", "s/D", warnings
    if axis_mode == "p/(gammaD)-s/D":
        if "gamma_kN_m3" not in frame or frame["gamma_kN_m3"].isna().all():
            raise ValueError("Для p/(γD) требуется gamma_kN_m3 в metadata.")
        diameter_m = frame["D_mm"] / 1000.0
        x = frame["p_kPa"] / (frame["gamma_kN_m3"] * diameter_m)
        y = frame["settlement_mm"] / frame["D_mm"]
        return x, y, "p/(γD)", "s/D", warnings
    raise ValueError(f"Неизвестный режим осей: {axis_mode}")


def _configure_antonov_axes(
    ax: mpl.axes.Axes,
    *,
    xlabel: str,
    ylabel: str,
    major_step: float | None = None,
    minor_step: float | None = None,
) -> None:
    ax.xaxis.set_label_position("top")
    ax.xaxis.tick_top()
    ax.tick_params(axis="x", which="both", top=True, bottom=False, labeltop=True, labelbottom=False)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.invert_yaxis()
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.9)
        spine.set_color("black")
    if major_step is not None and major_step > 0:
        ax.xaxis.set_major_locator(MultipleLocator(major_step))
    if minor_step is not None and minor_step > 0:
        ax.xaxis.set_minor_locator(MultipleLocator(minor_step))
    else:
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.grid(True, which="major", color="0.68", linewidth=0.55)
    ax.grid(True, which="minor", color="0.86", linewidth=0.35)
    ax.set_axisbelow(True)


def _reasonable_limits(
    ax: mpl.axes.Axes,
    x_values: list[float],
    y_values: list[float],
    *,
    fixed_axes: tuple[float, float, float, float] | None,
) -> None:
    if fixed_axes is not None:
        xmin, xmax, ymin, ymax = map(float, fixed_axes)
        ax.set_xlim(xmin, xmax)
        # Set in ascending physical order, then invert.
        ax.set_ylim(ymin, ymax)
        if not ax.yaxis_inverted():
            ax.invert_yaxis()
        return
    finite_x = np.asarray(x_values, dtype=float)
    finite_y = np.asarray(y_values, dtype=float)
    finite_x = finite_x[np.isfinite(finite_x)]
    finite_y = finite_y[np.isfinite(finite_y)]
    if len(finite_x):
        xmin = min(0.0, float(finite_x.min()))
        xmax = float(finite_x.max())
        span = max(xmax - xmin, 1.0)
        ax.set_xlim(xmin, xmax + 0.08 * span)
    if len(finite_y):
        ymin = min(0.0, float(finite_y.min()))
        ymax = float(finite_y.max())
        span = max(ymax - ymin, 1.0)
        ax.set_ylim(ymin, ymax + 0.10 * span)
        if not ax.yaxis_inverted():
            ax.invert_yaxis()


def _failure_x(part: pd.DataFrame, x: pd.Series) -> tuple[float | None, float | None, bool]:
    ordered = (
        part.sort_values("sequence_no", kind="stable") if "sequence_no" in part else part
    )
    x_local = x.loc[ordered.index]
    failure_mask = ordered.get(
        "is_failure", pd.Series(False, index=ordered.index)
    ).astype(bool)
    failure_positions = np.flatnonzero(failure_mask.to_numpy())
    if len(failure_positions):
        failure_position = int(failure_positions[0])
        row_index = ordered.index[failure_position]
        value = x_local.loc[row_index]
        before = ordered.iloc[:failure_position]
        stable_idx = before[
            x_local.loc[before.index].notna() & _stable_status_mask(before)
        ].index
        lower = float(x_local.loc[stable_idx[-1]]) if len(stable_idx) else None
        return float(value) if pd.notna(value) else None, lower, False
    valid = x_local[_stable_status_mask(ordered)].dropna()
    return (float(valid.max()) if not valid.empty else None), None, True


def _draw_failure_event(
    ax: mpl.axes.Axes,
    part: pd.DataFrame,
    x: pd.Series,
    *,
    label_level: int = 0,
    curve_label: str | None = None,
) -> list[float]:
    event_x, lower, censored = _failure_x(part, x)
    if event_x is None:
        return []
    if censored:
        label = "Fu > Fmax" if x.name == "F_kN" else "pu > pmax"
        if curve_label:
            label += f", {curve_label}"
        ax.annotate(
            label,
            xy=(event_x, 0.995),
            xycoords=("data", "axes fraction"),
            xytext=(event_x, 0.955 - 0.055 * label_level),
            textcoords=("data", "axes fraction"),
            ha="center",
            va="top",
            fontsize=8,
            color="black",
            annotation_clip=False,
        )
    else:
        if lower is not None:
            ax.axvspan(min(lower, event_x), max(lower, event_x), color="black", alpha=0.06, zorder=0)
        ax.axvline(event_x, color="black", linestyle=":", linewidth=1.0)
        symbol = "Fu" if x.name == "F_kN" else ("p/pu" if x.name == "p_over_pu" else "pu")
        label = "разрушение" + (f", {curve_label}" if curve_label else "")
        if lower is not None:
            label += f"\n{lower:.3g} < {symbol} ≤ {event_x:.3g}"
        ax.annotate(
            label,
            xy=(event_x, 0.995),
            xycoords=("data", "axes fraction"),
            xytext=(event_x, 0.955 - 0.055 * label_level),
            textcoords=("data", "axes fraction"),
            arrowprops={"arrowstyle": "-|>", "color": "black", "lw": 0.8},
            ha="center",
            va="top",
            fontsize=8,
            color="black",
            annotation_clip=False,
        )
    return [event_x]


def _label_curve_numbers(
    ax: mpl.axes.Axes, endpoints: list[tuple[float, float, int]], y_span: float
) -> None:
    if not endpoints:
        return
    ordered = sorted(endpoints, key=lambda item: item[1])
    gap = max(0.035 * y_span, 1e-9)
    adjusted: list[tuple[float, float, float, int]] = []
    previous = -np.inf
    for x, y, number in ordered:
        target = max(y, previous + gap)
        adjusted.append((x, y, target, number))
        previous = target
    x_limits = ax.get_xlim()
    x_offset = 0.012 * abs(float(x_limits[1] - x_limits[0]))
    for x, y, target, number in adjusted:
        ax.annotate(
            str(number),
            xy=(x, y),
            xytext=(x + x_offset, target),
            textcoords="data",
            fontsize=9,
            ha="left",
            va="center",
            arrowprops=(
                None
                if np.isclose(target, y)
                else {"arrowstyle": "-", "color": "black", "lw": 0.6}
            ),
            color="black",
        )


def _ordered_test_part(part: pd.DataFrame) -> pd.DataFrame:
    if "sequence_no" in part:
        return part.sort_values("sequence_no", kind="stable")
    return part


def _aggregate_points(
    group: pd.DataFrame,
    *,
    axis_mode: str,
    statistic: Literal["mean", "median"],
    confidence: float,
    bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    result = aggregate_group_curve(
        group,
        axis_mode=axis_mode,
        statistic=statistic,
        confidence=confidence,
        bootstrap=bootstrap,
        seed=seed,
    ).copy()
    required = {"x", "y", "n", "measured_n", "interpolated_n"}
    missing = sorted(required.difference(result.columns))
    if missing:
        raise ValueError(f"Агрегация кривой не вернула столбцы: {missing}.")
    result["x"] = pd.to_numeric(result["x"], errors="coerce")
    result["y"] = pd.to_numeric(result["y"], errors="coerce")
    if "draw_marker" not in result:
        result["draw_marker"] = pd.to_numeric(
            result["interpolated_n"], errors="coerce"
        ).fillna(0).eq(0)
    result["draw_marker"] = result["draw_marker"].fillna(False).astype(bool)
    finite = np.isfinite(result["x"]) & np.isfinite(result["y"])
    result = result.loc[finite].sort_values("x", kind="stable").reset_index(drop=True)
    if result.empty:
        raise ValueError("После агрегации не осталось конечных точек для графика.")
    return result


def _raw_plot_points(
    part: pd.DataFrame,
    x: pd.Series,
    y: pd.Series,
    *,
    axis_mode: str,
    group: str,
    test_id: str,
    curve_number: int,
    selection_method: str,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    ordered = _ordered_test_part(part)
    px = x.loc[ordered.index]
    py = y.loc[ordered.index]
    valid = px.notna() & py.notna() & ~ordered.get(
        "is_failure", pd.Series(False, index=ordered.index)
    ).astype(bool)
    points = pd.DataFrame(
        {
            "group": group,
            "test_id": test_id,
            "curve_number": curve_number,
            "selection_method": selection_method,
            "axis_mode": axis_mode,
            "x": px[valid].to_numpy(dtype=float),
            "y": py[valid].to_numpy(dtype=float),
            "n": 1,
            "measured_n": 1,
            "interpolated_n": 0,
            "draw_marker": True,
        }
    )
    return points, px, py, valid


def plot_curves(
    frame: pd.DataFrame,
    *,
    mode: str = "raw_protocol",
    axis_mode: str = "p-s",
    ci_method: str = "t",
    fixed_axes: tuple[float, float, float, float] | None = None,
    major_step: float | None = None,
    minor_step: float | None = None,
    pcr_result: PCRResult | None = None,
    bootstrap: int = 500,
    seed: int = 202604,
    confidence: float = 0.95,
    selections: CurveSelectionInput = None,
) -> PlotOutput:
    if frame.empty:
        raise ValueError("Нет данных для графика.")
    if mode not in {"raw_protocol", "antonov_publication", "group_mean_ci", "diagnostic", "normalized"}:
        raise ValueError(f"Неизвестный режим графика: {mode}")
    if mode == "diagnostic":
        return plot_pcr_diagnostic(
            frame,
            result=pcr_result,
            fixed_axes=fixed_axes,
            major_step=major_step,
            minor_step=minor_step,
            bootstrap=bootstrap,
            seed=seed,
        )
    if mode == "normalized" and axis_mode not in {
        "p-s/D",
        "p/pu-s/D",
        "F/(gammaD3)-s/D",
        "p/(gammaD)-s/D",
    }:
        raise ValueError("Режим normalized требует нормированных осей: p–s/D, p/pu–s/D или γD.")
    if mode == "group_mean_ci" and axis_mode not in {"F-s", "p-s", "p-s/D"}:
        raise ValueError("Режим group_mean_ci поддерживает только F–s, p–s и p–s/D.")
    if ci_method not in {"t", "simultaneous"}:
        raise ValueError("ci_method должен быть t или simultaneous.")
    x, y, xlabel, ylabel, warnings = _axis_spec(frame, axis_mode)
    x = pd.Series(x, index=frame.index, name=getattr(x, "name", None))
    y = pd.Series(y, index=frame.index)

    with mpl.rc_context(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.linewidth": 0.9,
            "savefig.bbox": "tight",
            "svg.fonttype": "none",
        }
    ):
        fig, ax = plt.subplots(figsize=(7.2, 5.0), constrained_layout=True)
        _configure_antonov_axes(
            ax, xlabel=xlabel, ylabel=ylabel, major_step=major_step, minor_step=minor_step
        )
        groups = sorted(frame["group"].dropna().astype(str).unique().tolist())
        group_style = {name: LINE_STYLES[i % len(LINE_STYLES)] for i, name in enumerate(groups)}
        tests = sorted(frame["test_id"].dropna().astype(str).unique().tolist())
        test_marker = {name: MARKERS[i % len(MARKERS)] for i, name in enumerate(tests)}
        curve_map: dict[int, str] = {}
        endpoints: list[tuple[float, float, int]] = []
        all_x: list[float] = []
        all_y: list[float] = []
        selection_records: list[dict[str, Any]] = []
        plotted_point_tables: list[pd.DataFrame] = []

        if mode in {"raw_protocol", "normalized"}:
            for number, test_id in enumerate(tests, 1):
                part = frame[frame["test_id"].astype(str) == test_id]
                group_name = str(part["group"].iloc[0])
                points, px, py, valid = _raw_plot_points(
                    part,
                    x,
                    y,
                    axis_mode=axis_mode,
                    group=group_name,
                    test_id=test_id,
                    curve_number=number,
                    selection_method="individual_curves",
                )
                ordered = _ordered_test_part(part)
                # Protocol sequence_no is preserved: no sorting by load and no smoothing.
                line = ax.plot(
                    px[valid].to_numpy(),
                    py[valid].to_numpy(),
                    color="black",
                    linestyle=group_style[group_name],
                    marker=test_marker[str(test_id)],
                    linewidth=0.8,
                    markersize=4.0,
                    markerfacecolor="white",
                    markeredgecolor="black",
                    label=str(test_id),
                )[0]
                line.set_gid(f"raw-{test_id}")
                values_x = px[valid].to_numpy(dtype=float)
                values_y = py[valid].to_numpy(dtype=float)
                all_x.extend(values_x.tolist())
                all_y.extend(values_y.tolist())
                if len(values_x):
                    endpoints.append((values_x[-1], values_y[-1], number))
                curve_map[number] = f"{test_id} — {group_name}"
                plotted_point_tables.append(points)
                all_x.extend(
                    _draw_failure_event(
                        ax,
                        ordered,
                        x,
                        label_level=(number - 1) % 3,
                        curve_label=str(number),
                    )
                )

        elif mode == "antonov_publication":
            resolved = resolve_curve_selections(frame, selections)
            number = 0
            for group_name in groups:
                group = frame[frame["group"].astype(str) == group_name]
                decision = resolved[group_name]
                selection_records.append(asdict(decision))
                if decision.method in {"mean_curve", "median_curve"}:
                    number += 1
                    statistic: Literal["mean", "median"] = (
                        "mean" if decision.method == "mean_curve" else "median"
                    )
                    aggregate = _aggregate_points(
                        group,
                        axis_mode=axis_mode,
                        statistic=statistic,
                        confidence=confidence,
                        bootstrap=bootstrap,
                        seed=seed,
                    )
                    px = aggregate["x"].to_numpy(dtype=float)
                    py = aggregate["y"].to_numpy(dtype=float)
                    marker_mask = aggregate["draw_marker"].to_numpy(dtype=bool)
                    line = ax.plot(
                        px,
                        py,
                        color="black",
                        linestyle=group_style[group_name],
                        linewidth=1.6,
                    )[0]
                    line.set_gid(f"aggregate-{group_name}-{decision.method}")
                    ax.plot(
                        px[marker_mask],
                        py[marker_mask],
                        linestyle="none",
                        color="black",
                        marker=MARKERS[(number - 1) % len(MARKERS)],
                        markerfacecolor="white",
                        markersize=4.5,
                    )[0].set_gid(f"aggregate-markers-{group_name}-{decision.method}")
                    label = "средняя" if statistic == "mean" else "медиана"
                    tests_in_group = group["test_id"].nunique()
                    curve_map[number] = f"{group_name}, {label}, n={tests_in_group}"
                    endpoints.append((float(px[-1]), float(py[-1]), number))
                    all_x.extend(px.tolist())
                    all_y.extend(py.tolist())
                    rendered = aggregate.copy()
                    rendered["group"] = group_name
                    rendered["test_id"] = pd.NA
                    rendered["curve_number"] = number
                    rendered["selection_method"] = decision.method
                    rendered["axis_mode"] = axis_mode
                    plotted_point_tables.append(rendered)
                    # Failure bounds are never pooled into an aggregate curve.
                    continue

                test_ids = (
                    [str(decision.test_id)]
                    if decision.method == "manual_representative"
                    else sorted(group["test_id"].dropna().astype(str).unique().tolist())
                )
                for test_id in test_ids:
                    number += 1
                    part = group[group["test_id"].astype(str) == test_id]
                    points, px_s, py_s, valid = _raw_plot_points(
                        part,
                        x,
                        y,
                        axis_mode=axis_mode,
                        group=group_name,
                        test_id=test_id,
                        curve_number=number,
                        selection_method=decision.method,
                    )
                    px = px_s[valid].to_numpy(dtype=float)
                    py = py_s[valid].to_numpy(dtype=float)
                    line = ax.plot(
                        px,
                        py,
                        color="black",
                        linestyle=group_style[group_name],
                        marker=test_marker[test_id],
                        markerfacecolor="white",
                        markeredgecolor="black",
                        linewidth=1.4,
                        markersize=4.5,
                    )[0]
                    line.set_gid(f"publication-{group_name}-{test_id}")
                    if len(px):
                        endpoints.append((float(px[-1]), float(py[-1]), number))
                    all_x.extend(px.tolist())
                    all_y.extend(py.tolist())
                    if decision.method == "manual_representative":
                        curve_map[number] = f"{group_name}, репрезентативная {test_id}"
                    else:
                        curve_map[number] = f"{group_name}, индивидуальная {test_id}"
                    plotted_point_tables.append(points)
                    all_x.extend(
                        _draw_failure_event(
                            ax,
                            _ordered_test_part(part),
                            x,
                            label_level=(number - 1) % 3,
                            curve_label=str(number),
                        )
                    )

        elif mode == "group_mean_ci":
            number = 0
            for group_name in groups:
                group = frame[frame["group"].astype(str) == group_name]
                number += 1
                selection_records.append(
                    {
                        **asdict(
                            CurveSelectionDecision(group=group_name, method="mean_curve")
                        ),
                        "source": "group_mean_ci",
                    }
                )
                for test_id in sorted(
                    group["test_id"].dropna().astype(str).unique().tolist()
                ):
                    part = group[group["test_id"].astype(str) == test_id]
                    _, px, py, valid = _raw_plot_points(
                        part,
                        x,
                        y,
                        axis_mode=axis_mode,
                        group=group_name,
                        test_id=test_id,
                        curve_number=number,
                        selection_method="group_mean_ci_source",
                    )
                    ax.plot(
                        px[valid],
                        py[valid],
                        color="0.45",
                        linestyle=group_style[str(group_name)],
                        marker=test_marker[str(test_id)],
                        markerfacecolor="white",
                        linewidth=0.55,
                        markersize=3.2,
                    )
                    all_x.extend(px[valid].astype(float).tolist())
                    all_y.extend(py[valid].astype(float).tolist())
                mean = _aggregate_points(
                    group,
                    axis_mode=axis_mode,
                    statistic="mean",
                    confidence=confidence,
                    bootstrap=bootstrap,
                    seed=seed,
                )
                low_name, high_name = (
                    ("t_ci_low", "t_ci_high")
                    if ci_method == "t"
                    else ("simultaneous_low", "simultaneous_high")
                )
                px = mean["x"].to_numpy(dtype=float)
                py = mean["y"].to_numpy(dtype=float)
                low = mean[low_name].to_numpy(dtype=float)
                high = mean[high_name].to_numpy(dtype=float)
                ax.fill_between(px, low, high, color="black", alpha=0.10, linewidth=0)
                line = ax.plot(
                    px,
                    py,
                    color="black",
                    linestyle=group_style[group_name],
                    linewidth=1.7,
                )[0]
                line.set_gid(f"group-mean-{group_name}")
                marker_mask = mean["draw_marker"].to_numpy(dtype=bool)
                ax.plot(
                    px[marker_mask],
                    py[marker_mask],
                    linestyle="none",
                    marker=MARKERS[(number - 1) % len(MARKERS)],
                    color="black",
                    markerfacecolor="white",
                    markersize=4.3,
                )[0].set_gid(f"group-mean-markers-{group_name}")
                for x_value, y_value, count in zip(px, py, mean["n"], strict=True):
                    ax.annotate(
                        f"n={count}",
                        (x_value, y_value),
                        xytext=(0, 5),
                        textcoords="offset points",
                        fontsize=6.5,
                        ha="center",
                    )
                all_x.extend(px.tolist())
                all_y.extend(np.concatenate([py, low[np.isfinite(low)], high[np.isfinite(high)]]).tolist())
                if len(px):
                    endpoints.append((float(px[-1]), float(py[-1]), number))
                curve_map[number] = f"{group_name}, средняя и 95% ДИ, n={group['test_id'].nunique()}"
                rendered = mean.copy()
                rendered["group"] = group_name
                rendered["test_id"] = pd.NA
                rendered["curve_number"] = number
                rendered["selection_method"] = "mean_curve"
                rendered["axis_mode"] = axis_mode
                plotted_point_tables.append(rendered)
                # Failure and censoring are shown only in the individual interval plot.

        _reasonable_limits(ax, all_x, all_y, fixed_axes=fixed_axes)
        y_limits = ax.get_ylim()
        _label_curve_numbers(ax, endpoints, abs(float(y_limits[0] - y_limits[1])))
        caption = "Кривые нагрузки–осадки. " + "; ".join(
            f"{number} — {description}" for number, description in curve_map.items()
        )
        caption += (
            ". Маркеры — уровни без интерполированных вкладов; "
            "линии между точками — ломаные без spline-сглаживания."
        )
        fig.canvas.draw()
        plotted_points = (
            pd.concat(plotted_point_tables, ignore_index=True, sort=False)
            if plotted_point_tables
            else pd.DataFrame()
        )
        return PlotOutput(
            fig,
            caption,
            warnings,
            curve_map,
            selection_records=selection_records,
            plotted_points=plotted_points,
        )


def _finite_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _row_bool(row: pd.Series, *names: str) -> bool:
    for name in names:
        if name not in row:
            continue
        value = row.get(name)
        if value is None or bool(pd.isna(value)):
            continue
        return bool(value)
    return False


def _failure_bounds(
    row: pd.Series,
    capacity_axis: Literal["force", "pressure"],
) -> tuple[float | None, float | None]:
    if capacity_axis == "force":
        lower = _finite_number(row.get("Fu_lower"))
        upper = _finite_number(row.get("Fu_upper"))
    else:
        lower = _finite_number(row.get("pu_lower"))
        upper = _finite_number(row.get("pu_upper"))
    row_axis = str(row.get("capacity_axis") or row.get("capacity_kind") or "").strip().lower()
    if lower is None and (not row_axis or row_axis == capacity_axis):
        lower = _finite_number(row.get("lower_bound"))
    if upper is None and (not row_axis or row_axis == capacity_axis):
        upper = _finite_number(row.get("upper_bound"))
    return lower, upper


def plot_failure_intervals(
    failure_table: pd.DataFrame,
    *,
    capacity_axis: Literal["force", "pressure"] = "force",
) -> PlotOutput:
    """Plot individual capacity intervals without a pooled point estimate."""

    if capacity_axis not in {"force", "pressure"}:
        raise ValueError("capacity_axis должен быть force или pressure.")
    if failure_table.empty:
        raise ValueError("Нет данных о разрушении и цензурировании.")
    if "test_id" not in failure_table:
        raise ValueError("В таблице разрушения отсутствует test_id.")
    ordered = failure_table.copy()
    ordered["test_id"] = ordered["test_id"].astype(str)
    if ordered["test_id"].duplicated().any():
        duplicates = sorted(ordered.loc[ordered["test_id"].duplicated(), "test_id"].unique())
        raise ValueError(f"В таблице разрушения повторяются test_id: {', '.join(duplicates)}.")
    ordered = ordered.sort_values("test_id", kind="stable").reset_index(drop=True)

    records: list[dict[str, Any]] = []
    all_bounds: list[float] = []
    for _, row in ordered.iterrows():
        lower, upper = _failure_bounds(row, capacity_axis)
        observed = _row_bool(
            row,
            "failure_observed",
            "observed_failure",
            "failure_reached",
        )
        right_censored = _row_bool(row, "right_censored")
        explicit_type = str(row.get("censoring_type") or "").strip().lower()
        if explicit_type in {"right", "right_censored"}:
            right_censored = True
        interval_censored = lower is not None and upper is not None and (
            _row_bool(row, "interval_censored") or (observed and lower < upper)
        )
        if right_censored:
            censoring_type = "right_censored"
        elif interval_censored:
            censoring_type = "interval_censored"
        elif observed and upper is not None and lower == upper:
            censoring_type = "observed_exact"
        else:
            censoring_type = "indeterminate"
        records.append(
            {
                "test_id": str(row["test_id"]),
                "capacity_axis": capacity_axis,
                "lower_bound": lower,
                "upper_bound": upper,
                "failure_observed": observed,
                "observed_failure": observed,
                "interval_censored": interval_censored,
                "right_censored": right_censored,
                "censoring_type": censoring_type,
            }
        )
        all_bounds.extend(value for value in (lower, upper) if value is not None)

    finite_bounds = np.asarray(all_bounds, dtype=float)
    if len(finite_bounds):
        data_min = float(finite_bounds.min())
        data_max = float(finite_bounds.max())
        data_span = max(data_max - data_min, abs(data_max) * 0.05, 1.0)
    else:
        data_min, data_max, data_span = 0.0, 1.0, 1.0
    arrow_length = max(data_span * 0.16, 1e-9)

    warnings: list[str] = []
    with mpl.rc_context(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.linewidth": 0.9,
            "savefig.bbox": "tight",
            "svg.fonttype": "none",
        }
    ):
        fig, ax = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
        ax.xaxis.set_label_position("top")
        ax.xaxis.tick_top()
        ax.tick_params(axis="x", which="both", top=True, bottom=False, labeltop=True, labelbottom=False)
        ax.set_xlabel("Fu, кН" if capacity_axis == "force" else "pu, кПа")
        y_positions = np.arange(len(records), dtype=float)
        ax.set_yticks(y_positions, [record["test_id"] for record in records])
        ax.set_ylabel("test ID")
        ax.set_ylim(len(records) - 0.35, -0.65)
        ax.grid(True, axis="x", which="major", color="0.72", linewidth=0.5)
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.grid(True, axis="x", which="minor", color="0.88", linewidth=0.35)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("black")
            spine.set_linewidth(0.9)

        arrow_ends: list[float] = []
        for y_position, record in zip(y_positions, records, strict=True):
            test_id = record["test_id"]
            lower = record["lower_bound"]
            upper = record["upper_bound"]
            kind = record["censoring_type"]
            if kind == "interval_censored" and lower is not None and upper is not None:
                interval = ax.plot(
                    [lower, upper],
                    [y_position, y_position],
                    color="black",
                    linewidth=1.2,
                )[0]
                interval.set_gid(f"failure-interval-{test_id}")
                lower_artist = ax.plot(
                    lower,
                    y_position,
                    marker="o",
                    markerfacecolor="white",
                    markeredgecolor="black",
                    linestyle="none",
                    markersize=5.2,
                )[0]
                lower_artist.set_gid(f"failure-open-lower-{test_id}")
                upper_artist = ax.plot(
                    upper,
                    y_position,
                    marker="o",
                    markerfacecolor="black",
                    markeredgecolor="black",
                    linestyle="none",
                    markersize=5.2,
                )[0]
                upper_artist.set_gid(f"failure-closed-upper-{test_id}")
            elif kind == "right_censored" and lower is not None:
                arrow_end = lower + arrow_length
                arrow_ends.append(arrow_end)
                lower_artist = ax.plot(
                    lower,
                    y_position,
                    marker="o",
                    markerfacecolor="white",
                    markeredgecolor="black",
                    linestyle="none",
                    markersize=4.8,
                )[0]
                lower_artist.set_gid(f"failure-right-lower-{test_id}")
                annotation = ax.annotate(
                    "",
                    xy=(arrow_end, y_position),
                    xytext=(lower, y_position),
                    arrowprops={"arrowstyle": "-|>", "color": "black", "lw": 1.1},
                )
                annotation.set_gid(f"failure-right-arrow-{test_id}")
            elif kind == "observed_exact" and upper is not None:
                exact = ax.plot(
                    upper,
                    y_position,
                    marker="o",
                    markerfacecolor="black",
                    markeredgecolor="black",
                    linestyle="none",
                    markersize=5.2,
                )[0]
                exact.set_gid(f"failure-observed-{test_id}")
            else:
                warnings.append(
                    f"{test_id}: недостаточно границ для отображения интервала {capacity_axis}."
                )
                marker_x = lower if lower is not None else upper
                if marker_x is not None:
                    indeterminate = ax.plot(
                        marker_x,
                        y_position,
                        marker="x",
                        color="black",
                        linestyle="none",
                        markersize=5.5,
                    )[0]
                    indeterminate.set_gid(f"failure-indeterminate-{test_id}")
                else:
                    missing_text = ax.text(
                        0.01,
                        y_position,
                        "нет границ",
                        transform=ax.get_yaxis_transform(),
                        ha="left",
                        va="center",
                        fontsize=8,
                        color="black",
                    )
                    missing_text.set_gid(f"failure-indeterminate-{test_id}")

        xmin = min(0.0, data_min)
        xmax_data = max([data_max, *arrow_ends]) if arrow_ends else data_max
        span = max(xmax_data - xmin, 1.0)
        ax.set_xlim(xmin, xmax_data + 0.08 * span)
        observed_n = sum(bool(record["failure_observed"]) for record in records)
        interval_n = sum(bool(record["interval_censored"]) for record in records)
        right_n = sum(bool(record["right_censored"]) for record in records)
        caption = (
            "Индивидуальные интервалы предельной нагрузки: "
            f"наблюдалось разрушение — {observed_n}; интервально цензурировано — {interval_n}; "
            f"правоцензурировано — {right_n}. Сводная точечная оценка не рассчитывалась."
        )
        fig.canvas.draw()
        curve_map = {index + 1: record["test_id"] for index, record in enumerate(records)}
        return PlotOutput(
            fig,
            caption,
            warnings,
            curve_map,
            plotted_points=pd.DataFrame(records),
        )


def plot_pcr_diagnostic(
    frame: pd.DataFrame,
    *,
    result: PCRResult | None = None,
    fixed_axes: tuple[float, float, float, float] | None = None,
    major_step: float | None = None,
    minor_step: float | None = None,
    bootstrap: int = 500,
    seed: int = 202604,
) -> PlotOutput:
    test_ids = sorted(frame["test_id"].dropna().astype(str).unique().tolist())
    if len(test_ids) != 1:
        raise ValueError(
            "Диагностика pcr требует ровно одно явно отфильтрованное испытание."
        )
    selected_test = test_ids[0]
    part = frame[frame["test_id"].astype(str) == selected_test]
    result = result or fit_segmented_pcr(part, bootstrap=bootstrap, seed=seed)
    used = part.loc[result.used_indices].sort_values("p_kPa", kind="stable")
    p = used["p_kPa"].to_numpy(dtype=float)
    s = used["settlement_mm"].to_numpy(dtype=float)
    fitted = np.asarray(result.fitted, dtype=float)
    residuals = np.asarray(result.residuals, dtype=float)
    with mpl.rc_context({"font.family": "DejaVu Sans", "font.size": 9, "svg.fonttype": "none"}):
        fig, (ax, residual_ax) = plt.subplots(
            2,
            1,
            figsize=(7.2, 6.2),
            gridspec_kw={"height_ratios": [3.0, 1.0]},
            constrained_layout=True,
        )
        _configure_antonov_axes(
            ax,
            xlabel="p, кПа",
            ylabel="s, мм",
            major_step=major_step,
            minor_step=minor_step,
        )
        ax.plot(p, s, color="black", marker="o", markerfacecolor="white", linewidth=0.8, label="измерения")
        ax.plot(p, fitted, color="black", linestyle="--", linewidth=1.5, label="сегментированная модель")
        ax.axvline(result.pcr_auto, color="black", linestyle=":", linewidth=1.2)
        if result.pcr_ci_low is not None and result.pcr_ci_high is not None:
            ax.axvspan(result.pcr_ci_low, result.pcr_ci_high, color="black", alpha=0.09)
        ax.annotate(
            f"pcr={result.pcr_auto:.3g} кПа",
            xy=(result.pcr_auto, float(np.interp(result.pcr_auto, p, fitted))),
            xytext=(8, -12),
            textcoords="offset points",
            fontsize=8,
        )
        if result.pcr_manual is not None:
            ax.axvline(result.pcr_manual, color="0.35", linestyle="-.", linewidth=1.1)
            ax.annotate(
                f"подтверждено: {result.pcr_manual:.3g}",
                xy=(result.pcr_manual, float(np.interp(result.pcr_manual, p, fitted))),
                xytext=(8, 12),
                textcoords="offset points",
                fontsize=8,
            )
        _reasonable_limits(ax, p.tolist(), np.concatenate([s, fitted]).tolist(), fixed_axes=fixed_axes)
        ax.legend(frameon=False, loc="lower right", fontsize=8)

        residual_ax.axhline(0.0, color="black", linewidth=0.8)
        residual_ax.scatter(p, residuals, marker="o", facecolors="white", edgecolors="black", s=22)
        residual_ax.set_xlabel("p, кПа")
        residual_ax.set_ylabel("остатки, мм")
        residual_ax.grid(True, which="major", color="0.78", linewidth=0.45)
        residual_ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        for spine in residual_ax.spines.values():
            spine.set_visible(True)
            spine.set_color("black")
        caption = (
            f"Диагностика pcr для {selected_test}: непрерывная сегментированная регрессия, "
            f"R²={result.r2:.3f}, AIC={result.aic:.2f}, BIC={result.bic:.2f}; "
            f"wild bootstrap, валидных повторов {result.bootstrap_valid}."
        )
        fig.canvas.draw()
        return PlotOutput(fig, caption, [], {1: selected_test})


def plot_stamp_schematic(metadata: dict[str, Any]) -> Figure:
    """Optional simple vector scheme driven only by supplied metadata."""

    fig, ax = plt.subplots(figsize=(5.2, 2.8), constrained_layout=True)
    ax.set_aspect("equal")
    ax.axis("off")
    diameter = float(metadata.get("stamp_diameter_mm", 300.0))
    radius = diameter / 2.0
    layers = int((metadata.get("reinforcement") or {}).get("layers", 0))
    ax.add_patch(plt.Rectangle((-radius, 0), diameter, diameter * 0.12, facecolor="0.75", edgecolor="black"))
    ax.plot([0, 0], [-diameter * 0.45, 0], color="black", linewidth=1.2)
    ax.annotate("F", xy=(0, -diameter * 0.10), xytext=(0, -diameter * 0.40), arrowprops={"arrowstyle": "-|>", "color": "black"}, ha="center")
    for layer in range(layers):
        depth = diameter * (0.32 + layer * 0.22)
        ax.plot([-diameter * 0.8, diameter * 0.8], [depth, depth], color="black", linestyle="--")
        ax.text(diameter * 0.84, depth, f"слой {layer + 1}", va="center", fontsize=8)
    ax.set_xlim(-diameter, diameter)
    ax.set_ylim(diameter, -diameter * 0.55)
    return fig


def export_figure(figure: Figure, file_format: str) -> bytes:
    fmt = file_format.lower().lstrip(".")
    if fmt not in {"svg", "pdf", "png"}:
        raise ValueError("Поддерживаются SVG, PDF и PNG.")
    buffer = BytesIO()
    kwargs: dict[str, Any] = {
        "format": fmt,
        "bbox_inches": "tight",
        "facecolor": "white",
        "metadata": {"Title": "Soil Stamp Antonov", "Creator": f"soil-stamp-antonov {VERSION}"},
    }
    if fmt == "png":
        kwargs["dpi"] = 600
    figure.savefig(buffer, **kwargs)
    return buffer.getvalue()
