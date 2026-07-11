"""Matplotlib publication and diagnostic figures."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from matplotlib.ticker import AutoMinorLocator, MultipleLocator

from .analysis import fit_segmented_pcr, group_mean_curve
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
    x_local = x.loc[part.index]
    failure = part[part.get("is_failure", pd.Series(False, index=part.index)).astype(bool)]
    if not failure.empty:
        row_index = failure.index[0]
        value = x_local.loc[row_index]
        before = part.loc[:row_index].iloc[:-1]
        stable_idx = before[x_local.loc[before.index].notna() & _stable_status_mask(before)].index
        lower = float(x_local.loc[stable_idx[-1]]) if len(stable_idx) else None
        return float(value) if pd.notna(value) else None, lower, False
    valid = x_local[_stable_status_mask(part)].dropna()
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


def _draw_group_failure_event(
    ax: mpl.axes.Axes,
    group: pd.DataFrame,
    x: pd.Series,
    *,
    curve_number: int,
    label_level: int,
) -> list[float]:
    """Draw one non-overlapping publication event per plotted group curve."""

    reached: list[tuple[float, float | None]] = []
    censored: list[float] = []
    all_events: list[float] = []
    for _, part in group.groupby("test_id", sort=False):
        event_x, lower, is_censored = _failure_x(part, x)
        if event_x is None:
            continue
        all_events.append(event_x)
        if is_censored:
            censored.append(event_x)
        else:
            reached.append((event_x, lower))
    if reached:
        event_x = float(np.mean([item[0] for item in reached]))
        lower_values = [item[1] for item in reached if item[1] is not None]
        lower = float(np.mean(lower_values)) if lower_values else None
        if lower is not None:
            ax.axvspan(min(lower, event_x), max(lower, event_x), color="black", alpha=0.06, zorder=0)
        ax.axvline(event_x, color="black", linestyle=":", linewidth=1.0)
        label = f"разрушение, кривая {curve_number}"
        if censored:
            label += " (+ценз.)"
        if lower is not None:
            symbol = "Fu" if x.name == "F_kN" else ("p/pu" if x.name == "p_over_pu" else "pu")
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
    elif censored:
        event_x = float(max(censored))
        if x.name == "F_kN":
            censor_label = f"Fu > Fmax, кривая {curve_number}"
        elif x.name == "p_kPa":
            censor_label = f"pu > pmax, кривая {curve_number}"
        else:
            censor_label = f"предел не достигнут, кривая {curve_number}"
        ax.annotate(
            censor_label,
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
    return all_events


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
        groups = list(dict.fromkeys(frame["group"].astype(str)))
        group_style = {name: LINE_STYLES[i % len(LINE_STYLES)] for i, name in enumerate(groups)}
        tests = list(dict.fromkeys(frame["test_id"].astype(str)))
        test_marker = {name: MARKERS[i % len(MARKERS)] for i, name in enumerate(tests)}
        curve_map: dict[int, str] = {}
        endpoints: list[tuple[float, float, int]] = []
        all_x: list[float] = []
        all_y: list[float] = []

        if mode in {"raw_protocol", "normalized"}:
            for number, (test_id, part) in enumerate(frame.groupby("test_id", sort=False), 1):
                group_name = str(part["group"].iloc[0])
                px = x.loc[part.index]
                py = y.loc[part.index]
                valid = px.notna() & py.notna() & ~part.get("is_failure", pd.Series(False, index=part.index)).astype(bool)
                # Source sequence is preserved: no sorting and no smoothing.
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
                all_x.extend(
                    _draw_failure_event(
                        ax,
                        part,
                        x,
                        label_level=(number - 1) % 3,
                        curve_label=str(number),
                    )
                )

        elif mode == "antonov_publication":
            number = 0
            for group_name, group in frame.groupby("group", sort=False):
                number += 1
                tests_in_group = group["test_id"].nunique()
                if tests_in_group >= 2 and axis_mode == "p-s":
                    mean = group_mean_curve(group, bootstrap=bootstrap, seed=seed)
                    px = mean["p_kPa"].to_numpy(dtype=float)
                    py = mean["mean_settlement_mm"].to_numpy(dtype=float)
                    marker_mask = mean["all_measured"].to_numpy(dtype=bool)
                    ax.plot(px, py, color="black", linestyle=group_style[str(group_name)], linewidth=1.6)
                    ax.plot(
                        px[marker_mask],
                        py[marker_mask],
                        linestyle="none",
                        color="black",
                        marker=MARKERS[(number - 1) % len(MARKERS)],
                        markerfacecolor="white",
                        markersize=4.5,
                    )
                    descriptor = f"{group_name}, средняя, n={tests_in_group}"
                else:
                    representative_id = str(group["test_id"].iloc[0])
                    representative = group[group["test_id"].astype(str) == representative_id]
                    px_s = x.loc[representative.index]
                    py_s = y.loc[representative.index]
                    valid = px_s.notna() & py_s.notna() & ~representative.get(
                        "is_failure", pd.Series(False, index=representative.index)
                    ).astype(bool)
                    px = px_s[valid].to_numpy(dtype=float)
                    py = py_s[valid].to_numpy(dtype=float)
                    ax.plot(
                        px,
                        py,
                        color="black",
                        linestyle=group_style[str(group_name)],
                        marker=MARKERS[(number - 1) % len(MARKERS)],
                        markerfacecolor="white",
                        linewidth=1.4,
                        markersize=4.5,
                    )
                    descriptor = f"{group_name}, репрезентативная {representative_id}"
                if len(px):
                    endpoints.append((float(px[-1]), float(py[-1]), number))
                all_x.extend(px.tolist())
                all_y.extend(py.tolist())
                curve_map[number] = descriptor
                all_x.extend(
                    _draw_group_failure_event(
                        ax,
                        group,
                        x,
                        curve_number=number,
                        label_level=(number - 1) % 3,
                    )
                )

        elif mode == "group_mean_ci":
            number = 0
            for group_name, group in frame.groupby("group", sort=False):
                number += 1
                for test_id, part in group.groupby("test_id", sort=False):
                    px = x.loc[part.index]
                    py = y.loc[part.index]
                    valid = px.notna() & py.notna() & ~part.get(
                        "is_failure", pd.Series(False, index=part.index)
                    ).astype(bool)
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
                if axis_mode != "p-s":
                    warnings.append("Средняя с ДИ рассчитывается в координатах p–s; выбрана только ломаная повторностей.")
                    curve_map[number] = str(group_name)
                    continue
                mean = group_mean_curve(group, bootstrap=bootstrap, seed=seed)
                low_name, high_name = (
                    ("t_ci_low_mm", "t_ci_high_mm")
                    if ci_method == "t"
                    else ("simultaneous_low_mm", "simultaneous_high_mm")
                )
                px = mean["p_kPa"].to_numpy(dtype=float)
                py = mean["mean_settlement_mm"].to_numpy(dtype=float)
                low = mean[low_name].to_numpy(dtype=float)
                high = mean[high_name].to_numpy(dtype=float)
                ax.fill_between(px, low, high, color="black", alpha=0.10, linewidth=0)
                ax.plot(px, py, color="black", linestyle=group_style[str(group_name)], linewidth=1.7)
                marker_mask = mean["all_measured"].to_numpy(dtype=bool)
                ax.plot(
                    px[marker_mask],
                    py[marker_mask],
                    linestyle="none",
                    marker=MARKERS[(number - 1) % len(MARKERS)],
                    color="black",
                    markerfacecolor="white",
                    markersize=4.3,
                )
                for p_value, s_value, count in zip(px, py, mean["n"], strict=True):
                    ax.annotate(f"n={count}", (p_value, s_value), xytext=(0, 5), textcoords="offset points", fontsize=6.5, ha="center")
                all_x.extend(px.tolist())
                all_y.extend(np.concatenate([py, low[np.isfinite(low)], high[np.isfinite(high)]]).tolist())
                if len(px):
                    endpoints.append((float(px[-1]), float(py[-1]), number))
                curve_map[number] = f"{group_name}, средняя и 95% ДИ, n={group['test_id'].nunique()}"
                all_x.extend(
                    _draw_group_failure_event(
                        ax,
                        group,
                        x,
                        curve_number=number,
                        label_level=(number - 1) % 3,
                    )
                )

        _reasonable_limits(ax, all_x, all_y, fixed_axes=fixed_axes)
        y_limits = ax.get_ylim()
        _label_curve_numbers(ax, endpoints, abs(float(y_limits[0] - y_limits[1])))
        caption = "Кривые нагрузки–осадки. " + "; ".join(
            f"{number} — {description}" for number, description in curve_map.items()
        )
        caption += ". Точки — измерения; линии между ними — ломаные без spline-сглаживания."
        fig.canvas.draw()
        return PlotOutput(fig, caption, warnings, curve_map)


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
    selected_test = str(frame["test_id"].iloc[0])
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
