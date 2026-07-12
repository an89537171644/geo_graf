"""Scientific calculations for plate-load curves.

The module contains no UI state and does not mutate input frames.  This makes
analysis runs deterministic for a fixed random seed and suitable for tests.
"""

from __future__ import annotations

import math
from dataclasses import replace
from datetime import datetime, timezone
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize_scalar

from .data import AuditTrail, _stable_status_mask
from .methodology import (
    ModulusOverrides,
    ModulusResolution,
    legacy_modulus_resolution,
    resolve_modulus_method,
)
from .schema import ModulusResult, PCRResult


EPS = np.finfo(float).eps


def _finite_loading_points(
    frame: pd.DataFrame,
    *,
    pressure: str = "p_kPa",
    settlement: str = "settlement_mm",
    branches: Iterable[str] = ("loading", "hold"),
) -> pd.DataFrame:
    mask = np.isfinite(pd.to_numeric(frame[pressure], errors="coerce"))
    mask &= np.isfinite(pd.to_numeric(frame[settlement], errors="coerce"))
    if "is_failure" in frame:
        mask &= ~frame["is_failure"].fillna(False).astype(bool)
    mask &= _stable_status_mask(frame)
    if "branch" in frame:
        branch_tuple = tuple(branches)
        mask &= frame["branch"].isin(branch_tuple)
        if set(branch_tuple).issubset({"loading", "hold"}):
            first_cycle = pd.Series(False, index=frame.index)
            for _, part in frame.groupby("test_id", sort=False):
                ordered = part.sort_values("sequence_no", kind="stable") if "sequence_no" in part else part
                stop_positions = np.flatnonzero(
                    ordered["branch"].isin(["unloading", "reloading"]).to_numpy()
                )
                stop = int(stop_positions[0]) if len(stop_positions) else len(ordered)
                first_cycle.loc[ordered.index[:stop]] = True
            mask &= first_cycle
    points = frame.loc[mask].copy()
    points[pressure] = pd.to_numeric(points[pressure], errors="coerce")
    points[settlement] = pd.to_numeric(points[settlement], errors="coerce")
    return points


def _deduplicate_pressure(
    points: pd.DataFrame, pressure: str, settlement: str
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Use the last accepted reading at each repeated pressure level."""

    if points.empty:
        return np.array([]), np.array([]), []
    columns = [pressure, settlement]
    if "sequence_no" in points:
        columns.append("sequence_no")
    work = points[columns].copy()
    work["_index"] = points.index
    if "sequence_no" in work:
        work = work.sort_values("sequence_no", kind="stable")
    grouped = (
        work.groupby(pressure, sort=True, as_index=False)
        .agg({settlement: "last", "_index": "last"})
        .sort_values(pressure, kind="stable")
    )
    return (
        grouped[pressure].to_numpy(dtype=float),
        grouped[settlement].to_numpy(dtype=float),
        grouped["_index"].astype(int).tolist(),
    )


def _hinge_fit_at(p: np.ndarray, s: np.ndarray, pcr: float) -> dict[str, Any]:
    x = np.column_stack([np.ones_like(p), p, np.maximum(0.0, p - pcr)])
    coefficients, _, rank, _ = np.linalg.lstsq(x, s, rcond=None)
    fitted = x @ coefficients
    residuals = s - fitted
    rss = float(residuals @ residuals)
    return {
        "coefficients": coefficients,
        "fitted": fitted,
        "residuals": residuals,
        "rss": max(rss, np.finfo(float).tiny),
        "rank": int(rank),
    }


def _fit_segmented_core(
    p: np.ndarray, s: np.ndarray, *, min_side: int = 3
) -> dict[str, Any]:
    n = len(p)
    if n < 2 * min_side:
        raise ValueError(f"Для pcr нужно не менее {2 * min_side} различных уровней давления.")
    order = np.argsort(p, kind="stable")
    p = p[order]
    s = s[order]
    if np.any(np.diff(p) <= 0):
        raise ValueError("Для pcr нужны различные возрастающие уровни давления.")
    low = float(p[min_side - 1])
    high = float(p[n - min_side])
    if not low < high:
        raise ValueError("Недостаточный диапазон кандидатов pcr.")

    grid = np.linspace(low, high, max(161, 20 * n))
    rss_grid = np.array([_hinge_fit_at(p, s, candidate)["rss"] for candidate in grid])
    best_index = int(np.argmin(rss_grid))
    left = grid[max(0, best_index - 2)]
    right = grid[min(len(grid) - 1, best_index + 2)]
    optimized = minimize_scalar(
        lambda candidate: _hinge_fit_at(p, s, float(candidate))["rss"],
        bounds=(float(left), float(right)),
        method="bounded",
        options={"xatol": max((high - low) * 1e-10, 1e-10)},
    )
    pcr = float(optimized.x if optimized.success else grid[best_index])
    fit = _hinge_fit_at(p, s, pcr)
    total = float(np.sum((s - np.mean(s)) ** 2))
    r2 = 1.0 - fit["rss"] / total if total > EPS else float("nan")
    k = 4  # a, b1, b2 and the profiled breakpoint
    aic = n * math.log(fit["rss"] / n) + 2 * k
    bic = n * math.log(fit["rss"] / n) + k * math.log(n)
    linear_x = np.column_stack([np.ones_like(p), p])
    linear_coef, *_ = np.linalg.lstsq(linear_x, s, rcond=None)
    linear_residual = s - linear_x @ linear_coef
    linear_rss = max(float(linear_residual @ linear_residual), np.finfo(float).tiny)
    linear_bic = n * math.log(linear_rss / n) + 2 * math.log(n)
    hinge_delta = float(fit["coefficients"][2])
    slope_scale = max(
        abs(float(fit["coefficients"][1])),
        abs(float(fit["coefficients"][1] + hinge_delta)),
        1e-12,
    )
    at_boundary = bool(
        np.isclose(pcr, low, rtol=0, atol=(high - low) * 1e-3)
        or np.isclose(pcr, high, rtol=0, atol=(high - low) * 1e-3)
    )
    identifiable = bool(
        not at_boundary
        and (linear_bic - bic) >= 2.0
        and float(fit["coefficients"][1]) > 0
        and float(fit["coefficients"][1] + hinge_delta) > 0
        and hinge_delta >= 0.05 * slope_scale
    )
    return {
        **fit,
        "p": p,
        "s": s,
        "pcr": pcr,
        "r2": r2,
        "aic": float(aic),
        "bic": float(bic),
        "at_boundary": at_boundary,
        "linear_bic": float(linear_bic),
        "bic_improvement_over_linear": float(linear_bic - bic),
        "identifiable": identifiable,
    }


def _fit_independent_two_line(p: np.ndarray, s: np.ndarray, min_side: int) -> dict[str, Any]:
    n = len(p)
    best: dict[str, Any] | None = None
    for split in range(min_side, n - min_side + 1):
        left_x = np.column_stack([np.ones(split), p[:split]])
        right_x = np.column_stack([np.ones(n - split), p[split:]])
        left_coef, *_ = np.linalg.lstsq(left_x, s[:split], rcond=None)
        right_coef, *_ = np.linalg.lstsq(right_x, s[split:], rcond=None)
        fitted = np.concatenate([left_x @ left_coef, right_x @ right_coef])
        rss = max(float(np.sum((s - fitted) ** 2)), np.finfo(float).tiny)
        k = 5  # two intercepts, two slopes and split
        candidate = {
            "method": "independent_two_line_bic",
            "split_index": split,
            "pcr": float((p[split - 1] + p[split]) / 2),
            "left_intercept": float(left_coef[0]),
            "left_slope": float(left_coef[1]),
            "right_intercept": float(right_coef[0]),
            "right_slope": float(right_coef[1]),
            "rss": rss,
            "aic": float(n * math.log(rss / n) + 2 * k),
            "bic": float(n * math.log(rss / n) + k * math.log(n)),
            "fitted": fitted.tolist(),
        }
        if best is None or candidate["bic"] < best["bic"]:
            best = candidate
    if best is None:
        raise ValueError("Не удалось построить независимую двухлинейную модель.")
    return best


def fit_segmented_pcr(
    frame: pd.DataFrame,
    *,
    pressure: str = "p_kPa",
    settlement: str = "settlement_mm",
    min_side: int = 3,
    bootstrap: int = 500,
    seed: int = 202604,
) -> PCRResult:
    """Fit the continuous hinge model and residual-bootstrap its breakpoint."""

    points = _finite_loading_points(frame, pressure=pressure, settlement=settlement)
    p, s, used_indices = _deduplicate_pressure(points, pressure, settlement)
    core = _fit_segmented_core(p, s, min_side=min_side)
    if not core["identifiable"]:
        raise ValueError(
            "Излом pcr не идентифицируется: сегментированная модель не улучшает "
            "линейную по BIC, решение находится на границе либо податливость после излома не возрастает."
        )
    rng = np.random.default_rng(seed)
    residuals = core["residuals"] - np.mean(core["residuals"])
    pcr_boot: list[float] = []
    # Wild bootstrap preserves the pressure design and tolerates mild
    # heteroscedasticity better than resampling individual protocol rows.
    for _ in range(max(0, int(bootstrap))):
        signs = rng.choice(np.array([-1.0, 1.0]), size=len(residuals))
        simulated = core["fitted"] + residuals * signs
        try:
            fitted_boot = _fit_segmented_core(p, simulated, min_side=min_side)
        except (ValueError, np.linalg.LinAlgError):
            continue
        if fitted_boot["identifiable"]:
            pcr_boot.append(float(fitted_boot["pcr"]))
    if len(pcr_boot) >= max(20, int(0.2 * max(bootstrap, 1))):
        ci_low, ci_high = np.quantile(pcr_boot, [0.025, 0.975]).tolist()
    else:
        ci_low = ci_high = None
    coefficients = core["coefficients"]
    alternative = _fit_independent_two_line(core["p"], core["s"], min_side)
    alternative["continuous_at_boundary"] = core["at_boundary"]
    alternative["linear_bic"] = core["linear_bic"]
    alternative["bic_improvement_over_linear"] = core["bic_improvement_over_linear"]
    alternative["bootstrap_type"] = "wild_residual_rademacher"
    alternative["bootstrap_seed"] = seed
    alternative["p_min_used_kPa"] = float(np.min(core["p"]))
    alternative["p_max_used_kPa"] = float(np.max(core["p"]))
    return PCRResult(
        method="continuous_segmented_hinge",
        pcr_auto=float(core["pcr"]),
        pcr_ci_low=float(ci_low) if ci_low is not None else None,
        pcr_ci_high=float(ci_high) if ci_high is not None else None,
        intercept=float(coefficients[0]),
        slope_before=float(coefficients[1]),
        slope_after=float(coefficients[1] + coefficients[2]),
        hinge_delta=float(coefficients[2]),
        r2=float(core["r2"]),
        aic=float(core["aic"]),
        bic=float(core["bic"]),
        n=len(p),
        used_indices=used_indices,
        fitted=core["fitted"].tolist(),
        residuals=core["residuals"].tolist(),
        bootstrap_valid=len(pcr_boot),
        alternative=alternative,
    )


def confirm_manual_pcr(
    result: PCRResult,
    value_kpa: float,
    *,
    reason: str,
    audit: AuditTrail,
    scope: str,
    user: str = "local-user",
) -> PCRResult:
    value = float(value_kpa)
    if not np.isfinite(value):
        raise ValueError("Ручное pcr должно быть конечным числом.")
    lower = float((result.alternative or {}).get("p_min_used_kPa", 0.0))
    upper = float((result.alternative or {}).get("p_max_used_kPa", np.inf))
    if value < max(0.0, lower) or value > upper:
        raise ValueError(
            f"Ручное pcr должно находиться в испытанном диапазоне {max(0.0, lower):g}–{upper:g} кПа."
        )
    if not str(reason).strip():
        raise ValueError("Для ручного подтверждения pcr требуется непустое обоснование.")
    confirmed_at = datetime.now(timezone.utc).isoformat()
    updated = replace(
        result,
        pcr_manual=value,
        manual_reason=str(reason).strip(),
        manual_author=str(user).strip() or "local-user",
        manual_confirmed_at_utc=confirmed_at,
    )
    audit.record(
        "confirm_manual_pcr",
        scope=scope,
        reason=reason,
        parameters={"pcr_auto_kPa": result.pcr_auto, "pcr_manual_kPa": value},
        before=result.to_dict(),
        after=updated.to_dict(),
        user=user,
        method="manual_confirmation",
    )
    return updated


def _modulus_from_slope(
    slope_mm_per_kpa: float, diameter_mm: float, nu: float, shape_factor: float
) -> float:
    if slope_mm_per_kpa <= 0 or not np.isfinite(slope_mm_per_kpa):
        return float("nan")
    slope_m_per_kpa = slope_mm_per_kpa / 1000.0
    diameter_m = diameter_mm / 1000.0
    return (1.0 - nu**2) * shape_factor * diameter_m / slope_m_per_kpa


def estimate_moduli(
    frame: pd.DataFrame,
    *,
    p_min_kpa: float | None = None,
    p_max_kpa: float | None = None,
    nu: float | None = None,
    shape_factor: float | None = None,
    resolution: ModulusResolution | None = None,
    tangent_window: int = 3,
    bootstrap: int = 500,
    seed: int = 202604,
) -> pd.DataFrame:
    """Calculate apparent moduli under a resolved methodology contract.

    Calls without ``resolution`` remain numerically compatible, but their
    rows are explicitly marked ``diagnostic_unapproved_v1`` and can never be
    primary results.
    """

    points = _finite_loading_points(frame)
    p, s, source_indices = _deduplicate_pressure(points, "p_kPa", "settlement_mm")
    if len(p) < 2:
        raise ValueError("Недостаточно точек для расчета модуля.")
    available_range = (float(np.min(p)), float(np.max(p)))
    if resolution is None:
        resolution = legacy_modulus_resolution(
            p_min_kpa=p_min_kpa,
            p_max_kpa=p_max_kpa,
            nu=nu,
            shape_factor=shape_factor,
            available_p_range=available_range,
        )
    elif any(value is not None for value in (p_min_kpa, p_max_kpa, nu, shape_factor)):
        raise ValueError(
            "Нельзя смешивать resolution с legacy-параметрами p_min/p_max/nu/shape_factor."
        )
    nu = resolution.nu
    shape_factor = resolution.shape_factor
    if not 0 <= nu < 0.5:
        raise ValueError("Коэффициент Пуассона должен быть в диапазоне [0; 0,5).")
    if shape_factor <= 0:
        raise ValueError("Коэффициент формы должен быть положительным.")
    lower = float(
        available_range[0] if resolution.p_min_kpa is None else resolution.p_min_kpa
    )
    upper = float(
        available_range[1] if resolution.p_max_kpa is None else resolution.p_max_kpa
    )
    selected = (p >= lower) & (p <= upper)
    p_sel, s_sel = p[selected], s[selected]
    selected_indices = np.asarray(source_indices, dtype=int)[selected].astype(int).tolist()
    if len(p_sel) < 2:
        raise ValueError("В выбранном диапазоне нужно не менее двух точек.")
    d_values = pd.to_numeric(points.get("D_mm", pd.Series(dtype=float)), errors="coerce").dropna()
    if d_values.empty:
        raise ValueError("Для E_stamp_app требуется диаметр D.")
    diameter = float(d_values.iloc[0])
    if not np.isfinite(diameter) or diameter <= 0:
        raise ValueError("Для E_stamp_app диаметр D должен быть конечным и положительным.")
    if not np.allclose(d_values.to_numpy(), diameter, rtol=1e-6, atol=1e-6):
        raise ValueError("E_stamp_app нельзя объединять для разных диаметров штампа.")

    def method_contract(
        indices: list[int],
        *,
        primary_method: bool = False,
        calculation_valid: bool = True,
        calculation_note: str = "",
    ) -> dict[str, Any]:
        row_is_primary = bool(primary_method and resolution.is_primary and calculation_valid)
        row_review_status = resolution.review_status
        methodology_note = resolution.methodology_note
        if primary_method and not calculation_valid:
            row_review_status = "review_required"
            methodology_note = f"{methodology_note} {calculation_note}".strip()
        return {
            "profile_id": resolution.profile_id,
            "profile_version": resolution.profile_version,
            "is_primary": row_is_primary,
            "review_status": row_review_status,
            "p_range_source": resolution.p_range_source,
            "nu_source": resolution.nu_source,
            "shape_factor_source": resolution.shape_factor_source,
            "used_indices": [int(index) for index in indices],
            "methodology_note": methodology_note,
            "profile_source": resolution.profile_source,
            "p_range_origin": resolution.p_range_origin,
            "requested_p_min_kPa": (
                resolution.p_min_kpa
                if resolution.p_range_source != "diagnostic_full_curve"
                else None
            ),
            "requested_p_max_kPa": (
                resolution.p_max_kpa
                if resolution.p_range_source != "diagnostic_full_curve"
                else None
            ),
        }

    rows: list[ModulusResult] = []
    x = np.column_stack([np.ones(len(p_sel)), p_sel])
    coef, *_ = np.linalg.lstsq(x, s_sel, rcond=None)
    fitted = x @ coef
    residuals = s_sel - fitted
    rss = float(residuals @ residuals)
    total = float(np.sum((s_sel - np.mean(s_sel)) ** 2))
    r2 = 1.0 - rss / total if total > EPS else float("nan")
    slope = float(coef[1])
    e_reg = _modulus_from_slope(slope, diameter, nu, shape_factor)
    settlement_scale = max(float(np.max(np.abs(s_sel))), 1.0)
    settlement_has_variation = bool(
        float(np.ptp(s_sel)) > np.finfo(float).eps * settlement_scale
    )
    regression_valid = bool(
        settlement_has_variation and np.isfinite(slope) and slope > 0 and np.isfinite(e_reg) and e_reg > 0
    )
    regression_calculation_note = (
        "Регрессионный E понижен до diagnostic: наклон/деформация не дают "
        "конечного положительного модуля."
        if not regression_valid
        else ""
    )
    rng = np.random.default_rng(seed)
    bootstrap_e: list[float] = []
    centered = residuals - np.mean(residuals)
    for _ in range(max(0, int(bootstrap))):
        simulated = fitted + rng.choice(centered, size=len(centered), replace=True)
        boot_coef, *_ = np.linalg.lstsq(x, simulated, rcond=None)
        value = _modulus_from_slope(float(boot_coef[1]), diameter, nu, shape_factor)
        if np.isfinite(value):
            bootstrap_e.append(value)
    if len(p_sel) >= 3 and rss > EPS and len(bootstrap_e) >= 20:
        e_low, e_high = np.quantile(bootstrap_e, [0.025, 0.975]).tolist()
    else:
        e_low = e_high = None
    rows.append(
        ModulusResult(
            method="E_regression",
            E_stamp_app_kPa=e_reg,
            p_min_kPa=float(p_sel[0]),
            p_max_kPa=float(p_sel[-1]),
            n=len(p_sel),
            r2=float(r2),
            ci_low_kPa=float(e_low) if e_low is not None else None,
            ci_high_kPa=float(e_high) if e_high is not None else None,
            nu=nu,
            shape_factor=shape_factor,
            slope_m_per_kPa=slope / 1000.0,
            note=(
                "Основной условный показатель; ДИ — residual bootstrap."
                if resolution.is_primary and regression_valid
                else "Диагностический показатель; ДИ — residual bootstrap."
            ),
            **method_contract(
                selected_indices,
                primary_method=True,
                calculation_valid=regression_valid,
                calculation_note=regression_calculation_note,
            ),
        )
    )
    delta_p = float(p_sel[-1] - p_sel[0])
    delta_s = float(s_sel[-1] - s_sel[0])
    secant_slope = delta_s / delta_p if abs(delta_p) > EPS else float("nan")
    rows.append(
        ModulusResult(
            method="E_secant",
            E_stamp_app_kPa=_modulus_from_slope(secant_slope, diameter, nu, shape_factor),
            p_min_kPa=float(p_sel[0]),
            p_max_kPa=float(p_sel[-1]),
            n=2,
            r2=None,
            ci_low_kPa=None,
            ci_high_kPa=None,
            nu=nu,
            shape_factor=shape_factor,
            slope_m_per_kPa=secant_slope / 1000.0,
            note="Секущая по границам выбранного диапазона; R² неприменим.",
            **method_contract([selected_indices[0], selected_indices[-1]]),
        )
    )

    window = max(3, int(tangent_window))
    if window % 2 == 0:
        window += 1
    half = window // 2
    for i in range(len(p_sel)):
        start = max(0, i - half)
        stop = min(len(p_sel), i + half + 1)
        if stop - start < 3:
            if start == 0:
                stop = min(len(p_sel), 3)
            else:
                start = max(0, len(p_sel) - 3)
        local_p, local_s = p_sel[start:stop], s_sel[start:stop]
        if len(local_p) < 3:
            continue
        local_x = np.column_stack([np.ones(len(local_p)), local_p])
        local_coef, *_ = np.linalg.lstsq(local_x, local_s, rcond=None)
        local_fit = local_x @ local_coef
        local_total = float(np.sum((local_s - np.mean(local_s)) ** 2))
        local_rss = float(np.sum((local_s - local_fit) ** 2))
        local_r2 = 1.0 - local_rss / local_total if local_total > EPS else float("nan")
        local_slope = float(local_coef[1])
        rows.append(
            ModulusResult(
                method=f"E_tangent@{p_sel[i]:g}",
                E_stamp_app_kPa=_modulus_from_slope(local_slope, diameter, nu, shape_factor),
                p_min_kPa=float(local_p[0]),
                p_max_kPa=float(local_p[-1]),
                n=len(local_p),
                r2=float(local_r2),
                ci_low_kPa=None,
                ci_high_kPa=None,
                nu=nu,
                shape_factor=shape_factor,
                slope_m_per_kPa=local_slope / 1000.0,
                note="Локальная линейная регрессия.",
                **method_contract(selected_indices[start:stop]),
            )
        )
    for i in range(1, len(p_sel)):
        dp = float(p_sel[i] - p_sel[i - 1])
        ds = float(s_sel[i] - s_sel[i - 1])
        inc_slope = ds / dp if abs(dp) > EPS else float("nan")
        rows.append(
            ModulusResult(
                method=f"E_incremental_diagnostic#{i}",
                E_stamp_app_kPa=_modulus_from_slope(inc_slope, diameter, nu, shape_factor),
                p_min_kPa=float(p_sel[i - 1]),
                p_max_kPa=float(p_sel[i]),
                n=2,
                r2=None,
                ci_low_kPa=None,
                ci_high_kPa=None,
                nu=nu,
                shape_factor=shape_factor,
                slope_m_per_kPa=inc_slope / 1000.0,
                note="Только диагностический соседний инкремент; не основной результат.",
                **method_contract(selected_indices[i - 1 : i + 1]),
            )
        )
    table = pd.DataFrame([row.to_dict() for row in rows])
    resolved_contract = resolution.to_dict()
    if resolution.is_primary and not regression_valid:
        resolved_contract["is_primary"] = False
        resolved_contract["review_status"] = "review_required"
        resolved_contract["methodology_note"] = (
            f"{resolution.methodology_note} {regression_calculation_note}".strip()
        )
    table.attrs["modulus_resolution"] = resolved_contract
    return table


def calculate_moduli_for_test(
    frame: pd.DataFrame,
    metadata: dict[str, Any] | None,
    test_id: str,
    *,
    overrides: ModulusOverrides | dict[str, Any] | None = None,
    manual_confirmation: ModulusOverrides | dict[str, Any] | None = None,
    pcr_result: PCRResult | None = None,
    tangent_window: int = 3,
    bootstrap: int = 500,
    seed: int = 202604,
) -> pd.DataFrame:
    """Resolve and calculate one test through the API shared by CLI and GUI."""

    scoped = frame
    if "test_id" in frame:
        scoped = frame[frame["test_id"].astype(str) == str(test_id)]
        if scoped.empty:
            raise ValueError(f"Испытание {test_id} не найдено для расчёта модуля.")
    points = _finite_loading_points(scoped)
    p, _, _ = _deduplicate_pressure(points, "p_kPa", "settlement_mm")
    if len(p) < 2:
        raise ValueError("Недостаточно точек для расчета модуля.")
    resolution = resolve_modulus_method(
        metadata,
        str(test_id),
        overrides=overrides,
        manual_confirmation=manual_confirmation,
        pcr_result=pcr_result,
        available_p_range=(float(np.min(p)), float(np.max(p))),
    )
    return estimate_moduli(
        scoped,
        resolution=resolution,
        tangent_window=tangent_window,
        bootstrap=bootstrap,
        seed=seed,
    )


def modulus_sensitivity(
    slope_mm_per_kpa: float,
    diameter_mm: float,
    *,
    nu_values: Iterable[float] = (0.20, 0.25, 0.30, 0.35, 0.40, 0.45),
    shape_factors: Iterable[float] = (0.8, 1.0, 1.2),
) -> pd.DataFrame:
    rows = []
    for nu in nu_values:
        for factor in shape_factors:
            rows.append(
                {
                    "nu": float(nu),
                    "shape_factor": float(factor),
                    "E_stamp_app_kPa": _modulus_from_slope(
                        float(slope_mm_per_kpa), float(diameter_mm), float(nu), float(factor)
                    ),
                }
            )
    return pd.DataFrame(rows)


def _test_curves(
    frame: pd.DataFrame,
    *,
    pressure: str = "p_kPa",
    settlement: str = "settlement_mm",
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    curves: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for test_id, part in frame.groupby("test_id", sort=False):
        points = _finite_loading_points(part, pressure=pressure, settlement=settlement)
        p, s, _ = _deduplicate_pressure(points, pressure, settlement)
        if len(p) >= 1:
            curves[str(test_id)] = (p, s)
    return curves


def _common_union_grid(curves: dict[str, tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    if not curves:
        return np.array([])
    levels = np.unique(np.concatenate([p for p, _ in curves.values()]))
    return levels.astype(float)


def _curve_matrix(
    curves: dict[str, tuple[np.ndarray, np.ndarray]], grid: np.ndarray, tolerance: float
) -> tuple[list[str], np.ndarray, np.ndarray]:
    ids = list(curves)
    values = np.full((len(ids), len(grid)), np.nan)
    measured = np.zeros((len(ids), len(grid)), dtype=bool)
    for row, test_id in enumerate(ids):
        p, s = curves[test_id]
        inside = (grid >= p.min() - tolerance) & (grid <= p.max() + tolerance)
        values[row, inside] = np.interp(grid[inside], p, s)
        for column, level in enumerate(grid):
            hits = np.flatnonzero(np.isclose(p, level, rtol=0.0, atol=tolerance))
            if len(hits):
                values[row, column] = float(np.mean(s[hits]))
                measured[row, column] = True
    return ids, values, measured


def _column_nanmean(values: np.ndarray) -> np.ndarray:
    counts = np.sum(np.isfinite(values), axis=0)
    return np.divide(
        np.nansum(values, axis=0),
        counts,
        out=np.full(values.shape[1], np.nan, dtype=float),
        where=counts > 0,
    )


def group_mean_curve(
    frame: pd.DataFrame,
    *,
    confidence: float = 0.95,
    bootstrap: int = 1000,
    seed: int = 202604,
    pressure_tolerance_kpa: float = 1e-8,
) -> pd.DataFrame:
    """Mean curve on the union of real levels, with exact t and max-t bands."""

    curves = _test_curves(frame)
    grid = _common_union_grid(curves)
    if len(grid) == 0:
        raise ValueError("У повторностей нет общей области давления.")
    ids, matrix, measured = _curve_matrix(curves, grid, pressure_tolerance_kpa)
    n = np.sum(np.isfinite(matrix), axis=0)
    mean = _column_nanmean(matrix)
    sd = np.array(
        [np.nanstd(matrix[:, j], ddof=1) if n[j] >= 2 else np.nan for j in range(len(grid))]
    )
    se = sd / np.sqrt(n)
    alpha = 1.0 - confidence
    critical = np.array(
        [stats.t.ppf(1.0 - alpha / 2.0, int(count - 1)) if count >= 2 else np.nan for count in n]
    )
    t_low = mean - critical * se
    t_high = mean + critical * se

    rng = np.random.default_rng(seed)
    boot_means: list[np.ndarray] = []
    if len(ids) >= 2:
        for _ in range(max(0, int(bootstrap))):
            selected = rng.integers(0, len(ids), size=len(ids))
            boot_means.append(_column_nanmean(matrix[selected]))
    if boot_means:
        boot = np.vstack(boot_means)
        point_low = np.nanquantile(boot, alpha / 2.0, axis=0)
        point_high = np.nanquantile(boot, 1.0 - alpha / 2.0, axis=0)
        scale = np.where(np.isfinite(se) & (se > EPS), se, np.nan)
        valid_scale = np.isfinite(scale)
        if valid_scale.any():
            standardized = np.abs((boot[:, valid_scale] - mean[valid_scale]) / scale[valid_scale])
            valid_rows = np.isfinite(standardized).any(axis=1)
            maxima = np.nanmax(standardized[valid_rows], axis=1) if valid_rows.any() else np.array([])
            maxima = maxima[np.isfinite(maxima)]
            q = float(np.quantile(maxima, confidence)) if len(maxima) else np.nan
            sim_low = mean - q * se
            sim_high = mean + q * se
        else:
            # Identical curves: the empirical band collapses to the mean.
            sim_low = mean.copy()
            sim_high = mean.copy()
    else:
        point_low = point_high = sim_low = sim_high = np.full(len(grid), np.nan)
    insufficient = n < 2
    t_low[insufficient] = np.nan
    t_high[insufficient] = np.nan
    point_low[insufficient] = np.nan
    point_high[insufficient] = np.nan
    sim_low[insufficient] = np.nan
    sim_high[insufficient] = np.nan
    measured_n = measured.sum(axis=0)
    return pd.DataFrame(
        {
            "p_kPa": grid,
            "mean_settlement_mm": mean,
            "t_ci_low_mm": t_low,
            "t_ci_high_mm": t_high,
            "bootstrap_ci_low_mm": point_low,
            "bootstrap_ci_high_mm": point_high,
            "simultaneous_low_mm": sim_low,
            "simultaneous_high_mm": sim_high,
            "n": n.astype(int),
            "measured_n": measured_n.astype(int),
            "interpolated_n": (n - measured_n).astype(int),
            "all_measured": measured_n == n,
            "descriptive_small_n": n < 5,
            "confidence": confidence,
            "bootstrap_seed": seed,
        }
    )


def _benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    adjusted = np.full_like(p_values, np.nan, dtype=float)
    finite = np.isfinite(p_values)
    if not finite.any():
        return adjusted
    values = p_values[finite]
    order = np.argsort(values)
    ranked = values[order]
    m = len(ranked)
    corrected = ranked * m / np.arange(1, m + 1)
    corrected = np.minimum.accumulate(corrected[::-1])[::-1]
    restored = np.empty(m, dtype=float)
    restored[order] = np.clip(corrected, 0.0, 1.0)
    adjusted[finite] = restored
    return adjusted


def _bootstrap_interval(samples: list[np.ndarray], columns: int) -> tuple[np.ndarray, np.ndarray]:
    low = np.full(columns, np.nan)
    high = np.full(columns, np.nan)
    if not samples:
        return low, high
    matrix = np.vstack(samples)
    for column in range(columns):
        finite = matrix[:, column][np.isfinite(matrix[:, column])]
        if len(finite) >= 20:
            low[column], high[column] = np.quantile(finite, [0.025, 0.975])
    return low, high


def _pair_mapping(group_frame: pd.DataFrame) -> dict[str, str]:
    if "pair_id" not in group_frame:
        return {}
    unique = group_frame[["test_id", "pair_id"]].dropna().drop_duplicates()
    mapping: dict[str, str] = {}
    duplicated: set[str] = set()
    for _, row in unique.iterrows():
        pair = str(row["pair_id"])
        test_id = str(row["test_id"])
        if pair in mapping and mapping[pair] != test_id:
            duplicated.add(pair)
        else:
            mapping[pair] = test_id
    for pair in duplicated:
        mapping.pop(pair, None)
    return mapping


def compare_groups(
    frame: pd.DataFrame,
    baseline_group: str,
    reinforced_group: str,
    *,
    bootstrap: int = 1000,
    seed: int = 202604,
) -> pd.DataFrame:
    """Compare groups with paired bootstrap/sign permutation when pairs exist."""

    if "group" not in frame:
        raise ValueError("Для сравнения нужен столбец group.")
    baseline = frame[frame["group"].astype(str) == str(baseline_group)]
    reinforced = frame[frame["group"].astype(str) == str(reinforced_group)]
    base_curves = _test_curves(baseline)
    reinf_curves = _test_curves(reinforced)
    if not base_curves or not reinf_curves:
        raise ValueError("В каждой сравниваемой группе нужна хотя бы одна кривая.")
    base_pairs = _pair_mapping(baseline)
    reinf_pairs = _pair_mapping(reinforced)
    common_pairs = [pair for pair in base_pairs if pair in reinf_pairs]
    paired = bool(common_pairs) and (
        len(common_pairs) == len(base_curves) == len(reinf_curves)
        and len(base_pairs) == len(base_curves)
        and len(reinf_pairs) == len(reinf_curves)
    )
    if paired:
        base_curves_used = {
            pair: base_curves[base_pairs[pair]]
            for pair in common_pairs
            if base_pairs[pair] in base_curves and reinf_pairs[pair] in reinf_curves
        }
        common_pairs = list(base_curves_used)
        reinf_curves_used = {pair: reinf_curves[reinf_pairs[pair]] for pair in common_pairs}
        if not common_pairs:
            paired = False
    if not paired:
        base_curves_used = base_curves
        reinf_curves_used = reinf_curves
    all_curves = {
        **{f"b:{k}": v for k, v in base_curves_used.items()},
        **{f"r:{k}": v for k, v in reinf_curves_used.items()},
    }
    grid = _common_union_grid(all_curves)
    if len(grid) == 0:
        raise ValueError("У групп нет общей области давления.")
    _, base_matrix, _ = _curve_matrix(base_curves_used, grid, 1e-8)
    _, reinf_matrix, _ = _curve_matrix(reinf_curves_used, grid, 1e-8)
    shared_support = np.isfinite(base_matrix).any(axis=0) & np.isfinite(reinf_matrix).any(axis=0)
    grid = grid[shared_support]
    base_matrix = base_matrix[:, shared_support]
    reinf_matrix = reinf_matrix[:, shared_support]
    if len(grid) == 0:
        raise ValueError("У групп нет общей области давления.")
    base_mean = _column_nanmean(base_matrix)
    reinf_mean = _column_nanmean(reinf_matrix)
    n_base_at = np.sum(np.isfinite(base_matrix), axis=0).astype(int)
    n_reinf_at = np.sum(np.isfinite(reinf_matrix), axis=0).astype(int)
    n_pairs_at = (
        np.sum(np.isfinite(base_matrix) & np.isfinite(reinf_matrix), axis=0).astype(int)
        if paired
        else np.zeros(len(grid), dtype=int)
    )
    delta = base_mean - reinf_mean
    ratio = np.divide(
        reinf_mean,
        base_mean,
        out=np.full_like(base_mean, np.nan),
        where=np.abs(base_mean) > EPS,
    )
    reduction = np.divide(
        delta * 100.0,
        base_mean,
        out=np.full_like(base_mean, np.nan),
        where=np.abs(base_mean) > EPS,
    )
    rng = np.random.default_rng(seed)
    boot_ratio: list[np.ndarray] = []
    boot_delta: list[np.ndarray] = []
    boot_effect: list[np.ndarray] = []
    for _ in range(max(0, int(bootstrap))):
        if paired:
            selected = rng.integers(0, len(base_matrix), len(base_matrix))
            b_sample = base_matrix[selected]
            r_sample = reinf_matrix[selected]
            b = _column_nanmean(b_sample)
            r = _column_nanmean(r_sample)
            difference_sample = b_sample - r_sample
            effect_vector = np.full(len(grid), np.nan)
            for column in range(len(grid)):
                differences = difference_sample[:, column]
                differences = differences[np.isfinite(differences)]
                if len(differences) >= 2 and np.std(differences, ddof=1) > EPS:
                    effect_vector[column] = np.mean(differences) / np.std(differences, ddof=1)
            boot_effect.append(effect_vector)
        else:
            b_sample = base_matrix[rng.integers(0, len(base_matrix), len(base_matrix))]
            r_sample = reinf_matrix[rng.integers(0, len(reinf_matrix), len(reinf_matrix))]
            b = _column_nanmean(b_sample)
            r = _column_nanmean(r_sample)
            effect_vector = np.full(len(grid), np.nan)
            for column in range(len(grid)):
                base_values = b_sample[:, column][np.isfinite(b_sample[:, column])]
                reinf_values = r_sample[:, column][np.isfinite(r_sample[:, column])]
                degrees_boot = len(base_values) + len(reinf_values) - 2
                if len(base_values) >= 2 and len(reinf_values) >= 2 and degrees_boot > 1:
                    pooled = math.sqrt(
                        ((len(base_values) - 1) * np.var(base_values, ddof=1)
                        + (len(reinf_values) - 1) * np.var(reinf_values, ddof=1))
                        / degrees_boot
                    )
                    if pooled > EPS:
                        correction = 1.0 - 3.0 / (4.0 * degrees_boot - 1.0)
                        effect_vector[column] = (
                            (np.mean(base_values) - np.mean(reinf_values)) / pooled * correction
                        )
            boot_effect.append(effect_vector)
        boot_delta.append(b - r)
        boot_ratio.append(np.divide(r, b, out=np.full_like(b, np.nan), where=np.abs(b) > EPS))
    ratio_low, ratio_high = _bootstrap_interval(boot_ratio, len(grid))
    delta_low, delta_high = _bootstrap_interval(boot_delta, len(grid))
    effect_low, effect_high = _bootstrap_interval(boot_effect, len(grid))

    permutations = min(max(int(bootstrap), 1000), 10000)
    observed = np.abs(delta)
    permutation_p = np.full(len(grid), np.nan)
    effect_size = np.full(len(grid), np.nan)
    if paired:
        pair_differences = base_matrix - reinf_matrix
        for column in range(len(grid)):
            differences = pair_differences[:, column]
            differences = differences[np.isfinite(differences)]
            n_pairs = len(differences)
            if n_pairs < 2:
                continue
            pair_sd = np.std(differences, ddof=1)
            if pair_sd > EPS:
                effect_size[column] = np.mean(differences) / pair_sd
            if n_pairs <= 15:
                combinations = np.arange(2**n_pairs, dtype=np.uint32)[:, None]
                bits = (combinations >> np.arange(n_pairs, dtype=np.uint32)) & 1
                signs = np.where(bits == 0, -1.0, 1.0)
                permuted = np.abs(signs @ differences / n_pairs)
                permutation_p[column] = np.mean(permuted >= observed[column])
            else:
                signs = rng.choice(np.array([-1.0, 1.0]), size=(permutations, n_pairs))
                permuted = np.abs(signs @ differences / n_pairs)
                permutation_p[column] = (
                    np.sum(permuted >= observed[column]) + 1.0
                ) / (len(permuted) + 1.0)
        effect_name = "Cohen_dz_paired"
    else:
        for column in range(len(grid)):
            base_values = base_matrix[:, column][np.isfinite(base_matrix[:, column])]
            reinf_values = reinf_matrix[:, column][np.isfinite(reinf_matrix[:, column])]
            if len(base_values) < 2 or len(reinf_values) < 2:
                continue
            degrees = len(base_values) + len(reinf_values) - 2
            pooled = math.sqrt(
                ((len(base_values) - 1) * np.var(base_values, ddof=1)
                + (len(reinf_values) - 1) * np.var(reinf_values, ddof=1))
                / degrees
            )
            if pooled > EPS and degrees > 1:
                correction = 1.0 - 3.0 / (4.0 * degrees - 1.0)
                effect_size[column] = (
                    (np.mean(base_values) - np.mean(reinf_values)) / pooled * correction
                )
            combined = np.concatenate([base_values, reinf_values])
            permuted_values = np.empty(permutations)
            for permutation in range(permutations):
                shuffled = rng.permutation(combined)
                permuted_values[permutation] = abs(
                    np.mean(shuffled[: len(base_values)])
                    - np.mean(shuffled[len(base_values) :])
                )
            permutation_p[column] = (
                np.sum(permuted_values >= observed[column]) + 1.0
            ) / (permutations + 1.0)
        effect_name = "Hedges_g_independent"
    insufficient = (n_pairs_at < 2) if paired else ((n_base_at < 2) | (n_reinf_at < 2))
    for array in (ratio_low, ratio_high, delta_low, delta_high, effect_low, effect_high):
        array[insufficient] = np.nan
    permutation_p[insufficient] = np.nan
    effect_size[insufficient] = np.nan
    permutation_fdr = _benjamini_hochberg(permutation_p)
    return pd.DataFrame(
        {
            "p_kPa": grid,
            "s_baseline_mm": base_mean,
            "s_reinforced_mm": reinf_mean,
            "k_s": ratio,
            "k_s_ci_low": ratio_low,
            "k_s_ci_high": ratio_high,
            "delta_s_mm": delta,
            "delta_s_ci_low_mm": delta_low,
            "delta_s_ci_high_mm": delta_high,
            "settlement_reduction_percent": reduction,
            "n_baseline": n_base_at,
            "n_reinforced": n_reinf_at,
            "n_pairs": n_pairs_at,
            "analysis_design": "paired" if paired else "independent",
            "effect_size": effect_size,
            "effect_size_name": effect_name,
            "effect_size_ci_low": effect_low,
            "effect_size_ci_high": effect_high,
            "permutation_p": permutation_p,
            "permutation_p_fdr_bh": permutation_fdr,
            "descriptive_small_n": (
                n_pairs_at < 5
                if paired
                else np.minimum(n_base_at, n_reinf_at) < 5
            ),
            "bootstrap_seed": seed,
        }
    )


def derivative_diagnostics(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for test_id, part in frame.groupby("test_id", sort=False):
        points = part.sort_values("sequence_no", kind="stable") if "sequence_no" in part else part
        for i in range(1, len(points)):
            previous, current = points.iloc[i - 1], points.iloc[i]
            previous_branch = str(previous.get("branch", ""))
            current_branch = str(current.get("branch", ""))
            if (
                previous_branch != current_branch
                or current_branch not in {"loading", "reloading"}
                or not _stable_status_mask(points.iloc[[i - 1, i]]).all()
                or pd.isna(previous.get("p_kPa"))
                or pd.isna(current.get("p_kPa"))
                or pd.isna(previous.get("settlement_mm"))
                or pd.isna(current.get("settlement_mm"))
            ):
                continue
            dp = float(current["p_kPa"] - previous["p_kPa"])
            ds = float(current["settlement_mm"] - previous["settlement_mm"])
            rows.append(
                {
                    "test_id": str(test_id),
                    "from_sequence": int(previous.get("sequence_no", i - 1)),
                    "to_sequence": int(current.get("sequence_no", i)),
                    "p_mid_kPa": float((current["p_kPa"] + previous["p_kPa"]) / 2),
                    "delta_s_mm": ds,
                    "ds_dp_mm_per_kPa": ds / dp if abs(dp) > EPS else np.nan,
                    "dp_ds_kPa_per_mm": dp / ds if abs(ds) > EPS else np.nan,
                }
            )
    return pd.DataFrame(rows)


def value_at_pressure(frame: pd.DataFrame, pressure_kpa: float) -> pd.DataFrame:
    rows = []
    target = float(pressure_kpa)
    for test_id, (p, s) in _test_curves(frame).items():
        if p.min() <= target <= p.max():
            rows.append({"test_id": test_id, "p_kPa": target, "settlement_mm": float(np.interp(target, p, s))})
    return pd.DataFrame(rows)


def pressure_at_settlement(frame: pd.DataFrame, settlement_mm: float) -> pd.DataFrame:
    rows = []
    target = float(settlement_mm)
    for test_id, (p, s) in _test_curves(frame).items():
        # Use only monotone loading curves; multiple crossings are reported.
        crossings: list[float] = []
        for i in range(1, len(p)):
            lo, hi = sorted((s[i - 1], s[i]))
            if lo <= target <= hi and not np.isclose(s[i], s[i - 1]):
                fraction = (target - s[i - 1]) / (s[i] - s[i - 1])
                crossings.append(float(p[i - 1] + fraction * (p[i] - p[i - 1])))
        for crossing_index, value in enumerate(crossings, 1):
            rows.append(
                {
                    "test_id": test_id,
                    "settlement_mm": target,
                    "p_kPa": value,
                    "crossing": crossing_index,
                }
            )
    return pd.DataFrame(rows)


def deformation_work(frame: pd.DataFrame) -> pd.DataFrame:
    """Integrate p ds in protocol order; result unit kJ/m² (kPa·m)."""

    rows = []
    for test_id, part in frame.groupby("test_id", sort=False):
        if "sequence_no" in part:
            part = part.sort_values("sequence_no", kind="stable")
        total = 0.0
        by_branch: dict[str, float] = {}
        integrated_segments = 0
        skipped_gaps = 0
        for i in range(1, len(part)):
            previous, current = part.iloc[i - 1], part.iloc[i]
            if any(
                pd.isna(value)
                for value in (
                    previous.get("p_kPa"),
                    previous.get("settlement_mm"),
                    current.get("p_kPa"),
                    current.get("settlement_mm"),
                )
            ):
                skipped_gaps += 1
                continue
            ds_m = float(current["settlement_mm"] - previous["settlement_mm"]) / 1000.0
            mean_p = float(current["p_kPa"] + previous["p_kPa"]) / 2.0
            value = mean_p * ds_m
            total += value
            integrated_segments += 1
            branch = str(current.get("branch", "unknown"))
            by_branch[branch] = by_branch.get(branch, 0.0) + value
        row = {
            "test_id": str(test_id),
            "W_total_kJ_m2": total,
            "integrated_segments": integrated_segments,
            "skipped_gaps": skipped_gaps,
        }
        row.update({f"W_{key}_kJ_m2": value for key, value in by_branch.items()})
        rows.append(row)
    return pd.DataFrame(rows)


def hysteresis_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for test_id, part in frame.groupby("test_id", sort=False):
        if "sequence_no" in part:
            part = part.sort_values("sequence_no", kind="stable")
        unloading_positions = np.flatnonzero(part["branch"].eq("unloading").to_numpy())
        if not len(unloading_positions):
            continue
        first = int(unloading_positions[0])
        before = part.iloc[:first]
        peak_candidates = before[before["settlement_mm"].notna() & before["p_kPa"].notna()]
        unload_segment = part.iloc[first:]
        reloading_positions = np.flatnonzero(unload_segment["branch"].eq("reloading").to_numpy())
        if len(reloading_positions):
            unload_segment = unload_segment.iloc[: int(reloading_positions[0])]
        unload = unload_segment[
            unload_segment["settlement_mm"].notna() & unload_segment["p_kPa"].notna()
        ]
        if peak_candidates.empty or unload.empty:
            continue
        peak = peak_candidates.iloc[-1]
        min_pressure = float(unload["p_kPa"].min())
        residual = unload[np.isclose(unload["p_kPa"], min_pressure, rtol=0, atol=1e-9)].iloc[-1]
        s_peak = float(peak["settlement_mm"])
        s_residual = float(residual["settlement_mm"])
        start_candidates = part.iloc[: first + 1]
        start_candidates = start_candidates[
            start_candidates["p_kPa"].notna() & start_candidates["settlement_mm"].notna()
        ]
        start_pressure = float(start_candidates.iloc[0]["p_kPa"]) if not start_candidates.empty else np.nan
        pressure_span = float(peak["p_kPa"] - start_pressure) if np.isfinite(start_pressure) else np.nan
        closure_tolerance = max(abs(pressure_span) * 0.01, 1e-6) if np.isfinite(pressure_span) else 1e-6
        loop_closed = bool(
            np.isfinite(start_pressure)
            and abs(float(residual["p_kPa"]) - start_pressure) <= closure_tolerance
        )
        first_valid_positions = np.flatnonzero(
            (part["p_kPa"].notna() & part["settlement_mm"].notna()).to_numpy()
        )
        cycle_work = np.nan
        if len(first_valid_positions):
            start_position = int(first_valid_positions[0])
            cycle = pd.concat([part.iloc[start_position:first], unload_segment])
            cycle_work = 0.0
            integrated = 0
            for position in range(1, len(cycle)):
                previous, current = cycle.iloc[position - 1], cycle.iloc[position]
                if any(
                    pd.isna(value)
                    for value in (
                        previous.get("p_kPa"),
                        previous.get("settlement_mm"),
                        current.get("p_kPa"),
                        current.get("settlement_mm"),
                    )
                ):
                    continue
                cycle_work += (
                    float(previous["p_kPa"] + current["p_kPa"])
                    / 2.0
                    * float(current["settlement_mm"] - previous["settlement_mm"])
                    / 1000.0
                )
                integrated += 1
            if integrated == 0:
                cycle_work = np.nan
        rows.append(
            {
                "test_id": str(test_id),
                "s_peak_mm": s_peak,
                "s_residual_mm": s_residual,
                "s_recoverable_mm": s_peak - s_residual,
                "residual_pressure_kPa": float(residual["p_kPa"]),
                "hysteresis_energy_kJ_m2": (
                    abs(cycle_work)
                    if loop_closed and np.isfinite(cycle_work)
                    else np.nan
                ),
                "loop_closed": loop_closed,
                "note": (
                    "Контур замкнут в пределах 1% диапазона давления."
                    if loop_closed
                    else "Контур не замкнут; энергия гистерезиса не выдаётся. Остаточная осадка дана при измеренном давлении."
                ),
            }
        )
    return pd.DataFrame(rows)


def time_stabilization(
    frame: pd.DataFrame,
    *,
    rate_threshold_mm_per_min: float = 0.01,
    consecutive_intervals: int = 2,
) -> pd.DataFrame:
    if "timestamp" not in frame:
        return pd.DataFrame()
    rows = []
    for (test_id, stage), part in frame.groupby(["test_id", "stage"], sort=False):
        part = part.copy()
        part["_time"] = pd.to_datetime(part["timestamp"], errors="coerce", utc=True)
        part = part[part["_time"].notna() & part["settlement_mm"].notna()].sort_values("_time")
        if len(part) < consecutive_intervals + 1:
            continue
        dt = part["_time"].diff().dt.total_seconds().to_numpy() / 60.0
        ds = part["settlement_mm"].diff().to_numpy(dtype=float)
        rate = np.divide(ds, dt, out=np.full_like(ds, np.nan), where=dt > 0)
        stable = np.abs(rate) <= rate_threshold_mm_per_min
        stabilization_time = None
        for end in range(consecutive_intervals, len(stable)):
            if np.all(stable[end - consecutive_intervals + 1 : end + 1]):
                stabilization_time = (part.iloc[end]["_time"] - part.iloc[0]["_time"]).total_seconds() / 60.0
                break
        rows.append(
            {
                "test_id": str(test_id),
                "stage": stage,
                "stabilization_time_min": stabilization_time,
                "threshold_mm_per_min": rate_threshold_mm_per_min,
                "n": len(part),
            }
        )
    return pd.DataFrame(rows)


def center_and_tilt(
    frame: pd.DataFrame,
    indicator_positions_mm: dict[str, tuple[float, float]] | None,
    *,
    indicator_sign: float = 1.0,
    reference_sign: float = -1.0,
    scale_to_mm: float = 1.0,
    indicator_resolution_mm: float = 0.0,
) -> pd.DataFrame:
    """Fit indicator plane only for rows with explicitly confirmed calibration."""

    if not indicator_positions_mm or len(indicator_positions_mm) < 3:
        return pd.DataFrame()
    raw_columns = [name for name in indicator_positions_mm if name in frame.columns]
    calibrated_channel_mode = any(
        f"{name}_settlement_mm" in frame for name in indicator_positions_mm
    )
    columns = (
        [name for name in indicator_positions_mm if f"{name}_settlement_mm" in frame]
        if calibrated_channel_mode
        else raw_columns
    )
    if len(columns) < 3:
        return pd.DataFrame()
    positions = np.array([indicator_positions_mm[name] for name in columns], dtype=float)
    design = np.column_stack([np.ones(len(columns)), positions])
    span = max(
        (
            float(np.linalg.norm(positions[i] - positions[j]))
            for i in range(len(positions))
            for j in range(i + 1, len(positions))
        ),
        default=0.0,
    )
    rows = []
    for index, row in frame.iterrows():
        if not calibrated_channel_mode and not bool(row.get("indicator_calibration_confirmed", False)):
            continue
        row_indicator_sign = float(row.get("indicator_sign", indicator_sign))
        row_reference_sign = float(row.get("reference_sign", reference_sign))
        row_scale = float(row.get("indicator_scale_to_mm", scale_to_mm))
        row_calibration_factor = float(row.get("indicator_calibration_factor", 1.0))
        calibrated_columns = [f"{name}_settlement_mm" for name in columns]
        if calibrated_channel_mode:
            z = pd.to_numeric(row[calibrated_columns], errors="coerce").to_numpy(dtype=float)
        else:
            z = (
                pd.to_numeric(row[columns], errors="coerce").to_numpy(dtype=float)
                * row_indicator_sign
                * row_scale
                * row_calibration_factor
            )
        if bool(row.get("reference_channel_used", False)) and "reference_indicator" in frame:
            reference_column = (
                "reference_indicator_settlement_mm"
                if "reference_indicator_settlement_mm" in frame
                else "reference_indicator"
            )
            reference = pd.to_numeric(pd.Series([row.get(reference_column)]), errors="coerce").iloc[0]
            if pd.isna(reference):
                continue
            if calibrated_channel_mode and reference_column.endswith("_settlement_mm"):
                z = z + float(reference)
            else:
                z = (
                    z
                    + row_reference_sign
                    * float(reference)
                    * row_scale
                    * row_calibration_factor
                )
        finite = np.isfinite(z)
        if finite.sum() < 3 or np.linalg.matrix_rank(design[finite]) < 3:
            continue
        coefficients, *_ = np.linalg.lstsq(design[finite], z[finite], rcond=None)
        bx, by = float(coefficients[1]), float(coefficients[2])
        magnitude = math.hypot(bx, by)
        row_resolution = float(row.get("indicator_resolution_mm", indicator_resolution_mm))
        direction_resolved = bool(
            span > 0 and magnitude * span >= max(row_resolution, np.finfo(float).eps * span)
        )
        rows.append(
            {
                "source_index": index,
                "test_id": str(row["test_id"]),
                "stage": row["stage"],
                "center_settlement_mm": float(coefficients[0]),
                "tilt_magnitude_mm_per_mm": magnitude,
                "tilt_direction_deg": (
                    (math.degrees(math.atan2(by, bx)) + 360.0) % 360.0
                    if direction_resolved
                    else np.nan
                ),
                "tilt_direction_resolved": direction_resolved,
                "n_indicators": int(finite.sum()),
                "indicator_mode": row.get("indicator_mode"),
                "indicator_unit": row.get("indicator_unit"),
                "indicator_calibration_factor": row_calibration_factor,
            }
        )
    return pd.DataFrame(rows)
