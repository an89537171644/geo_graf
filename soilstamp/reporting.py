"""Human-readable Russian reporting and reproducibility bundle export."""

from __future__ import annotations

import io
import hashlib
import json
import math
import platform
import sys
import zipfile
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any

import matplotlib
import numpy
import pandas
import scipy

from .data import AuditTrail
from .indicators import (
    indicator_aggregation_frame,
    indicator_audit_frame,
    indicator_event_frame,
    indicator_passport_frame,
)
from .provenance import (
    canonical_json_bytes,
    effective_conversion_parameters,
    passport_completeness,
)
from .schema import PCRResult, ProvenanceRecord, VERSION


def decimals_for_resolution(resolution: float | None, default: int = 2) -> int:
    if resolution is None or not math.isfinite(float(resolution)) or float(resolution) <= 0:
        return default
    exponent = Decimal(str(float(resolution))).normalize().as_tuple().exponent
    return max(0, -int(exponent))


def format_ru(
    value: float | int | None,
    *,
    resolution: float | None = None,
    uncertainty: float | None = None,
    unit: str = "",
) -> str:
    if value is None:
        return "—"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(numeric):
        return "—"
    effective = resolution
    if uncertainty is not None and math.isfinite(float(uncertainty)) and float(uncertainty) > 0:
        # Report no more digits than the uncertainty supports.
        effective = max(float(resolution or 0.0), float(uncertainty))
    decimals = decimals_for_resolution(effective)
    if resolution is not None and math.isfinite(float(resolution)) and float(resolution) > 0:
        step = Decimal(str(float(resolution)))
        rounded = (Decimal(str(numeric)) / step).quantize(Decimal("1"), rounding=ROUND_HALF_UP) * step
        numeric = float(rounded)
    text = f"{numeric:.{decimals}f}".replace(".", ",")
    return f"{text} {unit}".rstrip()


def software_versions() -> dict[str, str]:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": numpy.__version__,
        "pandas": pandas.__version__,
        "scipy": scipy.__version__,
        "matplotlib": matplotlib.__version__,
        "soil_stamp_antonov": VERSION,
    }


def _modulus_text(value: Any, fallback: str = "—") -> str:
    """Render a methodology field without exposing pandas missing values."""

    if value is None:
        return fallback
    try:
        if pandas.isna(value):
            return fallback
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    return text or fallback


def _modulus_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return bool(value) if math.isfinite(float(value)) else False
    return str(value).strip().casefold() in {"1", "true", "yes", "да"}


def _finite_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _modulus_resolution_mpa(row: pandas.Series) -> float:
    """Choose MPa rounding from the reported confidence interval.

    The uncertainty is shown with one significant digit, or two when its
    leading digit is one or two.  The same decimal place is then used for the
    estimate and both confidence limits.
    """

    estimate = _finite_float(row.get("E_stamp_app_kPa"))
    low = _finite_float(row.get("ci_low_kPa"))
    high = _finite_float(row.get("ci_high_kPa"))
    if estimate is None or low is None or high is None:
        return 0.01
    uncertainty = max(abs(estimate - low), abs(high - estimate)) / 1000.0
    if not math.isfinite(uncertainty) or uncertainty <= 0:
        return 0.01
    exponent = math.floor(math.log10(uncertainty))
    leading = uncertainty / (10**exponent)
    significant_digits = 2 if leading < 3 else 1
    return float(10 ** (exponent - significant_digits + 1))


def _modulus_used_rows(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return json.dumps(list(value), ensure_ascii=False)
    text = _modulus_text(value)
    return text if text != "—" else "не указаны"


def _modulus_profile(row: pandas.Series) -> str:
    profile_id = _modulus_text(row.get("profile_id"), "diagnostic_unapproved_v1")
    profile_version = _modulus_text(row.get("profile_version"), "legacy")
    return f"{profile_id}@{profile_version}"


def _indicator_tables_for_scope(
    prepared,
    test_ids: list[str] | None = None,
) -> tuple[
    pandas.DataFrame,
    pandas.DataFrame,
    pandas.DataFrame,
    pandas.DataFrame,
]:
    """Return calibration artefacts limited to the report's test scope."""

    tables = (
        indicator_audit_frame(prepared),
        indicator_event_frame(prepared),
        indicator_passport_frame(prepared),
        indicator_aggregation_frame(prepared),
    )
    if test_ids is None:
        return tables
    selected = {str(value) for value in test_ids}
    scoped: list[pandas.DataFrame] = []
    for table in tables:
        if not table.empty and "test_id" in table:
            table = table[table["test_id"].astype(str).isin(selected)].copy()
        scoped.append(table)
    return scoped[0], scoped[1], scoped[2], scoped[3]


def build_markdown_report(
    *,
    metadata: dict[str, Any],
    prepared,
    validation_issues: list[Any],
    failures,
    pcr_results: dict[str, PCRResult] | None = None,
    moduli=None,
    group_comparisons: list[Any] | None = None,
    figure_caption: str | None = None,
    plot_warnings: list[str] | None = None,
    audit: AuditTrail | None = None,
    provenance: ProvenanceRecord | dict[str, Any] | None = None,
    passport_status: dict[str, Any] | None = None,
    import_info: dict[str, Any] | None = None,
    source_test_ids: list[str] | None = None,
    source_row_count: int | None = None,
) -> str:
    load_resolution = float(metadata.get("load_resolution_kN", 0.01))
    pressure_resolution_values = pandas.to_numeric(
        prepared.get("pressure_resolution_kPa"), errors="coerce"
    ).dropna()
    pressure_resolution = (
        float(pressure_resolution_values.median())
        if len(pressure_resolution_values) and float(pressure_resolution_values.median()) > 0
        else 0.1
    )
    diameters = pandas.to_numeric(prepared.get("D_mm"), errors="coerce").dropna().unique()
    diameter_text = (
        format_ru(float(diameters[0]), resolution=1, unit="мм")
        if len(diameters) == 1
        else ", ".join(format_ru(float(value), resolution=1, unit="мм") for value in sorted(diameters))
        if len(diameters) > 1
        else "—"
    )
    selected_test_ids = prepared["test_id"].dropna().astype(str).unique().tolist()
    all_source_ids = [str(value) for value in (source_test_ids or selected_test_ids)]
    excluded_test_ids = [value for value in all_source_ids if value not in selected_test_ids]
    scope_text = "полный набор" if not excluded_test_ids else "подмножество исходного набора"
    lines = [
        f"# Отчёт Soil Stamp Antonov {VERSION}",
        "",
        "## Набор данных",
        "",
        f"- Испытаний: {prepared['test_id'].nunique()}",
        f"- Область анализа: **{scope_text}**.",
        f"- Включённые test_id: {', '.join(f'`{value}`' for value in selected_test_ids) or '—'}.",
        f"- Исключённые test_id: {', '.join(f'`{value}`' for value in excluded_test_ids) or 'нет'}.",
        f"- Строк исходного протокола: {source_row_count if source_row_count is not None else len(prepared)}",
        f"- Строк в активном prepared-слое: {len(prepared)}",
        f"- Активный слой: `{prepared['correction_mode'].iloc[0] if len(prepared) else '—'}`",
        f"- Диаметр(ы) штампа: {diameter_text}",
        "",
        "## Паспорт проекта",
        "",
    ]
    passport = passport_status or passport_completeness(metadata)
    if passport.get("complete"):
        lines.append("- Обязательные поля паспорта заполнены.")
    else:
        missing = ", ".join(f"`{name}`" for name in passport.get("missing", []))
        lines.append(f"- Паспорт неполон. Не заполнены: {missing or '—'}.")
    for name, value in passport.get("fields", {}).items():
        rendered = json.dumps(value, ensure_ascii=False, default=str) if value is not None else "—"
        lines.append(f"- `{name}`: {rendered}")
    lines.extend(["", "## Преобразование нагрузки/давления", ""])
    for row in effective_conversion_parameters(metadata, selected_test_ids):
        lines.append(
            f"- `{row['test_id']}`: `{row['formula']}`; unit={row['load_unit']}; "
            f"factor={row['load_factor']}; lever={row['lever_ratio']}; "
            f"D={row['stamp_diameter_mm']}; A={row['stamp_area_m2']}."
        )
    lines.extend(["", "## Индикаторные каналы", ""])
    (
        indicator_audit,
        indicator_events,
        indicator_passports,
        indicator_aggregation,
    ) = _indicator_tables_for_scope(prepared, selected_test_ids)
    if indicator_passports.empty:
        raw_channels = [
            name
            for name in ("indicator_1", "indicator_2", "indicator_3", "indicator_4")
            if name in prepared
            and pandas.to_numeric(prepared[name], errors="coerce").notna().any()
        ]
        if raw_channels:
            lines.append(
                "- Эффективные поканальные паспорта не сформированы; исходные показания "
                f"{', '.join(f'`{name}`' for name in raw_channels)} сохранены без скрытого преобразования."
            )
        else:
            lines.append("- Индикаторные каналы в выбранных строках отсутствуют.")
    else:
        lines.append(f"- Эффективных поканальных паспортов: {len(indicator_passports)}.")
        for _, row in indicator_passports.iterrows():
            valid_from = str(row.get("verification_date") or "—")
            valid_until = str(row.get("verification_valid_until") or "—")
            verification_status = str(row.get("verification_status") or "review_required")
            evaluation_date = str(row.get("verification_evaluation_date") or "—")
            evaluation_source = str(
                row.get("verification_evaluation_date_source") or "unknown"
            )
            evaluation_rule = str(
                row.get("verification_evaluation_rule") or "—"
            )
            instrument_id = row.get("instrument_id") or "—"
            serial_number = row.get("serial_number") or "—"
            compatibility = (
                "; режим совместимости с ранее откалиброванной осадкой"
                if bool(row.get("compatibility_mode", False))
                else ""
            )
            lines.append(
                f"- `{row.get('test_id')}` / `{row.get('channel')}`: "
                f"тип={row.get('indicator_type') or '—'}; "
                f"instrument_id={instrument_id}; заводской №={serial_number}; "
                f"mode=`{row.get('mode')}`; диапазон={row.get('range_mm')} мм; "
                f"цена деления={row.get('division_mm')} мм; "
                f"коэффициент={row.get('correction_factor')}; "
                f"координаты=({row.get('x_mm')}, {row.get('y_mm')}) мм; "
                f"назначение=`{row.get('assignment_status') or '—'}`; "
                f"поверка={valid_from}…{valid_until}; статус=`{verification_status}`; "
                f"дата оценки={evaluation_date} ({evaluation_source}); "
                f"правило=`{evaluation_rule}`{compatibility}."
            )
    if indicator_audit.empty:
        lines.append("- Таблица преобразования индикаторов пуста.")
    else:
        statuses = (
            indicator_audit.get("processing_status", pandas.Series(dtype="string"))
            .fillna("unknown")
            .astype(str)
            .value_counts()
            .to_dict()
        )
        flags = (
            indicator_audit.get(
                "quality_flags", pandas.Series("", index=indicator_audit.index)
            )
            .fillna("")
            .astype(str)
        )
        zero_crossings = (
            int(indicator_events["event_type"].eq("zero_crossing").sum())
            if not indicator_events.empty and "event_type" in indicator_events
            else int(flags.str.contains(r"(?:^|;)zero_crossing(?:;|$)", regex=True).sum())
        )
        reverse_points = int(flags.str.contains("reverse_motion", regex=False).sum())
        correction_events = (
            int(indicator_events["event_type"].eq("zero_correction_applied").sum())
            if not indicator_events.empty and "event_type" in indicator_events
            else int(flags.str.contains("zero_correction_applied", regex=False).sum())
        )
        qc_points = int(flags.ne("").sum())
        status_text = ", ".join(f"{name}={count}" for name, count in statuses.items()) or "нет"
        lines.append(
            f"- Таблица преобразования: {len(indicator_audit)} строк; QC-статусы: {status_text}."
        )
        lines.append(
            f"- Журнал: переходов через ноль — {zero_crossings}; точек обратного хода — "
            f"{reverse_points}; коррекций нуля — {correction_events}; точек с QC-флагами — {qc_points}."
        )
    lines.extend(["", "## Агрегация осадки", ""])
    if indicator_aggregation.empty:
        lines.append(
            "- Таблица агрегации пуста: осадка по индикаторным каналам не формировалась."
        )
    else:
        for test_id, part in indicator_aggregation.groupby("test_id", sort=False):
            methods = sorted(
                set(part.get("aggregation_method", pandas.Series(dtype="string")).dropna().astype(str))
            )
            statuses = (
                part.get("aggregation_status", pandas.Series(dtype="string"))
                .fillna("unknown")
                .astype(str)
                .value_counts()
                .to_dict()
            )
            status_text = ", ".join(
                f"{name}={count}" for name, count in statuses.items()
            )
            required = sorted(
                set(part.get("channels_required", pandas.Series(dtype="string")).dropna().astype(str))
            )
            used = sorted(
                set(part.get("channels_used", pandas.Series(dtype="string")).dropna().astype(str))
            )
            missing = sorted(
                set(part.get("missing_channels", pandas.Series(dtype="string")).dropna().astype(str))
            )
            lines.append(
                f"- `{test_id}`: метод={', '.join(f'`{value}`' for value in methods) or '—'}; "
                f"статусы={status_text or '—'}; required={required or ['[]']}; "
                f"used={used or ['[]']}; missing={missing or ['[]']}."
            )
            plane_rank_values = pandas.to_numeric(
                part.get(
                    "plane_rank",
                    pandas.Series(float("nan"), index=part.index),
                ),
                errors="coerce",
            )
            plane = part[plane_rank_values.notna()]
            if not plane.empty:
                ranks = sorted(
                    set(pandas.to_numeric(plane["plane_rank"], errors="coerce").dropna().astype(int))
                )
                residual = pandas.to_numeric(
                    plane.get("plane_residual_rms_mm"), errors="coerce"
                ).dropna()
                tilt = pandas.to_numeric(
                    plane.get("tilt_magnitude_mm_per_mm"), errors="coerce"
                ).dropna()
                lines.append(
                    f"  - plane_center: rank={ranks or '—'}; "
                    f"residual RMS max={float(residual.max()) if len(residual) else '—'} мм; "
                    f"tilt max={float(tilt.max()) if len(tilt) else '—'} мм/мм."
                )
    lines.extend(["", "## Контроль качества", ""])
    if validation_issues:
        for issue in validation_issues:
            payload = issue.to_dict() if hasattr(issue, "to_dict") else dict(issue)
            location = ":".join(
                str(value)
                for value in (payload.get("sheet"), payload.get("row"), payload.get("column"))
                if value not in (None, "")
            )
            details = []
            if payload.get("raw_value") not in (None, ""):
                details.append(f"raw={payload.get('raw_value')!r}")
            if payload.get("suggested_action"):
                details.append(f"действие: {payload.get('suggested_action')}")
            details.append(f"blocks={bool(payload.get('blocks_processing'))}")
            lines.append(
                f"- **{payload.get('level', 'info')} / {payload.get('code', '')}:** "
                f"{payload.get('message', '')}"
                + (f" [{location}]" if location else "")
                + ("; " + "; ".join(details) if details else "")
            )
    else:
        lines.append("- Замечаний в автоматической проверке нет.")
    lines.extend(["", "## Разрушение и цензурирование", ""])
    for _, row in failures.iterrows():
        if bool(row["failure_reached"]):
            if pandas.notna(row["F_last_stable"]) and pandas.notna(row["F_failure_step"]):
                localized_failure = (
                    f"{format_ru(row['F_last_stable'], resolution=load_resolution)} < Fu ≤ "
                    f"{format_ru(row['F_failure_step'], resolution=load_resolution, unit='кН')}"
                )
                capacity_kind = "force"
            elif pandas.notna(row.get("p_last_stable")) and pandas.notna(row.get("p_failure_step")):
                localized_failure = (
                    f"{format_ru(row['p_last_stable'], resolution=pressure_resolution)} < pu ≤ "
                    f"{format_ru(row['p_failure_step'], resolution=pressure_resolution, unit='кПа')}"
                )
                capacity_kind = "pressure"
            else:
                localized_failure = "разрушение зафиксировано; интервал неполон"
                capacity_kind = "unknown"
            lines.append(f"- `{row['test_id']}`: {localized_failure}")
            if capacity_kind == "force":
                lines.append(
                    "  - последняя устойчивая нагрузка: "
                    + format_ru(row["F_last_stable"], resolution=load_resolution, unit="кН")
                )
                lines.append(
                    "  - ступень разрушения: "
                    + format_ru(row["F_failure_step"], resolution=load_resolution, unit="кН")
                )
            elif capacity_kind == "pressure":
                lines.append(
                    "  - последнее устойчивое давление: "
                    + format_ru(row["p_last_stable"], resolution=pressure_resolution, unit="кПа")
                )
                lines.append(
                    "  - давление ступени разрушения: "
                    + format_ru(row["p_failure_step"], resolution=pressure_resolution, unit="кПа")
                )
            if pandas.isna(row["s_failure"]):
                lines.append("  - осадка при разрушении не измерена; фиктивная точка не создавалась.")
        else:
            if pandas.notna(row["Fu_lower"]):
                censor_text = "Fu > " + format_ru(
                    row["Fu_lower"], resolution=load_resolution, unit="кН"
                )
            else:
                censor_text = "pu > " + format_ru(
                    row.get("pu_lower"), resolution=pressure_resolution, unit="кПа"
                )
            lines.append(f"- `{row['test_id']}`: {censor_text} (правое цензурирование)")
    if pcr_results:
        lines.extend(["", "## Начальное критическое давление", ""])
        for test_id, result in pcr_results.items():
            ci = (
                f"; 95% ДИ {format_ru(result.pcr_ci_low, resolution=0.1)}–{format_ru(result.pcr_ci_high, resolution=0.1)} кПа"
                if result.pcr_ci_low is not None and result.pcr_ci_high is not None
                else "; ДИ неустойчив"
            )
            lines.append(
                f"- `{test_id}`: pcr(auto) = {format_ru(result.pcr_auto, resolution=0.1, unit='кПа')}"
                f"{ci}; R²={format_ru(result.r2, resolution=0.001)}; n={result.n}."
            )
            if result.pcr_manual is not None:
                lines.append(
                    f"  - подтверждено вручную: {format_ru(result.pcr_manual, resolution=0.1, unit='кПа')}; "
                    f"причина: {result.manual_reason}. Автоматический результат сохранён."
                )
    if moduli is not None and len(moduli):
        lines.extend(["", "## Условный штамповый модуль", ""])
        for _, row in moduli[moduli["method"].isin(["E_regression", "E_secant"])].iterrows():
            review_status = _modulus_text(row.get("review_status"), "review_required")
            primary = _modulus_bool(row.get("is_primary")) and review_status == "approved"
            result_class = "PRIMARY" if primary else "DIAGNOSTIC"
            resolution_mpa = _modulus_resolution_mpa(row)
            estimate_kpa = _finite_float(row.get("E_stamp_app_kPa"))
            estimate_mpa = estimate_kpa / 1000.0 if estimate_kpa is not None else None
            ci_low_kpa = _finite_float(row.get("ci_low_kPa"))
            ci_high_kpa = _finite_float(row.get("ci_high_kPa"))
            ci_text = ""
            if ci_low_kpa is not None and ci_high_kpa is not None:
                ci_text = (
                    "; 95% ДИ "
                    f"{format_ru(ci_low_kpa / 1000.0, resolution=resolution_mpa)}–"
                    f"{format_ru(ci_high_kpa / 1000.0, resolution=resolution_mpa, unit='МПа')}"
                )
            test_id = _modulus_text(row.get("test_id"), "не указан")
            p_range_source = _modulus_text(row.get("p_range_source"), "не указан")
            p_range_origin = _modulus_text(row.get("p_range_origin"), "не указан")
            profile_source = _modulus_text(row.get("profile_source"), "не указан")
            lines.append(
                f"- `{test_id}` / `{row['method']}` / `{_modulus_profile(row)}`: "
                f"**{result_class}**; review_status=`{review_status}`; "
                "E_stamp_app = "
                f"{format_ru(estimate_mpa, resolution=resolution_mpa, unit='МПа')}"
                f"{ci_text}; "
                f"p={format_ru(row['p_min_kPa'], resolution=0.1)}–{format_ru(row['p_max_kPa'], resolution=0.1)} кПа; "
                f"источник диапазона=`{p_range_source}` (origin=`{p_range_origin}`); "
                f"n={int(row['n'])}; ν={format_ru(row['nu'], resolution=0.01)} "
                f"(source=`{_modulus_text(row.get('nu_source'), 'legacy')}`); "
                f"коэффициент формы={format_ru(row['shape_factor'], resolution=0.01)} "
                f"(source=`{_modulus_text(row.get('shape_factor_source'), 'legacy')}`); "
                f"profile_source=`{profile_source}`; "
                f"использованные строки={_modulus_used_rows(row.get('used_indices'))}."
            )
            requested_min = _finite_float(row.get("requested_p_min_kPa"))
            requested_max = _finite_float(row.get("requested_p_max_kPa"))
            if requested_min is not None or requested_max is not None:
                lines.append(
                    "  - запрошенный диапазон: "
                    f"{format_ru(requested_min, resolution=0.1)}–"
                    f"{format_ru(requested_max, resolution=0.1, unit='кПа')}."
                )
            methodology_note = _modulus_text(row.get("methodology_note"), "не указано")
            lines.append(f"  - методическое примечание: {methodology_note}")

    if group_comparisons:
        lines.extend(["", "## Сравнение групп", ""])
        rendered_comparisons = 0
        for comparison in group_comparisons:
            if not isinstance(comparison, pandas.DataFrame) or comparison.empty:
                continue
            row = comparison.iloc[0]
            baseline_group = row.get("baseline_group", "—")
            reinforced_group = row.get("reinforced_group", "—")
            pairing_status = row.get("pairing_status", "unknown")
            pairing_reason = row.get("pairing_reason", "")
            pairing_warning = row.get("pairing_warning", "")
            lines.append(
                f"- `{baseline_group}` vs `{reinforced_group}`: "
                f"pairing_status=`{pairing_status}`; pairing_reason=`{pairing_reason or '—'}`."
            )
            if pandas.notna(pairing_warning) and str(pairing_warning).strip():
                lines.append(f"  - Предупреждение: {str(pairing_warning).strip()}")
            rendered_comparisons += 1
        if rendered_comparisons == 0:
            lines.append("- Таблицы сравнения групп отсутствуют.")
    if plot_warnings:
        lines.extend(["", "## Предупреждения графика", ""])
        lines.extend(f"- {warning}" for warning in plot_warnings)
    if figure_caption:
        lines.extend(["", "## Подпись рисунка", "", figure_caption])
    lines.extend(
        [
            "",
            "## Воспроизводимость",
            "",
            "- Исходные строки не сортировались по нагрузке и не сглаживались spline.",
            "- Интерполяция групповых кривых выполнялась только внутри диапазона испытания.",
            "- Доверительные интервалы указаны вместе с методом и seed в таблицах результатов.",
            f"- Записей audit trail: {len(audit.events) if audit else 0}.",
            "- Версии ПО: `" + json.dumps(software_versions(), ensure_ascii=False) + "`",
            "",
            "> E_stamp_app является условным/эквивалентным штамповым модулем. "
            "Нормативная формула и коэффициент формы должны быть зафиксированы в методике проекта.",
        ]
    )
    if import_info:
        lines.extend(
            [
                "",
                "### Импорт",
                "",
                f"- Формат: `{import_info.get('format', '—')}`.",
                f"- Режим: `{import_info.get('import_mode', '—')}`.",
                f"- Строк: {import_info.get('rows', '—')}.",
                f"- Листы/mapping: `{json.dumps(import_info.get('sheets', []), ensure_ascii=False, default=str)}`.",
                f"- Raw cells: {import_info.get('raw_cell_count', 0)}.",
            ]
        )
    if provenance:
        payload = provenance.to_dict() if hasattr(provenance, "to_dict") else dict(provenance)
        lines.extend(
            [
                "",
                "### Provenance",
                "",
                f"- SHA-256 исходного файла: `{payload.get('input_file_sha256')}`.",
                f"- SHA-256 metadata: `{payload.get('metadata_sha256')}`.",
                f"- SHA-256 config: `{payload.get('config_sha256')}`.",
                f"- Версия алгоритма: `{payload.get('program_version')}`.",
                f"- Git commit: `{payload.get('git_commit') or 'нет первого commit'}`.",
                f"- Git dirty: `{payload.get('git_dirty')}`.",
                f"- SHA-256 дерева исходников: `{payload.get('source_tree_sha256') or '—'}`.",
                f"- Время обработки UTC: `{payload.get('processing_timestamp_utc')}`.",
            ]
        )
        evaluations = payload.get("metrology_evaluations") or []
        lines.append(f"- Оценок срока поверки: {len(evaluations)}.")
        for evaluation in evaluations:
            if not isinstance(evaluation, dict):
                continue
            lines.append(
                f"  - `{evaluation.get('test_id')}` / `{evaluation.get('channel')}`: "
                f"status=`{evaluation.get('verification_status')}`; "
                f"evaluation_date={evaluation.get('verification_evaluation_date') or '—'}; "
                f"source=`{evaluation.get('verification_evaluation_date_source') or 'unknown'}`; "
                f"rule=`{evaluation.get('verification_evaluation_rule') or '—'}`."
            )
    return "\n".join(lines)


def reproducibility_bundle(
    *,
    raw,
    prepared,
    metadata: dict[str, Any],
    audit: AuditTrail,
    report_markdown: str,
    result_tables: dict[str, Any] | None = None,
    figures: dict[str, bytes] | None = None,
    run_parameters: dict[str, Any] | None = None,
    provenance: ProvenanceRecord | dict[str, Any] | None = None,
    raw_cells=None,
    import_issues: list[Any] | None = None,
    source_file_name: str | None = None,
    source_file_bytes: bytes | None = None,
    metadata_file_name: str | None = None,
    metadata_file_bytes: bytes | None = None,
    config_snapshot: dict[str, Any] | None = None,
    scope: dict[str, Any] | None = None,
) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        manifest: list[dict[str, Any]] = []

        def safe_component(value: Any, fallback: str) -> str:
            basename = Path(str(value)).name
            cleaned = "".join(
                character if character.isalnum() or character in {"-", "_", "."} else "_"
                for character in basename
            ).strip(".")
            return cleaned or fallback

        def write(name: str, payload: bytes | str) -> None:
            content = payload.encode("utf-8") if isinstance(payload, str) else payload
            archive.writestr(name, content)
            manifest.append(
                {"path": name, "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}
            )

        write("data/raw_protocol.csv", raw.to_csv(index=False).encode("utf-8-sig"))
        write("data/prepared_machine.csv", prepared.to_csv(index=False).encode("utf-8"))
        if raw_cells is not None and hasattr(raw_cells, "to_csv") and len(raw_cells):
            write("data/raw_cells.csv", raw_cells.to_csv(index=False).encode("utf-8-sig"))
        if source_file_bytes is not None:
            safe_name = safe_component(source_file_name or "source_input.bin", "source_input.bin")
            write(f"source/{safe_name}", source_file_bytes)
        if metadata_file_bytes is not None:
            safe_metadata_name = safe_component(
                metadata_file_name or "metadata_original.json", "metadata_original.json"
            )
            write(f"source/{safe_metadata_name}", metadata_file_bytes)
        if config_snapshot is not None:
            write("config/processing_config.canonical.json", canonical_json_bytes(config_snapshot))
        write("metadata.json", json.dumps(metadata, ensure_ascii=False, indent=2, default=str))
        write("audit.json", audit.to_json())
        write("report_ru.md", report_markdown.encode("utf-8"))
        if provenance:
            payload = provenance.to_dict() if hasattr(provenance, "to_dict") else dict(provenance)
            write("provenance.json", json.dumps(payload, ensure_ascii=False, indent=2, default=str))
        if import_issues:
            issue_payload = [item.to_dict() if hasattr(item, "to_dict") else dict(item) for item in import_issues]
            write("validation/issues.json", json.dumps(issue_payload, ensure_ascii=False, indent=2, default=str))
        write(
            "analysis_run.json",
            json.dumps(
                {"parameters": run_parameters or {}, "software": software_versions()},
                ensure_ascii=False,
                indent=2,
                default=str,
            ),
        )
        effective_result_tables = dict(result_tables or {})
        (
            indicator_audit,
            indicator_events,
            indicator_passports,
            indicator_aggregation,
        ) = _indicator_tables_for_scope(prepared)
        effective_result_tables.setdefault("indicator_processing_audit", indicator_audit)
        effective_result_tables.setdefault("indicator_processing_events", indicator_events)
        effective_result_tables.setdefault(
            "indicator_calibration_parameters", indicator_passports
        )
        effective_result_tables.setdefault(
            "indicator_aggregation_results", indicator_aggregation
        )
        for name, table in effective_result_tables.items():
            safe_result_name = safe_component(name, "result")
            if hasattr(table, "to_csv"):
                write(
                    f"results/{safe_result_name}.csv",
                    table.to_csv(index=False).encode("utf-8"),
                )
            else:
                write(
                    f"results/{safe_result_name}.json",
                    json.dumps(table, ensure_ascii=False, indent=2, default=str),
                )
        for name, payload in (figures or {}).items():
            write(f"figures/{safe_component(name, 'figure.bin')}", payload)
        archive.writestr(
            "manifest.json",
            json.dumps(
                {"scope": scope or {}, "files": manifest},
                ensure_ascii=False,
                indent=2,
            ).encode("utf-8"),
        )
    return buffer.getvalue()
