"""Auditable calibration and unwrapping of settlement indicators.

The service is intentionally independent from Antonov plotting and failure
analysis.  It transforms every channel before aggregation and keeps one audit
row per source point.  A scale crossing is accepted only when the configured
physical limits leave one possible revolution number; no value is clamped,
filled, rounded or silently corrected.
"""

from __future__ import annotations

import hashlib
import json
import math
from copy import deepcopy
from dataclasses import asdict
from datetime import date
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .schema import IndicatorPassport, IndicatorProcessingResult, ValidationIssue


INDICATOR_PROCESSING_SCHEMA = "indicator-processing/1.0"
INDICATOR_ALGORITHM_VERSION = "1.0.0"

_MODE_ALIASES = {
    "direct": "increasing",
    "direct_scale": "increasing",
    "direct_wrapped": "increasing_wrapped",
    "direct_scale_wrapped": "increasing_wrapped",
    "reverse": "decreasing",
    "reverse_scale": "decreasing",
    "reverse_wrapped": "decreasing_wrapped",
    "reverse_scale_wrapped": "decreasing_wrapped",
    "increasing": "increasing",
    "increasing_wrapped": "increasing_wrapped",
    "decreasing": "decreasing",
    "decreasing_wrapped": "decreasing_wrapped",
    "direct_displacement": "cumulative_settlement",
    "ready_settlement": "cumulative_settlement",
    "cumulative_settlement": "cumulative_settlement",
}
INDICATOR_MODES = tuple(sorted(set(_MODE_ALIASES.values())))

_UNIT_TO_MM = {
    "mm": 1.0,
    "мм": 1.0,
    "cm": 10.0,
    "см": 10.0,
    "m": 1000.0,
    "м": 1000.0,
}
_PASSPORT_CONTAINERS = ("indicator_passports", "indicator_channels")
_VERTICAL_CHANNELS = tuple(f"indicator_{index}" for index in range(1, 5))


def canonical_indicator_mode(value: Any) -> str:
    return _MODE_ALIASES.get(str(value or "").strip().casefold(), "")


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def _effective_metadata(metadata: dict[str, Any] | None, test_id: str) -> dict[str, Any]:
    if not isinstance(metadata, dict):
        return {}
    base = {key: deepcopy(value) for key, value in metadata.items() if key != "tests"}
    tests = metadata.get("tests")
    if isinstance(tests, dict) and isinstance(tests.get(str(test_id)), dict):
        base = _deep_merge(base, tests[str(test_id)])
    return base


def _as_finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _as_integer(value: Any) -> int | None:
    number = _as_finite(value)
    if number is None or not math.isclose(number, round(number), abs_tol=1e-9):
        return None
    return int(round(number))


def _first(mapping: dict[str, Any], *names: str) -> Any:
    for name in names:
        if name in mapping:
            return mapping[name]
    return None


def _instrument_registry(metadata: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows: list[Any] = []
    direct = metadata.get("instruments")
    if isinstance(direct, dict):
        rows.append(direct)
    elif isinstance(direct, list):
        rows.extend(direct)
    passport = metadata.get("project_passport")
    nested = passport.get("instruments") if isinstance(passport, dict) else None
    if isinstance(nested, dict):
        rows.append(nested)
    elif isinstance(nested, list):
        rows.extend(nested)
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        identifier = str(row.get("instrument_id") or "").strip()
        if identifier:
            result[identifier] = deepcopy(row)
    return result


def _nested_passport_mapping(metadata: dict[str, Any], channel: str) -> tuple[dict[str, Any], str]:
    common = metadata.get("indicator_passport")
    result = deepcopy(common) if isinstance(common, dict) else {}
    source = "metadata.indicator_passport" if result else ""
    for container_name in _PASSPORT_CONTAINERS:
        container = metadata.get(container_name)
        if not isinstance(container, dict):
            continue
        channel_mapping = container.get(channel)
        if isinstance(channel_mapping, dict):
            result = _deep_merge(result, channel_mapping)
            source = f"metadata.{container_name}.{channel}"
    instrument_id = str(result.get("instrument_id") or "").strip()
    registered = _instrument_registry(metadata).get(instrument_id)
    if registered:
        result = _deep_merge(registered, result)
    return result, source


def _legacy_passport_mapping(metadata: dict[str, Any], channel: str) -> tuple[dict[str, Any], str]:
    raw_mode = str(metadata.get("indicator_mode") or "").strip().casefold()
    mode = canonical_indicator_mode(raw_mode)
    if not mode:
        return {}, ""
    # Wrapped/raw modes require a full passport.  The old common calibration
    # remains readable only for already accumulated displacement.
    if mode != "cumulative_settlement":
        return {}, ""
    unit_key = str(metadata.get("indicator_unit") or "").strip().casefold().replace(" ", "")
    unit_scale = _UNIT_TO_MM.get(unit_key)
    factor = _as_finite(metadata.get("indicator_calibration_factor"))
    sign_field = "reference_sign" if channel == "reference_indicator" else "indicator_sign"
    sign = _as_finite(metadata.get(sign_field))
    division = _as_finite(metadata.get("indicator_resolution_mm"))
    instrument_id = str(metadata.get("indicator_instrument_id") or "").strip()
    if not instrument_id:
        registry = _instrument_registry(metadata)
        instrument_id = next(iter(registry), "")
    return (
        {
            "mode": "cumulative_settlement",
            "type": "legacy pre-calibrated displacement",
            "serial_number": instrument_id,
            "instrument_id": instrument_id,
            "range_mm": _as_finite(metadata.get("indicator_range_mm")) or 0.0,
            "division_mm": division,
            "correction_factor": (
                factor * unit_scale
                if factor is not None and unit_scale is not None
                else None
            ),
            "verification_date": "",
            "verification_valid_until": "",
            "initial_reading": None,
            "zero_correction_mm": 0.0,
            "cumulative_sign": sign,
            "compatibility_mode": True,
            "raw_unit": metadata.get("indicator_unit"),
            "channel": channel,
        },
        "metadata.legacy_indicator_*",
    )


def _issue(
    level: str,
    code: str,
    message: str,
    *,
    test_id: str,
    channel: str,
    rows: Iterable[int] = (),
    raw_value: Any = None,
) -> ValidationIssue:
    return ValidationIssue(
        level,
        code,
        message,
        test_id=str(test_id),
        rows=[int(row) for row in rows],
        column=channel,
        raw_value=raw_value,
    )


def resolve_indicator_passport(
    metadata: dict[str, Any] | None,
    test_id: str,
    channel: str,
    *,
    rows: Iterable[int] = (),
) -> tuple[IndicatorPassport | None, list[ValidationIssue]]:
    """Resolve a test/channel passport without scientific defaults."""

    effective = _effective_metadata(metadata, str(test_id))
    mapping, source = _nested_passport_mapping(effective, channel)
    if not mapping:
        mapping, source = _legacy_passport_mapping(effective, channel)
    if not mapping:
        return None, []

    compatibility = bool(mapping.get("compatibility_mode"))
    raw_mode = _first(mapping, "mode", "indicator_mode", "scale_mode")
    mode = canonical_indicator_mode(raw_mode)
    issues: list[ValidationIssue] = []
    if not mode:
        issues.append(
            _issue(
                "error",
                "unsupported_indicator_mode",
                f"Неизвестный режим индикатора {raw_mode!r}.",
                test_id=str(test_id),
                channel=channel,
                rows=rows,
                raw_value=raw_mode,
            )
        )

    indicator_type = str(_first(mapping, "type", "indicator_type", "model") or "").strip()
    serial_number = str(_first(mapping, "serial_number", "factory_number") or "").strip()
    instrument_id = str(mapping.get("instrument_id") or serial_number).strip() or None
    range_mm = _as_finite(_first(mapping, "range_mm", "scale_range_mm", "dial_period_mm"))
    division_mm = _as_finite(_first(mapping, "division_mm", "resolution_mm", "scale_division_mm"))
    factor = _as_finite(_first(mapping, "correction_factor", "calibration_factor"))
    verification_date = str(_first(mapping, "verification_date", "calibration_date") or "").strip()
    valid_until = str(_first(mapping, "verification_valid_until", "valid_until") or "").strip()
    initial = _as_finite(_first(mapping, "initial_reading", "initial_reading_mm"))
    zero_present = any(name in mapping for name in ("zero_correction_mm", "zero_correction"))
    zero = _as_finite(_first(mapping, "zero_correction_mm", "zero_correction"))
    maximum = _as_finite(_first(mapping, "max_increment_mm", "max_step_mm"))
    reverse = _as_finite(mapping.get("reverse_tolerance_mm"))
    travel = _as_finite(_first(mapping, "travel_range_mm", "instrument_travel_mm"))
    initial_turn = _as_integer(mapping.get("initial_turn", 0))
    cumulative_sign = _as_finite(mapping.get("cumulative_sign", 1.0))

    required_text = {
        "indicator_type": indicator_type,
        "serial_number": serial_number,
        "verification_date": verification_date,
        "verification_valid_until": valid_until,
    }
    if not compatibility:
        for field, value in required_text.items():
            if not value:
                issues.append(
                    _issue(
                        "error",
                        "missing_indicator_passport_field",
                        f"В паспорте {channel} не заполнено поле {field}.",
                        test_id=str(test_id),
                        channel=channel,
                        rows=rows,
                    )
                )
    if range_mm is None or (range_mm <= 0 and not compatibility):
        issues.append(
            _issue(
                "error",
                "invalid_indicator_range",
                "Диапазон шкалы должен быть конечным положительным числом.",
                test_id=str(test_id),
                channel=channel,
                rows=rows,
                raw_value=_first(mapping, "range_mm", "scale_range_mm", "dial_period_mm"),
            )
        )
    if division_mm is None or division_mm <= 0:
        issues.append(
            _issue(
                "error",
                "invalid_indicator_division",
                "Цена деления индикатора должна быть конечным положительным числом (например, 0,01 мм).",
                test_id=str(test_id),
                channel=channel,
                rows=rows,
                raw_value=_first(mapping, "division_mm", "resolution_mm", "scale_division_mm"),
            )
        )
    if factor is None or factor <= 0:
        issues.append(
            _issue(
                "error",
                "invalid_indicator_correction_factor",
                "Поправочный коэффициент должен быть конечным положительным числом.",
                test_id=str(test_id),
                channel=channel,
                rows=rows,
                raw_value=_first(mapping, "correction_factor", "calibration_factor"),
            )
        )
    raw_scale_mode = mode in {"increasing", "increasing_wrapped", "decreasing", "decreasing_wrapped"}
    if raw_scale_mode and initial is None:
        issues.append(
            _issue(
                "error",
                "missing_initial_indicator_reading",
                "Для исходной шкалы требуется явное начальное показание.",
                test_id=str(test_id),
                channel=channel,
                rows=rows,
            )
        )
    if (
        raw_scale_mode
        and initial is not None
        and range_mm is not None
        and not (0 <= initial < range_mm)
    ):
        issues.append(
            _issue(
                "error",
                "initial_indicator_reading_out_of_range",
                "Начальное показание должно находиться в диапазоне [0; range_mm).",
                test_id=str(test_id),
                channel=channel,
                rows=rows,
                raw_value=initial,
            )
        )
    if (
        raw_scale_mode
        and initial is not None
        and division_mm is not None
        and not math.isclose(
            initial / division_mm,
            round(initial / division_mm),
            abs_tol=1e-6,
        )
    ):
        issues.append(
            _issue(
                "error",
                "initial_reading_off_scale_division",
                "Начальное показание не соответствует сетке цены деления.",
                test_id=str(test_id),
                channel=channel,
                rows=rows,
                raw_value=initial,
            )
        )
    if not compatibility and (not zero_present or zero is None):
        issues.append(
            _issue(
                "error",
                "missing_zero_correction",
                "Нулевая коррекция должна быть задана явно, включая значение 0,0 мм.",
                test_id=str(test_id),
                channel=channel,
                rows=rows,
            )
        )
    if raw_scale_mode and (maximum is None or maximum <= 0):
        issues.append(
            _issue(
                "error",
                "missing_max_indicator_increment",
                "Для исходной шкалы требуется явное max_increment_mm: без него нельзя отличить переход через ноль от скачка.",
                test_id=str(test_id),
                channel=channel,
                rows=rows,
            )
        )
    if (
        raw_scale_mode
        and maximum is not None
        and factor is not None
        and range_mm is not None
        and (
            not math.isfinite(range_mm * factor)
            or range_mm * factor <= 0
            or not math.isfinite(maximum / factor)
        )
    ):
        issues.append(
            _issue(
                "error",
                "invalid_indicator_numeric_scale",
                "Сочетание range_mm, correction_factor и max_increment_mm выходит за числовой диапазон.",
                test_id=str(test_id),
                channel=channel,
                rows=rows,
            )
        )
    if reverse is not None and reverse < 0:
        issues.append(
            _issue(
                "error",
                "invalid_reverse_tolerance",
                "reverse_tolerance_mm не может быть отрицательным.",
                test_id=str(test_id),
                channel=channel,
                rows=rows,
                raw_value=reverse,
            )
        )
    if travel is not None and travel <= 0:
        issues.append(
            _issue(
                "error",
                "invalid_indicator_travel_range",
                "Полный механический ход должен быть положительным.",
                test_id=str(test_id),
                channel=channel,
                rows=rows,
                raw_value=travel,
            )
        )
    if initial_turn is None:
        issues.append(
            _issue(
                "error",
                "invalid_initial_turn",
                "initial_turn должен быть целым числом.",
                test_id=str(test_id),
                channel=channel,
                rows=rows,
                raw_value=mapping.get("initial_turn"),
            )
        )
    if cumulative_sign is None or not math.isclose(abs(cumulative_sign), 1.0):
        issues.append(
            _issue(
                "error",
                "invalid_indicator_sign",
                "Знак готовой осадки должен быть равен +1 или -1.",
                test_id=str(test_id),
                channel=channel,
                rows=rows,
                raw_value=mapping.get("cumulative_sign"),
            )
        )

    if not compatibility and verification_date and valid_until:
        try:
            verified = date.fromisoformat(verification_date)
            expires = date.fromisoformat(valid_until)
        except ValueError:
            issues.append(
                _issue(
                    "error",
                    "invalid_verification_date",
                    "Даты поверки должны иметь формат YYYY-MM-DD.",
                    test_id=str(test_id),
                    channel=channel,
                    rows=rows,
                )
            )
        else:
            if expires < verified:
                issues.append(
                    _issue(
                        "error",
                        "invalid_verification_period",
                        "Дата окончания поверки раньше даты поверки.",
                        test_id=str(test_id),
                        channel=channel,
                        rows=rows,
                    )
                )
            elif expires < date.today():
                issues.append(
                    _issue(
                        "warning",
                        "expired_indicator_verification",
                        f"Срок поверки {channel} истёк {valid_until}; преобразование сохранено, статус видим в QC.",
                        test_id=str(test_id),
                        channel=channel,
                        rows=rows,
                    )
                )

    if any(item.level == "error" for item in issues):
        return None, issues
    assert mode and division_mm is not None and factor is not None
    effective_reverse = reverse if reverse is not None else division_mm * factor
    passport = IndicatorPassport(
        channel=channel,
        indicator_type=indicator_type,
        serial_number=serial_number,
        range_mm=float(range_mm or 0.0),
        division_mm=float(division_mm),
        correction_factor=float(factor),
        verification_date=verification_date,
        verification_valid_until=valid_until,
        mode=mode,
        initial_reading=initial,
        zero_correction_mm=float(zero or 0.0),
        max_increment_mm=maximum,
        reverse_tolerance_mm=float(effective_reverse),
        travel_range_mm=travel,
        initial_turn=int(initial_turn or 0),
        cumulative_sign=float(cumulative_sign or 1.0),
        instrument_id=instrument_id,
        source_path=source,
        compatibility_mode=compatibility,
    )
    if compatibility:
        issues.append(
            _issue(
                "warning",
                "legacy_shared_indicator_calibration",
                "Использована совместимая общая калибровка старого формата; для нового расчёта рекомендуется поканальный паспорт.",
                test_id=str(test_id),
                channel=channel,
                rows=rows,
            )
        )
    return passport, issues


def _branch_series(part: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    suggested = pd.Series("loading", index=part.index, dtype="object")
    loads = pd.to_numeric(part.get("load", pd.Series(np.nan, index=part.index)), errors="coerce")
    unloaded = False
    previous: float | None = None
    for index, value in loads.items():
        if previous is None or not math.isfinite(float(value)):
            suggested.loc[index] = "loading"
        else:
            delta = float(value) - previous
            if math.isclose(delta, 0.0, abs_tol=1e-12):
                suggested.loc[index] = "hold"
            elif delta < 0:
                suggested.loc[index] = "unloading"
                unloaded = True
            else:
                suggested.loc[index] = "reloading" if unloaded else "loading"
        if math.isfinite(float(value)):
            previous = float(value)
    if "branch" not in part:
        return suggested, pd.Series("load_direction_suggested", index=part.index)
    supplied = part["branch"].astype("string").str.strip().str.casefold()
    valid = supplied.isin(("loading", "hold", "unloading", "reloading"))
    return supplied.where(valid, suggested), pd.Series(
        np.where(valid, "protocol", "load_direction_suggested"), index=part.index
    )


def _stable_event_id(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return "IND-" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]


def _append_event(
    events: list[dict[str, Any]],
    *,
    test_id: str,
    channel: str,
    row_index: int,
    sequence_index: int,
    event_type: str,
    reason: str,
    method: str,
    raw_before: float | None,
    raw_after: float | None,
    turn_before: int | None,
    turn_after: int | None,
    increment_mm: float | None,
    correction_mm: float = 0.0,
    candidates: list[int] | None = None,
) -> str:
    key = {
        "schema": INDICATOR_PROCESSING_SCHEMA,
        "test_id": test_id,
        "channel": channel,
        "row_index": row_index,
        "sequence_index": sequence_index,
        "event_type": event_type,
        "turn_after": turn_after,
    }
    event_id = _stable_event_id(key)
    events.append(
        {
            "event_id": event_id,
            "schema_version": INDICATOR_PROCESSING_SCHEMA,
            "algorithm_version": INDICATOR_ALGORITHM_VERSION,
            "test_id": test_id,
            "channel": channel,
            "row_index": row_index,
            "sequence_index": sequence_index,
            "event_type": event_type,
            "reason": reason,
            "method": method,
            "raw_before": raw_before,
            "raw_after": raw_after,
            "turn_before": turn_before,
            "turn_after": turn_after,
            "increment_mm": increment_mm,
            "correction_mm": correction_mm,
            "candidate_turns": json.dumps(candidates or [], ensure_ascii=False),
        }
    )
    return event_id


def _turn_column(part: pd.DataFrame, channel: str) -> str | None:
    for name in (
        f"{channel}_turn",
        f"{channel}_turn_number",
        f"{channel}_revolution",
        f"{channel}_revolution_number",
    ):
        if name in part:
            return name
    return None


def _expected_direction(branch: str) -> int | None:
    if branch in {"loading", "reloading"}:
        return 1
    if branch == "unloading":
        return -1
    return None


def _audit_base(
    part: pd.DataFrame,
    index: int,
    *,
    test_id: str,
    channel: str,
    passport: IndicatorPassport | None,
    branch: str,
    branch_source: str,
) -> dict[str, Any]:
    row = part.loc[index]
    sequence_value = row.get("sequence_index", row.get("sequence_no", index))
    sequence_number = _as_integer(sequence_value)
    if sequence_number is None:
        sequence_number = int(index)
    return {
        "schema_version": INDICATOR_PROCESSING_SCHEMA,
        "algorithm_version": INDICATOR_ALGORITHM_VERSION,
        "test_id": test_id,
        "channel": channel,
        "sheet_name": row.get("sheet_name"),
        "source_row": row.get("source_row", index),
        "row_index": int(index),
        "sequence_index": sequence_number,
        "branch": branch,
        "branch_source": branch_source,
        "indicator_type": passport.indicator_type if passport else None,
        "serial_number": passport.serial_number if passport else None,
        "instrument_id": passport.instrument_id if passport else None,
        "mode": passport.mode if passport else None,
        "original_reading": row.get(channel),
        "raw_reading": np.nan,
        "turn_number": np.nan,
        "unwrapped_reading": np.nan,
        "computed_increment_mm": np.nan,
        "cumulative_before_correction_mm": np.nan,
        "applied_correction_mm": np.nan,
        "cumulative_settlement_mm": np.nan,
        "reference_correction_mm": 0.0,
        "settlement_effective_mm": np.nan,
        "warning": "",
        "quality_flags": "",
        "processing_status": "unprocessed",
        "conversion_method": "",
        "correction_record_ids": "",
    }


def _process_channel(
    part: pd.DataFrame,
    *,
    test_id: str,
    channel: str,
    passport: IndicatorPassport,
    branches: pd.Series,
    branch_sources: pd.Series,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[ValidationIssue], dict[int, float | None]]:
    rows: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    issues: list[ValidationIssue] = []
    values: dict[int, float | None] = {}
    turn_column = _turn_column(part, channel)
    mode = passport.mode
    wrapped = mode.endswith("_wrapped")
    raw_scale = mode != "cumulative_settlement"
    scale_direction = -1.0 if mode.startswith("decreasing") else 1.0
    previous_raw: float | None = (
        float(passport.initial_reading) if raw_scale and passport.initial_reading is not None else None
    )
    previous_turn = passport.initial_turn
    previous_unwrapped: float | None = (
        previous_raw + previous_turn * passport.range_mm
        if previous_raw is not None
        else None
    )
    previous_ready: float | None = None
    correction_logged = False

    for index in part.index:
        branch = str(branches.loc[index])
        branch_source = str(branch_sources.loc[index])
        audit = _audit_base(
            part,
            int(index),
            test_id=test_id,
            channel=channel,
            passport=passport,
            branch=branch,
            branch_source=branch_source,
        )
        sequence = int(audit["sequence_index"])
        flags: list[str] = []
        record_ids: list[str] = []
        original = part.at[index, channel]
        raw = _as_finite(original)
        provided = not pd.isna(original) and str(original).strip() != ""
        if raw is None:
            flag = "invalid_reading" if provided else "missing_reading"
            flags.append(flag)
            audit["processing_status"] = "error" if provided else "missing"
            audit["warning"] = flag
            audit["quality_flags"] = flag
            event_id = _append_event(
                events,
                test_id=test_id,
                channel=channel,
                row_index=int(index),
                sequence_index=sequence,
                event_type=flag,
                reason="Исходное показание отсутствует или не является конечным числом; значение не заполнено.",
                method="preserve_nan",
                raw_before=previous_raw,
                raw_after=None,
                turn_before=previous_turn,
                turn_after=None,
                increment_mm=None,
            )
            record_ids.append(event_id)
            audit["correction_record_ids"] = ";".join(record_ids)
            rows.append(audit)
            values[int(index)] = None
            if provided:
                issues.append(
                    _issue(
                        "error",
                        "invalid_indicator_reading",
                        f"{channel}: показание должно быть конечным числом.",
                        test_id=test_id,
                        channel=channel,
                        rows=[int(index)],
                        raw_value=original,
                    )
                )
            else:
                issues.append(
                    _issue(
                        "warning",
                        "missing_indicator_reading",
                        f"{channel}: пропуск сохранён как NaN, перенос соседнего показания не выполнялся.",
                        test_id=test_id,
                        channel=channel,
                        rows=[int(index)],
                    )
                )
            continue

        audit["raw_reading"] = raw
        if not math.isclose(raw / passport.division_mm, round(raw / passport.division_mm), abs_tol=1e-6):
            flags.append("off_scale_division")

        if mode == "cumulative_settlement":
            raw_cumulative = passport.cumulative_sign * raw * passport.correction_factor
            increment = raw_cumulative if previous_ready is None else raw_cumulative - previous_ready
            corrected = raw_cumulative + passport.zero_correction_mm
            fatal = False
            if previous_ready is not None and not passport.compatibility_mode:
                expected = _expected_direction(branch)
                tolerance = float(passport.reverse_tolerance_mm or 0.0)
                reverse_motion = (
                    expected is not None
                    and (
                        (expected > 0 and increment < 0)
                        or (expected < 0 and increment > 0)
                    )
                )
                if reverse_motion and abs(increment) <= tolerance + 1e-12:
                    flags.append("small_reverse_motion")
                    record_ids.append(
                        _append_event(
                            events,
                            test_id=test_id,
                            channel=channel,
                            row_index=int(index),
                            sequence_index=sequence,
                            event_type="small_reverse_motion",
                            reason="Небольшой обратный ход готовой осадки сохранён без обнуления.",
                            method="cumulative_settlement",
                            raw_before=previous_raw,
                            raw_after=raw,
                            turn_before=0,
                            turn_after=0,
                            increment_mm=increment,
                        )
                    )
                elif reverse_motion:
                    flags.append("unexpected_reverse_motion")
                    fatal = True
                    record_ids.append(
                        _append_event(
                            events,
                            test_id=test_id,
                            channel=channel,
                            row_index=int(index),
                            sequence_index=sequence,
                            event_type="unexpected_reverse_motion",
                            reason="Обратный ход готовой осадки превышает паспортный допуск; raw сохранён, точка заблокирована.",
                            method="cumulative_settlement",
                            raw_before=previous_raw,
                            raw_after=raw,
                            turn_before=0,
                            turn_after=0,
                            increment_mm=increment,
                        )
                    )
                    issues.append(
                        _issue(
                            "error",
                            "unexpected_indicator_reverse_motion",
                            f"{channel}: обратный ход {increment:g} мм превышает допуск {tolerance:g} мм.",
                            test_id=test_id,
                            channel=channel,
                            rows=[int(index)],
                            raw_value=raw,
                        )
                    )
            audit.update(
                {
                    "turn_number": 0,
                    "unwrapped_reading": raw,
                    "computed_increment_mm": increment,
                    "cumulative_before_correction_mm": raw_cumulative,
                    "applied_correction_mm": np.nan if fatal else passport.zero_correction_mm,
                    "cumulative_settlement_mm": np.nan if fatal else corrected,
                    "settlement_effective_mm": np.nan if fatal else corrected,
                    "processing_status": "error" if fatal else "ok",
                    "conversion_method": "cumulative_settlement: s=sign·raw·factor+zero_correction",
                }
            )
            if previous_ready is not None and math.isclose(increment, 0.0, abs_tol=1e-12):
                flags.append("repeated_reading")
            elif 0 < abs(increment) < passport.division_mm * passport.correction_factor:
                flags.append("below_resolution")
            if not fatal:
                previous_ready = raw_cumulative
                previous_raw = raw
            previous_turn = 0
            values[int(index)] = None if fatal else corrected
        else:
            assert previous_unwrapped is not None
            if raw < 0 or raw >= passport.range_mm:
                flags.append("reading_out_of_scale_range")
                issues.append(
                    _issue(
                        "error",
                        "indicator_reading_out_of_range",
                        f"{channel}: показание {raw:g} вне диапазона [0; {passport.range_mm:g}).",
                        test_id=test_id,
                        channel=channel,
                        rows=[int(index)],
                        raw_value=raw,
                    )
                )
                audit["processing_status"] = "error"
                audit["warning"] = ";".join(flags)
                audit["quality_flags"] = ";".join(flags)
                rows.append(audit)
                values[int(index)] = None
                continue

            explicit_turn_value = part.at[index, turn_column] if turn_column else None
            explicit_turn_provided = (
                turn_column is not None
                and not pd.isna(explicit_turn_value)
                and str(explicit_turn_value).strip() != ""
            )
            candidate_turns: list[int] = []
            chosen_turn: int | None = None
            method = ""
            if explicit_turn_provided:
                chosen_turn = _as_integer(explicit_turn_value)
                if chosen_turn is None:
                    issues.append(
                        _issue(
                            "error",
                            "invalid_explicit_turn",
                            f"{turn_column} должен содержать целое число оборотов.",
                            test_id=test_id,
                            channel=channel,
                            rows=[int(index)],
                            raw_value=explicit_turn_value,
                        )
                    )
                    flags.append("invalid_explicit_turn")
                else:
                    method = "explicit_turn_number"
                    flags.append("explicit_turn_number")
            elif not wrapped:
                chosen_turn = passport.initial_turn
                method = "non_wrapped_scale"
            else:
                assert passport.max_increment_mm is not None
                expected = _expected_direction(branch)
                tolerance = float(passport.reverse_tolerance_mm or 0.0)
                dial_limit = passport.max_increment_mm / passport.correction_factor
                lower = (previous_unwrapped - raw - dial_limit) / passport.range_mm
                upper = (previous_unwrapped - raw + dial_limit) / passport.range_mm
                all_min = math.ceil(lower - 1e-12)
                all_max = math.floor(upper + 1e-12)
                directed_min, directed_max = all_min, all_max
                dial_tolerance = tolerance / passport.correction_factor
                if expected is not None:
                    # q·(raw+kR-prev) must have the branch sign, allowing the
                    # declared reverse tolerance.  Tighten the integer interval
                    # analytically; never enumerate a user-sized range.
                    require_lower = (expected > 0 and scale_direction > 0) or (
                        expected < 0 and scale_direction < 0
                    )
                    if require_lower:
                        bound = (
                            previous_unwrapped - raw - dial_tolerance
                        ) / passport.range_mm
                        directed_min = max(directed_min, math.ceil(bound - 1e-12))
                    else:
                        bound = (
                            previous_unwrapped - raw + dial_tolerance
                        ) / passport.range_mm
                        directed_max = min(directed_max, math.floor(bound + 1e-12))
                all_count = max(0, all_max - all_min + 1)
                directed_count = max(0, directed_max - directed_min + 1)

                def candidate_summary(first: int, last: int, count: int) -> list[int]:
                    if count <= 0:
                        return []
                    if count <= 20:
                        return list(range(first, last + 1))
                    return [first, last]

                candidate_turns = candidate_summary(
                    directed_min, directed_max, directed_count
                )
                within_step_candidates = candidate_summary(all_min, all_max, all_count)
                if directed_count == 1:
                    chosen_turn = directed_min
                    method = "automatic_unique_turn_candidate"
                elif directed_count == 0 and all_count == 1:
                    # Preserve a uniquely identified reverse movement so that
                    # the direction check below can record it explicitly.
                    chosen_turn = all_min
                    method = "automatic_unique_reverse_candidate"
                elif directed_count > 1 or (directed_count == 0 and all_count > 1):
                    ambiguous_candidates = candidate_turns or within_step_candidates
                    candidate_turns = list(ambiguous_candidates)
                    ambiguous_count = directed_count or all_count
                    display = (
                        str(ambiguous_candidates)
                        if ambiguous_count <= 20
                        else f"{ambiguous_candidates[0]}…{ambiguous_candidates[-1]} ({ambiguous_count} вариантов)"
                    )
                    flags.append("ambiguous_scale_crossing")
                    issues.append(
                        _issue(
                            "error",
                            "ambiguous_indicator_turn",
                            f"{channel}: допустимы несколько номеров оборота {display}; задайте {channel}_turn_number явно.",
                            test_id=test_id,
                            channel=channel,
                            rows=[int(index)],
                            raw_value=raw,
                        )
                    )
                else:
                    flags.append("invalid_jump")
                    issues.append(
                        _issue(
                            "error",
                            "invalid_indicator_jump",
                            f"{channel}: ни один переход не укладывается в направление ветви и max_increment_mm={passport.max_increment_mm:g}.",
                            test_id=test_id,
                            channel=channel,
                            rows=[int(index)],
                            raw_value=raw,
                        )
                    )

            if chosen_turn is None:
                audit.update(
                    {
                        "processing_status": "error",
                        "warning": ";".join(flags),
                        "quality_flags": ";".join(flags),
                        "conversion_method": method or "blocked_no_unique_turn",
                    }
                )
                event_id = _append_event(
                    events,
                    test_id=test_id,
                    channel=channel,
                    row_index=int(index),
                    sequence_index=sequence,
                    event_type=flags[-1],
                    reason="Преобразование заблокировано: скрытый выбор оборота запрещён.",
                    method=audit["conversion_method"],
                    raw_before=previous_raw,
                    raw_after=raw,
                    turn_before=previous_turn,
                    turn_after=None,
                    increment_mm=None,
                    candidates=candidate_turns,
                )
                audit["correction_record_ids"] = event_id
                rows.append(audit)
                values[int(index)] = None
                continue

            unwrapped = raw + chosen_turn * passport.range_mm
            increment = scale_direction * (unwrapped - previous_unwrapped) * passport.correction_factor
            if method == "explicit_turn_number":
                record_ids.append(
                    _append_event(
                        events,
                        test_id=test_id,
                        channel=channel,
                        row_index=int(index),
                        sequence_index=sequence,
                        event_type="explicit_turn_number",
                        reason="Использован явно записанный оператором номер оборота; автоматический выбор не выполнялся.",
                        method=method,
                        raw_before=previous_raw,
                        raw_after=raw,
                        turn_before=previous_turn,
                        turn_after=chosen_turn,
                        increment_mm=increment,
                    )
                )
            expected = _expected_direction(branch)
            tolerance = float(passport.reverse_tolerance_mm or 0.0)
            reverse_motion = (
                expected is not None
                and ((expected > 0 and increment < 0) or (expected < 0 and increment > 0))
            )
            if reverse_motion and abs(increment) <= tolerance + 1e-12:
                flags.append("small_reverse_motion")
                record_ids.append(
                    _append_event(
                        events,
                        test_id=test_id,
                        channel=channel,
                        row_index=int(index),
                        sequence_index=sequence,
                        event_type="small_reverse_motion",
                        reason="Небольшой обратный ход сохранён без обнуления или зажима.",
                        method=method,
                        raw_before=previous_raw,
                        raw_after=raw,
                        turn_before=previous_turn,
                        turn_after=chosen_turn,
                        increment_mm=increment,
                    )
                )
            elif reverse_motion:
                flags.append("unexpected_reverse_motion")
                record_ids.append(
                    _append_event(
                        events,
                        test_id=test_id,
                        channel=channel,
                        row_index=int(index),
                        sequence_index=sequence,
                        event_type="unexpected_reverse_motion",
                        reason="Обратный ход превышает паспортный допуск; значение сохранено в raw, научное преобразование точки заблокировано.",
                        method=method,
                        raw_before=previous_raw,
                        raw_after=raw,
                        turn_before=previous_turn,
                        turn_after=chosen_turn,
                        increment_mm=increment,
                    )
                )
                issues.append(
                    _issue(
                        "error",
                        "unexpected_indicator_reverse_motion",
                        f"{channel}: обратный ход {increment:g} мм превышает допуск {tolerance:g} мм.",
                        test_id=test_id,
                        channel=channel,
                        rows=[int(index)],
                        raw_value=raw,
                    )
                )
            if passport.max_increment_mm is not None and abs(increment) > passport.max_increment_mm + 1e-12:
                flags.append("invalid_jump")
                record_ids.append(
                    _append_event(
                        events,
                        test_id=test_id,
                        channel=channel,
                        row_index=int(index),
                        sequence_index=sequence,
                        event_type="invalid_jump",
                        reason="Приращение превышает явно заданный предел; значение не применено.",
                        method=method,
                        raw_before=previous_raw,
                        raw_after=raw,
                        turn_before=previous_turn,
                        turn_after=chosen_turn,
                        increment_mm=increment,
                    )
                )
                issues.append(
                    _issue(
                        "error",
                        "invalid_indicator_jump",
                        f"{channel}: приращение {increment:g} мм превышает max_increment_mm={passport.max_increment_mm:g}.",
                        test_id=test_id,
                        channel=channel,
                        rows=[int(index)],
                        raw_value=raw,
                    )
                )
            if math.isclose(increment, 0.0, abs_tol=1e-12):
                flags.append("repeated_reading")
            elif 0 < abs(increment) < passport.division_mm * passport.correction_factor - 1e-12:
                flags.append("below_resolution")

            raw_cumulative = (
                scale_direction
                * (unwrapped - (float(passport.initial_reading) + passport.initial_turn * passport.range_mm))
                * passport.correction_factor
            )
            corrected = raw_cumulative + passport.zero_correction_mm
            fatal = any(flag in {"invalid_jump", "unexpected_reverse_motion"} for flag in flags)
            if passport.travel_range_mm is not None and abs(raw_cumulative) > passport.travel_range_mm + 1e-12:
                flags.append("travel_range_exceeded")
                fatal = True
                record_ids.append(
                    _append_event(
                        events,
                        test_id=test_id,
                        channel=channel,
                        row_index=int(index),
                        sequence_index=sequence,
                        event_type="travel_range_exceeded",
                        reason="Накопленное перемещение превышает полный механический ход; точка заблокирована.",
                        method=method,
                        raw_before=previous_raw,
                        raw_after=raw,
                        turn_before=previous_turn,
                        turn_after=chosen_turn,
                        increment_mm=increment,
                    )
                )
                issues.append(
                    _issue(
                        "error",
                        "indicator_travel_range_exceeded",
                        f"{channel}: накопленное перемещение {raw_cumulative:g} мм превышает ход {passport.travel_range_mm:g} мм.",
                        test_id=test_id,
                        channel=channel,
                        rows=[int(index)],
                    )
                )
            if chosen_turn != previous_turn:
                flags.append("zero_crossing")
                crossing_reason = (
                    "Переход через ноль зарегистрирован, но точка заблокирована контролем качества."
                    if fatal
                    else (
                        "Переход через ноль применён по явно заданному номеру оборота."
                        if method == "explicit_turn_number"
                        else "Переход через ноль принят по единственному допустимому номеру оборота."
                    )
                )
                record_ids.append(
                    _append_event(
                        events,
                        test_id=test_id,
                        channel=channel,
                        row_index=int(index),
                        sequence_index=sequence,
                        event_type="zero_crossing",
                        reason=crossing_reason,
                        method=method,
                        raw_before=previous_raw,
                        raw_after=raw,
                        turn_before=previous_turn,
                        turn_after=chosen_turn,
                        increment_mm=increment,
                        candidates=candidate_turns,
                    )
                )
            audit.update(
                {
                    "turn_number": chosen_turn,
                    "unwrapped_reading": unwrapped,
                    "computed_increment_mm": increment,
                    "cumulative_before_correction_mm": raw_cumulative,
                    "applied_correction_mm": np.nan if fatal else passport.zero_correction_mm,
                    "cumulative_settlement_mm": np.nan if fatal else corrected,
                    "settlement_effective_mm": np.nan if fatal else corrected,
                    "processing_status": "error" if fatal else "ok",
                    "conversion_method": method,
                }
            )
            values[int(index)] = None if fatal else corrected
            if not fatal:
                previous_raw = raw
                previous_turn = chosen_turn
                previous_unwrapped = unwrapped

        if (
            passport.zero_correction_mm != 0
            and not correction_logged
            and audit["processing_status"] != "error"
        ):
            flags.append("zero_correction_applied")
            record_ids.append(
                _append_event(
                    events,
                    test_id=test_id,
                    channel=channel,
                    row_index=int(index),
                    sequence_index=sequence,
                    event_type="zero_correction_applied",
                    reason="Применена явно заданная паспортная коррекция нуля.",
                    method="additive_zero_correction",
                    raw_before=raw,
                    raw_after=raw,
                    turn_before=int(audit["turn_number"]) if not pd.isna(audit["turn_number"]) else None,
                    turn_after=int(audit["turn_number"]) if not pd.isna(audit["turn_number"]) else None,
                    increment_mm=float(audit["computed_increment_mm"]),
                    correction_mm=passport.zero_correction_mm,
                )
            )
            correction_logged = True
        if branch_source != "protocol":
            flags.append("branch_inferred_from_load")
        audit["quality_flags"] = ";".join(dict.fromkeys(flags))
        audit["warning"] = audit["quality_flags"]
        if audit["processing_status"] == "ok" and flags:
            audit["processing_status"] = "warning"
        audit["correction_record_ids"] = ";".join(dict.fromkeys(record_ids))
        rows.append(audit)
    return rows, events, issues, values


def _unprocessed_rows(
    part: pd.DataFrame,
    *,
    test_id: str,
    channel: str,
    branches: pd.Series,
    branch_sources: pd.Series,
    warning: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index in part.index:
        row = _audit_base(
            part,
            int(index),
            test_id=test_id,
            channel=channel,
            passport=None,
            branch=str(branches.loc[index]),
            branch_source=str(branch_sources.loc[index]),
        )
        row["warning"] = warning
        row["quality_flags"] = warning
        row["processing_status"] = "unprocessed"
        row["conversion_method"] = "none"
        rows.append(row)
    return rows


def _passport_row(test_id: str, passport: IndicatorPassport) -> dict[str, Any]:
    row = asdict(passport)
    row.update(
        {
            "test_id": str(test_id),
            "schema_version": INDICATOR_PROCESSING_SCHEMA,
            "algorithm_version": INDICATOR_ALGORITHM_VERSION,
            "confirmed": True,
            "reverse_tolerance_source": (
                "passport" if passport.reverse_tolerance_mm != passport.division_mm * passport.correction_factor else "one_effective_division"
            ),
            "formula": (
                "s=sign·raw·factor+zero_correction"
                if passport.mode == "cumulative_settlement"
                else "Δs=q·factor·(raw[i]-raw[i-1]+k·range); s=ΣΔs+zero_correction"
            ),
        }
    )
    return row


def process_indicator_frame(
    frame: pd.DataFrame,
    metadata: dict[str, Any] | None,
) -> IndicatorProcessingResult:
    """Process all indicator channels and return deterministic audit artefacts.

    Direct ``settlement`` remains authoritative where supplied.  When any row
    of a test needs indicator-derived settlement, channel continuity is checked
    over the whole test: an invalid intermediate raw point is not silently
    skipped.  Fully auxiliary indicators are preserved as raw with warnings.
    """

    result = IndicatorProcessingResult()
    if "test_id" not in frame or frame.empty:
        return result
    vertical_channels = [channel for channel in _VERTICAL_CHANNELS if channel in frame]
    if not vertical_channels:
        return result

    for test_value, part in frame.groupby("test_id", sort=False):
        test_id = str(test_value)
        part_rows = [int(index) for index in part.index]
        settlement = (
            pd.to_numeric(part["settlement"], errors="coerce")
            if "settlement" in part
            else pd.Series(np.nan, index=part.index)
        )
        indicator_present = (
            part[vertical_channels]
            .apply(pd.to_numeric, errors="coerce")
            .notna()
            .any(axis=1)
        )
        fallback_mask = settlement.isna() & indicator_present
        fallback_required = bool(fallback_mask.any())
        branches, branch_sources = _branch_series(part)
        channel_values: dict[str, dict[int, float | None]] = {}
        test_passports: dict[str, IndicatorPassport] = {}

        for channel in vertical_channels:
            passport, passport_issues = resolve_indicator_passport(
                metadata, test_id, channel, rows=part_rows
            )
            if not fallback_required:
                for issue in passport_issues:
                    if issue.level == "error":
                        issue.level = "warning"
                        issue.blocks_processing = False
            result.issues.extend(passport_issues)
            if passport is None:
                raw_present = pd.to_numeric(part[channel], errors="coerce").notna().any()
                if raw_present:
                    result.audit_rows.extend(
                        _unprocessed_rows(
                            part,
                            test_id=test_id,
                            channel=channel,
                            branches=branches,
                            branch_sources=branch_sources,
                            warning="passport_missing",
                        )
                    )
                    channel_needed = bool(
                        (
                            fallback_mask
                            & pd.to_numeric(part[channel], errors="coerce").notna()
                        ).any()
                    )
                    if channel_needed:
                        result.issues.append(
                            _issue(
                                "error",
                                "missing_indicator_channel_passport",
                                f"{channel}: raw-показания участвуют в строках без settlement, но поканальный паспорт отсутствует.",
                                test_id=test_id,
                                channel=channel,
                                rows=part.index[
                                    fallback_mask
                                    & pd.to_numeric(part[channel], errors="coerce").notna()
                                ].tolist(),
                            )
                        )
                continue
            test_passports[channel] = passport
            result.passport_rows.append(_passport_row(test_id, passport))
            audit_rows, event_rows, channel_issues, values = _process_channel(
                part,
                test_id=test_id,
                channel=channel,
                passport=passport,
                branches=branches,
                branch_sources=branch_sources,
            )
            if not fallback_required:
                for issue in channel_issues:
                    if issue.level == "error":
                        issue.level = "warning"
                        issue.blocks_processing = False
            result.audit_rows.extend(audit_rows)
            result.event_rows.extend(event_rows)
            result.issues.extend(channel_issues)
            channel_values[channel] = values
            result.channel_settlement_by_row.setdefault(channel, {}).update(values)

        rows_needing_indicators = settlement.index[fallback_mask].tolist()
        if fallback_required and not test_passports:
            legacy = (
                "indicator_requires_calibration" in part
                and part["indicator_requires_calibration"].fillna(False).astype(bool).any()
            )
            code = "uncalibrated_legacy_indicator" if legacy else "indicator_mode_not_confirmed"
            message = (
                "Legacy-показания индикатора требуют явного поканального паспорта."
                if legacy
                else "Для расчёта осадки по indicator_* требуется подтверждённый паспорт и режим шкалы."
            )
            result.issues.append(
                _issue(
                    "error",
                    code,
                    message,
                    test_id=test_id,
                    channel="indicator_passports",
                    rows=rows_needing_indicators,
                )
            )
        elif not fallback_required and not test_passports:
            result.issues.append(
                _issue(
                    "warning",
                    "uncalibrated_indicators_ignored",
                    "Прямая settlement использована; indicator_* сохранены как raw и не участвовали в расчёте.",
                    test_id=test_id,
                    channel="indicator_passports",
                    rows=part_rows,
                )
            )

        # Reference correction is independently calibrated when a passport is
        # supplied.  The old shared reference_sign path remains in data.py for
        # backward compatibility only.
        reference_values: dict[int, float | None] = {}
        reference_required = False
        if (
            "reference_indicator" in part
            and pd.to_numeric(part["reference_indicator"], errors="coerce").notna().any()
        ):
            reference_passport, reference_issues = resolve_indicator_passport(
                metadata, test_id, "reference_indicator", rows=part_rows
            )
            if reference_passport is not None:
                reference_required = True
                result.passport_rows.append(_passport_row(test_id, reference_passport))
                ref_audit, ref_events, ref_issues, reference_values = _process_channel(
                    part,
                    test_id=test_id,
                    channel="reference_indicator",
                    passport=reference_passport,
                    branches=branches,
                    branch_sources=branch_sources,
                )
                result.audit_rows.extend(ref_audit)
                result.event_rows.extend(ref_events)
                reference_issues.extend(ref_issues)
                result.channel_settlement_by_row.setdefault(
                    "reference_indicator", {}
                ).update(reference_values)
            elif fallback_required and pd.to_numeric(part["reference_indicator"], errors="coerce").notna().any():
                effective = _effective_metadata(metadata, test_id)
                if effective.get("reference_sign") is None:
                    reference_issues.append(
                        _issue(
                            "error",
                            "missing_reference_indicator_passport",
                            "reference_indicator требует собственного паспорта либо совместимого reference_sign.",
                            test_id=test_id,
                            channel="reference_indicator",
                            rows=part_rows,
                        )
                    )
            result.issues.extend(reference_issues)

        for index in part.index:
            available = [
                values.get(int(index))
                for values in channel_values.values()
                if values.get(int(index)) is not None
            ]
            derived = float(np.mean(available)) if available else None
            reference = reference_values.get(int(index))
            if derived is not None and reference_required:
                derived = None if reference is None else derived + float(reference)
            result.settlement_by_row[int(index)] = derived

        # Add the effective aggregate and reference correction to every
        # vertical audit row without changing the channel result.
        for audit in result.audit_rows:
            if audit["test_id"] != test_id or audit["channel"] not in vertical_channels:
                continue
            index = int(audit["row_index"])
            reference = reference_values.get(index)
            audit["reference_correction_mm"] = (
                float(reference)
                if reference is not None
                else (np.nan if reference_required else 0.0)
            )
            audit["settlement_effective_mm"] = result.settlement_by_row.get(index)

    return result


def indicator_audit_frame(result_or_frame: IndicatorProcessingResult | pd.DataFrame) -> pd.DataFrame:
    if isinstance(result_or_frame, IndicatorProcessingResult):
        return pd.DataFrame(result_or_frame.audit_rows)
    return pd.DataFrame(result_or_frame.attrs.get("indicator_processing_audit", []))


def indicator_event_frame(result_or_frame: IndicatorProcessingResult | pd.DataFrame) -> pd.DataFrame:
    if isinstance(result_or_frame, IndicatorProcessingResult):
        return pd.DataFrame(result_or_frame.event_rows)
    return pd.DataFrame(result_or_frame.attrs.get("indicator_processing_events", []))


def indicator_passport_frame(result_or_frame: IndicatorProcessingResult | pd.DataFrame) -> pd.DataFrame:
    if isinstance(result_or_frame, IndicatorProcessingResult):
        return pd.DataFrame(result_or_frame.passport_rows)
    return pd.DataFrame(result_or_frame.attrs.get("indicator_calibration_parameters", []))
