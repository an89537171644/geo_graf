"""Data preparation, non-destructive corrections, branch and failure handling."""

from __future__ import annotations

import hashlib
import json
import math
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .indicators import (
    INDICATOR_PROCESSING_SCHEMA,
    INDICATOR_MODES,
    canonical_indicator_mode,
    process_indicator_frame,
)
from .schema import (
    BRANCHES,
    CORRECTION_MODES,
    FAILURE_WORDS,
    FailureResult,
    ValidationIssue,
    merged_metadata,
)


_FORCE_TO_KN = {
    "kn": 1.0,
    "кн": 1.0,
    "n": 0.001,
    "н": 0.001,
    "mn": 1000.0,
    "мн": 1000.0,
    "kgf": 0.00980665,
    "кгс": 0.00980665,
    "tf": 9.80665,
    "тс": 9.80665,
}
_PRESSURE_TO_KPA = {
    "kpa": 1.0,
    "кпа": 1.0,
    "pa": 0.001,
    "па": 0.001,
    "mpa": 1000.0,
    "мпа": 1000.0,
}
_SETTLEMENT_TO_MM = {"mm": 1.0, "мм": 1.0, "cm": 10.0, "см": 10.0, "m": 1000.0, "м": 1000.0}
_SUPPORTED_PRECALIBRATED_INDICATOR_MODES = {
    "direct_displacement",
    "cumulative_settlement",
    "increasing",
    "increasing_wrapped",
    "decreasing",
    "decreasing_wrapped",
    "direct_scale",
    "direct_scale_wrapped",
    "reverse_scale",
    "reverse_scale_wrapped",
}


def _unit_key(value: Any) -> str:
    return str(value or "").strip().casefold().replace(" ", "")


def _load_kind_and_factor(metadata: dict[str, Any]) -> tuple[str, float]:
    unit = _unit_key(metadata.get("load_unit", "kN"))
    kind = str(metadata.get("load_kind", "force")).strip().casefold()
    if unit in _PRESSURE_TO_KPA and kind == "force" and "load_kind" not in metadata:
        kind = "pressure"
    if kind in {"force", "f", "сила", "нагрузка"}:
        if unit not in _FORCE_TO_KN:
            raise ValueError(f"Неподдерживаемая единица силы: {metadata.get('load_unit')}")
        return "force", _FORCE_TO_KN[unit]
    if kind in {"pressure", "p", "давление"}:
        if unit not in _PRESSURE_TO_KPA:
            raise ValueError(f"Неподдерживаемая единица давления: {metadata.get('load_unit')}")
        return "pressure", _PRESSURE_TO_KPA[unit]
    raise ValueError(f"Неподдерживаемый load_kind: {metadata.get('load_kind')}")


def _settlement_factor(metadata: dict[str, Any]) -> float:
    unit = _unit_key(metadata.get("settlement_unit", "mm"))
    if unit not in _SETTLEMENT_TO_MM:
        raise ValueError(f"Неподдерживаемая единица осадки: {metadata.get('settlement_unit')}")
    return _SETTLEMENT_TO_MM[unit]


def _stable_hash(value: Any) -> str:
    if isinstance(value, pd.DataFrame):
        payload = value.to_json(orient="split", date_format="iso", double_precision=12)
    else:
        payload = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class AuditTrail:
    """Append-only in-memory audit trail suitable for JSON export.

    Existing events are returned as copies. Reversal is represented by a new
    event; the class deliberately exposes no delete or update operation.
    """

    def __init__(self, events: Iterable[dict[str, Any]] | None = None) -> None:
        self._events = deepcopy(list(events or []))

    def record(
        self,
        action: str,
        *,
        scope: str,
        reason: str,
        parameters: dict[str, Any] | None = None,
        before: Any = None,
        after: Any = None,
        user: str = "local-user",
        method: str | None = None,
    ) -> dict[str, Any]:
        if not reason or not reason.strip():
            raise ValueError("Для ручного решения требуется непустая причина.")
        def readable_value(value: Any) -> Any:
            if value is None or isinstance(value, (str, int, float, bool)):
                return deepcopy(value)
            if isinstance(value, (list, tuple, dict)):
                try:
                    encoded = json.dumps(value, ensure_ascii=False, default=str)
                except (TypeError, ValueError):
                    return None
                return deepcopy(value) if len(encoded) <= 4000 else None
            return None

        event = {
            "event_id": len(self._events) + 1,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "user": user,
            "action": action,
            "scope": scope,
            "reason": reason.strip(),
            "parameters": deepcopy(parameters or {}),
            "before_hash": _stable_hash(before) if before is not None else None,
            "after_hash": _stable_hash(after) if after is not None else None,
            "before_value": readable_value(before),
            "after_value": readable_value(after),
            "method": method,
        }
        self._events.append(event)
        return deepcopy(event)

    @property
    def events(self) -> list[dict[str, Any]]:
        return deepcopy(self._events)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.events)

    def to_json(self) -> str:
        return json.dumps(self.events, ensure_ascii=False, indent=2)


def metadata_for_test(metadata: dict[str, Any] | None, test_id: str) -> dict[str, Any]:
    result = merged_metadata(metadata)
    tests = result.pop("tests", {}) or {}
    if isinstance(tests, dict) and str(test_id) in tests:
        override = tests[str(test_id)] or {}
        if (
            isinstance(override, dict)
            and ("stamp_diameter_mm" in override or "stamp_shape" in override)
            and "stamp_area_m2" not in override
        ):
            # A test-specific geometry must not inherit an area calculated for
            # a different global diameter.
            result["stamp_area_m2"] = None
        result.update(override)
    return result


def _explicit_metadata_for_test(
    metadata: dict[str, Any] | None, test_id: str
) -> dict[str, Any]:
    """Resolve only user-supplied values, without scientific defaults."""

    if not isinstance(metadata, dict):
        return {}
    result = {key: value for key, value in metadata.items() if key != "tests"}
    tests = metadata.get("tests")
    if isinstance(tests, dict):
        override = tests.get(str(test_id))
        if isinstance(override, dict):
            for key, value in override.items():
                if (
                    key in {"indicator_passport", "indicator_passports", "indicator_channels"}
                    and isinstance(value, dict)
                    and isinstance(result.get(key), dict)
                ):
                    merged = deepcopy(result[key])
                    for nested_key, nested_value in value.items():
                        if isinstance(nested_value, dict) and isinstance(merged.get(nested_key), dict):
                            merged[nested_key] = {**merged[nested_key], **nested_value}
                        else:
                            merged[nested_key] = deepcopy(nested_value)
                    result[key] = merged
                else:
                    result[key] = deepcopy(value)
    return result


def _indicator_instrument_ids(metadata: dict[str, Any]) -> list[str]:
    identifiers: list[str] = []
    direct = metadata.get("indicator_instrument_id")
    if direct is not None and str(direct).strip():
        identifiers.append(str(direct).strip())
    instruments = metadata.get("instruments")
    if isinstance(instruments, dict):
        instruments = [instruments]
    if isinstance(instruments, list):
        identifiers.extend(
            str(item.get("instrument_id")).strip()
            for item in instruments
            if isinstance(item, dict) and item.get("instrument_id")
        )
    passport = metadata.get("project_passport")
    passport_instruments = passport.get("instruments") if isinstance(passport, dict) else None
    if isinstance(passport_instruments, dict):
        passport_instruments = [passport_instruments]
    if isinstance(passport_instruments, list):
        identifiers.extend(
            str(item.get("instrument_id")).strip()
            for item in passport_instruments
            if isinstance(item, dict) and item.get("instrument_id")
        )
    for container_name in ("indicator_passports", "indicator_channels"):
        container = metadata.get(container_name)
        if not isinstance(container, dict):
            continue
        for item in container.values():
            if not isinstance(item, dict):
                continue
            identifier = item.get("instrument_id")
            if identifier is not None and str(identifier).strip():
                identifiers.append(str(identifier).strip())
    common = metadata.get("indicator_passport")
    if isinstance(common, dict):
        identifier = common.get("instrument_id")
        if identifier is not None and str(identifier).strip():
            identifiers.append(str(identifier).strip())
    return list(dict.fromkeys(value for value in identifiers if value))


def _indicator_calibration_parameters(
    metadata: dict[str, Any], *, reference_used: bool = False
) -> dict[str, Any]:
    nested: dict[str, Any] = {}
    common = metadata.get("indicator_passport")
    if isinstance(common, dict):
        nested.update(common)
    for container_name in ("indicator_passports", "indicator_channels"):
        container = metadata.get(container_name)
        if not isinstance(container, dict):
            continue
        first = next(
            (value for key, value in container.items() if key.startswith("indicator_") and isinstance(value, dict)),
            None,
        )
        if first:
            nested.update(first)
            break
    if nested:
        mode = canonical_indicator_mode(
            nested.get("mode") or nested.get("indicator_mode") or nested.get("scale_mode")
        )
        try:
            factor = float(
                nested.get("correction_factor", nested.get("calibration_factor"))
            )
        except (TypeError, ValueError):
            factor = None
        try:
            division = float(
                nested.get("division_mm", nested.get("resolution_mm"))
            )
        except (TypeError, ValueError):
            division = None
        sign = -1.0 if mode.startswith("decreasing") else 1.0
        reference_passport = any(
            isinstance(metadata.get(name), dict)
            and isinstance(metadata[name].get("reference_indicator"), dict)
            for name in ("indicator_passports", "indicator_channels")
        )
        reference_sign = None
        reference_valid = not reference_used or reference_passport
        instrument_ids = _indicator_instrument_ids(metadata)
        confirmed = bool(
            mode in INDICATOR_MODES
            and factor is not None
            and math.isfinite(factor)
            and factor > 0
            and division is not None
            and math.isfinite(division)
            and division > 0
            and instrument_ids
            and reference_valid
        )
        return {
            "confirmed": confirmed,
            "mode": mode,
            "unit": "mm",
            "scale_to_mm": 1.0,
            "factor": factor,
            "sign": sign,
            "resolution_mm": (
                division * factor
                if division is not None and factor is not None
                else None
            ),
            "reference_sign": reference_sign,
            "reference_used": bool(reference_used),
            "instrument_ids": instrument_ids,
        }
    mode = str(metadata.get("indicator_mode") or "").strip().casefold()
    unit = _unit_key(metadata.get("indicator_unit"))
    try:
        scale_to_mm = _SETTLEMENT_TO_MM[unit]
    except KeyError:
        scale_to_mm = None
    try:
        factor = float(metadata.get("indicator_calibration_factor"))
    except (TypeError, ValueError):
        factor = None
    try:
        sign = float(metadata.get("indicator_sign"))
    except (TypeError, ValueError):
        sign = None
    try:
        resolution = float(metadata.get("indicator_resolution_mm"))
    except (TypeError, ValueError):
        resolution = None
    try:
        reference_sign = float(metadata.get("reference_sign"))
    except (TypeError, ValueError):
        reference_sign = None
    instrument_ids = _indicator_instrument_ids(metadata)
    sign_valid = sign is not None and math.isfinite(sign) and math.isclose(abs(sign), 1.0)
    reference_valid = not reference_used
    confirmed = bool(
        mode in _SUPPORTED_PRECALIBRATED_INDICATOR_MODES
        and scale_to_mm is not None
        and factor is not None
        and math.isfinite(factor)
        and factor > 0
        and sign_valid
        and resolution is not None
        and math.isfinite(resolution)
        and resolution > 0
        and instrument_ids
        and reference_valid
    )
    return {
        "confirmed": confirmed,
        "mode": mode,
        "unit": str(metadata.get("indicator_unit") or ""),
        "scale_to_mm": scale_to_mm,
        "factor": factor,
        "sign": sign,
        "resolution_mm": resolution,
        "reference_sign": reference_sign,
        "reference_used": bool(reference_used),
        "instrument_ids": instrument_ids,
    }


def stamp_area_m2(metadata: dict[str, Any]) -> float | None:
    explicit = metadata.get("stamp_area_m2")
    if explicit is not None:
        try:
            value = float(explicit)
            return value if value > 0 else None
        except (TypeError, ValueError):
            return None
    diameter = metadata.get("stamp_diameter_mm")
    if diameter is None:
        return None
    try:
        diameter_m = float(diameter) / 1000.0
    except (TypeError, ValueError):
        return None
    if diameter_m <= 0:
        return None
    shape = str(metadata.get("stamp_shape", "circle")).casefold()
    if shape in {"circle", "round", "круг", "круглый"}:
        return math.pi * diameter_m**2 / 4.0
    return None


def validate_measurements(
    frame: pd.DataFrame, metadata: dict[str, Any] | None = None
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    if metadata is not None and not isinstance(metadata, dict):
        return [ValidationIssue("error", "invalid_metadata", "Metadata должны быть JSON-объектом.")]
    if isinstance(metadata, dict) and "tests" in metadata and not isinstance(metadata.get("tests"), dict):
        issues.append(
            ValidationIssue("error", "invalid_tests_metadata", "Поле metadata.tests должно быть объектом.")
        )
    positions = metadata.get("indicator_positions_mm") if isinstance(metadata, dict) else None
    if positions is not None:
        valid_positions = isinstance(positions, dict) and all(
            isinstance(value, (list, tuple))
            and len(value) == 2
            and all(isinstance(number, (int, float)) and math.isfinite(float(number)) for number in value)
            for value in positions.values()
        )
        if not valid_positions:
            issues.append(
                ValidationIssue(
                    "error",
                    "invalid_indicator_positions",
                    "indicator_positions_mm должен задавать конечные координаты [x, y].",
                )
            )
    for name in ("test_id", "stage", "load"):
        if name not in frame.columns:
            issues.append(ValidationIssue("error", "missing_column", f"Нет обязательного столбца {name}."))
    if issues:
        return issues
    if frame.empty:
        return [ValidationIssue("error", "empty_table", "Таблица измерений пуста.")]

    sources = [c for c in ["settlement", "indicator_1", "indicator_2", "indicator_3", "indicator_4"] if c in frame]
    if not sources:
        issues.append(
            ValidationIssue(
                "error",
                "missing_settlement_source",
                "Нужен settlement или хотя бы один столбец indicator_1..4.",
            )
        )
    supplied_missing = (
        pd.to_numeric(frame["settlement"], errors="coerce").isna()
        if "settlement" in frame
        else pd.Series(True, index=frame.index)
    )
    test_ids_for_measurement = frame["test_id"].astype("string")
    test_requires_indicator = pd.Series(False, index=frame.index)
    for test_id in test_ids_for_measurement.dropna().unique():
        mask = test_ids_for_measurement == test_id
        test_requires_indicator.loc[mask] = bool(supplied_missing.loc[mask].any())
    numeric_measurement_columns = [
        column
        for column in [
            "settlement",
            "indicator_1",
            "indicator_2",
            "indicator_3",
            "indicator_4",
            "reference_indicator",
            "horizontal_indicator",
        ]
        if column in frame
    ]
    for column in numeric_measurement_columns:
        original = frame[column]
        numeric = pd.to_numeric(original, errors="coerce")
        provided = original.notna() & original.astype(str).str.strip().ne("")
        invalid = provided & ~np.isfinite(numeric)
        blocking_invalid = (
            invalid
            if column == "settlement"
            else invalid & test_requires_indicator
        )
        auxiliary_invalid = invalid & ~blocking_invalid
        if blocking_invalid.any():
            issues.append(
                ValidationIssue(
                    "error",
                    "invalid_measurement",
                    f"{column} должен содержать только конечные числа или пустые значения.",
                    rows=frame.index[blocking_invalid].tolist(),
                    column=column,
                )
            )
        if auxiliary_invalid.any():
            issues.append(
                ValidationIssue(
                    "warning",
                    "invalid_auxiliary_indicator_ignored",
                    f"Нечисловые auxiliary-значения {column} сохранены как raw и не использованы: прямая settlement авторитетна.",
                    rows=frame.index[auxiliary_invalid].tolist(),
                    column=column,
                )
            )
    indicator_columns = _indicator_columns(frame)
    indicator_present = (
        frame[indicator_columns]
        .apply(pd.to_numeric, errors="coerce")
        .notna()
        .any(axis=1)
        if indicator_columns
        else pd.Series(False, index=frame.index)
    )
    indicator_fallback_rows = supplied_missing & indicator_present
    legacy_uncalibrated = (
        frame["indicator_requires_calibration"].fillna(False).astype(bool)
        if "indicator_requires_calibration" in frame
        else pd.Series(False, index=frame.index)
    )
    test_ids_as_text = frame["test_id"].astype("string")
    for test_id in frame["test_id"].dropna().astype(str).unique():
        test_mask = test_ids_as_text == str(test_id)
        affected = indicator_fallback_rows & test_mask.fillna(False) & ~legacy_uncalibrated
        auxiliary = (~supplied_missing) & indicator_present & test_mask.fillna(False)
        reference_used = bool(
            "reference_indicator" in frame
            and pd.to_numeric(
                frame.loc[affected | auxiliary, "reference_indicator"], errors="coerce"
            ).notna().any()
        )
        explicit = _explicit_metadata_for_test(metadata, test_id)
        calibration = _indicator_calibration_parameters(
            explicit, reference_used=reference_used
        )
        if affected.any():
            rows = frame.index[affected].tolist()
            mode = calibration["mode"]
            if not mode:
                issues.append(
                    ValidationIssue(
                        "error",
                        "indicator_mode_not_confirmed",
                        "Показания indicator_* нельзя автоматически считать осадкой: физический режим шкалы не подтверждён.",
                        test_id=test_id,
                        rows=rows,
                        column="indicator_mode",
                        suggested_action="Укажите direct_displacement/cumulative_settlement только для уже преобразованных показаний; развёртка шкалы выполняется в отдельном gate.",
                    )
                )
                continue
            if mode not in _SUPPORTED_PRECALIBRATED_INDICATOR_MODES:
                issues.append(
                    ValidationIssue(
                        "error",
                        "unsupported_indicator_mode",
                        f"Режим indicator_mode={mode!r} ещё не реализован безопасно.",
                        test_id=test_id,
                        rows=rows,
                        column="indicator_mode",
                        raw_value=mode,
                        suggested_action="До TASK 02 используйте прямую settlement либо уже преобразованные cumulative_settlement/direct_displacement.",
                    )
                )
                continue
            has_nested_passport = isinstance(explicit.get("indicator_passport"), dict) or any(
                isinstance(explicit.get(name), dict)
                and any(isinstance(value, dict) for value in explicit[name].values())
                for name in ("indicator_passports", "indicator_channels")
            )
            required_fields = (
                ()
                if has_nested_passport
                else (
                    "indicator_unit",
                    "indicator_calibration_factor",
                    "indicator_sign",
                    "indicator_resolution_mm",
                )
            )
            for field in required_fields:
                value = explicit.get(field)
                if value is None or (isinstance(value, str) and not value.strip()):
                    issues.append(
                        ValidationIssue(
                            "error",
                            "missing_indicator_calibration_metadata",
                            f"Для расчёта из indicator_* нужно явно задать metadata.{field}.",
                            test_id=test_id,
                            rows=rows,
                            column=field,
                            suggested_action="Заполните паспорт калибровки; научные defaults не подставляются.",
                        )
                    )
            if not calibration["instrument_ids"]:
                issues.append(
                    ValidationIssue(
                        "error",
                        "missing_indicator_calibration_metadata",
                        "Не указан идентификатор индикатора в indicator_instrument_id или паспорте приборов.",
                        test_id=test_id,
                        rows=rows,
                        column="indicator_instrument_id",
                        suggested_action="Свяжите показания с instrument_id и записью о калибровке.",
                    )
                )
            if explicit.get("indicator_unit") is not None and calibration["scale_to_mm"] is None:
                issues.append(
                    ValidationIssue(
                        "error",
                        "unsupported_indicator_unit",
                        f"Неподдерживаемая единица индикатора: {explicit.get('indicator_unit')!r}.",
                        test_id=test_id,
                        rows=rows,
                        column="indicator_unit",
                        raw_value=explicit.get("indicator_unit"),
                    )
                )
            factor = calibration["factor"]
            if explicit.get("indicator_calibration_factor") is not None and not (
                factor is not None and math.isfinite(factor) and factor > 0
            ):
                issues.append(
                    ValidationIssue(
                        "error",
                        "invalid_indicator_calibration_factor",
                        "indicator_calibration_factor должен быть конечным положительным числом.",
                        test_id=test_id,
                        rows=rows,
                        column="indicator_calibration_factor",
                        raw_value=explicit.get("indicator_calibration_factor"),
                    )
                )
            sign = calibration["sign"]
            if explicit.get("indicator_sign") is not None and not (
                sign is not None and math.isfinite(sign) and math.isclose(abs(sign), 1.0)
            ):
                issues.append(
                    ValidationIssue(
                        "error",
                        "invalid_indicator_sign",
                        "indicator_sign должен быть равен +1 или -1.",
                        test_id=test_id,
                        rows=rows,
                        column="indicator_sign",
                        raw_value=explicit.get("indicator_sign"),
                    )
                )
            resolution = calibration["resolution_mm"]
            if explicit.get("indicator_resolution_mm") is not None and not (
                resolution is not None and math.isfinite(resolution) and resolution > 0
            ):
                issues.append(
                    ValidationIssue(
                        "error",
                        "invalid_resolution_metadata",
                        "indicator_resolution_mm должен быть конечным положительным числом для расчёта по индикаторам.",
                        test_id=test_id,
                        rows=rows,
                        column="indicator_resolution_mm",
                        raw_value=explicit.get("indicator_resolution_mm"),
                    )
                )
            if reference_used:
                reference_has_passport = any(
                    isinstance(explicit.get(name), dict)
                    and isinstance(explicit[name].get("reference_indicator"), dict)
                    for name in ("indicator_passports", "indicator_channels")
                )
                if reference_has_passport:
                    pass
                else:
                    issues.append(
                        ValidationIssue(
                            "error",
                            "missing_reference_indicator_passport",
                            "Для используемого reference_indicator нужен собственный явный поканальный паспорт.",
                            test_id=test_id,
                            rows=rows,
                            column="reference_indicator",
                        )
                    )
        elif auxiliary.any() and not calibration["confirmed"]:
            issues.append(
                ValidationIssue(
                    "warning",
                    "uncalibrated_indicators_ignored",
                    "Прямая settlement использована без подмены, а неподтверждённые indicator_* исключены из расчёта центра и крена.",
                    test_id=test_id,
                    rows=frame.index[auxiliary].tolist(),
                    suggested_action="Для функций индикаторов явно задайте mode, unit, calibration factor/sign, resolution и instrument_id.",
                )
            )
    if indicator_columns and "reference_indicator" in frame:
        reference_missing = pd.to_numeric(frame["reference_indicator"], errors="coerce").isna()
        affected = reference_missing & supplied_missing & frame[indicator_columns].notna().any(axis=1)
        if affected.any():
            issues.append(
                ValidationIssue(
                    "warning",
                    "missing_reference_indicator",
                    "Reference indicator пропущен: осадка по индикаторам оставлена NaN, нуль не подставлялся.",
                    rows=frame.index[affected].tolist(),
                )
            )
    load_num = pd.to_numeric(frame["load"], errors="coerce")
    bad_load = frame.index[~np.isfinite(load_num)].tolist()
    if bad_load:
        issues.append(
            ValidationIssue("error", "non_numeric_load", "Нагрузка должна быть числом.", rows=bad_load)
        )
    if frame["test_id"].isna().any():
        issues.append(
            ValidationIssue(
                "error",
                "missing_test_id",
                "У части строк отсутствует test_id.",
                rows=frame.index[frame["test_id"].isna()].tolist(),
            )
        )
    duplicates = frame.duplicated(subset=["test_id", "stage"], keep=False)
    if duplicates.any():
        issues.append(
            ValidationIssue(
                "info",
                "repeated_stage",
                "Повторяющиеся stage сохранены: это допустимо для выдержки, порядок строк не изменяется.",
                rows=frame.index[duplicates].tolist(),
            )
        )
    if "branch" in frame:
        values = set(frame["branch"].dropna().astype(str).str.casefold())
        unknown = sorted(values.difference(BRANCHES))
        if unknown:
            issues.append(
                ValidationIssue(
                    "warning",
                    "unknown_branch",
                    "Неизвестные ветви будут заменены предложенной классификацией: " + ", ".join(unknown),
                )
            )

    indicator_preview_frame = frame.reset_index(drop=True)
    indicator_result = process_indicator_frame(indicator_preview_frame, metadata)
    failure_mask = _failure_mask(frame)
    failure_preview_mask = _failure_mask(indicator_preview_frame)
    if failure_preview_mask.any():
        settlement = pd.Series(np.nan, index=indicator_preview_frame.index, dtype=float)
        for test_id, part in indicator_preview_frame.groupby("test_id", sort=False):
            meta = metadata_for_test(metadata, str(test_id))
            supplied = (
                pd.to_numeric(part["settlement"], errors="coerce")
                * _settlement_factor(meta)
                if "settlement" in part
                else pd.Series(np.nan, index=part.index, dtype=float)
            )
            derived = pd.Series(
                {
                    int(index): indicator_result.settlement_by_row.get(int(index))
                    for index in part.index
                },
                index=part.index,
                dtype="float64",
            )
            settlement.loc[part.index] = supplied.where(supplied.notna(), derived)
        no_s = failure_preview_mask & settlement.isna()
        if no_s.any():
            issues.append(
                ValidationIssue(
                    "info",
                    "failure_without_settlement",
                    "Разрушение без осадки будет показано событием, без фиктивной точки.",
                    rows=[frame.index[int(position)] for position in no_s[no_s].index],
                )
            )
    if "status" in frame:
        unknown_status = ~failure_mask & ~_stable_status_mask(frame)
        if unknown_status.any():
            issues.append(
                ValidationIssue(
                    "warning",
                    "unaccepted_status",
                    "Точки с неподтверждённым status сохранены в raw, но исключены из pcr/E.",
                    rows=frame.index[unknown_status].tolist(),
                )
            )

    for test_id in frame["test_id"].dropna().astype(str).unique():
        test_meta = metadata_for_test(metadata, test_id)
        for name in ("lever_ratio", "load_factor", "indicator_sign", "reference_sign"):
            try:
                value = float(test_meta.get(name))
                valid_number = math.isfinite(value) and (
                    value > 0 if name in {"lever_ratio", "load_factor"} else True
                )
            except (TypeError, ValueError):
                valid_number = False
            if not valid_number:
                issues.append(
                    ValidationIssue(
                        "error",
                        "invalid_metadata_number",
                        f"Metadata.{name} должно быть конечным числом"
                        + (" > 0." if name in {"lever_ratio", "load_factor"} else "."),
                        test_id=test_id,
                    )
                )
        try:
            load_zero = float(test_meta.get("load_zero", 0.0))
            valid_load_zero = math.isfinite(load_zero)
        except (TypeError, ValueError):
            valid_load_zero = False
        if not valid_load_zero:
            issues.append(
                ValidationIssue(
                    "error",
                    "invalid_load_zero",
                    "Metadata.load_zero должно быть конечным числом.",
                    test_id=test_id,
                )
            )
        for name in (
            "stamp_diameter_mm",
            "stamp_area_m2",
            "shape_factor",
            "gamma_kN_m3",
            "pu_kPa_confirmed",
        ):
            optional = test_meta.get(name)
            if optional is None:
                continue
            try:
                valid_optional = math.isfinite(float(optional)) and float(optional) > 0
            except (TypeError, ValueError):
                valid_optional = False
            if not valid_optional:
                issues.append(
                    ValidationIssue(
                        "error",
                        "invalid_positive_metadata",
                        f"Metadata.{name} должно быть конечным положительным числом.",
                        test_id=test_id,
                    )
                )
        for name in (
            "load_resolution",
            "load_resolution_kN",
            "pressure_resolution_kPa",
            "indicator_resolution_mm",
        ):
            optional = test_meta.get(name)
            if optional is None:
                continue
            try:
                valid_optional = math.isfinite(float(optional)) and float(optional) >= 0
            except (TypeError, ValueError):
                valid_optional = False
            if not valid_optional:
                issues.append(
                    ValidationIssue(
                        "error",
                        "invalid_resolution_metadata",
                        f"Metadata.{name} должно быть конечным неотрицательным числом.",
                        test_id=test_id,
                    )
                )
        try:
            poisson = float(test_meta.get("poisson_ratio"))
            valid_poisson = math.isfinite(poisson) and 0 <= poisson < 0.5
        except (TypeError, ValueError):
            valid_poisson = False
        if not valid_poisson:
            issues.append(
                ValidationIssue(
                    "error",
                    "invalid_poisson_ratio",
                    "Metadata.poisson_ratio должно быть в диапазоне [0; 0,5).",
                    test_id=test_id,
                )
            )
        reinforcement = test_meta.get("reinforcement")
        if reinforcement is not None and not isinstance(reinforcement, dict):
            issues.append(
                ValidationIssue(
                    "error",
                    "invalid_reinforcement_metadata",
                    "Metadata.reinforcement должно быть объектом.",
                    test_id=test_id,
                )
            )
        try:
            load_kind, _ = _load_kind_and_factor(test_meta)
        except ValueError as exc:
            load_kind = None
            issues.append(ValidationIssue("error", "unsupported_load_unit", str(exc), test_id=test_id))
        try:
            _settlement_factor(test_meta)
        except ValueError as exc:
            issues.append(
                ValidationIssue("error", "unsupported_settlement_unit", str(exc), test_id=test_id)
            )
        area = stamp_area_m2(test_meta)
        if area is None:
            if load_kind == "pressure":
                code = "missing_stamp_area_for_force"
                message = "Площадь штампа не задана: p доступно напрямую, но F и E_stamp_app недоступны."
            else:
                code = "missing_stamp_area"
                message = "Не задана корректная площадь или геометрия штампа; давление p недоступно."
            issues.append(
                ValidationIssue(
                    "warning",
                    code,
                    message,
                    test_id=test_id,
                )
            )
        explicit_area = test_meta.get("stamp_area_m2")
        diameter = test_meta.get("stamp_diameter_mm")
        shape = str(test_meta.get("stamp_shape", "circle")).casefold()
        if explicit_area is not None and diameter is not None and shape in {"circle", "round", "круг", "круглый"}:
            try:
                geometric = math.pi * (float(diameter) / 1000.0) ** 2 / 4.0
                inconsistent = geometric <= 0 or not math.isclose(
                    float(explicit_area), geometric, rel_tol=0.02
                )
            except (TypeError, ValueError):
                inconsistent = True
            if inconsistent:
                issues.append(
                    ValidationIssue(
                        "error",
                        "inconsistent_stamp_geometry",
                        "Площадь штампа противоречит указанному диаметру более чем на 2%.",
                        test_id=test_id,
                    )
                )
    for indicator_issue in indicator_result.issues:
        duplicate = any(
            item.code == indicator_issue.code
            and str(item.test_id or "") == str(indicator_issue.test_id or "")
            and str(item.column or "") == str(indicator_issue.column or "")
            and sorted(item.rows) == sorted(indicator_issue.rows)
            for item in issues
        )
        if not duplicate:
            issues.append(indicator_issue)
    return issues


def _indicator_columns(frame: pd.DataFrame) -> list[str]:
    return [f"indicator_{index}" for index in range(1, 5) if f"indicator_{index}" in frame]


def classify_branches(
    frame: pd.DataFrame, *, load_column: str = "F_kN", tolerance: float | None = None
) -> pd.Series:
    """Suggest branches in source order; never sorts or rewrites load values."""

    if load_column not in frame:
        load_column = "load"
    result = pd.Series(index=frame.index, dtype="object")
    for _, group in frame.groupby("test_id", sort=False):
        local_load_column = load_column if load_column in group else "load"
        selected_load = pd.to_numeric(group[local_load_column], errors="coerce")
        if not np.isfinite(selected_load).any() and "p_kPa" in group:
            local_load_column = "p_kPa"
            resolution_column = "pressure_resolution_kPa"
        else:
            resolution_column = "load_resolution_kN"
        if tolerance is None and resolution_column in group:
            resolution = pd.to_numeric(group[resolution_column], errors="coerce").dropna()
            local_tolerance = max(float(resolution.median()) / 2.0, 1e-12) if len(resolution) else 1e-9
        else:
            local_tolerance = float(tolerance if tolerance is not None else 1e-9)
        loads = pd.to_numeric(group[local_load_column], errors="coerce").to_numpy(dtype=float)
        labels: list[str] = []
        has_unloaded = False
        for i, current in enumerate(loads):
            if i == 0 or not np.isfinite(current) or not np.isfinite(loads[i - 1]):
                labels.append("loading")
                continue
            delta = current - loads[i - 1]
            if abs(delta) <= local_tolerance:
                labels.append("hold")
            elif delta < 0:
                labels.append("unloading")
                has_unloaded = True
            else:
                labels.append("reloading" if has_unloaded else "loading")
        result.loc[group.index] = labels
    return result


def prepare_measurements(
    frame: pd.DataFrame,
    metadata: dict[str, Any] | None = None,
    *,
    strict_metadata: bool = True,
) -> tuple[pd.DataFrame, list[ValidationIssue]]:
    """Create calibrated columns while retaining every source row and its order.

    New/public calls are default-safe and require explicit physical metadata.
    Pipelines that already ran ``validate_project_metadata`` may opt out to
    avoid duplicate diagnostics.
    """

    project_issues: list[ValidationIssue] = []
    if strict_metadata:
        from .provenance import validate_project_metadata

        project_issues = validate_project_metadata(
            metadata if isinstance(metadata, dict) else {}, strict=True
        )
    issues = [*project_issues, *validate_measurements(frame, metadata)]
    if any(bool(item.blocks_processing) for item in issues):
        return frame.copy(deep=True), issues
    result = frame.copy(deep=True).reset_index(drop=True)
    if "source_row" not in result:
        result["source_row"] = range(len(result))
    if "sequence_index" not in result:
        result["sequence_index"] = range(len(result))
    if "source_load_unit" not in result:
        result["source_load_unit"] = result["load_unit"] if "load_unit" in result else None
    result["sequence_no"] = result.groupby("test_id", sort=False).cumcount()
    result["test_id"] = result["test_id"].astype(str)
    result["load"] = pd.to_numeric(result["load"], errors="coerce")
    indicator_processing = process_indicator_frame(result, metadata)

    raw_parts: list[pd.Series] = []
    force_parts: list[pd.Series] = []
    pressure_parts: list[pd.Series] = []
    diameter_parts: list[pd.Series] = []
    area_parts: list[pd.Series] = []
    gamma_parts: list[pd.Series] = []
    resolution_parts: list[pd.Series] = []
    pressure_resolution_parts: list[pd.Series] = []
    load_kind_parts: list[pd.Series] = []
    load_unit_parts: list[pd.Series] = []
    load_unit_factor_parts: list[pd.Series] = []
    load_factor_parts: list[pd.Series] = []
    load_zero_parts: list[pd.Series] = []
    lever_ratio_parts: list[pd.Series] = []
    effective_load_coefficient_parts: list[pd.Series] = []
    pu_parts: list[pd.Series] = []
    indicator_sign_parts: list[pd.Series] = []
    reference_sign_parts: list[pd.Series] = []
    settlement_scale_parts: list[pd.Series] = []
    indicator_mode_parts: list[pd.Series] = []
    indicator_unit_parts: list[pd.Series] = []
    indicator_scale_parts: list[pd.Series] = []
    indicator_calibration_factor_parts: list[pd.Series] = []
    indicator_calibration_confirmed_parts: list[pd.Series] = []
    reference_channel_used_parts: list[pd.Series] = []
    indicator_instrument_parts: list[pd.Series] = []
    indicator_resolution_parts: list[pd.Series] = []
    group_parts: list[pd.Series] = []
    pair_parts: list[pd.Series] = []
    for test_id, part in result.groupby("test_id", sort=False):
        meta = metadata_for_test(metadata, test_id)
        explicit = _explicit_metadata_for_test(metadata, test_id)
        reference_used = bool(
            "reference_indicator" in part
            and pd.to_numeric(part["reference_indicator"], errors="coerce").notna().any()
        )
        indicator_calibration = _indicator_calibration_parameters(
            explicit, reference_used=reference_used
        )
        aggregation_rows_for_test = [
            row
            for row in indicator_processing.aggregation_rows
            if str(row.get("test_id")) == str(test_id)
        ]
        required_channels: set[str] = set()
        if aggregation_rows_for_test:
            try:
                required_channels = set(
                    json.loads(aggregation_rows_for_test[0]["channels_required"])
                )
            except (TypeError, ValueError, json.JSONDecodeError):
                required_channels = set()
        effective_passports = [
            row
            for row in indicator_processing.passport_rows
            if str(row.get("test_id")) == str(test_id)
            and str(row.get("channel", "")).startswith("indicator_")
            and str(row.get("channel")) != "reference_indicator"
            and str(row.get("channel")) in required_channels
        ]
        if effective_passports:
            modes = {str(row.get("mode")) for row in effective_passports}
            factors = {
                float(row["correction_factor"])
                for row in effective_passports
                if row.get("correction_factor") is not None
            }
            signs = {
                (
                    -1.0
                    if str(row.get("mode", "")).startswith("decreasing")
                    else float(row.get("cumulative_sign", 1.0))
                )
                for row in effective_passports
            }
            effective_resolutions = [
                float(row["division_mm"]) * float(row["correction_factor"])
                for row in effective_passports
                if row.get("division_mm") is not None
                and row.get("correction_factor") is not None
            ]
            indicator_calibration.update(
                {
                    "confirmed": all(
                        row.get("assignment_status") == "confirmed"
                        and row.get("verification_status")
                        == "valid_at_experiment"
                        for row in effective_passports
                    ),
                    "mode": next(iter(modes)) if len(modes) == 1 else "per_channel",
                    "unit": "mm",
                    "scale_to_mm": 1.0,
                    "factor": next(iter(factors)) if len(factors) == 1 else np.nan,
                    "sign": next(iter(signs)) if len(signs) == 1 else np.nan,
                    "resolution_mm": max(effective_resolutions, default=np.nan),
                    "instrument_ids": list(
                        dict.fromkeys(
                            str(row.get("instrument_id"))
                            for row in effective_passports
                            if row.get("instrument_id")
                        )
                    ),
                }
            )
        else:
            indicator_calibration["confirmed"] = False
        if "settlement" in part:
            supplied_settlement = (
                pd.to_numeric(part["settlement"], errors="coerce")
                * _settlement_factor(meta)
            )
        else:
            supplied_settlement = pd.Series(np.nan, index=part.index, dtype=float)
        derived_settlement = pd.Series(
            {
                int(index): indicator_processing.settlement_by_row.get(int(index))
                for index in part.index
            },
            index=part.index,
            dtype="float64",
        )
        raw = supplied_settlement.where(supplied_settlement.notna(), derived_settlement)
        load_kind, load_factor = _load_kind_and_factor(meta)
        ratio = float(meta.get("lever_ratio", 1.0))
        user_factor = float(meta.get("load_factor", 1.0))
        load_zero = float(meta.get("load_zero", 0.0))
        area = stamp_area_m2(meta)
        if load_kind == "force":
            force = (part["load"] - load_zero) * load_factor * user_factor * ratio
            pressure = force / area if area else pd.Series(np.nan, index=part.index)
            if meta.get("load_resolution") is not None:
                resolution_kn = (
                    float(meta["load_resolution"]) * load_factor * user_factor * ratio
                )
            else:
                resolution_kn = float(meta.get("load_resolution_kN", 0.0) or 0.0)
            pressure_resolution = resolution_kn / area if area else 0.0
        else:
            pressure = (part["load"] - load_zero) * load_factor * user_factor
            force = pressure * area if area else pd.Series(np.nan, index=part.index)
            if meta.get("pressure_resolution_kPa") is not None:
                pressure_resolution = float(meta["pressure_resolution_kPa"])
            elif meta.get("load_resolution") is not None:
                pressure_resolution = float(meta["load_resolution"]) * load_factor * user_factor
            else:
                pressure_resolution = 0.0
            resolution_kn = pressure_resolution * area if area else 0.0
        diameter = float(meta.get("stamp_diameter_mm") or np.nan)
        default_group = meta.get("group") or meta.get("reinforcement", {}).get("type", "series")
        raw_parts.append(pd.Series(raw.to_numpy(), index=part.index))
        force_parts.append(pd.Series(force.to_numpy(), index=part.index))
        pressure_parts.append(pd.Series(np.asarray(pressure, dtype=float), index=part.index))
        diameter_parts.append(pd.Series(diameter, index=part.index))
        area_parts.append(pd.Series(float(area) if area else np.nan, index=part.index))
        gamma_parts.append(pd.Series(float(meta.get("gamma_kN_m3") or np.nan), index=part.index))
        resolution_parts.append(pd.Series(resolution_kn, index=part.index))
        pressure_resolution_parts.append(pd.Series(pressure_resolution, index=part.index))
        load_kind_parts.append(pd.Series(load_kind, index=part.index))
        load_unit_parts.append(pd.Series(str(meta.get("load_unit")), index=part.index))
        load_unit_factor_parts.append(pd.Series(load_factor, index=part.index))
        load_factor_parts.append(pd.Series(user_factor, index=part.index))
        load_zero_parts.append(pd.Series(load_zero, index=part.index))
        lever_ratio_parts.append(pd.Series(ratio, index=part.index))
        effective_load_coefficient_parts.append(
            pd.Series(
                load_factor * user_factor * (ratio if load_kind == "force" else 1.0),
                index=part.index,
            )
        )
        pu_parts.append(pd.Series(float(meta.get("pu_kPa_confirmed") or np.nan), index=part.index))
        indicator_sign_parts.append(
            pd.Series(
                float(indicator_calibration.get("sign") or meta.get("indicator_sign", 1.0)),
                index=part.index,
            )
        )
        reference_sign_parts.append(
            pd.Series(float(meta.get("reference_sign", -1.0)), index=part.index)
        )
        settlement_scale_parts.append(pd.Series(_settlement_factor(meta), index=part.index))
        indicator_mode_parts.append(
            pd.Series(indicator_calibration["mode"] or None, index=part.index, dtype="object")
        )
        indicator_unit_parts.append(
            pd.Series(indicator_calibration["unit"] or None, index=part.index, dtype="object")
        )
        indicator_scale_parts.append(
            pd.Series(indicator_calibration["scale_to_mm"], index=part.index, dtype="float64")
        )
        indicator_calibration_factor_parts.append(
            pd.Series(indicator_calibration["factor"], index=part.index, dtype="float64")
        )
        indicator_calibration_confirmed_parts.append(
            pd.Series(bool(indicator_calibration["confirmed"]), index=part.index)
        )
        aggregation_for_test = {
            int(row["row_index"]): row
            for row in indicator_processing.aggregation_rows
            if str(row.get("test_id")) == str(test_id)
        }
        reference_channel_used_parts.append(
            pd.Series(
                {
                    int(index): bool(
                        aggregation_for_test.get(int(index), {}).get(
                            "reference_correction_mm"
                        )
                        is not None
                        and aggregation_for_test.get(int(index), {}).get(
                            "aggregation_status"
                        )
                        == "ok"
                    )
                    for index in part.index
                },
                index=part.index,
            )
        )
        indicator_instrument_parts.append(
            pd.Series(
                ",".join(indicator_calibration["instrument_ids"]) or None,
                index=part.index,
                dtype="object",
            )
        )
        indicator_resolution_parts.append(
            pd.Series(
                float(
                    indicator_calibration.get("resolution_mm")
                    or meta.get("indicator_resolution_mm", 0.0)
                    or 0.0
                ),
                index=part.index,
            )
        )
        if "group" in part:
            groups = part["group"].fillna(default_group).astype(str)
        else:
            groups = pd.Series(str(default_group), index=part.index)
        group_parts.append(groups)
        default_pair = meta.get("pair_id")
        if "pair_id" in part:
            pairs = part["pair_id"].where(part["pair_id"].notna(), default_pair)
        else:
            pairs = pd.Series(default_pair, index=part.index)
        pair_parts.append(pairs.astype("string"))

    result["settlement_raw_mm"] = pd.concat(raw_parts).sort_index()
    result["settlement_mm"] = result["settlement_raw_mm"]
    result["correction_mode"] = "raw"
    result["F_kN"] = pd.concat(force_parts).sort_index()
    result["p_kPa"] = pd.concat(pressure_parts).sort_index()
    result["D_mm"] = pd.concat(diameter_parts).sort_index()
    result["stamp_area_m2"] = pd.concat(area_parts).sort_index()
    result["gamma_kN_m3"] = pd.concat(gamma_parts).sort_index()
    result["load_resolution_kN"] = pd.concat(resolution_parts).sort_index()
    result["pressure_resolution_kPa"] = pd.concat(pressure_resolution_parts).sort_index()
    result["load_kind"] = pd.concat(load_kind_parts).sort_index()
    result["load_unit"] = pd.concat(load_unit_parts).sort_index()
    result["load_unit_factor_to_base"] = pd.concat(load_unit_factor_parts).sort_index()
    result["load_factor"] = pd.concat(load_factor_parts).sort_index()
    result["load_zero"] = pd.concat(load_zero_parts).sort_index()
    result["lever_ratio"] = pd.concat(lever_ratio_parts).sort_index()
    result["effective_load_coefficient"] = pd.concat(
        effective_load_coefficient_parts
    ).sort_index()
    result["pu_kPa_confirmed"] = pd.concat(pu_parts).sort_index()
    result["indicator_sign"] = pd.concat(indicator_sign_parts).sort_index()
    result["reference_sign"] = pd.concat(reference_sign_parts).sort_index()
    result["settlement_scale_to_mm"] = pd.concat(settlement_scale_parts).sort_index()
    result["indicator_mode"] = pd.concat(indicator_mode_parts).sort_index()
    result["indicator_unit"] = pd.concat(indicator_unit_parts).sort_index()
    result["indicator_scale_to_mm"] = pd.concat(indicator_scale_parts).sort_index()
    result["indicator_calibration_factor"] = pd.concat(
        indicator_calibration_factor_parts
    ).sort_index()
    result["indicator_calibration_confirmed"] = pd.concat(
        indicator_calibration_confirmed_parts
    ).sort_index()
    result["reference_channel_used"] = pd.concat(
        reference_channel_used_parts
    ).sort_index()
    result["indicator_instrument_id"] = pd.concat(indicator_instrument_parts).sort_index()
    result["indicator_resolution_mm"] = pd.concat(indicator_resolution_parts).sort_index()
    for channel, channel_values in indicator_processing.channel_settlement_by_row.items():
        result[f"{channel}_settlement_mm"] = pd.Series(
            {
                int(index): channel_values.get(int(index))
                for index in result.index
            },
            index=result.index,
            dtype="float64",
        )
    aggregation_by_index = {
        int(row["row_index"]): row for row in indicator_processing.aggregation_rows
    }
    aggregation_columns = (
        "aggregated_settlement_mm",
        "aggregation_method",
        "channels_required",
        "channels_used",
        "missing_channels",
        "aggregation_status",
        "plane_rank",
        "plane_residual_rms_mm",
        "tilt_magnitude_mm_per_mm",
        "tilt_direction_deg",
        "tilt_direction_resolved",
    )
    for column in aggregation_columns:
        result[column] = pd.Series(
            {
                int(index): aggregation_by_index.get(int(index), {}).get(column)
                for index in result.index
            },
            index=result.index,
        )
    result["group"] = pd.concat(group_parts).sort_index()
    result["pair_id"] = pd.concat(pair_parts).sort_index()
    suggested = classify_branches(result)
    if "branch" not in result:
        result["branch"] = suggested
    else:
        supplied = result["branch"].astype("string").str.casefold()
        valid = supplied.isin(BRANCHES)
        result["branch"] = supplied.where(valid, suggested)
    result["branch_suggested"] = suggested
    result["is_failure"] = _failure_mask(result)
    result["is_measured"] = result["settlement_raw_mm"].notna()
    result.attrs["indicator_processing_audit"] = deepcopy(
        indicator_processing.audit_rows
    )
    result.attrs["indicator_processing_events"] = deepcopy(
        indicator_processing.event_rows
    )
    result.attrs["indicator_calibration_parameters"] = deepcopy(
        indicator_processing.passport_rows
    )
    result.attrs["indicator_aggregation_results"] = deepcopy(
        indicator_processing.aggregation_rows
    )
    result.attrs["metrology_evaluations"] = [
        {
            "test_id": row.get("test_id"),
            "channel": row.get("channel"),
            "instrument_id": row.get("instrument_id"),
            "verification_date": row.get("verification_date"),
            "verification_valid_until": row.get(
                "verification_valid_until"
            ),
            "assignment_status": row.get("assignment_status"),
            "verification_status": row.get("verification_status"),
            "verification_evaluation_date": row.get(
                "verification_evaluation_date"
            ),
            "verification_evaluation_date_source": row.get(
                "verification_evaluation_date_source"
            ),
            "verification_evaluation_rule": row.get(
                "verification_evaluation_rule"
            ),
        }
        for row in indicator_processing.passport_rows
    ]
    result.attrs["indicator_processing_schema"] = INDICATOR_PROCESSING_SCHEMA
    return result, issues


def apply_settlement_correction(
    frame: pd.DataFrame,
    mode: str,
    *,
    seating_offsets_mm: dict[str, float] | None = None,
    audit: AuditTrail | None = None,
    reason: str = "Выбран слой данных",
    user: str = "local-user",
) -> tuple[pd.DataFrame, list[ValidationIssue]]:
    """Return a new curve variant; raw measurements are never overwritten."""

    if mode not in CORRECTION_MODES:
        raise ValueError(f"Неизвестный режим коррекции: {mode}")
    if "settlement_raw_mm" not in frame:
        raise ValueError("Сначала вызовите prepare_measurements().")
    before = frame[["test_id", "sequence_no", "settlement_raw_mm", "settlement_mm"]].copy()
    result = frame.copy(deep=True)
    issues: list[ValidationIssue] = []
    result["settlement_mm"] = result["settlement_raw_mm"]
    result["correction_offset_mm"] = 0.0
    if mode == "zero_shifted":
        for test_id, part in result.groupby("test_id", sort=False):
            if np.isfinite(pd.to_numeric(part["F_kN"], errors="coerce")).any():
                zero_values = part["F_kN"]
                resolution_column = "load_resolution_kN"
            else:
                zero_values = part["p_kPa"]
                resolution_column = "pressure_resolution_kPa"
            resolution = pd.to_numeric(part.get(resolution_column), errors="coerce").dropna()
            tolerance = max(float(resolution.median()) / 2.0, 1e-12) if len(resolution) else 1e-9
            force_zero = np.isclose(zero_values, 0.0, atol=tolerance)
            initial_sequence = part["sequence_no"].min() if "sequence_no" in part else part.index.min()
            initial_branch = part.get("branch", pd.Series("loading", index=part.index)).eq("loading")
            initial_row = (
                part["sequence_no"].eq(initial_sequence)
                if "sequence_no" in part
                else pd.Series(part.index == initial_sequence, index=part.index)
            )
            zero = part[
                force_zero
                & part["settlement_raw_mm"].notna()
                & (initial_branch | initial_row)
            ]
            if zero.empty:
                issues.append(
                    ValidationIssue(
                        "warning",
                        "missing_measured_zero",
                        "Нулевая точка не измерена: сдвиг не применен и (0;0) не добавлен.",
                        test_id=str(test_id),
                    )
                )
                continue
            offset = float(zero.iloc[0]["settlement_raw_mm"])
            result.loc[part.index, "settlement_mm"] = part["settlement_raw_mm"] - offset
            result.loc[part.index, "correction_offset_mm"] = offset
    elif mode == "seating_corrected":
        offsets = seating_offsets_mm or {}
        for test_id, part in result.groupby("test_id", sort=False):
            if str(test_id) not in offsets:
                issues.append(
                    ValidationIssue(
                        "warning",
                        "missing_seating_offset",
                        "Посадочная поправка не задана явно; автоматическая поправка не выполнялась.",
                        test_id=str(test_id),
                    )
                )
                continue
            offset = float(offsets[str(test_id)])
            if not np.isfinite(offset):
                raise ValueError(f"Посадочная поправка {test_id} должна быть конечным числом.")
            result.loc[part.index, "settlement_mm"] = part["settlement_raw_mm"] - offset
            result.loc[part.index, "correction_offset_mm"] = offset
    result["correction_mode"] = mode
    if audit is not None:
        audit.record(
            "apply_curve_variant",
            scope=",".join(result["test_id"].unique()),
            reason=reason,
            parameters={"mode": mode, "seating_offsets_mm": seating_offsets_mm or {}},
            before=before,
            after=result[["test_id", "sequence_no", "settlement_raw_mm", "settlement_mm"]],
            user=user,
            method=mode,
        )
    return result, issues


def apply_manual_point_correction(
    frame: pd.DataFrame,
    *,
    test_id: str,
    sequence_no: int,
    corrected_settlement_mm: float,
    reason: str,
    audit: AuditTrail,
    user: str = "local-user",
) -> pd.DataFrame:
    corrected_value = float(corrected_settlement_mm)
    if not np.isfinite(corrected_value):
        raise ValueError("Ручная осадка должна быть конечным числом.")
    result = frame.copy(deep=True)
    mask = (result["test_id"].astype(str) == str(test_id)) & (result["sequence_no"] == sequence_no)
    if mask.sum() != 1:
        raise ValueError("Точка для ручной коррекции не найдена или неоднозначна.")
    previous = float(result.loc[mask, "settlement_mm"].iloc[0])
    if not np.isfinite(previous):
        raise ValueError("Ручная коррекция не должна создавать осадку там, где измерение отсутствует.")
    result.loc[mask, "settlement_mm"] = corrected_value
    result.loc[mask, "correction_mode"] = "manual"
    audit.record(
        "manual_point_correction",
        scope=f"{test_id}:{sequence_no}",
        reason=reason,
        parameters={"old_mm": previous, "new_mm": corrected_value},
        before={"settlement_mm": previous},
        after={"settlement_mm": corrected_value},
        user=user,
        method="manual",
    )
    return result


def _failure_mask(frame: pd.DataFrame) -> pd.Series:
    if "status" not in frame:
        return pd.Series(False, index=frame.index)
    text = (
        frame["status"]
        .fillna("")
        .astype(str)
        .str.casefold()
        .str.replace(r"[_-]+", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    controlled = {
        *(str(word).casefold() for word in FAILURE_WORDS),
        "failure confirmed",
        "failure step",
        "confirmed failure",
        "разрушение подтверждено",
        "ступень разрушения",
    }
    return text.isin(controlled)


def _stable_status_mask(frame: pd.DataFrame) -> pd.Series:
    if "status" not in frame:
        return pd.Series(True, index=frame.index)
    text = frame["status"].fillna("").astype(str).str.casefold()
    invalid = pd.Series(False, index=frame.index)
    for phrase in (
        "unstable",
        "not stable",
        "invalid",
        "excluded",
        "нестабил",
        "не стаб",
        "неустойч",
        "брак",
    ):
        invalid |= text.str.contains(phrase, regex=False)
    accepted = text.str.strip().eq("")
    for phrase in (
        "stable",
        "stabilized",
        "accepted",
        "ok",
        "устойчив",
        "стабил",
        "норма",
        "outlier",
        "выброс",
        "no failure",
        "not failed",
        "failure not",
        "не достиг",
    ):
        accepted |= text.str.contains(phrase, regex=False)
    return accepted & ~invalid & ~_failure_mask(frame)


FAILURE_ANALYSIS_CONTRACT_VERSION = "failure-analysis/1.0"


def _finite_optional_float(value: Any) -> float | None:
    """Return a finite float without converting a missing bound into a value."""

    if value is None or pd.isna(value):
        return None
    number = float(value)
    return number if np.isfinite(number) else None


def _failure_sequence_value(row: pd.Series | None) -> int | float | str | None:
    if row is None or "sequence_no" not in row or pd.isna(row.get("sequence_no")):
        return None
    value = row.get("sequence_no")
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return int(number) if number.is_integer() else number
    return str(value)


def _failure_capacity_fields(
    payload: dict[str, Any],
    *,
    load_kind: str | None = None,
) -> tuple[str, float | None, float | None, str | None]:
    """Select the available physical coordinate without inventing conversions."""

    force_lower = _finite_optional_float(payload.get("Fu_lower"))
    force_upper = _finite_optional_float(payload.get("Fu_upper"))
    pressure_lower = _finite_optional_float(payload.get("pu_lower"))
    pressure_upper = _finite_optional_float(payload.get("pu_upper"))
    if load_kind == "pressure":
        return "pressure", pressure_lower, pressure_upper, "kPa"
    if load_kind == "force":
        return "force", force_lower, force_upper, "kN"
    if force_lower is not None or force_upper is not None:
        return "force", force_lower, force_upper, "kN"
    if pressure_lower is not None or pressure_upper is not None:
        return "pressure", pressure_lower, pressure_upper, "kPa"
    return "unknown", None, None, None


def failure_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Summarize failure events and censoring without fabricating point estimates.

    ``failure_observed`` records whether a failure event was documented.  It is
    deliberately orthogonal to ``censoring_type``: in a step-loading protocol a
    documented failure normally has an interval-censored capacity.
    """

    required = {"test_id", "F_kN", "p_kPa"}
    if not required.issubset(frame):
        raise ValueError(f"Не хватает столбцов: {sorted(required.difference(frame.columns))}")
    rows: list[dict[str, Any]] = []
    for test_id, part in frame.groupby("test_id", sort=False):
        ordered = part.sort_values("sequence_no", kind="stable") if "sequence_no" in part else part
        load_kinds = (
            ordered["load_kind"].dropna().astype(str).str.strip().str.casefold().unique().tolist()
            if "load_kind" in ordered
            else []
        )
        declared_load_kind = (
            load_kinds[0]
            if len(load_kinds) == 1 and load_kinds[0] in {"force", "pressure"}
            else None
        )
        fail_mask = (
            ordered["is_failure"].astype(bool)
            if "is_failure" in ordered
            else _failure_mask(ordered)
        )
        failure_event_count = int(fail_mask.sum())
        failure_row: pd.Series | None = None
        last: pd.Series | None = None
        lower_bound_row: pd.Series | None = None
        if fail_mask.any():
            failure_pos = int(np.flatnonzero(fail_mask.to_numpy())[0])
            failure_row = ordered.iloc[failure_pos]
            before = ordered.iloc[:failure_pos]
            stable = before[
                (before["F_kN"].notna() | before["p_kPa"].notna()) & _stable_status_mask(before)
            ]
            last = stable.iloc[-1] if not stable.empty else None
            f_failure = float(failure_row["F_kN"]) if pd.notna(failure_row["F_kN"]) else None
            f_stable = float(last["F_kN"]) if last is not None and pd.notna(last["F_kN"]) else None
            p_failure = float(failure_row["p_kPa"]) if pd.notna(failure_row["p_kPa"]) else None
            p_stable = (
                float(last["p_kPa"]) if last is not None and pd.notna(last["p_kPa"]) else None
            )
            s_stable = (
                float(last["settlement_mm"])
                if last is not None and "settlement_mm" in last and pd.notna(last["settlement_mm"])
                else None
            )
            s_failure = (
                float(failure_row["settlement_mm"])
                if "settlement_mm" in failure_row and pd.notna(failure_row["settlement_mm"])
                else None
            )
            lower_bound_row = last
            result = FailureResult(
                test_id=str(test_id),
                failure_reached=True,
                right_censored=False,
                F_last_stable=f_stable,
                F_failure_step=f_failure,
                Fu_lower=f_stable,
                Fu_upper=f_failure,
                p_last_stable=p_stable,
                p_failure_step=p_failure,
                pu_lower=p_stable,
                pu_upper=p_failure,
                s_last_stable=s_stable,
                s_failure=s_failure,
                display=(
                    f"{f_stable:g} < Fu ≤ {f_failure:g} кН"
                    if f_stable is not None and f_failure is not None
                    else f"{p_stable:g} < pu ≤ {p_failure:g} кПа"
                    if p_stable is not None and p_failure is not None
                    else "Разрушение зафиксировано; интервал неполон"
                ),
            )
        else:
            stable = ordered[
                _stable_status_mask(ordered) & (ordered["F_kN"].notna() | ordered["p_kPa"].notna())
            ]
            fmax_raw = pd.to_numeric(stable["F_kN"], errors="coerce").max()
            fmax = float(fmax_raw) if pd.notna(fmax_raw) else None
            pmax_raw = pd.to_numeric(stable["p_kPa"], errors="coerce").max()
            pmax = float(pmax_raw) if pd.notna(pmax_raw) else None
            if fmax is not None:
                last_at_max = stable[np.isclose(stable["F_kN"], fmax, rtol=0, atol=1e-12)]
            elif pmax is not None:
                last_at_max = stable[np.isclose(stable["p_kPa"], pmax, rtol=0, atol=1e-12)]
            else:
                last_at_max = stable.iloc[0:0]
            lower_bound_row = last_at_max.iloc[-1] if not last_at_max.empty else None
            last = lower_bound_row
            s_at_max = (
                float(lower_bound_row["settlement_mm"])
                if lower_bound_row is not None
                and "settlement_mm" in lower_bound_row
                and pd.notna(lower_bound_row["settlement_mm"])
                else None
            )
            has_valid_lower_bound = fmax is not None or pmax is not None
            result = FailureResult(
                test_id=str(test_id),
                failure_reached=False,
                right_censored=has_valid_lower_bound,
                F_last_stable=fmax,
                F_failure_step=None,
                Fu_lower=fmax,
                Fu_upper=None,
                p_last_stable=pmax,
                p_failure_step=None,
                pu_lower=pmax,
                pu_upper=None,
                s_last_stable=s_at_max,
                s_failure=None,
                display=(
                    f"Fu > {fmax:g} кН"
                    if fmax is not None
                    else f"pu > {pmax:g} кПа"
                    if pmax is not None
                    else "Предельная нагрузка не определена"
                ),
            )
        payload = asdict(result)
        capacity_kind, capacity_lower, capacity_upper, capacity_unit = _failure_capacity_fields(
            payload, load_kind=declared_load_kind
        )
        warnings: list[str] = []
        if len(load_kinds) > 1:
            warnings.append("inconsistent_load_kind")
        if failure_event_count > 1:
            warnings.append(f"multiple_failure_events:{failure_event_count};first_sequence_used")
        failure_observed = failure_event_count > 0
        valid_interval = (
            failure_observed
            and capacity_lower is not None
            and capacity_upper is not None
            and capacity_lower < capacity_upper
        )
        if valid_interval:
            censoring_type = "interval_censored"
        elif not failure_observed and capacity_lower is not None:
            censoring_type = "right_censored"
        else:
            censoring_type = "indeterminate"
            if failure_observed:
                if capacity_lower is None or capacity_upper is None:
                    warnings.append("missing_interval_bound")
                else:
                    warnings.append("invalid_interval_bounds")
            else:
                warnings.append("missing_valid_lower_bound")
        classification_status = "review_required" if warnings else "ok"
        if censoring_type == "indeterminate":
            payload["display"] = (
                "Разрушение зафиксировано; границы требуют проверки"
                if failure_observed
                else "Предельная нагрузка не определена; требуется проверка"
            )
        elif censoring_type == "interval_censored":
            symbol = "Fu" if capacity_kind == "force" else "pu"
            display_unit = "кН" if capacity_kind == "force" else "кПа"
            payload["display"] = (
                f"{capacity_lower:g} < {symbol} ≤ {capacity_upper:g} {display_unit}"
            )
        else:
            symbol = "Fu" if capacity_kind == "force" else "pu"
            display_unit = "кН" if capacity_kind == "force" else "кПа"
            payload["display"] = f"{symbol} > {capacity_lower:g} {display_unit}"
        classification_warning = ";".join(warnings) if warnings else None
        payload.update(
            {
                "failure_observed": failure_observed,
                "interval_censored": censoring_type == "interval_censored",
                "right_censored": censoring_type == "right_censored",
                "censoring_type": censoring_type,
                "classification_status": classification_status,
                "classification_warning": classification_warning,
                "capacity_kind": capacity_kind,
                "lower_bound": capacity_lower,
                "upper_bound": capacity_upper,
                "capacity_lower": capacity_lower,
                "capacity_upper": capacity_upper,
                "capacity_unit": capacity_unit,
                "lower_inclusive": False,
                "upper_inclusive": (
                    True if censoring_type == "interval_censored" else None
                ),
                "warning": classification_warning,
                "capacity_lower_inclusive": False if capacity_lower is not None else None,
                "capacity_upper_inclusive": (
                    True if censoring_type == "interval_censored" else None
                ),
                "failure_event_count": failure_event_count,
                "failure_sequence_no": _failure_sequence_value(failure_row),
                "last_stable_sequence_no": _failure_sequence_value(last),
                "lower_bound_sequence_no": _failure_sequence_value(lower_bound_row),
                "upper_bound_sequence_no": _failure_sequence_value(failure_row),
            }
        )
        rows.append(payload)
    rows.sort(key=lambda row: str(row["test_id"]))
    return pd.DataFrame(rows)


def failure_analysis_contract() -> dict[str, Any]:
    """Return the versioned, deliberately conservative failure-analysis contract."""

    return {
        "contract_version": FAILURE_ANALYSIS_CONTRACT_VERSION,
        "default_summary_method": "none",
        "supported_summary_methods": ["none"],
        "supported_capacity_axes": ["auto", "force", "pressure"],
        "point_estimate_policy": (
            "No aggregate point estimate is produced unless a separately versioned "
            "censored-data method is explicitly supported and selected."
        ),
    }


def failure_analysis_summary(
    failures: pd.DataFrame,
    *,
    summary_method: str = "none",
    capacity_axis: str = "auto",
) -> dict[str, Any]:
    """Count failure/censoring states under ``failure-analysis/1.0``.

    The current contract intentionally supports no aggregate estimator.  This
    prevents interval bounds and right-censored lower bounds from being reduced
    to a hidden arithmetic mean.
    """

    method = str(summary_method).strip()
    if method != "none":
        raise ValueError(
            "Неподдерживаемый метод сводного анализа разрушения: "
            f"{summary_method!r}. В failure-analysis/1.0 доступен только 'none'."
        )
    axis = str(capacity_axis).strip()
    if axis not in {"auto", "force", "pressure"}:
        raise ValueError("capacity_axis должен быть одним из: auto, force, pressure.")
    if "censoring_type" not in failures:
        raise ValueError("Сначала сформируйте таблицу через failure_summary().")
    censoring = failures["censoring_type"].fillna("indeterminate").astype(str)
    observed_source = failures.get(
        "failure_observed", failures.get("failure_reached", pd.Series(False, index=failures.index))
    )
    observed = observed_source.fillna(False).astype(bool)
    review = (
        failures.get("classification_status", pd.Series("review_required", index=failures.index))
        .fillna("review_required")
        .astype(str)
    )
    return {
        "contract_version": FAILURE_ANALYSIS_CONTRACT_VERSION,
        "summary_method": method,
        "capacity_axis": axis,
        "capacity_unit": {"force": "kN", "pressure": "kPa"}.get(axis),
        "analysis_status": "descriptive_only_no_point_estimate",
        "point_estimate": None,
        "point_estimate_unit": None,
        "n_tests": int(len(failures)),
        "n_failure_observed": int(observed.sum()),
        "n_interval_censored": int(censoring.eq("interval_censored").sum()),
        "n_right_censored": int(censoring.eq("right_censored").sum()),
        "n_indeterminate": int(censoring.eq("indeterminate").sum()),
        "n_review_required": int(review.eq("review_required").sum()),
    }
