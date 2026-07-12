"""Validation rules for lossless manual plate-load drafts.

This module validates primary input only.  It deliberately performs no load,
pressure, settlement, Antonov-curve or failure-model calculations.  Every
diagnostic points back to a stable ``manual_row_uuid`` and the editor row.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Iterable
from uuid import UUID

import numpy as np
import pandas as pd

from .indicators import canonical_indicator_mode
from .io import parse_decimal
from .manual_entry_models import (
    MANUAL_BRANCHES,
    MANUAL_PROTOCOL_TYPES,
    MANUAL_ROW_STATUSES,
    MANUAL_TEST_SCOPES,
    ManualDraft,
    ManualPoint,
)
from .schema import ValidationIssue


_LOAD_UNITS = {
    "force": {"n", "н", "kn", "кн", "mn", "мн", "kgf", "кгс", "tf", "тс"},
    "pressure": {"pa", "па", "kpa", "кпа", "mpa", "мпа"},
}
_SETTLEMENT_UNITS = {"mm", "мм", "cm", "см", "m", "м"}
_STAMP_SHAPES = {"circle", "round", "круг", "круглый", "custom"}
_TERMINAL_STATUSES = {"failure", "instrument_limit", "stopped_without_failure"}
_INDICATOR_CHANNELS = tuple(f"indicator_{index}" for index in range(1, 5))
_ALL_METROLOGY_CHANNELS = (*_INDICATOR_CHANNELS, "reference_indicator")
_AGGREGATION_METHODS = {
    "all_channels_mean",
    "selected_channels_mean",
    "plane_center",
    "primary_channel",
    "no_aggregation",
}


@dataclass(slots=True)
class ManualValidationResult:
    """Structured validation outcome consumed by GUI and adapters."""

    issues: list[ValidationIssue] = field(default_factory=list)
    adapter_issues: list[ValidationIssue] = field(default_factory=list)
    pipeline_issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def blocking_issues(self) -> list[ValidationIssue]:
        return [item for item in self.issues if bool(item.blocks_processing)]

    @property
    def blocking(self) -> bool:
        return bool(self.blocking_issues)

    @property
    def can_analyze(self) -> bool:
        return not self.blocking

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(item.to_dict() for item in self.issues)


def _blank(value: Any) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def _finite(value: Any) -> float | None:
    return parse_decimal(value)


def _issue_key(issue: ValidationIssue) -> tuple[Any, ...]:
    return (
        issue.level,
        issue.code,
        issue.test_id,
        tuple(issue.rows),
        issue.row,
        issue.column,
        issue.entity_id,
        repr(issue.raw_value),
    )


def merge_manual_issues(*collections: Iterable[ValidationIssue]) -> list[ValidationIssue]:
    """Keep issue order while removing exact adapter/pipeline duplicates."""

    result: list[ValidationIssue] = []
    seen: set[tuple[Any, ...]] = set()
    for collection in collections:
        for issue in collection:
            key = _issue_key(issue)
            if key in seen:
                continue
            seen.add(key)
            result.append(issue)
    return result


def _passport_issue(
    draft: ManualDraft,
    code: str,
    message: str,
    *,
    column: str,
    raw_value: Any = None,
    level: str = "error",
) -> ValidationIssue:
    return ValidationIssue(
        level,
        code,
        message,
        test_id=draft.passport.test_id or None,
        column=column,
        raw_value=raw_value,
        entity_id=draft.draft_id or None,
        suggested_action="Исправьте поле паспорта ручного испытания.",
    )


def _row_issue(
    draft: ManualDraft,
    point: ManualPoint,
    position: int,
    code: str,
    message: str,
    *,
    column: str,
    raw_value: Any = None,
    level: str = "error",
) -> ValidationIssue:
    return ValidationIssue(
        level,
        code,
        message,
        test_id=draft.passport.test_id or None,
        rows=[position],
        row=position + 1,
        column=column,
        raw_value=raw_value,
        entity_id=point.manual_row_uuid or None,
        suggested_action=f"Перейдите к строке {position + 1} и исправьте {column}.",
    )


def _require_text(draft: ManualDraft, issues: list[ValidationIssue], name: str) -> None:
    value = getattr(draft.passport, name)
    if _blank(value):
        issues.append(
            _passport_issue(
                draft,
                "missing_manual_passport_field",
                f"В паспорте не заполнено обязательное поле {name}.",
                column=name,
                raw_value=value,
            )
        )


def _require_number(
    draft: ManualDraft,
    issues: list[ValidationIssue],
    name: str,
    *,
    positive: bool = False,
    nonnegative: bool = False,
) -> float | None:
    raw = getattr(draft.passport, name)
    value = _finite(raw)
    invalid = value is None
    if positive:
        invalid = invalid or bool(value is not None and value <= 0)
    if nonnegative:
        invalid = invalid or bool(value is not None and value < 0)
    if invalid:
        condition = "положительным" if positive else "неотрицательным" if nonnegative else "конечным"
        issues.append(
            _passport_issue(
                draft,
                "invalid_manual_passport_number",
                f"Поле {name} должно быть {condition} числом.",
                column=name,
                raw_value=raw,
            )
        )
        return None
    return value


def _validate_date_field(
    draft: ManualDraft,
    issues: list[ValidationIssue],
    name: str,
    *,
    required: bool = True,
) -> date | None:
    raw = getattr(draft.passport, name)
    if _blank(raw):
        if required:
            _require_text(draft, issues, name)
        return None
    try:
        return date.fromisoformat(str(raw).strip())
    except ValueError:
        issues.append(
            _passport_issue(
                draft,
                "invalid_manual_date",
                f"Поле {name} должно иметь формат YYYY-MM-DD.",
                column=name,
                raw_value=raw,
            )
        )
        return None


def _indicator_number(
    draft: ManualDraft,
    issues: list[ValidationIssue],
    channel: str,
    indicator: Any,
    name: str,
    *,
    positive: bool = False,
    nonnegative: bool = False,
    required: bool = True,
) -> float | None:
    raw = getattr(indicator, name)
    if _blank(raw) and not required:
        return None
    value = _finite(raw)
    invalid = value is None
    if positive:
        invalid = invalid or bool(value is not None and value <= 0)
    if nonnegative:
        invalid = invalid or bool(value is not None and value < 0)
    if invalid:
        condition = (
            "положительным"
            if positive
            else "неотрицательным"
            if nonnegative
            else "конечным"
        )
        issues.append(
            _passport_issue(
                draft,
                "invalid_manual_indicator_passport_number",
                f"{channel}.{name} должно быть {condition} числом.",
                column=f"indicator_passports.{channel}.{name}",
                raw_value=raw,
            )
        )
        return None
    return value


def _indicator_date(
    draft: ManualDraft,
    issues: list[ValidationIssue],
    channel: str,
    indicator: Any,
    name: str,
) -> date | None:
    raw = getattr(indicator, name)
    if _blank(raw):
        issues.append(
            _passport_issue(
                draft,
                "missing_manual_indicator_passport_field",
                f"В паспорте {channel} не заполнено {name}.",
                column=f"indicator_passports.{channel}.{name}",
                raw_value=raw,
            )
        )
        return None
    try:
        return date.fromisoformat(str(raw).strip())
    except ValueError:
        issues.append(
            _passport_issue(
                draft,
                "invalid_manual_indicator_verification_date",
                f"{channel}.{name} должно иметь формат YYYY-MM-DD.",
                column=f"indicator_passports.{channel}.{name}",
                raw_value=raw,
            )
        )
        return None


def _validate_indicator_passport(
    draft: ManualDraft,
    issues: list[ValidationIssue],
    channel: str,
    indicator: Any,
    *,
    experiment_date: date | None,
) -> None:
    prefix = f"indicator_passports.{channel}"
    for name in ("type", "serial_number", "instrument_id", "mode"):
        raw = getattr(indicator, name)
        if _blank(raw):
            issues.append(
                _passport_issue(
                    draft,
                    "missing_manual_indicator_passport_field",
                    f"В паспорте {channel} не заполнено {name}.",
                    column=f"{prefix}.{name}",
                    raw_value=raw,
                )
            )

    mode = canonical_indicator_mode(indicator.mode)
    if not mode:
        issues.append(
            _passport_issue(
                draft,
                "unsupported_manual_dial_mode",
                f"Неизвестный режим шкалы в паспорте {channel}.",
                column=f"{prefix}.mode",
                raw_value=indicator.mode,
            )
        )
    dial_range = _indicator_number(
        draft, issues, channel, indicator, "range_mm", positive=True
    )
    division = _indicator_number(
        draft, issues, channel, indicator, "division_mm", positive=True
    )
    _indicator_number(
        draft, issues, channel, indicator, "correction_factor", positive=True
    )
    _indicator_number(draft, issues, channel, indicator, "zero_correction_mm")
    if dial_range is not None and division is not None and division > dial_range:
        issues.append(
            _passport_issue(
                draft,
                "manual_dial_resolution_exceeds_range",
                f"Цена деления {channel} не может превышать диапазон шкалы.",
                column=f"{prefix}.division_mm",
                raw_value=indicator.division_mm,
            )
        )
    if mode and mode != "cumulative_settlement":
        initial = _indicator_number(
            draft, issues, channel, indicator, "initial_reading"
        )
        _indicator_number(
            draft, issues, channel, indicator, "max_increment_mm", positive=True
        )
        if initial is not None and dial_range is not None and not (0 <= initial < dial_range):
            issues.append(
                _passport_issue(
                    draft,
                    "manual_initial_reading_out_of_range",
                    f"Начальное показание {channel} должно быть в [0; range).",
                    column=f"{prefix}.initial_reading",
                    raw_value=indicator.initial_reading,
                )
            )
    elif not _blank(indicator.initial_reading):
        _indicator_number(
            draft,
            issues,
            channel,
            indicator,
            "initial_reading",
            required=False,
        )
    _indicator_number(
        draft,
        issues,
        channel,
        indicator,
        "reverse_tolerance_mm",
        nonnegative=True,
        required=False,
    )
    _indicator_number(
        draft,
        issues,
        channel,
        indicator,
        "travel_range_mm",
        positive=True,
        required=False,
    )
    initial_turn = indicator.initial_turn
    if isinstance(initial_turn, bool) or not isinstance(initial_turn, int):
        issues.append(
            _passport_issue(
                draft,
                "invalid_manual_indicator_initial_turn",
                f"{channel}.initial_turn должен быть целым числом.",
                column=f"{prefix}.initial_turn",
                raw_value=initial_turn,
            )
        )
    cumulative_sign = _indicator_number(
        draft, issues, channel, indicator, "cumulative_sign"
    )
    if cumulative_sign is not None and not math.isclose(
        abs(cumulative_sign), 1.0, abs_tol=1e-12
    ):
        issues.append(
            _passport_issue(
                draft,
                "invalid_manual_indicator_sign",
                f"{channel}.cumulative_sign должен быть +1 или -1.",
                column=f"{prefix}.cumulative_sign",
                raw_value=indicator.cumulative_sign,
            )
        )
    for coordinate in ("x_mm", "y_mm"):
        _indicator_number(
            draft,
            issues,
            channel,
            indicator,
            coordinate,
            required=False,
        )

    verified = _indicator_date(
        draft, issues, channel, indicator, "verification_date"
    )
    valid_until = _indicator_date(
        draft, issues, channel, indicator, "verification_valid_until"
    )
    if verified is not None and valid_until is not None and valid_until < verified:
        issues.append(
            _passport_issue(
                draft,
                "invalid_manual_verification_period",
                f"Срок поверки {channel} заканчивается раньше даты поверки.",
                column=f"{prefix}.verification_valid_until",
                raw_value=indicator.verification_valid_until,
            )
        )
    if experiment_date is None:
        issues.append(
            _passport_issue(
                draft,
                "manual_indicator_verification_review_required",
                f"Дата опыта неизвестна: статус поверки {channel} требует проверки.",
                column="test_date",
                raw_value=draft.passport.test_date,
                level="warning",
            )
        )
    elif verified is not None and experiment_date < verified:
        issues.append(
            _passport_issue(
                draft,
                "indicator_verification_not_yet_valid_at_test",
                f"Поверка {channel} ещё не действовала на дату испытания.",
                column=f"{prefix}.verification_date",
                raw_value=indicator.verification_date,
                level="warning",
            )
        )
    elif valid_until is not None and experiment_date > valid_until:
        issues.append(
            _passport_issue(
                draft,
                "indicator_verification_expired_at_test",
                f"На дату испытания срок поверки {channel} истёк.",
                column=f"{prefix}.verification_valid_until",
                raw_value=indicator.verification_valid_until,
                level="warning",
            )
        )

    if indicator.assignment_status != "confirmed":
        issues.append(
            _passport_issue(
                draft,
                "manual_indicator_assignment_not_confirmed",
                f"Поканальное назначение {channel} не подтверждено инженером.",
                column=f"{prefix}.assignment_status",
                raw_value=indicator.assignment_status,
            )
        )


def _validate_aggregation(
    draft: ManualDraft,
    issues: list[ValidationIssue],
    active_channels: list[str],
) -> None:
    passport = draft.passport
    method = passport.settlement_aggregation
    configured = list(passport.settlement_aggregation_channels)
    primary = passport.settlement_primary_channel
    if method not in _AGGREGATION_METHODS:
        issues.append(
            _passport_issue(
                draft,
                "invalid_manual_settlement_aggregation",
                "Выберите поддерживаемую политику агрегации осадки.",
                column="settlement_aggregation",
                raw_value=method,
            )
        )
        return
    if passport.settlement_missing_channel_policy not in {
        "block",
        "allow_if_solvable",
    }:
        issues.append(
            _passport_issue(
                draft,
                "invalid_manual_missing_channel_policy",
                "Политика пропуска должна быть block или allow_if_solvable.",
                column="settlement_missing_channel_policy",
                raw_value=passport.settlement_missing_channel_policy,
            )
        )
    if len(configured) != len(set(configured)):
        issues.append(
            _passport_issue(
                draft,
                "duplicate_manual_aggregation_channel",
                "Список каналов агрегации не должен содержать повторов.",
                column="settlement_aggregation_channels",
                raw_value=configured,
            )
        )
    unknown = [channel for channel in configured if channel not in active_channels]
    if unknown:
        issues.append(
            _passport_issue(
                draft,
                "inactive_manual_aggregation_channel",
                "Агрегация ссылается на неактивные каналы: " + ", ".join(unknown),
                column="settlement_aggregation_channels",
                raw_value=configured,
            )
        )

    if method == "no_aggregation":
        issues.append(
            _passport_issue(
                draft,
                "manual_settlement_aggregation_not_selected",
                "Осадка не формируется, пока инженер не выберет метод агрегации.",
                column="settlement_aggregation",
                raw_value=method,
            )
        )
        return
    if method == "all_channels_mean" and configured != active_channels:
        issues.append(
            _passport_issue(
                draft,
                "manual_all_channels_basis_mismatch",
                "Для all_channels_mean заранее зафиксируйте все активные каналы в их порядке.",
                column="settlement_aggregation_channels",
                raw_value=configured,
            )
        )
    if method in {"selected_channels_mean", "plane_center"} and not configured:
        issues.append(
            _passport_issue(
                draft,
                "missing_manual_aggregation_channels",
                "Для выбранного метода нужен непустой фиксированный список каналов.",
                column="settlement_aggregation_channels",
                raw_value=configured,
            )
        )
    if method == "primary_channel":
        if primary not in active_channels:
            issues.append(
                _passport_issue(
                    draft,
                    "invalid_manual_primary_channel",
                    "Основной канал должен быть выбран из активных каналов.",
                    column="settlement_primary_channel",
                    raw_value=primary,
                )
            )
    elif primary is not None:
        issues.append(
            _passport_issue(
                draft,
                "unused_manual_primary_channel",
                "Основной канал задан, хотя выбран другой метод агрегации.",
                column="settlement_primary_channel",
                raw_value=primary,
                level="warning",
            )
        )
    if (
        method != "plane_center"
        and passport.settlement_missing_channel_policy != "block"
    ):
        issues.append(
            _passport_issue(
                draft,
                "manual_missing_policy_not_applicable",
                "allow_if_solvable допустим только для plane_center.",
                column="settlement_missing_channel_policy",
                raw_value=passport.settlement_missing_channel_policy,
            )
        )
    if method != "plane_center":
        return
    if len(configured) < 3:
        issues.append(
            _passport_issue(
                draft,
                "insufficient_manual_plane_channels",
                "Для plane_center нужны минимум три заранее выбранных канала.",
                column="settlement_aggregation_channels",
                raw_value=configured,
            )
        )
        return
    coordinates: list[tuple[float, float]] = []
    for channel in configured:
        indicator = passport.indicator_passports.get(channel)
        if indicator is None:
            continue
        x_value = _finite(indicator.x_mm)
        y_value = _finite(indicator.y_mm)
        if x_value is None or y_value is None:
            issues.append(
                _passport_issue(
                    draft,
                    "missing_manual_plane_coordinate",
                    f"Для {channel} задайте конечные x_mm и y_mm.",
                    column=f"indicator_passports.{channel}.x_mm/y_mm",
                    raw_value=[indicator.x_mm, indicator.y_mm],
                )
            )
            continue
        coordinates.append((x_value, y_value))
    if len(coordinates) == len(configured):
        design = np.column_stack(
            [np.ones(len(coordinates)), np.asarray(coordinates, dtype=float)]
        )
        if int(np.linalg.matrix_rank(design)) < 3:
            issues.append(
                _passport_issue(
                    draft,
                    "collinear_manual_plane_coordinates",
                    "Координаты каналов plane_center коллинеарны.",
                    column="settlement_aggregation_channels",
                    raw_value=coordinates,
                )
            )


def _validate_passport(draft: ManualDraft) -> list[ValidationIssue]:
    passport = draft.passport
    issues: list[ValidationIssue] = []

    for name in (
        "project_name",
        "series_name",
        "operator",
        "laboratory_or_site",
        "group_name",
        "soil_type",
        "soil_batch",
        "reinforcement_type",
        "stamp_shape",
        "load_kind",
        "load_unit",
        "settlement_unit",
    ):
        _require_text(draft, issues, name)
    if not passport.test_id:
        issues.append(
            _passport_issue(
                draft,
                "missing_manual_test_id",
                "Заполните test_name или archive_number.",
                column="test_name/archive_number",
            )
        )
    if (
        isinstance(passport.pair_id, str)
        and passport.pair_id
        and passport.pair_id != passport.pair_id.strip()
    ):
        issues.append(
            _passport_issue(
                draft,
                "noncanonical_manual_pair_id",
                "ID пары содержит пробелы по краям и не будет использоваться как "
                "подтверждение парного дизайна.",
                column="pair_id",
                raw_value=passport.pair_id,
                level="warning",
            )
        )

    test_date = _validate_date_field(draft, issues, "test_date", required=False)

    if passport.test_scope not in MANUAL_TEST_SCOPES:
        issues.append(
            _passport_issue(
                draft,
                "invalid_manual_test_scope",
                f"test_scope должен быть одним из: {', '.join(MANUAL_TEST_SCOPES)}.",
                column="test_scope",
                raw_value=passport.test_scope,
            )
        )
    if passport.protocol_type not in MANUAL_PROTOCOL_TYPES:
        issues.append(
            _passport_issue(
                draft,
                "invalid_manual_protocol_type",
                f"protocol_type должен быть одним из: {', '.join(MANUAL_PROTOCOL_TYPES)}.",
                column="protocol_type",
                raw_value=passport.protocol_type,
            )
        )

    kind = str(passport.load_kind).strip().casefold()
    if kind not in _LOAD_UNITS:
        issues.append(
            _passport_issue(
                draft,
                "invalid_manual_load_kind",
                "load_kind должен быть force или pressure.",
                column="load_kind",
                raw_value=passport.load_kind,
            )
        )
    elif str(passport.load_unit).strip().casefold() not in _LOAD_UNITS[kind]:
        issues.append(
            _passport_issue(
                draft,
                "manual_load_unit_kind_conflict",
                "Единица нагрузки не соответствует load_kind.",
                column="load_unit",
                raw_value=passport.load_unit,
            )
        )
    if str(passport.settlement_unit).strip().casefold() not in _SETTLEMENT_UNITS:
        issues.append(
            _passport_issue(
                draft,
                "unsupported_manual_settlement_unit",
                "Неподдерживаемая единица осадки.",
                column="settlement_unit",
                raw_value=passport.settlement_unit,
            )
        )
    if str(passport.stamp_shape).strip().casefold() not in _STAMP_SHAPES:
        issues.append(
            _passport_issue(
                draft,
                "unsupported_manual_stamp_shape",
                "Неподдерживаемая форма штампа.",
                column="stamp_shape",
                raw_value=passport.stamp_shape,
            )
        )

    _require_number(draft, issues, "stamp_diameter_mm", positive=True)
    if not _blank(passport.stamp_area_m2):
        _require_number(draft, issues, "stamp_area_m2", positive=True)
    _require_number(draft, issues, "load_factor", positive=True)
    _require_number(draft, issues, "load_zero")
    _require_number(draft, issues, "lever_ratio", positive=True)

    count = passport.number_of_indicators
    active_channels: list[str] = []
    if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= 4:
        issues.append(
            _passport_issue(
                draft,
                "invalid_manual_indicator_count",
                "number_of_indicators должен быть целым числом от 1 до 4.",
                column="number_of_indicators",
                raw_value=count,
            )
        )
    else:
        active_channels = list(_INDICATOR_CHANNELS[:count])
        for channel in active_channels:
            indicator = passport.indicator_passports.get(channel)
            if indicator is None:
                issues.append(
                    _passport_issue(
                        draft,
                        "missing_manual_indicator_channel_passport",
                        f"Для активного канала {channel} нужен отдельный паспорт.",
                        column=f"indicator_passports.{channel}",
                    )
                )
                continue
            _validate_indicator_passport(
                draft,
                issues,
                channel,
                indicator,
                experiment_date=test_date,
            )
        for channel in _INDICATOR_CHANNELS[count:]:
            if passport.indicator_passports.get(channel) is not None:
                issues.append(
                    _passport_issue(
                        draft,
                        "inactive_manual_indicator_passport",
                        f"Паспорт {channel} сохранён, но канал неактивен и не участвует в расчёте.",
                        column=f"indicator_passports.{channel}",
                        level="warning",
                    )
                )

    reference = passport.indicator_passports.get("reference_indicator")
    if reference is not None:
        _validate_indicator_passport(
            draft,
            issues,
            "reference_indicator",
            reference,
            experiment_date=test_date,
        )
    if passport.metrology_status not in {
        "draft",
        "migration_review_required",
        "confirmed",
    }:
        issues.append(
            _passport_issue(
                draft,
                "invalid_manual_metrology_status",
                "Неизвестный статус поканальной метрологии.",
                column="metrology_status",
                raw_value=passport.metrology_status,
            )
        )
    elif passport.metrology_status == "migration_review_required":
        issues.append(
            _passport_issue(
                draft,
                "manual_metrology_migration_review_required",
                "Старый общий паспорт сохранён, но его распределение по каналам ещё не подтверждено.",
                column="metrology_status",
                raw_value=passport.metrology_status,
            )
        )
    elif passport.metrology_status != "confirmed":
        issues.append(
            _passport_issue(
                draft,
                "manual_metrology_confirmation_required",
                "Подтвердите поканальные паспорта перед расчётом.",
                column="metrology_status",
                raw_value=passport.metrology_status,
            )
        )
    _validate_aggregation(draft, issues, active_channels)

    reinforcement_type = str(passport.reinforcement_type).strip().casefold()
    reinforcement = passport.reinforcement
    if passport.is_reinforced:
        if reinforcement_type in {"", "none", "без армирования"}:
            issues.append(
                _passport_issue(
                    draft,
                    "missing_manual_reinforcement_type",
                    "Для армированного опыта укажите тип армирования.",
                    column="reinforcement_type",
                    raw_value=passport.reinforcement_type,
                )
            )
        if _blank(reinforcement.material):
            issues.append(
                _passport_issue(
                    draft,
                    "missing_manual_reinforcement_field",
                    "Для армированного опыта укажите material.",
                    column="reinforcement.material",
                    raw_value=reinforcement.material,
                )
            )
        layers = _finite(reinforcement.number_of_layers)
        if layers is None or layers <= 0 or not math.isclose(layers, round(layers), abs_tol=1e-9):
            issues.append(
                _passport_issue(
                    draft,
                    "invalid_manual_reinforcement_layers",
                    "number_of_layers должен быть положительным целым числом.",
                    column="reinforcement.number_of_layers",
                    raw_value=reinforcement.number_of_layers,
                )
            )
        for name in (
            "depth_mm",
            "spacing_mm",
            "length_mm",
            "width_mm",
            "bar_diameter_or_aperture_mm",
        ):
            raw = getattr(reinforcement, name)
            if not _blank(raw) and (_finite(raw) is None or float(_finite(raw)) <= 0):
                issues.append(
                    _passport_issue(
                        draft,
                        "invalid_manual_reinforcement_dimension",
                        f"reinforcement.{name} должно быть положительным числом.",
                        column=f"reinforcement.{name}",
                        raw_value=raw,
                    )
                )
        if not isinstance(reinforcement.custom_parameters, dict):
            issues.append(
                _passport_issue(
                    draft,
                    "invalid_manual_reinforcement_parameters",
                    "reinforcement.custom_parameters должен быть JSON-объектом.",
                    column="reinforcement.custom_parameters",
                    raw_value=reinforcement.custom_parameters,
                )
            )
    elif reinforcement_type not in {"none", "без армирования"}:
        issues.append(
            _passport_issue(
                draft,
                "manual_reinforcement_flag_conflict",
                "is_reinforced=false несовместим с указанным типом армирования.",
                column="reinforcement_type",
                raw_value=passport.reinforcement_type,
            )
        )
    return issues


def _valid_uuid(value: str) -> bool:
    try:
        UUID(str(value))
    except (ValueError, TypeError, AttributeError):
        return False
    return True


def _timestamp(value: Any) -> pd.Timestamp | None:
    if _blank(value):
        return None
    parsed = pd.to_datetime(value, errors="coerce", utc=True)
    return None if pd.isna(parsed) else parsed


def _validate_rows(draft: ManualDraft) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    rows = draft.rows
    indicator_count = draft.passport.number_of_indicators
    active_count = indicator_count if isinstance(indicator_count, int) and 1 <= indicator_count <= 4 else 0
    parsed_loads: list[float | None] = []
    elapsed_values: list[float | None] = []
    timestamp_values: list[pd.Timestamp | None] = []
    measurement_positions: list[int] = []

    uuid_positions: dict[str, list[int]] = {}
    sequence_positions: dict[int, list[int]] = {}
    for position, point in enumerate(rows):
        uuid_positions.setdefault(point.manual_row_uuid, []).append(position)
        sequence_positions.setdefault(point.sequence_no, []).append(position)
        if (
            isinstance(point.sequence_no, bool)
            or not isinstance(point.sequence_no, int)
            or point.sequence_no < 1
        ):
            issues.append(
                _row_issue(
                    draft,
                    point,
                    position,
                    "invalid_manual_sequence_no",
                    "sequence_no должен быть положительным целым числом.",
                    column="sequence_no",
                    raw_value=point.sequence_no,
                )
            )
        if not _valid_uuid(point.manual_row_uuid):
            issues.append(
                _row_issue(
                    draft,
                    point,
                    position,
                    "invalid_manual_row_uuid",
                    "manual_row_uuid должен быть непустым UUID.",
                    column="manual_row_uuid",
                    raw_value=point.manual_row_uuid,
                )
            )
        if point.source_type != "manual":
            issues.append(
                _row_issue(
                    draft,
                    point,
                    position,
                    "invalid_manual_source_type",
                    "Для ручной строки source_type должен быть manual.",
                    column="source_type",
                    raw_value=point.source_type,
                )
            )
        if point.source_row is not None:
            issues.append(
                _row_issue(
                    draft,
                    point,
                    position,
                    "manual_source_row_must_be_null",
                    "Для ручного ввода source_row должен оставаться null.",
                    column="source_row",
                    raw_value=point.source_row,
                )
            )
        for name in ("created_by", "created_at", "modified_by", "modified_at"):
            if _blank(getattr(point, name)):
                issues.append(
                    _row_issue(
                        draft,
                        point,
                        position,
                        "missing_manual_provenance",
                        f"Не заполнено поле происхождения {name}.",
                        column=name,
                        raw_value=getattr(point, name),
                    )
                )
        created_at = _timestamp(point.created_at)
        modified_at = _timestamp(point.modified_at)
        for name, raw, parsed in (
            ("created_at", point.created_at, created_at),
            ("modified_at", point.modified_at, modified_at),
        ):
            if not _blank(raw) and parsed is None:
                issues.append(
                    _row_issue(
                        draft,
                        point,
                        position,
                        "invalid_manual_provenance_timestamp",
                        f"{name} должен быть корректной датой/временем ISO 8601.",
                        column=name,
                        raw_value=raw,
                    )
                )
        if created_at is not None and modified_at is not None and modified_at < created_at:
            issues.append(
                _row_issue(
                    draft,
                    point,
                    position,
                    "manual_provenance_time_order",
                    "modified_at не может быть раньше created_at.",
                    column="modified_at",
                    raw_value=point.modified_at,
                )
            )

        if point.branch not in MANUAL_BRANCHES:
            issues.append(
                _row_issue(
                    draft,
                    point,
                    position,
                    "invalid_manual_branch",
                    f"branch должен быть одним из: {', '.join(MANUAL_BRANCHES)}.",
                    column="branch",
                    raw_value=point.branch,
                )
            )
        if point.row_status not in MANUAL_ROW_STATUSES:
            issues.append(
                _row_issue(
                    draft,
                    point,
                    position,
                    "invalid_manual_row_status",
                    f"row_status должен быть одним из: {', '.join(MANUAL_ROW_STATUSES)}.",
                    column="row_status",
                    raw_value=point.row_status,
                )
            )
        elif point.row_status == "invalid":
            issues.append(
                _row_issue(
                    draft,
                    point,
                    position,
                    "manual_row_marked_invalid",
                    "Строка явно помечена invalid и останется видимой; анализ заблокирован до её исправления или удаления.",
                    column="row_status",
                    raw_value=point.row_status,
                )
            )

        load = parse_decimal(point.load_raw, family="load")
        parsed_loads.append(load)
        if point.row_status != "invalid" and load is None:
            issues.append(
                _row_issue(
                    draft,
                    point,
                    position,
                    "invalid_manual_load",
                    "Для этой строки требуется конечное числовое load_raw.",
                    column="load_raw",
                    raw_value=point.load_raw,
                )
            )

        elapsed = parse_decimal(point.elapsed_time_s)
        elapsed_values.append(elapsed)
        if not _blank(point.elapsed_time_s) and (elapsed is None or elapsed < 0):
            issues.append(
                _row_issue(
                    draft,
                    point,
                    position,
                    "invalid_manual_elapsed_time",
                    "elapsed_time_s должно быть конечным неотрицательным числом.",
                    column="elapsed_time_s",
                    raw_value=point.elapsed_time_s,
                )
            )
        timestamp = _timestamp(point.timestamp)
        timestamp_values.append(timestamp)
        if not _blank(point.timestamp) and timestamp is None:
            issues.append(
                _row_issue(
                    draft,
                    point,
                    position,
                    "invalid_manual_timestamp",
                    "timestamp должен быть корректной датой/временем ISO 8601.",
                    column="timestamp",
                    raw_value=point.timestamp,
                )
            )

        parsed_indicators: list[float | None] = []
        for number in range(1, 5):
            column = f"indicator_{number}_raw"
            raw = getattr(point, column)
            parsed = parse_decimal(raw, family="measurement")
            parsed_indicators.append(parsed)
            if not _blank(raw) and parsed is None:
                issues.append(
                    _row_issue(
                        draft,
                        point,
                        position,
                        "invalid_manual_indicator",
                        f"{column} должно быть конечным числом без суффикса единицы.",
                        column=column,
                        raw_value=raw,
                    )
                )
            if number > active_count and not _blank(raw):
                issues.append(
                    _row_issue(
                        draft,
                        point,
                        position,
                        "inactive_manual_indicator_value",
                        "Значение введено в неактивный канал индикатора.",
                        column=column,
                        raw_value=raw,
                    )
                )

        if point.row_status == "measurement":
            active = parsed_indicators[:active_count]
            if not any(value is not None for value in active):
                issues.append(
                    _row_issue(
                        draft,
                        point,
                        position,
                        "missing_manual_measurement",
                        "В строке measurement нужно хотя бы одно показание активного индикатора.",
                        column="indicator_1_raw",
                    )
                )
            elif load is not None:
                measurement_positions.append(position)
            missing_active = [index + 1 for index, value in enumerate(active) if value is None]
            if active and missing_active and len(missing_active) < len(active):
                issues.append(
                    _row_issue(
                        draft,
                        point,
                        position,
                        "partial_manual_indicator_readings",
                        "Отсутствуют показания отдельных активных индикаторов: "
                        + ", ".join(map(str, missing_active)),
                        column="indicator_raw",
                        raw_value=missing_active,
                        level="warning",
                    )
                )
        # A failure row intentionally may have no indicator reading.  Its load
        # remains mandatory so the existing interval-censoring logic can use it.

    for value, positions in uuid_positions.items():
        if value and len(positions) > 1:
            for position in positions:
                issues.append(
                    _row_issue(
                        draft,
                        rows[position],
                        position,
                        "duplicate_manual_row_uuid",
                        "manual_row_uuid должен быть уникальным.",
                        column="manual_row_uuid",
                        raw_value=value,
                    )
                )
    for value, positions in sequence_positions.items():
        if len(positions) > 1:
            for position in positions:
                issues.append(
                    _row_issue(
                        draft,
                        rows[position],
                        position,
                        "duplicate_manual_sequence_no",
                        "sequence_no должен быть уникальным.",
                        column="sequence_no",
                        raw_value=value,
                    )
                )
    for position in range(1, len(rows)):
        previous = rows[position - 1].sequence_no
        current = rows[position].sequence_no
        if (
            isinstance(previous, int)
            and not isinstance(previous, bool)
            and isinstance(current, int)
            and not isinstance(current, bool)
            and current <= previous
        ):
            issues.append(
                _row_issue(
                    draft,
                    rows[position],
                    position,
                    "manual_sequence_order",
                    "sequence_no должны возрастать в порядке строк без скрытой сортировки.",
                    column="sequence_no",
                    raw_value=current,
                )
            )

        previous_load = parsed_loads[position - 1]
        current_load = parsed_loads[position]
        if (
            previous_load is not None
            and current_load is not None
            and current_load < previous_load - 1e-12
            and rows[position].branch != "unloading"
            and rows[position].row_status != "invalid"
        ):
            issues.append(
                _row_issue(
                    draft,
                    rows[position],
                    position,
                    "manual_load_decrease_outside_unloading",
                    "Уменьшение нагрузки допустимо только в ветви unloading.",
                    column="load_raw",
                    raw_value=rows[position].load_raw,
                )
            )

    last_elapsed: float | None = None
    last_timestamp: pd.Timestamp | None = None
    for position, (elapsed, timestamp) in enumerate(
        zip(elapsed_values, timestamp_values)
    ):
        if elapsed is not None:
            if last_elapsed is not None and elapsed < last_elapsed:
                issues.append(
                    _row_issue(
                        draft,
                        rows[position],
                        position,
                        "manual_elapsed_time_order",
                        "elapsed_time_s не может уменьшаться относительно предыдущего зарегистрированного времени.",
                        column="elapsed_time_s",
                        raw_value=rows[position].elapsed_time_s,
                    )
                )
            last_elapsed = elapsed
        if timestamp is not None:
            if last_timestamp is not None and timestamp < last_timestamp:
                issues.append(
                    _row_issue(
                        draft,
                        rows[position],
                        position,
                        "manual_timestamp_order",
                        "timestamp не может уменьшаться относительно предыдущей зарегистрированной метки.",
                        column="timestamp",
                        raw_value=rows[position].timestamp,
                    )
                )
            last_timestamp = timestamp

    if len(measurement_positions) < 2:
        issues.append(
            _passport_issue(
                draft,
                "insufficient_manual_measurements",
                "Для анализа нужны как минимум две корректные строки measurement.",
                column="rows",
                raw_value=len(measurement_positions),
            )
        )

    duplicate_groups: dict[tuple[Any, ...], list[int]] = {}
    for position, point in enumerate(rows):
        signature = (
            point.stage_no,
            point.branch,
            point.elapsed_time_s,
            point.timestamp,
            point.load_raw,
            point.indicator_1_raw,
            point.indicator_2_raw,
            point.indicator_3_raw,
            point.indicator_4_raw,
            point.row_status,
            point.comment,
        )
        duplicate_groups.setdefault(signature, []).append(position)
    for positions in duplicate_groups.values():
        if len(positions) < 2:
            continue
        for position in positions:
            issues.append(
                _row_issue(
                    draft,
                    rows[position],
                    position,
                    "duplicate_manual_row",
                    "Полностью повторяющаяся строка сохранена и требует проверки.",
                    column="sequence_no",
                    raw_value=rows[position].sequence_no,
                    level="warning",
                )
            )

    failure_positions = [
        position for position, point in enumerate(rows) if point.row_status == "failure"
    ]
    if len(failure_positions) > 1:
        for position in failure_positions:
            issues.append(
                _row_issue(
                    draft,
                    rows[position],
                    position,
                    "multiple_manual_failure_rows",
                    "В одной ревизии испытания допустима только одна строка failure.",
                    column="row_status",
                    raw_value="failure",
                )
            )
    if failure_positions:
        failure_position = failure_positions[0]
        stable_before = any(
            point.row_status == "measurement" and position < failure_position
            for position, point in enumerate(rows)
        )
        if not stable_before:
            issues.append(
                _row_issue(
                    draft,
                    rows[failure_position],
                    failure_position,
                    "manual_failure_without_stable_predecessor",
                    "Строка failure должна следовать после измеренной устойчивой точки.",
                    column="row_status",
                    raw_value="failure",
                )
            )
        for position in range(failure_position + 1, len(rows)):
            if rows[position].row_status == "measurement":
                issues.append(
                    _row_issue(
                        draft,
                        rows[position],
                        position,
                        "manual_measurement_after_failure",
                        "После failure нельзя добавлять обычную measurement в той же ревизии.",
                        column="row_status",
                        raw_value=rows[position].row_status,
                    )
                )
    terminal_positions = [
        position for position, point in enumerate(rows) if point.row_status in _TERMINAL_STATUSES
    ]
    if terminal_positions:
        first_terminal = min(terminal_positions)
        for position in range(first_terminal + 1, len(rows)):
            if rows[position].row_status == "measurement" and not any(
                item.code == "manual_measurement_after_failure" and item.row == position + 1
                for item in issues
            ):
                issues.append(
                    _row_issue(
                        draft,
                        rows[position],
                        position,
                        "manual_measurement_after_terminal_status",
                        "После терминального статуса нужна новая ревизия или явное продолжение протокола.",
                        column="row_status",
                        raw_value=rows[position].row_status,
                    )
                )
    return issues


def validate_manual_draft(draft: ManualDraft) -> ManualValidationResult:
    """Validate a draft without dropping or repairing any row."""

    if not isinstance(draft, ManualDraft):
        issue = ValidationIssue(
            "error",
            "invalid_manual_draft_type",
            "Ожидается экземпляр ManualDraft.",
            raw_value=type(draft).__name__,
        )
        return ManualValidationResult(issues=[issue], adapter_issues=[issue])
    issues = merge_manual_issues(_validate_passport(draft), _validate_rows(draft))
    return ManualValidationResult(issues=issues, adapter_issues=list(issues))


__all__ = [
    "ManualValidationResult",
    "merge_manual_issues",
    "validate_manual_draft",
]
