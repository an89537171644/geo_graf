"""Streamlit adapter for the versioned manual-entry domain services.

Widget state is never the scientific source of truth.  It is reconciled with
``ManualEntryService`` and only an explicit activation freezes a draft for the
existing calculation pipeline.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from .data import failure_summary
from .indicators import indicator_audit_frame, indicator_event_frame
from .manual_entry_models import (
    MANUAL_BRANCHES,
    MANUAL_EDITOR_COLUMNS,
    MANUAL_PROTOCOL_TYPES,
    MANUAL_ROW_STATUSES,
    MANUAL_TEST_SCOPES,
    ManualAuditEvent,
    ManualDraft,
    ManualIndicatorPassport,
    utc_now_iso,
)
from .manual_entry_service import (
    EDITOR_UUID_COLUMN,
    HistoryEmptyError,
    ManualEntryService,
    ManualEntryServiceError,
)
from .manual_entry_validation import (
    ManualValidationResult,
    merge_manual_issues,
    validate_manual_draft,
)
from .plotting import plot_curves
from .schema import ValidationIssue


MANUAL_SERVICE_KEY = "manual_entry_service"
MANUAL_ACTIVE_DRAFT_KEY = "manual_active_draft"
MANUAL_ACTIVE_HASH_KEY = "manual_active_hash"
MANUAL_EDITOR_GENERATION_KEY = "manual_editor_generation"
MANUAL_SOURCE_REQUEST_KEY = "manual_source_request"


@dataclass(slots=True)
class ManualPreview:
    prepared: pd.DataFrame
    issues: list[Any]
    bundle: Any


def _ui_safe_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Make mixed object columns Arrow-safe without changing stored values."""

    result = frame.copy(deep=True)

    def text_value(value: Any) -> str:
        if value is None or value is pd.NA:
            return ""
        if isinstance(value, (dict, list, tuple)):
            return json.dumps(value, ensure_ascii=False, default=str)
        try:
            if bool(pd.isna(value)):
                return ""
        except (TypeError, ValueError):
            pass
        return str(value)

    for column in result.columns:
        if pd.api.types.is_object_dtype(result[column].dtype):
            result[column] = result[column].map(text_value)
    result.attrs.clear()
    return result


def get_manual_service(*, author: str = "local-user") -> ManualEntryService:
    service = st.session_state.get(MANUAL_SERVICE_KEY)
    if not isinstance(service, ManualEntryService):
        service = ManualEntryService(author=author)
        st.session_state[MANUAL_SERVICE_KEY] = service
        st.session_state.setdefault(MANUAL_EDITOR_GENERATION_KEY, 0)
    service.author = author or service.author
    return service


def active_manual_draft() -> ManualDraft | None:
    payload = st.session_state.get(MANUAL_ACTIVE_DRAFT_KEY)
    if not isinstance(payload, dict):
        return None
    try:
        return ManualDraft.from_dict(payload)
    except ValueError:
        return None


def _bump_editor() -> None:
    st.session_state[MANUAL_EDITOR_GENERATION_KEY] = int(
        st.session_state.get(MANUAL_EDITOR_GENERATION_KEY, 0)
    ) + 1


def _bounded_widget_value(
    key: str, *, minimum: int, maximum: int, default: int
) -> int:
    """Clamp persisted widget state after row insertion/deletion reruns."""

    try:
        current = int(st.session_state.get(key, default))
    except (TypeError, ValueError):
        current = default
    bounded = min(maximum, max(minimum, current))
    if key in st.session_state and st.session_state[key] != bounded:
        st.session_state[key] = bounded
    return bounded


def _option_index(options: list[str], value: str) -> tuple[list[str], int]:
    if value and value not in options:
        options = [value, *options]
    return options, options.index(value) if value in options else 0


def _indicator_passport_widgets(
    channel: str,
    indicator: ManualIndicatorPassport | None,
    *,
    key_prefix: str,
) -> dict[str, Any]:
    """Render one concrete channel passport and return its exact raw payload."""

    current = indicator or ManualIndicatorPassport()
    widget_prefix = f"{key_prefix}_{channel}"
    mode_options, mode_index = _option_index(
        [
            "increasing",
            "increasing_wrapped",
            "decreasing",
            "decreasing_wrapped",
            "cumulative_settlement",
        ],
        current.mode,
    )
    assignment_options, assignment_index = _option_index(
        ["draft", "review_required", "confirmed", "migration_review_required"],
        current.assignment_status,
    )
    sign_options, sign_index = _option_index(
        ["1", "-1"], current.cumulative_sign or "1"
    )
    c1, c2, c3, c4 = st.columns(4)
    indicator_type = c1.text_input(
        "Тип *", value=current.type, key=f"{widget_prefix}_type"
    )
    serial_number = c2.text_input(
        "Заводской № *",
        value=current.serial_number,
        key=f"{widget_prefix}_serial",
    )
    instrument_id = c3.text_input(
        "ID прибора *",
        value=current.instrument_id or "",
        key=f"{widget_prefix}_instrument_id",
        help="ID задаётся явно и не подменяется заводским номером.",
    )
    mode = c4.selectbox(
        "Режим шкалы *",
        mode_options,
        index=mode_index,
        key=f"{widget_prefix}_mode",
    )
    range_mm = c1.text_input(
        "Диапазон, мм *", value=current.range_mm or "", key=f"{widget_prefix}_range"
    )
    division_mm = c2.text_input(
        "Цена деления, мм *",
        value=current.division_mm or "",
        key=f"{widget_prefix}_division",
    )
    correction_factor = c3.text_input(
        "Поправочный коэффициент *",
        value=current.correction_factor or "",
        key=f"{widget_prefix}_factor",
    )
    cumulative_sign = c4.selectbox(
        "Знак готовой осадки *",
        sign_options,
        index=sign_index,
        key=f"{widget_prefix}_sign",
    )
    initial_reading = c1.text_input(
        "Начальное показание",
        value=current.initial_reading or "",
        key=f"{widget_prefix}_initial",
    )
    initial_turn = int(
        c2.number_input(
            "Начальный оборот *",
            value=int(current.initial_turn or 0),
            step=1,
            key=f"{widget_prefix}_initial_turn",
        )
    )
    zero_correction_mm = c3.text_input(
        "Коррекция нуля, мм *",
        value=current.zero_correction_mm or "",
        key=f"{widget_prefix}_zero",
    )
    max_increment_mm = c4.text_input(
        "Макс. приращение, мм",
        value=current.max_increment_mm or "",
        key=f"{widget_prefix}_max_increment",
    )
    reverse_tolerance_mm = c1.text_input(
        "Допуск обратного хода, мм",
        value=current.reverse_tolerance_mm or "",
        key=f"{widget_prefix}_reverse",
    )
    travel_range_mm = c2.text_input(
        "Полный ход, мм",
        value=current.travel_range_mm or "",
        key=f"{widget_prefix}_travel",
    )
    verification_date = c3.text_input(
        "Дата поверки *",
        value=current.verification_date,
        key=f"{widget_prefix}_verified",
    )
    verification_valid_until = c4.text_input(
        "Поверка действует до *",
        value=current.verification_valid_until,
        key=f"{widget_prefix}_valid_until",
    )
    x_mm = c1.text_input(
        "x, мм", value=current.x_mm or "", key=f"{widget_prefix}_x"
    )
    y_mm = c2.text_input(
        "y, мм", value=current.y_mm or "", key=f"{widget_prefix}_y"
    )
    assignment_status = c3.selectbox(
        "Назначение канала *",
        assignment_options,
        index=assignment_index,
        key=f"{widget_prefix}_assignment",
    )
    return {
        "type": indicator_type,
        "serial_number": serial_number,
        "instrument_id": instrument_id or None,
        "range_mm": range_mm or None,
        "division_mm": division_mm or None,
        "correction_factor": correction_factor or None,
        "mode": mode,
        "initial_reading": initial_reading or None,
        "initial_turn": initial_turn,
        "zero_correction_mm": zero_correction_mm or None,
        "max_increment_mm": max_increment_mm or None,
        "reverse_tolerance_mm": reverse_tolerance_mm or None,
        "travel_range_mm": travel_range_mm or None,
        "verification_date": verification_date,
        "verification_valid_until": verification_valid_until,
        "x_mm": x_mm or None,
        "y_mm": y_mm or None,
        "cumulative_sign": cumulative_sign,
        "assignment_status": assignment_status,
    }


def _passport_form(service: ManualEntryService, actor: str, *, key_prefix: str) -> None:
    passport = service.draft.passport
    st.subheader("1. Паспорт опыта")
    st.caption("Физические параметры применяются только после явного сохранения формы.")
    scope_options, scope_index = _option_index(
        list(MANUAL_TEST_SCOPES), passport.test_scope
    )
    protocol_options, protocol_index = _option_index(
        list(MANUAL_PROTOCOL_TYPES), passport.protocol_type
    )
    load_kinds, load_kind_index = _option_index(
        ["force", "pressure"], passport.load_kind
    )
    load_units, load_unit_index = _option_index(
        ["kN", "N", "MN", "kgf", "tf", "kPa", "Pa", "MPa"],
        passport.load_unit,
    )
    shape_options, shape_index = _option_index(
        ["circle", "custom"], passport.stamp_shape
    )
    current_n = passport.number_of_indicators or 1
    current_n = min(4, max(1, int(current_n)))
    indicator_payload = {
        channel: values.to_dict() if values is not None else None
        for channel, values in passport.indicator_passports.items()
    }

    with st.form(f"{key_prefix}_passport_form_{service.draft.draft_id}"):
        c1, c2, c3 = st.columns(3)
        project_name = c1.text_input("Проект *", value=passport.project_name)
        series_name = c2.text_input("Серия *", value=passport.series_name)
        archive_number = c3.text_input(
            "Архивный номер / ID *", value=passport.archive_number
        )
        test_name = c1.text_input("Название опыта", value=passport.test_name)
        test_date = c2.text_input(
            "Дата YYYY-MM-DD (необязательно)",
            value=passport.test_date,
            help="Без даты опыта статус поверки будет review_required.",
        )
        operator = c3.text_input("Оператор *", value=passport.operator)
        laboratory_or_site = c1.text_input(
            "Лаборатория / площадка *", value=passport.laboratory_or_site
        )
        test_scope = c2.selectbox(
            "Область *", scope_options, index=scope_index
        )
        protocol_type = c3.selectbox(
            "Тип протокола *", protocol_options, index=protocol_index
        )
        group_name = c1.text_input("Группа опыта *", value=passport.group_name)
        baseline_group = c2.text_input(
            "Контрольная группа (необязательно)",
            value=passport.baseline_group,
            help="Имя контрольной серии; оно не является идентификатором пары.",
        )
        pair_id = c3.text_input(
            "ID пары / блока (необязательно)",
            value=passport.pair_id or "",
            help=(
                "Одинаковый непустой ID задают только двум действительно сопоставленным "
                "опытам. Из контрольной группы ID пары не выводится."
            ),
        )
        st.caption(
            "Контрольная группа и ID пары имеют разный смысл. Пустой ID пары означает, "
            "что парность не заявлена."
        )
        soil_type = c1.text_input("Тип грунта *", value=passport.soil_type)
        soil_batch = c1.text_input("Партия грунта *", value=passport.soil_batch)
        is_reinforced = c2.checkbox(
            "Армированный грунт", value=passport.is_reinforced
        )
        reinforcement_type = c3.text_input(
            "Тип армирования *", value=passport.reinforcement_type
        )

        st.markdown("##### Штамп и нагрузка")
        c1, c2, c3, c4 = st.columns(4)
        stamp_shape = c1.selectbox(
            "Форма штампа *", shape_options, index=shape_index
        )
        stamp_diameter = c2.text_input(
            "Диаметр/размер, мм *", value=passport.stamp_diameter_mm or ""
        )
        stamp_area = c3.text_input(
            "Площадь, м² (custom)", value=passport.stamp_area_m2 or ""
        )
        load_kind = c4.selectbox(
            "Тип нагрузки *", load_kinds, index=load_kind_index
        )
        load_unit = c1.selectbox(
            "Единица нагрузки *", load_units, index=load_unit_index
        )
        load_factor = c2.text_input(
            "Коэффициент нагрузки *", value=passport.load_factor or ""
        )
        load_zero = c3.text_input("Нуль нагрузки *", value=passport.load_zero or "")
        lever_ratio = c4.text_input(
            "Передаточное отношение *", value=passport.lever_ratio or ""
        )

        st.markdown("##### Индикаторы")
        c1, c2, c3, c4 = st.columns(4)
        number_of_indicators = int(
            c1.number_input(
                "Количество *", min_value=1, max_value=4, value=current_n, step=1
            )
        )
        reference_enabled = c2.checkbox(
            "Опорный индикатор",
            value=passport.indicator_passports.get("reference_indicator") is not None,
            help="Паспорт reference_indicator хранится отдельно от вертикальных каналов.",
        )
        metrology_options, metrology_index = _option_index(
            ["draft", "confirmed", "migration_review_required"],
            passport.metrology_status,
        )
        metrology_status = c3.selectbox(
            "Статус метрологии *", metrology_options, index=metrology_index
        )
        aggregation_options, aggregation_index = _option_index(
            [
                "all_channels_mean",
                "selected_channels_mean",
                "plane_center",
                "primary_channel",
                "no_aggregation",
            ],
            passport.settlement_aggregation,
        )
        settlement_aggregation = c4.selectbox(
            "Агрегация осадки *",
            aggregation_options,
            index=aggregation_index,
        )
        active_channels = [
            f"indicator_{index}" for index in range(1, number_of_indicators + 1)
        ]
        configured_default = [
            channel
            for channel in passport.settlement_aggregation_channels
            if channel in active_channels
        ]
        aggregation_channels = st.multiselect(
            "Фиксированный состав каналов",
            active_channels,
            default=configured_default,
            help=(
                "Состав сохраняется до обработки и не меняется по строкам. "
                "Для all_channels_mean выберите все активные каналы."
            ),
        )
        primary_options = ["", *active_channels]
        primary_default = (
            passport.settlement_primary_channel
            if passport.settlement_primary_channel in active_channels
            else ""
        )
        primary_channel = c1.selectbox(
            "Основной канал",
            primary_options,
            index=primary_options.index(primary_default),
        )
        missing_options, missing_index = _option_index(
            ["block", "allow_if_solvable"],
            passport.settlement_missing_channel_policy,
        )
        missing_policy = c2.selectbox(
            "Пропуск канала *", missing_options, index=missing_index
        )
        metrology_reason = c3.text_input(
            "Причина подтверждения метрологии",
            value="",
            help="Обязательна при смене статуса на confirmed.",
        )
        st.caption(
            "Общего сохраняемого шаблона нет: каждый блок ниже является отдельным "
            "паспортом конкретного канала."
        )
        for channel in active_channels:
            with st.expander(
                f"Паспорт {channel}", expanded=channel == "indicator_1"
            ):
                indicator_payload[channel] = _indicator_passport_widgets(
                    channel,
                    passport.indicator_passports.get(channel),
                    key_prefix=f"{key_prefix}_{service.draft.draft_id}",
                )
        if reference_enabled:
            with st.expander("Паспорт reference_indicator", expanded=False):
                indicator_payload["reference_indicator"] = (
                    _indicator_passport_widgets(
                        "reference_indicator",
                        passport.indicator_passports.get("reference_indicator"),
                        key_prefix=f"{key_prefix}_{service.draft.draft_id}",
                    )
                )
        else:
            indicator_payload["reference_indicator"] = None
        if passport.legacy_common_indicator_passport is not None:
            with st.expander("Legacy-общий паспорт (только чтение)"):
                st.warning(
                    "Эти значения не участвуют в расчёте. Заполните первый "
                    "поканальный паспорт вручную, затем используйте явное копирование."
                )
                st.json(passport.legacy_common_indicator_passport.to_dict())

        reinforcement_changes: dict[str, Any] = {}
        if is_reinforced:
            st.markdown("##### Параметры армирования")
            reinforcement = passport.reinforcement
            c1, c2, c3, c4 = st.columns(4)
            reinforcement_changes = {
                "reinforcement.material": c1.text_input(
                    "Материал *", value=reinforcement.material
                ),
                "reinforcement.number_of_layers": c2.text_input(
                    "Число слоёв *", value=reinforcement.number_of_layers or ""
                ),
                "reinforcement.depth_mm": c3.text_input(
                    "Глубина, мм *", value=reinforcement.depth_mm or ""
                ),
                "reinforcement.spacing_mm": c4.text_input(
                    "Шаг, мм *", value=reinforcement.spacing_mm or ""
                ),
                "reinforcement.length_mm": c1.text_input(
                    "Длина, мм *", value=reinforcement.length_mm or ""
                ),
                "reinforcement.width_mm": c2.text_input(
                    "Ширина, мм *", value=reinforcement.width_mm or ""
                ),
                "reinforcement.bar_diameter_or_aperture_mm": c3.text_input(
                    "Диаметр/ячейка, мм *",
                    value=reinforcement.bar_diameter_or_aperture_mm or "",
                ),
                "reinforcement.orientation": c4.text_input(
                    "Ориентация *", value=reinforcement.orientation
                ),
            }
            custom_text = st.text_area(
                "Дополнительные параметры JSON",
                value=json.dumps(
                    reinforcement.custom_parameters,
                    ensure_ascii=False,
                    indent=2,
                ),
            )
        else:
            custom_text = json.dumps(
                passport.reinforcement.custom_parameters, ensure_ascii=False
            )

        comment = st.text_area("Комментарий", value=passport.comment)
        submitted = st.form_submit_button("Применить паспорт", type="primary")

    if submitted:
        try:
            def reject_json_constant(value: str) -> None:
                raise ValueError(f"Недопустимая JSON-константа {value}.")

            custom_parameters = json.loads(
                custom_text or "{}", parse_constant=reject_json_constant
            )
            if not isinstance(custom_parameters, dict):
                raise ValueError("custom_parameters должен быть JSON-объектом.")
            previous_indicator_payload = {
                channel: values.to_dict() if values is not None else None
                for channel, values in passport.indicator_passports.items()
            }
            metrology_changed = any(
                (
                    test_date != passport.test_date,
                    number_of_indicators != passport.number_of_indicators,
                    indicator_payload != previous_indicator_payload,
                    settlement_aggregation != passport.settlement_aggregation,
                    list(aggregation_channels)
                    != list(passport.settlement_aggregation_channels),
                    (primary_channel or None) != passport.settlement_primary_channel,
                    missing_policy != passport.settlement_missing_channel_policy,
                    metrology_status != passport.metrology_status,
                )
            )
            if (
                metrology_status == "confirmed"
                and metrology_changed
                and not metrology_reason.strip()
            ):
                raise ValueError(
                    "Для изменения или подтверждения поканальной метрологии укажите причину."
                )
            changes: dict[str, Any] = {
                "project_name": project_name,
                "series_name": series_name,
                "test_name": test_name,
                "archive_number": archive_number,
                "test_date": test_date,
                "operator": operator,
                "laboratory_or_site": laboratory_or_site,
                "test_scope": test_scope,
                "protocol_type": protocol_type,
                "group_name": group_name,
                "is_reinforced": is_reinforced,
                "baseline_group": baseline_group,
                "pair_id": pair_id if pair_id != "" else None,
                "soil_type": soil_type,
                "soil_batch": soil_batch,
                "reinforcement_type": reinforcement_type,
                "stamp_shape": stamp_shape,
                "stamp_diameter_mm": stamp_diameter,
                "stamp_area_m2": stamp_area,
                "load_kind": load_kind,
                "load_unit": load_unit,
                "load_factor": load_factor,
                "load_zero": load_zero,
                "lever_ratio": lever_ratio,
                "number_of_indicators": number_of_indicators,
                "indicator_passports": indicator_payload,
                "settlement_aggregation": settlement_aggregation,
                "settlement_aggregation_channels": aggregation_channels,
                "settlement_primary_channel": primary_channel or None,
                "settlement_missing_channel_policy": missing_policy,
                "metrology_status": metrology_status,
                "comment": comment,
                "reinforcement.custom_parameters": custom_parameters,
                **reinforcement_changes,
            }
            if service.update_passport(
                changes,
                author=actor,
                reason=metrology_reason.strip() or "manual_edit",
            ):
                _bump_editor()
                st.success("Паспорт сохранён; изменение записано в аудит.")
                st.rerun()
        except (ValueError, ManualEntryServiceError) as exc:
            st.error(str(exc))

    with st.expander("Явно копировать паспорт между каналами"):
        assigned_channels = [
            channel
            for channel, values in service.draft.passport.indicator_passports.items()
            if values is not None
        ]
        if not assigned_channels:
            st.info(
                "Сначала вручную заполните и сохраните паспорт исходного канала."
            )
        else:
            copy_source = st.selectbox(
                "Исходный канал",
                assigned_channels,
                key=f"{key_prefix}_passport_copy_source",
            )
            copy_targets = st.multiselect(
                "Целевые каналы",
                [
                    channel
                    for channel in (
                        "indicator_1",
                        "indicator_2",
                        "indicator_3",
                        "indicator_4",
                        "reference_indicator",
                    )
                    if channel != copy_source
                ],
                key=f"{key_prefix}_passport_copy_targets",
            )
            copy_reason = st.text_input(
                "Причина копирования *",
                key=f"{key_prefix}_passport_copy_reason",
                help="Команда и причина сохраняются в audit trail для каждого канала.",
            )
            if st.button(
                "Копировать паспорт в каналы",
                key=f"{key_prefix}_passport_copy_button",
            ):
                try:
                    changed = service.copy_indicator_passport(
                        copy_source,
                        copy_targets,
                        author=actor,
                        reason=copy_reason,
                    )
                except ManualEntryServiceError as exc:
                    st.error(str(exc))
                else:
                    if changed:
                        _bump_editor()
                        st.success(
                            "Паспорта скопированы явно; независимые копии и причина записаны в аудит."
                        )
                        st.rerun()
                    else:
                        st.info("Изменений нет.")


def _run_action(action, *, success: str) -> None:
    try:
        action()
    except (ManualEntryServiceError, HistoryEmptyError) as exc:
        st.error(str(exc))
        return
    _bump_editor()
    st.success(success)
    st.rerun()


def _table_editor(service: ManualEntryService, actor: str, *, key_prefix: str) -> None:
    st.subheader("2. Таблица первичных отсчётов")
    st.caption(
        "Raw-значения хранятся как текст. Поддерживаются Tab/Enter/стрелки и "
        "обычная вставка из Excel; sequence_no рассчитывается по видимому порядку."
    )
    n_indicators = service.draft.passport.number_of_indicators or 1
    frame = service.editor_frame(n_indicators=n_indicators)
    visible_columns = [
        column for column in frame.columns if column != EDITOR_UUID_COLUMN
    ]
    editor_key = f"{key_prefix}_editor_{st.session_state.get(MANUAL_EDITOR_GENERATION_KEY, 0)}"
    column_config: dict[str, Any] = {
        EDITOR_UUID_COLUMN: None,
        "sequence_no": st.column_config.NumberColumn("seq", disabled=True),
        "stage_no": st.column_config.TextColumn("stage"),
        "branch": st.column_config.SelectboxColumn(
            "branch", options=list(MANUAL_BRANCHES), required=True
        ),
        "elapsed_time_s": st.column_config.TextColumn("time, s"),
        "timestamp": st.column_config.TextColumn("timestamp"),
        "load_raw": st.column_config.TextColumn("load raw"),
        "row_status": st.column_config.SelectboxColumn(
            "status", options=list(MANUAL_ROW_STATUSES), required=True
        ),
        "comment": st.column_config.TextColumn("comment"),
    }
    for index in range(1, 5):
        column_config[f"indicator_{index}_raw"] = st.column_config.TextColumn(
            f"indicator {index} raw"
        )
    edited = st.data_editor(
        frame,
        key=editor_key,
        column_order=visible_columns,
        column_config=column_config,
        disabled=["sequence_no"],
        hide_index=True,
        num_rows="fixed",
        width="stretch",
    )
    try:
        service.apply_editor_frame(
            edited, author=actor, confirm_deletions=False, reason="manual_edit"
        )
    except ManualEntryServiceError as exc:
        st.error(f"Изменение таблицы отклонено: {exc}")

    row_count = len(service.draft.rows)
    selected_key = f"{key_prefix}_selected_row_value"
    selected_default = _bounded_widget_value(
        selected_key,
        minimum=1,
        maximum=max(1, row_count),
        default=1,
    )
    selected = int(
        st.number_input(
            "Активная строка (1-based)",
            min_value=1,
            max_value=max(1, row_count),
            value=selected_default,
            step=1,
            key=selected_key,
        )
    )
    selected_index = selected - 1
    c1, c2, c3, c4, c5 = st.columns(5)
    if c1.button("Добавить", key=f"{key_prefix}_add"):
        _run_action(
            lambda: service.add_row(author=actor), success="Строка добавлена."
        )
    if c2.button("Вставить до", key=f"{key_prefix}_before"):
        _run_action(
            lambda: service.insert_row(selected_index, position="before", author=actor),
            success="Строка вставлена.",
        )
    if c3.button("Вставить после", key=f"{key_prefix}_after"):
        _run_action(
            lambda: service.insert_row(selected_index, position="after", author=actor),
            success="Строка вставлена.",
        )
    if c4.button("Дублировать", key=f"{key_prefix}_duplicate"):
        _run_action(
            lambda: service.duplicate_row(selected_index, author=actor),
            success="Строка продублирована с новым UUID.",
        )
    if c5.button("Перенумеровать", key=f"{key_prefix}_renumber"):
        _run_action(lambda: service.renumber(author=actor), success="Строки перенумерованы.")

    confirm_delete = st.checkbox(
        "Подтверждаю удаление активной строки",
        key=f"{key_prefix}_confirm_delete",
    )
    c1, c2, c3, c4 = st.columns(4)
    if c1.button("Удалить", key=f"{key_prefix}_delete"):
        _run_action(
            lambda: service.delete_row(
                selected_index, confirmed=confirm_delete, author=actor
            ),
            success="Строка удалена; снимок сохранён в аудите.",
        )
    if c2.button("Заполнить ступени", key=f"{key_prefix}_fill_stages"):
        _run_action(
            lambda: service.fill_stages(start=0, step=1, author=actor),
            success="Номера ступеней заполнены.",
        )
    if c3.button(
        "Отменить", key=f"{key_prefix}_undo", disabled=not service.can_undo
    ):
        _run_action(lambda: service.undo(author=actor), success="Изменение отменено.")
    if c4.button(
        "Повторить", key=f"{key_prefix}_redo", disabled=not service.can_redo
    ):
        _run_action(lambda: service.redo(author=actor), success="Изменение повторено.")

    with st.expander("Буфер обмена и прямоугольные диапазоны"):
        paste_text = st.text_area(
            "TSV из Excel",
            key=f"{key_prefix}_paste_text",
            placeholder="0\tloading\t0\t0,00\t9,80",
        )
        c1, c2 = st.columns(2)
        paste_row_key = f"{key_prefix}_paste_row"
        paste_row_default = _bounded_widget_value(
            paste_row_key,
            minimum=1,
            maximum=max(1, row_count + 1),
            default=min(selected, max(1, row_count + 1)),
        )
        paste_row = int(
            c1.number_input(
                "Начальная строка",
                min_value=1,
                max_value=max(1, row_count + 1),
                value=paste_row_default,
                step=1,
                key=paste_row_key,
            )
        )
        paste_column = c2.selectbox(
            "Начальный столбец",
            list(MANUAL_EDITOR_COLUMNS[1:]),
            key=f"{key_prefix}_paste_column",
        )
        if st.button("Вставить TSV", key=f"{key_prefix}_paste"):
            _run_action(
                lambda: service.paste_block(
                    paste_row - 1,
                    paste_column,
                    paste_text,
                    expand_rows=True,
                    author=actor,
                ),
                success="Блок вставлен атомарно.",
            )

        c1, c2 = st.columns(2)
        range_start_key = f"{key_prefix}_range_start"
        range_start_default = _bounded_widget_value(
            range_start_key,
            minimum=1,
            maximum=max(1, row_count),
            default=1,
        )
        range_start = int(
            c1.number_input(
                "Первая строка диапазона",
                min_value=1,
                max_value=max(1, row_count),
                value=range_start_default,
                key=range_start_key,
            )
        )
        range_end_key = f"{key_prefix}_range_end"
        range_end_default = _bounded_widget_value(
            range_end_key,
            minimum=range_start,
            maximum=max(range_start, row_count),
            default=max(range_start, row_count),
        )
        range_end = int(
            c2.number_input(
                "Последняя строка диапазона",
                min_value=range_start,
                max_value=max(range_start, row_count),
                value=range_end_default,
                key=range_end_key,
            )
        )
        range_columns = st.multiselect(
            "Столбцы диапазона",
            list(MANUAL_EDITOR_COLUMNS[1:]),
            default=["stage_no", "branch", "elapsed_time_s", "load_raw"],
            key=f"{key_prefix}_range_columns",
        )
        c1, c2 = st.columns(2)
        if c1.button("Очистить диапазон", key=f"{key_prefix}_clear"):
            cells = [
                (row, column)
                for row in range(range_start - 1, range_end)
                for column in range_columns
            ]
            _run_action(
                lambda: service.clear_cells(cells, author=actor),
                success="Ячейки очищены в пустые значения, нули не подставлялись.",
            )
        if c2.button("Копировать диапазон", key=f"{key_prefix}_copy"):
            if not range_columns:
                st.error("Выберите хотя бы один столбец.")
            else:
                indices = [MANUAL_EDITOR_COLUMNS.index(name) for name in range_columns]
                if indices != list(range(min(indices), max(indices) + 1)):
                    st.error("Для TSV-копирования выберите смежные столбцы.")
                else:
                    st.session_state[f"{key_prefix}_copied_tsv"] = service.copy_block(
                        range_start - 1,
                        range_end - 1,
                        min(indices),
                        max(indices),
                    )
        copied = st.session_state.get(f"{key_prefix}_copied_tsv")
        if copied is not None:
            st.text_area(
                "Скопированный TSV",
                value=copied,
                key=f"{key_prefix}_copied_output",
            )


def _validation_panel(
    service: ManualEntryService,
    validation: ManualValidationResult,
    *,
    key_prefix: str,
) -> None:
    st.subheader("3. Валидация")
    errors = sum(bool(issue.blocks_processing) for issue in validation.issues)
    warnings = len(validation.issues) - errors
    c1, c2, c3 = st.columns(3)
    c1.metric("Критических ошибок", errors)
    c2.metric("Остальных сообщений", warnings)
    c3.metric("Анализ", "доступен" if validation.can_analyze else "заблокирован")
    if validation.issues:
        frame = validation.to_frame()
        st.dataframe(_ui_safe_frame(frame), hide_index=True, width="stretch")
        targets = [
            (index, issue)
            for index, issue in enumerate(validation.issues)
            if getattr(issue, "rows", None)
        ]
        if targets:
            labels = [
                f"{issue.code}: строка {int(issue.rows[0]) + 1}, {issue.column or 'строка'}"
                for _, issue in targets
            ]
            selected = st.selectbox(
                "Перейти к ошибке", labels, key=f"{key_prefix}_issue_target"
            )
            if st.button("Выбрать строку ошибки", key=f"{key_prefix}_jump_issue"):
                position = labels.index(selected)
                issue = targets[position][1]
                st.session_state[f"{key_prefix}_selected_row_value"] = int(
                    issue.rows[0]
                ) + 1
                st.rerun()
    else:
        st.success("Ошибок не обнаружено.")

    editor = service.editor_frame(
        n_indicators=service.draft.passport.number_of_indicators or 1
    ).drop(columns=[EDITOR_UUID_COLUMN], errors="ignore")
    error_rows = {
        int(row)
        for issue in validation.issues
        if bool(issue.blocks_processing)
        for row in getattr(issue, "rows", [])
        if isinstance(row, int)
    }
    if error_rows and len(editor):
        styles = pd.DataFrame("", index=editor.index, columns=editor.columns)
        for issue in validation.blocking_issues:
            for row in getattr(issue, "rows", []):
                if not isinstance(row, int) or row not in styles.index:
                    continue
                if issue.column in styles.columns:
                    styles.at[row, issue.column] = "background-color: #ffd6d6"
                else:
                    styles.loc[row, :] = "background-color: #ffd6d6"

        st.caption("Подсветка ячеек и строк с критическими ошибками")
        st.dataframe(
            editor.style.apply(lambda _: styles, axis=None),
            hide_index=True,
            width="stretch",
        )


def _preview(
    service: ManualEntryService,
    validation: ManualValidationResult,
    *,
    computed: ManualPreview | None = None,
) -> ManualPreview | None:
    st.subheader("4. Предварительный расчёт")
    st.caption(
        "Предварительный просмотр не является утверждённым отчётным графиком. "
        "Расчёт выполняется тем же pipeline, что и Excel."
    )
    if not validation.can_analyze:
        st.warning("Исправьте критические ошибки. Черновик при этом можно скачать.")
        return computed
    if computed is None:
        st.error("Предварительный pipeline не сформировал проверяемый результат.")
        return None
    prepared = computed.prepared
    pipeline_issues = computed.issues
    blockers = [issue for issue in pipeline_issues if bool(issue.blocks_processing)]
    if blockers:
        st.error("Pipeline обнаружил блокирующие ошибки.")
        st.dataframe(
            _ui_safe_frame(pd.DataFrame([issue.to_dict() for issue in pipeline_issues])),
            hide_index=True,
            width="stretch",
        )
        return computed

    raw_columns = [
        column
        for column in (
            "source_sequence_no",
            "raw_stage",
            "raw_load",
            "raw_indicator_1",
            "raw_indicator_2",
            "raw_indicator_3",
            "raw_indicator_4",
            "row_status",
            "manual_row_uuid",
        )
        if column in prepared
    ]
    calculated = [
        column
        for column in (
            "parsed_stage",
            "parsed_load",
            "parsed_indicator",
            "stage",
            "branch",
            "F_kN",
            "p_kPa",
            "settlement_raw_mm",
            "settlement_mm",
            "is_failure",
        )
        if column in prepared
    ]
    st.dataframe(
        _ui_safe_frame(prepared[[*raw_columns, *calculated]]),
        hide_index=True,
        width="stretch",
    )
    conversion = indicator_audit_frame(prepared)
    if not conversion.empty:
        st.caption("Проверяемое преобразование показаний индикаторов")
        conversion_columns = [
            "test_id",
            "channel",
            "sequence_index",
            "manual_row_uuid",
            "original_reading",
            "turn_number",
            "computed_increment_mm",
            "cumulative_before_correction_mm",
            "applied_correction_mm",
            "cumulative_settlement_mm",
            "warning",
            "conversion_method",
        ]
        st.dataframe(
            _ui_safe_frame(
                conversion[
                    [column for column in conversion_columns if column in conversion]
                ]
            ),
            hide_index=True,
            width="stretch",
        )
    events = indicator_event_frame(prepared)
    if not events.empty:
        with st.expander("Переходы шкалы, обратный ход и коррекции"):
            st.dataframe(
                _ui_safe_frame(events), hide_index=True, width="stretch"
            )
    nonblocking = [
        issue for issue in pipeline_issues if not bool(issue.blocks_processing)
    ]
    if nonblocking:
        st.caption("Предупреждения общего pipeline")
        st.dataframe(
            _ui_safe_frame(pd.DataFrame(issue.to_dict() for issue in nonblocking)),
            hide_index=True,
            width="stretch",
        )
    area = pd.to_numeric(prepared.get("stamp_area_m2"), errors="coerce").dropna()
    st.metric("Площадь штампа, м²", f"{float(area.iloc[0]):.6g}" if len(area) else "—")
    failures = failure_summary(prepared)
    st.dataframe(failures, hide_index=True, width="stretch")
    try:
        axis_mode = "p-s" if pd.to_numeric(prepared["p_kPa"], errors="coerce").notna().any() else "F-s"
        output = plot_curves(prepared, mode="raw_protocol", axis_mode=axis_mode)
        st.pyplot(output.figure, width="stretch")
        plt.close(output.figure)
        for warning in output.warnings:
            st.warning(warning)
    except Exception as exc:
        st.warning(f"Предварительный график пока недоступен: {exc}")
    return computed


def _draft_controls(
    service: ManualEntryService,
    validation: ManualValidationResult,
    actor: str,
    *,
    key_prefix: str,
) -> None:
    st.markdown("##### Черновик и передача в анализ")
    active_hash = st.session_state.get(MANUAL_ACTIVE_HASH_KEY)
    if active_hash and active_hash != service.draft.sha256:
        st.warning(
            "Активный расчёт относится к предыдущему snapshot. Текущие правки ещё не переданы в анализ."
        )
    st.caption(
        f"schema={service.draft.schema_version}; draft SHA-256={service.draft.sha256[:16]}…; "
        f"audit events={len(service.draft.audit_events)}"
    )
    c1, c2 = st.columns(2)
    c1.download_button(
        "Сохранить черновик JSON",
        service.to_json().encode("utf-8"),
        f"{service.draft.passport.test_id or 'manual_test'}_draft.json",
        "application/json",
        key=f"{key_prefix}_download_draft",
    )
    uploaded = c2.file_uploader(
        "Открыть черновик JSON",
        type=["json"],
        key=f"{key_prefix}_draft_upload",
    )
    if st.button("Загрузить выбранный черновик", key=f"{key_prefix}_load_draft"):
        if uploaded is None:
            st.error("Выберите JSON-файл черновика.")
        else:
            try:
                replacement = ManualEntryService.from_json(
                    uploaded.getvalue(), author=actor
                )
            except ValueError as exc:
                st.error(str(exc))
            else:
                st.session_state[MANUAL_SERVICE_KEY] = replacement
                _bump_editor()
                st.success("Черновик открыт без нормализации raw-значений.")
                st.rerun()

    if st.button(
        "Передать snapshot в анализ",
        type="primary",
        disabled=not validation.can_analyze,
        key=f"{key_prefix}_activate",
    ):
        event = ManualAuditEvent.create(
            author=actor,
            action="activate_analysis_snapshot",
            entity_id=service.draft.draft_id,
            field=None,
            old_value={"active_sha256": active_hash},
            new_value={"source_type": "manual", "status": "activated"},
            reason="manual_activation",
        )
        service.draft.audit_events.append(event)
        service.draft.updated_at = utc_now_iso()
        st.session_state[MANUAL_ACTIVE_DRAFT_KEY] = service.draft.to_dict()
        st.session_state[MANUAL_ACTIVE_HASH_KEY] = service.draft.sha256
        # The source radio is already instantiated in this run.  Defer its
        # state change until the top of the next run to satisfy Streamlit's
        # widget-state contract.
        st.session_state[MANUAL_SOURCE_REQUEST_KEY] = True
        st.success("Snapshot заморожен и будет передан в общий pipeline.")
        st.rerun()

    with st.expander("Audit trail ручного черновика"):
        st.dataframe(
            _ui_safe_frame(
                pd.DataFrame([event.to_dict() for event in service.draft.audit_events])
            ),
            hide_index=True,
            width="stretch",
        )
    st.info("Сохранение в SQLite-архив появится только в TASK 13; сейчас доступен JSON-черновик.")


def render_manual_entry(*, key_prefix: str = "manual_entry") -> None:
    """Render all four TASK 12 zones without activating data implicitly."""

    st.header("Ввод вручную")
    st.caption("Источник первичных данных: manual · активное состояние: черновик")
    actor = st.text_input(
        "Автор изменений *",
        value="local-user",
        key=f"{key_prefix}_actor",
    ).strip() or "local-user"
    service = get_manual_service(author=actor)
    _passport_form(service, actor, key_prefix=key_prefix)
    _table_editor(service, actor, key_prefix=key_prefix)
    computed: ManualPreview | None = None
    try:
        from .manual_entry_adapter import adapt_manual_draft

        bundle = adapt_manual_draft(service.draft)
        prepared, pipeline_issues = bundle.prepare()
        validation = ManualValidationResult(
            issues=merge_manual_issues(bundle.issues, pipeline_issues),
            adapter_issues=list(bundle.validation.adapter_issues),
            pipeline_issues=merge_manual_issues(
                bundle.validation.pipeline_issues, pipeline_issues
            ),
        )
        computed = ManualPreview(
            prepared=prepared,
            issues=list(pipeline_issues),
            bundle=bundle,
        )
    except Exception as exc:
        base_validation = validate_manual_draft(service.draft)
        runtime_issue = ValidationIssue(
            "error",
            "manual_preview_exception",
            f"Предварительный pipeline завершился ошибкой: {exc}",
            raw_value=str(exc),
            suggested_action="Проверьте паспорт и исходные строки; черновик можно сохранить.",
        )
        validation = ManualValidationResult(
            issues=merge_manual_issues(base_validation.issues, [runtime_issue]),
            adapter_issues=list(base_validation.issues),
            pipeline_issues=[runtime_issue],
        )
    _validation_panel(service, validation, key_prefix=key_prefix)
    _preview(service, validation, computed=computed)
    _draft_controls(service, validation, actor, key_prefix=key_prefix)


__all__ = [
    "MANUAL_ACTIVE_DRAFT_KEY",
    "MANUAL_ACTIVE_HASH_KEY",
    "MANUAL_EDITOR_GENERATION_KEY",
    "MANUAL_SERVICE_KEY",
    "MANUAL_SOURCE_REQUEST_KEY",
    "active_manual_draft",
    "get_manual_service",
    "render_manual_entry",
]
