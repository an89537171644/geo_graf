from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
from pandas.testing import assert_frame_equal
from streamlit.testing.v1 import AppTest

from soilstamp.gui_manual_entry import (
    MANUAL_SERVICE_KEY,
    _localized_manual_issue_frame,
)
from soilstamp.manual_entry_validation import validate_manual_draft
from soilstamp.schema import ValidationIssue


ROOT = Path(__file__).resolve().parents[1]
MANUAL_SOURCE = "Ввод вручную"


def _open_manual_entry() -> AppTest:
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120).run(
        timeout=120
    )
    assert not app.exception

    source = next(item for item in app.sidebar.radio if item.label == "Источник")
    assert MANUAL_SOURCE in source.options
    source.set_value(MANUAL_SOURCE)
    app.run(timeout=120)

    assert not app.exception
    return app


def _button(app: AppTest, label: str):
    return next(item for item in app.button if item.label == label)


def test_manual_issue_display_localizes_prose_without_mutating_canonical_issue() -> None:
    issue = ValidationIssue(
        "warning",
        "manual_machine_code",
        (
            "В строке measurement поле metadata.load_factor и sequence_no "
            "используют default."
        ),
        column="sequence_no",
        raw_value="metadata.load_factor=default; measurement; sequence_no",
        suggested_action="Исправьте indicator_1_raw в metadata JSON.",
    )
    canonical = pd.DataFrame([issue.to_dict()])
    snapshot = canonical.copy(deep=True)

    displayed = _localized_manual_issue_frame(canonical)

    assert_frame_equal(canonical, snapshot)
    assert issue.code == "manual_machine_code"
    assert "metadata.load_factor" in issue.message
    assert "measurement" in issue.message
    assert displayed.loc[0, "Код"] == "manual_machine_code"
    assert displayed.loc[0, "Столбец"] == "№ по порядку"
    assert "измерение" in displayed.loc[0, "Сообщение"]
    assert "Коэффициент пересчёта нагрузки" in displayed.loc[0, "Сообщение"]
    assert "№ по порядку" in displayed.loc[0, "Сообщение"]
    assert "значение по умолчанию" in displayed.loc[0, "Сообщение"]
    assert "Показание индикатора 1" in displayed.loc[0, "Рекомендуемое действие"]
    assert "JSON паспорта и служебных параметров" in displayed.loc[
        0, "Рекомендуемое действие"
    ]
    assert displayed.loc[0, "Исходное значение"] == issue.raw_value


def test_manual_source_opens_without_excel_and_renders_four_zones() -> None:
    app = _open_manual_entry()

    assert [tab.label for tab in app.tabs] == [MANUAL_SOURCE]
    assert [item.value for item in app.subheader] == [
        "1. Паспорт опыта",
        "2. Таблица первичных отсчётов",
        "3. Валидация",
        "4. Предварительный расчёт",
    ]
    labels = {item.label for item in app.text_input}
    assert "Контрольная группа (необязательно)" in labels
    assert "ID пары / блока (необязательно)" in labels


    issue_table = next(
        table.value
        for table in app.dataframe
        if hasattr(table.value, "columns")
        and {"Код", "Сообщение", "Столбец", "Рекомендуемое действие"}.issubset(
            table.value.columns
        )
    )
    visible_issue_prose = " ".join(
        issue_table[["Сообщение", "Столбец", "Рекомендуемое действие"]]
        .astype(str)
        .to_numpy()
        .ravel()
    )
    forbidden_visible_tokens = {
        "measurement",
        "metadata",
        "default",
        "project_name",
        "series_name",
        "laboratory_or_site",
        "group_name",
        "soil_type",
        "soil_batch",
        "test_name",
        "archive_number",
        "stamp_diameter_mm",
        "stamp_area_m2",
        "load_factor",
        "load_zero",
        "lever_ratio",
        "number_of_indicators",
        "settlement_aggregation",
        "indicator_resolution_mm",
        "division_mm",
        "load_raw",
        "indicator_1_raw",
        "sequence_no",
    }
    assert not {
        token
        for token in forbidden_visible_tokens
        if re.search(
            rf"(?<![\w-]){re.escape(token)}(?![\w-])",
            visible_issue_prose,
            flags=re.IGNORECASE,
        )
    }
    assert "значение по умолчанию" in visible_issue_prose
    assert "missing_manual_measurement" in set(issue_table["Код"])

    canonical_validation = validate_manual_draft(
        app.session_state[MANUAL_SERVICE_KEY].draft
    )
    assert any("measurement" in issue.message for issue in canonical_validation.issues)


def test_invalid_manual_draft_can_be_downloaded_but_not_activated() -> None:
    app = _open_manual_entry()

    download = next(
        item
        for item in app.download_button
        if item.label == "Сохранить черновик JSON"
    )
    activate = _button(app, "Передать снимок в анализ")

    assert download.disabled is False
    assert activate.disabled is True
    assert any(
        "Исправьте критические ошибки" in str(item.value)
        for item in app.warning
    )


def test_gui_preserves_literal_pair_id_for_auditable_validation() -> None:
    app = _open_manual_entry()
    pair_input = next(
        item
        for item in app.text_input
        if item.label == "ID пары / блока (необязательно)"
    )
    pair_input.set_value(" P1 ")
    _button(app, "Применить паспорт").click()
    app.run(timeout=120)

    assert not app.exception
    service = app.session_state[MANUAL_SERVICE_KEY]
    assert service.draft.passport.pair_id == " P1 "
    matching_events = [
        event
        for event in service.draft.audit_events
        if event.field == "pair_id"
    ]
    assert matching_events
    assert matching_events[-1].new_value == " P1 "


def test_valid_manual_demo_can_be_activated_in_common_pipeline() -> None:
    demo = ROOT / "examples" / "manual_entry_demo.json"
    assert demo.is_file()
    app = _open_manual_entry()

    uploader = next(
        item
        for item in app.get("file_uploader")
        if item.label == "Открыть черновик JSON"
    )
    uploader.upload(demo.name, demo.read_bytes(), "application/json")
    app.run(timeout=120)
    assert not app.exception

    _button(app, "Загрузить выбранный черновик").click()
    app.run(timeout=120)
    assert not app.exception

    # Re-saving the one-indicator passport must not create phantom serials or
    # turn the previously valid draft into a blocking state.
    _button(app, "Применить паспорт").click()
    app.run(timeout=120)
    assert not app.exception

    activate = _button(app, "Передать снимок в анализ")
    assert activate.disabled is False
    activate.click()
    app.run(timeout=120)
    # AppTest exposes the stop-rendered standalone tree after the handler's
    # st.rerun request.  One further cycle exercises the active snapshot path.
    app.run(timeout=120)

    assert not app.exception
    assert [tab.label for tab in app.tabs] == [
        "Импорт и контроль качества",
        "Коррекции",
        "Графики",
        "pcr и E",
        "Сравнение групп",
        "Доп. анализ",
        "Отчёт и журнал",
        MANUAL_SOURCE,
    ]
    source = next(item for item in app.sidebar.radio if item.label == "Источник")
    assert source.value == MANUAL_SOURCE


def test_gui_copy_to_channels_is_explicit_and_audited() -> None:
    demo = ROOT / "examples" / "manual_entry_demo.json"
    app = _open_manual_entry()
    uploader = next(
        item
        for item in app.get("file_uploader")
        if item.label == "Открыть черновик JSON"
    )
    uploader.upload(demo.name, demo.read_bytes(), "application/json")
    app.run(timeout=120)
    _button(app, "Загрузить выбранный черновик").click()
    app.run(timeout=120)

    target = next(
        item for item in app.multiselect if item.label == "Целевые каналы"
    )
    target.set_value(["indicator_2"])
    reason = next(
        item for item in app.text_input if item.label == "Причина копирования *"
    )
    reason.set_value("установка одинаковой серии с последующей проверкой")
    _button(app, "Копировать паспорт в каналы").click()
    app.run(timeout=120)

    assert not app.exception
    service = app.session_state[MANUAL_SERVICE_KEY]
    copied = service.draft.passport.indicator_passports["indicator_2"]
    assert copied is not None
    assert copied.assignment_status == "review_required"
    events = [
        event
        for event in service.draft.audit_events
        if event.action == "copy_indicator_passport"
        and event.entity_id.endswith(":indicator_2")
    ]
    assert events
    assert events[-1].reason == "установка одинаковой серии с последующей проверкой"


def test_confirmed_metrology_cannot_change_without_explicit_reason() -> None:
    demo = ROOT / "examples" / "manual_entry_demo.json"
    app = _open_manual_entry()
    uploader = next(
        item
        for item in app.get("file_uploader")
        if item.label == "Открыть черновик JSON"
    )
    uploader.upload(demo.name, demo.read_bytes(), "application/json")
    app.run(timeout=120)
    _button(app, "Загрузить выбранный черновик").click()
    app.run(timeout=120)

    instrument = next(
        item for item in app.text_input if item.label == "ID прибора *"
    )
    instrument.set_value("DEMO-IND-CHANGED")
    _button(app, "Применить паспорт").click()
    app.run(timeout=120)

    service = app.session_state[MANUAL_SERVICE_KEY]
    current = service.draft.passport.indicator_passports["indicator_1"]
    assert current is not None
    assert current.instrument_id == "DEMO-IND-001"
    assert any(
        "укажите причину" in str(item.value).casefold() for item in app.error
    )

    reason = next(
        item
        for item in app.text_input
        if item.label == "Причина подтверждения метрологии"
    )
    reason.set_value("замена идентификатора по паспорту прибора")
    _button(app, "Применить паспорт").click()
    app.run(timeout=120)

    service = app.session_state[MANUAL_SERVICE_KEY]
    changed = service.draft.passport.indicator_passports["indicator_1"]
    assert changed is not None
    assert changed.instrument_id == "DEMO-IND-CHANGED"
    assert any(
        event.field == "indicator_passports"
        and event.reason == "замена идентификатора по паспорту прибора"
        for event in service.draft.audit_events
    )


def test_confirmed_experiment_date_cannot_change_without_explicit_reason() -> None:
    demo = ROOT / "examples" / "manual_entry_demo.json"
    app = _open_manual_entry()
    uploader = next(
        item
        for item in app.get("file_uploader")
        if item.label == "Открыть черновик JSON"
    )
    uploader.upload(demo.name, demo.read_bytes(), "application/json")
    app.run(timeout=120)
    _button(app, "Загрузить выбранный черновик").click()
    app.run(timeout=120)

    test_date = next(
        item
        for item in app.text_input
        if item.label == "Дата YYYY-MM-DD (необязательно)"
    )
    original_date = str(test_date.value)
    test_date.set_value("2026-07-12")
    _button(app, "Применить паспорт").click()
    app.run(timeout=120)

    service = app.session_state[MANUAL_SERVICE_KEY]
    assert service.draft.passport.test_date == original_date
    assert service.draft.passport.metrology_status == "confirmed"
    assert any(
        "укажите причину" in str(item.value).casefold() for item in app.error
    )
