from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

from soilstamp.gui_manual_entry import MANUAL_SERVICE_KEY


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


def test_invalid_manual_draft_can_be_downloaded_but_not_activated() -> None:
    app = _open_manual_entry()

    download = next(
        item
        for item in app.download_button
        if item.label == "Сохранить черновик JSON"
    )
    activate = _button(app, "Передать snapshot в анализ")

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

    activate = _button(app, "Передать snapshot в анализ")
    assert activate.disabled is False
    activate.click()
    app.run(timeout=120)
    # AppTest exposes the stop-rendered standalone tree after the handler's
    # st.rerun request.  One further cycle exercises the active snapshot path.
    app.run(timeout=120)

    assert not app.exception
    assert [tab.label for tab in app.tabs] == [
        "Импорт и QC",
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
