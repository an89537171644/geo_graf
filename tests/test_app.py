from __future__ import annotations

import csv
from io import BytesIO, StringIO
from pathlib import Path

from openpyxl import Workbook
from streamlit.testing.v1 import AppTest


ROOT = Path(__file__).resolve().parents[1]


def _user_upload_app(protocol_name: str, protocol_bytes: bytes, mime_type: str) -> AppTest:
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120).run()
    source = app.sidebar.radio[0]
    source.set_value(source.options[1])
    app.run()
    uploaders = app.get("file_uploader")
    uploaders[0].upload(protocol_name, protocol_bytes, mime_type)
    uploaders[1].upload(
        "metadata.json",
        (ROOT / "examples" / "demo_metadata.json").read_bytes(),
        "application/json",
    )
    app.run(timeout=120)
    return app


def _demo_xlsx_bytes(*, two_sheets: bool = False) -> bytes:
    text = (ROOT / "examples" / "demo_protocol.csv").read_text(encoding="utf-8")
    rows = list(csv.reader(StringIO(text)))
    workbook_rows = rows[:7] if two_sheets else rows
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Protocol"
    for row in workbook_rows:
        sheet.append(row)
    if two_sheets:
        second = workbook.create_sheet("Protocol copy")
        for row in workbook_rows:
            second.append(row)
    buffer = BytesIO()
    workbook.save(buffer)
    workbook.close()
    return buffer.getvalue()


def test_streamlit_user_csv_path_has_no_exceptions() -> None:
    app = _user_upload_app(
        "demo_protocol.csv",
        (ROOT / "examples" / "demo_protocol.csv").read_bytes(),
        "text/csv",
    )

    assert not app.exception
    assert len(app.tabs) == 8
    assert app.tabs[-1].label == "Ввод вручную"
    assert any(item.value == "Паспорта индикаторов" for item in app.subheader)
    assert any(
        item.value == "Преобразование показаний индикаторов" for item in app.subheader
    )
    assert any(
        item.label == "Скачать indicator_processing_audit.csv"
        for item in app.download_button
    )


def test_streamlit_user_strict_xlsx_path_has_no_exceptions() -> None:
    app = _user_upload_app(
        "demo_protocol.xlsx",
        _demo_xlsx_bytes(),
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    assert not app.exception
    assert len(app.tabs) == 8


def test_corrupt_xlsx_has_controlled_diagnostic_download() -> None:
    app = _user_upload_app(
        "broken.xlsx",
        b"PK\x03\x04not-a-valid-xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    assert not app.exception
    assert len(app.tabs) == 0
    assert any(
        item.label == "Скачать диагностический пакет ZIP"
        for item in app.download_button
    )


def test_interactive_mapping_preview_and_confirmation_are_scoped_to_sheet() -> None:
    app = _user_upload_app(
        "demo_protocol.xlsx",
        _demo_xlsx_bytes(two_sheets=True),
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
    # strict defaults to all sheets, so duplicated IDs across the two protocol
    # sheets must block before the user explicitly switches to interactive scope.
    assert len(app.tabs) == 0
    assert any(
        item.label == "Скачать диагностический пакет ZIP"
        for item in app.download_button
    )
    import_mode = next(
        item for item in app.sidebar.radio if item.label == "Режим Excel-импорта"
    )
    import_mode.set_value(import_mode.options[1])
    app.run(timeout=120)

    assert not app.exception
    assert any(item.value == "Предпросмотр сопоставления Excel" for item in app.subheader)
    confirmation = next(
        item
        for item in app.sidebar.checkbox
        if item.label == "Подтверждаю сопоставление столбцов"
    )
    assert confirmation.value is False
    confirmation.set_value(True)
    app.run(timeout=120)
    assert not app.exception
    assert len(app.tabs) == 8

    sheet = next(item for item in app.sidebar.selectbox if item.label == "Лист Excel")
    sheet.set_value(sheet.options[1])
    app.run(timeout=120)
    new_confirmation = next(
        item
        for item in app.sidebar.checkbox
        if item.label == "Подтверждаю сопоставление столбцов"
    )
    assert new_confirmation.value is False
    assert len(app.tabs) == 0
