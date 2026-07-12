from __future__ import annotations

import csv
import json
from io import BytesIO, StringIO
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import Workbook
from streamlit.testing.v1 import AppTest

from soilstamp.analysis import calculate_moduli_for_test
from soilstamp.data import prepare_measurements
from soilstamp.methodology import ModulusOverrides


ROOT = Path(__file__).resolve().parents[1]


def _user_upload_app(
    protocol_name: str,
    protocol_bytes: bytes,
    mime_type: str,
    *,
    metadata_bytes: bytes | None = None,
) -> AppTest:
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120).run()
    source = app.sidebar.radio[0]
    source.set_value(source.options[1])
    app.run()
    uploaders = app.get("file_uploader")
    uploaders[0].upload(protocol_name, protocol_bytes, mime_type)
    uploaders[1].upload(
        "metadata.json",
        metadata_bytes or (ROOT / "examples" / "demo_metadata.json").read_bytes(),
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
    assert any(item.value == "Агрегация осадки" for item in app.subheader)
    assert any(
        item.label == "Скачать indicator_processing_audit.csv"
        for item in app.download_button
    )


def test_streamlit_singleton_subset_does_not_reuse_metadata_mean_decision() -> None:
    app = _user_upload_app(
        "demo_protocol.csv",
        (ROOT / "examples" / "demo_protocol.csv").read_bytes(),
        "text/csv",
    )
    tests = next(item for item in app.sidebar.multiselect if item.label == "Испытания")
    tests.set_value(["B-01"])
    app.run(timeout=120)
    graph_mode = next(item for item in app.selectbox if item.label == "Режим")
    graph_mode.set_value("antonov_publication")
    app.run(timeout=120)

    assert not app.exception
    assert any(
        "индивидуальная B-01" in str(item.value)
        for item in app.caption
    )


def test_streamlit_metadata_curve_selection_survives_same_context_rerun() -> None:
    app = _user_upload_app(
        "demo_protocol.csv",
        (ROOT / "examples" / "demo_protocol.csv").read_bytes(),
        "text/csv",
    )
    graph_mode = next(item for item in app.selectbox if item.label == "Режим")
    graph_mode.set_value("antonov_publication")
    app.run(timeout=120)
    bootstrap = next(item for item in app.number_input if item.label == "Bootstrap")
    bootstrap.set_value(400)
    app.run(timeout=120)

    assert not app.exception


def test_streamlit_user_strict_xlsx_path_has_no_exceptions() -> None:
    app = _user_upload_app(
        "demo_protocol.xlsx",
        _demo_xlsx_bytes(),
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    assert not app.exception
    assert len(app.tabs) == 8


def test_streamlit_modulus_matches_shared_direct_api_with_conflicting_metadata() -> None:
    metadata = json.loads(
        (ROOT / "examples" / "demo_metadata.json").read_text(encoding="utf-8")
    )
    metadata["modulus_method"] = {
        "profile_id": "antonov_round_stamp_v1",
        "nu": 0.25,
        "shape_factor": 1.0,
        "p_range_kPa": [0.0, 150.0],
        "p_range_source": "explicit",
        "approval": {
            "status": "approved",
            "author": "old-engineer",
            "timestamp_utc": "2026-07-12T05:00:00+00:00",
            "reason": "Superseded coefficients for parity test.",
        },
    }
    protocol_bytes = (ROOT / "examples" / "demo_protocol.csv").read_bytes()
    app = _user_upload_app(
        "demo_protocol.csv",
        protocol_bytes,
        "text/csv",
        metadata_bytes=json.dumps(metadata, ensure_ascii=False).encode("utf-8"),
    )

    profile = next(item for item in app.selectbox if item.label == "Профиль методики E")
    assert profile.value == "antonov_round_stamp_v1"
    assert next(item for item in app.number_input if item.label == "ν").value == 0.30
    assert (
        next(item for item in app.number_input if item.label == "Коэффициент формы").value
        == 0.80
    )
    next(
        item
        for item in app.checkbox
        if item.label == "Подтверждаю диапазон и параметры для условного E_stamp_app"
    ).set_value(True)
    next(
        item for item in app.text_input if item.label == "Обоснование решения"
    ).set_value("Engineer confirms displayed Antonov coefficients.")
    next(item for item in app.button if item.label == "Рассчитать E").click()
    app.run(timeout=120)

    assert not app.exception
    gui_primary = next(
        table.value.iloc[0]
        for table in app.dataframe
        if hasattr(table.value, "columns")
        and "profile_id" in table.value.columns
        and bool(table.value["is_primary"].fillna(False).any())
    )

    raw = pd.read_csv(BytesIO(protocol_bytes))
    prepared, issues = prepare_measurements(raw, metadata, strict_metadata=False)
    assert not [issue for issue in issues if issue.level == "error"]
    direct = calculate_moduli_for_test(
        prepared,
        metadata,
        "B-01",
        manual_confirmation=ModulusOverrides(
            p_range_kpa=(0.0, 150.0),
            p_range_source="explicit",
            nu=0.30,
            shape_factor=0.80,
            approval_status="approved",
            author="demo-operator",
            timestamp_utc="2026-07-12T06:00:00+00:00",
            reason="Engineer confirms displayed Antonov coefficients.",
        ),
        bootstrap=500,
        seed=202604,
    )
    direct_primary = direct[direct["is_primary"]].iloc[0]

    assert gui_primary["nu"] == direct_primary["nu"] == 0.30
    assert gui_primary["shape_factor"] == direct_primary["shape_factor"] == 0.80
    for field in (
        "profile_id",
        "profile_version",
        "review_status",
        "p_range_source",
        "nu_source",
        "shape_factor_source",
    ):
        assert gui_primary[field] == direct_primary[field]
    assert np.isclose(
        gui_primary["E_stamp_app_kPa"],
        direct_primary["E_stamp_app_kPa"],
        rtol=1e-12,
    )


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
