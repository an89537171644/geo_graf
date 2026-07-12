from __future__ import annotations

import csv
import json
from io import BytesIO, StringIO
from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile

import numpy as np
import pandas as pd
import streamlit.testing.v1.app_test as app_test_module
from openpyxl import Workbook
from streamlit.runtime.memory_media_file_storage import MemoryMediaFileStorage
from streamlit.testing.v1 import AppTest

from soilstamp.analysis import calculate_moduli_for_test
from soilstamp.data import prepare_measurements
from soilstamp.methodology import ModulusOverrides
from soilstamp.reporting import build_markdown_report
from soilstamp.schema import ValidationIssue
from soilstamp.ui import label_for_enum


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
        item.label == "Скачать журнал преобразования показаний (CSV)"
        for item in app.download_button
    )
    passport_frame = next(
        table.value
        for table in app.dataframe
        if hasattr(table.value, "columns")
        and set(table.value.columns) == {"Поле", "Значение", "Заполнено"}
    )
    assert set(passport_frame["Поле"]) == {
        "Идентификатор проекта",
        "Наименование серии",
        "Статус армирования",
        "Контрольная группа",
        "Идентификатор пары",
        "Партия грунта",
        "Дата испытания",
        "Оператор",
        "Геометрия штампа",
        "Размеры лотка, мм",
        "Плотность сухого грунта, кг/м³",
        "Влажность, %",
        "Тип грунта",
        "Схема армирования",
        "Приборы и поверка",
    }


def test_download_labels_are_russian_but_indicator_files_remain_canonical() -> None:
    storages: list[MemoryMediaFileStorage] = []

    class CapturingStorage(MemoryMediaFileStorage):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            storages.append(self)

    with patch.object(app_test_module, "MemoryMediaFileStorage", CapturingStorage):
        app = _user_upload_app(
            "demo_protocol.xlsx",
            _demo_xlsx_bytes(),
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    assert not app.exception
    final_storage = storages[-1]
    labels_by_filename: dict[str, str] = {}
    content_by_filename: dict[str, bytes] = {}
    for element in app.download_button:
        stored = final_storage.get_file(element.proto.url.rsplit("/", 1)[-1])
        if stored.filename:
            labels_by_filename[stored.filename] = element.label
            content_by_filename[stored.filename] = stored.content

    expected_labels = {
        "indicator_calibration_parameters.csv": (
            "Скачать параметры калибровки индикаторов (CSV)"
        ),
        "indicator_processing_audit.csv": (
            "Скачать журнал преобразования показаний (CSV)"
        ),
        "indicator_processing_events.csv": (
            "Скачать журнал событий индикаторов (CSV)"
        ),
        "indicator_aggregation_results.csv": (
            "Скачать результаты агрегации осадки (CSV)"
        ),
        "raw_cells.csv": "Скачать исходные ячейки (CSV)",
    }
    assert expected_labels.items() <= labels_by_filename.items()

    canonical_columns = {
        "indicator_calibration_parameters.csv": {"test_id", "channel", "mode"},
        "indicator_processing_audit.csv": {
            "test_id",
            "channel",
            "original_reading",
        },
        "indicator_processing_events.csv": {"test_id", "channel", "event_type"},
        "indicator_aggregation_results.csv": {
            "test_id",
            "row_index",
            "aggregation_status",
        },
        "raw_cells.csv": {"sheet_name", "source_row", "raw_value"},
    }
    for filename, required_columns in canonical_columns.items():
        decoded = content_by_filename[filename].decode("utf-8-sig")
        header = set(next(csv.reader(StringIO(decoded))))
        assert required_columns <= header


def test_report_preview_is_localized_but_markdown_download_is_byte_identical() -> None:
    storages: list[MemoryMediaFileStorage] = []
    canonical_reports: list[str] = []

    class CapturingStorage(MemoryMediaFileStorage):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            storages.append(self)

    def capture_report(*args, **kwargs) -> str:
        report = build_markdown_report(*args, **kwargs)
        canonical_reports.append(report)
        return report

    with (
        patch.object(app_test_module, "MemoryMediaFileStorage", CapturingStorage),
        patch(
            "soilstamp.reporting.build_markdown_report",
            side_effect=capture_report,
        ),
    ):
        app = _user_upload_app(
            "demo_protocol.csv",
            (ROOT / "examples" / "demo_protocol.csv").read_bytes(),
            "text/csv",
        )

    assert not app.exception
    canonical_report = canonical_reports[-1]
    preview = next(
        str(element.value)
        for element in app.markdown
        if "Активный слой:" in str(element.value)
    )

    assert "Активный слой: `Исходные данные`" in preview
    assert "режим=`Готовая накопленная осадка`" in preview
    assert "назначение=`Проверка после переноса данных`" in preview
    assert "статус=`Требуется проверка`" in preview
    assert "метод=`Не определено`" in preview
    assert (
        "статусы=Не применяется для непосредственно заданной осадки=" in preview
    )
    assert "QC-статусы: Без замечаний=" in preview
    assert ", Предупреждение=" in preview
    assert ", Нет данных=" in preview
    assert "Активный слой: `raw`" not in preview
    assert "mode=`cumulative_settlement`" not in preview
    assert "назначение=`migration_review_required`" not in preview
    assert "статус=`review_required`" not in preview

    assert "Активный слой: `raw`" in canonical_report
    assert "mode=`cumulative_settlement`" in canonical_report
    final_storage = storages[-1]
    markdown_download = next(
        final_storage.get_file(element.proto.url.rsplit("/", 1)[-1])
        for element in app.download_button
        if element.label == "Скачать отчёт Markdown"
    )
    assert markdown_download.filename == "soil_stamp_report_ru.md"
    assert markdown_download.content == canonical_report.encode("utf-8")


def test_streamlit_builds_shared_html_xlsx_and_approval_downloads() -> None:
    app = _user_upload_app(
        "demo_protocol.csv",
        (ROOT / "examples" / "demo_protocol.csv").read_bytes(),
        "text/csv",
    )
    next(
        item for item in app.button if item.label == "Собрать пакет воспроизводимости"
    ).click()
    app.run(timeout=180)

    assert not app.exception
    labels = {item.label for item in app.download_button}
    assert {
        "Скачать HTML-отчёт",
        "Скачать XLSX-отчёт",
        "Скачать пакет согласования ZIP",
        "Скачать реестр SHA-256",
        "Скачать пакет воспроизводимости ZIP",
    }.issubset(labels)


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
    bootstrap = next(
        item
        for item in app.number_input
        if item.label == "Число повторов бутстрепа"
    )
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
    display_profile = "Методический профиль"
    display_primary = "Основной результат"
    gui_primary = next(
        table.value.iloc[0]
        for table in app.dataframe
        if hasattr(table.value, "columns")
        and display_profile in table.value.columns
        and bool(table.value[display_primary].fillna(False).any())
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

    # The Streamlit table is a Russian display projection, while the frame in
    # session state remains the canonical machine representation.
    assert gui_primary["Коэффициент Пуассона ν"] == direct_primary["nu"] == 0.30
    assert (
        gui_primary["Коэффициент формы"]
        == direct_primary["shape_factor"]
        == 0.80
    )
    assert gui_primary[display_profile] == label_for_enum(
        "methodology_profile", direct_primary["profile_id"]
    )
    assert gui_primary["Статус проверки"] == label_for_enum(
        "review", direct_primary["review_status"]
    )

    canonical_gui = next(
        frame
        for frame in app.session_state.analysis_tables.values()
        if isinstance(frame, pd.DataFrame)
        and "profile_id" in frame.columns
        and "is_primary" in frame.columns
        and bool(frame["is_primary"].fillna(False).any())
    )
    canonical_gui_primary = canonical_gui[canonical_gui["is_primary"]].iloc[0]
    assert canonical_gui_primary["profile_id"] == "antonov_round_stamp_v1"
    for field in (
        "profile_id",
        "profile_version",
        "review_status",
        "p_range_source",
        "nu_source",
        "shape_factor_source",
    ):
        assert canonical_gui_primary[field] == direct_primary[field]
    assert np.isclose(
        canonical_gui_primary["E_stamp_app_kPa"],
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


def test_processing_exception_diagnostics_keep_canonical_suggested_action() -> None:
    storages: list[MemoryMediaFileStorage] = []

    class CapturingStorage(MemoryMediaFileStorage):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            storages.append(self)

    with (
        patch.object(app_test_module, "MemoryMediaFileStorage", CapturingStorage),
        patch(
            "soilstamp.data.prepare_measurements",
            side_effect=RuntimeError("forced processing exception"),
        ),
    ):
        app = _user_upload_app(
            "demo_protocol.csv",
            (ROOT / "examples" / "demo_protocol.csv").read_bytes(),
            "text/csv",
    )

    assert not app.exception
    assert storages
    final_storage = storages[-1]

    downloads: dict[str, bytes] = {}
    labels_by_filename: dict[str, str] = {}
    for element in app.download_button:
        stored = final_storage.get_file(element.proto.url.rsplit("/", 1)[-1])
        if stored.filename:
            downloads[stored.filename] = stored.content
            labels_by_filename[stored.filename] = element.label

    assert labels_by_filename["issues.csv"] == (
        "Скачать замечания контроля качества (CSV)"
    )

    canonical_action = (
        "Скачайте диагностический ZIP и проверьте metadata/калибровку."
    )
    issues = pd.read_csv(BytesIO(downloads["issues.csv"]))
    runtime_issue = issues.loc[issues["code"] == "processing_exception"].iloc[0]
    assert runtime_issue["suggested_action"] == canonical_action

    with ZipFile(BytesIO(downloads["soil_stamp_import_diagnostics.zip"])) as archive:
        bundled_issues = pd.read_csv(BytesIO(archive.read("diagnostics/issues.csv")))
    bundled_runtime_issue = bundled_issues.loc[
        bundled_issues["code"] == "processing_exception"
    ].iloc[0]
    assert bundled_runtime_issue["suggested_action"] == canonical_action

    display_frame = next(
        table.value
        for table in app.dataframe
        if hasattr(table.value, "columns")
        and "Рекомендуемое действие" in table.value.columns
    )
    display_runtime_issue = display_frame.loc[
        display_frame["Код"] == "processing_exception"
    ].iloc[0]
    assert display_runtime_issue["Рекомендуемое действие"] == (
        "Скачайте диагностический ZIP и проверьте паспорт, служебные параметры "
        "и калибровку."
    )


def test_qc_prose_is_localized_without_mutating_canonical_issue() -> None:
    canonical_message = (
        "Прямая settlement использована; indicator_* сохранены как raw."
    )
    canonical_action = (
        "Статус migration_review_required; задайте mode, unit и calibration factor."
    )
    canonical_raw_value = "user settlement indicator_1 comment"
    injected: list[ValidationIssue] = []

    def prepare_with_controlled_issue(*args, **kwargs):
        prepared, issues = prepare_measurements(*args, **kwargs)
        issue = ValidationIssue(
            "warning",
            "controlled_qc_localization",
            canonical_message,
            column="indicator_1",
            raw_value=canonical_raw_value,
            suggested_action=canonical_action,
        )
        injected.append(issue)
        return prepared, [*issues, issue]

    with patch(
        "soilstamp.data.prepare_measurements", side_effect=prepare_with_controlled_issue
    ):
        app = _user_upload_app(
            "demo_protocol.csv",
            (ROOT / "examples" / "demo_protocol.csv").read_bytes(),
            "text/csv",
        )

    assert not app.exception
    display_frame = next(
        table.value
        for table in app.dataframe
        if hasattr(table.value, "columns")
        and "Код" in table.value.columns
        and "controlled_qc_localization" in table.value["Код"].astype(str).tolist()
    )
    display_issue = display_frame.loc[
        display_frame["Код"] == "controlled_qc_localization"
    ].iloc[0]
    assert display_issue["Сообщение"] == (
        "Прямая осадка использована; индикаторные каналы сохранены как исходные данные."
    )
    assert display_issue["Рекомендуемое действие"] == (
        "Статус требуется проверка после переноса данных; задайте режим, единица "
        "и коэффициент калибровки."
    )
    assert display_issue["Столбец"] == "Индикатор 1"
    assert display_issue["Исходное значение"] == canonical_raw_value

    canonical_issue = injected[-1]
    assert canonical_issue.message == canonical_message
    assert canonical_issue.suggested_action == canonical_action
    assert canonical_issue.column == "indicator_1"
    assert canonical_issue.raw_value == canonical_raw_value


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
