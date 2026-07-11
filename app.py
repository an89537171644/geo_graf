"""Streamlit interface for Soil Stamp Antonov."""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from soilstamp.analysis import (
    center_and_tilt,
    compare_groups,
    confirm_manual_pcr,
    deformation_work,
    derivative_diagnostics,
    estimate_moduli,
    fit_segmented_pcr,
    group_mean_curve,
    hysteresis_metrics,
    modulus_sensitivity,
    pressure_at_settlement,
    time_stabilization,
    value_at_pressure,
)
from soilstamp.data import (
    AuditTrail,
    apply_manual_point_correction,
    apply_settlement_correction,
    failure_summary,
    prepare_measurements,
)
from soilstamp.io import (
    inspect_excel_schema,
    read_metadata_json,
    read_protocol,
    validate_import_metadata_consistency,
)
from soilstamp.indicators import (
    indicator_audit_frame,
    indicator_event_frame,
    indicator_passport_frame,
)
from soilstamp.plotting import export_figure, plot_curves, plot_stamp_schematic
from soilstamp.provenance import (
    build_provenance,
    effective_conversion_parameters,
    passport_completeness,
    validate_project_metadata,
    value_sha256,
)
from soilstamp.reporting import build_markdown_report, reproducibility_bundle
from soilstamp.schema import VERSION, ValidationIssue


BASE_DIR = Path(__file__).resolve().parent
DEMO_PROTOCOL = BASE_DIR / "examples" / "demo_protocol.csv"
DEMO_METADATA = BASE_DIR / "examples" / "demo_metadata.json"


st.set_page_config(
    page_title=f"Soil Stamp Antonov {VERSION}",
    page_icon="📐",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _dataset_hash(raw: pd.DataFrame, metadata: dict) -> str:
    payload = raw.to_csv(index=False).encode("utf-8") + json.dumps(
        metadata, sort_keys=True, ensure_ascii=False, default=str
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def provenance_key(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()[:12]


def _display_safe_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a UI-only string copy for object columns; exports remain lossless."""

    result = frame.copy(deep=True)

    def text_value(value):
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
        dtype = result[column].dtype
        if pd.api.types.is_object_dtype(dtype) or pd.api.types.is_string_dtype(dtype):
            result[column] = result[column].map(text_value)
    # Processing artefacts live in attrs and are exported separately.  Passing
    # them through every Arrow preview is both redundant and expensive.
    result.attrs.clear()
    return result


def _scope_indicator_table(table: pd.DataFrame, test_ids: list[str]) -> pd.DataFrame:
    if table.empty or "test_id" not in table:
        return table.copy()
    selected = {str(value) for value in test_ids}
    return table[table["test_id"].astype(str).isin(selected)].copy()


def _issue_frame(issues) -> pd.DataFrame:
    return pd.DataFrame([item.to_dict() if hasattr(item, "to_dict") else item for item in issues])


def _diagnostic_bundle(input_context: dict, issues) -> bytes:
    """Package exact inputs plus machine-readable diagnostics before processing."""

    issue_frame = _issue_frame(issues)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        source_name = Path(str(input_context.get("source_file_name") or "protocol.bin")).name
        metadata_name = Path(str(input_context.get("metadata_file_name") or "metadata.json")).name
        archive.writestr(f"source/{source_name}", input_context.get("source_file_bytes", b""))
        archive.writestr(
            f"metadata/{metadata_name}", input_context.get("metadata_file_bytes", b"")
        )
        archive.writestr("diagnostics/issues.csv", issue_frame.to_csv(index=False).encode("utf-8-sig"))
        raw_cells = input_context.get("raw_cells")
        if isinstance(raw_cells, pd.DataFrame) and len(raw_cells):
            archive.writestr(
                "diagnostics/raw_cells.csv", raw_cells.to_csv(index=False).encode("utf-8-sig")
            )
        archive.writestr(
            "diagnostics/import_diagnostics.json",
            json.dumps(
                {
                    "import_info": input_context.get("import_info"),
                    "column_mapping": input_context.get("column_mapping"),
                    "provenance": input_context["provenance"].to_dict(),
                    "provenance_config": input_context.get("provenance_config"),
                    "source_file_name": input_context.get("source_file_name"),
                    "metadata_file_name": input_context.get("metadata_file_name"),
                },
                ensure_ascii=False,
                indent=2,
                default=str,
            ).encode("utf-8"),
        )
    return buffer.getvalue()


def _render_blocking_diagnostics(input_context: dict, issues, *, key_prefix: str) -> None:
    issue_frame = _issue_frame(issues)
    st.dataframe(_display_safe_frame(issue_frame), width="stretch", hide_index=True)
    st.download_button(
        "Скачать диагностический пакет ZIP",
        _diagnostic_bundle(input_context, issues),
        "soil_stamp_import_diagnostics.zip",
        "application/zip",
        key=f"{key_prefix}_zip",
    )
    st.download_button(
        "Скачать issues.csv",
        issue_frame.to_csv(index=False).encode("utf-8-sig"),
        "issues.csv",
        "text/csv",
        key=f"{key_prefix}_issues",
    )


def _reset_for_dataset(key: str, raw: pd.DataFrame, input_context: dict) -> None:
    st.session_state.dataset_key = key
    st.session_state.dataset_provenance = input_context["provenance"]
    st.session_state.audit = AuditTrail()
    st.session_state.audit.record(
        "import_dataset",
        scope=",".join(raw["test_id"].astype(str).unique()) if "test_id" in raw else "dataset",
        reason="Импорт выбранного набора данных",
        parameters={
            "rows": len(raw),
            "dataset_sha256": key,
            "import": input_context["import_info"],
            "provenance": input_context["provenance"].to_dict(),
        },
        after=raw,
        method=f"{input_context['import_info'].get('format', 'unknown')}_import",
    )
    st.session_state.manual_overrides = []
    st.session_state.seating_offsets = {}
    st.session_state.pcr_results = {}
    st.session_state.pcr_latest = {}
    st.session_state.analysis_tables = {}
    st.session_state.figure_exports = {}
    st.session_state.bundle_cache = {}
    st.session_state.processing_provenance = {}
    st.session_state.revision = 0


def _load_inputs() -> dict:
    st.sidebar.header("Данные")
    source = st.sidebar.radio("Источник", ["Демонстрационный набор", "Загрузить файлы"])
    if source == "Демонстрационный набор":
        protocol_bytes = DEMO_PROTOCOL.read_bytes()
        metadata_bytes = DEMO_METADATA.read_bytes()
        imported = read_protocol(protocol_bytes, filename=DEMO_PROTOCOL.name)
        metadata = read_metadata_json(metadata_bytes)
        config = {"import_mode": "strict", "source": "demo"}
        provenance = build_provenance(
            input_source=protocol_bytes,
            metadata_source=metadata_bytes,
            config=config,
            project_root=BASE_DIR,
        )
        return {
            "raw": imported.frame,
            "metadata": metadata,
            "import_info": imported.info,
            "import_issues": imported.issues,
            "raw_cells": imported.raw_cells,
            "provenance": provenance,
            "passport": passport_completeness(metadata),
            "source_file_name": DEMO_PROTOCOL.name,
            "source_file_bytes": protocol_bytes,
            "metadata_file_name": DEMO_METADATA.name,
            "metadata_file_bytes": metadata_bytes,
            "column_mapping": None,
            "provenance_config": config,
        }

    protocol_file = st.sidebar.file_uploader(
        "Протокол CSV/XLSX", type=["csv", "txt", "xlsx", "xlsm"]
    )
    metadata_file = st.sidebar.file_uploader("Metadata JSON", type=["json"])
    if protocol_file is None or metadata_file is None:
        st.info("Загрузите протокол и metadata JSON. Пользовательские файлы не подменяются demo-набором.")
        st.stop()

    protocol_bytes = protocol_file.getvalue()
    file_key = provenance_key(protocol_bytes)
    metadata_bytes = metadata_file.getvalue()
    metadata = read_metadata_json(metadata_bytes)

    def uploaded_context(imported, config: dict) -> dict:
        provenance = build_provenance(
            input_source=protocol_bytes,
            metadata_source=metadata_bytes,
            config=config,
            project_root=BASE_DIR,
        )
        return {
            "raw": imported.frame,
            "metadata": metadata,
            "import_info": imported.info,
            "import_issues": imported.issues,
            "raw_cells": imported.raw_cells,
            "provenance": provenance,
            "passport": passport_completeness(metadata),
            "source_file_name": protocol_file.name,
            "source_file_bytes": protocol_bytes,
            "metadata_file_name": metadata_file.name,
            "metadata_file_bytes": metadata_bytes,
            "column_mapping": config.get("column_mapping"),
            "provenance_config": config,
        }

    suffix = Path(protocol_file.name).suffix.casefold()
    import_mode = "strict"
    selected_sheet = None
    selected_header_row = None
    column_mapping = None
    preview_imported = None
    partial_sheet_scope = False

    def schema_failure_context(exc: Exception) -> dict:
        failed_import = read_protocol(
            protocol_bytes,
            filename=protocol_file.name,
            import_mode="strict",
        )
        failed_import.issues.append(
            ValidationIssue(
                "error",
                "excel_schema_inspection_failed",
                f"Не удалось безопасно прочитать схему Excel: {exc}",
                sheet=selected_sheet,
                row=selected_header_row,
                raw_value=str(exc),
                suggested_action="Скачайте диагностический ZIP и переэкспортируйте исходную книгу.",
            )
        )
        failed_import.info["requested_import_mode"] = import_mode
        failed_import.info["blocking_issue_count"] = sum(
            bool(item.blocks_processing) for item in failed_import.issues
        )
        return uploaded_context(
            failed_import,
            {
                "import_mode": import_mode,
                "sheet_name": selected_sheet,
                "header_row": selected_header_row,
                "column_mapping": None,
                "schema_inspection_failed": True,
            },
        )

    if suffix in {".xlsx", ".xlsm"}:
        import_mode = st.sidebar.radio(
            "Режим Excel-импорта",
            ["strict", "interactive", "heuristic"],
            captions=[
                "Неизвестная схема блокирует расчёт",
                "Явное сохранённое сопоставление",
                "Legacy-совместимость с предупреждением",
            ],
        )
        try:
            schema = inspect_excel_schema(protocol_bytes)
        except Exception as exc:
            return schema_failure_context(exc)
        sheet_names = [item["sheet_name"] for item in schema["sheets"]]
        if import_mode == "interactive":
            sheet_scope = "Один лист"
        else:
            sheet_scope = st.sidebar.radio(
                "Область листов Excel",
                ["Все листы", "Один лист"],
                captions=[
                    "По умолчанию: проверяются межлистовые ID и схемы",
                    "Явно ограничить импорт одним листом",
                ],
            )
        selected_schema = None
        if sheet_scope == "Один лист":
            selected_sheet = st.sidebar.selectbox("Лист Excel", sheet_names)
            selected_schema = next(
                item for item in schema["sheets"] if item["sheet_name"] == selected_sheet
            )
            suggested_row = selected_schema.get("header_row")
            selected_header_row = int(
                st.sidebar.number_input(
                    "Строка заголовков (1-based)",
                    min_value=1,
                    max_value=200_000,
                    value=int(suggested_row or 1),
                    step=1,
                    key=f"header_row_{file_key}_{selected_sheet}",
                )
            )
            try:
                selected_schema = inspect_excel_schema(
                    protocol_bytes,
                    sheet_name=selected_sheet,
                    header_row=selected_header_row,
                )["sheets"][0]
            except Exception as exc:
                return schema_failure_context(exc)
            partial_sheet_scope = import_mode != "interactive"
        else:
            st.sidebar.caption(
                "Будут импортированы все распознанные листы; служебные листы фиксируются в QC."
            )
        if import_mode == "interactive":
            mapping_file = st.sidebar.file_uploader(
                "Сохранённый mapping JSON (необязательно)",
                type=["json"],
                key=f"mapping_file_{file_key}_{selected_sheet}_{selected_header_row}",
            )
            saved_mapping = {}
            mapping_file_sha256 = None
            if mapping_file is not None:
                mapping_bytes = mapping_file.getvalue()
                saved_mapping = json.loads(mapping_bytes.decode("utf-8-sig"))
                if not isinstance(saved_mapping, dict):
                    raise ValueError("Mapping JSON должен быть объектом.")
                mapping_file_sha256 = hashlib.sha256(mapping_bytes).hexdigest()
            mapping_state_key = provenance_key(
                json.dumps(
                    {
                        "input_file_sha256": hashlib.sha256(protocol_bytes).hexdigest(),
                        "sheet_name": selected_sheet,
                        "header_row": selected_header_row,
                        "mapping_file_sha256": mapping_file_sha256,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                ).encode("utf-8")
            )
            header_items = selected_schema.get("headers", [])
            displays = ["—"] + [f"{item['column']}: {item['value']}" for item in header_items]
            display_to_column = {
                f"{item['column']}: {item['value']}": item["column"] for item in header_items
            }
            suggested = selected_schema.get("suggested_mapping", {})
            field_labels = {
                "test_id": "ID испытания *",
                "stage": "Ступень *",
                "load": "Нагрузка/давление *",
                "settlement": "Осадка",
                "indicator_1": "Индикатор 1",
                "indicator_2": "Индикатор 2",
                "indicator_3": "Индикатор 3",
                "indicator_4": "Индикатор 4",
                "reference_indicator": "Опорный индикатор",
                "horizontal_indicator": "Горизонтальный индикатор",
                "branch": "Ветвь",
                "timestamp": "Время",
                "status": "Статус",
                "comment": "Комментарий",
                "group": "Группа",
                "pair_id": "ID пары",
            }
            available_columns = {item["column"] for item in header_items}
            column_mapping = dict(saved_mapping)
            with st.sidebar.expander("Сопоставление столбцов", expanded=True):
                for field, label in field_labels.items():
                    saved_selector = saved_mapping.get(field)
                    saved_column = None
                    if saved_selector is not None:
                        selector_text = str(saved_selector).strip()
                        if selector_text.upper() in available_columns:
                            saved_column = selector_text.upper()
                        else:
                            saved_column = next(
                                (
                                    item["column"]
                                    for item in header_items
                                    if str(item["value"]).strip() == selector_text
                                ),
                                None,
                            )
                    suggested_column = saved_column or suggested.get(field)
                    default_display = next(
                        (name for name, column in display_to_column.items() if column == suggested_column),
                        "—",
                    )
                    selected = st.selectbox(
                        label,
                        displays,
                        index=displays.index(default_display),
                        key=f"mapping_{field}_{mapping_state_key}",
                    )
                    if selected != "—":
                        column_mapping[field] = display_to_column[selected]
                    else:
                        column_mapping.pop(field, None)
            st.sidebar.download_button(
                "Скачать mapping JSON",
                json.dumps(column_mapping, ensure_ascii=False, indent=2).encode("utf-8"),
                "excel_column_mapping.json",
                "application/json",
            )
            mapping_context = {
                "input_file_sha256": hashlib.sha256(protocol_bytes).hexdigest(),
                "sheet_name": selected_sheet,
                "header_row": selected_header_row,
                "column_mapping": column_mapping,
            }
            mapping_signature = provenance_key(
                json.dumps(mapping_context, ensure_ascii=False, sort_keys=True).encode("utf-8")
            )
            preview_imported = read_protocol(
                protocol_bytes,
                filename=protocol_file.name,
                import_mode="interactive",
                column_mapping=column_mapping,
                sheet_name=selected_sheet,
                header_row=selected_header_row,
            )
            st.subheader("Предпросмотр сопоставления Excel")
            st.caption(
                f"Лист: {selected_sheet}; строка заголовков: {selected_header_row}; "
                f"подпись mapping: {mapping_signature}"
            )
            if preview_imported.frame.empty:
                st.warning("По текущему mapping не распознано ни одной строки протокола.")
            else:
                preview_columns = [
                    column
                    for column in (
                        "sheet_name",
                        "source_row",
                        "source_columns",
                        "test_id",
                        "stage",
                        "load",
                        "settlement",
                        "indicator_1",
                        "indicator_2",
                        "indicator_3",
                        "indicator_4",
                        "reference_indicator",
                        "raw_stage",
                        "raw_load",
                        "parsed_stage",
                        "parsed_load",
                        "status",
                    )
                    if column in preview_imported.frame.columns
                ]
                st.dataframe(
                    _display_safe_frame(preview_imported.frame[preview_columns].head(10)),
                    width="stretch",
                    hide_index=True,
                )
            if preview_imported.issues:
                with st.expander("Диагностика предпросмотра", expanded=True):
                    st.dataframe(
                        _display_safe_frame(_issue_frame(preview_imported.issues)),
                        width="stretch",
                        hide_index=True,
                    )
            if len(preview_imported.raw_cells):
                with st.expander("Исходные координаты ячеек предпросмотра"):
                    st.dataframe(
                        _display_safe_frame(preview_imported.raw_cells.head(50)),
                        width="stretch",
                        hide_index=True,
                    )
            mapping_confirmed = st.sidebar.checkbox(
                "Подтверждаю сопоставление столбцов",
                key=f"confirm_mapping_{mapping_signature}",
            )
            if not mapping_confirmed:
                st.info("Проверьте предпросмотр и явно подтвердите mapping перед расчётом.")
                st.stop()

    config = {
        "import_mode": import_mode,
        "sheet_scope": (
            "single_confirmed"
            if import_mode == "interactive"
            else "single_explicit" if partial_sheet_scope else "all"
        ),
        "sheet_name": selected_sheet,
        "header_row": selected_header_row,
        "column_mapping": column_mapping,
    }
    imported = (
        preview_imported
        if preview_imported is not None
        else read_protocol(
            protocol_bytes,
            filename=protocol_file.name,
            import_mode=import_mode,
            column_mapping=column_mapping,
            sheet_name=selected_sheet,
            header_row=selected_header_row,
        )
    )
    if partial_sheet_scope:
        imported.issues.append(
            ValidationIssue(
                "warning",
                "partial_excel_sheet_scope",
                f"Импорт явно ограничен листом {selected_sheet!r}; остальные листы не входят в анализ.",
                sheet=selected_sheet,
                suggested_action="Для полного проекта выберите область «Все листы».",
            )
        )
        imported.info["partial_sheet_scope"] = True
    imported.info["blocking_issue_count"] = sum(
        bool(item.blocks_processing) for item in imported.issues
    )
    return uploaded_context(imported, config)


def _apply_overrides(prepared: pd.DataFrame, active_mode: str) -> pd.DataFrame:
    result = prepared.copy(deep=True)
    for item in st.session_state.get("manual_overrides", []):
        if item.get("base_mode") != active_mode:
            continue
        mask = (result["test_id"].astype(str) == item["test_id"]) & (
            result["sequence_no"] == item["sequence_no"]
        )
        result.loc[mask, "settlement_mm"] = item["value_mm"]
        result.loc[mask, "manual_override"] = True
    if "manual_override" not in result:
        result["manual_override"] = False
    result["manual_override"] = result["manual_override"].fillna(False).astype(bool)
    return result
try:
    input_context = _load_inputs()
    raw = input_context["raw"]
    metadata = input_context["metadata"]
    import_info = input_context["import_info"]
    metadata_issues = validate_project_metadata(
        metadata,
        strict=import_info.get("import_mode", "strict") in {"strict", "interactive"},
    )
    consistency_issues = validate_import_metadata_consistency(
        raw,
        metadata,
        import_info,
        strict=import_info.get("import_mode", "strict") in {"strict", "interactive"},
    )
    import_issues = [*input_context["import_issues"], *metadata_issues, *consistency_issues]
except Exception as exc:
    st.error(f"Не удалось прочитать данные: {exc}")
    st.stop()

import_blocking = [item for item in import_issues if bool(item.blocks_processing)]
if import_blocking:
    st.title(f"Soil Stamp Antonov {VERSION}")
    st.error("Схема импорта содержит блокирующие ошибки. Исправьте mapping или выберите другой режим.")
    _render_blocking_diagnostics(input_context, import_issues, key_prefix="import_blocking")
    st.stop()

dataset_key = hashlib.sha256(
    (
        _dataset_hash(raw, metadata)
        + input_context["provenance"].input_file_sha256
        + input_context["provenance"].metadata_sha256
        + input_context["provenance"].config_sha256
    ).encode("utf-8")
).hexdigest()
if st.session_state.get("dataset_key") != dataset_key:
    _reset_for_dataset(dataset_key, raw, input_context)
else:
    input_context["provenance"] = st.session_state.get(
        "dataset_provenance", input_context["provenance"]
    )
    st.session_state.dataset_provenance = input_context["provenance"]

try:
    base_prepared, validation_issues = prepare_measurements(
        raw, metadata, strict_metadata=False
    )
    indicator_processing_audit = indicator_audit_frame(base_prepared)
    indicator_processing_events = indicator_event_frame(base_prepared)
    indicator_calibration_parameters = indicator_passport_frame(base_prepared)
    # Pandas deep-copies attrs through most analysis operations.  Keep the
    # sizeable indicator artefacts in dedicated frames and use an attrs-free
    # working layer; they are reattached only to the report snapshot below.
    base_prepared.attrs.clear()
except Exception as exc:
    st.title(f"Soil Stamp Antonov {VERSION}")
    st.error(f"Metadata или калибровка некорректны: {exc}")
    runtime_issue = ValidationIssue(
        "error",
        "processing_exception",
        f"Подготовка измерений завершилась исключением: {exc}",
        raw_value=str(exc),
        suggested_action="Скачайте диагностический ZIP и проверьте metadata/калибровку.",
    )
    _render_blocking_diagnostics(
        input_context,
        [*import_issues, runtime_issue],
        key_prefix="processing_exception",
    )
    st.stop()
blocking = [item for item in validation_issues if bool(item.blocks_processing)]
if blocking:
    st.title(f"Soil Stamp Antonov {VERSION}")
    st.error("Импорт содержит блокирующие ошибки.")
    _render_blocking_diagnostics(
        input_context,
        [*import_issues, *validation_issues],
        key_prefix="measurement_blocking",
    )
    st.stop()

st.sidebar.header("Рабочий слой")
correction_mode = st.sidebar.radio(
    "Осадка",
    ["raw", "zero_shifted", "seating_corrected"],
    captions=[
        "Исходная кривая",
        "Сдвиг только по измеренной нулевой точке",
        "Явно заданная посадочная поправка",
    ],
)
prepared, correction_issues = apply_settlement_correction(
    base_prepared,
    correction_mode,
    seating_offsets_mm=st.session_state.seating_offsets,
)
prepared = _apply_overrides(prepared, correction_mode)
all_issues = [*import_issues, *validation_issues, *correction_issues]
failures = failure_summary(prepared)

st.sidebar.header("Выбор")
test_options = prepared["test_id"].astype(str).unique().tolist()
selected_tests = st.sidebar.multiselect("Испытания", test_options, default=test_options)
filtered = prepared[prepared["test_id"].isin(selected_tests)].copy()
if filtered.empty:
    st.warning("Выберите хотя бы одно испытание.")
    st.stop()
selected_indicator_audit = _scope_indicator_table(
    indicator_processing_audit, selected_tests
)
selected_indicator_events = _scope_indicator_table(
    indicator_processing_events, selected_tests
)
selected_indicator_passports = _scope_indicator_table(
    indicator_calibration_parameters, selected_tests
)

st.title(f"Soil Stamp Antonov {VERSION}")
st.caption(
    f"Слой: {correction_mode} · ревизия {st.session_state.revision} · "
    f"{filtered['test_id'].nunique()} испытаний · source SHA-256 "
    f"{input_context['provenance'].input_file_sha256[:12]}"
)

tabs = st.tabs(
    [
        "Импорт и QC",
        "Коррекции",
        "Графики",
        "pcr и E",
        "Сравнение групп",
        "Доп. анализ",
        "Отчёт и журнал",
    ]
)

with tabs[0]:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Строк протокола", len(raw))
    c2.metric("Испытаний", raw["test_id"].nunique())
    c3.metric("Измеренных осадок", int(prepared["is_measured"].sum()))
    c4.metric("Событий разрушения", int(prepared["is_failure"].sum()))
    st.subheader("Параметры импорта")
    st.json(import_info, expanded=False)
    st.dataframe(
        pd.DataFrame(
            effective_conversion_parameters(
                metadata, raw["test_id"].dropna().astype(str).unique().tolist()
            )
        ),
        width="stretch",
        hide_index=True,
    )
    st.subheader("Паспорт проекта")
    passport = input_context["passport"]
    if passport["complete"]:
        st.success("Обязательные поля паспорта заполнены.")
    else:
        st.warning("Паспорт неполон: " + ", ".join(passport["missing"]))
    with st.expander("Поля паспорта"):
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "field": name,
                        "value": (
                            json.dumps(value, ensure_ascii=False, default=str)
                            if isinstance(value, (dict, list, tuple))
                            else "" if value is None else str(value)
                        ),
                        "filled": name not in passport["missing"],
                    }
                    for name, value in passport["fields"].items()
                ]
            ),
            width="stretch",
            hide_index=True,
        )
    st.subheader("Паспорта индикаторов")
    st.caption("Только для чтения: эффективные параметры, использованные при преобразовании.")
    if selected_indicator_passports.empty:
        st.info("Для выбранных испытаний поканальные паспорта индикаторов не сформированы.")
    else:
        passport_columns = [
            "test_id",
            "channel",
            "indicator_type",
            "serial_number",
            "instrument_id",
            "mode",
            "range_mm",
            "division_mm",
            "correction_factor",
            "initial_reading",
            "initial_turn",
            "zero_correction_mm",
            "verification_date",
            "verification_valid_until",
            "max_increment_mm",
            "reverse_tolerance_mm",
            "travel_range_mm",
            "source_path",
            "compatibility_mode",
        ]
        st.dataframe(
            _display_safe_frame(
                selected_indicator_passports[
                    [
                        column
                        for column in passport_columns
                        if column in selected_indicator_passports
                    ]
                ]
            ),
            width="stretch",
            hide_index=True,
        )
        st.download_button(
            "Скачать indicator_calibration_parameters.csv",
            selected_indicator_passports.to_csv(index=False).encode("utf-8-sig"),
            "indicator_calibration_parameters.csv",
            "text/csv",
        )
    st.subheader("Преобразование показаний индикаторов")
    if selected_indicator_audit.empty:
        st.info("Таблица преобразования для выбранных испытаний пуста.")
    else:
        zero_crossings = (
            int(selected_indicator_events["event_type"].eq("zero_crossing").sum())
            if "event_type" in selected_indicator_events
            else 0
        )
        status = selected_indicator_audit.get(
            "processing_status", pd.Series("", index=selected_indicator_audit.index)
        )
        status = status.fillna("").astype(str)
        c1, c2, c3 = st.columns(3)
        c1.metric("Строк преобразования", len(selected_indicator_audit))
        c2.metric("Переходов через ноль", zero_crossings)
        c3.metric("QC не ok", int(status.ne("ok").sum()))
        audit_columns = [
            "test_id",
            "channel",
            "source_row",
            "sequence_index",
            "branch",
            "original_reading",
            "raw_reading",
            "turn_number",
            "computed_increment_mm",
            "cumulative_before_correction_mm",
            "applied_correction_mm",
            "cumulative_settlement_mm",
            "settlement_effective_mm",
            "warning",
            "processing_status",
            "conversion_method",
            "correction_record_ids",
        ]
        st.dataframe(
            _display_safe_frame(
                selected_indicator_audit[
                    [column for column in audit_columns if column in selected_indicator_audit]
                ]
            ),
            width="stretch",
            hide_index=True,
        )
        c1, c2 = st.columns(2)
        c1.download_button(
            "Скачать indicator_processing_audit.csv",
            selected_indicator_audit.to_csv(index=False).encode("utf-8-sig"),
            "indicator_processing_audit.csv",
            "text/csv",
        )
        c2.download_button(
            "Скачать indicator_processing_events.csv",
            selected_indicator_events.to_csv(index=False).encode("utf-8-sig"),
            "indicator_processing_events.csv",
            "text/csv",
        )
        with st.expander("Журнал переходов, обратного хода и коррекций"):
            if selected_indicator_events.empty:
                st.info("Событий преобразования нет.")
            else:
                st.dataframe(
                    _display_safe_frame(selected_indicator_events),
                    width="stretch",
                    hide_index=True,
                )
    st.subheader("Provenance")
    st.json(input_context["provenance"].to_dict(), expanded=False)
    if len(input_context["raw_cells"]):
        with st.expander("Исходные Excel-ячейки и распознанные значения"):
            st.dataframe(
                _display_safe_frame(input_context["raw_cells"]),
                width="stretch",
                hide_index=True,
            )
            st.download_button(
                "Скачать raw_cells.csv",
                input_context["raw_cells"].to_csv(index=False).encode("utf-8-sig"),
                "raw_cells.csv",
                "text/csv",
            )
    if all_issues:
        st.subheader("Контроль качества")
        st.dataframe(
            _display_safe_frame(_issue_frame(all_issues)),
            width="stretch",
            hide_index=True,
        )
    else:
        st.success("Автоматическая проверка не обнаружила замечаний.")
    st.subheader("Хронологический протокол")
    protocol_columns = [
        "sheet_name",
        "source_row",
        "source_columns",
        "test_id",
        "sequence_no",
        "sequence_index",
        "raw_stage",
        "raw_load",
        "raw_indicator",
        "parsed_stage",
        "parsed_load",
        "parsed_indicator",
        "stage",
        "branch",
        "branch_suggested",
        "F_kN",
        "p_kPa",
        "settlement_raw_mm",
        "settlement_mm",
        "status",
        "comment",
    ]
    st.dataframe(
        _display_safe_frame(
            filtered[[column for column in protocol_columns if column in filtered]]
        ),
        width="stretch",
        hide_index=True,
    )
    st.subheader("Разрушение и цензурирование")
    st.dataframe(failures[failures["test_id"].isin(selected_tests)], width="stretch", hide_index=True)
    with st.expander("Metadata"):
        st.json(metadata)

with tabs[1]:
    st.write(
        "Исходный слой хранится отдельно. Нулевая коррекция использует только реально измеренную "
        "точку при F=0; новая точка (0;0) не создаётся."
    )
    correction_preview = filtered[
        ["test_id", "sequence_no", "F_kN", "settlement_raw_mm", "settlement_mm", "correction_mode"]
    ].copy()
    correction_preview["difference_mm"] = (
        correction_preview["settlement_mm"] - correction_preview["settlement_raw_mm"]
    )
    st.dataframe(_display_safe_frame(correction_preview), width="stretch", hide_index=True)

    st.subheader("Посадочная поправка")
    with st.form("seating_form"):
        offsets = {}
        cols = st.columns(min(3, len(test_options)))
        for index, test_id in enumerate(test_options):
            offsets[test_id] = cols[index % len(cols)].number_input(
                f"{test_id}, мм",
                value=float(st.session_state.seating_offsets.get(test_id, 0.0)),
                step=0.01,
                format="%.3f",
            )
        seating_reason = st.text_input("Причина/методика поправки", key="seating_reason")
        apply_seating = st.form_submit_button("Создать ревизию посадочной поправки")
    if apply_seating:
        try:
            _, _ = apply_settlement_correction(
                base_prepared,
                "seating_corrected",
                seating_offsets_mm=offsets,
                audit=st.session_state.audit,
                reason=seating_reason,
            )
            st.session_state.seating_offsets = offsets
            st.session_state.revision += 1
            st.session_state.pcr_results = {}
            st.session_state.pcr_latest = {}
            st.session_state.analysis_tables = {}
            st.session_state.figure_exports = {}
            st.session_state.bundle_cache = {}
            st.success("Новая ревизия записана в audit trail.")
            st.rerun()
        except ValueError as exc:
            st.error(str(exc))

    st.subheader("Ручная коррекция одной точки")
    if correction_mode == "raw":
        st.info("Слой raw доступен только для чтения. Выберите zero_shifted или seating_corrected.")
    else:
        with st.form("manual_point_form"):
            manual_test = st.selectbox("Испытание", test_options, key="manual_test")
            test_rows = prepared[prepared["test_id"].astype(str) == manual_test]
            sequence_options = test_rows["sequence_no"].astype(int).tolist()
            manual_sequence = st.selectbox("sequence_no", sequence_options)
            current_row = test_rows[test_rows["sequence_no"] == manual_sequence].iloc[0]
            current_value = current_row["settlement_mm"]
            manual_value = st.number_input(
                "Новое значение s, мм",
                value=float(current_value) if pd.notna(current_value) else 0.0,
                step=0.01,
                format="%.3f",
            )
            manual_reason = st.text_input("Обоснование", key="manual_reason")
            apply_manual = st.form_submit_button("Создать ревизию ручной коррекции")
        if apply_manual:
            try:
                apply_manual_point_correction(
                    prepared,
                    test_id=manual_test,
                    sequence_no=int(manual_sequence),
                    corrected_settlement_mm=float(manual_value),
                    reason=manual_reason,
                    audit=st.session_state.audit,
                )
                st.session_state.manual_overrides.append(
                    {
                        "test_id": manual_test,
                        "sequence_no": int(manual_sequence),
                        "value_mm": float(manual_value),
                        "base_mode": correction_mode,
                    }
                )
                st.session_state.revision += 1
                st.session_state.pcr_results = {}
                st.session_state.pcr_latest = {}
                st.session_state.analysis_tables = {}
                st.session_state.figure_exports = {}
                st.session_state.bundle_cache = {}
                st.success("Коррекция добавлена как новая ревизия; raw не изменён.")
                st.rerun()
            except ValueError as exc:
                st.error(str(exc))

with tabs[2]:
    controls = st.columns([1.25, 1.15, 1.0, 1.0])
    graph_mode = controls[0].selectbox(
        "Режим",
        ["raw_protocol", "antonov_publication", "group_mean_ci", "diagnostic", "normalized"],
    )
    if graph_mode == "normalized":
        axis_options = [
            "p-s/D",
            "p/pu-s/D",
            "F/(gammaD3)-s/D",
            "p/(gammaD)-s/D",
        ]
        default_axis = 0
    elif graph_mode == "diagnostic":
        axis_options = ["p-s"]
        default_axis = 0
    else:
        axis_options = ["F-s", "p-s", "p-s/D"]
        default_axis = 1 if filtered["D_mm"].nunique() > 1 else 0
    axis_mode = controls[1].selectbox("Оси", axis_options, index=default_axis)
    ci_method = controls[2].selectbox("95% ДИ", ["t", "simultaneous_bootstrap"])
    bootstrap_graph = controls[3].number_input("Bootstrap", min_value=100, max_value=5000, value=300, step=100)
    grid_controls = st.columns(3)
    major_step = grid_controls[0].number_input("Major step (0 = auto)", min_value=0.0, value=0.0, step=10.0)
    minor_step = grid_controls[1].number_input("Minor step (0 = auto)", min_value=0.0, value=0.0, step=5.0)
    fixed = grid_controls[2].checkbox("Фиксированный масштаб")
    fixed_axes = None
    if fixed:
        limits = st.columns(4)
        fixed_axes = (
            limits[0].number_input("x min", value=0.0),
            limits[1].number_input("x max", value=500.0),
            limits[2].number_input("s min", value=0.0),
            limits[3].number_input("s max", value=10.0),
        )
    diagnostic_result = None
    if graph_mode == "diagnostic":
        diagnostic_test = st.selectbox("Испытание для диагностики", selected_tests, key="diag_test")
        graph_data = filtered[filtered["test_id"].astype(str) == diagnostic_test]
        diagnostic_context = (
            f"{dataset_key}:{st.session_state.revision}:{correction_mode}:{diagnostic_test}"
        )
        latest_key = st.session_state.pcr_latest.get(diagnostic_context)
        diagnostic_result = st.session_state.pcr_results.get(latest_key) if latest_key else None
    else:
        graph_data = filtered
    try:
        plot_output = plot_curves(
            graph_data,
            mode=graph_mode,
            axis_mode=axis_mode,
            ci_method="t" if ci_method == "t" else "simultaneous",
            fixed_axes=fixed_axes,
            major_step=major_step or None,
            minor_step=minor_step or None,
            pcr_result=diagnostic_result,
            bootstrap=int(bootstrap_graph),
            seed=202604,
        )
        for warning in plot_output.warnings:
            st.warning(warning)
        st.pyplot(plot_output.figure, width="stretch")
        st.caption(plot_output.caption)
        export_spec = {
            "dataset": dataset_key,
            "revision": st.session_state.revision,
            "correction_mode": correction_mode,
            "tests": selected_tests,
            "mode": graph_mode,
            "axis": axis_mode,
            "ci": ci_method,
            "fixed": fixed_axes,
            "major": major_step,
            "minor": minor_step,
            "bootstrap": int(bootstrap_graph),
            "pcr": diagnostic_result.to_dict() if diagnostic_result is not None else None,
        }
        export_key = hashlib.sha1(
            json.dumps(export_spec, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()
        if st.button("Подготовить SVG, PDF и PNG 600 dpi", key="prepare_figure_exports"):
            with st.spinner("Формирование публикационных файлов…"):
                st.session_state.figure_exports[export_key] = {
                    "svg": export_figure(plot_output.figure, "svg"),
                    "pdf": export_figure(plot_output.figure, "pdf"),
                    "png": export_figure(plot_output.figure, "png"),
                }
        cached_exports = st.session_state.figure_exports.get(export_key)
        if cached_exports:
            exports = st.columns(3)
            exports[0].download_button(
                "SVG",
                cached_exports["svg"],
                "soil_stamp_antonov.svg",
                "image/svg+xml",
                width="stretch",
            )
            exports[1].download_button(
                "PDF",
                cached_exports["pdf"],
                "soil_stamp_antonov.pdf",
                "application/pdf",
                width="stretch",
            )
            exports[2].download_button(
                "PNG 600 dpi",
                cached_exports["png"],
                "soil_stamp_antonov_600dpi.png",
                "image/png",
                width="stretch",
            )
    except Exception as exc:
        plot_output = None
        st.error(f"График не построен: {exc}")

    if st.checkbox("Показать схему штампа и армирования"):
        schematic_figure = plot_stamp_schematic(metadata)
        st.pyplot(schematic_figure, width="content")
        plt.close(schematic_figure)

with tabs[3]:
    analysis_test = st.selectbox("Испытание", selected_tests, key="analysis_test")
    test_frame = filtered[filtered["test_id"].astype(str) == analysis_test]
    a1, a2, a3 = st.columns(3)
    bootstrap_n = a1.number_input("Bootstrap pcr/E", min_value=100, max_value=10000, value=500, step=100)
    seed = a2.number_input("Seed", min_value=0, value=202604, step=1)
    calculate = a3.button("Рассчитать pcr", type="primary", width="stretch")
    result_context = f"{dataset_key}:{st.session_state.revision}:{correction_mode}:{analysis_test}"
    result_key = f"{result_context}:{int(bootstrap_n)}:{int(seed)}"
    if calculate:
        try:
            with st.spinner("Сегментированная регрессия и bootstrap…"):
                st.session_state.pcr_results[result_key] = fit_segmented_pcr(
                    test_frame, bootstrap=int(bootstrap_n), seed=int(seed)
                )
                st.session_state.pcr_latest[result_context] = result_key
                st.session_state.bundle_cache = {}
        except Exception as exc:
            st.error(f"pcr не рассчитано: {exc}")
    pcr_result = st.session_state.pcr_results.get(result_key)
    if pcr_result:
        metrics = st.columns(4)
        metrics[0].metric("pcr auto, кПа", f"{pcr_result.pcr_auto:.2f}")
        metrics[1].metric(
            "95% ДИ, кПа",
            (
                f"{pcr_result.pcr_ci_low:.2f}–{pcr_result.pcr_ci_high:.2f}"
                if pcr_result.pcr_ci_low is not None
                else "неустойчив"
            ),
        )
        metrics[2].metric("R²", f"{pcr_result.r2:.4f}")
        metrics[3].metric("AIC / BIC", f"{pcr_result.aic:.1f} / {pcr_result.bic:.1f}")
        diag = plot_curves(
            test_frame, mode="diagnostic", axis_mode="p-s", pcr_result=pcr_result
        )
        st.pyplot(diag.figure, width="stretch")
        plt.close(diag.figure)
        with st.expander("Коэффициенты, остатки и альтернативная двухлинейная модель"):
            st.json(pcr_result.to_dict())
        with st.form("manual_pcr_form"):
            manual_pcr_value = st.number_input("Подтверждённое pcr, кПа", value=float(pcr_result.pcr_auto))
            manual_pcr_reason = st.text_input("Обоснование ручного решения", key="pcr_reason")
            confirm_pcr = st.form_submit_button("Сохранить рядом с автоматическим")
        if confirm_pcr:
            try:
                st.session_state.pcr_results[result_key] = confirm_manual_pcr(
                    pcr_result,
                    manual_pcr_value,
                    reason=manual_pcr_reason,
                    audit=st.session_state.audit,
                    scope=analysis_test,
                )
                st.session_state.bundle_cache = {}
                st.success("Автоматический результат сохранён; ручное подтверждение добавлено отдельно.")
                st.rerun()
            except ValueError as exc:
                st.error(str(exc))
    else:
        st.info("Запустите расчёт pcr. Используются устойчивые точки первой ветви loading.")

    st.divider()
    st.subheader("E_stamp_app и жёсткость")
    finite_p = test_frame.loc[
        (test_frame["branch"] == "loading") & test_frame["p_kPa"].notna() & test_frame["settlement_mm"].notna(),
        "p_kPa",
    ]
    if finite_p.nunique() >= 2:
        pmin_data, pmax_data = float(finite_p.min()), float(finite_p.max())
        e_controls = st.columns(4)
        p_range = e_controls[0].slider(
            "Диапазон p, кПа",
            min_value=pmin_data,
            max_value=pmax_data,
            value=(pmin_data, pmax_data),
        )
        nu = e_controls[1].number_input("ν", min_value=0.0, max_value=0.49, value=float(metadata.get("poisson_ratio", 0.30)), step=0.01)
        shape_factor = e_controls[2].number_input("Коэффициент формы", min_value=0.01, value=float(metadata.get("shape_factor", 1.0)), step=0.05)
        calculate_e = e_controls[3].button("Рассчитать E", width="stretch")
        e_spec = {
            "dataset": dataset_key,
            "revision": st.session_state.revision,
            "correction_mode": correction_mode,
            "test_id": analysis_test,
            "p_range": [float(p_range[0]), float(p_range[1])],
            "nu": float(nu),
            "shape_factor": float(shape_factor),
            "bootstrap": int(bootstrap_n),
            "seed": int(seed),
        }
        e_key = "E:" + hashlib.sha1(
            json.dumps(e_spec, sort_keys=True).encode("utf-8")
        ).hexdigest()
        if calculate_e:
            try:
                e_result = estimate_moduli(
                    test_frame,
                    p_min_kpa=p_range[0],
                    p_max_kpa=p_range[1],
                    nu=float(nu),
                    shape_factor=float(shape_factor),
                    bootstrap=int(bootstrap_n),
                    seed=int(seed),
                )
                e_result.insert(0, "test_id", analysis_test)
                e_result.attrs["analysis_spec"] = e_spec
                st.session_state.analysis_tables[e_key] = e_result
                st.session_state.bundle_cache = {}
            except Exception as exc:
                st.error(f"Модуль не рассчитан: {exc}")
        moduli = st.session_state.analysis_tables.get(e_key)
        if moduli is not None:
            primary = moduli[moduli["method"].isin(["E_regression", "E_secant"])]
            st.dataframe(primary, width="stretch", hide_index=True)
            with st.expander("E_tangent и E_incremental_diagnostic"):
                st.dataframe(moduli[~moduli.index.isin(primary.index)], width="stretch", hide_index=True)
            regression_row = primary[primary["method"] == "E_regression"].iloc[0]
            sensitivity = modulus_sensitivity(
                regression_row["slope_m_per_kPa"] * 1000.0,
                float(test_frame["D_mm"].dropna().iloc[0]),
            )
            with st.expander("Чувствительность к ν и коэффициенту формы"):
                st.dataframe(sensitivity, width="stretch", hide_index=True)
    else:
        st.warning("Недостаточно loading-точек для E.")

with tabs[4]:
    groups = filtered["group"].astype(str).unique().tolist()
    if len(groups) < 2:
        st.info("Для сравнения нужны минимум две группы.")
    else:
        gcols = st.columns(4)
        baseline_group = gcols[0].selectbox("Baseline", groups, index=0)
        reinforced_candidates = [name for name in groups if name != baseline_group]
        reinforced_group = gcols[1].selectbox("Reinforced", reinforced_candidates)
        compare_bootstrap = gcols[2].number_input("Bootstrap сравнения", 100, 10000, 1000, 100)
        run_compare = gcols[3].button("Сравнить", type="primary", width="stretch")
        comparison_spec = {
            "dataset": dataset_key,
            "revision": st.session_state.revision,
            "correction_mode": correction_mode,
            "selected_tests": sorted(selected_tests),
            "baseline_group": baseline_group,
            "reinforced_group": reinforced_group,
            "bootstrap": int(compare_bootstrap),
            "seed": 202604,
        }
        comparison_key = "CMP:" + hashlib.sha1(
            json.dumps(comparison_spec, sort_keys=True).encode("utf-8")
        ).hexdigest()
        if run_compare:
            try:
                comparison_result = compare_groups(
                    filtered,
                    baseline_group,
                    reinforced_group,
                    bootstrap=int(compare_bootstrap),
                    seed=202604,
                )
                comparison_result.insert(0, "baseline_group", baseline_group)
                comparison_result.insert(1, "reinforced_group", reinforced_group)
                comparison_result.attrs["analysis_spec"] = comparison_spec
                st.session_state.analysis_tables[comparison_key] = comparison_result
                st.session_state.bundle_cache = {}
            except Exception as exc:
                st.error(f"Сравнение не рассчитано: {exc}")
        comparison = st.session_state.analysis_tables.get(comparison_key)
        if comparison is not None:
            st.dataframe(comparison, width="stretch", hide_index=True)
            fig, axes = plt.subplots(2, 1, figsize=(7.2, 6.2), sharex=True, constrained_layout=True)
            axes[0].plot(comparison["p_kPa"], comparison["k_s"], color="black", marker="o", markerfacecolor="white")
            axes[0].fill_between(
                comparison["p_kPa"], comparison["k_s_ci_low"], comparison["k_s_ci_high"], color="black", alpha=0.1
            )
            axes[0].axhline(1.0, color="0.5", linestyle="--")
            axes[0].set_ylabel("k_s = s_reinf/s_base")
            axes[1].plot(comparison["p_kPa"], comparison["delta_s_mm"], color="black", marker="s", markerfacecolor="white")
            axes[1].fill_between(
                comparison["p_kPa"], comparison["delta_s_ci_low_mm"], comparison["delta_s_ci_high_mm"], color="black", alpha=0.1
            )
            axes[1].set_ylabel("Δs, мм")
            axes[1].set_xlabel("p, кПа")
            for axis in axes:
                axis.grid(True, color="0.82", linewidth=0.5)
            st.pyplot(fig, width="stretch")
            plt.close(fig)
            if min(comparison["n_baseline"].min(), comparison["n_reinforced"].min()) < 5:
                st.warning("n < 5: сравнение следует трактовать как описательное.")

with tabs[5]:
    st.subheader("Инкременты, податливость и жёсткость")
    derivatives = derivative_diagnostics(filtered)
    st.dataframe(derivatives, width="stretch", hide_index=True)
    qcols = st.columns(2)
    target_p = qcols[0].number_input("Осадка при p, кПа", value=100.0)
    target_s = qcols[1].number_input("Давление при s, мм", value=1.0)
    settlement_at_target = value_at_pressure(filtered, target_p)
    pressure_at_target = pressure_at_settlement(filtered, target_s)
    st.dataframe(settlement_at_target, width="stretch", hide_index=True)
    st.dataframe(pressure_at_target, width="stretch", hide_index=True)

    st.subheader("Работа деформирования и разгрузка")
    work_table = deformation_work(filtered)
    st.dataframe(work_table, width="stretch", hide_index=True)
    hysteresis = hysteresis_metrics(filtered)
    if hysteresis.empty:
        st.info("В выбранных испытаниях нет полной ветви разгрузки.")
    else:
        st.dataframe(hysteresis, width="stretch", hide_index=True)

    st.subheader("Время стабилизации")
    rate = st.number_input("Порог |ds/dt|, мм/мин", min_value=0.0001, value=0.01, format="%.4f")
    stabilization = time_stabilization(filtered, rate_threshold_mm_per_min=float(rate))
    if stabilization.empty:
        st.info("Для расчёта нужны повторные временные измерения внутри ступени.")
    else:
        st.dataframe(stabilization, width="stretch", hide_index=True)

    st.subheader("Осадка центра и крен")
    tilt = center_and_tilt(
        filtered,
        metadata.get("indicator_positions_mm"),
    )
    if tilt.empty:
        if any(f"indicator_{index}" in filtered.columns for index in range(1, 5)) and not bool(
            filtered.get("indicator_calibration_confirmed", pd.Series(False, index=filtered.index)).any()
        ):
            st.info(
                "Расчёт центра и крена отключён: калибровка indicator_* не подтверждена явно. "
                "Прямая settlement при этом остаётся активной."
            )
        else:
            st.info("Направление крена доступно только при координатах минимум трёх индикаторов.")
    else:
        st.dataframe(tilt, width="stretch", hide_index=True)

with tabs[6]:
    pcr_by_test = {}
    for test_id in selected_tests:
        context = f"{dataset_key}:{st.session_state.revision}:{correction_mode}:{test_id}"
        latest_key = st.session_state.pcr_latest.get(context)
        if latest_key and latest_key in st.session_state.pcr_results:
            pcr_by_test[test_id] = st.session_state.pcr_results[latest_key]
    e_tables = [
        value
        for value in st.session_state.analysis_tables.values()
        if isinstance(value, pd.DataFrame)
        and value.attrs.get("analysis_spec", {}).get("dataset") == dataset_key
        and value.attrs.get("analysis_spec", {}).get("revision") == st.session_state.revision
        and value.attrs.get("analysis_spec", {}).get("correction_mode") == correction_mode
        and value.attrs.get("analysis_spec", {}).get("test_id") in selected_tests
    ]
    report_moduli = pd.concat(e_tables, ignore_index=True) if e_tables else None
    current_caption = plot_output.caption if "plot_output" in locals() and plot_output is not None else None
    current_warnings = plot_output.warnings if "plot_output" in locals() and plot_output is not None else []

    def analysis_table_in_scope(analysis_table) -> bool:
        if not isinstance(analysis_table, pd.DataFrame):
            return False
        spec = analysis_table.attrs.get("analysis_spec", {})
        spec_tests = spec.get("selected_tests", [spec.get("test_id")])
        return bool(
            spec.get("dataset") == dataset_key
            and spec.get("revision") == st.session_state.revision
            and spec.get("correction_mode") == correction_mode
            and set(filter(None, spec_tests)).issubset(set(selected_tests))
        )

    analysis_specs_for_provenance = []
    for analysis_key, analysis_table in st.session_state.analysis_tables.items():
        if analysis_table_in_scope(analysis_table):
            analysis_specs_for_provenance.append(
                {
                    "key": analysis_key,
                    "spec": analysis_table.attrs.get("analysis_spec", {}),
                }
            )
    processing_config = {
        "import": input_context["provenance_config"],
        "revision": st.session_state.revision,
        "correction_mode": correction_mode,
        "seating_offsets_mm": st.session_state.seating_offsets,
        "manual_overrides": st.session_state.manual_overrides,
        "selected_tests": sorted(selected_tests),
        "graph": locals().get("export_spec"),
        "pcr": {key: value.to_dict() for key, value in pcr_by_test.items()},
        "analysis_specs": analysis_specs_for_provenance,
        "target_pressure_kPa": float(target_p),
        "target_settlement_mm": float(target_s),
        "stabilization_rate_mm_min": float(rate),
        "audit_decisions": [
            {
                key: value
                for key, value in event.items()
                if key not in {"event_id", "timestamp_utc"}
            }
            for event in st.session_state.audit.events
        ],
    }
    processing_config_key = value_sha256(processing_config)
    if "processing_provenance" not in st.session_state:
        st.session_state.processing_provenance = {}
    if processing_config_key not in st.session_state.processing_provenance:
        st.session_state.processing_provenance[processing_config_key] = build_provenance(
            input_source=input_context["source_file_bytes"],
            metadata_source=input_context["metadata_file_bytes"],
            config=processing_config,
            project_root=BASE_DIR,
        )
    processing_provenance = st.session_state.processing_provenance[processing_config_key]
    report_prepared = filtered.copy(deep=False)
    report_prepared.attrs["indicator_processing_audit"] = (
        selected_indicator_audit.to_dict(orient="records")
    )
    report_prepared.attrs["indicator_processing_events"] = (
        selected_indicator_events.to_dict(orient="records")
    )
    report_prepared.attrs["indicator_calibration_parameters"] = (
        selected_indicator_passports.to_dict(orient="records")
    )
    report = build_markdown_report(
        metadata=metadata,
        prepared=report_prepared,
        validation_issues=all_issues,
        failures=failures[failures["test_id"].isin(selected_tests)],
        pcr_results=pcr_by_test,
        moduli=report_moduli,
        figure_caption=current_caption,
        plot_warnings=current_warnings,
        audit=st.session_state.audit,
        provenance=processing_provenance,
        passport_status=passport_completeness(metadata, selected_tests),
        import_info=import_info,
        source_test_ids=raw["test_id"].dropna().astype(str).unique().tolist(),
        source_row_count=len(raw),
    )
    st.subheader("Отчёт")
    st.markdown(report)
    st.download_button("Скачать отчёт Markdown", report.encode("utf-8"), "soil_stamp_report_ru.md", "text/markdown")

    result_tables = {
        "failure_summary": failures[failures["test_id"].isin(selected_tests)],
        "audit": st.session_state.audit.to_frame(),
        "pcr": {key: value.to_dict() for key, value in pcr_by_test.items()},
        "derivatives": derivatives,
        "settlement_at_target_pressure": settlement_at_target,
        "pressure_at_target_settlement": pressure_at_target,
        "deformation_work": work_table,
        "hysteresis": hysteresis,
        "stabilization": stabilization,
        "center_and_tilt": tilt,
        "validation_issues": [item.to_dict() for item in all_issues],
        "provenance": processing_provenance.to_dict(),
        "conversion_parameters": pd.DataFrame(
            effective_conversion_parameters(metadata, selected_tests)
        ),
        "indicator_processing_audit": selected_indicator_audit,
        "indicator_processing_events": selected_indicator_events,
        "indicator_calibration_parameters": selected_indicator_passports,
    }
    analysis_manifest = {}
    for name, table in st.session_state.analysis_tables.items():
        spec = table.attrs.get("analysis_spec", {}) if isinstance(table, pd.DataFrame) else {}
        if analysis_table_in_scope(table):
            safe_name = hashlib.sha1(name.encode()).hexdigest()[:10]
            result_tables[f"analysis_{safe_name}"] = table
            analysis_manifest[f"analysis_{safe_name}"] = spec
    result_tables["analysis_manifest"] = analysis_manifest
    bundle_spec = {
        "dataset": dataset_key,
        "revision": st.session_state.revision,
        "correction_mode": correction_mode,
        "tests": selected_tests,
        "caption": current_caption,
        "graph": locals().get("export_spec"),
        "pcr": {key: value.to_dict() for key, value in pcr_by_test.items()},
        "analysis_manifest": analysis_manifest,
        "target_pressure_kPa": float(target_p),
        "target_settlement_mm": float(target_s),
        "stabilization_rate_mm_min": float(rate),
        "audit_events": len(st.session_state.audit.events),
        "provenance_config_sha256": processing_provenance.config_sha256,
    }
    bundle_key = hashlib.sha1(
        json.dumps(bundle_spec, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()
    if st.button("Собрать пакет воспроизводимости", key="prepare_bundle", type="primary"):
        with st.spinner("Формирование ZIP с данными, результатами и рисунками…"):
            figure_payloads = {}
            if "plot_output" in locals() and plot_output is not None:
                graph_exports = st.session_state.figure_exports.get(
                    locals().get("export_key", ""), {}
                )
                if not graph_exports:
                    graph_exports = {
                        "svg": export_figure(plot_output.figure, "svg"),
                        "pdf": export_figure(plot_output.figure, "pdf"),
                        "png": export_figure(plot_output.figure, "png"),
                    }
                figure_payloads = {
                    "current.svg": graph_exports["svg"],
                    "current.pdf": graph_exports["pdf"],
                    "current_600dpi.png": graph_exports["png"],
                }
            st.session_state.bundle_cache[bundle_key] = reproducibility_bundle(
                raw=raw,
                prepared=filtered,
                metadata=metadata,
                audit=st.session_state.audit,
                report_markdown=report,
                result_tables=result_tables,
                figures=figure_payloads,
                run_parameters={
                    "dataset_sha256": dataset_key,
                    "revision": st.session_state.revision,
                    "correction_mode": correction_mode,
                    "selected_tests": selected_tests,
                    "graph": locals().get("export_spec"),
                    "pcr": {key: value.to_dict() for key, value in pcr_by_test.items()},
                    "analysis_manifest": analysis_manifest,
                    "target_pressure_kPa": float(target_p),
                    "target_settlement_mm": float(target_s),
                    "stabilization_rate_mm_min": float(rate),
                },
                provenance=processing_provenance,
                raw_cells=input_context["raw_cells"],
                import_issues=all_issues,
                source_file_name=input_context["source_file_name"],
                source_file_bytes=input_context["source_file_bytes"],
                metadata_file_name=input_context["metadata_file_name"],
                metadata_file_bytes=input_context["metadata_file_bytes"],
                config_snapshot=processing_config,
                scope={
                    "source_test_ids": raw["test_id"].dropna().astype(str).unique().tolist(),
                    "selected_test_ids": sorted(selected_tests),
                    "excluded_test_ids": sorted(set(raw["test_id"].dropna().astype(str)) - set(selected_tests)),
                    "source_rows": len(raw),
                    "prepared_rows": len(filtered),
                },
            )
    cached_bundle = st.session_state.bundle_cache.get(bundle_key)
    if cached_bundle:
        st.download_button(
            "Скачать пакет воспроизводимости ZIP",
            cached_bundle,
            "soil_stamp_reproducibility.zip",
            "application/zip",
        )
    st.subheader("Audit trail")
    audit_frame = st.session_state.audit.to_frame()
    if audit_frame.empty:
        st.info("Ручных решений пока нет.")
    else:
        st.dataframe(audit_frame, width="stretch", hide_index=True)
        st.download_button(
            "Скачать audit JSON",
            st.session_state.audit.to_json().encode("utf-8"),
            "audit.json",
            "application/json",
        )

if "plot_output" in locals() and plot_output is not None:
    plt.close(plot_output.figure)
