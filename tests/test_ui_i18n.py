from __future__ import annotations

import csv
import json
import re
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
import streamlit.testing.v1.app_test as app_test_module
from pandas.testing import assert_frame_equal
from streamlit.runtime.memory_media_file_storage import MemoryMediaFileStorage
from streamlit.testing.v1 import AppTest

from soilstamp.ui.formatters import display_dataframe, format_method, format_status
from soilstamp.ui.i18n import display_text, label_for_enum


ROOT = Path(__file__).resolve().parents[1]


# This is the normative subset supplied in UI_TRANSLATION_DICTIONARY.csv.
# Keeping it here makes an accidental rename of a machine token fail loudly.
REQUIRED_ENUM_LABELS = (
    ("correction", "raw", "Исходные данные"),
    ("correction", "zero_shifted", "Нулевой уровень по измеренной точке"),
    ("correction", "seating_corrected", "С посадочной поправкой"),
    ("import", "strict", "Строгий импорт"),
    ("import", "interactive", "Сопоставление столбцов"),
    ("import", "heuristic", "Совместимость со старым форматом"),
    ("graph", "raw_protocol", "Протокол испытания"),
    ("graph", "antonov_publication", "Публикационный график по Антонову"),
    ("graph", "group_mean_ci", "Средняя кривая и 95% доверительный интервал"),
    ("graph", "diagnostic", "Диагностика критического давления"),
    ("graph", "normalized", "Нормированные координаты"),
    ("curve", "mean_curve", "Средняя кривая"),
    ("curve", "median_curve", "Медианная кривая"),
    ("curve", "manual_representative", "Выбранный инженером опыт"),
    ("curve", "individual_curves", "Все индивидуальные кривые"),
    ("load_kind", "force", "Сила"),
    ("load_kind", "pressure", "Давление"),
    ("group", "baseline", "Контрольная группа"),
    ("group", "reinforced", "Армированная группа"),
    ("stamp_shape", "circle", "Круглый штамп"),
    ("stamp_shape", "custom", "Другая форма"),
    ("indicator_mode", "increasing", "Прямая шкала"),
    (
        "indicator_mode",
        "increasing_wrapped",
        "Прямая шкала с переходом через ноль",
    ),
    ("indicator_mode", "decreasing", "Обратная шкала"),
    (
        "indicator_mode",
        "decreasing_wrapped",
        "Обратная шкала с переходом через ноль",
    ),
    (
        "indicator_mode",
        "cumulative_settlement",
        "Готовая накопленная осадка",
    ),
    ("status", "draft", "Черновик"),
    ("status", "review_required", "Требуется проверка"),
    ("status", "confirmed", "Подтверждено"),
    (
        "status",
        "migration_review_required",
        "Проверка после переноса данных",
    ),
    ("missing_policy", "block", "Блокировать расчёт"),
    (
        "missing_policy",
        "allow_if_solvable",
        "Разрешить при достаточных данных",
    ),
    ("branch", "loading", "Нагружение"),
    ("branch", "hold", "Выдержка"),
    ("branch", "unloading", "Разгрузка"),
    ("branch", "reloading", "Повторное нагружение"),
    ("branch", "cyclic", "Циклическая ветвь"),
    ("row_status", "measurement", "Измерение"),
    ("row_status", "failure", "Разрушение"),
    ("row_status", "instrument_limit", "Предел хода индикатора"),
    (
        "row_status",
        "stopped_without_failure",
        "Остановлено без разрушения",
    ),
    ("row_status", "invalid", "Недопустимая строка"),
)


ADDITIONAL_REQUIRED_DOMAINS = {
    "aggregation": (
        "all_channels_mean",
        "selected_channels_mean",
        "plane_center",
        "primary_channel",
        "no_aggregation",
    ),
    "censoring": ("interval_censored", "right_censored", "indeterminate"),
    "methodology_profile": (
        "antonov_round_stamp_v1",
        "custom_v1",
        "diagnostic_unapproved_v1",
    ),
    "review": ("review_required", "approved", "unsigned"),
}


FORBIDDEN_MACHINE_OPTIONS = {
    machine_value for _, machine_value, _ in REQUIRED_ENUM_LABELS
} | {
    value for values in ADDITIONAL_REQUIRED_DOMAINS.values() for value in values
}


EXPECTED_VISIBLE_MACHINE_VALUES = {
    "raw",
    "raw_protocol",
    "force",
    "diagnostic_unapproved_v1",
    "baseline",
    "reinforced",
    "circle",
    "block",
    "draft",
    "no_aggregation",
    "increasing",
}


FORBIDDEN_WIDGET_LABELS = {
    "baseline",
    "reinforced",
    "bootstrap",
    "bootstrap pcr/e",
    "seed",
    "major step (0 = auto)",
    "minor step (0 = auto)",
    "x min",
    "x max",
    "metadata",
    "provenance",
    "audit trail",
    "qc",
}


# Representative canonical headings from every required visible table family.
# They may remain in CSV/JSON downloads, but not as the rendered table heading.
FORBIDDEN_VISIBLE_COLUMNS = {
    "test_id",
    "sequence_no",
    "sequence_index",
    "branch",
    "row_status",
    "F_kN",
    "p_kPa",
    "settlement_raw_mm",
    "settlement_mm",
    "load_kind",
    "indicator_type",
    "serial_number",
    "verification_status",
    "original_reading",
    "turn_number",
    "computed_increment_mm",
    "applied_correction_mm",
    "cumulative_settlement_mm",
    "processing_status",
    "conversion_method",
    "aggregation_method",
    "aggregation_status",
    "level",
    "code",
    "message",
    "censoring_type",
    "classification_status",
    "lower_bound",
    "upper_bound",
    "correction_mode",
    "method",
    "profile_id",
    "review_status",
    "event_id",
    "timestamp_utc",
    "action",
    "scope",
    "reason",
    "before_hash",
    "after_hash",
    "field",
    "value",
    "formula",
    "raw_before",
    "raw_after",
    "correction_mm",
    "candidate_turns",
    "capacity_lower_inclusive",
    "capacity_upper_inclusive",
    "failure_event_count",
    "lower_bound_sequence_no",
    "upper_bound_sequence_no",
    "difference_mm",
    "ds_dp_mm_per_kPa",
    "dp_ds_kPa_per_mm",
    "crossing",
    "W_total_kJ_m2",
    "integrated_segments",
    "skipped_gaps",
    "W_loading_kJ_m2",
    "W_hold_kJ_m2",
    "W_unloading_kJ_m2",
    "W_reloading_kJ_m2",
    "W_cyclic_kJ_m2",
    "W_unknown_kJ_m2",
    "s_peak_mm",
    "s_residual_mm",
    "s_recoverable_mm",
    "residual_pressure_kPa",
    "hysteresis_energy_kJ_m2",
    "loop_closed",
    "user",
    "before_value",
    "after_value",
}


P0_COLUMN_LABELS = {
    "formula": "Формула пересчёта",
    "raw_before": "Показание до события",
    "raw_after": "Показание после события",
    "correction_mm": "Коррекция, мм",
    "candidate_turns": "Допустимые номера оборотов",
    "capacity_lower_inclusive": "Нижняя граница несущей способности включена",
    "capacity_upper_inclusive": "Верхняя граница несущей способности включена",
    "failure_event_count": "Количество событий разрушения",
    "lower_bound_sequence_no": "№ ступени нижней границы",
    "upper_bound_sequence_no": "№ ступени верхней границы",
    "difference_mm": "Разность осадок, мм",
    "ds_dp_mm_per_kPa": "Приращение осадки ds/dp, мм/кПа",
    "dp_ds_kPa_per_mm": "Жёсткость dp/ds, кПа/мм",
    "crossing": "Пересечение",
    "W_total_kJ_m2": "Полная работа деформации, кДж/м²",
    "integrated_segments": "Проинтегрировано сегментов",
    "skipped_gaps": "Пропущено разрывов данных",
    "W_loading_kJ_m2": "Работа при нагружении, кДж/м²",
    "W_hold_kJ_m2": "Работа при выдержке, кДж/м²",
    "W_unloading_kJ_m2": "Работа при разгрузке, кДж/м²",
    "W_reloading_kJ_m2": "Работа при повторном нагружении, кДж/м²",
    "W_cyclic_kJ_m2": "Работа на циклической ветви, кДж/м²",
    "W_unknown_kJ_m2": "Работа на неопределённой ветви, кДж/м²",
    "s_peak_mm": "Пиковая осадка, мм",
    "s_residual_mm": "Остаточная осадка, мм",
    "s_recoverable_mm": "Восстанавливаемая осадка, мм",
    "residual_pressure_kPa": "Давление при остаточной осадке, кПа",
    "hysteresis_energy_kJ_m2": "Энергия гистерезиса, кДж/м²",
    "loop_closed": "Контур гистерезиса замкнут",
    "user": "Пользователь",
    "before_value": "Значение до изменения",
    "after_value": "Значение после изменения",
}


P0_VALUE_LABELS = (
    ("status", "warning", "Предупреждение"),
    ("status", "info", "Информация"),
    ("status", "unresolved", "Не определено"),
    (
        "status",
        "not_applied_direct_settlement",
        "Не применяется для непосредственно заданной осадки",
    ),
    ("audit", "missing_reading", "Показание отсутствует"),
    ("audit", "repeated_reading", "Показание повторяется"),
    ("method", "preserve_nan", "Сохранение пропущенного значения NaN"),
    ("method", "unresolved", "Способ не определён"),
    ("missing_policy", "unresolved", "Политика не определена"),
    ("aggregation", "unresolved", "Агрегация не определена"),
    ("audit", "import_dataset", "Импорт набора данных"),
)


@pytest.mark.parametrize(("domain", "machine_value", "expected"), REQUIRED_ENUM_LABELS)
def test_required_russian_enum_labels(
    domain: str, machine_value: str, expected: str
) -> None:
    assert label_for_enum(domain, machine_value) == expected


@pytest.mark.parametrize(
    ("domain", "machine_value"),
    [
        (domain, machine_value)
        for domain, machine_values in ADDITIONAL_REQUIRED_DOMAINS.items()
        for machine_value in machine_values
    ],
)
def test_additional_required_domains_have_russian_display_labels(
    domain: str, machine_value: str
) -> None:
    rendered = label_for_enum(domain, machine_value)

    assert rendered != machine_value
    assert re.search(r"[А-Яа-яЁё]", rendered)


def test_unknown_machine_tokens_are_losslessly_preserved() -> None:
    token = "future_schema_token_v99"

    assert label_for_enum("future_domain", token) == token
    assert format_status(token) == token
    assert format_method(token) == token


def test_failure_classification_ok_is_localized_only_in_display_copy() -> None:
    source = pd.DataFrame({"classification_status": ["ok"]})

    rendered = display_dataframe(source, "failure")

    assert rendered.to_dict(orient="records") == [
        {"Статус классификации": "Без замечаний"}
    ]
    assert source.to_dict(orient="records") == [{"classification_status": "ok"}]


def test_generated_visible_text_localizes_group_and_branch_tokens_only() -> None:
    source = "B-01 — baseline; R-01 — reinforced; loading; custom_token"

    rendered = display_text(source)

    assert rendered == (
        "B-01 — контрольная группа; R-01 — армированная группа; "
        "нагружение; custom_token"
    )
    assert source == "B-01 — baseline; R-01 — reinforced; loading; custom_token"


def test_status_and_method_formatters_are_display_only() -> None:
    status = "review_required"
    method = "E_regression"

    assert format_status(status) == label_for_enum("status", status)
    assert format_method(method) != method
    assert status == "review_required"
    assert method == "E_regression"


@pytest.mark.parametrize(("domain", "machine_value", "expected"), P0_VALUE_LABELS)
def test_p0_rendered_machine_values_are_russian(
    domain: str, machine_value: str, expected: str
) -> None:
    assert label_for_enum(domain, machine_value) == expected


def test_p0_rendered_column_labels_do_not_change_canonical_columns() -> None:
    source = pd.DataFrame({column: ["machine-value"] for column in P0_COLUMN_LABELS})
    snapshot = source.copy(deep=True)

    displayed = display_dataframe(source, "generic")

    assert_frame_equal(source, snapshot)
    assert tuple(source.columns) == tuple(P0_COLUMN_LABELS)
    assert tuple(displayed.columns) == tuple(P0_COLUMN_LABELS.values())


def test_indicator_flags_and_statuses_localize_only_in_display_copy() -> None:
    source = pd.DataFrame(
        {
            "warning": ["missing_reading;repeated_reading"],
            "quality_flags": ["repeated_reading"],
            "processing_status": ["info"],
            "aggregation_status": ["not_applied_direct_settlement"],
            "conversion_method": ["preserve_nan"],
            "missing_channel_policy": ["unresolved"],
        }
    )
    snapshot = source.copy(deep=True)
    canonical_csv = source.to_csv(index=False, lineterminator="\n")

    displayed = display_dataframe(source, "indicator_processing")

    assert_frame_equal(source, snapshot)
    assert source.to_csv(index=False, lineterminator="\n") == canonical_csv
    assert displayed.loc[0, "Предупреждение"] == (
        "Показание отсутствует; Показание повторяется"
    )
    assert displayed.loc[0, "Признаки качества"] == "Показание повторяется"
    assert displayed.loc[0, "Статус обработки"] == "Информация"
    assert displayed.loc[0, "Статус агрегации"] == (
        "Не применяется для непосредственно заданной осадки"
    )
    assert displayed.loc[0, "Способ преобразования"] == (
        "Сохранение пропущенного значения NaN"
    )
    assert displayed.loc[0, "Политика пропусков каналов"] == "Политика не определена"


def test_display_dataframe_does_not_mutate_canonical_frame_or_serialization() -> None:
    source = pd.DataFrame(
        {
            "test_id": ["T-01", "T-02"],
            "sequence_no": [0, 1],
            "F_kN": [0.0, 10.0],
            "p_kPa": [0.0, 100.0],
            "settlement_mm": [0.0, 0.4],
            "branch": ["loading", "unloading"],
            "row_status": ["measurement", "failure"],
            "review_status": ["review_required", "confirmed"],
            "method": ["E_regression", "E_secant"],
        }
    )
    source.attrs["machine_contract"] = {"version": "measurement/1.0"}
    snapshot = source.copy(deep=True)
    snapshot.attrs = {"machine_contract": {"version": "measurement/1.0"}}
    canonical_columns = tuple(source.columns)
    canonical_csv = source.to_csv(index=False, lineterminator="\n")
    canonical_json = source.to_json(orient="records", force_ascii=False)

    displayed = display_dataframe(source, "protocol")

    assert displayed is not source
    assert_frame_equal(source, snapshot)
    assert source.attrs == snapshot.attrs
    assert tuple(source.columns) == canonical_columns
    assert source.to_csv(index=False, lineterminator="\n") == canonical_csv
    assert source.to_json(orient="records", force_ascii=False) == canonical_json
    assert not FORBIDDEN_VISIBLE_COLUMNS.intersection(map(str, displayed.columns))
    assert "loading" not in displayed.astype(str).to_numpy()
    assert "failure" not in displayed.astype(str).to_numpy()


def test_qc_display_headings_are_unique_for_arrow_rendering() -> None:
    source = pd.DataFrame(
        {
            "level": ["error"],
            "severity": ["error"],
            "code": ["example"],
            "message": ["example"],
            "sheet": ["Лист1"],
            "row": [2],
            "column": ["A"],
            "suggested_action": ["example"],
            "blocks_processing": [True],
            "entity_id": ["T-01"],
        }
    )

    displayed = display_dataframe(source, "issues")

    assert displayed.columns.is_unique
    assert "level" not in displayed.columns
    assert "severity" not in displayed.columns


def _visible_widget_text(app: AppTest) -> tuple[list[str], list[str]]:
    labels: list[str] = []
    options: list[str] = []
    element_kinds = (
        "radio",
        "selectbox",
        "multiselect",
        "number_input",
        "text_input",
        "text_area",
        "checkbox",
        "button",
        "download_button",
        "file_uploader",
        "tabs",
        "expander",
    )
    for kind in element_kinds:
        for element in getattr(app, kind):
            label = getattr(element, "label", None)
            if label not in (None, ""):
                labels.append(str(label))
            element_options = getattr(element, "options", None)
            if element_options is not None:
                options.extend(str(option) for option in element_options)
    return labels, options


def _visible_dataframe_columns(app: AppTest) -> list[str]:
    visible: list[str] = []
    for table in app.dataframe:
        frame = table.value
        if not hasattr(frame, "columns"):
            continue
        config = json.loads(table.proto.columns) if table.proto.columns else {}
        for column in frame.columns:
            configured = config.get(str(column), {})
            visible.append(str(configured.get("label") or column))
    return visible


def _captured_downloads(app: AppTest, storage: MemoryMediaFileStorage) -> dict[str, bytes]:
    result: dict[str, bytes] = {}
    for element in app.download_button:
        stored = storage.get_file(element.proto.url.rsplit("/", 1)[-1])
        if stored.filename:
            result[stored.filename] = stored.content
    return result


def test_streamlit_visible_ui_is_russian_but_machine_download_is_canonical() -> None:
    storages: list[MemoryMediaFileStorage] = []

    class CapturingStorage(MemoryMediaFileStorage):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            storages.append(self)

    with patch.object(app_test_module, "MemoryMediaFileStorage", CapturingStorage):
        app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120).run(
            timeout=120
        )

    assert not app.exception
    assert len(storages) == 1

    labels, options = _visible_widget_text(app)
    label_keys = [label.strip().casefold() for label in labels]
    label_violations: dict[str, list[str]] = {}
    for forbidden in FORBIDDEN_WIDGET_LABELS:
        matches = [
            label
            for label, key in zip(labels, label_keys, strict=True)
            if key == forbidden
            or key.startswith(forbidden + " ")
            or (forbidden == "audit trail" and key.startswith("audit trail"))
            or (
                forbidden == "qc"
                and re.search(r"(?:^|\W)qc(?:$|\W)", key, flags=re.IGNORECASE)
            )
        ]
        if matches:
            label_violations[forbidden] = sorted(set(matches))

    option_violations = sorted(FORBIDDEN_MACHINE_OPTIONS.intersection(options))
    column_violations = sorted(
        FORBIDDEN_VISIBLE_COLUMNS.intersection(_visible_dataframe_columns(app))
    )
    assert not (label_violations or option_violations or column_violations), json.dumps(
        {
            "english_labels": label_violations,
            "machine_options": option_violations,
            "canonical_table_headings": column_violations,
        },
        ensure_ascii=False,
        indent=2,
    )

    scalar_machine_values = {
        str(element.value)
        for kind in ("radio", "selectbox")
        for element in getattr(app, kind)
        if not isinstance(element.value, (list, tuple, set, dict))
    }
    assert EXPECTED_VISIBLE_MACHINE_VALUES.issubset(scalar_machine_values)

    # format_func changes only the shown option: Streamlit keeps canonical values.
    correction = next(item for item in app.radio if item.value == "raw")
    graph = next(item for item in app.selectbox if item.value == "raw_protocol")
    profile = next(
        item for item in app.selectbox if item.value == "diagnostic_unapproved_v1"
    )
    assert correction.options[0] == label_for_enum("correction", "raw")
    assert graph.options[0] == label_for_enum("graph", "raw_protocol")
    assert profile.options[0] == label_for_enum(
        "methodology_profile", "antonov_round_stamp_v1"
    )
    assert correction.value == "raw"
    assert graph.value == "raw_protocol"
    assert profile.value == "diagnostic_unapproved_v1"

    downloads = _captured_downloads(app, storages[0])
    audit_bytes = downloads["indicator_processing_audit.csv"]
    reader = csv.DictReader(StringIO(audit_bytes.decode("utf-8-sig")))
    rows = list(reader)
    assert reader.fieldnames is not None
    assert {
        "test_id",
        "branch",
        "processing_status",
        "conversion_method",
    }.issubset(reader.fieldnames)
    assert "Испытание" not in reader.fieldnames
    assert {row["branch"] for row in rows} == {
        "loading",
        "hold",
        "unloading",
        "reloading",
    }
    assert {row["processing_status"] for row in rows} == {
        "ok",
        "warning",
        "missing",
    }
    assert {row["conversion_method"] for row in rows} == {
        "",
        "cumulative_settlement: s=sign·raw·factor+zero_correction",
    }
