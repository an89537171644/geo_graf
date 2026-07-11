from __future__ import annotations

from io import BytesIO

import numpy as np
import pandas as pd
import pytest
from openpyxl import Workbook

from soilstamp.data import failure_summary, prepare_measurements
from soilstamp.indicators import indicator_audit_frame
from soilstamp.io import read_protocol_excel
from soilstamp.manual_entry_adapter import (
    ManualExperimentSource,
    adapt_manual_draft,
)
from soilstamp.manual_entry_models import ManualDraft, ManualPoint


def _draft(
    *,
    readings: tuple[str | None, ...] = ("0", "0,20", "0,50"),
    loads: tuple[str, ...] = ("0", "1,0", "2,0"),
    mode: str = "cumulative_settlement",
    initial: str | None = None,
) -> ManualDraft:
    draft = ManualDraft.create(initial_rows=len(readings), author="engineer")
    passport = draft.passport
    values = {
        "project_name": "Doctoral project",
        "series_name": "Series A",
        "test_name": "M-001",
        "test_date": "2026-06-01",
        "operator": "Engineer",
        "laboratory_or_site": "Laboratory 1",
        "test_scope": "laboratory",
        "protocol_type": "static_step",
        "group_name": "baseline",
        "baseline_group": "baseline",
        "soil_type": "sand",
        "soil_batch": "batch-1",
        "reinforcement_type": "none",
        "stamp_shape": "circle",
        "stamp_diameter_mm": "300",
        "load_kind": "force",
        "load_unit": "kN",
        "load_factor": "1",
        "load_zero": "0",
        "lever_ratio": "1",
        "settlement_unit": "mm",
        "dial_mode": mode,
        "dial_range_mm": "10",
        "dial_resolution_mm": "0,01",
        "dial_correction_factor": "1",
        "dial_initial_reading": initial,
        "dial_zero_correction_mm": "0",
        "dial_max_increment_mm": "2",
        "dial_reverse_tolerance_mm": "0,02",
        "dial_travel_range_mm": "50",
        "indicator_type": "ИЧ-10",
        "indicator_serial_numbers": ["IND-001"],
        "verification_date": "2026-01-15",
        "verification_valid_until": "2030-01-15",
        "number_of_indicators": 1,
    }
    for name, value in values.items():
        setattr(passport, name, value)
    for index, (row, load, reading) in enumerate(zip(draft.rows, loads, readings)):
        row.stage_no = str(index)
        row.elapsed_time_s = str(index * 60)
        row.load_raw = load
        row.indicator_1_raw = reading
        row.comment = f"point {index}"
    return draft


def _workbook_bytes(readings: tuple[float, ...]) -> bytes:
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Protocol"
    sheet.append(["test_id", "stage", "load, kN", "indicator_1", "branch", "status"])
    for index, reading in enumerate(readings):
        sheet.append(["M-001", index, float(index), reading, "loading", "stable"])
    buffer = BytesIO()
    workbook.save(buffer)
    workbook.close()
    return buffer.getvalue()


def test_adapter_preserves_decimal_comma_uuid_and_manual_provenance() -> None:
    draft = _draft()
    bundle = adapt_manual_draft(draft)

    assert bundle.can_analyze
    assert bundle.raw["load"].tolist() == pytest.approx([0.0, 1.0, 2.0])
    assert bundle.raw["indicator_1"].tolist() == pytest.approx([0.0, 0.2, 0.5])
    assert bundle.raw["source_type"].eq("manual").all()
    assert bundle.raw["source_row"].isna().all()
    assert bundle.raw["manual_row_uuid"].tolist() == [row.manual_row_uuid for row in draft.rows]
    assert bundle.raw_cells.loc[
        bundle.raw_cells["canonical_field"].eq("indicator_1"), "raw_value"
    ].tolist() == ["0", "0,20", "0,50"]
    assert bundle.import_info["source_type"] == "manual"
    assert bundle.source_bytes == draft.to_json(indent=None).encode("utf-8")


def test_manual_source_loads_source_neutral_experiment_points() -> None:
    draft = _draft()
    draft.rows[1].stage_no = "01"
    draft.rows[1].elapsed_time_s = "1,50"
    experiments = ManualExperimentSource(draft).load()

    assert len(experiments) == 1
    assert experiments[0].test_id == "M-001"
    assert len(experiments[0].points) == 3
    point = experiments[0].points[1]
    assert point.load_raw == "1,0"
    assert point.stage == 1
    assert point.stage_raw == "01"
    assert point.elapsed_time_s == pytest.approx(1.5)
    assert point.elapsed_time_raw == "1,50"
    assert point.indicator_raws == {"indicator_1": "0,20"}
    assert point.source_type == "manual"
    assert point.source_row is None
    assert point.manual_row_uuid == draft.rows[1].manual_row_uuid
    assert point.created_by == "engineer"


def test_manual_and_equivalent_excel_use_identical_scientific_pipeline() -> None:
    draft = _draft()
    bundle = adapt_manual_draft(draft)
    manual, manual_issues = bundle.prepare()

    imported = read_protocol_excel(_workbook_bytes((0.0, 0.2, 0.5)), import_mode="strict")
    excel, excel_issues = prepare_measurements(
        imported.frame, bundle.metadata, strict_metadata=True
    )
    assert not [issue for issue in manual_issues if issue.level == "error"]
    assert not [issue for issue in excel_issues if issue.level == "error"]
    scientific = [
        "F_kN",
        "p_kPa",
        "settlement_raw_mm",
        "settlement_mm",
        "D_mm",
        "stamp_area_m2",
    ]
    pd.testing.assert_frame_equal(
        manual[scientific].reset_index(drop=True),
        excel[scientific].reset_index(drop=True),
        check_dtype=False,
        rtol=1e-12,
        atol=1e-12,
    )


def test_wrapped_manual_and_excel_indicator_conversion_are_equivalent() -> None:
    draft = _draft(
        readings=("9,80", "0,20", "0,70"),
        mode="increasing_wrapped",
        initial="9,80",
    )
    bundle = adapt_manual_draft(draft)
    manual, manual_issues = bundle.prepare()
    imported = read_protocol_excel(
        _workbook_bytes((9.80, 0.20, 0.70)), import_mode="strict"
    )
    excel, excel_issues = prepare_measurements(
        imported.frame, bundle.metadata, strict_metadata=True
    )

    assert not [issue for issue in manual_issues if issue.level == "error"]
    assert not [issue for issue in excel_issues if issue.level == "error"]
    assert manual["settlement_mm"].tolist() == pytest.approx([0.0, 0.4, 0.9])
    assert excel["settlement_mm"].tolist() == pytest.approx(
        manual["settlement_mm"].tolist()
    )
    audit_columns = [
        "channel",
        "original_reading",
        "turn_number",
        "computed_increment_mm",
        "cumulative_before_correction_mm",
        "applied_correction_mm",
        "cumulative_settlement_mm",
        "processing_status",
        "conversion_method",
        "warning",
    ]
    pd.testing.assert_frame_equal(
        indicator_audit_frame(manual)[audit_columns].reset_index(drop=True),
        indicator_audit_frame(excel)[audit_columns].reset_index(drop=True),
        check_dtype=False,
    )


def test_failure_without_indicator_keeps_failure_settlement_null() -> None:
    draft = _draft()
    failure = ManualPoint.create(4, author="engineer")
    failure.stage_no = "3"
    failure.elapsed_time_s = "180"
    failure.load_raw = "2,5"
    failure.indicator_1_raw = None
    failure.row_status = "failure"
    failure.comment = "explicit failure"
    draft.rows.append(failure)

    bundle = adapt_manual_draft(draft)
    prepared, issues = bundle.prepare()
    summary = failure_summary(prepared).iloc[0]

    assert not [issue for issue in issues if issue.level == "error"]
    assert bool(summary["failure_reached"])
    assert summary["F_failure_step"] == pytest.approx(2.5)
    assert pd.isna(summary["s_failure"])
    assert np.isnan(prepared.loc[prepared["is_failure"], "settlement_mm"]).all()


def test_cyclic_branch_is_preserved_without_reclassification() -> None:
    draft = _draft()
    draft.passport.protocol_type = "cyclic"
    draft.rows[1].branch = "cyclic"

    bundle = adapt_manual_draft(draft)
    prepared, issues = bundle.prepare()

    assert not [issue for issue in issues if bool(issue.blocks_processing)]
    assert bundle.raw.loc[1, "branch"] == "cyclic"
    assert prepared.loc[1, "branch"] == "cyclic"
    audit = indicator_audit_frame(prepared)
    cyclic_audit = audit[audit["sequence_index"].eq(1)].iloc[0]
    assert cyclic_audit["branch"] == "cyclic"
    assert cyclic_audit["branch_source"] == "protocol"


@pytest.mark.parametrize(
    ("manual_status", "canonical_status"),
    [
        ("measurement", "stable"),
        ("failure", "failure"),
        ("instrument_limit", "instrument_limit"),
        ("stopped_without_failure", "stopped_without_failure"),
        ("invalid", "invalid"),
    ],
)
def test_manual_status_is_preserved_with_explicit_canonical_mapping(
    manual_status: str, canonical_status: str
) -> None:
    draft = _draft()
    draft.rows[-1].row_status = manual_status

    raw = adapt_manual_draft(draft).raw.iloc[-1]

    assert raw["row_status"] == manual_status
    assert raw["status"] == canonical_status


def test_critical_adapter_or_pipeline_issue_blocks_analysis_without_hiding_rows() -> None:
    draft = _draft()
    draft.rows[1].load_raw = "not-a-number"
    bundle = adapt_manual_draft(draft)
    prepared, issues = bundle.prepare()

    assert not bundle.can_analyze
    assert any(issue.code == "invalid_manual_load" for issue in issues)
    assert len(prepared) == len(draft.rows)
    assert prepared.loc[1, "manual_row_uuid"] == draft.rows[1].manual_row_uuid
    assert "F_kN" not in prepared
    assert bundle.validation.adapter_issues
    assert bundle.validation.pipeline_issues
