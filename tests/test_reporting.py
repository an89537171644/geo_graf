from __future__ import annotations

import json
import zipfile
from io import BytesIO

import numpy as np
import pandas as pd
import pytest

from soilstamp.data import AuditTrail, failure_summary, prepare_measurements as _prepare_measurements
from soilstamp.provenance import build_provenance, passport_completeness
from soilstamp.reporting import build_markdown_report, format_ru, reproducibility_bundle


def prepare_measurements(*args, **kwargs):
    kwargs.setdefault("strict_metadata", False)
    return _prepare_measurements(*args, **kwargs)


def test_russian_decimal_and_no_false_precision() -> None:
    assert format_ru(12.34567, resolution=0.1, unit="кПа") == "12,3 кПа"
    assert format_ru(12.34567, resolution=0.01) == "12,35"
    assert format_ru(12.34567, resolution=0.01, uncertainty=0.2) == "12,3"
    assert format_ru(13.1, resolution=0.5) == "13,0"
    assert format_ru(13.1, resolution=0.25) == "13,00"


def test_failure_interval_in_report_uses_decimal_comma() -> None:
    raw = pd.DataFrame(
        {
            "test_id": ["T1", "T1"],
            "stage": [1, 2],
            "load": [1.25, 2.5],
            "settlement": [0.5, np.nan],
            "status": ["stable", "failure"],
        }
    )
    metadata = {
        "stamp_shape": "custom",
        "stamp_area_m2": 0.1,
        "load_resolution_kN": 0.01,
    }
    prepared, issues = prepare_measurements(raw, metadata)
    report = build_markdown_report(
        metadata=metadata,
        prepared=prepared,
        validation_issues=issues,
        failures=failure_summary(prepared),
        audit=AuditTrail(),
    )
    assert "1,25 < Fu ≤ 2,50 кН" in report
    assert "1.25 < Fu" not in report


def test_report_records_failure_contract_counts_and_curve_selection() -> None:
    raw = pd.DataFrame(
        {
            "test_id": ["T1", "T1"],
            "stage": [1, 2],
            "load": [1.0, 2.0],
            "settlement": [0.2, np.nan],
            "status": ["stable", "failure"],
        }
    )
    metadata = {"stamp_area_m2": 0.1, "load_resolution_kN": 0.01}
    prepared, issues = prepare_measurements(raw, metadata)
    report = build_markdown_report(
        metadata=metadata,
        prepared=prepared,
        validation_issues=issues,
        failures=failure_summary(prepared),
        failure_analysis={
            "contract_version": "failure-analysis/1.0",
            "summary_method": "none",
            "point_estimate": None,
            "n_failure_observed": 1,
            "n_interval_censored": 1,
            "n_right_censored": 0,
            "n_indeterminate": 0,
        },
        curve_selections=[
            {
                "group": "baseline",
                "method": "manual_representative",
                "test_id": "T1",
                "author": "engineer",
                "timestamp_utc": "2026-07-12T06:00:00+00:00",
                "reason": "Representative specimen confirmed.",
            }
        ],
        audit=AuditTrail(),
    )

    assert "Наблюдавшихся разрушений: 1" in report
    assert "Интервально цензурированных: 1" in report
    assert "Метод сводной оценки: `none`" in report
    assert "Сводная точечная оценка Fu/pu не рассчитывалась" in report
    assert "method=`manual_representative`" in report
    assert "author=`engineer`" in report


def test_report_uses_canonical_pressure_capacity_and_indeterminate_state() -> None:
    pressure_raw = pd.DataFrame(
        {
            "test_id": ["P1", "P1"],
            "stage": [1, 2],
            "load": [100.0, 150.0],
            "settlement": [0.2, np.nan],
            "status": ["stable", "failure"],
        }
    )
    pressure_metadata = {
        "load_kind": "pressure",
        "load_unit": "kPa",
        "stamp_area_m2": 0.1,
    }
    pressure, issues = prepare_measurements(pressure_raw, pressure_metadata)
    pressure_report = build_markdown_report(
        metadata=pressure_metadata,
        prepared=pressure,
        validation_issues=issues,
        failures=failure_summary(pressure),
        audit=AuditTrail(),
    )

    assert "100,0 < pu ≤ 150,0 кПа" in pressure_report
    assert "< Fu ≤" not in pressure_report

    indeterminate = pressure.iloc[[0]].copy()
    indeterminate[["F_kN", "p_kPa"]] = np.nan
    indeterminate["status"] = "invalid"
    indeterminate["is_failure"] = False
    indeterminate_failures = failure_summary(indeterminate)
    indeterminate_report = build_markdown_report(
        metadata=pressure_metadata,
        prepared=indeterminate,
        validation_issues=[],
        failures=indeterminate_failures,
        audit=AuditTrail(),
    )

    assert "требуется инженерная проверка (indeterminate)" in indeterminate_report
    assert "(правое цензурирование)" not in indeterminate_report


def test_report_exposes_group_pairing_decision_and_fallback_warning() -> None:
    metadata = {"stamp_area_m2": 0.1, "indicator_resolution_mm": 0.01}
    raw = pd.DataFrame(
        {
            "test_id": ["B1", "R1"],
            "stage": [1, 1],
            "load": [1.0, 1.0],
            "settlement": [0.2, 0.1],
        }
    )
    prepared, issues = prepare_measurements(raw, metadata)
    comparison = pd.DataFrame(
        [
            {
                "baseline_group": "baseline",
                "reinforced_group": "reinforced",
                "pairing_status": "independent_fallback",
                "pairing_reason": "missing_pair_id:both_groups",
                "pairing_warning": (
                    "Парный дизайн не подтверждён; выполнен independent analysis."
                ),
            }
        ]
    )

    report = build_markdown_report(
        metadata=metadata,
        prepared=prepared,
        validation_issues=issues,
        failures=failure_summary(prepared),
        group_comparisons=[comparison],
        audit=AuditTrail(),
    )

    assert "## Сравнение групп" in report
    assert "pairing_status=`independent_fallback`" in report
    assert "pairing_reason=`missing_pair_id:both_groups`" in report
    assert "Парный дизайн не подтверждён; выполнен independent analysis." in report


def test_report_and_bundle_include_provenance_source_and_manifest() -> None:
    source = b"test_id,stage,load,settlement\nT1,1,1,0.2\n"
    metadata = {
        "load_kind": "force",
        "load_unit": "kN",
        "load_factor": 1.0,
        "load_zero": 0.0,
        "lever_ratio": 1.0,
        "settlement_unit": "mm",
        "indicator_resolution_mm": 0.01,
        "stamp_area_m2": 0.1,
    }
    raw = pd.DataFrame(
        {"test_id": ["T1"], "stage": [1], "load": [1.0], "settlement": [0.2]}
    )
    prepared, issues = prepare_measurements(raw, metadata)
    config = {"import_mode": "strict"}
    provenance = build_provenance(
        input_source=source,
        metadata_source=metadata,
        config=config,
    )
    report = build_markdown_report(
        metadata=metadata,
        prepared=prepared,
        validation_issues=issues,
        failures=failure_summary(prepared),
        audit=AuditTrail(),
        provenance=provenance,
        passport_status=passport_completeness(metadata),
        import_info={"format": "csv", "import_mode": "strict", "rows": 1},
    )
    bundle = reproducibility_bundle(
        raw=raw,
        prepared=prepared,
        metadata=metadata,
        audit=AuditTrail(),
        report_markdown=report,
        provenance=provenance,
        source_file_name="protocol.csv",
        source_file_bytes=source,
        metadata_file_name="metadata_original.json",
        metadata_file_bytes=b'{"load_unit":"kN"}',
        config_snapshot=config,
        result_tables={"../../escape": {"ok": True}},
        figures={"../bad.svg": b"<svg/>"},
        additional_root_files={
            "report.html": b"<html></html>",
            "report.xlsx": b"PK-report",
        },
        additional_files={
            "approval/report.html": b"<html>approval</html>",
            "approval/source/protocol.csv": source,
        },
    )

    assert provenance.input_file_sha256 in report
    assert "Паспорт неполон" in report
    assert "## Индикаторные каналы" in report
    assert "SHA-256 дерева исходников" in report
    with zipfile.ZipFile(BytesIO(bundle)) as archive:
        names = set(archive.namelist())
        manifest = json.loads(archive.read("manifest.json"))
        assert "source/protocol.csv" in names
        assert "provenance.json" in names
        assert "source/metadata_original.json" in names
        assert "config/processing_config.canonical.json" in names
        assert "results/indicator_processing_audit.csv" in names
        assert "results/indicator_processing_events.csv" in names
        assert "results/indicator_calibration_parameters.csv" in names
        assert "results/indicator_aggregation_results.csv" in names
        assert all(".." not in name for name in names)
        assert "results/escape.json" in names
        assert "figures/bad.svg" in names
        assert archive.read("report.html") == b"<html></html>"
        assert archive.read("report.xlsx") == b"PK-report"
        assert archive.read("approval/report.html") == b"<html>approval</html>"
        assert archive.read("approval/source/protocol.csv") == source
        assert archive.read("source/protocol.csv") == source
        import hashlib

        assert (
            hashlib.sha256(archive.read("config/processing_config.canonical.json")).hexdigest()
            == provenance.config_sha256
        )
        assert any(item["path"] == "source/protocol.csv" for item in manifest["files"])


def test_reproducibility_bundle_rejects_unsafe_or_colliding_additional_paths() -> None:
    kwargs = {
        "raw": pd.DataFrame(),
        "prepared": pd.DataFrame(),
        "metadata": {},
        "audit": AuditTrail(),
        "report_markdown": "# Report",
    }
    with pytest.raises(ValueError, match="Unsafe reproducibility bundle path"):
        reproducibility_bundle(**kwargs, additional_files={"../escape": b"x"})
    with pytest.raises(ValueError, match="file/directory path collision"):
        reproducibility_bundle(
            **kwargs,
            additional_files={"approval": b"file", "approval/report.html": b"html"},
        )


def test_report_summarizes_indicator_passport_crossings_and_qc() -> None:
    metadata = {
        "stamp_area_m2": 0.1,
        "load_resolution_kN": 0.01,
        "indicator_resolution_mm": 0.01,
    }
    raw = pd.DataFrame(
        {
            "test_id": ["T1", "T1"],
            "stage": [1, 2],
            "load": [1.0, 2.0],
            "settlement": [0.0, 0.2],
        }
    )
    prepared, issues = prepare_measurements(raw, metadata)
    prepared.attrs["indicator_calibration_parameters"] = [
        {
            "test_id": "T1",
            "channel": "indicator_1",
            "indicator_type": "ИЧ-10",
            "serial_number": "SN-42",
            "instrument_id": "SN-42",
            "mode": "increasing_wrapped",
            "range_mm": 10.0,
            "division_mm": 0.01,
            "correction_factor": 1.0,
            "verification_date": "2026-01-01",
            "verification_valid_until": "2027-01-01",
            "verification_status": "valid",
            "verification_evaluation_date": "2026-06-01",
            "verification_evaluation_date_source": "experiment_date",
            "verification_evaluation_rule": (
                "verification_date <= experiment_date <= verification_valid_until"
            ),
            "x_mm": 0.0,
            "y_mm": 0.0,
            "assignment_status": "confirmed",
            "compatibility_mode": False,
        }
    ]
    prepared.attrs["indicator_processing_audit"] = [
        {
            "test_id": "T1",
            "channel": "indicator_1",
            "processing_status": "ok",
            "quality_flags": "",
        },
        {
            "test_id": "T1",
            "channel": "indicator_1",
            "processing_status": "warning",
            "quality_flags": "zero_crossing;small_reverse_motion",
        },
    ]
    prepared.attrs["indicator_processing_events"] = [
        {"test_id": "T1", "channel": "indicator_1", "event_type": "zero_crossing"},
        {
            "test_id": "T1",
            "channel": "indicator_1",
            "event_type": "small_reverse_motion",
        },
    ]
    prepared.attrs["indicator_aggregation_results"] = [
        {
            "test_id": "T1",
            "row_index": 0,
            "sequence_index": 0,
            "aggregation_method": "primary_channel",
            "channels_required": '["indicator_1"]',
            "channels_used": '["indicator_1"]',
            "missing_channels": "[]",
            "aggregation_status": "ok",
            "plane_rank": None,
            "plane_residual_rms_mm": None,
            "tilt_magnitude_mm_per_mm": None,
        }
    ]

    report = build_markdown_report(
        metadata=metadata,
        prepared=prepared,
        validation_issues=issues,
        failures=failure_summary(prepared),
        audit=AuditTrail(),
    )

    assert "SN-42" in report
    assert "mode=`increasing_wrapped`" in report
    assert "переходов через ноль — 1" in report
    assert "точек обратного хода — 1" in report
    assert "ok=1, warning=1" in report
    assert "## Агрегация осадки" in report
    assert "primary_channel" in report
    assert "статус=`valid`" in report
    assert "дата оценки=2026-06-01 (experiment_date)" in report


def test_report_marks_primary_modulus_and_shows_method_provenance() -> None:
    metadata = {"stamp_area_m2": 0.1, "indicator_resolution_mm": 0.01}
    raw = pd.DataFrame(
        {
            "test_id": ["T1", "T1", "T1"],
            "stage": [1, 2, 3],
            "load": [1.0, 2.0, 3.0],
            "settlement": [0.0, 0.2, 0.5],
        }
    )
    prepared, issues = prepare_measurements(raw, metadata)
    moduli = pd.DataFrame(
        [
            {
                "test_id": "T1",
                "method": "E_regression",
                "E_stamp_app_kPa": 12543.0,
                "p_min_kPa": 10.0,
                "p_max_kPa": 80.0,
                "n": 3,
                "nu": 0.3,
                "shape_factor": 0.8,
                "ci_low_kPa": 12000.0,
                "ci_high_kPa": 13000.0,
                "profile_id": "antonov_round_stamp_v1",
                "profile_version": "1",
                "profile_source": "metadata.tests.T1",
                "is_primary": True,
                "review_status": "approved",
                "p_range_source": "explicit",
                "p_range_origin": "manual_confirmation",
                "requested_p_min_kPa": 10.0,
                "requested_p_max_kPa": 80.0,
                "nu_source": "method_profile",
                "shape_factor_source": "method_profile",
                "used_indices": "[0, 1, 2]",
                "methodology_note": "Диапазон подтверждён инженером.",
            }
        ]
    )

    report = build_markdown_report(
        metadata=metadata,
        prepared=prepared,
        validation_issues=issues,
        failures=failure_summary(prepared),
        moduli=moduli,
        audit=AuditTrail(),
    )

    assert "`T1` / `E_regression` / `antonov_round_stamp_v1@1`" in report
    assert "**PRIMARY**; review_status=`approved`" in report
    assert "12,5 МПа; 95% ДИ 12,0–13,0 МПа" in report
    assert "источник диапазона=`explicit` (origin=`manual_confirmation`)" in report
    assert "ν=0,30 (source=`method_profile`)" in report
    assert "использованные строки=[0, 1, 2]" in report
    assert "запрошенный диапазон: 10,0–80,0 кПа" in report
    assert "Диапазон подтверждён инженером." in report


def test_report_treats_legacy_and_review_required_moduli_as_diagnostic() -> None:
    metadata = {"stamp_area_m2": 0.1, "indicator_resolution_mm": 0.01}
    raw = pd.DataFrame(
        {
            "test_id": ["T1", "T1"],
            "stage": [1, 2],
            "load": [1.0, 2.0],
            "settlement": [0.0, 0.2],
        }
    )
    prepared, issues = prepare_measurements(raw, metadata)
    legacy_moduli = pd.DataFrame(
        [
            {
                "test_id": "T1",
                "method": "E_regression",
                "E_stamp_app_kPa": 12500.0,
                "p_min_kPa": 10.0,
                "p_max_kPa": 80.0,
                "n": 2,
                "nu": 0.3,
                "shape_factor": 1.0,
                "is_primary": True,
                "review_status": "review_required",
            }
        ]
    )

    report = build_markdown_report(
        metadata=metadata,
        prepared=prepared,
        validation_issues=issues,
        failures=failure_summary(prepared),
        moduli=legacy_moduli,
        audit=AuditTrail(),
    )

    modulus_line = next(line for line in report.splitlines() if "E_regression" in line)
    assert "diagnostic_unapproved_v1@legacy" in modulus_line
    assert "**DIAGNOSTIC**; review_status=`review_required`" in modulus_line
    assert "**PRIMARY**" not in modulus_line
