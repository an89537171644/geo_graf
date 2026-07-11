from __future__ import annotations

import json
import zipfile
from io import BytesIO

import numpy as np
import pandas as pd

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
        assert all(".." not in name for name in names)
        assert "results/escape.json" in names
        assert "figures/bad.svg" in names
        assert archive.read("source/protocol.csv") == source
        import hashlib

        assert (
            hashlib.sha256(archive.read("config/processing_config.canonical.json")).hexdigest()
            == provenance.config_sha256
        )
        assert any(item["path"] == "source/protocol.csv" for item in manifest["files"])


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
