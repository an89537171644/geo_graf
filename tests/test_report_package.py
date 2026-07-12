from __future__ import annotations

import hashlib
import json
import zipfile
from io import BytesIO

import pandas as pd
import pytest

from soilstamp.report_package import (
    build_approval_report_package,
    build_formula_and_range_records,
    build_review_required_registry,
    collect_approval_artifacts,
)


def _package():
    raw = pd.DataFrame(
        {"test_id": ["T-01"], "raw_load": ["001,20"], "comment": ["=unsafe"]}
    )
    prepared = pd.DataFrame(
        {"test_id": ["T-01"], "p_kPa": [12.3456789012345], "settlement_mm": [0.2]}
    )
    return build_approval_report_package(
        artifacts={
            "source/protocol/protocol.csv": b"test_id,load\nT-01,001,20\n",
            "figures/antonov.svg": b"<svg/>",
        },
        metadata={"project": "Approval test"},
        raw=raw,
        prepared=prepared,
        failures=pd.DataFrame(),
        formulas={"pressure": "p = F / A"},
        review_required=["Engineering review required"],
    )


def test_approval_report_package_is_deterministic_and_manifested() -> None:
    first = _package()
    second = _package()

    assert first.html == second.html
    assert first.xlsx == second.xlsx
    assert first.archive == second.archive
    manifest = json.loads(first.artifact_manifest_json)
    assert manifest["schema_version"] == "approval-report-package/1.0"

    with zipfile.ZipFile(BytesIO(first.archive)) as archive:
        assert archive.namelist() == sorted(archive.namelist())
        assert archive.read("source/protocol/protocol.csv") == (
            b"test_id,load\nT-01,001,20\n"
        )
        for record in manifest["files"]:
            payload = archive.read(record["path"])
            assert len(payload) == record["bytes"]
            assert hashlib.sha256(payload).hexdigest() == record["sha256"]


@pytest.mark.parametrize(
    "path",
    ["../escape.csv", "/absolute.csv", "bad\\path.csv", "C:/drive.csv", "bad:name.csv"],
)
def test_approval_report_package_rejects_unsafe_artifact_paths(path: str) -> None:
    with pytest.raises(ValueError, match="Unsafe report artifact path"):
        build_approval_report_package(
            artifacts={path: b"x"},
            metadata={},
            raw=pd.DataFrame(),
            prepared=pd.DataFrame(),
        )


@pytest.mark.parametrize(
    "path", ["report.html", "report.xlsx", "artifact_manifest.json"]
)
def test_approval_report_package_rejects_reserved_artifact_paths(path: str) -> None:
    with pytest.raises(ValueError, match="Reserved report artifact path"):
        build_approval_report_package(
            artifacts={path: b"x", "source/protocol.csv": b"raw"},
            metadata={},
            raw=pd.DataFrame(),
            prepared=pd.DataFrame(),
        )


def test_approval_report_package_requires_exact_source_and_portable_unique_paths() -> None:
    with pytest.raises(ValueError, match="exact source artifact"):
        build_approval_report_package(
            artifacts={"results/result.csv": b"x"},
            metadata={},
            raw=pd.DataFrame(),
            prepared=pd.DataFrame(),
        )
    with pytest.raises(ValueError, match="Duplicate portable"):
        build_approval_report_package(
            artifacts={
                "source/protocol/A.csv": b"a",
                "source/protocol/a.csv": b"b",
            },
            metadata={},
            raw=pd.DataFrame(),
            prepared=pd.DataFrame(),
        )


@pytest.mark.parametrize(
    "artifacts",
    [
        {"source": b"file", "source/protocol/input.csv": b"child"},
        {"source/protocol": b"file", "source/protocol/input.csv": b"child"},
        {"Source/Protocol": b"file", "source/protocol/input.csv": b"child"},
        {"report.html/assets/style.css": b"collision"},
    ],
)
def test_approval_report_package_rejects_file_directory_collisions(
    artifacts: dict[str, bytes],
) -> None:
    with pytest.raises(ValueError, match="file/directory collision"):
        build_approval_report_package(
            artifacts=artifacts,
            metadata={},
            raw=pd.DataFrame(),
            prepared=pd.DataFrame(),
        )


@pytest.mark.parametrize(
    "artifacts",
    [
        {"source": b"not-a-descendant"},
        {"source/metadata/project.json": b"{}"},
        {"source/protocol.csv": b"ambiguous-root-file"},
    ],
)
def test_approval_report_package_requires_protocol_source_descendant(
    artifacts: dict[str, bytes],
) -> None:
    with pytest.raises(ValueError, match=r"under source/protocol/"):
        build_approval_report_package(
            artifacts=artifacts,
            metadata={},
            raw=pd.DataFrame(),
            prepared=pd.DataFrame(),
        )


def test_collect_artifacts_preserves_exact_source_and_machine_values() -> None:
    source = b"test_id;load\r\nT-1;001,20\r\n"
    artifacts = collect_approval_artifacts(
        raw=pd.DataFrame({"test_id": ["T-1"], "load": ["001,20"]}),
        prepared=pd.DataFrame({"p_kPa": [1.2345678901234567]}),
        source_file_name="protocol.csv",
        source_file_bytes=source,
        result_tables={"result": {"value": 1.2345678901234567}},
    )

    assert artifacts["source/protocol/protocol.csv"] == source
    assert b"1.2345678901234567" in artifacts["data/prepared_machine.csv"]
    assert b"1.2345678901234567" in artifacts["results/result.json"]


def test_review_registry_and_formula_ranges_are_explicit() -> None:
    registry = build_review_required_registry(
        passport_status={"missing": ["operator"]},
        qc_issues=[{"level": "warning", "message": "Check input", "test_id": "T-1"}],
        indicator_passports=[
            {
                "test_id": "T-1",
                "channel": "indicator_1",
                "assignment_status": "review_required",
                "verification_status": "expired",
            }
        ],
        failures=[
            {
                "test_id": "T-1",
                "classification_status": "review_required",
                "classification_warning": "invalid bounds",
            }
        ],
        moduli=[
            {
                "test_id": "T-1",
                "review_status": "review_required",
                "methodology_note": "Select range",
            }
        ],
    )
    categories = {record["category"] for record in registry}
    assert {
        "project_passport",
        "qc_issue",
        "indicator_assignment",
        "indicator_verification",
        "failure_classification",
        "modulus_methodology",
    }.issubset(categories)

    formulas = build_formula_and_range_records(
        conversion_parameters=pd.DataFrame(
            [{"test_id": "T-1", "formula": "p = F / A"}]
        ),
        moduli=pd.DataFrame(
            [
                {
                    "test_id": "T-1",
                    "method": "E_regression",
                    "p_min_kPa": 25.0,
                    "p_max_kPa": 75.0,
                }
            ]
        ),
    )
    assert any(record["expression"] == "p = F / A" for record in formulas)
    assert any(
        record.get("range") == "25.0 <= p_kPa <= 75.0" for record in formulas
    )


def test_indicator_formula_records_match_direct_inverse_wrapped_and_ready_modes() -> None:
    passports = [
        {
            "test_id": "T-direct",
            "channel": "indicator_1",
            "mode": "direct_scale",
            "range_mm": 10.0,
            "division_mm": 0.01,
            "correction_factor": 1.002,
            "initial_reading": 2.0,
            "initial_turn": 0,
            "zero_correction_mm": 0.1,
        },
        {
            "test_id": "T-inverse",
            "channel": "indicator_1",
            "mode": "reverse_scale",
            "range_mm": 10.0,
            "division_mm": 0.01,
            "correction_factor": 0.999,
            "initial_reading": 8.0,
            "initial_turn": 0,
            "zero_correction_mm": -0.02,
        },
        {
            "test_id": "T-wrapped",
            "channel": "indicator_1",
            "mode": "increasing_wrapped",
            "range_mm": 10.0,
            "division_mm": 0.01,
            "correction_factor": 1.0,
            "initial_reading": 9.8,
            "initial_turn": 2,
            "zero_correction_mm": 0.0,
            "max_increment_mm": 1.0,
            "travel_range_mm": 25.0,
        },
        {
            "test_id": "T-ready",
            "channel": "indicator_1",
            "mode": "ready_settlement",
            "division_mm": 0.01,
            "correction_factor": 1.02,
            "zero_correction_mm": 0.1,
            "cumulative_sign": -1.0,
        },
    ]

    records = build_formula_and_range_records(indicator_passports=passports)
    indicator_records = {
        record["scope"]: record
        for record in records
        if str(record["record_id"]).startswith("indicator_")
    }

    direct = indicator_records["T-direct"]
    direct_contract = json.loads(direct["notes"])["formula_contract"]
    assert direct_contract["mode_canonical"] == "increasing"
    assert direct_contract["reading_model"] == "absolute_dial"
    assert direct_contract["direction_multiplier_q"] == 1
    assert "q = +1" in direct["expression"]
    assert "s_i = q * f * (u_i - u_0) + c0" in direct["expression"]
    assert "0 <= raw_i < R (10.0 mm)" in direct["range"]
    assert "no automatic wrap inference" in direct["range"]

    inverse = indicator_records["T-inverse"]
    inverse_contract = json.loads(inverse["notes"])["formula_contract"]
    assert inverse_contract["mode_canonical"] == "decreasing"
    assert inverse_contract["scale_direction"] == "inverse_decreasing"
    assert inverse_contract["direction_multiplier_q"] == -1
    assert "q = -1" in inverse["expression"]

    wrapped = indicator_records["T-wrapped"]
    wrapped_contract = json.loads(wrapped["notes"])["formula_contract"]
    assert wrapped_contract["reading_model"] == "wrapped_dial"
    assert wrapped_contract["initial_turn"] == 2
    assert "u_i = raw_i + turn_i * R" in wrapped["expression"]
    assert "unique admissible inferred integer" in wrapped["range"]
    assert "abs(delta_s_i) <= 1.0 mm" in wrapped["range"]
    assert "abs(s_i - c0) <= 25.0 mm" in wrapped["range"]

    ready = indicator_records["T-ready"]
    ready_contract = json.loads(ready["notes"])["formula_contract"]
    assert ready_contract["mode_canonical"] == "cumulative_settlement"
    assert ready_contract["reading_model"] == "ready_cumulative_settlement"
    assert ready_contract["cumulative_sign"] == -1.0
    assert ready_contract["dial_range_used"] is False
    assert "v_i = sign * raw_i * f" in ready["expression"]
    assert "s_i = v_i + c0" in ready["expression"]
    assert "turn_i = 0 and dial unwrapping is not applied" in ready["range"]
