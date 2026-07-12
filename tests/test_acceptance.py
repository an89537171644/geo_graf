from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path
from typing import Any

import pytest
import pandas as pd

from soilstamp.acceptance import (
    ACCEPTANCE_CASE_VERSION,
    AcceptanceCriticalMismatch,
    AcceptanceManifestError,
    load_acceptance_manifest,
    run_acceptance_manifest,
)


def _input_files(root: Path) -> None:
    (root / "protocol.csv").write_text("test_id,load\nT-01,0\n", encoding="utf-8")
    (root / "metadata.json").write_text("{}\n", encoding="utf-8")


def _case(**updates: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "case_id": "synthetic_static",
        "source_type": "csv",
        "input_files": {
            "protocol": "protocol.csv",
            "metadata": "metadata.json",
        },
        "metadata": {"data_class": "synthetic", "cli_args": []},
        "expected_outputs": {
            "required_artifacts": ["stable.txt"],
            "scientific_flags": [
                {
                    "artifact": "failure_analysis.json",
                    "pointer": "/contract_version",
                    "expected": "failure-analysis/1.0",
                }
            ],
            "failure_rows": [
                {
                    "selector": {"test_id": "T-01"},
                    "expected": {
                        "failure_observed": True,
                        "interval_censored": True,
                        "right_censored": False,
                        "censoring_type": "interval-observed",
                        "lower_bound": 100.0,
                        "upper_bound": 120.0,
                    },
                }
            ],
            "modulus_rows": [
                {
                    "selector": {"test_id": "T-01", "method": "E_regression"},
                    "expected": {
                        "E_stamp_app_kPa": 1234.5,
                        "profile_id": "diagnostic_unapproved_v1",
                        "p_range_source": "diagnostic_full_curve",
                        "review_status": "review_required",
                    },
                    "tolerance": "E",
                }
            ],
            # Empty selector exercises the one-row CSV contract.
            "golden_values": [
                {
                    "artifact": "single.csv",
                    "selector": {},
                    "field": "value",
                    "expected": 2.5,
                    "tolerance": "default",
                },
                {
                    "artifact": "approval_report.zip!nested/value.csv",
                    "selector": {},
                    "field": "value",
                    "expected": 7,
                },
            ],
            "hashes": {
                "stable.txt": hashlib.sha256(b"stable\n").hexdigest(),
            },
            "report_package": True,
        },
        "tolerances": {
            "default": {"absolute": 1e-12, "relative": 0.0},
            "E": {"absolute": 0.1, "relative": 1e-9},
        },
        "expected_warnings": [],
        "expected_review_status": "review_required",
        "independent_calculation_reference": "independent/calc-T-01.md#result",
        "reviewer": None,
        "signoff_status": "unsigned",
    }
    payload.update(updates)
    return payload


def _manifest(root: Path, case: dict[str, Any] | None = None) -> Path:
    path = root / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "contract_version": ACCEPTANCE_CASE_VERSION,
                "unsigned_engineering_gates": [
                    "three_real_tests_engineer_signoff",
                    "methodology_source_engineer_review",
                ],
                "cases": [case or _case()],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return path


def _zip_write(archive: zipfile.ZipFile, name: str, payload: bytes) -> None:
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    archive.writestr(info, payload)


def _production_artifacts(output: Path, *, unsafe_archive: bool = False) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "failure_analysis.json").write_text(
        json.dumps(
            {
                "contract_version": "failure-analysis/1.0",
                "summary_method": "none",
                "point_estimate": None,
            }
        ),
        encoding="utf-8",
    )
    (output / "failure_summary.csv").write_text(
        "test_id,failure_reached,failure_observed,interval_censored,right_censored,"
        "censoring_type,classification_status,lower_bound,upper_bound\n"
        "T-01,True,True,True,False,interval-observed,classified,100,120\n",
        encoding="utf-8",
    )
    (output / "moduli.csv").write_text(
        "test_id,method,E_stamp_app_kPa,profile_id,profile_version,p_range_source,"
        "review_status,used_indices\n"
        "T-01,E_regression,1234.51,diagnostic_unapproved_v1,1.0,"
        'diagnostic_full_curve,review_required,"[0, 1, 2]"\n',
        encoding="utf-8",
    )
    (output / "validation_issues.csv").write_text("severity,code,message\n", encoding="utf-8")
    (output / "analysis_warnings.json").write_text("[]\n", encoding="utf-8")
    (output / "stable.txt").write_bytes(b"stable\n")
    (output / "single.csv").write_text("value\n2.5\n", encoding="utf-8")

    report_html = b"<!doctype html><title>Approval</title>\n"
    report_xlsx = b"deterministic-test-workbook\n"
    nested = b"value\n7\n"
    artifacts = {
        "report.html": report_html,
        "report.xlsx": report_xlsx,
        "nested/value.csv": nested,
    }
    records = [
        {
            "path": name,
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
        for name, payload in sorted(artifacts.items())
    ]
    manifest = {
        "schema_version": "approval-report-package/1.0",
        "files": records,
    }
    manifest_bytes = (json.dumps(manifest, sort_keys=True, indent=2) + "\n").encode()
    (output / "report.html").write_bytes(report_html)
    (output / "report.xlsx").write_bytes(report_xlsx)
    (output / "artifact_manifest.json").write_bytes(manifest_bytes)
    with zipfile.ZipFile(output / "approval_report.zip", "w") as archive:
        for name, payload in sorted(artifacts.items()):
            _zip_write(archive, name, payload)
        _zip_write(archive, "artifact_manifest.json", manifest_bytes)
        if unsafe_archive:
            _zip_write(archive, "../escape.txt", b"unsafe")


def _runner(case: dict[str, Any], inputs: dict[str, Path], output: Path) -> None:
    assert inputs["protocol"].is_file()
    assert case["case_id"]
    _production_artifacts(output)


def test_acceptance_run_passes_synthetic_case_and_writes_deterministic_reports(
    tmp_path: Path,
) -> None:
    _input_files(tmp_path)
    manifest = _manifest(tmp_path)
    out = tmp_path / "results"

    first = run_acceptance_manifest(manifest, out, production_runner=_runner)
    first_payloads = {
        path.name: path.read_bytes()
        for path in (first.json_report, first.markdown_report, first.html_report)
    }
    second = run_acceptance_manifest(manifest, out, production_runner=_runner)

    assert first.passed
    assert first.exit_code == 0
    assert first.synthetic_acceptance_passed
    assert not first.engineering_acceptance
    assert second.passed
    assert first_payloads == {
        path.name: path.read_bytes()
        for path in (second.json_report, second.markdown_report, second.html_report)
    }
    payload = json.loads(second.json_report.read_text(encoding="utf-8"))
    assert payload["candidate_status"] == "candidate_for_engineering_acceptance"
    assert payload["synthetic_acceptance_passed"] is True
    assert payload["engineering_acceptance"] is False
    assert payload["technical_status"] == "pass"
    assert "three_real_tests_engineer_signoff" in {
        gate["gate_id"] for gate in payload["unsigned_engineering_gates"]
    }
    assert "not an engineering approval" in second.html_report.read_text(encoding="utf-8")


def test_repeated_run_cannot_reuse_stale_artifacts(tmp_path: Path) -> None:
    _input_files(tmp_path)
    manifest = _manifest(tmp_path)
    out = tmp_path / "results"

    def first_runner(case: dict[str, Any], inputs: dict[str, Path], output: Path) -> None:
        _production_artifacts(output)
        (output / "stale.txt").write_text("must disappear", encoding="utf-8")

    run_acceptance_manifest(manifest, out, production_runner=first_runner)

    def second_runner(case: dict[str, Any], inputs: dict[str, Path], output: Path) -> None:
        assert not (output / "stale.txt").exists()
        _production_artifacts(output)

    result = run_acceptance_manifest(manifest, out, production_runner=second_runner)
    assert result.passed


def test_input_hash_mismatch_blocks_production_execution(tmp_path: Path) -> None:
    _input_files(tmp_path)
    case = _case(
        metadata={
            "data_class": "synthetic",
            "cli_args": [],
            "input_sha256": {"protocol": "0" * 64},
        }
    )
    called = False

    def forbidden_runner(case: dict[str, Any], inputs: dict[str, Path], output: Path) -> None:
        nonlocal called
        called = True

    result = run_acceptance_manifest(
        _manifest(tmp_path, case),
        tmp_path / "results",
        production_runner=forbidden_runner,
    )

    assert result.exit_code == 1
    assert not called
    production = [
        check for check in result.cases[0].checks if check.category == "production_pipeline"
    ]
    assert len(production) == 1
    assert production[0].actual == "not_executed"


def test_critical_golden_mismatch_sets_nonzero_and_can_raise(tmp_path: Path) -> None:
    _input_files(tmp_path)
    case = _case()
    case["expected_outputs"]["modulus_rows"][0]["expected"]["E_stamp_app_kPa"] = 9000
    result = run_acceptance_manifest(
        _manifest(tmp_path, case), tmp_path / "results", production_runner=_runner
    )

    assert not result.passed
    assert result.exit_code == 1
    assert result.critical_failure_count == 1
    with pytest.raises(AcceptanceCriticalMismatch):
        result.raise_for_failure()
    assert json.loads(result.json_report.read_text(encoding="utf-8"))["critical_failure_count"] == 1
    markdown = result.markdown_report.read_text(encoding="utf-8")
    rendered_html = result.html_report.read_text(encoding="utf-8")
    assert "| Expected | Actual |" in markdown
    assert "9000" in markdown and "1234.51" in markdown
    assert "<th>Expected</th><th>Actual</th>" in rendered_html
    assert "9000" in rendered_html and "1234.51" in rendered_html


def test_real_unsigned_case_is_never_accepted(tmp_path: Path) -> None:
    _input_files(tmp_path)
    case = _case(
        case_id="real_unsigned",
        metadata={"data_class": "real", "cli_args": []},
        reviewer=None,
        signoff_status="unsigned",
    )
    result = run_acceptance_manifest(
        _manifest(tmp_path, case), tmp_path / "results", production_runner=_runner
    )

    assert result.exit_code == 1
    assert not result.engineering_acceptance
    assert "real_unsigned_engineer_signoff" in result.unsigned_engineering_gate_ids
    failed = [
        check
        for check in result.cases[0].checks
        if check.category == "engineering_signoff" and not check.passed
    ]
    assert len(failed) == 1


def test_structured_unsigned_gate_is_preserved_and_safely_rendered(tmp_path: Path) -> None:
    _input_files(tmp_path)
    manifest = _manifest(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["unsigned_engineering_gates"] = [
        {
            "gate_id": "real_tests_engineer_signoff",
            "status": "unsigned",
            "required_evidence": "Three <real> tests signed by an engineer.",
        }
    ]
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    result = run_acceptance_manifest(manifest, tmp_path / "results", production_runner=_runner)

    assert result.passed
    assert result.unsigned_engineering_gate_ids == ("real_tests_engineer_signoff",)
    assert result.unsigned_engineering_gates[0]["required_evidence"].startswith("Three")
    rendered = result.html_report.read_text(encoding="utf-8")
    assert "<real>" not in rendered
    assert "&lt;real&gt;" in rendered


@pytest.mark.parametrize(
    "unsafe",
    ["../outside.csv", "/absolute.csv", "C:/windows.csv", "nested\\windows.csv"],
)
def test_manifest_rejects_unsafe_input_paths(tmp_path: Path, unsafe: str) -> None:
    _input_files(tmp_path)
    case = _case(input_files={"protocol": unsafe, "metadata": "metadata.json"})
    manifest = _manifest(tmp_path, case)

    with pytest.raises(AcceptanceManifestError):
        load_acceptance_manifest(manifest)


def test_manifest_requires_complete_contract_and_portable_unique_case_ids(
    tmp_path: Path,
) -> None:
    _input_files(tmp_path)
    case = _case()
    del case["reviewer"]
    with pytest.raises(AcceptanceManifestError, match="reviewer"):
        load_acceptance_manifest(_manifest(tmp_path, case))

    duplicate = tmp_path / "duplicate.json"
    first = _case(case_id="Case")
    second = _case(case_id="case")
    duplicate.write_text(
        json.dumps({"contract_version": ACCEPTANCE_CASE_VERSION, "cases": [first, second]}),
        encoding="utf-8",
    )
    with pytest.raises(AcceptanceManifestError, match="Duplicate portable"):
        load_acceptance_manifest(duplicate)


@pytest.mark.parametrize(
    ("metadata", "reviewer", "message"),
    [
        ({"data_class": "synthetic"}, "engineer", "synthetic/anonymized"),
        ({"data_class": "real"}, None, "without an identified reviewer"),
        (
            {"data_class": "real"},
            {"date": "2026-07-12"},
            "without an identified reviewer",
        ),
    ],
)
def test_manifest_rejects_fake_signed_acceptance(
    tmp_path: Path,
    metadata: dict[str, str],
    reviewer: str | dict[str, str] | None,
    message: str,
) -> None:
    _input_files(tmp_path)
    case = _case(
        metadata={**metadata, "cli_args": []},
        reviewer=reviewer,
        signoff_status="approved",
    )
    with pytest.raises(AcceptanceManifestError, match=message):
        load_acceptance_manifest(_manifest(tmp_path, case))


def test_report_package_rejects_unsafe_archive_member(tmp_path: Path) -> None:
    _input_files(tmp_path)

    def unsafe_runner(case: dict[str, Any], inputs: dict[str, Path], output: Path) -> None:
        _production_artifacts(output, unsafe_archive=True)

    result = run_acceptance_manifest(
        _manifest(tmp_path), tmp_path / "results", production_runner=unsafe_runner
    )

    assert result.exit_code == 1
    report_checks = [
        check for check in result.cases[0].checks if check.category == "report_package"
    ]
    assert any(
        not check.passed and "unsafe archive member" in check.message for check in report_checks
    )


def test_case_without_e_expectations_does_not_require_moduli_artifact(
    tmp_path: Path,
) -> None:
    _input_files(tmp_path)
    case = _case(expected_review_status=[])
    case["expected_outputs"]["modulus_rows"] = []

    def no_modulus_runner(case: dict[str, Any], inputs: dict[str, Path], output: Path) -> None:
        _production_artifacts(output)
        (output / "moduli.csv").unlink()

    result = run_acceptance_manifest(
        _manifest(tmp_path, case), tmp_path / "results", production_runner=no_modulus_runner
    )

    assert result.passed
    assert not (tmp_path / "results" / "cases" / "synthetic_static" / "moduli.csv").exists()


def test_production_failure_is_escaped_and_recorded_in_all_reports(tmp_path: Path) -> None:
    _input_files(tmp_path)

    def failing_runner(case: dict[str, Any], inputs: dict[str, Path], output: Path) -> None:
        raise RuntimeError("<script>alert('no')</script>")

    result = run_acceptance_manifest(
        _manifest(tmp_path), tmp_path / "results", production_runner=failing_runner
    )

    assert result.exit_code == 1
    assert "<script>" in result.markdown_report.read_text(encoding="utf-8")
    rendered_html = result.html_report.read_text(encoding="utf-8")
    assert "<script>alert" not in rendered_html
    assert "&lt;script&gt;" in rendered_html


def test_default_csv_execution_uses_cli_production_runner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _input_files(tmp_path)
    observed: dict[str, Any] = {}

    def fake_run(namespace: Any) -> Path:
        observed["protocol"] = namespace.protocol
        observed["metadata"] = namespace.metadata
        observed["out"] = namespace.out
        _production_artifacts(namespace.out)
        return namespace.out / "reproducibility.zip"

    monkeypatch.setattr("soilstamp.cli.run", fake_run)
    result = run_acceptance_manifest(_manifest(tmp_path), tmp_path / "results")

    assert result.passed
    assert observed["protocol"] == tmp_path / "protocol.csv"
    assert observed["metadata"] == tmp_path / "metadata.json"
    assert observed["out"] == tmp_path / "results" / "cases" / "synthetic_static"


def test_manual_equivalence_materializes_real_xlsx_and_uses_production_importer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from soilstamp.io import read_protocol
    from soilstamp.manual_entry_models import ManualDraft

    _input_files(tmp_path)
    (tmp_path / "manual_draft.json").write_text("{}\n", encoding="utf-8")
    (tmp_path / "excel_projection.json").write_text(
        json.dumps(
            {
                "contract_version": "declarative-xlsx-projection/1.0",
                "status": "declarative_only",
                "binary_xlsx_claimed": False,
                "sheet_name": "Protocol",
                "header_row": 1,
                "headers": [
                    "test_id",
                    "stage",
                    "load, kN",
                    "indicator_1",
                    "branch",
                    "status",
                    "comment",
                    "group",
                ],
                "rows": [["T-01", 0, 0.0, 0.0, "loading", "stable", "", "baseline"]],
            }
        ),
        encoding="utf-8",
    )
    manual_prepared = pd.DataFrame(
        {
            "test_id": ["T-01"],
            "sequence_no": [0],
            "F_kN": [0.0],
            "p_kPa": [0.0],
            "settlement_mm": [0.0],
            "stamp_area_m2": [0.07068583470577035],
            "branch": ["loading"],
            "status": ["stable"],
        }
    )

    class FakeBundle:
        metadata_bytes = b"{}\n"

        def prepare(self) -> tuple[pd.DataFrame, list[Any]]:
            return manual_prepared.copy(), []

    sentinel = object()
    monkeypatch.setattr(
        ManualDraft,
        "from_json",
        classmethod(lambda cls, payload: sentinel),
    )
    monkeypatch.setattr(
        "soilstamp.manual_entry_adapter.adapt_manual_draft",
        lambda draft: FakeBundle(),
    )
    observed: dict[str, Any] = {}

    def fake_cli_run(namespace: Any) -> Path:
        observed["suffix"] = namespace.protocol.suffix
        observed["is_zip"] = zipfile.is_zipfile(namespace.protocol)
        imported = read_protocol(
            namespace.protocol.read_bytes(),
            filename=namespace.protocol.name,
            import_mode="strict",
        )
        observed["import_format"] = imported.info["format"]
        observed["rows"] = len(imported.frame)
        _production_artifacts(namespace.out)
        serialized = manual_prepared.copy()
        serialized["stamp_area_m2"] = 0.0706858347057703
        serialized.to_csv(namespace.out / "prepared.csv", index=False)
        return namespace.out / "reproducibility.zip"

    monkeypatch.setattr("soilstamp.cli.run", fake_cli_run)
    case = _case(
        case_id="excel_manual",
        source_type="equivalence",
        input_files={
            "manual_draft": "manual_draft.json",
            "excel_projection": "excel_projection.json",
            "metadata": "metadata.json",
        },
        metadata={
            "data_class": "synthetic",
            "cli_args": [],
            "equivalence_columns": [
                "test_id",
                "sequence_no",
                "F_kN",
                "p_kPa",
                "settlement_mm",
                "stamp_area_m2",
                "branch",
                "status",
            ],
            "equivalence_tolerance": "strict",
        },
    )
    result = run_acceptance_manifest(_manifest(tmp_path, case), tmp_path / "results")

    assert result.passed
    assert observed == {
        "suffix": ".xlsx",
        "is_zip": True,
        "import_format": "xlsx",
        "rows": 1,
    }
    materialized = (
        tmp_path
        / "results"
        / "cases"
        / "excel_manual"
        / "materialized"
        / "equivalent_excel_manual.xlsx"
    )
    assert materialized.is_file()
    parity_checks = [
        check for check in result.cases[0].checks if check.category == "input_equivalence"
    ]
    assert len(parity_checks) == 1
    assert parity_checks[0].passed
    assert "1 rows" in parity_checks[0].message
    assert parity_checks[0].actual["real_xlsx_import"] is True
    assert parity_checks[0].actual["import_mode"] == "strict"
    assert parity_checks[0].actual["rows_compared"] == 1
    assert parity_checks[0].actual["fields_compared"] == 8


def test_paired_case_executes_public_compare_groups_and_checks_pairing_golden(
    tmp_path: Path,
) -> None:
    _input_files(tmp_path)
    prepared = pd.DataFrame(
        [
            ("B1", "baseline", "P1", 0.0, 0.2),
            ("B1", "baseline", "P1", 100.0, 2.0),
            ("R1", "reinforced", "P1", 0.0, 0.1),
            ("R1", "reinforced", "P1", 100.0, 1.2),
            ("B2", "baseline", "P2", 0.0, 0.3),
            ("B2", "baseline", "P2", 100.0, 2.4),
            ("R2", "reinforced", "P2", 0.0, 0.2),
            ("R2", "reinforced", "P2", 100.0, 1.4),
        ],
        columns=["test_id", "group", "pair_id", "p_kPa", "settlement_mm"],
    )
    prepared["branch"] = "loading"
    prepared["status"] = "stable"
    prepared["sequence_no"] = prepared.groupby("test_id").cumcount()

    def paired_runner(case: dict[str, Any], inputs: dict[str, Path], output: Path) -> None:
        _production_artifacts(output)
        prepared.to_csv(output / "prepared.csv", index=False)

    case = _case(case_id="paired")
    case["expected_outputs"]["group_comparison"] = {
        "baseline_group": "baseline",
        "reinforced_group": "reinforced",
        "bootstrap": 20,
        "seed": 9,
        "artifact": "group_comparison.csv",
    }
    case["expected_outputs"]["golden_values"].extend(
        [
            {
                "artifact": "group_comparison.csv",
                "selector": {"p_kPa": "100.0"},
                "field": "analysis_design",
                "expected": "paired",
            },
            {
                "artifact": "group_comparison.csv",
                "selector": {"p_kPa": "100.0"},
                "field": "pairing_status",
                "expected": "paired_validated",
            },
            {
                "artifact": "group_comparison.csv",
                "selector": {"p_kPa": "100.0"},
                "field": "n_pairs",
                "expected": 2,
            },
            {
                "artifact": "group_comparison.csv",
                "selector": {"p_kPa": "100.0"},
                "field": "pair_ids_used",
                "expected": "P1,P2",
            },
        ]
    )
    result = run_acceptance_manifest(
        _manifest(tmp_path, case), tmp_path / "results", production_runner=paired_runner
    )

    assert result.passed
    comparison = pd.read_csv(tmp_path / "results" / "cases" / "paired" / "group_comparison.csv")
    assert set(comparison["analysis_design"]) == {"paired"}
    assert set(comparison["pairing_status"]) == {"paired_validated"}
    assert set(comparison["n_pairs"]) == {2}
