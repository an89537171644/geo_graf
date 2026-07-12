"""Reproducible engineering-acceptance runner for release candidates.

The acceptance runner deliberately consumes the files written by the normal
production pipeline.  It does not reproduce any scientific calculation.  Its
job is limited to executing that pipeline, comparing its artifacts with an
explicit ``acceptance-case/1.0`` contract, and recording the comparison.
"""

from __future__ import annotations

import csv
import hashlib
import html
import importlib.util
import json
import math
import re
import shutil
import zipfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

import pandas as pd


ACCEPTANCE_CASE_VERSION = "acceptance-case/1.0"
ACCEPTANCE_REPORT_VERSION = "acceptance-report/1.0"
CANDIDATE_STATUS = "candidate_for_engineering_acceptance"

_REQUIRED_CASE_FIELDS = {
    "case_id",
    "source_type",
    "input_files",
    "metadata",
    "expected_outputs",
    "tolerances",
    "expected_warnings",
    "expected_review_status",
    "independent_calculation_reference",
    "reviewer",
    "signoff_status",
}
_CASE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_INPUT_KEY = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,63}$")
_SIGNED_STATUSES = {"signed", "approved", "accepted"}
_REAL_DATA_CLASSES = {"real", "real_test", "real_experiment", "field_real"}
_REVIEWER_IDENTITY_KEYS = {
    "name",
    "full_name",
    "reviewer_id",
    "engineer",
    "engineer_id",
    "email",
}
_MAX_ARCHIVE_MEMBERS = 10_000
_MAX_ARCHIVE_MEMBER_BYTES = 512 * 1024 * 1024

ProductionRunner = Callable[[Mapping[str, Any], Mapping[str, Path], Path], object]


class AcceptanceManifestError(ValueError):
    """Raised before execution when the acceptance contract is malformed."""


class AcceptanceCriticalMismatch(RuntimeError):
    """Raised on request when an acceptance run contains critical failures."""

    def __init__(self, result: "AcceptanceRunResult") -> None:
        self.result = result
        super().__init__(
            f"Acceptance run has {result.critical_failure_count} critical mismatch(es)."
        )


@dataclass(frozen=True, slots=True)
class AcceptanceCheck:
    check_id: str
    category: str
    passed: bool
    message: str
    expected: Any = None
    actual: Any = None
    critical: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "check_id": self.check_id,
            "category": self.category,
            "status": "pass" if self.passed else "fail",
            "critical": self.critical,
            "message": self.message,
            "expected": _json_value(self.expected),
            "actual": _json_value(self.actual),
        }


@dataclass(frozen=True, slots=True)
class AcceptanceCaseResult:
    case_id: str
    source_type: str
    data_class: str
    signoff_status: str
    reviewer: Any
    checks: tuple[AcceptanceCheck, ...]

    @property
    def passed(self) -> bool:
        return not any(check.critical and not check.passed for check in self.checks)

    @property
    def critical_failure_count(self) -> int:
        return sum(check.critical and not check.passed for check in self.checks)

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "source_type": self.source_type,
            "data_class": self.data_class,
            "technical_status": "pass" if self.passed else "fail",
            "signoff_status": self.signoff_status,
            "reviewer": _json_value(self.reviewer),
            "checks": [check.to_dict() for check in self.checks],
        }


@dataclass(frozen=True, slots=True)
class AcceptanceRunResult:
    cases: tuple[AcceptanceCaseResult, ...]
    unsigned_engineering_gates: tuple[dict[str, Any], ...]
    json_report: Path
    markdown_report: Path
    html_report: Path
    contract_version: str = ACCEPTANCE_CASE_VERSION
    report_version: str = ACCEPTANCE_REPORT_VERSION
    candidate_status: str = CANDIDATE_STATUS
    engineering_acceptance: bool = False

    @property
    def passed(self) -> bool:
        return all(case.passed for case in self.cases)

    @property
    def synthetic_acceptance_passed(self) -> bool:
        synthetic = [case for case in self.cases if case.data_class != "real"]
        return bool(synthetic) and all(case.passed for case in synthetic)

    @property
    def critical_failure_count(self) -> int:
        return sum(case.critical_failure_count for case in self.cases)

    @property
    def unsigned_engineering_gate_ids(self) -> tuple[str, ...]:
        return tuple(str(gate["gate_id"]) for gate in self.unsigned_engineering_gates)

    @property
    def exit_code(self) -> int:
        return 0 if self.passed else 1

    def raise_for_failure(self) -> None:
        if not self.passed:
            raise AcceptanceCriticalMismatch(self)

    def to_dict(self) -> dict[str, Any]:
        return {
            "report_version": self.report_version,
            "contract_version": self.contract_version,
            "candidate_status": self.candidate_status,
            "technical_status": "pass" if self.passed else "fail",
            "synthetic_acceptance_passed": self.synthetic_acceptance_passed,
            # A successful synthetic run is never an engineering approval.
            "engineering_acceptance": False,
            "critical_failure_count": self.critical_failure_count,
            "unsigned_engineering_gates": [
                _json_value(gate) for gate in self.unsigned_engineering_gates
            ],
            "cases": [case.to_dict() for case in self.cases],
        }


@dataclass(slots=True)
class _CaseContext:
    case: Mapping[str, Any]
    inputs: Mapping[str, Path]
    output_dir: Path
    checks: list[AcceptanceCheck] = field(default_factory=list)

    @property
    def case_id(self) -> str:
        return str(self.case["case_id"])

    @property
    def expected(self) -> Mapping[str, Any]:
        value = self.case["expected_outputs"]
        return value if isinstance(value, Mapping) else {}

    @property
    def tolerances(self) -> Mapping[str, Any]:
        value = self.case["tolerances"]
        return value if isinstance(value, Mapping) else {}

    def add(
        self,
        check_id: str,
        category: str,
        passed: bool,
        message: str,
        *,
        expected: Any = None,
        actual: Any = None,
        critical: bool = True,
    ) -> None:
        self.checks.append(
            AcceptanceCheck(
                check_id=f"{self.case_id}:{check_id}",
                category=category,
                passed=bool(passed),
                message=message,
                expected=expected,
                actual=actual,
                critical=critical,
            )
        )


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if hasattr(value, "item"):
        try:
            return _json_value(value.item())
        except (TypeError, ValueError):
            pass
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_value(item) for item in value]
    return str(value)


def _safe_relative_path(value: Any, *, field_name: str) -> PurePosixPath:
    if not isinstance(value, str) or not value:
        raise AcceptanceManifestError(f"{field_name} must be a non-empty relative path.")
    if "\x00" in value or "\\" in value:
        raise AcceptanceManifestError(f"{field_name} must use a safe portable POSIX relative path.")
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise AcceptanceManifestError(f"{field_name} escapes its allowed directory.")
    if ":" in path.parts[0]:
        raise AcceptanceManifestError(f"{field_name} must not contain a drive prefix.")
    return path


def _resolve_beneath(root: Path, relative: PurePosixPath, *, field_name: str) -> Path:
    root = root.resolve()
    candidate = root.joinpath(*relative.parts).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise AcceptanceManifestError(f"{field_name} resolves outside {root}.") from exc
    return candidate


def _input_path_value(value: Any, *, field_name: str) -> tuple[str, str | None]:
    if isinstance(value, str):
        return value, None
    if not isinstance(value, Mapping):
        raise AcceptanceManifestError(f"{field_name} must be a path or path/hash object.")
    extra = sorted(set(value) - {"path", "sha256"})
    if extra or not isinstance(value.get("path"), str):
        raise AcceptanceManifestError(f"{field_name} accepts only path and optional sha256 fields.")
    digest = value.get("sha256")
    if digest is not None and (
        not isinstance(digest, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", digest)
    ):
        raise AcceptanceManifestError(f"{field_name}.sha256 must contain 64 hex digits.")
    return str(value["path"]), digest.casefold() if isinstance(digest, str) else None


def _case_data_class(case: Mapping[str, Any]) -> str:
    metadata = case.get("metadata")
    configured = metadata.get("data_class") if isinstance(metadata, Mapping) else None
    value = str(configured or "synthetic").strip().casefold()
    source_type = str(case.get("source_type") or "").strip().casefold()
    if (
        value in _REAL_DATA_CLASSES
        or source_type in _REAL_DATA_CLASSES
        or source_type.startswith("real_")
    ):
        return "real"
    return "synthetic" if value in {"", "synthetic"} else value


def _reviewer_present(value: Any) -> bool:
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, Mapping):
        normalized = {str(key).strip().casefold(): item for key, item in value.items()}
        return any(str(normalized.get(key) or "").strip() for key in _REVIEWER_IDENTITY_KEYS)
    return False


def _validate_case(case: Any, *, index: int) -> None:
    if not isinstance(case, Mapping):
        raise AcceptanceManifestError(f"cases[{index}] must be a JSON object.")
    missing = sorted(_REQUIRED_CASE_FIELDS - set(case))
    if missing:
        raise AcceptanceManifestError(f"cases[{index}] is missing required fields: {missing!r}.")
    case_id = case.get("case_id")
    if not isinstance(case_id, str) or not _CASE_ID.fullmatch(case_id):
        raise AcceptanceManifestError(
            f"cases[{index}].case_id must be a portable identifier (letters, digits, ._-)."
        )
    if not isinstance(case.get("source_type"), str) or not str(case["source_type"]).strip():
        raise AcceptanceManifestError(f"cases[{index}].source_type must be non-empty text.")
    files = case.get("input_files")
    if not isinstance(files, Mapping) or not files:
        raise AcceptanceManifestError(f"cases[{index}].input_files must be a non-empty object.")
    for name, value in files.items():
        if not isinstance(name, str) or not _INPUT_KEY.fullmatch(name):
            raise AcceptanceManifestError(f"cases[{index}].input_files has unsafe key {name!r}.")
        raw_path, _ = _input_path_value(value, field_name=f"cases[{index}].input_files.{name}")
        _safe_relative_path(raw_path, field_name=f"cases[{index}].input_files.{name}")
    for name in ("metadata", "expected_outputs", "tolerances"):
        if not isinstance(case.get(name), Mapping):
            raise AcceptanceManifestError(f"cases[{index}].{name} must be an object.")
    input_hashes = case["metadata"].get("input_sha256", {})
    if not isinstance(input_hashes, Mapping):
        raise AcceptanceManifestError(f"cases[{index}].metadata.input_sha256 must be an object.")
    unknown_hashes = sorted(set(input_hashes) - set(files))
    if unknown_hashes:
        raise AcceptanceManifestError(
            f"cases[{index}].metadata.input_sha256 names unknown inputs: {unknown_hashes!r}."
        )
    if any(
        not isinstance(digest, str) or re.fullmatch(r"[0-9a-fA-F]{64}", digest) is None
        for digest in input_hashes.values()
    ):
        raise AcceptanceManifestError(
            f"cases[{index}].metadata.input_sha256 values must contain 64 hex digits."
        )
    warnings = case.get("expected_warnings")
    if not isinstance(warnings, (list, Mapping)):
        raise AcceptanceManifestError(
            f"cases[{index}].expected_warnings must be a list or matching object."
        )
    reference = case.get("independent_calculation_reference")
    if reference is not None and not isinstance(reference, str):
        raise AcceptanceManifestError(
            f"cases[{index}].independent_calculation_reference must be text or null."
        )
    if not isinstance(case.get("signoff_status"), str):
        raise AcceptanceManifestError(f"cases[{index}].signoff_status must be text.")
    signoff = str(case["signoff_status"]).strip().casefold()
    if signoff in _SIGNED_STATUSES and not _reviewer_present(case.get("reviewer")):
        raise AcceptanceManifestError(
            f"cases[{index}] uses a signed status without an identified reviewer."
        )
    if _case_data_class(case) != "real" and signoff in _SIGNED_STATUSES:
        raise AcceptanceManifestError(
            f"cases[{index}] is synthetic/anonymized and must not claim signed acceptance."
        )


def load_acceptance_manifest(path: Path | str) -> dict[str, Any]:
    """Read and validate an ``acceptance-case/1.0`` manifest without running it."""

    manifest_path = Path(path)
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AcceptanceManifestError(f"Cannot read acceptance manifest: {exc}") from exc
    if not isinstance(payload, dict):
        raise AcceptanceManifestError("Acceptance manifest must be a JSON object.")
    if payload.get("contract_version") != ACCEPTANCE_CASE_VERSION:
        raise AcceptanceManifestError(f"contract_version must be {ACCEPTANCE_CASE_VERSION!r}.")
    cases = payload.get("cases")
    if not isinstance(cases, list) or not cases:
        raise AcceptanceManifestError("Acceptance manifest must contain a non-empty cases list.")
    seen: set[str] = set()
    portable_seen: set[str] = set()
    for index, case in enumerate(cases):
        _validate_case(case, index=index)
        case_id = str(case["case_id"])
        portable = case_id.casefold()
        if case_id in seen or portable in portable_seen:
            raise AcceptanceManifestError(f"Duplicate portable case_id {case_id!r}.")
        seen.add(case_id)
        portable_seen.add(portable)
    gates = payload.get("unsigned_engineering_gates", [])
    if not isinstance(gates, list):
        raise AcceptanceManifestError("unsigned_engineering_gates must be a list.")
    seen_gates: set[str] = set()
    for index, gate in enumerate(gates):
        if isinstance(gate, str):
            gate_id = gate.strip()
        elif isinstance(gate, Mapping):
            gate_id = str(gate.get("gate_id") or "").strip()
            if str(gate.get("status") or "").strip().casefold() != "unsigned":
                raise AcceptanceManifestError(
                    f"unsigned_engineering_gates[{index}].status must be 'unsigned'."
                )
        else:
            gate_id = ""
        if not gate_id or not _CASE_ID.fullmatch(gate_id):
            raise AcceptanceManifestError(
                f"unsigned_engineering_gates[{index}] needs a portable gate_id."
            )
        if gate_id.casefold() in seen_gates:
            raise AcceptanceManifestError(f"Duplicate unsigned gate {gate_id!r}.")
        seen_gates.add(gate_id.casefold())
    return payload


def _resolve_inputs(
    case: Mapping[str, Any], manifest_dir: Path
) -> tuple[dict[str, Path], list[AcceptanceCheck]]:
    result: dict[str, Path] = {}
    checks: list[AcceptanceCheck] = []
    metadata = case.get("metadata")
    declared_hashes = metadata.get("input_sha256", {}) if isinstance(metadata, Mapping) else {}
    for key, value in sorted(case["input_files"].items()):
        raw_path, expected_hash = _input_path_value(
            value, field_name=f"{case['case_id']}.input_files.{key}"
        )
        relative = _safe_relative_path(raw_path, field_name=f"{case['case_id']}.input_files.{key}")
        path = _resolve_beneath(
            manifest_dir, relative, field_name=f"{case['case_id']}.input_files.{key}"
        )
        if not path.is_file():
            raise AcceptanceManifestError(
                f"{case['case_id']}.input_files.{key} does not name a readable file."
            )
        result[str(key)] = path
        if expected_hash is None and isinstance(declared_hashes, Mapping):
            declared = declared_hashes.get(key)
            expected_hash = declared.casefold() if isinstance(declared, str) else None
        if expected_hash is not None:
            actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            checks.append(
                AcceptanceCheck(
                    check_id=f"{case['case_id']}:input_hash:{key}",
                    category="hash",
                    passed=actual_hash == expected_hash,
                    message=f"Exact input hash for {key} matches the manifest.",
                    expected=expected_hash,
                    actual=actual_hash,
                )
            )
    return result, checks


def _cli_arguments(case: Mapping[str, Any], inputs: Mapping[str, Path], out: Path) -> list[str]:
    metadata = case.get("metadata")
    extra = metadata.get("cli_args", []) if isinstance(metadata, Mapping) else []
    if not isinstance(extra, list) or any(not isinstance(item, str) for item in extra):
        raise AcceptanceManifestError(f"{case['case_id']}.metadata.cli_args must be a string list.")
    if any(item == "--out" or item.startswith("--out=") for item in extra):
        raise AcceptanceManifestError(
            "metadata.cli_args must not override the acceptance output directory."
        )
    protocol = inputs.get("protocol")
    metadata_path = inputs.get("metadata")
    if protocol is None or metadata_path is None:
        raise AcceptanceManifestError(
            f"{case['case_id']} requires input_files.protocol and input_files.metadata."
        )
    return [str(protocol), str(metadata_path), "--out", str(out), *extra]


def _normal_source_type(case: Mapping[str, Any], inputs: Mapping[str, Path]) -> bool:
    source_type = str(case["source_type"]).strip().casefold().replace("-", "_")
    if source_type in {"csv", "xlsx", "excel", "protocol", "synthetic", "anonymized"}:
        return True
    if source_type.endswith("_csv") or source_type.endswith("_xlsx"):
        return True
    protocol = inputs.get("protocol")
    return protocol is not None and protocol.suffix.casefold() in {".csv", ".xlsx", ".xlsm"}


def _run_cli_case(case: Mapping[str, Any], inputs: Mapping[str, Path], out: Path) -> None:
    from .cli import build_parser, run

    args = build_parser().parse_args(_cli_arguments(case, inputs, out))
    run(args)


def _scalar_equal(left: Any, right: Any) -> bool:
    if isinstance(left, str):
        lowered = left.strip().casefold()
        if isinstance(right, bool) and lowered in {"true", "false"}:
            return (lowered == "true") is right
        if right is None and lowered in {"", "none", "null", "nan"}:
            return True
    if isinstance(right, str) and not isinstance(left, str):
        return _scalar_equal(right, left)
    if not (isinstance(left, str) and isinstance(right, str)):
        left_number = _numeric(left)
        right_number = _numeric(right)
        if left_number is not None and right_number is not None:
            return left_number == right_number
    try:
        if pd.isna(left) and right is None:
            return True
    except (TypeError, ValueError):
        pass
    return _json_value(left) == _json_value(right)


def _equivalence_columns(case: Mapping[str, Any]) -> list[str]:
    expected = case.get("expected_outputs")
    raw = expected.get("equivalence_columns") if isinstance(expected, Mapping) else None
    if raw is None:
        metadata = case.get("metadata")
        raw = metadata.get("equivalence_columns") if isinstance(metadata, Mapping) else None
    if raw is None:
        return [
            "test_id",
            "sequence_no",
            "F_kN",
            "p_kPa",
            "settlement_mm",
            "branch",
            "status",
        ]
    if not isinstance(raw, list) or any(not isinstance(item, str) for item in raw):
        raise AcceptanceManifestError("expected_outputs.equivalence_columns must be a string list.")
    return raw


def _run_manual_equivalence(
    case: Mapping[str, Any], inputs: Mapping[str, Path], out: Path
) -> tuple[bool, dict[str, Any]]:
    from .manual_entry_adapter import adapt_manual_draft
    from .manual_entry_models import ManualDraft

    draft_path = inputs.get("manual_draft")
    existing_xlsx = inputs.get("equivalent_protocol")
    projection = inputs.get("excel_projection")
    if draft_path is None or (existing_xlsx is None and projection is None):
        raise AcceptanceManifestError(
            f"{case['case_id']} manual/equivalence execution requires manual_draft and "
            "either an XLSX equivalent_protocol or an excel_projection."
        )
    draft = ManualDraft.from_json(draft_path.read_bytes())
    bundle = adapt_manual_draft(draft)
    manual, issues = bundle.prepare()
    blocking = [issue for issue in issues if bool(issue.blocks_processing)]
    if blocking:
        return False, {
            "message": f"Manual production adapter reported {len(blocking)} blocking issue(s)."
        }
    out.mkdir(parents=True, exist_ok=True)
    manual.to_csv(out / "manual_prepared.csv", index=False)

    if projection is not None:
        helper = (
            Path(__file__).resolve().parents[1] / "acceptance" / "materialize_equivalent_xlsx.py"
        )
        declared_helper = inputs.get("xlsx_materializer")
        if declared_helper is not None and declared_helper.resolve() != helper.resolve():
            raise AcceptanceManifestError(
                "xlsx_materializer must reference the repository's reviewed acceptance helper."
            )
        if not helper.is_file():
            raise AcceptanceManifestError("The reviewed XLSX acceptance materializer is missing.")
        spec = importlib.util.spec_from_file_location(
            "_soilstamp_reviewed_acceptance_xlsx_materializer", helper
        )
        if spec is None or spec.loader is None:
            raise AcceptanceManifestError("Cannot load the reviewed XLSX materializer.")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        materialize = getattr(module, "materialize_equivalent_xlsx", None)
        if not callable(materialize):
            raise AcceptanceManifestError("XLSX materializer API is unavailable.")
        equivalent = Path(
            materialize(
                projection,
                out / "materialized" / "equivalent_excel_manual.xlsx",
            )
        )
    else:
        equivalent = Path(existing_xlsx)  # type: ignore[arg-type]
    if equivalent.suffix.casefold() not in {".xlsx", ".xlsm"} or not zipfile.is_zipfile(equivalent):
        raise AcceptanceManifestError(
            "Manual equivalence must execute a real OOXML .xlsx/.xlsm through the importer."
        )

    metadata_path = inputs.get("metadata")
    if metadata_path is None:
        metadata_path = out / "manual_adapter_metadata.json"
        metadata_path.write_bytes(bundle.metadata_bytes)
    case_metadata = case.get("metadata")
    cli_args = case_metadata.get("cli_args", []) if isinstance(case_metadata, Mapping) else []
    import_mode = "strict"
    for index, item in enumerate(cli_args):
        if item == "--import-mode" and index + 1 < len(cli_args):
            import_mode = str(cli_args[index + 1])
        elif isinstance(item, str) and item.startswith("--import-mode="):
            import_mode = item.partition("=")[2]
    if import_mode != "strict":
        raise AcceptanceManifestError(
            "Manual/XLSX equivalence must use the production strict XLSX importer."
        )
    cli_inputs = {"protocol": equivalent, "metadata": metadata_path}
    _run_cli_case(case, cli_inputs, out)

    canonical = pd.read_csv(out / "prepared.csv")
    columns = _equivalence_columns(case)
    missing = [
        name for name in columns if name not in manual.columns or name not in canonical.columns
    ]
    if missing:
        return False, {"message": "Equivalence columns are missing: " + ", ".join(missing)}
    if len(manual) != len(canonical):
        return False, {
            "message": f"Manual/canonical row counts differ ({len(manual)} != {len(canonical)})."
        }
    tolerances = case.get("tolerances", {})
    metadata = case.get("metadata")
    tolerance_name = (
        metadata.get("equivalence_tolerance", "equivalence")
        if isinstance(metadata, Mapping)
        else "equivalence"
    )
    tolerance = _resolve_tolerance(tolerances, tolerance_name)
    if tolerance is None and isinstance(tolerances, Mapping):
        for fallback in ("strict", "default"):
            if fallback in tolerances:
                tolerance = _resolve_tolerance(tolerances, fallback)
                break
    mismatches: list[str] = []
    for row_index, (manual_row, canonical_row) in enumerate(
        zip(manual[columns].to_dict("records"), canonical[columns].to_dict("records"), strict=True)
    ):
        for column in columns:
            if not _compare_value(manual_row[column], canonical_row[column], tolerance):
                mismatches.append(f"row {row_index + 1} {column}")
                if len(mismatches) >= 10:
                    break
        if len(mismatches) >= 10:
            break
    if mismatches:
        return False, {
            "message": "Manual/equivalent prepared data differ at " + ", ".join(mismatches)
        }
    return True, {
        "contract_version": "manual-xlsx-equivalence/1.0",
        "real_xlsx_import": True,
        "import_mode": import_mode,
        "rows_compared": len(manual),
        "fields_compared": len(columns),
        "columns": list(columns),
        "xlsx_sha256": hashlib.sha256(equivalent.read_bytes()).hexdigest(),
        "message": (
            f"Manual and strict XLSX production inputs agree for {len(manual)} rows "
            f"and {len(columns)} fields."
        ),
    }


def _default_production_runner(
    case: Mapping[str, Any], inputs: Mapping[str, Path], out: Path
) -> object:
    source_type = str(case["source_type"]).strip().casefold().replace("-", "_")
    if source_type in {"manual", "equivalence", "manual_equivalence"}:
        passed, evidence = _run_manual_equivalence(case, inputs, out)
        if not passed:
            raise RuntimeError(str(evidence.get("message") or "Manual/XLSX equivalence failed."))
        return {"input_equivalence": evidence}
    if _normal_source_type(case, inputs):
        _run_cli_case(case, inputs, out)
        return None
    raise AcceptanceManifestError(
        f"Unsupported source_type {case['source_type']!r}; it cannot be accepted implicitly."
    )


def _run_configured_production_extensions(context: _CaseContext) -> None:
    """Run explicitly requested public production analyses absent from CLI output."""

    configured = context.expected.get("group_comparison")
    if configured is None:
        return
    if not isinstance(configured, Mapping):
        raise AcceptanceManifestError("expected_outputs.group_comparison must be an object.")
    baseline = configured.get("baseline_group")
    reinforced = configured.get("reinforced_group")
    if not isinstance(baseline, str) or not baseline.strip():
        raise AcceptanceManifestError("group_comparison.baseline_group must be non-empty text.")
    if not isinstance(reinforced, str) or not reinforced.strip():
        raise AcceptanceManifestError("group_comparison.reinforced_group must be non-empty text.")
    raw_artifact = configured.get("artifact", "group_comparison.csv")
    artifact = _safe_relative_path(raw_artifact, field_name="group comparison artifact")
    target = _resolve_beneath(context.output_dir, artifact, field_name="group comparison artifact")
    bootstrap = configured.get("bootstrap", 1000)
    seed = configured.get("seed", 202604)
    if (
        not isinstance(bootstrap, int)
        or isinstance(bootstrap, bool)
        or bootstrap < 0
        or not isinstance(seed, int)
        or isinstance(seed, bool)
    ):
        raise AcceptanceManifestError("group_comparison bootstrap/seed must be integers.")

    from .analysis import compare_groups

    prepared = pd.read_csv(context.output_dir / "prepared.csv")
    comparison = compare_groups(
        prepared,
        baseline.strip(),
        reinforced.strip(),
        bootstrap=bootstrap,
        seed=seed,
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(target, index=False)
    context.add(
        "production_group_comparison",
        "production_pipeline",
        True,
        "Public compare_groups production analysis completed on prepared.csv.",
        expected={
            "baseline_group": baseline.strip(),
            "reinforced_group": reinforced.strip(),
        },
        actual={"rows": len(comparison), "artifact": artifact.as_posix()},
    )


def _safe_artifact_reference(value: Any) -> tuple[PurePosixPath, PurePosixPath | None]:
    if not isinstance(value, str) or value.count("!") > 1:
        raise AcceptanceManifestError(
            "Artifact reference must be a relative path or archive!member."
        )
    archive, separator, member = value.partition("!")
    outer = _safe_relative_path(archive, field_name="artifact")
    inner = _safe_relative_path(member, field_name="archive member") if separator else None
    return outer, inner


def _read_artifact(output_dir: Path, reference: Any) -> bytes:
    outer, inner = _safe_artifact_reference(reference)
    path = _resolve_beneath(output_dir, outer, field_name="artifact")
    if not path.is_file():
        raise FileNotFoundError(str(outer))
    if inner is None:
        return path.read_bytes()
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) > _MAX_ARCHIVE_MEMBERS:
            raise ValueError("Archive contains too many members.")
        target = inner.as_posix()
        matches = [info for info in archive.infolist() if info.filename == target]
        if len(matches) != 1:
            raise KeyError(f"Archive member {target!r} is missing or duplicated.")
        info = matches[0]
        if info.file_size > _MAX_ARCHIVE_MEMBER_BYTES:
            raise ValueError("Archive member is too large for acceptance inspection.")
        return archive.read(info)


def _json_pointer(payload: Any, pointer: Any) -> Any:
    if pointer in {None, ""}:
        return payload
    if not isinstance(pointer, str):
        raise AcceptanceManifestError("JSON pointer must be text.")
    if pointer.startswith("/"):
        parts = pointer[1:].split("/")
        parts = [part.replace("~1", "/").replace("~0", "~") for part in parts]
    else:
        parts = pointer.split(".")
    current = payload
    for part in parts:
        if isinstance(current, list):
            current = current[int(part)]
        elif isinstance(current, Mapping):
            current = current[part]
        else:
            raise KeyError(pointer)
    return current


def _resolve_tolerance(tolerances: Any, name: Any) -> tuple[float, float] | None:
    if isinstance(name, Mapping):
        spec = name
    elif isinstance(tolerances, Mapping):
        spec = tolerances.get(str(name)) if name not in {None, ""} else tolerances.get("default")
    else:
        spec = None
    if spec is None:
        return None
    if isinstance(spec, (int, float)) and not isinstance(spec, bool):
        absolute = float(spec)
        relative = 0.0
    elif isinstance(spec, Mapping):
        absolute = float(spec.get("absolute", spec.get("atol", 0.0)))
        relative = float(spec.get("relative", spec.get("rtol", 0.0)))
    else:
        raise AcceptanceManifestError(f"Invalid tolerance definition for {name!r}.")
    if not math.isfinite(absolute) or not math.isfinite(relative) or absolute < 0 or relative < 0:
        raise AcceptanceManifestError("Tolerances must be finite non-negative numbers.")
    return absolute, relative


def _numeric(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _compare_value(actual: Any, expected: Any, tolerance: tuple[float, float] | None) -> bool:
    actual_number = _numeric(actual)
    expected_number = _numeric(expected)
    if actual_number is not None and expected_number is not None:
        if tolerance is None:
            return actual_number == expected_number
        absolute, relative = tolerance
        return abs(actual_number - expected_number) <= absolute + relative * abs(expected_number)
    return _scalar_equal(actual, expected)


def _expected_value_spec(
    expected: Any, tolerances: Mapping[str, Any], default_tolerance: Any = None
) -> tuple[Any, tuple[float, float] | None]:
    if isinstance(expected, Mapping) and "value" in expected:
        return expected["value"], _resolve_tolerance(
            tolerances, expected.get("tolerance", default_tolerance)
        )
    return expected, _resolve_tolerance(tolerances, default_tolerance)


def _csv_rows(payload: bytes) -> list[dict[str, str]]:
    text = payload.decode("utf-8-sig")
    return list(csv.DictReader(text.splitlines()))


def _select_csv_row(rows: list[dict[str, Any]], selector: Any) -> dict[str, Any]:
    if selector is None or selector == {}:
        if len(rows) != 1:
            raise ValueError(f"Expected exactly one row, found {len(rows)}.")
        return rows[0]
    if not isinstance(selector, Mapping):
        raise AcceptanceManifestError("CSV selector must be an object.")
    matches = [
        row
        for row in rows
        if all(_scalar_equal(row.get(str(column)), value) for column, value in selector.items())
    ]
    if len(matches) != 1:
        raise ValueError(f"CSV selector matched {len(matches)} rows instead of one.")
    return matches[0]


def _check_required_artifacts(context: _CaseContext) -> None:
    configured = context.expected.get("required_artifacts", [])
    if not isinstance(configured, list) or any(not isinstance(item, str) for item in configured):
        raise AcceptanceManifestError("expected_outputs.required_artifacts must be a string list.")
    baseline = [
        "failure_analysis.json",
        "failure_summary.csv",
        "artifact_manifest.json",
        "report.html",
        "report.xlsx",
        "approval_report.zip",
    ]
    if _modulus_expected(context):
        baseline.append("moduli.csv")
    group_comparison = context.expected.get("group_comparison")
    if isinstance(group_comparison, Mapping):
        baseline.append(str(group_comparison.get("artifact", "group_comparison.csv")))
    for index, reference in enumerate(dict.fromkeys([*baseline, *configured])):
        try:
            payload = _read_artifact(context.output_dir, reference)
            passed = bool(payload)
            message = f"Required artifact {reference} exists and is non-empty."
        except (OSError, ValueError, KeyError, zipfile.BadZipFile) as exc:
            passed = False
            message = f"Required artifact {reference} is unavailable: {exc}"
        context.add(
            f"artifact:{index}:{reference}",
            "artifact",
            passed,
            message,
            expected="present_nonempty",
            actual="present_nonempty" if passed else "missing_or_invalid",
        )


def _modulus_expected(context: _CaseContext) -> bool:
    required = context.expected.get("required_artifacts", [])
    if isinstance(required, list) and "moduli.csv" in required:
        return True
    rows = context.expected.get("modulus_rows", [])
    if isinstance(rows, list) and bool(rows):
        return True
    if context.expected.get("check_modulus") is True:
        return True
    review = context.case.get("expected_review_status")
    if isinstance(review, Mapping):
        return "moduli" in review and bool(review["moduli"])
    if isinstance(review, list):
        return bool(review)
    return review not in {None, ""}


def _check_production_science_contract(context: _CaseContext) -> None:
    try:
        analysis = json.loads(_read_artifact(context.output_dir, "failure_analysis.json"))
        invariants = {
            "contract_version": (analysis.get("contract_version"), "failure-analysis/1.0"),
            "summary_method": (analysis.get("summary_method"), "none"),
            "point_estimate": (analysis.get("point_estimate"), None),
        }
        for name, (actual, expected) in invariants.items():
            context.add(
                f"science:failure_analysis:{name}",
                "scientific_flag",
                _scalar_equal(actual, expected),
                f"Failure-analysis invariant {name} is explicit.",
                expected=expected,
                actual=actual,
            )
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        context.add(
            "science:failure_analysis:read",
            "scientific_flag",
            False,
            f"Cannot inspect failure analysis: {exc}",
        )

    try:
        failure_rows = _csv_rows(_read_artifact(context.output_dir, "failure_summary.csv"))
        required = {
            "failure_observed",
            "interval_censored",
            "right_censored",
            "censoring_type",
            "classification_status",
            "lower_bound",
            "upper_bound",
        }
        columns = set(failure_rows[0]) if failure_rows else set()
        passed = bool(failure_rows) and required <= columns
        context.add(
            "science:failure_censoring_contract",
            "failure_censoring",
            passed,
            "Failure/censoring output exposes observed, interval, right-censored and bounds fields.",
            expected=sorted(required),
            actual=sorted(columns),
        )
    except (OSError, UnicodeError, csv.Error, ValueError, KeyError) as exc:
        context.add("science:failure_censoring_contract", "failure_censoring", False, str(exc))

    if _modulus_expected(context):
        try:
            modulus_rows = _csv_rows(_read_artifact(context.output_dir, "moduli.csv"))
            required = {
                "E_stamp_app_kPa",
                "profile_id",
                "profile_version",
                "p_range_source",
                "review_status",
                "used_indices",
            }
            columns = set(modulus_rows[0]) if modulus_rows else set()
            passed = bool(modulus_rows) and required <= columns
            context.add(
                "science:modulus_contract",
                "modulus",
                passed,
                "E output exposes value, profile, range source, review status and used range.",
                expected=sorted(required),
                actual=sorted(columns),
            )
        except (OSError, UnicodeError, csv.Error, ValueError, KeyError) as exc:
            context.add("science:modulus_contract", "modulus", False, str(exc))


def _check_scientific_flags(context: _CaseContext) -> None:
    checks = context.expected.get("scientific_flags", [])
    if isinstance(checks, Mapping):
        checks = [
            {"artifact": "failure_analysis.json", "pointer": pointer, "expected": expected}
            for pointer, expected in checks.items()
        ]
    if not isinstance(checks, list):
        raise AcceptanceManifestError("expected_outputs.scientific_flags must be a list or object.")
    for index, spec in enumerate(checks):
        if not isinstance(spec, Mapping) or "artifact" not in spec or "expected" not in spec:
            raise AcceptanceManifestError(
                "Each scientific flag needs artifact and expected fields."
            )
        actual: Any = None
        try:
            payload = json.loads(_read_artifact(context.output_dir, spec["artifact"]))
            actual = _json_pointer(payload, spec.get("pointer", ""))
            expected, tolerance = _expected_value_spec(
                spec["expected"], context.tolerances, spec.get("tolerance")
            )
            passed = _compare_value(actual, expected, tolerance)
            message = "Scientific flag matches its explicit expectation."
        except (OSError, ValueError, KeyError, IndexError, json.JSONDecodeError) as exc:
            expected = spec.get("expected")
            passed = False
            message = f"Cannot evaluate scientific flag: {exc}"
        context.add(
            f"scientific_flag:{index}",
            "scientific_flag",
            passed,
            message,
            expected=expected,
            actual=actual,
        )


def _check_expected_rows(
    context: _CaseContext,
    *,
    config_key: str,
    default_artifact: str,
    category: str,
) -> None:
    specs = context.expected.get(config_key, [])
    if not isinstance(specs, list):
        raise AcceptanceManifestError(f"expected_outputs.{config_key} must be a list.")
    cache: dict[str, list[dict[str, str]]] = {}
    for row_index, spec in enumerate(specs):
        if not isinstance(spec, Mapping) or not isinstance(spec.get("expected"), Mapping):
            raise AcceptanceManifestError(
                f"{config_key} entries need selector and expected objects."
            )
        artifact = str(spec.get("artifact", default_artifact))
        actual_row: Mapping[str, Any] = {}
        try:
            rows = cache.setdefault(
                artifact, _csv_rows(_read_artifact(context.output_dir, artifact))
            )
            actual_row = _select_csv_row(rows, spec.get("selector", {}))
            row_error: Exception | None = None
        except (OSError, UnicodeError, csv.Error, ValueError, KeyError) as exc:
            row_error = exc
        for field_name, configured_expected in spec["expected"].items():
            expected, tolerance = _expected_value_spec(
                configured_expected,
                context.tolerances,
                spec.get("tolerance", field_name),
            )
            actual = actual_row.get(str(field_name)) if row_error is None else None
            passed = row_error is None and _compare_value(actual, expected, tolerance)
            message = (
                f"{category} field {field_name} matches the golden value."
                if row_error is None
                else f"Cannot select {category} row: {row_error}"
            )
            context.add(
                f"{config_key}:{row_index}:{field_name}",
                category,
                passed,
                message,
                expected=expected,
                actual=actual,
            )


def _check_golden_values(context: _CaseContext) -> None:
    specs = context.expected.get("golden_values", context.expected.get("values", []))
    if not isinstance(specs, list):
        raise AcceptanceManifestError("expected_outputs.golden_values must be a list.")
    for index, spec in enumerate(specs):
        if not isinstance(spec, Mapping) or "artifact" not in spec or "expected" not in spec:
            raise AcceptanceManifestError("Golden values need artifact and expected fields.")
        actual: Any = None
        try:
            payload = _read_artifact(context.output_dir, spec["artifact"])
            if "pointer" in spec:
                actual = _json_pointer(json.loads(payload), spec["pointer"])
            else:
                rows = _csv_rows(payload)
                row = _select_csv_row(rows, spec.get("selector", {}))
                field_name = spec.get("field")
                if not isinstance(field_name, str):
                    raise AcceptanceManifestError("CSV golden values need a field name.")
                actual = row.get(field_name)
            expected, tolerance = _expected_value_spec(
                spec["expected"], context.tolerances, spec.get("tolerance")
            )
            passed = _compare_value(actual, expected, tolerance)
            message = "Golden value matches within its declared tolerance."
        except (
            OSError,
            UnicodeError,
            csv.Error,
            ValueError,
            KeyError,
            json.JSONDecodeError,
        ) as exc:
            expected = spec.get("expected")
            passed = False
            message = f"Cannot evaluate golden value: {exc}"
        context.add(
            f"golden:{index}",
            "golden_value",
            passed,
            message,
            expected=expected,
            actual=actual,
        )


def _check_expected_hashes(context: _CaseContext) -> None:
    hashes = context.expected.get("hashes", {})
    if not isinstance(hashes, Mapping):
        raise AcceptanceManifestError("expected_outputs.hashes must be an object.")
    for index, (reference, expected) in enumerate(sorted(hashes.items())):
        if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", expected):
            raise AcceptanceManifestError(f"Invalid SHA-256 expectation for {reference!r}.")
        actual: str | None = None
        try:
            actual = hashlib.sha256(_read_artifact(context.output_dir, reference)).hexdigest()
            passed = actual == expected.casefold()
            message = f"SHA-256 for {reference} matches."
        except (OSError, ValueError, KeyError, zipfile.BadZipFile) as exc:
            passed = False
            message = f"Cannot hash {reference}: {exc}"
        context.add(
            f"hash:{index}:{reference}",
            "hash",
            passed,
            message,
            expected=expected.casefold(),
            actual=actual,
        )


def _portable_zip_name(name: str) -> str | None:
    if "\\" in name or "\x00" in name:
        return None
    path = PurePosixPath(name)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        return None
    return path.as_posix()


def _check_report_package(context: _CaseContext) -> None:
    if context.expected.get("report_package", True) is not True:
        context.add(
            "report_package:required",
            "report_package",
            False,
            "Report-package verification cannot be disabled in an acceptance case.",
            expected=True,
            actual=context.expected.get("report_package"),
        )
        return
    problems: list[str] = []
    try:
        manifest_bytes = _read_artifact(context.output_dir, "artifact_manifest.json")
        manifest = json.loads(manifest_bytes)
        if manifest.get("schema_version") != "approval-report-package/1.0":
            problems.append("wrong artifact manifest schema_version")
        records = manifest.get("files")
        if not isinstance(records, list) or not records:
            problems.append("artifact manifest has no files")
            records = []
        archive_path = _resolve_beneath(
            context.output_dir,
            _safe_relative_path("approval_report.zip", field_name="approval archive"),
            field_name="approval archive",
        )
        with zipfile.ZipFile(archive_path) as archive:
            infos = archive.infolist()
            if len(infos) > _MAX_ARCHIVE_MEMBERS:
                problems.append("approval archive has too many members")
            portable: list[str] = []
            for info in infos:
                normalized = _portable_zip_name(info.filename)
                if normalized is None:
                    problems.append(f"unsafe archive member {info.filename!r}")
                    continue
                portable.append(normalized)
                if info.file_size > _MAX_ARCHIVE_MEMBER_BYTES:
                    problems.append(f"archive member is too large: {normalized}")
            if len({name.casefold() for name in portable}) != len(portable):
                problems.append("approval archive has duplicate portable paths")
            for record in records:
                if not isinstance(record, Mapping):
                    problems.append("artifact manifest contains a non-object record")
                    continue
                path = record.get("path")
                if not isinstance(path, str) or _portable_zip_name(path) is None:
                    problems.append(f"manifest has unsafe path {path!r}")
                    continue
                matches = [info for info in infos if info.filename == path]
                if len(matches) != 1:
                    problems.append(f"manifest member {path!r} is missing or duplicated")
                    continue
                payload = archive.read(matches[0])
                if len(payload) != record.get("bytes"):
                    problems.append(f"byte count mismatch for {path}")
                if hashlib.sha256(payload).hexdigest() != record.get("sha256"):
                    problems.append(f"SHA-256 mismatch for {path}")
            for report_name in ("report.html", "report.xlsx", "artifact_manifest.json"):
                if report_name not in portable:
                    problems.append(f"approval archive is missing {report_name}")
                else:
                    external = context.output_dir / report_name
                    if not external.is_file() or archive.read(report_name) != external.read_bytes():
                        problems.append(f"external and archived {report_name} differ")
        if "artifact_manifest.json" in portable:
            with zipfile.ZipFile(archive_path) as archive:
                if archive.read("artifact_manifest.json") != manifest_bytes:
                    problems.append("external and archived artifact manifests differ")
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        json.JSONDecodeError,
        zipfile.BadZipFile,
    ) as exc:
        problems.append(str(exc))
    context.add(
        "report_package:integrity",
        "report_package",
        not problems,
        "Approval report package manifest, paths, byte counts and SHA-256 hashes are valid."
        if not problems
        else "; ".join(problems[:20]),
        expected="internally_consistent",
        actual="internally_consistent" if not problems else problems[:20],
    )


def _warning_codes(output_dir: Path) -> set[str]:
    result: set[str] = set()
    validation = output_dir / "validation_issues.csv"
    if validation.is_file():
        for row in _csv_rows(validation.read_bytes()):
            code = str(row.get("code") or "").strip()
            if code:
                result.add(code)
    analysis = output_dir / "analysis_warnings.json"
    if analysis.is_file():
        payload = json.loads(analysis.read_text(encoding="utf-8-sig"))
        if isinstance(payload, list):
            for row in payload:
                if isinstance(row, Mapping):
                    code = str(row.get("code") or row.get("analysis") or "").strip()
                    if code:
                        result.add(code)
    return result


def _check_warnings(context: _CaseContext) -> None:
    configured = context.case["expected_warnings"]
    if isinstance(configured, Mapping):
        expected = configured.get("codes", [])
        mode = str(configured.get("match", "exact"))
    else:
        expected = configured
        mode = "exact"
    if not isinstance(expected, list) or any(not isinstance(item, str) for item in expected):
        raise AcceptanceManifestError("expected_warnings codes must be a string list.")
    if mode not in {"exact", "subset"}:
        raise AcceptanceManifestError("expected_warnings.match must be exact or subset.")
    try:
        actual_set = _warning_codes(context.output_dir)
        expected_set = set(expected)
        passed = expected_set <= actual_set if mode == "subset" else expected_set == actual_set
        message = f"Warning code set matches in {mode} mode."
    except (OSError, UnicodeError, csv.Error, json.JSONDecodeError) as exc:
        actual_set = set()
        expected_set = set(expected)
        passed = False
        message = f"Cannot inspect warning artifacts: {exc}"
    context.add(
        "warnings",
        "warning",
        passed,
        message,
        expected=sorted(expected_set),
        actual=sorted(actual_set),
    )


def _statuses_from_artifact(output_dir: Path, artifact: str, column: str) -> set[str]:
    try:
        rows = _csv_rows(_read_artifact(output_dir, artifact))
    except FileNotFoundError:
        return set()
    return {
        str(row.get(column) or "").strip() for row in rows if str(row.get(column) or "").strip()
    }


def _check_review_status(context: _CaseContext) -> None:
    configured = context.case["expected_review_status"]
    if isinstance(configured, Mapping):
        categories = {
            "moduli": ("moduli.csv", "review_status"),
            "failure": ("failure_summary.csv", "classification_status"),
        }
        for index, (category, expected_raw) in enumerate(sorted(configured.items())):
            if category not in categories:
                raise AcceptanceManifestError(f"Unsupported review-status category {category!r}.")
            expected = {
                str(value)
                for value in (expected_raw if isinstance(expected_raw, list) else [expected_raw])
            }
            artifact, column = categories[category]
            actual = _statuses_from_artifact(context.output_dir, artifact, column)
            context.add(
                f"review_status:{index}:{category}",
                "review_status",
                actual == expected,
                f"{category} review statuses match exactly.",
                expected=sorted(expected),
                actual=sorted(actual),
            )
        return
    expected = {
        str(value) for value in (configured if isinstance(configured, list) else [configured])
    }
    actual = _statuses_from_artifact(context.output_dir, "moduli.csv", "review_status")
    context.add(
        "review_status:moduli",
        "review_status",
        actual == expected,
        "Modulus review statuses match exactly.",
        expected=sorted(expected),
        actual=sorted(actual),
    )


def _case_result(context: _CaseContext) -> AcceptanceCaseResult:
    data_class = _case_data_class(context.case)
    signoff = str(context.case["signoff_status"]).strip().casefold()
    reviewer = context.case["reviewer"]
    reference = context.case["independent_calculation_reference"]
    context.add(
        "independent_calculation_reference",
        "traceability",
        isinstance(reference, str) and bool(reference.strip()),
        "Independent calculation reference is explicit.",
        expected="nonempty_reference",
        actual=reference,
    )
    if data_class == "real":
        signed = signoff in _SIGNED_STATUSES and _reviewer_present(reviewer)
        context.add(
            "real_engineering_signoff",
            "engineering_signoff",
            signed,
            "A real test can pass acceptance only with an identified engineer and signoff.",
            expected="signed_with_reviewer",
            actual={"signoff_status": signoff, "reviewer_present": _reviewer_present(reviewer)},
        )
    checks = tuple(sorted(context.checks, key=lambda item: item.check_id))
    return AcceptanceCaseResult(
        case_id=context.case_id,
        source_type=str(context.case["source_type"]),
        data_class=data_class,
        signoff_status=signoff,
        reviewer=reviewer,
        checks=checks,
    )


def _run_case(
    case: Mapping[str, Any],
    *,
    inputs: Mapping[str, Path],
    output_dir: Path,
    initial_checks: Sequence[AcceptanceCheck],
    production_runner: ProductionRunner,
) -> AcceptanceCaseResult:
    context = _CaseContext(case=case, inputs=inputs, output_dir=output_dir)
    context.checks.extend(initial_checks)
    if any(check.critical and not check.passed for check in initial_checks):
        context.add(
            "production_pipeline",
            "production_pipeline",
            False,
            "Production pipeline was not executed because input integrity checks failed.",
            expected="verified_inputs_then_completed",
            actual="not_executed",
        )
        return _case_result(context)
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        production_evidence = production_runner(case, inputs, output_dir)
        if isinstance(production_evidence, Mapping) and production_evidence.get(
            "input_equivalence"
        ):
            equivalence = production_evidence["input_equivalence"]
            equivalence_passed = (
                isinstance(equivalence, Mapping)
                and equivalence.get("contract_version") == "manual-xlsx-equivalence/1.0"
                and equivalence.get("real_xlsx_import") is True
                and equivalence.get("import_mode") == "strict"
                and isinstance(equivalence.get("rows_compared"), int)
                and equivalence["rows_compared"] > 0
                and isinstance(equivalence.get("fields_compared"), int)
                and equivalence["fields_compared"] > 0
            )
            context.add(
                "manual_xlsx_equivalence",
                "input_equivalence",
                equivalence_passed,
                str(
                    equivalence.get("message")
                    if isinstance(equivalence, Mapping)
                    else "Manual/XLSX equivalence completed."
                ),
                expected={
                    "contract_version": "manual-xlsx-equivalence/1.0",
                    "real_xlsx_import": True,
                    "import_mode": "strict",
                    "comparison": "row_by_row_scientific_fields",
                },
                actual=_json_value(equivalence),
            )
        _run_configured_production_extensions(context)
        context.add(
            "production_pipeline",
            "production_pipeline",
            True,
            "Production pipeline completed.",
            expected="completed",
            actual=(
                "completed" if production_evidence is None else _json_value(production_evidence)
            ),
        )
    except (Exception, SystemExit) as exc:  # argparse and production errors become report evidence.
        context.add(
            "production_pipeline",
            "production_pipeline",
            False,
            f"Production pipeline failed: {type(exc).__name__}: {exc}",
            expected="completed",
            actual="failed",
        )
        return _case_result(context)

    _check_required_artifacts(context)
    _check_production_science_contract(context)
    _check_scientific_flags(context)
    _check_expected_rows(
        context,
        config_key="failure_rows",
        default_artifact="failure_summary.csv",
        category="failure_censoring",
    )
    _check_expected_rows(
        context,
        config_key="modulus_rows",
        default_artifact="moduli.csv",
        category="modulus",
    )
    _check_golden_values(context)
    _check_expected_hashes(context)
    _check_report_package(context)
    _check_warnings(context)
    _check_review_status(context)
    return _case_result(context)


def _markdown_escape(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\r", " ").replace("\n", " ")


def _gate_text(gate: Any) -> str:
    if not isinstance(gate, Mapping):
        return str(gate)
    gate_id = str(gate.get("gate_id") or "")
    status = str(gate.get("status") or "unsigned")
    evidence = str(gate.get("required_evidence") or "").strip()
    return f"{gate_id} ({status})" + (f": {evidence}" if evidence else "")


def _compact_json(value: Any) -> str:
    return json.dumps(
        _json_value(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _render_markdown(payload: Mapping[str, Any]) -> str:
    lines = [
        "# Acceptance report",
        "",
        f"- Contract: `{payload['contract_version']}`",
        f"- Candidate status: `{payload['candidate_status']}`",
        f"- Technical status: **{str(payload['technical_status']).upper()}**",
        f"- Synthetic acceptance passed: `{str(payload['synthetic_acceptance_passed']).lower()}`",
        "- Engineering acceptance: `false`",
        "",
        "A passing synthetic run is evidence for a candidate for engineering acceptance; "
        "it is not an engineering approval or a final release.",
        "",
        "## Unsigned engineering gates",
        "",
    ]
    gates = payload.get("unsigned_engineering_gates") or []
    lines.extend(f"- {_markdown_escape(_gate_text(gate))}" for gate in gates)
    if not gates:
        lines.append("- None recorded in this manifest.")
    for case in payload["cases"]:
        lines.extend(
            [
                "",
                f"## Case `{_markdown_escape(case['case_id'])}`",
                "",
                f"Technical status: **{str(case['technical_status']).upper()}**; "
                f"signoff: `{_markdown_escape(case['signoff_status'])}`.",
                "",
                "| Check | Category | Status | Critical | Expected | Actual | Message |",
                "|---|---|---:|---:|---|---|---|",
            ]
        )
        for check in case["checks"]:
            lines.append(
                "| "
                + " | ".join(
                    _markdown_escape(value)
                    for value in (
                        check["check_id"],
                        check["category"],
                        check["status"],
                        str(check["critical"]).lower(),
                        _compact_json(check["expected"]),
                        _compact_json(check["actual"]),
                        check["message"],
                    )
                )
                + " |"
            )
    return "\n".join(lines) + "\n"


def _render_html(payload: Mapping[str, Any]) -> str:
    case_sections: list[str] = []
    for case in payload["cases"]:
        rows = "".join(
            "<tr>"
            f"<td><code>{html.escape(str(check['check_id']))}</code></td>"
            f"<td>{html.escape(str(check['category']))}</td>"
            f"<td>{html.escape(str(check['status']))}</td>"
            f"<td>{str(bool(check['critical'])).lower()}</td>"
            f"<td><code>{html.escape(_compact_json(check['expected']))}</code></td>"
            f"<td><code>{html.escape(_compact_json(check['actual']))}</code></td>"
            f"<td>{html.escape(str(check['message']))}</td>"
            "</tr>"
            for check in case["checks"]
        )
        case_sections.append(
            f"<section><h2>Case <code>{html.escape(str(case['case_id']))}</code></h2>"
            f"<p>Technical status: <strong>{html.escape(str(case['technical_status']).upper())}</strong>; "
            f"signoff: <code>{html.escape(str(case['signoff_status']))}</code>.</p>"
            "<table><thead><tr><th>Check</th><th>Category</th><th>Status</th>"
            f"<th>Critical</th><th>Expected</th><th>Actual</th><th>Message</th>"
            f"</tr></thead><tbody>{rows}</tbody></table></section>"
        )
    gates = (
        "".join(
            f"<li>{html.escape(_gate_text(gate))}</li>"
            for gate in payload.get("unsigned_engineering_gates", [])
        )
        or "<li>None recorded in this manifest.</li>"
    )
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        "<title>Acceptance report</title><style>body{font-family:system-ui,sans-serif;"
        "max-width:1200px;margin:2rem auto;padding:0 1rem}table{border-collapse:collapse;"
        "width:100%}th,td{border:1px solid #bbb;padding:.35rem;text-align:left}"
        "code{overflow-wrap:anywhere}</style></head><body><h1>Acceptance report</h1>"
        f"<dl><dt>Contract</dt><dd><code>{html.escape(str(payload['contract_version']))}</code></dd>"
        f"<dt>Candidate status</dt><dd><code>{html.escape(str(payload['candidate_status']))}</code></dd>"
        f"<dt>Technical status</dt><dd><strong>{html.escape(str(payload['technical_status']).upper())}</strong></dd>"
        f"<dt>Synthetic acceptance passed</dt><dd>{str(bool(payload['synthetic_acceptance_passed'])).lower()}</dd>"
        "<dt>Engineering acceptance</dt><dd>false</dd></dl>"
        "<p>A passing synthetic run is evidence for a candidate for engineering acceptance; "
        "it is not an engineering approval or a final release.</p>"
        f"<h2>Unsigned engineering gates</h2><ul>{gates}</ul>"
        + "".join(case_sections)
        + "</body></html>\n"
    )


def _write_reports(payload: Mapping[str, Any], out_dir: Path) -> tuple[Path, Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "acceptance_report.json"
    markdown_path = out_dir / "acceptance_report.md"
    html_path = out_dir / "acceptance_report.html"
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(_render_markdown(payload), encoding="utf-8")
    html_path.write_text(_render_html(payload), encoding="utf-8")
    return json_path, markdown_path, html_path


def _fresh_case_output(root: Path, case_id: str) -> Path:
    """Return an empty case directory without following stale symlinks."""

    root.mkdir(parents=True, exist_ok=True)
    cases_root = root / "cases"
    if cases_root.is_symlink():
        raise AcceptanceManifestError("Acceptance output cases directory must not be a symlink.")
    cases_root.mkdir(exist_ok=True)
    candidate = cases_root / case_id
    if candidate.is_symlink() or candidate.is_file():
        candidate.unlink()
    elif candidate.exists():
        resolved_root = root.resolve()
        resolved_candidate = candidate.resolve()
        try:
            resolved_candidate.relative_to(resolved_root)
        except ValueError as exc:
            raise AcceptanceManifestError(
                "Stale case output resolves outside the output root."
            ) from exc
        shutil.rmtree(resolved_candidate)
    return candidate


def run_acceptance_manifest(
    manifest_path: Path | str,
    out_dir: Path | str,
    *,
    production_runner: ProductionRunner | None = None,
) -> AcceptanceRunResult:
    """Execute cases and write deterministic JSON, Markdown and HTML reports.

    The return value carries ``exit_code`` for CLI integration.  Call
    :meth:`AcceptanceRunResult.raise_for_failure` when exception-based control
    flow is preferred.
    """

    manifest_file = Path(manifest_path).resolve()
    manifest = load_acceptance_manifest(manifest_file)
    root = Path(out_dir).resolve()
    runner = production_runner or _default_production_runner
    cases: list[AcceptanceCaseResult] = []
    for case in sorted(manifest["cases"], key=lambda item: str(item["case_id"])):
        inputs, input_checks = _resolve_inputs(case, manifest_file.parent)
        case_output = _fresh_case_output(root, str(case["case_id"]))
        cases.append(
            _run_case(
                case,
                inputs=inputs,
                output_dir=case_output,
                initial_checks=input_checks,
                production_runner=runner,
            )
        )

    gates: dict[str, dict[str, Any]] = {}
    for item in manifest.get("unsigned_engineering_gates", []):
        if isinstance(item, Mapping):
            record = {str(key): _json_value(value) for key, value in item.items()}
            gate_id = str(record["gate_id"])
        else:
            gate_id = str(item).strip()
            record = {"gate_id": gate_id, "status": "unsigned"}
        gates[gate_id.casefold()] = record
    for case in cases:
        if case.data_class == "real" and (
            case.signoff_status not in _SIGNED_STATUSES or not _reviewer_present(case.reviewer)
        ):
            gate_id = f"{case.case_id}_engineer_signoff"
            gates[gate_id.casefold()] = {
                "gate_id": gate_id,
                "status": "unsigned",
                "required_evidence": "Identified responsible engineer and explicit signoff.",
            }
    placeholder = AcceptanceRunResult(
        cases=tuple(cases),
        unsigned_engineering_gates=tuple(gates[key] for key in sorted(gates)),
        json_report=root / "acceptance_report.json",
        markdown_report=root / "acceptance_report.md",
        html_report=root / "acceptance_report.html",
    )
    json_path, markdown_path, html_path = _write_reports(placeholder.to_dict(), root)
    return AcceptanceRunResult(
        cases=placeholder.cases,
        unsigned_engineering_gates=placeholder.unsigned_engineering_gates,
        json_report=json_path,
        markdown_report=markdown_path,
        html_report=html_path,
    )


__all__ = [
    "ACCEPTANCE_CASE_VERSION",
    "ACCEPTANCE_REPORT_VERSION",
    "CANDIDATE_STATUS",
    "AcceptanceCaseResult",
    "AcceptanceCheck",
    "AcceptanceCriticalMismatch",
    "AcceptanceManifestError",
    "AcceptanceRunResult",
    "ProductionRunner",
    "load_acceptance_manifest",
    "run_acceptance_manifest",
]
