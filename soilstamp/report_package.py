"""Deterministic approval-report package shared by CLI and the web app."""

from __future__ import annotations

import hashlib
import json
import unicodedata
import zipfile
from dataclasses import dataclass
from io import BytesIO
from pathlib import PurePosixPath
from typing import Any, Mapping
from urllib.parse import quote

from .provenance import canonical_json_bytes
from .report_html import build_html_report_package
from .report_xlsx import build_xlsx_report_package


REPORT_PACKAGE_VERSION = "approval-report-package/1.0"
_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
_RESERVED_PATHS = {"artifact_manifest.json", "report.html", "report.xlsx"}
_MEDIA_TYPES = {
    ".csv": "text/csv",
    ".html": "text/html",
    ".jpeg": "image/jpeg",
    ".jpg": "image/jpeg",
    ".json": "application/json",
    ".md": "text/markdown",
    ".pdf": "application/pdf",
    ".png": "image/png",
    ".svg": "image/svg+xml",
    ".txt": "text/plain",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".zip": "application/zip",
}
_WINDOWS_RESERVED_NAMES = {
    "aux",
    "con",
    "nul",
    "prn",
    *(f"com{number}" for number in range(1, 10)),
    *(f"lpt{number}" for number in range(1, 10)),
}
_INDICATOR_MODE_ALIASES = {
    "direct": "increasing",
    "direct_scale": "increasing",
    "direct_wrapped": "increasing_wrapped",
    "direct_scale_wrapped": "increasing_wrapped",
    "reverse": "decreasing",
    "reverse_scale": "decreasing",
    "reverse_wrapped": "decreasing_wrapped",
    "reverse_scale_wrapped": "decreasing_wrapped",
    "increasing": "increasing",
    "increasing_wrapped": "increasing_wrapped",
    "decreasing": "decreasing",
    "decreasing_wrapped": "decreasing_wrapped",
    "direct_displacement": "cumulative_settlement",
    "ready_settlement": "cumulative_settlement",
    "cumulative_settlement": "cumulative_settlement",
}


@dataclass(frozen=True, slots=True)
class ApprovalReportPackage:
    html: bytes
    xlsx: bytes
    artifact_manifest_json: bytes
    archive: bytes
    manifest: dict[str, Any]


def _safe_leaf(value: str, fallback: str) -> str:
    candidate = PurePosixPath(str(value).replace("\\", "/")).name
    cleaned = "".join(
        character if character.isalnum() or character in {"-", "_", "."} else "_"
        for character in unicodedata.normalize("NFC", candidate)
    ).strip(". ")
    return cleaned or fallback


def _json_artifact_bytes(value: Any) -> bytes:
    if hasattr(value, "to_dict"):
        value = value.to_dict()
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            default=str,
        )
        + "\n"
    ).encode("utf-8")


def collect_approval_artifacts(
    *,
    raw: Any,
    prepared: Any,
    source_file_name: str,
    source_file_bytes: bytes,
    metadata_file_name: str | None = None,
    metadata_file_bytes: bytes | None = None,
    result_tables: Mapping[str, Any] | None = None,
    figures: Mapping[str, bytes] | None = None,
    audit: Any = None,
    provenance: Any = None,
    report_markdown: str | None = None,
    config_snapshot: Mapping[str, Any] | None = None,
) -> dict[str, bytes]:
    """Collect one self-contained artifact inventory shared by CLI and GUI."""

    if not isinstance(source_file_bytes, (bytes, bytearray, memoryview)):
        raise TypeError("source_file_bytes must contain the exact source bytes.")
    source_name = _safe_leaf(source_file_name, "protocol.bin")
    artifacts: dict[str, bytes] = {
        f"source/protocol/{source_name}": bytes(source_file_bytes),
    }
    if metadata_file_bytes is not None:
        metadata_name = _safe_leaf(metadata_file_name or "metadata.json", "metadata.json")
        artifacts[f"source/metadata/{metadata_name}"] = bytes(metadata_file_bytes)

    if hasattr(raw, "to_csv"):
        artifacts["data/raw_protocol_view.csv"] = raw.to_csv(index=False).encode(
            "utf-8-sig"
        )
    if hasattr(prepared, "to_csv"):
        artifacts["data/prepared_machine.csv"] = prepared.to_csv(index=False).encode(
            "utf-8"
        )

    portable_result_names: set[str] = set()
    for raw_name, value in sorted((result_tables or {}).items(), key=lambda item: str(item[0])):
        result_name = _safe_leaf(str(raw_name), "result")
        portable_name = result_name.casefold()
        if portable_name in portable_result_names:
            raise ValueError(f"Duplicate portable report result name: {result_name!r}.")
        portable_result_names.add(portable_name)
        if hasattr(value, "to_csv"):
            artifacts[f"results/{result_name}.csv"] = value.to_csv(index=False).encode(
                "utf-8"
            )
        else:
            artifacts[f"results/{result_name}.json"] = _json_artifact_bytes(value)

    portable_figure_names: set[str] = set()
    for raw_name, payload in sorted((figures or {}).items(), key=lambda item: str(item[0])):
        figure_name = _safe_leaf(str(raw_name), "figure.bin")
        portable_name = figure_name.casefold()
        if portable_name in portable_figure_names:
            raise ValueError(f"Duplicate portable report figure name: {figure_name!r}.")
        portable_figure_names.add(portable_name)
        artifacts[f"figures/{figure_name}"] = bytes(payload)

    if audit is not None:
        audit_value = audit.to_json() if hasattr(audit, "to_json") else audit
        artifacts["audit/audit.json"] = (
            audit_value.encode("utf-8")
            if isinstance(audit_value, str)
            else _json_artifact_bytes(audit_value)
        )
    if provenance is not None:
        artifacts["provenance/provenance.json"] = _json_artifact_bytes(provenance)
    if report_markdown is not None:
        artifacts["reports/report_ru.md"] = report_markdown.encode("utf-8")
    if config_snapshot is not None:
        artifacts["config/processing_config.canonical.json"] = canonical_json_bytes(
            dict(config_snapshot)
        )
    return artifacts


def _records(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if hasattr(value, "to_dict"):
        try:
            records = value.to_dict(orient="records")
        except TypeError:
            records = value.to_dict()
        if isinstance(records, list):
            return [dict(record) for record in records if isinstance(record, Mapping)]
        if isinstance(records, Mapping):
            return [dict(records)]
    if isinstance(value, Mapping):
        return [dict(value)]
    if isinstance(value, (list, tuple)):
        result: list[dict[str, Any]] = []
        for item in value:
            if hasattr(item, "to_dict"):
                item = item.to_dict()
            if isinstance(item, Mapping):
                result.append(dict(item))
            elif item is not None:
                result.append({"message": str(item)})
        return result
    return [{"message": str(value)}]


def build_review_required_registry(
    *,
    passport_status: Mapping[str, Any] | None = None,
    qc_issues: Any = None,
    indicator_passports: Any = None,
    indicator_audit: Any = None,
    indicator_aggregation: Any = None,
    failures: Any = None,
    moduli: Any = None,
    group_comparisons: Any = None,
    additional: Any = None,
) -> list[dict[str, Any]]:
    """Collect explicit engineering-review states without searching report text."""

    registry: list[dict[str, Any]] = []

    def add(category: str, row: Mapping[str, Any], message: str) -> None:
        registry.append(
            {
                "category": category,
                "scope": str(
                    row.get("test_id")
                    or row.get("channel")
                    or row.get("group")
                    or row.get("entity_id")
                    or "project"
                ),
                "status": "review_required",
                "message": str(message),
            }
        )

    passport = dict(passport_status or {})
    missing = passport.get("missing") or []
    if missing:
        add(
            "project_passport",
            {},
            "Missing required project-passport fields: "
            + ", ".join(str(value) for value in missing),
        )

    for row in _records(qc_issues):
        level = str(row.get("severity") or row.get("level") or "").casefold()
        if level in {"warning", "error", "critical"} or row.get("blocks_processing") is True:
            add("qc_issue", row, str(row.get("message") or row.get("code") or level))

    for row in _records(indicator_passports):
        assignment = str(row.get("assignment_status") or "").casefold()
        verification = str(row.get("verification_status") or "").casefold()
        if assignment and assignment != "confirmed":
            add(
                "indicator_assignment",
                row,
                f"Indicator assignment status is {assignment}.",
            )
        if verification and verification != "valid_at_experiment":
            add(
                "indicator_verification",
                row,
                f"Indicator verification status is {verification}.",
            )

    for row in _records(indicator_audit):
        warning = row.get("warning")
        if warning not in {None, "", False}:
            add("indicator_processing", row, str(warning))

    acceptable_aggregation = {"ok", "not_applied_direct_settlement", "no_aggregation"}
    for row in _records(indicator_aggregation):
        status = str(row.get("aggregation_status") or "").casefold()
        if status and status not in acceptable_aggregation:
            add("indicator_aggregation", row, f"Indicator aggregation status is {status}.")

    for row in _records(failures):
        status = str(row.get("classification_status") or "").casefold()
        if status == "review_required":
            add(
                "failure_classification",
                row,
                str(row.get("classification_warning") or "Failure bounds require review."),
            )

    for row in _records(moduli):
        status = str(row.get("review_status") or "").casefold()
        if status == "review_required":
            add(
                "modulus_methodology",
                row,
                str(row.get("methodology_note") or "Modulus methodology requires review."),
            )

    for row in _records(group_comparisons):
        warning = row.get("pairing_warning") or row.get("warning")
        status = str(row.get("review_status") or "").casefold()
        if warning or status == "review_required":
            add("group_comparison", row, str(warning or "Group comparison requires review."))

    for row in _records(additional):
        add("additional", row, str(row.get("message") or "Engineering review required."))

    unique: dict[str, dict[str, Any]] = {}
    for item in registry:
        key = json.dumps(item, ensure_ascii=False, sort_keys=True, default=str)
        unique[key] = item
    return [unique[key] for key in sorted(unique)]


def build_formula_and_range_records(
    *,
    conversion_parameters: Any = None,
    indicator_passports: Any = None,
    modulus_profiles: Any = None,
    moduli: Any = None,
    pcr_results: Any = None,
) -> list[dict[str, Any]]:
    """Build literal, auditable formula/range records for both report renderers."""

    records: list[dict[str, Any]] = []

    for index, row in enumerate(_records(conversion_parameters), start=1):
        scope = str(row.get("test_id") or f"conversion_{index}")
        records.append(
            {
                "record_id": f"conversion_{scope}",
                "name": "Load and pressure conversion",
                "expression": str(row.get("formula") or "F_kN = load; p_kPa = F_kN / A"),
                "scope": scope,
                "units": "F_kN; p_kPa",
                "notes": json.dumps(row, ensure_ascii=False, sort_keys=True, default=str),
            }
        )

    for index, row in enumerate(_records(indicator_passports), start=1):
        scope = str(
            row.get("test_id")
            or row.get("channel")
            or row.get("serial_number")
            or f"indicator_{index}"
        )
        raw_mode = str(
            row.get("mode") or row.get("indicator_mode") or row.get("scale_mode") or ""
        ).strip().casefold()
        mode = _INDICATOR_MODE_ALIASES.get(raw_mode, raw_mode)
        division = row.get("division_mm", row.get("resolution_mm"))
        factor = row.get("correction_factor", row.get("calibration_factor"))
        zero = row.get("zero_correction_mm", row.get("zero_correction"))
        range_mm = row.get(
            "range_mm", row.get("scale_range_mm", row.get("dial_period_mm"))
        )
        initial_reading = row.get("initial_reading", row.get("initial_reading_mm"))
        initial_turn = row.get("initial_turn")
        maximum = row.get("max_increment_mm", row.get("max_step_mm"))
        reverse_tolerance = row.get("reverse_tolerance_mm")
        travel_range = row.get(
            "travel_range_mm", row.get("instrument_travel_mm")
        )

        formula_contract: dict[str, Any] = {
            "mode_raw": raw_mode or None,
            "mode_canonical": mode or None,
            "division_mm": division,
            "correction_factor_f": factor,
            "zero_correction_c0_mm": zero,
            "range_mm_R": range_mm,
            "initial_reading": initial_reading,
            "initial_turn": initial_turn,
            "max_increment_mm": maximum,
            "reverse_tolerance_mm": reverse_tolerance,
            "travel_range_mm": travel_range,
        }
        range_parts: list[str] = []
        if mode == "cumulative_settlement":
            sign = row.get("cumulative_sign")
            expression = (
                "v_i = sign * raw_i * f; "
                "delta_s_i = v_i for i=0, otherwise v_i - v_(i-1); "
                "s_i = v_i + c0"
            )
            range_parts.extend(
                [
                    "raw_i is a finite directly supplied cumulative displacement",
                    "turn_i = 0 and dial unwrapping is not applied",
                    "sign must be -1 or +1",
                ]
            )
            formula_contract.update(
                {
                    "reading_model": "ready_cumulative_settlement",
                    "cumulative_sign": sign,
                    "turn_policy": "fixed_zero_no_unwrapping",
                    "dial_range_used": False,
                }
            )
        elif mode in {
            "increasing",
            "increasing_wrapped",
            "decreasing",
            "decreasing_wrapped",
        }:
            direction = 1 if mode.startswith("increasing") else -1
            wrapped = mode.endswith("_wrapped")
            expression = (
                f"q = {direction:+d}; u_i = raw_i + turn_i * R; "
                "u_0 = initial_reading + initial_turn * R; "
                "delta_s_i = q * f * (u_i - u_(i-1)); "
                "s_i = q * f * (u_i - u_0) + c0"
            )
            range_parts.append(f"0 <= raw_i < R ({range_mm} mm)")
            if wrapped:
                turn_policy = "explicit integer or unique admissible inferred integer"
                range_parts.append(
                    "turn_i is explicit or the unique admissible inferred integer"
                )
            else:
                turn_policy = "explicit integer when supplied, otherwise initial_turn"
                range_parts.append(
                    "turn_i is explicit when supplied, otherwise initial_turn; no automatic wrap inference"
                )
            formula_contract.update(
                {
                    "reading_model": "wrapped_dial" if wrapped else "absolute_dial",
                    "scale_direction": (
                        "direct_increasing" if direction == 1 else "inverse_decreasing"
                    ),
                    "direction_multiplier_q": direction,
                    "turn_policy": turn_policy,
                    "dial_range_used": True,
                }
            )
            if maximum is not None:
                range_parts.append(f"abs(delta_s_i) <= {maximum} mm")
            if travel_range is not None:
                range_parts.append(f"abs(s_i - c0) <= {travel_range} mm")
        else:
            expression = str(
                row.get("formula")
                or "No indicator conversion formula asserted: mode is unsupported or unspecified."
            )
            range_parts.append("indicator mode and admissible range require review")
            formula_contract.update(
                {
                    "reading_model": "unsupported_or_unspecified",
                    "turn_policy": "not asserted",
                    "dial_range_used": None,
                }
            )

        if division is not None:
            range_parts.append(
                f"division_mm = {division}; off-division and below-resolution values are flagged"
            )
        if reverse_tolerance is not None:
            range_parts.append(
                f"reverse motion tolerance = {reverse_tolerance} mm"
            )
        records.append(
            {
                "record_id": f"indicator_{index:04d}",
                "name": f"Indicator conversion ({mode or 'mode unspecified'})",
                "expression": expression,
                "scope": scope,
                "units": "raw, u, v, delta_s, s, c0, R: mm; f, q, sign: dimensionless",
                "range": "; ".join(range_parts),
                "notes": json.dumps(
                    {
                        "formula_contract": formula_contract,
                        "passport": row,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                    default=str,
                ),
            }
        )

    for index, row in enumerate(_records(modulus_profiles), start=1):
        profile_id = str(row.get("profile_id") or f"profile_{index}")
        records.append(
            {
                "record_id": f"modulus_profile_{profile_id}",
                "name": "Conditional modulus formula",
                "expression": str(
                    row.get("formula")
                    or "E_stamp_app = (1 - nu^2) * K_shape * D * dp/ds"
                ),
                "scope": profile_id,
                "units": "kPa",
                "range": "explicit pressure range required"
                if row.get("requires_explicit_range")
                else "profile-defined range policy",
                "notes": json.dumps(row, ensure_ascii=False, sort_keys=True, default=str),
            }
        )

    for index, row in enumerate(_records(moduli), start=1):
        lower = row.get("p_min_kPa")
        upper = row.get("p_max_kPa")
        range_text = (
            f"{lower} <= p_kPa <= {upper}"
            if lower is not None and upper is not None
            else None
        )
        records.append(
            {
                "record_id": f"modulus_result_{index:04d}",
                "name": str(row.get("method") or "Modulus result range"),
                "expression": "E_stamp_app = (1 - nu^2) * K_shape * D * dp/ds",
                "scope": str(row.get("test_id") or "project"),
                "units": "kPa",
                "range": range_text,
                "notes": json.dumps(
                    {
                        "used_indices": row.get("used_indices"),
                        "review_status": row.get("review_status"),
                        "profile_id": row.get("profile_id"),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                    default=str,
                ),
            }
        )

    pcr_rows: list[dict[str, Any]] = []
    if isinstance(pcr_results, Mapping):
        for test_id, value in sorted(pcr_results.items(), key=lambda item: str(item[0])):
            if hasattr(value, "to_dict"):
                value = value.to_dict()
            row = dict(value) if isinstance(value, Mapping) else {"value": value}
            pcr_rows.append({"test_id": str(test_id), **row})
    else:
        pcr_rows = _records(pcr_results)
    for index, row in enumerate(pcr_rows, start=1):
        lower = row.get("pcr_ci_low")
        upper = row.get("pcr_ci_high")
        records.append(
            {
                "record_id": f"pcr_{index:04d}",
                "name": str(row.get("method") or "Segmented pcr fit"),
                "expression": "s = a + b1*p + hinge_delta*max(0, p-pcr)",
                "scope": str(row.get("test_id") or "project"),
                "units": "kPa",
                "range": (
                    f"{lower} <= pcr_kPa <= {upper}"
                    if lower is not None and upper is not None
                    else "confidence interval unavailable"
                ),
                "notes": json.dumps(row, ensure_ascii=False, sort_keys=True, default=str),
            }
        )
    return records


def _safe_artifact_path(value: str) -> str:
    original = str(value)
    rendered = unicodedata.normalize("NFC", original)
    if (
        "\\" in rendered
        or ":" in rendered
        or any(unicodedata.category(character).startswith("C") for character in rendered)
    ):
        raise ValueError(f"Unsafe report artifact path: {value!r}.")
    path = PurePosixPath(rendered)
    if (
        not rendered
        or path.is_absolute()
        or ".." in path.parts
        or any(part in {"", "."} for part in path.parts)
    ):
        raise ValueError(f"Unsafe report artifact path: {value!r}.")
    for part in path.parts:
        if part.endswith((" ", ".")):
            raise ValueError(f"Unsafe report artifact path: {value!r}.")
        if part.split(".", 1)[0].casefold() in _WINDOWS_RESERVED_NAMES:
            raise ValueError(f"Unsafe report artifact path: {value!r}.")
    return path.as_posix()


def _artifact_href(path: str) -> str:
    return "/".join(quote(part, safe="-._~") for part in PurePosixPath(path).parts)


def _artifact_role(path: str) -> str:
    pure = PurePosixPath(path)
    first = pure.parts[0].casefold()
    suffix = pure.suffix.casefold()
    if first == "source" and len(pure.parts) > 1:
        return "source"
    if first == "figures" or suffix in {".svg", ".png", ".pdf", ".jpg", ".jpeg"}:
        return "figure"
    if first == "results":
        return "result"
    if first == "data" and pure.name.casefold().startswith("raw"):
        return "raw_view"
    if first == "data":
        return "prepared_data"
    if "audit" in pure.name.casefold():
        return "audit"
    return "artifact"


def _artifact_records(
    artifacts: Mapping[str, bytes],
) -> tuple[dict[str, bytes], list[dict[str, Any]]]:
    payloads: dict[str, bytes] = {}
    records: list[dict[str, Any]] = []
    portable_paths: dict[str, str] = {}
    portable_parts: dict[tuple[str, ...], str] = {
        tuple(PurePosixPath(path).parts): path for path in _RESERVED_PATHS
    }
    for raw_path, raw_payload in sorted(artifacts.items(), key=lambda item: str(item[0])):
        path = _safe_artifact_path(str(raw_path))
        if path.casefold() in _RESERVED_PATHS:
            raise ValueError(f"Reserved report artifact path: {path!r}.")
        portable_key = path.casefold()
        if portable_key in portable_paths:
            raise ValueError(
                "Duplicate portable report artifact path: "
                f"{portable_paths[portable_key]!r} and {path!r}."
            )
        parts = tuple(part.casefold() for part in PurePosixPath(path).parts)
        for existing_parts, existing_path in portable_parts.items():
            common_length = min(len(parts), len(existing_parts))
            if (
                parts[:common_length] == existing_parts[:common_length]
                and len(parts) != len(existing_parts)
            ):
                raise ValueError(
                    "Portable report artifact file/directory collision: "
                    f"{existing_path!r} and {path!r}."
                )
        if not isinstance(raw_payload, (bytes, bytearray, memoryview)):
            raise TypeError(f"Artifact {path!r} must be bytes.")
        payload = bytes(raw_payload)
        if path in payloads:
            raise ValueError(f"Duplicate report artifact path: {path!r}.")
        payloads[path] = payload
        portable_paths[portable_key] = path
        portable_parts[parts] = path
        records.append(
            {
                "path": path,
                "href": _artifact_href(path),
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "media_type": _MEDIA_TYPES.get(
                    PurePosixPath(path).suffix.casefold(), "application/octet-stream"
                ),
                "role": _artifact_role(path),
            }
        )
    return payloads, records


def _zip_info(path: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(path, date_time=_ZIP_TIMESTAMP)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    return info


def _deterministic_zip(files: Mapping[str, bytes]) -> bytes:
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for path, payload in sorted(files.items()):
            archive.writestr(_zip_info(path), payload)
    return buffer.getvalue()


def _plot_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    supported = {".svg", ".png", ".pdf", ".jpg", ".jpeg"}
    return [
        dict(record)
        for record in records
        if PurePosixPath(str(record["path"])).suffix.casefold() in supported
    ]


def build_approval_report_package(
    *,
    artifacts: Mapping[str, bytes],
    metadata: dict[str, Any],
    raw,
    prepared,
    indicator_passports=None,
    indicator_audit=None,
    qc_issues=None,
    failures=None,
    pcr_results=None,
    moduli=None,
    group_comparisons=None,
    audit=None,
    provenance=None,
    methodology=None,
    formulas=None,
    display_rounding=None,
    review_required=None,
    result_tables: Mapping[str, Any] | None = None,
    title: str | None = None,
    scope: Mapping[str, Any] | None = None,
    require_exact_source: bool = True,
) -> ApprovalReportPackage:
    """Build HTML, XLSX, a SHA-256 manifest and a self-contained ZIP."""

    artifact_payloads, artifact_records = _artifact_records(artifacts)
    source_paths = [
        str(record["path"])
        for record in artifact_records
        if record.get("role") == "source"
    ]
    protocol_source_paths = [
        path
        for path in source_paths
        if len(PurePosixPath(path).parts) > 2
        and PurePosixPath(path).parts[0].casefold() == "source"
        and PurePosixPath(path).parts[1].casefold() == "protocol"
    ]
    if require_exact_source and not protocol_source_paths:
        raise ValueError(
            "Approval report package requires at least one exact source artifact "
            "for the protocol under source/protocol/."
        )
    review_registry = json.loads(
        json.dumps(list(review_required or []), ensure_ascii=False, default=str)
    )
    manifest_reference = {
        "path": "artifact_manifest.json",
        "href": "artifact_manifest.json",
        "bytes": None,
        "sha256": "excluded:self-referential-manifest",
        "media_type": "application/json",
        "role": "manifest",
    }
    common = {
        "metadata": metadata,
        "raw": raw,
        "prepared": prepared,
        "indicator_passports": indicator_passports,
        "indicator_audit": indicator_audit,
        "qc_issues": qc_issues,
        "failures": failures,
        "pcr_results": pcr_results,
        "moduli": moduli,
        "group_comparisons": group_comparisons,
        "plots": _plot_records(artifact_records),
        "artifacts": [*artifact_records, manifest_reference],
        "audit": audit,
        "provenance": provenance,
        "methodology": methodology,
        "formulas": formulas,
        "display_rounding": display_rounding,
        "review_required": review_registry,
        "result_tables": dict(result_tables or {}),
        "title": title,
    }
    html = build_html_report_package(**common)
    xlsx = build_xlsx_report_package(**common)
    if not isinstance(html, bytes) or not isinstance(xlsx, bytes):
        raise TypeError("Report renderers must return bytes.")

    report_records = [
        {
            "path": "report.html",
            "href": "report.html",
            "bytes": len(html),
            "sha256": hashlib.sha256(html).hexdigest(),
            "media_type": "text/html",
            "role": "report",
        },
        {
            "path": "report.xlsx",
            "href": "report.xlsx",
            "bytes": len(xlsx),
            "sha256": hashlib.sha256(xlsx).hexdigest(),
            "media_type": (
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            ),
            "role": "report",
        },
    ]
    manifest = {
        "schema_version": REPORT_PACKAGE_VERSION,
        "approval_status": "review_required" if review_registry else "ready_for_review",
        "review_required": review_registry,
        "scope": json.loads(
            json.dumps(dict(scope or {}), ensure_ascii=False, default=str)
        ),
        "source_contract": {
            "authoritative_representation": "exact_source_bytes",
            "required": bool(require_exact_source),
            "paths": sorted(source_paths),
            "protocol_paths": sorted(protocol_source_paths),
            "raw_tables_are_views": True,
        },
        "manifest_hash_exclusions": ["artifact_manifest.json"],
        "files": sorted(
            [*artifact_records, *report_records], key=lambda record: str(record["path"])
        ),
    }
    manifest_json = (
        json.dumps(
            manifest,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            separators=(",", ": "),
        )
        + "\n"
    ).encode("utf-8")
    archive_files = {
        **artifact_payloads,
        "report.html": html,
        "report.xlsx": xlsx,
        "artifact_manifest.json": manifest_json,
    }
    return ApprovalReportPackage(
        html=html,
        xlsx=xlsx,
        artifact_manifest_json=manifest_json,
        archive=_deterministic_zip(archive_files),
        manifest=manifest,
    )
