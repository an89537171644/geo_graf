"""Verify the artifacts produced by the SoilStamp CLI demo.

The helper deliberately uses only the Python standard library so it can run in
the Windows and Linux CI jobs before any optional inspection tools are present.
It exits with a non-zero status and prints every detected problem, rather than
stopping at the first missing file.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import io
import json
import math
import struct
import sys
import xml.etree.ElementTree as ElementTree
import zipfile
from pathlib import Path, PurePosixPath
from typing import Iterable


REQUIRED_ARTIFACTS = (
    "prepared.csv",
    "indicator_aggregation_results.csv",
    "failure_summary.csv",
    "pcr.json",
    "moduli.csv",
    "report_ru.md",
    "antonov.svg",
    "antonov.pdf",
    "antonov_600dpi.png",
    "reproducibility.zip",
)

REQUIRED_CSV_COLUMNS = {
    "prepared.csv": {"test_id", "sequence_no", "settlement_mm", "F_kN", "p_kPa"},
    "indicator_aggregation_results.csv": {
        "test_id",
        "row_index",
        "aggregation_method",
        "channels_required",
        "channels_used",
        "missing_channels",
        "aggregation_status",
    },
    "failure_summary.csv": {"test_id", "failure_reached"},
    "moduli.csv": {
        "test_id",
        "method",
        "E_stamp_app_kPa",
        "profile_id",
        "profile_version",
        "is_primary",
        "review_status",
        "p_range_source",
        "nu_source",
        "shape_factor_source",
        "used_indices",
        "methodology_note",
    },
}

MODULUS_METHOD_COLUMNS = {
    "profile_id",
    "profile_version",
    "is_primary",
    "review_status",
    "p_range_source",
    "nu_source",
    "shape_factor_source",
    "used_indices",
    "methodology_note",
}

ZIP_EXTERNAL_COPIES = {
    "data/prepared_machine.csv": "prepared.csv",
    "results/indicator_aggregation_results.csv": "indicator_aggregation_results.csv",
    "results/failure_summary.csv": "failure_summary.csv",
    "results/pcr.json": "pcr.json",
    "results/moduli.csv": "moduli.csv",
    "report_ru.md": "report_ru.md",
    "figures/antonov.svg": "antonov.svg",
    "figures/antonov.pdf": "antonov.pdf",
    "figures/antonov_600dpi.png": "antonov_600dpi.png",
}

TEXT_ARTIFACT_SUFFIXES = {".csv", ".json", ".md"}


class ArtifactVerificationError(RuntimeError):
    """Raised when one or more demo artifacts fail verification."""

    def __init__(self, output_dir: Path, problems: Iterable[str]) -> None:
        self.output_dir = output_dir
        self.problems = tuple(problems)
        details = "\n".join(f"- {problem}" for problem in self.problems)
        super().__init__(f"Demo artifact verification failed in {output_dir}:\n{details}")


def _read_bytes(path: Path, problems: list[str]) -> bytes | None:
    try:
        payload = path.read_bytes()
    except OSError as exc:
        problems.append(f"cannot read {path.name}: {exc}")
        return None
    if not payload:
        problems.append(f"{path.name} is empty")
        return None
    return payload


def _check_csv(
    path: Path, required_columns: set[str], problems: list[str]
) -> tuple[set[str], list[dict[str | None, str | list[str] | None]]] | None:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            columns = set(reader.fieldnames or ())
            missing = sorted(required_columns - columns)
            if missing:
                problems.append(f"{path.name} is missing columns: {', '.join(missing)}")
            rows = list(reader)
    except (OSError, UnicodeError, csv.Error) as exc:
        problems.append(f"{path.name} is not a readable UTF-8 CSV: {exc}")
        return None

    if not rows:
        problems.append(f"{path.name} has no data rows")
    elif "test_id" in columns and not any((row.get("test_id") or "").strip() for row in rows):
        problems.append(f"{path.name} has no non-empty test_id")
    if any(None in row for row in rows):
        problems.append(f"{path.name} has rows wider than its header")
    return columns, rows


def _parse_csv_bool(value: object) -> bool | None:
    normalized = str(value or "").strip().casefold()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    return None


def _parse_used_indices(value: object) -> list[object] | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        try:
            parsed = ast.literal_eval(text)
        except (SyntaxError, ValueError):
            return None
    return parsed if isinstance(parsed, list) else None


def _check_moduli_rows(
    columns: set[str],
    rows: list[dict[str | None, str | list[str] | None]],
    problems: list[str],
) -> None:
    if not MODULUS_METHOD_COLUMNS.issubset(columns):
        return
    for row_number, row in enumerate(rows, start=2):
        test_id = str(row.get("test_id") or "").strip() or "<empty>"
        method = str(row.get("method") or "").strip() or "<empty>"
        label = f"moduli.csv row {row_number} ({test_id}/{method})"
        empty_fields = sorted(
            name for name in MODULUS_METHOD_COLUMNS if not str(row.get(name) or "").strip()
        )
        if empty_fields:
            problems.append(
                f"{label} has empty methodology fields: {', '.join(empty_fields)}"
            )

        used_indices = _parse_used_indices(row.get("used_indices"))
        if used_indices is None:
            problems.append(f"{label} used_indices is not a list")
        elif any(
            isinstance(index, bool) or not isinstance(index, int) or index < 0
            for index in used_indices
        ):
            problems.append(f"{label} used_indices must contain non-negative integers")

        is_primary = _parse_csv_bool(row.get("is_primary"))
        if is_primary is None:
            problems.append(f"{label} has invalid is_primary boolean")
            continue
        if not is_primary:
            continue

        try:
            e_value = float(str(row.get("E_stamp_app_kPa") or "").strip())
        except ValueError:
            e_value = float("nan")
        if not math.isfinite(e_value) or e_value <= 0:
            problems.append(f"{label} is primary without a finite positive E_stamp_app_kPa")

        review_status = str(row.get("review_status") or "").strip().casefold()
        if review_status != "approved":
            problems.append(
                f"{label} is primary but review_status is not approved"
            )
        profile_id = str(row.get("profile_id") or "").strip().casefold()
        if "diagnostic" in profile_id or "unapproved" in profile_id:
            problems.append(f"{label} is primary with an unapproved profile")
        range_tokens = " ".join(
            str(row.get(name) or "").strip().casefold()
            for name in ("p_range_source", "p_range_origin")
        )
        diagnostic_range_markers = (
            "diagnostic",
            "full_curve",
            "whole_curve",
            "implicit",
            "unapproved",
        )
        if any(marker in range_tokens for marker in diagnostic_range_markers):
            problems.append(f"{label} is primary with a diagnostic pressure range")


def _load_json(path: Path, problems: list[str]) -> object | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        problems.append(f"{path.name} is not valid UTF-8 JSON: {exc}")
        return None


def _check_pcr(path: Path, problems: list[str]) -> None:
    payload = _load_json(path, problems)
    if payload is None:
        return
    if not isinstance(payload, dict) or not payload:
        problems.append("pcr.json must contain a non-empty object keyed by test_id")
        return
    for test_id, result in payload.items():
        if not isinstance(test_id, str) or not test_id.strip() or not isinstance(result, dict):
            problems.append("pcr.json contains an invalid test result entry")
            continue
        if not isinstance(result.get("method"), str) or not result["method"].strip():
            problems.append(f"pcr.json result {test_id!r} has no method")
        value = result.get("pcr_auto")
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            problems.append(f"pcr.json result {test_id!r} has invalid pcr_auto")


def _check_report(path: Path, problems: list[str]) -> None:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        problems.append(f"report_ru.md is not readable UTF-8 text: {exc}")
        return
    if not text.strip():
        problems.append("report_ru.md is empty")
    elif not any(line.lstrip().startswith("#") for line in text.splitlines()):
        problems.append("report_ru.md has no Markdown heading")


def _check_svg(path: Path, problems: list[str]) -> None:
    try:
        root = ElementTree.fromstring(path.read_bytes())
    except (OSError, ElementTree.ParseError) as exc:
        problems.append(f"antonov.svg is not valid XML: {exc}")
        return
    if root.tag.rsplit("}", 1)[-1].lower() != "svg":
        problems.append("antonov.svg root element is not <svg>")


def _check_pdf(path: Path, problems: list[str]) -> None:
    payload = _read_bytes(path, problems)
    if payload is None:
        return
    if not payload.startswith(b"%PDF-"):
        problems.append("antonov.pdf has no PDF signature")
    if b"%%EOF" not in payload[-1024:]:
        problems.append("antonov.pdf has no PDF end marker")


def _check_png(path: Path, problems: list[str]) -> None:
    payload = _read_bytes(path, problems)
    if payload is None:
        return
    signature = b"\x89PNG\r\n\x1a\n"
    if len(payload) < 24 or not payload.startswith(signature):
        problems.append("antonov_600dpi.png has no valid PNG header")
        return
    if payload[12:16] != b"IHDR":
        problems.append("antonov_600dpi.png does not start with an IHDR chunk")
        return
    width, height = struct.unpack(">II", payload[16:24])
    if width == 0 or height == 0:
        problems.append("antonov_600dpi.png has invalid dimensions")


def _safe_zip_name(name: str) -> bool:
    path = PurePosixPath(name)
    return not path.is_absolute() and ".." not in path.parts and "\\" not in name


def _same_artifact_payload(member: str, archived: bytes, external: bytes) -> bool:
    """Compare text portably while keeping binary artifacts byte-exact.

    ``Path.write_text`` uses the platform newline convention, whereas
    ``ZipFile.writestr`` preserves the LF bytes generated by pandas/JSON.  The
    files are semantically identical on Windows despite that byte difference.
    """

    suffix = PurePosixPath(member).suffix.lower()
    if suffix not in TEXT_ARTIFACT_SUFFIXES:
        return archived == external
    try:
        archived_text = archived.decode("utf-8-sig")
        external_text = external.decode("utf-8-sig")
    except UnicodeDecodeError:
        return False
    if suffix == ".csv":
        # pandas may already emit CRLF on Windows before Path.write_text applies
        # newline translation, resulting in CRCRLF and apparent blank records.
        # Compare parsed records so this platform detail does not invalidate a
        # reproducibility bundle, while changed cell content is still caught.
        archived_rows = [
            row
            for row in csv.reader(io.StringIO(archived_text, newline=""))
            if any(cell != "" for cell in row)
        ]
        external_rows = [
            row
            for row in csv.reader(io.StringIO(external_text, newline=""))
            if any(cell != "" for cell in row)
        ]
        return archived_rows == external_rows
    archived_text = archived_text.replace("\r\n", "\n").replace("\r", "\n")
    external_text = external_text.replace("\r\n", "\n").replace("\r", "\n")
    return archived_text == external_text


def _check_zip(path: Path, output_dir: Path, problems: list[str]) -> None:
    try:
        with zipfile.ZipFile(path) as archive:
            infos = archive.infolist()
            names = [info.filename for info in infos]
            name_set = set(names)
            if len(names) != len(name_set):
                problems.append("reproducibility.zip contains duplicate member names")
            unsafe = sorted(name for name in names if not _safe_zip_name(name))
            if unsafe:
                problems.append(
                    "reproducibility.zip contains unsafe member names: " + ", ".join(unsafe)
                )
            corrupt = archive.testzip()
            if corrupt is not None:
                problems.append(f"reproducibility.zip has a corrupt member: {corrupt}")

            required = set(ZIP_EXTERNAL_COPIES) | {"manifest.json"}
            missing = sorted(required - name_set)
            if missing:
                problems.append(
                    "reproducibility.zip is missing members: " + ", ".join(missing)
                )

            _check_zip_manifest(archive, name_set, problems)
            for member, external_name in ZIP_EXTERNAL_COPIES.items():
                if member not in name_set:
                    continue
                member_payload = archive.read(member)
                if not member_payload:
                    problems.append(f"reproducibility.zip member {member} is empty")
                    continue
                external_path = output_dir / external_name
                if external_path.is_file() and not _same_artifact_payload(
                    member, member_payload, external_path.read_bytes()
                ):
                    problems.append(
                        f"reproducibility.zip member {member} differs from {external_name}"
                    )
    except (OSError, zipfile.BadZipFile, RuntimeError) as exc:
        problems.append(f"reproducibility.zip is not a readable ZIP archive: {exc}")


def _check_zip_manifest(
    archive: zipfile.ZipFile, names: set[str], problems: list[str]
) -> None:
    if "manifest.json" not in names:
        return
    try:
        manifest = json.loads(archive.read("manifest.json").decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError, OSError, RuntimeError) as exc:
        problems.append(f"reproducibility.zip manifest.json is invalid: {exc}")
        return
    if not isinstance(manifest, dict) or not isinstance(manifest.get("files"), list):
        problems.append("reproducibility.zip manifest.json has no files list")
        return

    declared: dict[str, dict[str, object]] = {}
    for entry in manifest["files"]:
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            problems.append("reproducibility.zip manifest has an invalid file entry")
            continue
        member = entry["path"]
        if member in declared:
            problems.append(f"reproducibility.zip manifest declares {member} more than once")
            continue
        declared[member] = entry

    expected_declarations = names - {"manifest.json"}
    missing_declarations = sorted(expected_declarations - declared.keys())
    extra_declarations = sorted(declared.keys() - names)
    if missing_declarations:
        problems.append(
            "reproducibility.zip manifest omits members: " + ", ".join(missing_declarations)
        )
    if extra_declarations:
        problems.append(
            "reproducibility.zip manifest references missing members: "
            + ", ".join(extra_declarations)
        )

    for member in sorted(declared.keys() & names):
        payload = archive.read(member)
        entry = declared[member]
        if entry.get("bytes") != len(payload):
            problems.append(f"reproducibility.zip manifest byte count mismatch for {member}")
        digest = hashlib.sha256(payload).hexdigest()
        if entry.get("sha256") != digest:
            problems.append(f"reproducibility.zip manifest SHA-256 mismatch for {member}")


def verify_demo_artifacts(output_dir: str | Path) -> None:
    """Validate a CLI demo output directory or raise ``ArtifactVerificationError``."""

    directory = Path(output_dir)
    problems: list[str] = []
    if not directory.is_dir():
        raise ArtifactVerificationError(directory, ["output directory does not exist"])

    for name in REQUIRED_ARTIFACTS:
        path = directory / name
        if not path.is_file():
            problems.append(f"missing required artifact: {name}")
        elif path.stat().st_size == 0:
            problems.append(f"required artifact is empty: {name}")

    for name, required_columns in REQUIRED_CSV_COLUMNS.items():
        path = directory / name
        if path.is_file() and path.stat().st_size:
            csv_result = _check_csv(path, required_columns, problems)
            if name == "moduli.csv" and csv_result is not None:
                columns, rows = csv_result
                _check_moduli_rows(columns, rows, problems)
    pcr_path = directory / "pcr.json"
    if pcr_path.is_file() and pcr_path.stat().st_size:
        _check_pcr(pcr_path, problems)
    report_path = directory / "report_ru.md"
    if report_path.is_file() and report_path.stat().st_size:
        _check_report(report_path, problems)
    svg_path = directory / "antonov.svg"
    if svg_path.is_file() and svg_path.stat().st_size:
        _check_svg(svg_path, problems)
    pdf_path = directory / "antonov.pdf"
    if pdf_path.is_file() and pdf_path.stat().st_size:
        _check_pdf(pdf_path, problems)
    png_path = directory / "antonov_600dpi.png"
    if png_path.is_file() and png_path.stat().st_size:
        _check_png(png_path, problems)
    zip_path = directory / "reproducibility.zip"
    if zip_path.is_file() and zip_path.stat().st_size:
        _check_zip(zip_path, directory, problems)

    if problems:
        raise ArtifactVerificationError(directory, problems)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify the files produced by the SoilStamp CLI demo."
    )
    parser.add_argument("output_dir", type=Path, help="CLI --out directory to verify")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        verify_demo_artifacts(args.output_dir)
    except ArtifactVerificationError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(f"Demo artifacts verified: {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
