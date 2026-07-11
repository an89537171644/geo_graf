from __future__ import annotations

import hashlib
import json
import struct
import subprocess
import sys
import zipfile
import zlib
from pathlib import Path

import pytest

from scripts.verify_demo_artifacts import ArtifactVerificationError, verify_demo_artifacts


def _png(width: int = 1, height: int = 1) -> bytes:
    def chunk(kind: bytes, payload: bytes) -> bytes:
        body = kind + payload
        return struct.pack(">I", len(payload)) + body + struct.pack(">I", zlib.crc32(body))

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(b"\x00\x00\x00\x00\x00"))
        + chunk(b"IEND", b"")
    )


def _artifact_payloads() -> dict[str, bytes]:
    return {
        "prepared.csv": (
            "test_id,sequence_no,settlement_mm,F_kN,p_kPa\n"
            "DEMO-01,1,0.20,10.0,31.83\n"
        ).encode(),
        "failure_summary.csv": (
            "test_id,failure_reached,F_last_stable\nDEMO-01,False,10.0\n"
        ).encode(),
        "pcr.json": json.dumps(
            {
                "DEMO-01": {
                    "method": "continuous_segmented_hinge",
                    "pcr_auto": 123.4,
                }
            },
            indent=2,
        ).encode(),
        "moduli.csv": (
            "test_id,method,E_stamp_app_kPa\nDEMO-01,E_regression,12500.0\n"
        ).encode(),
        "report_ru.md": "# Демонстрационный отчёт\n\nРезультат воспроизводим.\n".encode(
            "utf-8"
        ),
        "antonov.svg": b'<svg xmlns="http://www.w3.org/2000/svg" width="1" height="1"/>',
        "antonov.pdf": b"%PDF-1.4\n1 0 obj<</Type/Catalog>>endobj\n%%EOF\n",
        "antonov_600dpi.png": _png(),
    }


def _write_demo_artifacts(
    directory: Path,
    *,
    zip_overrides: dict[str, bytes] | None = None,
    bad_manifest_hash_for: str | None = None,
) -> None:
    payloads = _artifact_payloads()
    for name, payload in payloads.items():
        (directory / name).write_bytes(payload)

    zip_payloads = {
        "data/prepared_machine.csv": payloads["prepared.csv"],
        "results/failure_summary.csv": payloads["failure_summary.csv"],
        "results/pcr.json": payloads["pcr.json"],
        "results/moduli.csv": payloads["moduli.csv"],
        "report_ru.md": payloads["report_ru.md"],
        "figures/antonov.svg": payloads["antonov.svg"],
        "figures/antonov.pdf": payloads["antonov.pdf"],
        "figures/antonov_600dpi.png": payloads["antonov_600dpi.png"],
    }
    zip_payloads.update(zip_overrides or {})
    manifest = {
        "scope": {"selected_test_ids": ["DEMO-01"]},
        "files": [
            {
                "path": name,
                "bytes": len(payload),
                "sha256": (
                    "0" * 64
                    if name == bad_manifest_hash_for
                    else hashlib.sha256(payload).hexdigest()
                ),
            }
            for name, payload in zip_payloads.items()
        ],
    }
    with zipfile.ZipFile(directory / "reproducibility.zip", "w") as archive:
        for name, payload in zip_payloads.items():
            archive.writestr(name, payload)
        archive.writestr("manifest.json", json.dumps(manifest).encode())


def test_verify_demo_artifacts_accepts_complete_valid_output(tmp_path: Path) -> None:
    _write_demo_artifacts(tmp_path)

    verify_demo_artifacts(tmp_path)


def test_verify_demo_artifacts_aggregates_missing_and_invalid_files(tmp_path: Path) -> None:
    _write_demo_artifacts(tmp_path)
    (tmp_path / "moduli.csv").unlink()
    (tmp_path / "pcr.json").write_text("[]", encoding="utf-8")
    (tmp_path / "antonov.pdf").write_bytes(b"not a PDF")

    with pytest.raises(ArtifactVerificationError) as caught:
        verify_demo_artifacts(tmp_path)

    message = str(caught.value)
    assert "missing required artifact: moduli.csv" in message
    assert "pcr.json must contain a non-empty object" in message
    assert "antonov.pdf has no PDF signature" in message
    assert "differs from pcr.json" in message


def test_verify_demo_artifacts_checks_zip_manifest_and_external_copies(tmp_path: Path) -> None:
    member = "results/failure_summary.csv"
    _write_demo_artifacts(
        tmp_path,
        zip_overrides={member: b"test_id,failure_reached\nSTALE,True\n"},
        bad_manifest_hash_for=member,
    )

    with pytest.raises(ArtifactVerificationError) as caught:
        verify_demo_artifacts(tmp_path)

    assert f"manifest SHA-256 mismatch for {member}" in str(caught.value)
    assert f"member {member} differs from failure_summary.csv" in str(caught.value)


def test_verify_demo_artifacts_script_is_cross_platform_cli(tmp_path: Path) -> None:
    _write_demo_artifacts(tmp_path)
    # On Windows, pandas CRLF passed through Path.write_text can become CRCRLF.
    # The archive still contains LF, so the verifier must compare CSV records
    # rather than platform-specific newline bytes.
    for name in ("prepared.csv", "failure_summary.csv", "moduli.csv"):
        path = tmp_path / name
        path.write_bytes(path.read_bytes().replace(b"\n", b"\r\r\n"))
    script = Path(__file__).parents[1] / "scripts" / "verify_demo_artifacts.py"

    result = subprocess.run(
        [sys.executable, str(script), str(tmp_path)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "Demo artifacts verified:" in result.stdout
