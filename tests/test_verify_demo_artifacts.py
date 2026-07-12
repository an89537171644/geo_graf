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
        "indicator_aggregation_results.csv": (
            "test_id,row_index,aggregation_method,channels_required,channels_used,"
            "missing_channels,aggregation_status\n"
            'DEMO-01,0,primary_channel,"[""indicator_1""]",'
            '"[""indicator_1""]",[],ok\n'
        ).encode(),
        "moduli.csv": (
            "test_id,method,E_stamp_app_kPa,profile_id,profile_version,is_primary,"
            "review_status,p_range_source,nu_source,shape_factor_source,used_indices,"
            "methodology_note\n"
            "DEMO-01,E_regression,12500.0,antonov_round_stamp_v1,1,True,approved,"
            'explicit,profile,profile,"[0, 1, 2]",engineer-confirmed range\n'
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
    artifact_overrides: dict[str, bytes] | None = None,
    zip_overrides: dict[str, bytes] | None = None,
    bad_manifest_hash_for: str | None = None,
) -> None:
    payloads = _artifact_payloads()
    payloads.update(artifact_overrides or {})
    for name, payload in payloads.items():
        (directory / name).write_bytes(payload)

    zip_payloads = {
        "data/prepared_machine.csv": payloads["prepared.csv"],
        "results/indicator_aggregation_results.csv": payloads[
            "indicator_aggregation_results.csv"
        ],
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


def test_verify_demo_artifacts_requires_modulus_method_contract(tmp_path: Path) -> None:
    legacy_moduli = (
        "test_id,method,E_stamp_app_kPa\nDEMO-01,E_regression,12500.0\n"
    ).encode()
    _write_demo_artifacts(
        tmp_path,
        artifact_overrides={"moduli.csv": legacy_moduli},
    )

    with pytest.raises(ArtifactVerificationError) as caught:
        verify_demo_artifacts(tmp_path)

    message = str(caught.value)
    assert "moduli.csv is missing columns:" in message
    assert "profile_id" in message
    assert "used_indices" in message


def test_verify_demo_artifacts_rejects_invalid_primary_modulus(tmp_path: Path) -> None:
    invalid_moduli = (
        "test_id,method,E_stamp_app_kPa,profile_id,profile_version,is_primary,"
        "review_status,p_range_source,nu_source,shape_factor_source,used_indices,"
        "methodology_note\n"
        "DEMO-01,E_regression,12500.0,diagnostic_unapproved_v1,1,True,"
        "review_required,diagnostic_full_curve,profile,profile,not-a-list,diagnostic\n"
    ).encode()
    _write_demo_artifacts(
        tmp_path,
        artifact_overrides={"moduli.csv": invalid_moduli},
    )

    with pytest.raises(ArtifactVerificationError) as caught:
        verify_demo_artifacts(tmp_path)

    message = str(caught.value)
    assert "used_indices is not a list" in message
    assert "review_status is not approved" in message
    assert "primary with an unapproved profile" in message
    assert "primary with a diagnostic pressure range" in message


@pytest.mark.parametrize("value", ["nan", "inf", "0", "-1"])
def test_verify_demo_artifacts_rejects_nonpositive_or_nonfinite_primary_e(
    tmp_path: Path, value: str
) -> None:
    invalid_moduli = _artifact_payloads()["moduli.csv"].replace(
        b"12500.0", value.encode()
    )
    _write_demo_artifacts(
        tmp_path,
        artifact_overrides={"moduli.csv": invalid_moduli},
    )

    with pytest.raises(ArtifactVerificationError) as caught:
        verify_demo_artifacts(tmp_path)

    assert "primary without a finite positive E_stamp_app_kPa" in str(caught.value)


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
