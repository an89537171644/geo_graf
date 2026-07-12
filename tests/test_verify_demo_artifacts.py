from __future__ import annotations

import hashlib
import io
import json
import struct
import subprocess
import sys
import zipfile
import zlib
from pathlib import Path

import pytest

from scripts.verify_demo_artifacts import ArtifactVerificationError, verify_demo_artifacts


_HTML_SECTIONS = (
    "project-passport",
    "raw-data",
    "prepared-data",
    "indicator-passports",
    "indicator-audit",
    "qc-issues",
    "failure-censoring",
    "pcr",
    "moduli",
    "group-comparison",
    "plots-index",
    "audit-trail",
    "provenance",
    "methodology",
)
_XLSX_SHEETS = (
    "Project passport",
    "Raw data",
    "Prepared data",
    "Indicator passports",
    "Indicator audit",
    "QC issues",
    "Failure censoring",
    "pcr",
    "Moduli",
    "Group comparison",
    "Plots index",
    "Audit trail",
    "Provenance",
    "Methodology",
)


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
    report_html = _report_html()
    report_xlsx = _report_xlsx()
    payloads = {
        "prepared.csv": (
            "test_id,sequence_no,settlement_mm,F_kN,p_kPa\n"
            "DEMO-01,1,0.20,10.0,31.83\n"
        ).encode(),
        "failure_summary.csv": (
            "test_id,failure_reached,failure_observed,interval_censored,right_censored,"
            "censoring_type,classification_status,lower_bound,upper_bound,F_last_stable\n"
            "DEMO-01,False,False,False,True,right_censored,ok,10.0,,10.0\n"
        ).encode(),
        "failure_analysis.json": json.dumps(
            {
                "contract_version": "failure-analysis/1.0",
                "summary_method": "none",
                "point_estimate": None,
                "n_tests": 1,
                "n_failure_observed": 0,
                "n_interval_censored": 0,
                "n_right_censored": 1,
                "n_indeterminate": 0,
            },
            indent=2,
        ).encode(),
        "curve_selections.csv": (
            "group,method,test_id,author,timestamp_utc,reason\n"
            "baseline,mean_curve,,,,\n"
        ).encode(),
        "plotted_curve_points.csv": (
            "group,curve_number,selection_method,axis_mode,x,y,n,measured_n,"
            "interpolated_n,draw_marker\n"
            "baseline,1,mean_curve,p-s,31.83,0.20,1,1,0,True\n"
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
        "failure_intervals.svg": (
            b'<svg xmlns="http://www.w3.org/2000/svg" width="1" height="1"/>'
        ),
        "failure_intervals.pdf": b"%PDF-1.4\n1 0 obj<</Type/Catalog>>endobj\n%%EOF\n",
        "failure_intervals_600dpi.png": _png(),
        "report.html": report_html,
        "report.xlsx": report_xlsx,
    }
    payloads.update(_approval_report_payloads(report_html, report_xlsx))
    return payloads


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
        "results/failure_analysis.json": payloads["failure_analysis.json"],
        "results/curve_selections.csv": payloads["curve_selections.csv"],
        "results/plotted_curve_points.csv": payloads["plotted_curve_points.csv"],
        "results/pcr.json": payloads["pcr.json"],
        "results/moduli.csv": payloads["moduli.csv"],
        "report_ru.md": payloads["report_ru.md"],
        "figures/antonov.svg": payloads["antonov.svg"],
        "figures/antonov.pdf": payloads["antonov.pdf"],
        "figures/antonov_600dpi.png": payloads["antonov_600dpi.png"],
        "figures/failure_intervals.svg": payloads["failure_intervals.svg"],
        "figures/failure_intervals.pdf": payloads["failure_intervals.pdf"],
        "figures/failure_intervals_600dpi.png": payloads[
            "failure_intervals_600dpi.png"
        ],
        "audit.json": b"[]\n",
        "provenance.json": b'{"program_version":"test"}\n',
        "analysis_run.json": b'{"parameters":{},"software":{}}\n',
    }
    with zipfile.ZipFile(io.BytesIO(payloads["approval_report.zip"])) as approval:
        for info in approval.infolist():
            if not info.is_dir():
                zip_payloads[f"approval/{info.filename}"] = approval.read(info)
    zip_payloads["approval/approval_report.zip"] = payloads["approval_report.zip"]
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


def _report_html() -> bytes:
    sections = "".join(
        f'<section id="{section_id}"><h2>{section_id}</h2>'
        "<p>&lt;unsafe-token&gt; &amp; escaped</p></section>"
        for section_id in _HTML_SECTIONS
    )
    return (
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
        "<meta http-equiv=\"Content-Security-Policy\" "
        "content=\"default-src 'none'; style-src 'unsafe-inline'\">"
        "<title>Demo report</title><style>body{font-family:sans-serif}</style>"
        "</head><body><h1>Demo</h1>"
        "<p>Standalone UTF-8 report; no JavaScript or remote assets.</p>"
        '<aside class="review-banner review-required">Engineering review required.</aside>'
        '<a href="source/protocol/demo.csv">exact source</a>'
        '<a href="artifact_manifest.json">artifact manifest</a>'
        f"{sections}</body></html>"
    ).encode("utf-8")


def _report_xlsx(
    *,
    formula: bool = False,
    omit_sheet: str | None = None,
    hyperlink_target: str = "source/protocol/demo.csv",
) -> bytes:
    sheets = [name for name in _XLSX_SHEETS if name != omit_sheet]
    workbook_sheets = "".join(
        f'<sheet name="{name}" sheetId="{index}" r:id="rId{index}"/>'
        for index, name in enumerate(sheets, start=1)
    )
    workbook = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        f"<sheets>{workbook_sheets}</sheets></workbook>"
    ).encode()
    relationships = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        + "".join(
            '<Relationship Id="rId{0}" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
            'Target="worksheets/sheet{0}.xml"/>'.format(index)
            for index in range(1, len(sheets) + 1)
        )
        + "</Relationships>"
    ).encode()
    content_types = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/xl/workbook.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
        + "".join(
            '<Override PartName="/xl/worksheets/sheet{0}.xml" '
            'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'.format(
                index
            )
            for index in range(1, len(sheets) + 1)
        )
        + "</Types>"
    ).encode()
    package_rels = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
        'Target="xl/workbook.xml"/></Relationships>'
    ).encode()
    cell = "<c r=\"A1\"><f>1+1</f><v>2</v></c>" if formula else (
        '<c r="A1" t="inlineStr"><is><t>machine value</t></is></c>'
    )
    worksheet = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        f'<sheetData><row r="1">{cell}</row></sheetData>'
        '<hyperlinks><hyperlink ref="A1" r:id="rId1"/></hyperlinks></worksheet>'
    ).encode()
    hyperlink_relationships = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink" '
        f'Target="{hyperlink_target}" TargetMode="External"/></Relationships>'
    ).encode()
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", content_types)
        archive.writestr("_rels/.rels", package_rels)
        archive.writestr("xl/workbook.xml", workbook)
        archive.writestr("xl/_rels/workbook.xml.rels", relationships)
        for index in range(1, len(sheets) + 1):
            archive.writestr(f"xl/worksheets/sheet{index}.xml", worksheet)
            archive.writestr(
                f"xl/worksheets/_rels/sheet{index}.xml.rels",
                hyperlink_relationships,
            )
    return output.getvalue()


def _approval_report_payloads(report_html: bytes, report_xlsx: bytes) -> dict[str, bytes]:
    source_path = "source/protocol/demo.csv"
    source_payload = b"force,settlement\n10,0.2\n"
    entries = [
        {
            "path": source_path,
            "href": source_path,
            "bytes": len(source_payload),
            "sha256": hashlib.sha256(source_payload).hexdigest(),
            "media_type": "text/csv",
            "role": "source",
        },
        {
            "path": "report.html",
            "href": "report.html",
            "bytes": len(report_html),
            "sha256": hashlib.sha256(report_html).hexdigest(),
            "media_type": "text/html",
            "role": "report",
        },
        {
            "path": "report.xlsx",
            "href": "report.xlsx",
            "bytes": len(report_xlsx),
            "sha256": hashlib.sha256(report_xlsx).hexdigest(),
            "media_type": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            "role": "report",
        },
    ]
    manifest = {
        "schema_version": "approval-report-package/1.0",
        "approval_status": "ready_for_review",
        "review_required": [],
        "scope": {"selected_test_ids": ["DEMO-01"]},
        "source_contract": {
            "authoritative_representation": "exact_source_bytes",
            "required": True,
            "paths": [source_path],
            "raw_tables_are_views": True,
        },
        "manifest_hash_exclusions": ["artifact_manifest.json"],
        "files": entries,
    }
    manifest_json = (json.dumps(manifest, sort_keys=True, indent=2) + "\n").encode()
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(source_path, source_payload)
        archive.writestr("report.html", report_html)
        archive.writestr("report.xlsx", report_xlsx)
        archive.writestr("artifact_manifest.json", manifest_json)
    return {
        "artifact_manifest.json": manifest_json,
        "approval_report.zip": output.getvalue(),
    }


def _rewrite_zip_member(payload: bytes, member: str, replacement: bytes) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(payload)) as source, zipfile.ZipFile(
        output, "w", compression=zipfile.ZIP_DEFLATED
    ) as target:
        for info in source.infolist():
            target.writestr(
                info.filename,
                replacement if info.filename == member else source.read(info),
            )
    return output.getvalue()


def test_verify_demo_artifacts_rejects_implicit_failure_point_estimate(
    tmp_path: Path,
) -> None:
    invalid = json.loads(_artifact_payloads()["failure_analysis.json"])
    invalid["summary_method"] = "midpoint_mean"
    invalid["point_estimate"] = 125.0
    _write_demo_artifacts(
        tmp_path,
        artifact_overrides={
            "failure_analysis.json": json.dumps(invalid).encode("utf-8")
        },
    )

    with pytest.raises(ArtifactVerificationError) as caught:
        verify_demo_artifacts(tmp_path)

    assert "summary_method must be none" in str(caught.value)
    assert "must not contain an implicit point estimate" in str(caught.value)


def test_verify_demo_artifacts_rejects_inconsistent_failure_and_plot_points(
    tmp_path: Path,
) -> None:
    invalid_failure = (
        "test_id,failure_reached,failure_observed,interval_censored,right_censored,"
        "censoring_type,classification_status,lower_bound,upper_bound,F_last_stable\n"
        "DEMO-01,True,True,True,True,right_censored,ok,20.0,10.0,20.0\n"
    ).encode()
    invalid_points = (
        "group,curve_number,selection_method,axis_mode,x,y,n,measured_n,"
        "interpolated_n,draw_marker\n"
        "baseline,1,median_curve,p-s,garbage,nan,-7,5,9,True\n"
    ).encode()
    invalid_analysis = json.dumps(
        {
            "contract_version": "failure-analysis/1.0",
            "summary_method": "none",
            "point_estimate": None,
            "n_tests": 1,
            "n_failure_observed": 1,
            "n_interval_censored": 0,
            "n_right_censored": 1,
            "n_indeterminate": 0,
        }
    ).encode()
    _write_demo_artifacts(
        tmp_path,
        artifact_overrides={
            "failure_summary.csv": invalid_failure,
            "failure_analysis.json": invalid_analysis,
            "plotted_curve_points.csv": invalid_points,
        },
    )

    with pytest.raises(ArtifactVerificationError) as caught:
        verify_demo_artifacts(tmp_path)

    message = str(caught.value)
    assert "right-censored event is marked observed" in message
    assert "invalid right-censoring bounds" in message
    assert "non-finite x" in message
    assert "violates measured_n + interpolated_n = n" in message
    assert "disagrees with curve selection" in message


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


def test_verify_demo_artifacts_requires_safe_standalone_html_sections(
    tmp_path: Path,
) -> None:
    invalid_html = _report_html().replace(
        b'<section id="methodology">',
        b'<script src="https://example.invalid/payload.js"></script>',
    ).replace(b"&lt;unsafe-token&gt;", b"<unsafe-token>", 1)
    _write_demo_artifacts(
        tmp_path,
        artifact_overrides={"report.html": invalid_html},
    )

    with pytest.raises(ArtifactVerificationError) as caught:
        verify_demo_artifacts(tmp_path)

    message = str(caught.value)
    assert "report.html is missing sections: methodology" in message
    assert "report.html contains active or remote content" in message
    assert "report.html contains unescaped or unsupported elements:" in message
    assert "<unsafe-token>" in message


def test_verify_demo_artifacts_requires_all_xlsx_sheets_and_no_formulas(
    tmp_path: Path,
) -> None:
    invalid_xlsx = _report_xlsx(formula=True, omit_sheet="Methodology")
    _write_demo_artifacts(
        tmp_path,
        artifact_overrides={"report.xlsx": invalid_xlsx},
    )

    with pytest.raises(ArtifactVerificationError) as caught:
        verify_demo_artifacts(tmp_path)

    message = str(caught.value)
    assert "report.xlsx is missing logical sheets: Methodology" in message
    assert "report.xlsx must contain exactly 14 logical sheets" in message
    assert "report.xlsx contains executable formula cells" in message


@pytest.mark.parametrize(
    "target",
    [
        "../escape.csv",
        "%2e%2e/escape.csv",
        "%252e%252e/escape.csv",
        "/rooted.csv",
        "\\\\server\\share\\file.csv",
        "https://example.invalid/file.csv",
    ],
)
def test_verify_demo_artifacts_rejects_unsafe_xlsx_hyperlink_targets(
    tmp_path: Path,
    target: str,
) -> None:
    invalid_xlsx = _report_xlsx(hyperlink_target=target)
    _write_demo_artifacts(
        tmp_path,
        artifact_overrides={"report.xlsx": invalid_xlsx},
    )

    with pytest.raises(ArtifactVerificationError) as caught:
        verify_demo_artifacts(tmp_path)

    assert "report.xlsx has unsafe hyperlink target" in str(caught.value)


def test_verify_demo_artifacts_enforces_manifest_roles_and_exact_source_contract(
    tmp_path: Path,
) -> None:
    manifest = json.loads(_artifact_payloads()["artifact_manifest.json"])
    manifest["source_contract"]["required"] = False
    manifest["files"][0]["role"] = "artifact"
    invalid_manifest = (json.dumps(manifest, sort_keys=True, indent=2) + "\n").encode()
    _write_demo_artifacts(
        tmp_path,
        artifact_overrides={"artifact_manifest.json": invalid_manifest},
    )

    with pytest.raises(ArtifactVerificationError) as caught:
        verify_demo_artifacts(tmp_path)

    message = str(caught.value)
    assert "inconsistent source roles" in message
    assert "exact source is not required" in message
    assert "source contract paths do not match source roles" in message


def test_verify_demo_artifacts_rejects_unsafe_manifest_href(tmp_path: Path) -> None:
    manifest = json.loads(_artifact_payloads()["artifact_manifest.json"])
    manifest["files"][0]["href"] = "%252e%252e/escape.csv"
    invalid_manifest = (json.dumps(manifest, sort_keys=True, indent=2) + "\n").encode()
    _write_demo_artifacts(
        tmp_path,
        artifact_overrides={"artifact_manifest.json": invalid_manifest},
    )

    with pytest.raises(ArtifactVerificationError) as caught:
        verify_demo_artifacts(tmp_path)

    assert "has unsafe href '%252e%252e/escape.csv'" in str(caught.value)


def test_verify_demo_artifacts_requires_html_links_to_resolve_in_approval_subtree(
    tmp_path: Path,
) -> None:
    invalid_html = _report_html().replace(
        b"</body>", b'<a href="missing.bin">missing</a></body>'
    )
    _write_demo_artifacts(
        tmp_path,
        artifact_overrides={"report.html": invalid_html},
    )

    with pytest.raises(ArtifactVerificationError) as caught:
        verify_demo_artifacts(tmp_path)

    message = str(caught.value)
    assert "approval_report.zip report.html href 'missing.bin' does not resolve" in message
    assert "does not resolve in approval subtree" in message


def test_verify_demo_artifacts_checks_all_approval_hashes_and_external_copies(
    tmp_path: Path,
) -> None:
    payloads = _artifact_payloads()
    corrupted_approval = _rewrite_zip_member(
        payloads["approval_report.zip"],
        "source/protocol/demo.csv",
        b"changed exact source\n",
    )
    _write_demo_artifacts(
        tmp_path,
        artifact_overrides={"approval_report.zip": corrupted_approval},
    )

    with pytest.raises(ArtifactVerificationError) as caught:
        verify_demo_artifacts(tmp_path)

    message = str(caught.value)
    assert "approval_report.zip byte count mismatch for source/protocol/demo.csv" in message
    assert "approval_report.zip SHA-256 mismatch for source/protocol/demo.csv" in message


def test_verify_demo_artifacts_rejects_unsafe_duplicate_approval_members(
    tmp_path: Path,
) -> None:
    payloads = _artifact_payloads()
    output = io.BytesIO()
    with pytest.warns(UserWarning, match="Duplicate name"):
        with zipfile.ZipFile(
            io.BytesIO(payloads["approval_report.zip"])
        ) as source, zipfile.ZipFile(
            output, "w", compression=zipfile.ZIP_DEFLATED
        ) as target:
            for info in source.infolist():
                target.writestr(info.filename, source.read(info))
            target.writestr("report.html", payloads["report.html"])
            target.writestr("../escape.txt", b"unsafe")
            target.writestr("source", b"ancestor collision")
    _write_demo_artifacts(
        tmp_path,
        artifact_overrides={"approval_report.zip": output.getvalue()},
    )

    with pytest.raises(ArtifactVerificationError) as caught:
        verify_demo_artifacts(tmp_path)

    message = str(caught.value)
    assert "approval_report.zip contains duplicate member names" in message
    assert "approval_report.zip contains unsafe member names: ../escape.txt" in message
    assert "approval_report.zip contains file/directory ancestor collisions" in message


def test_verify_demo_artifacts_rejects_reproducibility_ancestor_collision(
    tmp_path: Path,
) -> None:
    _write_demo_artifacts(
        tmp_path,
        zip_overrides={"approval/source": b"ancestor collision"},
    )

    with pytest.raises(ArtifactVerificationError) as caught:
        verify_demo_artifacts(tmp_path)

    assert (
        "reproducibility.zip contains file/directory ancestor collisions"
        in str(caught.value)
    )


def test_verify_demo_artifacts_requires_exact_reproducibility_report_copies(
    tmp_path: Path,
) -> None:
    _write_demo_artifacts(
        tmp_path,
        zip_overrides={"approval/artifact_manifest.json": b"{}\n"},
    )

    with pytest.raises(ArtifactVerificationError) as caught:
        verify_demo_artifacts(tmp_path)

    assert (
        "reproducibility.zip member approval/artifact_manifest.json differs from exact external copy"
        in str(caught.value)
    )


def test_verify_demo_artifacts_script_is_cross_platform_cli(tmp_path: Path) -> None:
    _write_demo_artifacts(tmp_path)
    # On Windows, pandas CRLF passed through Path.write_text can become CRCRLF.
    # The archive still contains LF, so the verifier must compare CSV records
    # rather than platform-specific newline bytes.
    for name in (
        "prepared.csv",
        "failure_summary.csv",
        "curve_selections.csv",
        "plotted_curve_points.csv",
        "moduli.csv",
    ):
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
