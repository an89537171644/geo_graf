from __future__ import annotations

import hashlib
import re

import pandas as pd

from soilstamp.report_html import build_html_report_package


SECTIONS = [
    "Project passport",
    "Raw data",
    "Prepared data",
    "Indicator passports",
    "Indicator audit",
    "QC issues",
    "Failure/censoring",
    "pcr",
    "Moduli",
    "Group comparison",
    "Plots index",
    "Audit trail",
    "Provenance",
    "Methodology",
]


def _minimal_report(**overrides: object) -> bytes:
    arguments: dict[str, object] = {
        "metadata": {"project": "Test project", "project_passport": {"project_id": "P-1"}},
        "raw": pd.DataFrame({"test_id": ["T-1"], "load": ["001.00"]}),
        "prepared": pd.DataFrame({"test_id": ["T-1"], "F_kN": [1.0]}),
        "indicator_passports": [{"serial_number": "SN-1", "range_mm": 10.0}],
        "indicator_audit": [{"test_id": "T-1", "increment_mm": 0.01}],
        "qc_issues": [],
        "failures": [{"test_id": "T-1", "censoring_type": "right"}],
        "pcr_results": [{"test_id": "T-1", "pcr_auto": 75.0}],
        "moduli": [{"test_id": "T-1", "E_stamp_app_kPa": 12000.0}],
        "group_comparisons": [{"baseline_group": "A", "reinforced_group": "B"}],
        "plots": [],
        "artifacts": [],
        "audit": [{"event_id": 1, "action": "import"}],
        "provenance": {"input_file_sha256": "a" * 64, "program_version": "test"},
        "methodology": {"profile": "antonov_round_stamp_v1"},
    }
    arguments.update(overrides)
    return build_html_report_package(**arguments)  # type: ignore[arg-type]


def test_html_is_standalone_utf8_printable_and_has_exact_section_contract() -> None:
    report = _minimal_report().decode("utf-8")

    assert report.startswith("<!doctype html>\n<html lang=\"en\">")
    assert '<meta charset="utf-8">' in report
    assert "@media print" in report
    assert "<script" not in report.casefold()
    assert "<link" not in report.casefold()
    assert "@import" not in report.casefold()
    assert re.findall(r"<h2>(.*?)</h2>", report) == SECTIONS


def test_html_escapes_title_cells_and_urls_and_blocks_non_relative_links() -> None:
    report = _minimal_report(
        title='"><script>alert("title")</script>',
        raw=pd.DataFrame(
            {"comment": ['<img src=x onerror="alert(1)">& exact'], "test_id": ["T<1"]}
        ),
        artifacts=[
            {
                "path": '"><svg onload=alert(1)>',
                "href": "javascript:alert(1)",
                "bytes": 12,
                "sha256": "b" * 64,
                "media_type": "image/svg+xml",
            },
            {
                "path": "outside.txt",
                "href": "../outside.txt",
                "bytes": 2,
                "sha256": "c" * 64,
            },
        ],
    ).decode("utf-8")

    assert "<script>alert" not in report
    assert "&lt;script&gt;alert(&quot;" in report
    assert "<img src=x" not in report
    assert "&lt;img src=x onerror=&quot;" in report
    assert 'href="javascript:' not in report.casefold()
    assert 'href="../outside.txt"' not in report
    assert report.count("Unsafe link blocked") == 2


def test_html_blocks_encoded_traversal_schemes_separators_controls_and_double_encoding() -> None:
    unsafe_hrefs = [
        "%2e%2e/outside.txt",
        "%2E%2E%2Foutside.txt",
        "%252e%252e/outside.txt",
        "javascript%3Aalert(1)",
        "figures%2Foutside.svg",
        "figures/%5c..%5csecret.txt",
        "figures/%00secret.txt",
        "safe/%252Foutside.txt",
        "&#x2e;&#x2e;/outside.txt",
        "figures/%ff.txt",
        "figures/antonov.svg?download=1",
        "figures/antonov.svg#page",
        "figures/antonov.svg?",
        "figures/antonov.svg#",
        "C%3A/secret.txt",
        "figures/name%3Astream.svg",
    ]
    report = _minimal_report(
        artifacts=[
            {
                "path": f"artifact-{index}.bin",
                "href": href,
                "bytes": 1,
                "sha256": f"{index:x}" * 64,
            }
            for index, href in enumerate(unsafe_hrefs)
        ]
    ).decode("utf-8")

    assert report.count("Unsafe link blocked") == len(unsafe_hrefs)
    for href in unsafe_hrefs:
        assert f'href="{href}"' not in report


def test_html_preserves_safe_utf8_and_literal_percent_artifact_links() -> None:
    hrefs = [
        "figures/%D0%B3%D1%80%D0%B0%D1%84%D0%B8%D0%BA%20A.svg",
        "figures/100%25.svg",
        "figures/antonov.svg",
    ]
    report = _minimal_report(
        artifacts=[
            {
                "path": f"artifact-{index}.svg",
                "href": href,
                "bytes": 1,
                "sha256": f"{index + 1:x}" * 64,
            }
            for index, href in enumerate(hrefs)
        ]
    ).decode("utf-8")

    assert "Unsafe link blocked" not in report
    for href in hrefs:
        assert f'href="{href}"' in report


def test_raw_source_strings_remain_exact_and_machine_display_is_separate() -> None:
    raw = pd.DataFrame(
        {
            "load": ["000123.4500", "  +1,2300  "],
            "stage": ["01", "1e-03"],
            "empty": ["", "   "],
        }
    )
    prepared = pd.DataFrame({"value": [1.2345678901234567]})

    report = _minimal_report(
        raw=raw,
        prepared=prepared,
        display_rounding={"Prepared data.value": 2},
    ).decode("utf-8")

    for exact in ("000123.4500", "  +1,2300  ", "01", "1e-03", "   "):
        assert exact in report
    raw_digest = hashlib.sha256("000123.4500".encode()).hexdigest()
    assert f'data-raw-sha256="{raw_digest}"' in report
    assert "Representation notice" in report
    assert "exact source artifact listed in the manifest is authoritative" in report
    assert '<span class="display-value">1.23</span>' in report
    assert '<code class="machine-value">1.2345678901234567</code>' in report


def test_review_registry_formulas_and_ranges_are_explicitly_visible() -> None:
    report = _minimal_report(
        metadata={
            "project": "Review test",
            "project_passport": {"approval_status": "review_required"},
        },
        moduli=[
            {
                "test_id": "T-1",
                "review_status": "review_required",
                "p_min_kPa": 25.0,
                "p_max_kPa": 75.0,
            }
        ],
        formulas={
            "pressure_formula": "p = F / A",
            "approved_pressure_range_kPa": "25 <= p <= 75",
        },
    ).decode("utf-8")

    assert '<aside class="review-banner review-required">' in report
    assert "Review-required registry" in report
    assert report.count("REVIEW REQUIRED") >= 3
    assert "pressure_formula" in report
    assert "p = F / A" in report
    assert 'data-kind="formula"' in report
    assert "approved_pressure_range_kPa" in report
    assert "25 &lt;= p &lt;= 75" in report
    assert 'data-kind="range"' in report


def test_artifact_manifest_uses_integer_byte_count_and_is_deterministic() -> None:
    payload = b"<svg xmlns='http://www.w3.org/2000/svg'/>"
    digest = hashlib.sha256(payload).hexdigest()
    artifact = {
        "path": "figures/antonov.svg",
        "href": "figures/antonov.svg",
        "bytes": len(payload),
        "sha256": digest,
        "media_type": "image/svg+xml",
    }
    kwargs = {
        "artifacts": [artifact],
        "plots": [artifact],
        "provenance": {"source_tree_sha256": "d" * 64, "program_version": "test"},
    }

    first = _minimal_report(**kwargs)
    second = _minimal_report(**kwargs)
    report = first.decode("utf-8")

    assert first == second
    assert 'href="figures/antonov.svg"' in report
    assert digest in report
    assert f">{len(payload)}</td>" in report
    assert "image/svg+xml" in report
    assert "hash_mismatch" not in report


def test_result_tables_fallback_populates_named_sections_without_extra_sections() -> None:
    report = build_html_report_package(
        metadata={},
        raw=pd.DataFrame({"raw": ["01"]}),
        prepared=pd.DataFrame({"value": [1.0]}),
        result_tables={
            "indicator_calibration_parameters": [{"serial_number": "SN-77"}],
            "indicator_processing_audit": [{"event_type": "zero_crossing"}],
            "validation_issues": [{"code": "units"}],
            "failure_summary": [{"censoring_type": "interval"}],
            "pcr_results": [{"pcr_auto": 50.0}],
            "moduli": [{"E_stamp_app_kPa": 10000.0}],
            "group_comparisons": [{"pairing_status": "paired"}],
        },
    ).decode("utf-8")

    assert re.findall(r"<h2>(.*?)</h2>", report) == SECTIONS
    for expected in (
        "SN-77",
        "zero_crossing",
        "units",
        "interval",
        "50.0",
        "10000.0",
        "paired",
    ):
        assert expected in report
