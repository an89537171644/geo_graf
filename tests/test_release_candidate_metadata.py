from __future__ import annotations

import json
import re
from pathlib import Path

from soilstamp import __version__
from soilstamp.schema import VERSION


ROOT = Path(__file__).resolve().parents[1]
RC_VERSION = "0.5.0rc1"
FINAL_TASK06_RC_HEAD = "8a946352cd69fd23c00122bbb5aff4071c65793a"
MAIN_MERGE = "c0a8ef0ddec8bd98b94364179abf3cf8c897ab63"
TASK06_CI_RUN = "29199046654"


def test_release_candidate_version_is_consistent() -> None:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    declared = re.search(r'^version\s*=\s*"([^"]+)"', pyproject, re.MULTILINE)

    assert declared is not None
    assert declared.group(1) == RC_VERSION
    assert VERSION == RC_VERSION
    assert __version__ == RC_VERSION


def test_release_candidate_documents_do_not_claim_final_acceptance() -> None:
    required = [
        ROOT / "CHANGELOG.md",
        ROOT / "README.md",
        ROOT / "NIGHTLY_STATUS.md",
        ROOT / "docs" / "release_readiness.md",
        ROOT / "docs" / "known_limitations.md",
    ]
    for path in required:
        text = path.read_text(encoding="utf-8")
        assert RC_VERSION in text, path
        assert "candidate for engineering acceptance" in text, path


def test_methodology_traceability_remains_unsigned_without_bibliography() -> None:
    path = ROOT / "docs" / "methodology" / "antonov_round_stamp_v1.md"
    text = path.read_text(encoding="utf-8")
    required_fields = {
        "profile_id",
        "profile_version",
        "formula",
        "nu",
        "shape_factor",
        "stamp_shape",
        "source_title",
        "author",
        "year",
        "page_or_section",
        "source_file_hash_or_reference",
        "applicability",
        "limitations",
        "reviewer",
        "review_status",
    }
    present = {
        match.group(1)
        for match in re.finditer(r"^([a-z][a-z0-9_]*):", text, re.MULTILINE)
    }

    assert required_fields.issubset(present)
    assert "review_status: review_required_for_release" in text
    assert "source_title: null" in text
    assert 'reviewer: ""' in text


def test_missing_license_is_an_explicit_unsigned_gate() -> None:
    assert not (ROOT / "LICENSE").exists()
    decision = (ROOT / "docs" / "licensing_decision_required.md").read_text(
        encoding="utf-8"
    )
    assert "approved_for_distribution: false" in decision


def test_acceptance_inputs_are_checked_out_with_portable_lf_bytes() -> None:
    attributes = (ROOT / ".gitattributes").read_text(encoding="utf-8")
    assert "acceptance/** text eol=lf" in attributes.splitlines()


def test_merged_task06_evidence_is_recorded_without_final_release_claim() -> None:
    for path in (ROOT / "docs" / "release_readiness.md", ROOT / "NIGHTLY_STATUS.md"):
        text = path.read_text(encoding="utf-8")
        assert FINAL_TASK06_RC_HEAD in text, path
        assert MAIN_MERGE in text, path
        assert TASK06_CI_RUN in text, path
        assert "6/6 matrix jobs SUCCESS" in text, path
        assert "Required CI" in text and "SUCCESS" in text, path
        assert "not a final release" in text, path


def test_three_real_case_templates_remain_unsigned_and_unapproved() -> None:
    manifest_path = ROOT / "acceptance" / "real_cases" / "template_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert payload["contract_version"] == "acceptance-case/1.0"
    assert payload["template_only"] is True
    assert len(payload["cases"]) == 3
    assert [case["source_type"] for case in payload["cases"]] == ["csv", "xlsx", "csv"]
    for case in payload["cases"]:
        assert case["metadata"]["status"] == "pending_real_data"
        assert case["metadata"]["template_only"] is True
        assert case["reviewer"] == {
            "name": None,
            "organization": None,
            "reviewed_at": None,
        }
        assert case["signoff_status"] == "unsigned"
        assert all(
            item["sha256"] == "0" * 64 for item in case["input_files"].values()
        )

    serialized = json.dumps(payload, sort_keys=True)
    assert '"signoff_status": "approved"' not in serialized
    assert '"signoff_status": "accepted"' not in serialized


def test_real_case_guide_covers_all_seven_controlled_steps() -> None:
    text = (ROOT / "acceptance" / "real_cases" / "README.md").read_text(
        encoding="utf-8"
    )

    required_sections = (
        "## 1. Исходный файл",
        "## 2. Metadata",
        "## 3. Независимый расчёт",
        "## 4. Reviewer",
        "## 5. Допуски",
        "## 6. Подпись",
        "## 7. `signoff_status`",
    )
    assert all(section in text for section in required_sections)
    assert "signoff_status=unsigned" in text
    assert "signoff_status=signed" in text
    assert "input_files.signoff" in text
    assert "signoff_status=approved" not in text
    assert "signoff_status=accepted" not in text


def test_methodology_traceability_checklist_stays_open_without_bibliography() -> None:
    text = (
        ROOT / "docs" / "methodology" / "antonov_round_stamp_v1.md"
    ).read_text(encoding="utf-8")

    assert "## Checklist методической трассируемости" in text
    assert text.count("- [ ]") >= 15
    for field in (
        "source_title",
        "author",
        "year",
        "page_or_section",
        "source_file_hash_or_reference",
        "reviewer",
        "review_status",
    ):
        assert field in text
    assert "review_status: review_required_for_release" in text


def test_windows_portable_document_is_a_separate_unsigned_assignment() -> None:
    text = (ROOT / "docs" / "windows_portable_distribution_spec.md").read_text(
        encoding="utf-8"
    )

    assert "Draft technical assignment" in text
    assert "системного Python, Git, прав администратора" in text
    assert "engineering_acceptance=false" in text
    assert "SQLite остаётся отдельной будущей задачей" in text
    assert "не реализуется в portable PR" in text
    assert "не присваивает программе статус final release" in text
