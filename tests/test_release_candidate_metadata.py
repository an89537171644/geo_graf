from __future__ import annotations

import re
from pathlib import Path

from soilstamp import __version__
from soilstamp.schema import VERSION


ROOT = Path(__file__).resolve().parents[1]
RC_VERSION = "0.5.0rc1"


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
