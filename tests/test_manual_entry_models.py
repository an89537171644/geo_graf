from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import fields
from pathlib import Path

import pytest

from soilstamp.manual_entry_models import (
    MANUAL_DRAFT_SCHEMA_V1_0,
    MANUAL_DRAFT_SCHEMA_VERSION,
    ManualAuditEvent,
    ManualDraft,
    ManualPassport,
    ManualPoint,
    ManualReinforcement,
    migrate_manual_draft_payload,
)


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "manual_entry_demo.json"
SCHEMA = ROOT / "docs" / "manual-entry-draft-1.1.schema.json"


def _legacy_payload() -> dict:
    payload = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    payload["schema_version"] = MANUAL_DRAFT_SCHEMA_V1_0
    payload["passport"].pop("pair_id")
    return payload


def test_migration_1_0_to_1_1_is_lossless_and_idempotent() -> None:
    legacy = _legacy_payload()
    untouched = deepcopy(legacy)

    migrated = migrate_manual_draft_payload(legacy)
    migrated_again = migrate_manual_draft_payload(migrated)

    assert legacy == untouched
    assert migrated_again == migrated
    assert migrated["schema_version"] == MANUAL_DRAFT_SCHEMA_VERSION
    assert migrated["passport"]["baseline_group"] == legacy["passport"]["baseline_group"]
    assert migrated["passport"]["pair_id"] is None
    assert migrated["rows"] == legacy["rows"]
    assert migrated["audit_events"] == legacy["audit_events"]
    for name in (
        "draft_id",
        "status",
        "created_by",
        "created_at",
        "updated_at",
    ):
        assert migrated[name] == legacy[name]
    for name, value in legacy["passport"].items():
        assert migrated["passport"][name] == value

    restored = ManualDraft.from_dict(legacy)
    roundtrip = ManualDraft.from_json(restored.to_json())
    assert restored.schema_version == MANUAL_DRAFT_SCHEMA_VERSION
    assert restored.passport.pair_id is None
    assert [row.manual_row_uuid for row in restored.rows] == [
        row["manual_row_uuid"] for row in legacy["rows"]
    ]
    assert [event.event_id for event in restored.audit_events] == [
        event["event_id"] for event in legacy["audit_events"]
    ]
    assert roundtrip.to_dict() == restored.to_dict()
    assert roundtrip.sha256 == restored.sha256


def test_legacy_migration_rejects_a_pair_id_schema_collision() -> None:
    legacy = _legacy_payload()
    legacy["passport"]["pair_id"] = "P-01"

    with pytest.raises(ValueError, match="pair_id.*1.0"):
        migrate_manual_draft_payload(legacy)


def test_json_schema_matches_runtime_models_and_canonical_example() -> None:
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    example = json.loads(EXAMPLE.read_text(encoding="utf-8"))

    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert schema["additionalProperties"] is False
    assert schema["properties"]["schema_version"]["const"] == (
        MANUAL_DRAFT_SCHEMA_VERSION
    )
    assert set(schema["required"]) == set(schema["properties"]) == {
        field.name for field in fields(ManualDraft)
    }
    assert set(example) == set(schema["properties"])

    model_contracts = (
        ("manual_passport", ManualPassport, example["passport"]),
        ("manual_reinforcement", ManualReinforcement, example["passport"]["reinforcement"]),
        ("manual_point", ManualPoint, example["rows"][0]),
        ("audit_event", ManualAuditEvent, example["audit_events"][0]),
    )
    for definition_name, model, payload in model_contracts:
        definition = schema["$defs"][definition_name]
        expected = {field.name for field in fields(model)}
        assert definition["additionalProperties"] is False
        assert set(definition["required"]) == set(definition["properties"]) == expected
        assert set(payload) == expected

    restored = ManualDraft.from_json(EXAMPLE.read_bytes())
    assert restored.schema_version == MANUAL_DRAFT_SCHEMA_VERSION
    assert restored.passport.baseline_group == "baseline"
    assert restored.passport.pair_id is None
