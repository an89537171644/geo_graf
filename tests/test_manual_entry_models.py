from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import fields
from pathlib import Path

import pytest

from soilstamp.manual_entry_models import (
    MANUAL_DRAFT_SCHEMA_V1_0,
    MANUAL_DRAFT_SCHEMA_V1_1,
    MANUAL_DRAFT_SCHEMA_VERSION,
    MANUAL_INDICATOR_CHANNELS,
    ManualAuditEvent,
    ManualDraft,
    ManualIndicatorPassport,
    ManualLegacyIndicatorCommon,
    ManualPassport,
    ManualPoint,
    ManualReinforcement,
    migrate_manual_draft_payload,
)


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = ROOT / "docs" / "manual-entry-draft-1.2.schema.json"

V1_1_COMMON = {
    "dial_mode": "decreasing_wrapped",
    "dial_range_mm": "10,000",
    "dial_resolution_mm": "0,010",
    "dial_correction_factor": "1,0020",
    "dial_initial_reading": "9,800",
    "dial_zero_correction_mm": "-0,005",
    "dial_max_increment_mm": "2,00",
    "dial_reverse_tolerance_mm": "0,020",
    "dial_travel_range_mm": "50,0",
    "indicator_type": "ИЧ-10",
    "indicator_serial_numbers": ["A-001", "B-002"],
    "verification_date": "2026-01-15",
    "verification_valid_until": "2027-01-15",
}

V1_2_METROLOGY_FIELDS = {
    "indicator_passports",
    "legacy_common_indicator_passport",
    "settlement_aggregation",
    "settlement_aggregation_channels",
    "settlement_primary_channel",
    "settlement_missing_channel_policy",
    "metrology_status",
}


def _v1_1_payload() -> dict:
    draft = ManualDraft.create(author="legacy-author", initial_rows=2)
    draft.passport.project_name = "Legacy project"
    draft.passport.baseline_group = "control-series"
    draft.passport.number_of_indicators = 2
    draft.rows[0].load_raw = "0,000"
    draft.rows[0].indicator_1_raw = "09,800"
    draft.audit_events.append(
        ManualAuditEvent.create(
            author="legacy-author",
            action="update_passport",
            entity_id=f"{draft.draft_id}:passport",
            field="project_name",
            old_value="",
            new_value="Legacy project",
            reason="legacy edit",
        )
    )
    payload = draft.to_dict()
    payload["schema_version"] = MANUAL_DRAFT_SCHEMA_V1_1
    passport = payload["passport"]
    for name in V1_2_METROLOGY_FIELDS:
        passport.pop(name)
    passport.update(deepcopy(V1_1_COMMON))
    return payload


def _v1_0_payload() -> dict:
    payload = _v1_1_payload()
    payload["schema_version"] = MANUAL_DRAFT_SCHEMA_V1_0
    payload["passport"].pop("pair_id")
    return payload


@pytest.mark.parametrize("source", [_v1_0_payload, _v1_1_payload])
def test_staged_migration_to_1_2_is_lossless_and_idempotent(source) -> None:
    legacy = source()
    untouched = deepcopy(legacy)

    migrated = migrate_manual_draft_payload(legacy)
    migrated_again = migrate_manual_draft_payload(migrated)

    assert legacy == untouched
    assert migrated_again == migrated
    assert migrated["schema_version"] == MANUAL_DRAFT_SCHEMA_VERSION
    assert migrated["passport"]["baseline_group"] == "control-series"
    assert migrated["passport"]["pair_id"] is None
    assert migrated["passport"]["number_of_indicators"] == 2
    assert migrated["passport"]["legacy_common_indicator_passport"] == V1_1_COMMON
    assert migrated["passport"]["indicator_passports"] == {
        channel: None for channel in MANUAL_INDICATOR_CHANNELS
    }
    assert migrated["passport"]["settlement_aggregation"] == "no_aggregation"
    assert migrated["passport"]["settlement_aggregation_channels"] == []
    assert migrated["passport"]["settlement_primary_channel"] is None
    assert migrated["passport"]["settlement_missing_channel_policy"] == "block"
    assert migrated["passport"]["metrology_status"] == "migration_review_required"
    assert not (set(V1_1_COMMON) & set(migrated["passport"]))

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

    restored = ManualDraft.from_dict(legacy)
    roundtrip = ManualDraft.from_json(restored.to_json())
    assert restored.schema_version == MANUAL_DRAFT_SCHEMA_VERSION
    assert restored.passport.metrology_status == "migration_review_required"
    assert restored.passport.legacy_common_indicator_passport is not None
    assert restored.passport.legacy_common_indicator_passport.to_dict() == V1_1_COMMON
    assert [row.manual_row_uuid for row in restored.rows] == [
        row["manual_row_uuid"] for row in legacy["rows"]
    ]
    assert [event.event_id for event in restored.audit_events] == [
        event["event_id"] for event in legacy["audit_events"]
    ]
    assert roundtrip.to_dict() == restored.to_dict()
    assert roundtrip.sha256 == restored.sha256


def test_legacy_migration_rejects_a_pair_id_schema_collision() -> None:
    legacy = _v1_0_payload()
    legacy["passport"]["pair_id"] = "P-01"

    with pytest.raises(ValueError, match="pair_id.*1.0"):
        migrate_manual_draft_payload(legacy)


@pytest.mark.parametrize(
    "field,value",
    [
        ("indicator_passports", {}),
        ("legacy_common_indicator_passport", {}),
        ("settlement_aggregation", "all_channels_mean"),
        ("metrology_status", "confirmed"),
    ],
)
def test_1_1_migration_rejects_1_2_field_collisions(field: str, value) -> None:
    legacy = _v1_1_payload()
    legacy["passport"][field] = value

    with pytest.raises(ValueError, match="поля схемы 1.2"):
        migrate_manual_draft_payload(legacy)


def test_1_1_migration_rejects_incomplete_common_passport() -> None:
    legacy = _v1_1_payload()
    legacy["passport"].pop("dial_initial_reading")

    with pytest.raises(ValueError, match="неполон.*dial_initial_reading"):
        migrate_manual_draft_payload(legacy)


def test_current_1_2_migration_is_a_nonmutating_idempotent_copy() -> None:
    current = ManualDraft.create(author="current-author", initial_rows=1).to_dict()
    untouched = deepcopy(current)

    migrated = migrate_manual_draft_payload(current)

    assert current == untouched
    assert migrated == current
    assert migrated is not current
    assert migrated["passport"] is not current["passport"]
    assert migrate_manual_draft_payload(migrated) == migrated


def test_channel_passports_roundtrip_without_numeric_normalization() -> None:
    draft = ManualDraft.create(initial_rows=0)
    passport = ManualIndicatorPassport(
        type="ИЧ-10",
        serial_number="SN-01",
        instrument_id="INST-01",
        range_mm="10,000",
        division_mm="0,0100",
        correction_factor="1,0020",
        mode="decreasing_wrapped",
        initial_reading="09,800",
        initial_turn=-2,
        zero_correction_mm="-0,0050",
        max_increment_mm="2,000",
        reverse_tolerance_mm="0,020",
        travel_range_mm="50,00",
        verification_date="2026-01-15",
        verification_valid_until="2027-01-15",
        x_mm="-105,00",
        y_mm="0,00",
        cumulative_sign="-1,0",
        assignment_status="confirmed",
    )
    draft.passport.indicator_passports["indicator_1"] = passport
    draft.passport.number_of_indicators = 1
    draft.passport.settlement_aggregation = "primary_channel"
    draft.passport.settlement_aggregation_channels = ["indicator_1"]
    draft.passport.settlement_primary_channel = "indicator_1"
    draft.passport.metrology_status = "confirmed"

    restored = ManualDraft.from_json(draft.to_json())

    effective = restored.passport.indicator_passports["indicator_1"]
    assert effective is not None
    assert effective.to_dict() == passport.to_dict()
    assert effective.division_mm == "0,0100"
    assert effective.initial_turn == -2
    assert restored.to_dict() == draft.to_dict()


@pytest.mark.parametrize(
    "mutate,match",
    [
        (
            lambda payload: payload["passport"]["indicator_passports"].pop(
                "reference_indicator"
            ),
            "reference_indicator",
        ),
        (
            lambda payload: payload["passport"]["indicator_passports"].update(
                {"indicator_5": None}
            ),
            "indicator_5",
        ),
        (
            lambda payload: payload["passport"].update(
                {"settlement_aggregation_channels": ["indicator_1", "indicator_1"]}
            ),
            "повторы",
        ),
        (
            lambda payload: payload["passport"].update(
                {"settlement_primary_channel": "reference_indicator"}
            ),
            "settlement_primary_channel",
        ),
    ],
)
def test_current_contract_rejects_ambiguous_channel_registry(mutate, match: str) -> None:
    payload = ManualDraft.create(initial_rows=0).to_dict()
    mutate(payload)

    with pytest.raises(ValueError, match=match):
        ManualDraft.from_dict(payload)


def test_json_schema_matches_runtime_models() -> None:
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    current = ManualDraft.create(author="schema-test", initial_rows=1)
    current.audit_events.append(
        ManualAuditEvent.create(
            author="schema-test",
            action="schema_test",
            entity_id=current.draft_id,
            field=None,
            old_value=None,
            new_value=None,
            reason="schema test",
        )
    )
    payload = current.to_dict()

    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert schema["additionalProperties"] is False
    assert schema["properties"]["schema_version"]["const"] == (
        MANUAL_DRAFT_SCHEMA_VERSION
    )
    assert set(schema["required"]) == set(schema["properties"]) == {
        field.name for field in fields(ManualDraft)
    }
    assert set(payload) == set(schema["properties"])

    model_contracts = (
        ("manual_passport", ManualPassport, payload["passport"]),
        ("manual_reinforcement", ManualReinforcement, payload["passport"]["reinforcement"]),
        ("manual_point", ManualPoint, payload["rows"][0]),
        ("audit_event", ManualAuditEvent, payload["audit_events"][0]),
        (
            "manual_indicator_passport",
            ManualIndicatorPassport,
            ManualIndicatorPassport().to_dict(),
        ),
        (
            "manual_legacy_indicator_common",
            ManualLegacyIndicatorCommon,
            ManualLegacyIndicatorCommon().to_dict(),
        ),
    )
    for definition_name, model, model_payload in model_contracts:
        definition = schema["$defs"][definition_name]
        expected = {field.name for field in fields(model)}
        assert definition["additionalProperties"] is False
        assert set(definition["required"]) == set(definition["properties"]) == expected
        assert set(model_payload) == expected

    restored = ManualDraft.from_dict(payload)
    assert restored.schema_version == MANUAL_DRAFT_SCHEMA_VERSION
    assert restored.passport.metrology_status == "draft"
