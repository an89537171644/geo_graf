from __future__ import annotations

import json
from datetime import datetime

import pandas as pd
import pytest

from soilstamp.manual_entry_models import MANUAL_DRAFT_SCHEMA_VERSION, ManualDraft
from soilstamp.manual_entry_service import (
    ClipboardShapeError,
    ConfirmationRequiredError,
    EditorConflictError,
    HistoryEmptyError,
    ManualEntryService,
    editor_frame,
    parse_rectangular_tsv,
)


def _service(rows: int = 2) -> ManualEntryService:
    return ManualEntryService(ManualDraft.create(author="tester", initial_rows=rows))


def test_editor_frame_is_lossless_for_raw_text_none_and_hidden_indicators() -> None:
    service = _service(2)
    service.draft.rows[0].indicator_3_raw = "7,005"
    frame = service.editor_frame(n_indicators=2)

    assert frame.dtypes.eq(object).all()
    assert "indicator_3_raw" not in frame
    frame.at[0, "load_raw"] = "0,1250"
    frame.at[0, "indicator_1_raw"] = "09,80"
    frame.at[1, "load_raw"] = None
    original_uuids = [row.manual_row_uuid for row in service.draft.rows]

    assert service.apply_editor_frame(frame, author="engineer")
    assert service.draft.rows[0].load_raw == "0,1250"
    assert service.draft.rows[0].indicator_1_raw == "09,80"
    assert service.draft.rows[0].indicator_3_raw == "7,005"
    assert service.draft.rows[1].load_raw is None
    assert [row.manual_row_uuid for row in service.draft.rows] == original_uuids

    roundtrip = editor_frame(service.draft, n_indicators=4)
    audit_count = len(service.draft.audit_events)
    assert service.apply_editor_frame(roundtrip) is False
    assert len(service.draft.audit_events) == audit_count


def test_update_passport_and_reinforcement_fields_are_audited() -> None:
    service = _service()

    assert service.update_passport(
        {
            "project_name": "Опыт 01",
            "number_of_indicators": "2",
            "stamp_diameter_mm": "300,0",
            "reinforcement.material": "геосетка",
            "reinforcement.depth_mm": "50,0",
        },
        author="ivanov",
    )

    assert service.draft.passport.project_name == "Опыт 01"
    assert service.draft.passport.number_of_indicators == 2
    assert service.draft.passport.stamp_diameter_mm == "300,0"
    assert service.draft.passport.reinforcement.material == "геосетка"
    assert service.draft.passport.reinforcement.depth_mm == "50,0"
    assert {event.field for event in service.draft.audit_events} == {
        "project_name",
        "number_of_indicators",
        "stamp_diameter_mm",
        "reinforcement.material",
        "reinforcement.depth_mm",
    }
    assert all(event.author == "ivanov" for event in service.draft.audit_events)
    assert all(
        datetime.fromisoformat(event.timestamp).tzinfo is not None
        for event in service.draft.audit_events
    )


def test_add_insert_duplicate_delete_keep_uuid_and_auto_renumber() -> None:
    service = _service(2)
    first_uuid, second_uuid = [row.manual_row_uuid for row in service.draft.rows]

    appended_uuid = service.add_row({"load_raw": "30,0"})
    inserted_uuid = service.insert_row(
        second_uuid, position="before", values={"load_raw": "10,0"}
    )
    duplicate_uuid = service.duplicate_row(inserted_uuid, position="after")

    assert len({row.manual_row_uuid for row in service.draft.rows}) == 5
    assert duplicate_uuid != inserted_uuid
    assert service.draft.rows[0].manual_row_uuid == first_uuid
    assert service.draft.rows[-1].manual_row_uuid == appended_uuid
    assert [row.sequence_no for row in service.draft.rows] == [1, 2, 3, 4, 5]
    assert service.draft.rows[2].load_raw == "10,0"

    with pytest.raises(ConfirmationRequiredError):
        service.delete_row(second_uuid)
    assert any(row.manual_row_uuid == second_uuid for row in service.draft.rows)

    deleted = service.delete_row(second_uuid, confirmed=True, author="reviewer")
    assert deleted.manual_row_uuid == second_uuid
    assert [row.sequence_no for row in service.draft.rows] == [1, 2, 3, 4]
    assert any(
        event.action == "delete_row" and event.entity_id == second_uuid
        for event in service.draft.audit_events
    )


def test_fill_stages_is_selectable_and_audited_per_changed_cell() -> None:
    service = _service(4)
    selected = [service.draft.rows[1].manual_row_uuid, service.draft.rows[3].manual_row_uuid]

    assert service.fill_stages(start=10, step=5, rows=selected) == 2
    assert [row.stage_no for row in service.draft.rows] == [None, "10", None, "15"]
    events = [event for event in service.draft.audit_events if event.action == "fill_stages"]
    assert [event.new_value for event in events] == ["10", "15"]


def test_rectangular_tsv_paste_copy_and_clear_preserve_decimal_comma() -> None:
    service = _service(1)
    payload = "1,250\t9,80\n2,500\t9,45"

    assert parse_rectangular_tsv(payload) == [
        ["1,250", "9,80"],
        ["2,500", "9,45"],
    ]
    assert service.paste_block(0, "load_raw", payload, expand_rows=True) == 4
    assert len(service.draft.rows) == 2
    assert service.draft.rows[0].load_raw == "1,250"
    assert service.draft.rows[1].indicator_1_raw == "9,45"
    assert service.copy_block(0, 1, "load_raw", "indicator_1_raw") == payload

    first_uuid = service.draft.rows[0].manual_row_uuid
    assert service.clear_cells([(first_uuid, "load_raw"), (0, "indicator_1_raw")]) == 2
    assert service.draft.rows[0].load_raw is None
    assert service.draft.rows[0].indicator_1_raw is None
    assert [event.action for event in service.draft.audit_events].count("clear_cell") == 2


def test_clipboard_rejects_non_rectangular_and_out_of_bounds_blocks() -> None:
    service = _service(1)
    with pytest.raises(ClipboardShapeError, match="прямоугольным"):
        parse_rectangular_tsv("1\t2\n3")
    with pytest.raises(ClipboardShapeError, match="правую границу"):
        service.paste_block(0, "comment", "x\ty")
    with pytest.raises(ClipboardShapeError, match="нижнюю границу"):
        service.paste_block(0, "load_raw", "1\n2", expand_rows=False)


def test_apply_editor_frame_diffs_add_delete_and_reorder_by_stable_uuid() -> None:
    service = _service(2)
    frame = service.editor_frame(n_indicators=1)
    first_uuid = str(frame.at[0, "manual_row_uuid"])
    deleted_uuid = str(frame.at[1, "manual_row_uuid"])
    frame.at[0, "load_raw"] = "12,5"
    new_row = {column: None for column in frame.columns}
    new_row.update({"branch": "loading", "row_status": "measurement", "load_raw": "25,0"})
    edited = pd.concat([frame.iloc[[0]], pd.DataFrame([new_row])], ignore_index=True)

    audit_before = len(service.draft.audit_events)
    with pytest.raises(ConfirmationRequiredError):
        service.apply_editor_frame(edited)
    assert len(service.draft.audit_events) == audit_before
    assert len(service.draft.rows) == 2

    assert service.apply_editor_frame(
        edited, author="operator", confirm_deletions=True
    )
    assert len(service.draft.rows) == 2
    assert service.draft.rows[0].manual_row_uuid == first_uuid
    assert service.draft.rows[0].load_raw == "12,5"
    assert service.draft.rows[1].manual_row_uuid not in {first_uuid, deleted_uuid}
    assert service.draft.rows[1].load_raw == "25,0"
    assert [row.sequence_no for row in service.draft.rows] == [1, 2]
    actions = [event.action for event in service.draft.audit_events]
    assert "editor_cell_edit" in actions
    assert "editor_add_row" in actions
    assert "editor_delete_row" in actions


def test_editor_frame_rejects_duplicate_or_forged_uuid() -> None:
    service = _service(2)
    duplicate = service.editor_frame()
    duplicate.at[1, "manual_row_uuid"] = duplicate.at[0, "manual_row_uuid"]
    with pytest.raises(EditorConflictError, match="повторяется"):
        service.apply_editor_frame(duplicate)

    forged = service.editor_frame()
    forged.at[0, "manual_row_uuid"] = "forged-id"
    with pytest.raises(EditorConflictError, match="Неизвестный"):
        service.apply_editor_frame(forged)


def test_rejected_bulk_edits_are_atomic_and_do_not_write_audit() -> None:
    service = _service(2)
    before = service.draft.to_dict()
    invalid = service.editor_frame(n_indicators=1)
    invalid.at[0, "load_raw"] = "value-that-would-be-applied-first"
    invalid.at[1, "sequence_no"] = "not-an-integer"

    with pytest.raises(EditorConflictError, match="sequence_no"):
        service.apply_editor_frame(invalid)
    assert service.draft.to_dict() == before

    with pytest.raises(EditorConflictError, match="sequence_no"):
        service.paste_block(0, "sequence_no", "1\t10,0\nwrong\t20,0")
    assert service.draft.to_dict() == before


def test_undo_redo_restore_content_and_only_append_audit_history() -> None:
    service = _service(1)
    frame = service.editor_frame(n_indicators=1)
    frame.at[0, "load_raw"] = "5,5"
    service.apply_editor_frame(frame, author="operator")
    original_event_ids = [event.event_id for event in service.draft.audit_events]
    audit_after_edit = len(original_event_ids)

    assert service.undo(author="reviewer") == "apply_editor_frame"
    assert service.draft.rows[0].load_raw is None
    assert len(service.draft.audit_events) == audit_after_edit + 1
    assert [
        event.event_id for event in service.draft.audit_events[:audit_after_edit]
    ] == original_event_ids
    assert service.draft.audit_events[-1].action == "undo"

    assert service.redo(author="reviewer") == "apply_editor_frame"
    assert service.draft.rows[0].load_raw == "5,5"
    assert len(service.draft.audit_events) == audit_after_edit + 2
    assert service.draft.audit_events[-1].action == "redo"
    assert len({event.event_id for event in service.draft.audit_events}) == len(
        service.draft.audit_events
    )

    with pytest.raises(HistoryEmptyError):
        service.redo()


def test_new_mutation_after_undo_clears_redo_without_deleting_audit() -> None:
    service = _service(1)
    service.add_row()
    service.undo()
    audit_count = len(service.draft.audit_events)

    service.update_passport({"project_name": "Новая ветвь"})
    assert len(service.draft.audit_events) == audit_count + 1
    with pytest.raises(HistoryEmptyError):
        service.redo()


def test_structural_undo_redo_restores_the_same_generated_uuid() -> None:
    service = _service(1)
    added_uuid = service.add_row({"load_raw": "10,0"})

    service.undo()
    assert all(row.manual_row_uuid != added_uuid for row in service.draft.rows)

    service.redo()
    assert service.draft.rows[-1].manual_row_uuid == added_uuid
    assert service.draft.rows[-1].load_raw == "10,0"
    assert [event.action for event in service.draft.audit_events][-2:] == ["undo", "redo"]


def test_draft_json_roundtrip_preserves_decimal_comma_none_and_audit() -> None:
    service = _service(2)
    frame = service.editor_frame(n_indicators=1)
    frame.at[0, "load_raw"] = "1,2300"
    frame.at[0, "indicator_1_raw"] = "9,80"
    frame.at[1, "load_raw"] = None
    service.apply_editor_frame(frame, author="operator")
    service.update_passport(
        {"archive_number": "АРХ-1", "load_factor": "0,0100"}, author="operator"
    )

    payload = service.to_json()
    restored = ManualEntryService.from_json(payload)

    assert restored.draft.to_dict() == service.draft.to_dict()
    assert restored.draft.rows[0].load_raw == "1,2300"
    assert restored.draft.rows[1].load_raw is None
    assert restored.draft.passport.load_factor == "0,0100"
    assert restored.draft.sha256 == service.draft.sha256


@pytest.mark.parametrize(
    "payload,match",
    [
        ("{broken", "повреждён"),
        (
            json.dumps(
                {
                    "schema_version": "manual-entry-draft/999",
                    "rows": [],
                    "audit_events": [],
                }
            ),
            "версия",
        ),
        (
            json.dumps(
                {
                    "schema_version": MANUAL_DRAFT_SCHEMA_VERSION,
                    "rows": "not-a-list",
                    "audit_events": [],
                }
            ),
            "rows",
        ),
    ],
)
def test_corrupt_or_unsupported_draft_schema_is_rejected(payload: str, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        ManualEntryService.from_json(payload)


def test_draft_loader_rejects_malformed_audit_passport_and_source_provenance() -> None:
    service = _service(1)

    malformed_audit = service.draft.to_dict()
    malformed_audit["audit_events"] = [None]
    with pytest.raises(ValueError, match="audit_events"):
        ManualEntryService.from_json(json.dumps(malformed_audit, ensure_ascii=False))

    empty_audit = service.draft.to_dict()
    empty_audit["audit_events"] = [{}]
    with pytest.raises(ValueError, match="event_id"):
        ManualEntryService.from_json(json.dumps(empty_audit, ensure_ascii=False))

    malformed_passport = service.draft.to_dict()
    malformed_passport["passport"] = []
    with pytest.raises(ValueError, match="passport"):
        ManualEntryService.from_json(json.dumps(malformed_passport, ensure_ascii=False))

    hidden_source_repair = service.draft.to_dict()
    hidden_source_repair["rows"][0]["source_row"] = 17
    with pytest.raises(ValueError, match="source_row"):
        ManualEntryService.from_json(
            json.dumps(hidden_source_repair, ensure_ascii=False)
        )

    non_text_raw = service.draft.to_dict()
    non_text_raw["rows"][0]["load_raw"] = {"hidden": "conversion"}
    with pytest.raises(ValueError, match="row.load_raw"):
        ManualEntryService.from_json(json.dumps(non_text_raw, ensure_ascii=False))


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("is_reinforced", "false", "is_reinforced"),
        ("number_of_indicators", 1.5, "number_of_indicators"),
        ("indicator_serial_numbers", {"0": "I-1"}, "indicator_serial_numbers"),
        ("reinforcement", [1], "reinforcement"),
    ],
)
def test_draft_loader_rejects_lossy_passport_coercions(
    field: str, value, match: str
) -> None:
    payload = _service(1).draft.to_dict()
    payload["passport"][field] = value

    with pytest.raises(ValueError, match=match):
        ManualEntryService.from_json(json.dumps(payload, ensure_ascii=False))


def test_draft_loader_requires_complete_audit_event_schema() -> None:
    service = _service(1)
    service.update_passport({"project_name": "Project"})
    payload = service.draft.to_dict()
    for key in ("field", "old_value", "new_value"):
        malformed = json.loads(json.dumps(payload, ensure_ascii=False))
        malformed["audit_events"][0].pop(key)
        with pytest.raises(ValueError, match=key):
            ManualEntryService.from_json(json.dumps(malformed, ensure_ascii=False))


def test_draft_loader_rejects_nonstandard_json_nan() -> None:
    service = _service(1)
    payload = service.draft.to_dict()
    payload["passport"]["reinforcement"]["custom_parameters"] = {
        "invalid": float("nan")
    }

    with pytest.raises(ValueError, match="JSON-константа"):
        ManualEntryService.from_json(json.dumps(payload, ensure_ascii=False))


def test_nested_json_values_must_remain_finite_and_saveable() -> None:
    service = _service(1)
    before = service.draft.sha256

    with pytest.raises(ValueError, match="NaN или Infinity"):
        service.update_passport(
            {"reinforcement.custom_parameters": {"invalid": float("inf")}}
        )
    assert service.draft.sha256 == before

    custom_payload = service.draft.to_dict()
    custom_payload["passport"]["reinforcement"]["custom_parameters"] = {
        "invalid": "__HUGE_FLOAT__"
    }
    custom_json = json.dumps(custom_payload, ensure_ascii=False).replace(
        '"__HUGE_FLOAT__"', "1e999"
    )
    with pytest.raises(ValueError, match="NaN или Infinity"):
        ManualEntryService.from_json(custom_json)

    service.update_passport({"project_name": "Project"})
    audit_payload = service.draft.to_dict()
    audit_payload["audit_events"][0]["new_value"] = "__HUGE_FLOAT__"
    audit_json = json.dumps(audit_payload, ensure_ascii=False).replace(
        '"__HUGE_FLOAT__"', "1e999"
    )
    with pytest.raises(ValueError, match="NaN или Infinity"):
        ManualEntryService.from_json(audit_json)
