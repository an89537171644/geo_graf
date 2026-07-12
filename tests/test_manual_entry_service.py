from __future__ import annotations

import json
from copy import deepcopy
from datetime import datetime

import pandas as pd
import pytest

from soilstamp.manual_entry_models import (
    MANUAL_DRAFT_SCHEMA_VERSION,
    ManualDraft,
    ManualIndicatorPassport,
)
from soilstamp.manual_entry_service import (
    ClipboardShapeError,
    ConfirmationRequiredError,
    EditorConflictError,
    HistoryEmptyError,
    ManualEntryService,
    ManualEntryServiceError,
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


def test_sensitive_update_with_default_reason_invalidates_stale_confirmation() -> None:
    service = _service()
    service.draft.passport.indicator_passports["indicator_1"] = (
        ManualIndicatorPassport(
            serial_number="ORIGINAL-01",
            assignment_status="confirmed",
        )
    )
    service.draft.passport.metrology_status = "confirmed"
    changed_passports = service.draft.passport.to_dict()["indicator_passports"]
    changed_passports["indicator_1"]["serial_number"] = "REPLACED-02"

    assert service.update_passport(
        {
            "indicator_passports": changed_passports,
            "metrology_status": "confirmed",
        }
    )

    effective = service.draft.passport.indicator_passports["indicator_1"]
    assert effective is not None and effective.serial_number == "REPLACED-02"
    assert service.draft.passport.metrology_status == "draft"
    assert [event.action for event in service.draft.audit_events] == [
        "update_passport",
        "invalidate_metrology_confirmation",
    ]
    invalidation = service.draft.audit_events[-1]
    assert invalidation.field == "metrology_status"
    assert invalidation.old_value == "confirmed"
    assert invalidation.new_value == "draft"
    assert invalidation.reason == "manual_edit"


def test_sensitive_update_can_be_explicitly_reconfirmed_with_reason() -> None:
    service = _service()
    service.draft.passport.indicator_passports["indicator_1"] = (
        ManualIndicatorPassport(
            serial_number="ORIGINAL-01",
            assignment_status="confirmed",
        )
    )
    service.draft.passport.metrology_status = "confirmed"
    changed_passports = service.draft.passport.to_dict()["indicator_passports"]
    changed_passports["indicator_1"]["serial_number"] = "REPLACED-02"
    reason = "Инженер повторно проверил назначение после замены номера"

    assert service.update_passport(
        {
            "indicator_passports": changed_passports,
            "metrology_status": "confirmed",
        },
        author="metrologist",
        reason=reason,
    )

    assert service.draft.passport.metrology_status == "confirmed"
    assert [event.action for event in service.draft.audit_events] == [
        "update_passport",
        "reconfirm_metrology_after_update",
    ]
    reconfirmation = service.draft.audit_events[-1]
    assert reconfirmation.field == "metrology_status"
    assert reconfirmation.old_value == "confirmed"
    assert reconfirmation.new_value == "confirmed"
    assert reconfirmation.author == "metrologist"
    assert reconfirmation.reason == reason


def test_non_metrology_update_does_not_invalidate_confirmation() -> None:
    service = _service()
    service.draft.passport.metrology_status = "confirmed"

    assert service.update_passport({"project_name": "Updated project"})

    assert service.draft.passport.project_name == "Updated project"
    assert service.draft.passport.metrology_status == "confirmed"
    assert [event.action for event in service.draft.audit_events] == [
        "update_passport"
    ]
    assert service.draft.audit_events[0].field == "project_name"


def test_experiment_date_change_invalidates_confirmed_verification_basis() -> None:
    service = _service()
    service.draft.passport.test_date = "2026-01-15"
    service.draft.passport.metrology_status = "confirmed"

    assert service.update_passport({"test_date": "2027-02-20"})

    assert service.draft.passport.test_date == "2027-02-20"
    assert service.draft.passport.metrology_status == "draft"
    assert [event.action for event in service.draft.audit_events] == [
        "update_passport",
        "invalidate_metrology_confirmation",
    ]
    assert service.draft.audit_events[0].field == "test_date"
    invalidation = service.draft.audit_events[1]
    assert invalidation.old_value == "confirmed"
    assert invalidation.new_value == "draft"


@pytest.mark.parametrize("initial_status", ["draft", "migration_review_required"])
def test_default_reason_cannot_confirm_metrology_directly(initial_status: str) -> None:
    service = _service()
    service.draft.passport.metrology_status = initial_status
    before = service.draft.to_json(indent=None)

    with pytest.raises(ManualEntryServiceError, match="обоснование reason"):
        service.update_passport({"metrology_status": "confirmed"})

    assert service.draft.to_json(indent=None) == before
    assert service.draft.passport.metrology_status == initial_status
    assert service.draft.audit_events == []
    assert not service.can_undo


def test_explicit_reason_allows_direct_metrology_confirmation() -> None:
    service = _service()
    reason = "Инженер проверил все поканальные паспорта и агрегацию"

    assert service.update_passport(
        {"metrology_status": "confirmed"},
        author="metrologist",
        reason=reason,
    )

    assert service.draft.passport.metrology_status == "confirmed"
    assert len(service.draft.audit_events) == 1
    confirmation = service.draft.audit_events[0]
    assert confirmation.action == "update_passport"
    assert confirmation.field == "metrology_status"
    assert confirmation.old_value == "draft"
    assert confirmation.new_value == "confirmed"
    assert confirmation.reason == reason


def test_copy_indicator_passport_is_explicit_independent_and_audited() -> None:
    service = _service()
    source = ManualIndicatorPassport(
        type="ИЧ-10",
        serial_number="SOURCE-01",
        instrument_id="INST-01",
        range_mm="10,000",
        division_mm="0,010",
        correction_factor="1,002",
        mode="decreasing_wrapped",
        initial_reading="9,800",
        initial_turn=0,
        zero_correction_mm="0,000",
        max_increment_mm="2,000",
        reverse_tolerance_mm="0,020",
        travel_range_mm="50,0",
        verification_date="2026-01-15",
        verification_valid_until="2027-01-15",
        x_mm="-100,0",
        y_mm="0,0",
        cumulative_sign="1,0",
        assignment_status="confirmed",
    )
    service.draft.passport.indicator_passports["indicator_1"] = source
    service.draft.passport.metrology_status = "confirmed"

    assert service.copy_indicator_passport(
        "indicator_1",
        ["indicator_2", "reference_indicator"],
        author="metrologist",
        reason="Явно назначить одинаковый тип прибора двум каналам",
    )

    first = service.draft.passport.indicator_passports["indicator_2"]
    second = service.draft.passport.indicator_passports["reference_indicator"]
    assert first is not None and second is not None
    expected = source.to_dict()
    expected["assignment_status"] = "review_required"
    assert first.to_dict() == expected == second.to_dict()
    assert first is not source and second is not source and first is not second
    assert source.assignment_status == "confirmed"
    assert service.draft.passport.metrology_status == "draft"
    events = service.draft.audit_events
    assert [event.action for event in events] == [
        "copy_indicator_passport",
        "copy_indicator_passport",
        "invalidate_metrology_confirmation",
    ]
    assert [event.field for event in events] == [
        "indicator_passports.indicator_2",
        "indicator_passports.reference_indicator",
        "metrology_status",
    ]
    assert all(event.author == "metrologist" for event in events)
    assert [event.old_value for event in events] == [None, None, "confirmed"]
    assert [event.new_value for event in events] == [expected, expected, "draft"]

    source.serial_number = "CHANGED-AFTER-COPY"
    assert first.serial_number == "SOURCE-01"
    assert second.serial_number == "SOURCE-01"


def _draft_bytes_allowing_injected_nan(service: ManualEntryService) -> bytes:
    return json.dumps(
        service.draft.to_dict(),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=True,
    ).encode("utf-8")


def test_copy_indicator_passport_nan_validation_is_byte_atomic() -> None:
    service = _service()
    source = ManualIndicatorPassport(
        serial_number="SOURCE-01",
        assignment_status="confirmed",
    )
    # Simulate a corrupt integration bypassing the typed service API.  The copy
    # command must detect it while staging and leave every live byte untouched.
    source.range_mm = float("nan")  # type: ignore[assignment]
    service.draft.passport.indicator_passports["indicator_1"] = source
    service.draft.passport.metrology_status = "confirmed"
    before = _draft_bytes_allowing_injected_nan(service)
    updated_at = service.draft.updated_at
    audit_count = len(service.draft.audit_events)
    undo_count = len(service._undo_stack)
    redo_count = len(service._redo_stack)

    with pytest.raises(ValueError, match="range_mm"):
        service.copy_indicator_passport(
            "indicator_1",
            ["indicator_2"],
            author="metrologist",
            reason="Проверка атомарности",
        )

    assert _draft_bytes_allowing_injected_nan(service) == before
    assert service.draft.updated_at == updated_at
    assert len(service.draft.audit_events) == audit_count
    assert len(service._undo_stack) == undo_count
    assert len(service._redo_stack) == redo_count
    assert service.draft.passport.metrology_status == "confirmed"


def test_copy_indicator_passport_exact_prepared_target_is_true_noop() -> None:
    service = _service()
    source = ManualIndicatorPassport(
        type="ИЧ-10",
        serial_number="SOURCE-01",
        instrument_id="INST-01",
        assignment_status="confirmed",
    )
    prepared_target = deepcopy(source)
    prepared_target.assignment_status = "review_required"
    service.draft.passport.indicator_passports["indicator_1"] = source
    service.draft.passport.indicator_passports["indicator_2"] = prepared_target
    service.draft.passport.metrology_status = "confirmed"
    service.update_passport({"project_name": "Existing undo command"})
    before = service.draft.to_json(indent=None)
    updated_at = service.draft.updated_at
    audit_count = len(service.draft.audit_events)
    undo_count = len(service._undo_stack)
    redo_count = len(service._redo_stack)

    changed = service.copy_indicator_passport(
        "indicator_1",
        ["indicator_2"],
        author="metrologist",
        reason="Повторная команда без изменения",
    )

    assert changed is False
    assert service.draft.to_json(indent=None) == before
    assert service.draft.updated_at == updated_at
    assert len(service.draft.audit_events) == audit_count
    assert len(service._undo_stack) == undo_count
    assert len(service._redo_stack) == redo_count
    assert service.draft.passport.metrology_status == "confirmed"


@pytest.mark.parametrize(
    "source,targets,reason,match",
    [
        ("indicator_1", ["indicator_2"], "", "reason"),
        ("indicator_5", ["indicator_2"], "valid reason", "исходный"),
        ("indicator_1", [], "valid reason", "хотя бы один"),
        ("indicator_1", ["indicator_2", "indicator_2"], "valid reason", "повторы"),
        ("indicator_1", ["indicator_1"], "valid reason", "Исходный канал"),
    ],
)
def test_copy_indicator_passport_rejects_invalid_command_atomically(
    source: str, targets: list[str], reason: str, match: str
) -> None:
    service = _service()
    service.draft.passport.indicator_passports["indicator_1"] = (
        ManualIndicatorPassport(serial_number="SOURCE-01")
    )
    before = service.draft.to_dict()

    with pytest.raises(ManualEntryServiceError, match=match):
        service.copy_indicator_passport(
            source,
            targets,
            author="metrologist",
            reason=reason,
        )

    assert service.draft.to_dict() == before


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
        ("indicator_passports", {"indicator_1": None}, "indicator_passports"),
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
