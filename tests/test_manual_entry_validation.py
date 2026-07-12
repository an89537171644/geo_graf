from __future__ import annotations

from copy import deepcopy

import pytest

from soilstamp.manual_entry_models import (
    ManualDraft,
    ManualIndicatorPassport,
    ManualPoint,
)
from soilstamp.manual_entry_validation import validate_manual_draft


def _valid_draft() -> ManualDraft:
    draft = ManualDraft.create(initial_rows=3, author="engineer")
    passport = draft.passport
    values = {
        "project_name": "Project",
        "series_name": "Series",
        "test_name": "T-01",
        "test_date": "2026-06-01",
        "operator": "Engineer",
        "laboratory_or_site": "Lab",
        "test_scope": "laboratory",
        "protocol_type": "static_step",
        "group_name": "B",
        "baseline_group": "B",
        "soil_type": "sand",
        "soil_batch": "batch",
        "reinforcement_type": "none",
        "stamp_shape": "circle",
        "stamp_diameter_mm": "300",
        "load_kind": "force",
        "load_unit": "kN",
        "load_factor": "1",
        "load_zero": "0",
        "lever_ratio": "1",
        "settlement_unit": "mm",
        "number_of_indicators": 1,
        "settlement_aggregation": "primary_channel",
        "settlement_aggregation_channels": ["indicator_1"],
        "settlement_primary_channel": "indicator_1",
        "settlement_missing_channel_policy": "block",
        "metrology_status": "confirmed",
    }
    for name, value in values.items():
        setattr(passport, name, value)
    passport.indicator_passports["indicator_1"] = ManualIndicatorPassport(
        type="ИЧ-10",
        serial_number="I-1",
        instrument_id="I-1",
        range_mm="10",
        division_mm="0,01",
        correction_factor="1",
        mode="cumulative_settlement",
        initial_reading="0",
        initial_turn=0,
        zero_correction_mm="0",
        max_increment_mm="2",
        reverse_tolerance_mm="0,02",
        travel_range_mm="50",
        verification_date="2026-01-01",
        verification_valid_until="2030-01-01",
        x_mm="0",
        y_mm="0",
        cumulative_sign="1",
        assignment_status="confirmed",
    )
    for position, row in enumerate(draft.rows):
        row.stage_no = str(position)
        row.elapsed_time_s = str(position * 60)
        row.load_raw = str(position)
        row.indicator_1_raw = str(position / 10)
    return draft


def _codes(draft: ManualDraft) -> set[str]:
    return {issue.code for issue in validate_manual_draft(draft).issues}


def test_valid_manual_draft_can_be_analyzed() -> None:
    result = validate_manual_draft(_valid_draft())

    assert result.can_analyze
    assert not result.blocking
    assert result.blocking_issues == []


def test_inactive_indicator_passports_are_preserved_without_blocking() -> None:
    draft = _valid_draft()
    inactive = deepcopy(draft.passport.indicator_passports["indicator_1"])
    assert inactive is not None
    inactive.serial_number = "I-2"
    inactive.instrument_id = "I-2"
    draft.passport.indicator_passports["indicator_2"] = inactive

    result = validate_manual_draft(draft)

    assert result.can_analyze
    assert any(
        issue.code == "inactive_manual_indicator_passport"
        and not bool(issue.blocks_processing)
        for issue in result.issues
    )


@pytest.mark.parametrize(
    ("field", "value", "code"),
    [
        ("project_name", "", "missing_manual_passport_field"),
        ("test_scope", "ocean", "invalid_manual_test_scope"),
        ("protocol_type", "unknown", "invalid_manual_protocol_type"),
        ("load_unit", "MPa", "manual_load_unit_kind_conflict"),
        ("settlement_unit", "inch", "unsupported_manual_settlement_unit"),
        ("stamp_diameter_mm", "0", "invalid_manual_passport_number"),
        ("number_of_indicators", 5, "invalid_manual_indicator_count"),
    ],
)
def test_required_passport_enums_units_geometry_and_indicator_count(
    field: str, value, code: str
) -> None:
    draft = _valid_draft()
    setattr(draft.passport, field, value)

    result = validate_manual_draft(draft)

    assert code in {issue.code for issue in result.blocking_issues}
    assert not result.can_analyze


def test_reinforcement_fields_are_conditional() -> None:
    draft = _valid_draft()
    draft.passport.is_reinforced = True
    draft.passport.reinforcement_type = "geogrid"
    draft.passport.baseline_group = ""
    draft.passport.reinforcement.material = ""
    draft.passport.reinforcement.number_of_layers = "0"

    codes = _codes(draft)

    assert "missing_manual_passport_field" not in codes
    assert "missing_manual_reinforcement_field" in codes
    assert "invalid_manual_reinforcement_layers" in codes


def test_baseline_group_and_pair_id_are_optional_for_a_standalone_test() -> None:
    draft = _valid_draft()
    draft.passport.baseline_group = ""
    draft.passport.pair_id = None

    result = validate_manual_draft(draft)

    assert result.can_analyze
    assert not any(
        issue.column in {"baseline_group", "pair_id"}
        and bool(issue.blocks_processing)
        for issue in result.issues
    )


def test_pair_id_edge_whitespace_is_preserved_and_reported() -> None:
    draft = _valid_draft()
    draft.passport.pair_id = " P1 "

    result = validate_manual_draft(draft)
    issue = next(item for item in result.issues if item.code == "noncanonical_manual_pair_id")

    assert result.can_analyze
    assert issue.level == "warning"
    assert issue.column == "pair_id"
    assert issue.raw_value == " P1 "


def test_sequence_uuid_and_time_order_are_blocking_and_rows_are_addressable() -> None:
    draft = _valid_draft()
    draft.rows[1].sequence_no = draft.rows[0].sequence_no
    draft.rows[1].manual_row_uuid = draft.rows[0].manual_row_uuid
    draft.rows[2].elapsed_time_s = "30"

    result = validate_manual_draft(draft)

    codes = {issue.code for issue in result.blocking_issues}
    assert "duplicate_manual_sequence_no" in codes
    assert "duplicate_manual_row_uuid" in codes
    assert "manual_elapsed_time_order" in codes
    duplicate = next(
        issue for issue in result.issues if issue.code == "duplicate_manual_row_uuid"
    )
    assert duplicate.entity_id == draft.rows[0].manual_row_uuid
    assert duplicate.row in {1, 2}


def test_duplicate_measurement_rows_remain_visible_as_warning() -> None:
    draft = _valid_draft()
    first = draft.rows[0]
    second = draft.rows[1]
    second.stage_no = first.stage_no
    second.branch = first.branch
    second.elapsed_time_s = first.elapsed_time_s
    second.timestamp = first.timestamp
    second.load_raw = first.load_raw
    second.indicator_1_raw = first.indicator_1_raw
    second.comment = first.comment

    result = validate_manual_draft(draft)
    duplicate = [issue for issue in result.issues if issue.code == "duplicate_manual_row"]

    assert len(duplicate) == 2
    assert all(issue.level == "warning" for issue in duplicate)
    assert {issue.row for issue in duplicate} == {1, 2}


def test_load_decrease_is_allowed_only_for_unloading() -> None:
    unloading = _valid_draft()
    unloading.rows[1].load_raw = "2"
    unloading.rows[2].load_raw = "1"
    unloading.rows[2].branch = "unloading"
    assert "manual_load_decrease_outside_unloading" not in _codes(unloading)

    loading = deepcopy(unloading)
    loading.rows[2].branch = "loading"
    result = validate_manual_draft(loading)
    issue = next(
        issue
        for issue in result.blocking_issues
        if issue.code == "manual_load_decrease_outside_unloading"
    )
    assert issue.row == 3
    assert issue.entity_id == loading.rows[2].manual_row_uuid


def test_failure_after_stable_allows_missing_indicator() -> None:
    draft = _valid_draft()
    failure = ManualPoint.create(4, author="engineer")
    failure.stage_no = "3"
    failure.elapsed_time_s = "180"
    failure.load_raw = "3"
    failure.indicator_1_raw = None
    failure.row_status = "failure"
    draft.rows.append(failure)

    result = validate_manual_draft(draft)

    assert result.can_analyze
    assert "missing_manual_measurement" not in {issue.code for issue in result.issues}


def test_failure_order_and_measurement_count_are_enforced() -> None:
    draft = _valid_draft()
    draft.rows[0].row_status = "failure"
    draft.rows[0].indicator_1_raw = None

    codes = {issue.code for issue in validate_manual_draft(draft).blocking_issues}

    assert "manual_failure_without_stable_predecessor" in codes
    assert "manual_measurement_after_failure" in codes
    assert "insufficient_manual_measurements" not in codes  # two valid measurements remain


def test_explicit_invalid_row_is_visible_but_not_counted_as_measurement() -> None:
    draft = _valid_draft()
    draft.rows[1].row_status = "invalid"
    draft.rows[1].load_raw = "unreadable raw value"

    result = validate_manual_draft(draft)

    invalid = next(issue for issue in result.issues if issue.code == "manual_row_marked_invalid")
    assert invalid.level == "error"
    assert invalid.entity_id == draft.rows[1].manual_row_uuid
    assert not result.can_analyze


def test_fewer_than_two_measurements_is_critical() -> None:
    draft = _valid_draft()
    draft.rows[1].row_status = "invalid"
    draft.rows[2].row_status = "stopped_without_failure"

    result = validate_manual_draft(draft)

    assert any(
        issue.code == "insufficient_manual_measurements"
        for issue in result.blocking_issues
    )


def test_unknown_experiment_date_requires_review_without_system_date() -> None:
    draft = _valid_draft()
    draft.passport.test_date = ""

    result = validate_manual_draft(draft)

    reviews = [
        issue
        for issue in result.issues
        if issue.code == "manual_indicator_verification_review_required"
    ]
    assert reviews
    assert all(not bool(issue.blocks_processing) for issue in reviews)
    assert result.can_analyze


def test_all_channels_mean_requires_fixed_complete_active_basis() -> None:
    draft = _valid_draft()
    draft.passport.number_of_indicators = 2
    second = deepcopy(draft.passport.indicator_passports["indicator_1"])
    assert second is not None
    second.serial_number = "I-2"
    second.instrument_id = "I-2"
    draft.passport.indicator_passports["indicator_2"] = second
    draft.passport.settlement_aggregation = "all_channels_mean"
    draft.passport.settlement_aggregation_channels = ["indicator_1"]

    result = validate_manual_draft(draft)

    assert "manual_all_channels_basis_mismatch" in {
        issue.code for issue in result.blocking_issues
    }


def test_plane_center_rejects_collinear_channel_coordinates() -> None:
    draft = _valid_draft()
    draft.passport.number_of_indicators = 3
    for index in (2, 3):
        item = deepcopy(draft.passport.indicator_passports["indicator_1"])
        assert item is not None
        item.serial_number = f"I-{index}"
        item.instrument_id = f"I-{index}"
        item.x_mm = str((index - 1) * 100)
        item.y_mm = "0"
        draft.passport.indicator_passports[f"indicator_{index}"] = item
    draft.passport.settlement_aggregation = "plane_center"
    draft.passport.settlement_aggregation_channels = [
        "indicator_1",
        "indicator_2",
        "indicator_3",
    ]

    result = validate_manual_draft(draft)

    assert "collinear_manual_plane_coordinates" in {
        issue.code for issue in result.blocking_issues
    }


def test_migrated_common_passport_is_visible_but_not_effective() -> None:
    draft = _valid_draft()
    draft.passport.metrology_status = "migration_review_required"

    result = validate_manual_draft(draft)

    assert "manual_metrology_migration_review_required" in {
        issue.code for issue in result.blocking_issues
    }
    assert not result.can_analyze
