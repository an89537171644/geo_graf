"""Lossless adapter from manual drafts to the canonical import contract.

Manual entry is a source of primary data, not a scientific calculation mode.
The adapter parses raw strings, preserves stable UUID/provenance, and delegates
all processing to :func:`soilstamp.data.prepare_measurements`.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .data import prepare_measurements, validate_measurements
from .indicators import canonical_indicator_mode
from .io import parse_decimal
from .manual_entry_models import MANUAL_EDITOR_COLUMNS, ManualDraft, ManualPoint
from .manual_entry_validation import (
    ManualValidationResult,
    merge_manual_issues,
    validate_manual_draft,
)
from .provenance import validate_project_metadata
from .schema import Experiment, ValidationIssue
from .sources import experiments_from_frame


MANUAL_INPUT_SCHEMA = "manual-input/1.0"

_LOAD_KIND = {
    "force": "force",
    "сила": "force",
    "pressure": "pressure",
    "давление": "pressure",
}
_LOAD_UNIT = {
    "n": "N",
    "н": "N",
    "kn": "kN",
    "кн": "kN",
    "mn": "MN",
    "мн": "MN",
    "kgf": "kgf",
    "кгс": "kgf",
    "tf": "tf",
    "тс": "tf",
    "pa": "Pa",
    "па": "Pa",
    "kpa": "kPa",
    "кпа": "kPa",
    "mpa": "MPa",
    "мпа": "MPa",
}
_SETTLEMENT_UNIT = {
    "mm": "mm",
    "мм": "mm",
    "cm": "cm",
    "см": "cm",
    "m": "m",
    "м": "m",
}
_STAMP_SHAPE = {
    "circle": "circle",
    "round": "circle",
    "круг": "circle",
    "круглый": "circle",
    "custom": "custom",
}
_STATUS = {
    "measurement": "stable",
    "failure": "failure",
    "instrument_limit": "instrument_limit",
    "stopped_without_failure": "stopped_without_failure",
    "invalid": "invalid",
}


def _key(value: Any) -> str:
    return str(value or "").strip().casefold().replace(" ", "")


def _number(value: Any) -> float | None:
    return parse_decimal(value)


def _integer_or_number(value: Any) -> int | float | None:
    parsed = _number(value)
    if parsed is None:
        return None
    rounded = round(parsed)
    return int(rounded) if abs(parsed - rounded) <= 1e-12 else parsed


def _raw_or_none(value: Any) -> Any:
    return None if value is None or (isinstance(value, str) and not value.strip()) else value


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _reinforcement_metadata(draft: ManualDraft) -> dict[str, Any]:
    passport = draft.passport
    reinforcement = passport.reinforcement
    layers = _integer_or_number(reinforcement.number_of_layers)
    return {
        "type": passport.reinforcement_type,
        "is_reinforced": bool(passport.is_reinforced),
        "material": reinforcement.material or None,
        "layers": layers,
        "number_of_layers": layers,
        "depth_mm": _number(reinforcement.depth_mm),
        "spacing_mm": _number(reinforcement.spacing_mm),
        "length_mm": _number(reinforcement.length_mm),
        "width_mm": _number(reinforcement.width_mm),
        "bar_diameter_or_aperture_mm": _number(
            reinforcement.bar_diameter_or_aperture_mm
        ),
        "orientation": reinforcement.orientation or None,
        "custom_parameters": dict(reinforcement.custom_parameters)
        if isinstance(reinforcement.custom_parameters, dict)
        else {},
    }


def _indicator_passports(draft: ManualDraft) -> dict[str, dict[str, Any]]:
    passport = draft.passport
    count = passport.number_of_indicators
    if not isinstance(count, int) or isinstance(count, bool) or not 1 <= count <= 4:
        return {}
    serials = [str(value).strip() for value in passport.indicator_serial_numbers]
    mode = canonical_indicator_mode(passport.dial_mode) or passport.dial_mode
    common = {
        "type": passport.indicator_type,
        "range_mm": _number(passport.dial_range_mm),
        "division_mm": _number(passport.dial_resolution_mm),
        "correction_factor": _number(passport.dial_correction_factor),
        "verification_date": passport.verification_date,
        "verification_valid_until": passport.verification_valid_until,
        "mode": mode,
        "initial_reading": _number(passport.dial_initial_reading),
        "zero_correction_mm": _number(passport.dial_zero_correction_mm),
        "max_increment_mm": _number(passport.dial_max_increment_mm),
        "reverse_tolerance_mm": _number(passport.dial_reverse_tolerance_mm),
        "travel_range_mm": _number(passport.dial_travel_range_mm),
        "initial_turn": 0,
        "cumulative_sign": 1.0,
    }
    result: dict[str, dict[str, Any]] = {}
    for index in range(count):
        serial = serials[index] if index < len(serials) else ""
        result[f"indicator_{index + 1}"] = {
            **common,
            "serial_number": serial,
            "instrument_id": serial,
        }
    return result


def _manual_metadata(draft: ManualDraft) -> dict[str, Any]:
    passport = draft.passport
    test_id = passport.test_id
    reinforcement = _reinforcement_metadata(draft)
    indicator_passports = _indicator_passports(draft)
    instruments = [
        {
            "instrument_id": values.get("instrument_id"),
            "type": values.get("type"),
            "serial_number": values.get("serial_number"),
            "range_mm": values.get("range_mm"),
            "division_mm": values.get("division_mm"),
            "correction_factor": values.get("correction_factor"),
            "verification_date": values.get("verification_date"),
            "verification_valid_until": values.get("verification_valid_until"),
            # General project-passport compatibility aliases; metrological
            # semantics and original verification fields remain explicit.
            "calibration_date": values.get("verification_date"),
            "calibration_valid_until": values.get("verification_valid_until"),
        }
        for values in indicator_passports.values()
    ]
    load_kind = _LOAD_KIND.get(str(passport.load_kind).strip().casefold(), passport.load_kind)
    load_unit = _LOAD_UNIT.get(_key(passport.load_unit), passport.load_unit)
    settlement_unit = _SETTLEMENT_UNIT.get(
        _key(passport.settlement_unit), passport.settlement_unit
    )
    stamp_shape = _STAMP_SHAPE.get(
        str(passport.stamp_shape).strip().casefold(), passport.stamp_shape
    )
    test_metadata = {
        "group": passport.group_name or None,
        "baseline_group": passport.baseline_group or None,
        "pair_id": passport.pair_id or None,
        "soil_batch": passport.soil_batch or None,
        "experiment_date": passport.test_date or None,
        "operator": passport.operator or None,
        "reinforcement": reinforcement,
    }
    return {
        "project": passport.project_name,
        "series_name": passport.series_name,
        "source_type": "manual",
        "manual_draft_id": draft.draft_id,
        "manual_draft_schema_version": draft.schema_version,
        "manual_draft_sha256": draft.sha256,
        "test_scope": passport.test_scope,
        "protocol_type": passport.protocol_type,
        "group": passport.group_name or None,
        "baseline_group": passport.baseline_group or None,
        "pair_id": passport.pair_id or None,
        "load_kind": load_kind,
        "load_unit": load_unit,
        "load_factor": _number(passport.load_factor),
        "load_zero": _number(passport.load_zero),
        "lever_ratio": _number(passport.lever_ratio),
        "settlement_unit": settlement_unit,
        "stamp_shape": stamp_shape,
        "stamp_diameter_mm": _number(passport.stamp_diameter_mm),
        "stamp_area_m2": _number(passport.stamp_area_m2),
        "indicator_mode": canonical_indicator_mode(passport.dial_mode)
        or passport.dial_mode,
        "indicator_resolution_mm": _number(passport.dial_resolution_mm),
        "indicator_passports": indicator_passports,
        "instruments": instruments,
        "reinforcement": reinforcement,
        "soil": {"type": passport.soil_type or None, "batch": passport.soil_batch or None},
        "project_passport": {
            "project_id": passport.project_name or None,
            "series_name": passport.series_name or None,
            "reinforcement_status": passport.reinforcement_type or None,
            "baseline_group": passport.baseline_group or None,
            "pair_id": passport.pair_id or None,
            "soil_batch": passport.soil_batch or None,
            "experiment_date": passport.test_date or None,
            "operator": passport.operator or None,
            "soil_type": passport.soil_type or None,
            "laboratory_or_site": passport.laboratory_or_site or None,
            "archive_number": passport.archive_number or None,
            "test_name": passport.test_name or None,
            "instruments": instruments,
        },
        "tests": {test_id: test_metadata} if test_id else {},
    }


def _stage_value(raw: Any) -> Any:
    if _raw_or_none(raw) is None:
        return None
    parsed = parse_decimal(raw, family="stage")
    return _integer_or_number(parsed) if parsed is not None else raw


def _canonical_row(
    draft: ManualDraft, point: ManualPoint, position: int
) -> dict[str, Any]:
    indicators = {
        f"indicator_{number}": parse_decimal(
            point.indicator_raw(number), family="measurement"
        )
        for number in range(1, 5)
    }
    parsed_load = parse_decimal(point.load_raw, family="load")
    parsed_elapsed = parse_decimal(point.elapsed_time_s)
    stage = _stage_value(point.stage_no)
    raw_indicator = point.indicator_1_raw
    return {
        "test_id": draft.passport.test_id,
        "stage": stage,
        "load": parsed_load if parsed_load is not None else np.nan,
        **{
            name: value if value is not None else np.nan
            for name, value in indicators.items()
        },
        **{
            f"raw_indicator_{number}": point.indicator_raw(number)
            for number in range(1, 5)
        },
        "branch": point.branch,
        "timestamp": _raw_or_none(point.timestamp),
        "raw_timestamp": point.timestamp,
        "elapsed_time_s": parsed_elapsed if parsed_elapsed is not None else np.nan,
        "raw_elapsed_time_s": point.elapsed_time_s,
        "status": _STATUS.get(point.row_status, point.row_status),
        "row_status": point.row_status,
        "comment": point.comment,
        "group": draft.passport.group_name or None,
        "pair_id": draft.passport.pair_id or None,
        "sequence_no": point.sequence_no,
        "source_sequence_no": point.sequence_no,
        "sequence_index": position,
        "source_type": "manual",
        "source_row": None,
        "sheet_name": "Manual",
        "manual_row_uuid": point.manual_row_uuid,
        "created_by": point.created_by,
        "created_at": point.created_at,
        "modified_by": point.modified_by,
        "modified_at": point.modified_at,
        "raw_stage": point.stage_no,
        "parsed_stage": stage,
        "raw_load": point.load_raw,
        "parsed_load": parsed_load,
        "raw_indicator": raw_indicator,
        "parsed_indicator": indicators["indicator_1"],
        "source_load_unit": _LOAD_UNIT.get(
            _key(draft.passport.load_unit), draft.passport.load_unit
        ),
        "load_unit": _LOAD_UNIT.get(
            _key(draft.passport.load_unit), draft.passport.load_unit
        ),
        "failure_marker_raw": "failure" if point.row_status == "failure" else None,
        "source_columns": json.dumps(
            {name: name for name in MANUAL_EDITOR_COLUMNS},
            ensure_ascii=False,
            sort_keys=True,
        ),
    }


def _manual_frame(draft: ManualDraft) -> pd.DataFrame:
    frame = pd.DataFrame(
        _canonical_row(draft, point, position)
        for position, point in enumerate(draft.rows)
    )
    if "source_row" in frame:
        frame["source_row"] = pd.Series([None] * len(frame), dtype="object")
    count = draft.passport.number_of_indicators
    if isinstance(count, int) and not isinstance(count, bool) and 1 <= count <= 4:
        inactive = [f"indicator_{number}" for number in range(count + 1, 5)]
        inactive += [f"raw_indicator_{number}" for number in range(count + 1, 5)]
        frame = frame.drop(columns=[name for name in inactive if name in frame])
    return frame


def _raw_cells(draft: ManualDraft) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for position, point in enumerate(draft.rows):
        fields = {
            "sequence_no": point.sequence_no,
            "stage": point.stage_no,
            "branch": point.branch,
            "elapsed_time_s": point.elapsed_time_s,
            "timestamp": point.timestamp,
            "load": point.load_raw,
            "indicator_1": point.indicator_1_raw,
            "indicator_2": point.indicator_2_raw,
            "indicator_3": point.indicator_3_raw,
            "indicator_4": point.indicator_4_raw,
            "status": point.row_status,
            "comment": point.comment,
        }
        for canonical, raw in fields.items():
            if canonical == "load":
                parsed = parse_decimal(raw, family="load")
            elif canonical.startswith("indicator_"):
                parsed = parse_decimal(raw, family="measurement")
            elif canonical == "elapsed_time_s":
                parsed = parse_decimal(raw)
            elif canonical == "stage":
                parsed = _stage_value(raw)
            else:
                parsed = raw
            records.append(
                {
                    "source_type": "manual",
                    "source_row": None,
                    "sequence_index": position,
                    "sequence_no": point.sequence_no,
                    "manual_row_uuid": point.manual_row_uuid,
                    "source_column": canonical,
                    "canonical_field": canonical,
                    "raw_value": raw,
                    "parsed_value": parsed,
                }
            )
    result = pd.DataFrame(records)
    if "source_row" in result:
        result["source_row"] = pd.Series([None] * len(result), dtype="object")
    return result


@dataclass(slots=True)
class ManualInputBundle:
    """Canonical manual input plus its immutable source snapshots."""

    raw: pd.DataFrame
    metadata: dict[str, Any]
    raw_cells: pd.DataFrame
    import_info: dict[str, Any]
    source_bytes: bytes
    metadata_bytes: bytes
    validation: ManualValidationResult

    @property
    def frame(self) -> pd.DataFrame:
        return self.raw

    @property
    def info(self) -> dict[str, Any]:
        return self.import_info

    @property
    def issues(self) -> list[ValidationIssue]:
        return list(self.validation.issues)

    @property
    def blocking_issues(self) -> list[ValidationIssue]:
        return self.validation.blocking_issues

    @property
    def blocking(self) -> bool:
        return self.validation.blocking

    @property
    def can_analyze(self) -> bool:
        return self.validation.can_analyze

    def prepare_measurements(self) -> tuple[pd.DataFrame, list[ValidationIssue]]:
        """Run the established source-neutral pipeline, if validation permits."""

        if not self.can_analyze:
            return self.raw.copy(deep=True), self.issues
        prepared, pipeline = prepare_measurements(
            self.raw, self.metadata, strict_metadata=True
        )
        return prepared, merge_manual_issues(self.issues, pipeline)

    def prepare(self) -> tuple[pd.DataFrame, list[ValidationIssue]]:
        return self.prepare_measurements()


def adapt_manual_draft(draft: ManualDraft) -> ManualInputBundle:
    """Parse a draft into the same canonical frame/metadata used by Excel."""

    if not isinstance(draft, ManualDraft):
        raise TypeError("adapt_manual_draft ожидает экземпляр ManualDraft")
    adapter_validation = validate_manual_draft(draft)
    raw = _manual_frame(draft)
    metadata = _manual_metadata(draft)
    raw_cells = _raw_cells(draft)

    pipeline_issues = merge_manual_issues(
        validate_project_metadata(metadata, strict=True),
        validate_measurements(raw, metadata),
    )
    all_issues = merge_manual_issues(adapter_validation.issues, pipeline_issues)
    validation = ManualValidationResult(
        issues=all_issues,
        adapter_issues=list(adapter_validation.issues),
        pipeline_issues=pipeline_issues,
    )
    source_bytes = draft.to_json(indent=None).encode("utf-8")
    metadata_bytes = _canonical_json_bytes(metadata)
    import_info = {
        "format": "manual",
        "source_type": "manual",
        "schema": MANUAL_INPUT_SCHEMA,
        "draft_schema": draft.schema_version,
        "draft_id": draft.draft_id,
        "test_id": draft.passport.test_id,
        "rows": len(raw),
        "columns": list(raw.columns),
        "source_file_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "metadata_sha256": hashlib.sha256(metadata_bytes).hexdigest(),
        "issue_count": len(all_issues),
        "blocking_issue_count": len(validation.blocking_issues),
        "can_analyze": validation.can_analyze,
        "import_mode": "manual",
    }
    return ManualInputBundle(
        raw=raw,
        metadata=metadata,
        raw_cells=raw_cells,
        import_info=import_info,
        source_bytes=source_bytes,
        metadata_bytes=metadata_bytes,
        validation=validation,
    )


@dataclass(slots=True)
class ManualExperimentSource:
    """``ExperimentSource`` implementation backed by a manual draft."""

    draft: ManualDraft
    _bundle: ManualInputBundle | None = field(default=None, init=False, repr=False)

    @property
    def bundle(self) -> ManualInputBundle:
        if self._bundle is None:
            self._bundle = adapt_manual_draft(self.draft)
        return self._bundle

    def load(self) -> list[Experiment]:
        return experiments_from_frame(
            self.bundle.raw,
            self.bundle.metadata,
            source_type="manual",
        )

    def prepare(self) -> tuple[pd.DataFrame, list[ValidationIssue]]:
        return self.bundle.prepare_measurements()


__all__ = [
    "MANUAL_INPUT_SCHEMA",
    "ManualExperimentSource",
    "ManualInputBundle",
    "adapt_manual_draft",
]
