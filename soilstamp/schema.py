"""Column names, metadata defaults and lightweight result containers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


VERSION = "0.5.0b2.dev1"


REQUIRED_COLUMNS = ("test_id", "stage", "load")
OPTIONAL_PROTOCOL_COLUMNS = (
    "branch",
    "timestamp",
    "indicator_1",
    "indicator_2",
    "indicator_3",
    "indicator_4",
    "indicator_1_turn_number",
    "indicator_2_turn_number",
    "indicator_3_turn_number",
    "indicator_4_turn_number",
    "reference_indicator",
    "horizontal_indicator",
    "settlement",
    "status",
    "comment",
    "group",
)

BRANCHES = ("loading", "hold", "unloading", "reloading", "cyclic")
CORRECTION_MODES = ("raw", "zero_shifted", "seating_corrected")
IMPORT_MODES = ("strict", "interactive", "heuristic")
FAILURE_WORDS = (
    "failure",
    "failed",
    "collapse",
    "destroyed",
    "ушла",
    "разрушение",
    "провал",
)


@dataclass(slots=True)
class ValidationIssue:
    level: str
    code: str
    message: str
    test_id: str | None = None
    rows: list[int] = field(default_factory=list)
    sheet: str | None = None
    row: int | None = None
    column: str | None = None
    raw_value: Any = None
    suggested_action: str | None = None
    blocks_processing: bool | None = None
    entity_id: str | None = None

    def __post_init__(self) -> None:
        if self.blocks_processing is None:
            self.blocks_processing = self.level.casefold() == "error"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["severity"] = self.level
        return payload


@dataclass(slots=True)
class RawCell:
    """One source cell with its exact location and parsed interpretation."""

    sheet_name: str
    source_row: int
    source_column: str
    raw_value: Any
    parsed_value: Any = None
    canonical_field: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class RawRow:
    sheet_name: str
    source_row: int
    sequence_index: int
    cells: list[RawCell] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class MeasuredPoint:
    test_id: str
    sheet_name: str
    source_row: int
    source_columns: dict[str, str]
    sequence_index: int
    raw_stage: Any
    raw_indicator: Any
    raw_load: Any
    parsed_stage: Any
    parsed_indicator: float | None
    parsed_load: float | None
    load_unit: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ExperimentPoint:
    """Source-neutral primary point used by Excel and manual sources."""

    test_id: str
    sequence_no: int
    stage: Any = None
    stage_raw: Any = None
    branch: str | None = None
    elapsed_time_s: Any = None
    elapsed_time_raw: Any = None
    timestamp: Any = None
    timestamp_raw: Any = None
    load_raw: Any = None
    indicator_raws: dict[str, Any] = field(default_factory=dict)
    row_status: str | None = None
    comment: str | None = None
    source_type: str = "unknown"
    source_row: int | None = None
    manual_row_uuid: str | None = None
    created_by: str | None = None
    created_at: str | None = None
    modified_by: str | None = None
    modified_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ProcessingConfig:
    import_mode: str = "strict"
    column_mapping: dict[str, str] = field(default_factory=dict)
    load_kind: str | None = None
    load_unit: str | None = None
    load_factor: float | None = None
    load_zero: float | None = None
    lever_ratio: float | None = None
    stamp_diameter_mm: float | None = None
    stamp_area_m2: float | None = None
    settlement_unit: str | None = None
    indicator_mode: str | None = None
    indicator_unit: str | None = None
    indicator_calibration_factor: float | None = None
    indicator_sign: float | None = None
    reference_sign: float | None = None
    indicator_resolution_mm: float | None = None
    indicator_instrument_id: str | None = None
    indicator_passports: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class IndicatorPassport:
    """Effective metrological passport for one indicator channel.

    Scale readings, range and division are expressed in millimetres on the
    indicator dial. ``correction_factor`` is dimensionless; zero correction,
    motion tolerance and the maximum interval increment are in millimetres of
    settlement.
    """

    channel: str
    indicator_type: str
    serial_number: str
    range_mm: float
    division_mm: float
    correction_factor: float
    verification_date: str
    verification_valid_until: str
    mode: str
    initial_reading: float | None
    zero_correction_mm: float
    max_increment_mm: float | None = None
    reverse_tolerance_mm: float | None = None
    travel_range_mm: float | None = None
    initial_turn: int = 0
    cumulative_sign: float = 1.0
    instrument_id: str | None = None
    x_mm: float | None = None
    y_mm: float | None = None
    assignment_status: str = "review_required"
    verification_status: str = "review_required"
    verification_evaluation_date: str | None = None
    verification_evaluation_date_source: str | None = None
    verification_evaluation_rule: str = (
        "verification_date <= experiment_date <= verification_valid_until"
    )
    reverse_tolerance_source: str = "passport"
    cumulative_sign_source: str = "passport"
    initial_turn_source: str = "passport"
    source_path: str = "metadata.indicator_passports"
    compatibility_mode: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class IndicatorProcessingResult:
    """Deterministic point table, event log and effective passports."""

    audit_rows: list[dict[str, Any]] = field(default_factory=list)
    event_rows: list[dict[str, Any]] = field(default_factory=list)
    passport_rows: list[dict[str, Any]] = field(default_factory=list)
    aggregation_rows: list[dict[str, Any]] = field(default_factory=list)
    settlement_by_row: dict[int, float | None] = field(default_factory=dict)
    channel_settlement_by_row: dict[str, dict[int, float | None]] = field(
        default_factory=dict
    )
    issues: list[ValidationIssue] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class CorrectionRecord:
    id: str
    test_id: str
    kind: str
    old_value: Any
    new_value: Any
    author: str
    timestamp_utc: str
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class Experiment:
    test_id: str
    group: str | None = None
    pair_id: str | None = None
    soil_batch: str | None = None
    experiment_date: str | None = None
    operator: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    points: list[MeasuredPoint | ExperimentPoint] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class Project:
    project_id: str
    title: str
    series_name: str | None = None
    experiments: list[Experiment] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ProvenanceRecord:
    input_file_sha256: str
    metadata_sha256: str
    config_sha256: str
    program_version: str
    git_commit: str | None
    git_dirty: bool | None
    source_tree_sha256: str | None
    python_version: str
    dependency_versions: dict[str, str]
    metrology_evaluations: list[dict[str, Any]] = field(default_factory=list)
    processing_timestamp_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class FailureResult:
    test_id: str
    failure_reached: bool
    right_censored: bool
    F_last_stable: float | None
    F_failure_step: float | None
    Fu_lower: float | None
    Fu_upper: float | None
    p_last_stable: float | None
    p_failure_step: float | None
    pu_lower: float | None
    pu_upper: float | None
    s_last_stable: float | None
    s_failure: float | None
    display: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class PCRResult:
    method: str
    pcr_auto: float
    pcr_ci_low: float | None
    pcr_ci_high: float | None
    intercept: float
    slope_before: float
    slope_after: float
    hinge_delta: float
    r2: float
    aic: float
    bic: float
    n: int
    used_indices: list[int]
    fitted: list[float]
    residuals: list[float]
    bootstrap_valid: int
    pcr_manual: float | None = None
    manual_reason: str | None = None
    manual_author: str | None = None
    manual_confirmed_at_utc: str | None = None
    alternative: dict[str, Any] | None = None

    @property
    def pcr_effective(self) -> float:
        return self.pcr_manual if self.pcr_manual is not None else self.pcr_auto

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["pcr_effective"] = self.pcr_effective
        return result


@dataclass(slots=True)
class ModulusResult:
    method: str
    E_stamp_app_kPa: float
    p_min_kPa: float
    p_max_kPa: float
    n: int
    r2: float | None
    ci_low_kPa: float | None
    ci_high_kPa: float | None
    nu: float
    shape_factor: float
    slope_m_per_kPa: float | None
    profile_id: str = "diagnostic_unapproved_v1"
    profile_version: str = "1.0"
    is_primary: bool = False
    review_status: str = "review_required"
    p_range_source: str = "diagnostic_full_curve"
    nu_source: str = "legacy_default"
    shape_factor_source: str = "legacy_default"
    used_indices: list[int] = field(default_factory=list)
    methodology_note: str = ""
    profile_source: str = "legacy_default"
    p_range_origin: str = "observed_data"
    requested_p_min_kPa: float | None = None
    requested_p_max_kPa: float | None = None
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


DEFAULT_METADATA: dict[str, Any] = {
    "project": "Soil Stamp Antonov",
    "load_kind": "force",
    "load_unit": "kN",
    "load_factor": 1.0,
    "load_zero": 0.0,
    "settlement_unit": "mm",
    "stamp_shape": "circle",
    "stamp_diameter_mm": None,
    "stamp_area_m2": None,
    "lever_ratio": 1.0,
    "poisson_ratio": 0.30,
    "shape_factor": 1.0,
    "indicator_resolution_mm": 0.01,
    "load_resolution_kN": 0.01,
    "indicator_sign": 1.0,
    "reference_sign": -1.0,
    "gamma_kN_m3": None,
    "pu_kPa_confirmed": None,
    "reinforcement": {"type": "none", "layers": 0},
    "project_passport": {
        "project_id": None,
        "series_name": None,
        "reinforcement_status": None,
        "pair_id": None,
        "soil_batch": None,
        "experiment_date": None,
        "operator": None,
        "tray_dimensions_mm": None,
        "dry_density_kg_m3": None,
        "moisture_percent": None,
        "soil_type": None,
        "instruments": [],
    },
}


def merged_metadata(metadata: dict[str, Any] | None) -> dict[str, Any]:
    """Return defaults plus user metadata without mutating either mapping."""

    result = dict(DEFAULT_METADATA)
    if metadata:
        result.update(metadata)
    return result
