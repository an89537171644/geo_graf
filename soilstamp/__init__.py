"""Soil Stamp Antonov — reproducible plate-load test analysis."""

from .analysis import (
    calculate_moduli_for_test,
    compare_groups,
    deformation_work,
    estimate_moduli,
    fit_segmented_pcr,
    group_mean_curve,
    hysteresis_metrics,
)
from .methodology import (
    ModulusMethodProfile,
    ModulusOverrides,
    ModulusResolution,
    get_modulus_profile,
    modulus_profile_definitions,
    modulus_profile_ids,
    parse_pressure_range,
    resolve_modulus_method,
)
from .data import (
    AuditTrail,
    apply_settlement_correction,
    classify_branches,
    failure_summary,
    prepare_measurements,
    validate_measurements,
)
from .io import parse_decimal, read_protocol, read_protocol_excel
from .indicators import (
    indicator_audit_frame,
    indicator_event_frame,
    indicator_passport_frame,
    process_indicator_frame,
    resolve_indicator_passport,
)
from .provenance import build_provenance, passport_completeness
from .manual_entry_adapter import (
    ManualExperimentSource,
    ManualInputBundle,
    adapt_manual_draft,
)
from .manual_entry_models import (
    ManualAuditEvent,
    ManualDraft,
    ManualPassport,
    ManualPoint,
    ManualReinforcement,
)
from .manual_entry_validation import ManualValidationResult, validate_manual_draft
from .sources import ExcelExperimentSource, ExperimentSource
from .schema import (
    CorrectionRecord,
    Experiment,
    ExperimentPoint,
    IndicatorPassport,
    IndicatorProcessingResult,
    MeasuredPoint,
    ProcessingConfig,
    Project,
    ProvenanceRecord,
    RawCell,
    RawRow,
    VERSION,
)

__all__ = [
    "AuditTrail",
    "apply_settlement_correction",
    "classify_branches",
    "calculate_moduli_for_test",
    "compare_groups",
    "CorrectionRecord",
    "deformation_work",
    "estimate_moduli",
    "ExcelExperimentSource",
    "Experiment",
    "ExperimentPoint",
    "ExperimentSource",
    "IndicatorPassport",
    "IndicatorProcessingResult",
    "ManualAuditEvent",
    "ManualDraft",
    "ManualExperimentSource",
    "ManualInputBundle",
    "ManualPassport",
    "ManualPoint",
    "ManualReinforcement",
    "ManualValidationResult",
    "ModulusMethodProfile",
    "ModulusOverrides",
    "ModulusResolution",
    "failure_summary",
    "fit_segmented_pcr",
    "group_mean_curve",
    "hysteresis_metrics",
    "indicator_audit_frame",
    "indicator_event_frame",
    "indicator_passport_frame",
    "get_modulus_profile",
    "MeasuredPoint",
    "modulus_profile_definitions",
    "modulus_profile_ids",
    "ProcessingConfig",
    "Project",
    "ProvenanceRecord",
    "RawCell",
    "RawRow",
    "build_provenance",
    "adapt_manual_draft",
    "parse_decimal",
    "passport_completeness",
    "prepare_measurements",
    "parse_pressure_range",
    "process_indicator_frame",
    "read_protocol",
    "read_protocol_excel",
    "resolve_indicator_passport",
    "resolve_modulus_method",
    "validate_measurements",
    "validate_manual_draft",
]

__version__ = VERSION
