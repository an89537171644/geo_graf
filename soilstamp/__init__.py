"""Soil Stamp Antonov — reproducible plate-load test analysis."""

from .analysis import (
    compare_groups,
    deformation_work,
    estimate_moduli,
    fit_segmented_pcr,
    group_mean_curve,
    hysteresis_metrics,
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
    "failure_summary",
    "fit_segmented_pcr",
    "group_mean_curve",
    "hysteresis_metrics",
    "indicator_audit_frame",
    "indicator_event_frame",
    "indicator_passport_frame",
    "MeasuredPoint",
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
    "process_indicator_frame",
    "read_protocol",
    "read_protocol_excel",
    "resolve_indicator_passport",
    "validate_measurements",
    "validate_manual_draft",
]

__version__ = VERSION
