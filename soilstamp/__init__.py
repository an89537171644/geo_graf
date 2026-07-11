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
from .io import read_protocol, read_protocol_excel
from .indicators import (
    indicator_audit_frame,
    indicator_event_frame,
    indicator_passport_frame,
    process_indicator_frame,
    resolve_indicator_passport,
)
from .provenance import build_provenance, passport_completeness
from .schema import (
    CorrectionRecord,
    Experiment,
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
    "Experiment",
    "IndicatorPassport",
    "IndicatorProcessingResult",
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
    "passport_completeness",
    "prepare_measurements",
    "process_indicator_frame",
    "read_protocol",
    "read_protocol_excel",
    "resolve_indicator_passport",
    "validate_measurements",
]

__version__ = VERSION
