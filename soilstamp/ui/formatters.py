"""Small Russian display formatters with lossless unknown-value fallbacks."""

from __future__ import annotations

from typing import Any

import pandas as pd

from .column_labels_ru import (
    COMMON_COLUMN_LABELS_RU,
    TABLE_COLUMN_LABELS_RU,
    TABLE_KIND_ALIASES_RU,
    TABLE_VALUE_DOMAINS_RU,
)
from .labels_ru import lookup_ru_label


_STATUS_DOMAINS = (
    "status",
    "metrology",
    "review",
    "row_status",
    "failure",
    "censoring",
    "report",
)

_METHOD_DOMAINS = (
    "method",
    "methodology_profile",
    "curve",
    "graph",
    "aggregation",
    "indicator_mode",
    "correction",
    "import",
    "missing_policy",
    "load_kind",
    "stamp_shape",
)


def _first_known_label(value: Any, domains: tuple[str, ...]) -> str:
    original = "" if value is None else str(value)
    for domain in domains:
        rendered = lookup_ru_label(domain, value)
        if rendered != original:
            return rendered
    return original


def format_status(value: Any) -> str:
    """Format a status for the UI; unknown status strings remain unchanged."""

    return _first_known_label(value, _STATUS_DOMAINS)


def format_method(value: Any) -> str:
    """Format calculation and processing methods without changing their source value."""

    original = "" if value is None else str(value)
    normalized = original.strip().casefold()

    if normalized.startswith("e_tangent@"):
        return f"Касательный модуль E при {original.split('@', 1)[1]}"
    if normalized.startswith("e_incremental_diagnostic#"):
        return f"Диагностический приращённый модуль E №{original.split('#', 1)[1]}"
    if normalized.startswith("cumulative_settlement:"):
        detail = original.split(":", 1)[1].strip()
        prefix = lookup_ru_label("indicator_mode", "cumulative_settlement")
        return f"{prefix}: {detail}" if detail else prefix
    if normalized.endswith("_import"):
        source = original[: -len("_import")]
        rendered_source = lookup_ru_label("import", source)
        if rendered_source != source:
            return f"Импорт: {rendered_source}"

    return _first_known_label(value, _METHOD_DOMAINS)


def _normalize_table_kind(table_kind: Any) -> str:
    normalized = (
        str(table_kind or "generic")
        .strip()
        .casefold()
        .replace("-", "_")
        .replace(" ", "_")
    )
    return TABLE_KIND_ALIASES_RU.get(normalized, normalized)


def _format_single_domain_value(value: str, domain: str) -> str:
    if domain in {"status", "metrology", "review", "row_status", "failure", "report"}:
        if domain == "status":
            return format_status(value)
        return lookup_ru_label(domain, value)
    if domain == "method":
        return format_method(value)
    return lookup_ru_label(domain, value)


def _format_domain_value(value: Any, domain: str) -> Any:
    # Enum localization is deliberately limited to textual machine tokens.
    # Numeric, boolean, datetime and missing values retain their exact type.
    if not isinstance(value, str):
        return value

    rendered = _format_single_domain_value(value, domain)
    if rendered != value or ";" not in value:
        return rendered

    # Indicator warning/quality cells may contain a stable semicolon-delimited
    # list of machine tokens.  Localize each known item for display only.
    parts = [part.strip() for part in value.split(";")]
    localized = [_format_single_domain_value(part, domain) for part in parts]
    if any(display != machine for display, machine in zip(localized, parts, strict=True)):
        return "; ".join(localized)
    return value


def display_dataframe(frame: pd.DataFrame, table_kind: Any = "generic") -> pd.DataFrame:
    """Return a localized display copy without mutating canonical data.

    Only known enum-bearing columns receive localized cell values.  Unknown
    columns and values are preserved, as are the original index and row order.
    The returned frame is intended solely for UI rendering and must not be used
    as an export or calculation input.
    """

    if not isinstance(frame, pd.DataFrame):
        raise TypeError("display_dataframe ожидает pandas.DataFrame.")

    kind = _normalize_table_kind(table_kind)
    displayed = frame.copy(deep=True)
    value_domains = {
        **TABLE_VALUE_DOMAINS_RU.get("generic", {}),
        **TABLE_VALUE_DOMAINS_RU.get(kind, {}),
    }

    original_columns = tuple(displayed.columns)
    for position, column in enumerate(original_columns):
        if not isinstance(column, str):
            continue
        domain = value_domains.get(column)
        if domain is None:
            continue
        localized = displayed.iloc[:, position].map(
            lambda value, selected_domain=domain: _format_domain_value(value, selected_domain)
        )
        displayed.isetitem(position, localized)

    column_labels = {
        **COMMON_COLUMN_LABELS_RU,
        **TABLE_COLUMN_LABELS_RU.get(kind, {}),
    }
    displayed.columns = [
        column_labels.get(column, column) if isinstance(column, str) else column
        for column in original_columns
    ]
    return displayed
