"""Public Russian localization helpers for the engineering UI."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import re
from typing import Any

from .column_labels_ru import COMMON_COLUMN_LABELS_RU, TABLE_VALUE_DOMAINS_RU
from .formatters import display_dataframe, format_method, format_status
from .labels_ru import lookup_ru_label


def label_for_enum(domain: Any, value: Any) -> str:
    """Return a localized enum label or the original unknown token."""

    return lookup_ru_label(domain, value)


_VISIBLE_TEXT_TOKENS_RU: tuple[tuple[str, str], ...] = (
    ("baseline", "контрольная группа"),
    ("reinforced", "армированная группа"),
    ("loading", "нагружение"),
    ("hold", "выдержка"),
    ("unloading", "разгрузка"),
    ("reloading", "повторное нагружение"),
)


def display_text(value: Any) -> str:
    """Localize known standalone UI tokens in generated human-readable text.

    The source string is never modified and unknown text is preserved verbatim.
    This helper is intentionally not used for reports, exports or audit records.
    """

    rendered = "" if value is None else str(value)
    for machine_token, russian_label in _VISIBLE_TEXT_TOKENS_RU:
        rendered = re.sub(
            rf"(?<![\w-]){re.escape(machine_token)}(?![\w-])",
            russian_label,
            rendered,
            flags=re.IGNORECASE,
        )
    return rendered.replace("spline-сглаживания", "сглаживания сплайнами")


def _mapping_value_domain(key: Any, domain_by_key: Mapping[Any, str]) -> str | None:
    if key in domain_by_key:
        return domain_by_key[key]
    if not isinstance(key, str):
        return None

    generic = TABLE_VALUE_DOMAINS_RU.get("generic", {})
    if key in generic:
        return generic[key]

    normalized = key.strip().casefold()
    if normalized == "severity" or normalized.endswith("_status"):
        if normalized.startswith(("review_", "approval_", "signoff_")):
            return "review"
        if normalized.startswith("metrology_"):
            return "metrology"
        return "status"
    if normalized == "method" or normalized.endswith("_method"):
        return "method"
    if normalized in {"profile", "profile_id", "methodology_profile"}:
        return "methodology_profile"
    return None


def _display_scalar(value: Any, domain: str | None) -> Any:
    if domain is None or not isinstance(value, str):
        return deepcopy(value)
    if domain == "status":
        return format_status(value)
    if domain == "method":
        return format_method(value)
    return label_for_enum(domain, value)


def _display_value(
    value: Any,
    *,
    domain: str | None,
    domain_by_key: Mapping[Any, str],
) -> Any:
    if isinstance(value, Mapping):
        return _display_mapping(value, domain_by_key=domain_by_key)
    if isinstance(value, list):
        return [
            _display_value(item, domain=domain, domain_by_key=domain_by_key) for item in value
        ]
    if isinstance(value, tuple):
        return tuple(
            _display_value(item, domain=domain, domain_by_key=domain_by_key) for item in value
        )
    return _display_scalar(value, domain)


def _display_mapping(
    mapping: Mapping[Any, Any],
    *,
    domain_by_key: Mapping[Any, str],
) -> dict[Any, Any]:
    rendered: dict[Any, Any] = {}
    for key, value in mapping.items():
        display_key = COMMON_COLUMN_LABELS_RU.get(key, key) if isinstance(key, str) else key
        domain = _mapping_value_domain(key, domain_by_key)
        rendered[display_key] = _display_value(
            value,
            domain=domain,
            domain_by_key=domain_by_key,
        )
    return rendered


def display_mapping(
    mapping: Mapping[Any, Any],
    *,
    domain_by_key: Mapping[Any, str] | None = None,
) -> dict[Any, Any]:
    """Deep-copy and localize a mapping for ``st.json`` or UI captions.

    Known keys receive Russian display labels; known status and method values
    are localized recursively.  Unknown keys and values remain unchanged.
    """

    if not isinstance(mapping, Mapping):
        raise TypeError("display_mapping ожидает объект Mapping.")
    domains = dict(domain_by_key or {})
    return _display_mapping(mapping, domain_by_key=domains)


__all__ = [
    "display_dataframe",
    "display_mapping",
    "display_text",
    "format_method",
    "format_status",
    "label_for_enum",
]
