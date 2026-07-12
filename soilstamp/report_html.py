"""Deterministic, standalone HTML report-package rendering.

The renderer deliberately contains no JavaScript, remote styles, generated
timestamps or implicit numerical conversions of the raw protocol.  It accepts
plain Python records as well as pandas tables so the report-package facade can
share one data contract with the XLSX renderer.
"""

from __future__ import annotations

import base64
import hashlib
import html
import json
import math
import re
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, is_dataclass
from datetime import date, datetime
from decimal import Decimal, ROUND_HALF_UP
from numbers import Integral, Real
from typing import Any
from urllib.parse import unquote, urlsplit

import pandas as pd


_SECTIONS = (
    ("Project passport", "project-passport"),
    ("Raw data", "raw-data"),
    ("Prepared data", "prepared-data"),
    ("Indicator passports", "indicator-passports"),
    ("Indicator audit", "indicator-audit"),
    ("QC issues", "qc-issues"),
    ("Failure/censoring", "failure-censoring"),
    ("pcr", "pcr"),
    ("Moduli", "moduli"),
    ("Group comparison", "group-comparison"),
    ("Plots index", "plots-index"),
    ("Audit trail", "audit-trail"),
    ("Provenance", "provenance"),
    ("Methodology", "methodology"),
)

_RESULT_ALIASES = {
    "indicator_passports": (
        "indicator_passports",
        "indicator_calibration_parameters",
    ),
    "indicator_audit": (
        "indicator_audit",
        "indicator_processing_audit",
        "indicator_processing_events",
        "indicator_aggregation_results",
    ),
    "qc_issues": ("qc_issues", "validation_issues"),
    "failures": ("failures", "failure_summary", "failure_analysis"),
    "pcr_results": ("pcr", "pcr_results"),
    "moduli": ("moduli", "modulus_results"),
    "group_comparisons": ("group_comparison", "group_comparisons"),
    "plots": ("plots", "plots_index"),
}

_CSS = """
:root { color-scheme: light; font-family: Arial, Helvetica, sans-serif; }
body { color: #161616; background: #fff; margin: 0 auto; max-width: 1500px;
       padding: 24px; font-size: 14px; line-height: 1.35; }
h1 { margin: 0 0 8px; font-size: 25px; }
h2 { border-bottom: 2px solid #202020; margin-top: 32px; padding-bottom: 4px; }
h3 { margin: 20px 0 8px; font-size: 16px; }
.report-contract { color: #444; margin-bottom: 18px; }
.table-wrap { overflow-x: auto; margin: 8px 0 18px; }
table { border-collapse: collapse; min-width: 100%; }
th, td { border: 1px solid #aaa; padding: 5px 7px; text-align: left;
         vertical-align: top; }
th { background: #eee; position: sticky; top: 0; }
tbody tr:nth-child(even) { background: #f8f8f8; }
code, pre { font-family: "Courier New", Courier, monospace; white-space: pre-wrap;
            overflow-wrap: anywhere; }
.value-pair { display: grid; gap: 2px; }
.display-value::before { color: #555; content: "display: "; font-size: 11px; }
.machine-value { color: #242424; font-size: 11px; }
.value-pair .machine-value::before { color: #555; content: "machine: "; }
.raw-value { white-space: pre-wrap; }
.raw-value::before { color: #555; content: "raw: "; }
.review-required { background: #fff2cc !important; border-color: #8a6d00; }
.review-badge { background: #8a6d00; color: #fff; display: inline-block;
                font-size: 11px; font-weight: bold; margin: 0 5px 3px 0;
                padding: 2px 5px; }
.review-banner { border: 2px solid #8a6d00; margin: 12px 0; padding: 9px; }
.formula-value { border-left: 3px solid #2d5f88; }
.range-value { border-left: 3px solid #666; }
.empty { color: #666; font-style: italic; }
.blocked-link { color: #8b0000; font-weight: bold; }
.sha256 { font-family: "Courier New", Courier, monospace; overflow-wrap: anywhere; }
.legend { border: 1px solid #aaa; display: inline-grid; gap: 5px; padding: 8px; }
.manifest-status-ok { color: #1c5f26; }
.manifest-status-review-required, .manifest-status-hash-mismatch {
  color: #8b0000; font-weight: bold;
}
@media print {
  @page { margin: 12mm; size: A4 landscape; }
  body { max-width: none; margin: 0; padding: 0; font-size: 9pt; }
  h1, h2, h3 { break-after: avoid; }
  section, table, tr { break-inside: avoid; }
  .table-wrap { overflow: visible; }
  th { position: static; }
  a { color: #000; text-decoration: none; }
  a[href]::after { content: " [" attr(href) "]"; font-size: 8pt; }
}
""".strip()


def _escape(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _json_default(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return to_dict()
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, (bytes, bytearray, memoryview)):
        return {"base64": base64.b64encode(bytes(value)).decode("ascii")}
    if isinstance(value, (set, frozenset)):
        return sorted(value, key=_machine_text)
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return item()
        except (TypeError, ValueError):
            pass
    return str(value)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=_json_default,
        allow_nan=True,
    )


def _machine_text(value: Any) -> str:
    """Return a stable machine representation without parsing source strings."""

    if isinstance(value, str):
        return value
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, Integral):
        return str(int(value))
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, Real):
        return repr(float(value))
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, (bytes, bytearray, memoryview)):
        return base64.b64encode(bytes(value)).decode("ascii")
    if isinstance(value, Mapping) or (
        isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))
    ):
        return _canonical_json(value)
    if is_dataclass(value):
        return _canonical_json(asdict(value))
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return _canonical_json(to_dict())
    return str(value)


def _rounding_rule(
    display_rounding: Mapping[str, Any] | None,
    *,
    section: str,
    column: str,
) -> Any:
    if not isinstance(display_rounding, Mapping):
        return None
    for key in (f"{section}.{column}", column, section, "default"):
        if key in display_rounding:
            return display_rounding[key]
    return None


def _display_number(value: Real | Decimal, rule: Any) -> str:
    numeric = float(value)
    if not math.isfinite(numeric):
        return _machine_text(value)
    if isinstance(rule, Mapping):
        if "decimals" in rule:
            rule = int(rule["decimals"])
        elif "resolution" in rule:
            rule = float(rule["resolution"])
    if isinstance(rule, int) and not isinstance(rule, bool):
        return f"{numeric:.{max(0, rule)}f}"
    if isinstance(rule, Real) and not isinstance(rule, bool) and float(rule) > 0:
        step = Decimal(str(float(rule)))
        rounded = (Decimal(str(numeric)) / step).quantize(
            Decimal("1"), rounding=ROUND_HALF_UP
        ) * step
        decimals = max(0, -step.normalize().as_tuple().exponent)
        return f"{rounded:.{decimals}f}"
    return format(numeric, ".6g")


def _display_text(
    value: Any,
    *,
    section: str,
    column: str,
    display_rounding: Mapping[str, Any] | None,
) -> str:
    rule = _rounding_rule(display_rounding, section=section, column=column)
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (Real, Decimal)):
        return _display_number(value, rule)
    if value is None:
        return "not available"
    return _machine_text(value)


def _as_mapping(value: Any) -> dict[str, Any] | None:
    if isinstance(value, Mapping):
        return {str(key): item for key, item in value.items()}
    if is_dataclass(value):
        return asdict(value)
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        payload = to_dict()
        if isinstance(payload, Mapping):
            return {str(key): item for key, item in payload.items()}
    return None


def _mapping_is_record_collection(value: Mapping[Any, Any]) -> bool:
    return bool(value) and all(_as_mapping(item) is not None for item in value.values())


def _records(value: Any) -> tuple[list[str], list[list[Any]]]:
    """Convert supported table-like values without coercing their cells."""

    if value is None:
        return [], []
    if isinstance(value, pd.DataFrame):
        columns = [str(column) for column in value.columns]
        rows = [list(row) for row in value.itertuples(index=False, name=None)]
        return columns, rows
    if isinstance(value, pd.Series):
        return _records(value.to_frame(name=value.name or "value"))

    mapping = _as_mapping(value)
    if mapping is not None:
        if _mapping_is_record_collection(mapping):
            records: list[dict[str, Any]] = []
            for item_id, item in sorted(mapping.items(), key=lambda pair: pair[0]):
                record = _as_mapping(item) or {}
                records.append({"item_id": item_id, **record})
            return _records(records)
        rows = [[key, mapping[key]] for key in sorted(mapping)]
        return ["field", "value"], rows

    if isinstance(value, Iterable) and not isinstance(value, (str, bytes, bytearray)):
        items = (
            sorted(value, key=_machine_text)
            if isinstance(value, (set, frozenset))
            else list(value)
        )
        if not items:
            return [], []
        mapped = [_as_mapping(item) for item in items]
        if all(item is not None for item in mapped):
            columns = sorted({key for item in mapped for key in (item or {})})
            return columns, [[(item or {}).get(column) for column in columns] for item in mapped]
        return ["value"], [[item] for item in items]
    return ["value"], [[value]]


def _contains_review_required(value: Any) -> bool:
    return "review_required" in _machine_text(value).casefold()


def _column_kind(column: str) -> str | None:
    key = column.casefold().replace(" ", "_")
    if "formula" in key or "equation" in key:
        return "formula"
    if any(
        token in key
        for token in ("range", "interval", "_min", "_max", "lower", "upper", "bound")
    ):
        return "range"
    return None


def _render_cell(
    value: Any,
    *,
    section: str,
    column: str,
    raw: bool,
    display_rounding: Mapping[str, Any] | None,
    kind_override: str | None = None,
) -> str:
    machine = _machine_text(value)
    review = _contains_review_required(value)
    classes: list[str] = []
    kind = kind_override or _column_kind(column)
    if review:
        classes.append("review-required")
    if kind:
        classes.append(f"{kind}-value")
    attributes = f' class="{" ".join(classes)}"' if classes else ""
    if kind:
        attributes += f' data-kind="{kind}"'
    badge = '<span class="review-badge">REVIEW REQUIRED</span>' if review else ""
    if raw:
        digest = hashlib.sha256(machine.encode("utf-8")).hexdigest()
        content = (
            f'{badge}<code class="raw-value machine-value" '
            f'data-raw-sha256="{digest}">{_escape(machine)}</code>'
        )
    else:
        display = _display_text(
            value,
            section=section,
            column=column,
            display_rounding=display_rounding,
        )
        content = (
            f'{badge}<div class="value-pair"><span class="display-value">'
            f'{_escape(display)}</span><code class="machine-value">'
            f'{_escape(machine)}</code></div>'
        )
    return f"<td{attributes}>{content}</td>"


def _render_table(
    value: Any,
    *,
    section: str,
    raw: bool = False,
    display_rounding: Mapping[str, Any] | None = None,
) -> str:
    columns, rows = _records(value)
    if not columns:
        return '<p class="empty">No records supplied.</p>'
    head = "".join(f'<th scope="col">{_escape(column)}</th>' for column in columns)
    body_rows: list[str] = []
    for row in rows:
        row_review = any(_contains_review_required(value) for value in row)
        row_attr = ' class="review-required"' if row_review else ""
        record_kind = (
            _column_kind(_machine_text(row[0]))
            if columns and columns[0].casefold() == "field" and row
            else None
        )
        cells = "".join(
            _render_cell(
                value,
                section=section,
                column=column,
                raw=raw,
                display_rounding=display_rounding,
                kind_override=record_kind if index == 1 else None,
            )
            for index, (column, value) in enumerate(zip(columns, row))
        )
        body_rows.append(f"<tr{row_attr}>{cells}</tr>")
    body = "".join(body_rows) or f'<tr><td colspan="{len(columns)}">No rows.</td></tr>'
    return (
        '<div class="table-wrap"><table><thead><tr>'
        f"{head}</tr></thead><tbody>{body}</tbody></table></div>"
    )


def _flatten_mapping(value: Any, prefix: str = "") -> list[dict[str, Any]]:
    mapping = _as_mapping(value)
    if mapping is None:
        return [{"field": prefix or "value", "value": value}]
    rows: list[dict[str, Any]] = []
    for key in sorted(mapping):
        field = f"{prefix}.{key}" if prefix else key
        child = mapping[key]
        if _as_mapping(child) is not None:
            rows.extend(_flatten_mapping(child, field))
        else:
            rows.append({"field": field, "value": child})
    return rows


def _safe_href(value: Any) -> str | None:
    raw = _machine_text(value).strip()
    if not raw:
        return None
    entity_decoded = html.unescape(raw)
    if any(
        unicodedata.category(character).startswith("C")
        for character in entity_decoded
    ):
        return None
    # Package paths always use literal POSIX separators.  Decoding a separator
    # changes URL structure and is therefore never a legitimate artifact link.
    if re.search(r"%(?:2f|5c)", entity_decoded, flags=re.IGNORECASE):
        return None
    try:
        decoded = unquote(entity_decoded, encoding="utf-8", errors="strict")
        decoded_again = unquote(decoded, encoding="utf-8", errors="strict")
    except (UnicodeDecodeError, ValueError):
        return None
    # A second decoding pass changing the path is an encoded-path ambiguity.
    # Reject it instead of relying on browser/filesystem-specific decoding.
    if decoded_again != decoded:
        return None
    decoded = unicodedata.normalize("NFC", decoded)
    if any(unicodedata.category(character).startswith("C") for character in decoded):
        return None
    compact = re.sub(r"[\x00-\x20\x7f]+", "", decoded).casefold()
    if compact.startswith(("javascript:", "vbscript:", "data:")):
        return None
    try:
        parsed = urlsplit(decoded)
    except ValueError:
        return None
    # A report package links only to members beside the HTML document.  Remote,
    # root-relative, drive-letter and traversal URLs are never emitted.
    if (
        parsed.scheme
        or parsed.netloc
        or parsed.query
        or parsed.fragment
        or "?" in decoded
        or "#" in decoded
    ):
        return None
    if ":" in decoded or decoded.startswith(("/", "\\")) or "\\" in decoded:
        return None
    path_parts = parsed.path.split("/")
    if any(part in {"", ".", ".."} for part in path_parts):
        return None
    return raw


def _iter_artifact_items(value: Any) -> list[tuple[str | None, Any]]:
    if value is None:
        return []
    if isinstance(value, pd.DataFrame):
        return [(None, record) for record in value.to_dict(orient="records")]
    if isinstance(value, Mapping):
        record_keys = {"path", "href", "sha256", "hash", "media_type", "content", "payload"}
        if record_keys.intersection(str(key) for key in value):
            return [(None, value)]
        return [
            (str(key), item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        ]
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes, bytearray)):
        return [(None, item) for item in value]
    return [(None, value)]


def _artifact_record(default_path: str | None, value: Any, *, kind: str) -> dict[str, Any]:
    payload: bytes | None = None
    declared_sha: str | None = None
    path = default_path or "artifact"
    href: Any = path
    media_type = ""
    size: int | None = None

    if isinstance(value, (bytes, bytearray, memoryview)):
        payload = bytes(value)
    else:
        mapping = _as_mapping(value)
        if mapping is not None:
            path = str(
                mapping.get("path")
                or mapping.get("name")
                or mapping.get("artifact")
                or default_path
                or "artifact"
            )
            href = mapping.get("href", mapping.get("url", mapping.get("link", path)))
            raw_sha = mapping.get("sha256", mapping.get("hash"))
            declared_sha = str(raw_sha).casefold() if raw_sha not in (None, "") else None
            media_type = str(mapping.get("media_type", mapping.get("content_type", "")))
            content = mapping.get("content", mapping.get("payload"))
            if isinstance(content, (bytes, bytearray, memoryview)):
                payload = bytes(content)
            bytes_field = mapping.get("bytes", mapping.get("size_bytes", mapping.get("size")))
            # The report-package facade uses ``bytes`` for the integer byte count.
            if isinstance(bytes_field, Integral) and not isinstance(bytes_field, bool):
                size = int(bytes_field)
            elif isinstance(bytes_field, (bytes, bytearray, memoryview)):
                payload = bytes(bytes_field)
        elif isinstance(value, str) and re.fullmatch(r"[0-9a-fA-F]{64}", value):
            declared_sha = value.casefold()
        elif value is not None:
            href = value

    actual_sha = hashlib.sha256(payload).hexdigest() if payload is not None else None
    if payload is not None:
        size = len(payload)
    if actual_sha and declared_sha and actual_sha != declared_sha:
        status = "hash_mismatch"
        sha256 = actual_sha
    elif actual_sha:
        status = "ok"
        sha256 = actual_sha
    elif declared_sha and declared_sha.startswith("excluded:"):
        status = "ok"
        sha256 = declared_sha
    elif declared_sha and re.fullmatch(r"[0-9a-f]{64}", declared_sha):
        status = "ok"
        sha256 = declared_sha
    else:
        status = "review_required"
        sha256 = declared_sha or "not supplied"
    return {
        "kind": kind,
        "path": path,
        "href": href,
        "size": size,
        "media_type": media_type,
        "sha256": sha256,
        "declared_sha256": declared_sha,
        "status": status,
    }


def _artifact_records(plots: Any, artifacts: Any) -> list[dict[str, Any]]:
    records = [
        _artifact_record(path, value, kind=kind)
        for kind, source in (("plot", plots), ("artifact", artifacts))
        for path, value in _iter_artifact_items(source)
    ]
    return sorted(
        records,
        key=lambda row: (
            str(row["path"]),
            str(row["href"]),
            str(row["sha256"]),
            str(row["kind"]),
        ),
    )


def _render_artifact_manifest(plots: Any, artifacts: Any) -> str:
    records = _artifact_records(plots, artifacts)
    if not records:
        return '<p class="empty">No plot or artifact records supplied.</p>'
    rows: list[str] = []
    for record in records:
        href = _safe_href(record["href"])
        path = _escape(record["path"])
        if href is None:
            link = (
                f'<span class="blocked-link">Unsafe link blocked</span> '
                f'<code class="machine-value">{_escape(record["href"])}</code>'
            )
        else:
            link = f'<a href="{_escape(href)}">{path}</a>'
        status = str(record["status"])
        review_class = ' class="review-required"' if status != "ok" else ""
        declared = record["declared_sha256"]
        rows.append(
            f"<tr{review_class}><td>{_escape(record['kind'])}</td><td>{path}</td>"
            f"<td>{link}</td><td>{_escape(record['size'] if record['size'] is not None else 'not supplied')}</td>"
            f"<td>{_escape(record['media_type'])}</td>"
            f'<td class="sha256">{_escape(record["sha256"])}</td>'
            f'<td class="sha256">{_escape(declared or "not supplied")}</td>'
            f'<td class="manifest-status-{_escape(status)}">{_escape(status)}</td></tr>'
        )
    return (
        '<p>Artifact links and SHA-256 manifest:</p><div class="table-wrap"><table>'
        "<thead><tr><th>Kind</th><th>Path</th><th>Artifact link</th><th>Bytes</th>"
        "<th>Media type</th><th>SHA-256</th><th>Declared SHA-256</th><th>Status</th>"
        f"</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"
    )


def _result_values(
    explicit: Any,
    *,
    name: str,
    result_tables: Mapping[str, Any],
) -> tuple[list[tuple[str, Any]], set[str]]:
    if explicit is not None:
        return (
            [(name.replace("_", " "), explicit)],
            {alias for alias in _RESULT_ALIASES[name] if alias in result_tables},
        )
    entries: list[tuple[str, Any]] = []
    consumed: set[str] = set()
    for alias in _RESULT_ALIASES[name]:
        if alias in result_tables:
            entries.append((alias.replace("_", " "), result_tables[alias]))
            consumed.add(alias)
    return entries, consumed


def _prepared_attr(prepared: Any, names: Sequence[str]) -> Any:
    attrs = getattr(prepared, "attrs", None)
    if isinstance(attrs, Mapping):
        for name in names:
            if name in attrs:
                return attrs[name]
    return None


def _audit_value(audit: Any) -> Any:
    if audit is None:
        return None
    events = getattr(audit, "events", None)
    return events if events is not None else audit


def _render_entries(
    entries: Sequence[tuple[str, Any]],
    *,
    section: str,
    raw: bool = False,
    display_rounding: Mapping[str, Any] | None = None,
) -> str:
    if not entries:
        return '<p class="empty">No records supplied.</p>'
    rendered: list[str] = []
    for label, value in entries:
        rendered.append(f"<h3>{_escape(label)}</h3>")
        rendered.append(
            _render_table(
                value,
                section=section,
                raw=raw,
                display_rounding=display_rounding,
            )
        )
    return "".join(rendered)


def _review_registry_entry(source: str, location: str, value: Any) -> dict[str, Any]:
    return {
        "source": source,
        "location": location or "value",
        "status": "review_required",
        "detail": value,
    }


def _explicit_review_registry(value: Any) -> list[dict[str, Any]]:
    if value is None or value is False:
        return []
    if value is True:
        return [_review_registry_entry("report", "review_required", True)]
    if isinstance(value, str):
        return (
            [_review_registry_entry("report", "review_required", value)] if value else []
        )
    mapping = _as_mapping(value)
    if mapping is not None:
        return [
            _review_registry_entry("report", f"review_required.{key}", mapping[key])
            for key in sorted(mapping)
        ]
    if isinstance(value, Iterable) and not isinstance(value, (bytes, bytearray)):
        items = (
            sorted(value, key=_machine_text)
            if isinstance(value, (set, frozenset))
            else value
        )
        return [
            _review_registry_entry("report", f"review_required[{index}]", item)
            for index, item in enumerate(items)
        ]
    return [_review_registry_entry("report", "review_required", value)]


def _collect_review_required(
    value: Any,
    *,
    source: str,
    location: str = "",
) -> list[dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, pd.DataFrame):
        records: list[dict[str, Any]] = []
        for row_no, row in enumerate(value.itertuples(index=False, name=None)):
            for column, cell in zip(value.columns, row):
                records.extend(
                    _collect_review_required(
                        cell,
                        source=source,
                        location=f"row[{row_no}].{column}",
                    )
                )
        return records
    mapping = _as_mapping(value)
    if mapping is not None:
        records = []
        for key in sorted(mapping):
            child_location = f"{location}.{key}" if location else key
            records.extend(
                _collect_review_required(
                    mapping[key], source=source, location=child_location
                )
            )
        return records
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes, bytearray)):
        records = []
        for index, item in enumerate(value):
            child_location = f"{location}[{index}]" if location else f"[{index}]"
            records.extend(
                _collect_review_required(item, source=source, location=child_location)
            )
        return records
    if _contains_review_required(value):
        return [_review_registry_entry(source, location, value)]
    return []


def _review_banner(registry: Sequence[Mapping[str, Any]]) -> str:
    if not registry:
        return ""
    detail = f"{len(registry)} report item(s) require engineering review."
    return (
        '<aside class="review-banner review-required"><span class="review-badge">'
        f"REVIEW REQUIRED</span>{_escape(detail)}</aside>"
    )


def build_html_report_package(
    *,
    metadata: Mapping[str, Any] | None,
    raw: Any,
    prepared: Any,
    indicator_passports: Any = None,
    indicator_audit: Any = None,
    qc_issues: Any = None,
    failures: Any = None,
    pcr_results: Any = None,
    moduli: Any = None,
    group_comparisons: Any = None,
    plots: Any = None,
    artifacts: Any = None,
    audit: Any = None,
    provenance: Any = None,
    methodology: Any = None,
    formulas: Any = None,
    display_rounding: Mapping[str, Any] | None = None,
    review_required: Any = None,
    result_tables: Mapping[str, Any] | None = None,
    title: str | None = None,
) -> bytes:
    """Build a deterministic standalone UTF-8 HTML report.

    ``raw`` is rendered cell-for-cell from its existing Python values.  In
    particular, strings are never parsed as numbers, stripped or rounded.
    Other tables show a human-readable value separately from the exact machine
    representation.  ``display_rounding`` may map a column (or
    ``"Section.column"``) to decimal places or to a numerical resolution.

    Artifact records may contain ``path``, ``href``, integer ``bytes``,
    ``sha256`` and ``media_type``.  Byte payloads passed directly, or through a
    ``content``/``payload`` field, are hashed by this renderer.  Unsafe URL
    schemes are shown as text but never emitted as links.
    """

    metadata_dict = dict(metadata or {})
    tables = dict(result_tables or {})
    consumed: set[str] = set()

    if indicator_passports is None:
        indicator_passports = _prepared_attr(
            prepared, ("indicator_calibration_parameters", "indicator_passports")
        )
    if indicator_audit is None:
        prepared_audit = _prepared_attr(
            prepared, ("indicator_processing_audit", "indicator_audit")
        )
        if prepared_audit is not None:
            indicator_audit = prepared_audit

    section_entries: dict[str, list[tuple[str, Any]]] = {
        "Project passport": [("Metadata and project passport", _flatten_mapping(metadata_dict))],
        "Raw data": [("Lossless source values", raw)],
        "Prepared data": [("Prepared machine values", prepared)],
        "Indicator passports": [],
        "Indicator audit": [],
        "QC issues": [],
        "Failure/censoring": [],
        "pcr": [],
        "Moduli": [],
        "Group comparison": [],
        "Audit trail": [("Append-only events", _audit_value(audit))],
        "Provenance": [("Hashes, versions and sources", _flatten_mapping(provenance))],
        "Methodology": [("Method definitions", _flatten_mapping(methodology))],
    }

    explicit_values = {
        "indicator_passports": indicator_passports,
        "indicator_audit": indicator_audit,
        "qc_issues": qc_issues,
        "failures": failures,
        "pcr_results": pcr_results,
        "moduli": moduli,
        "group_comparisons": group_comparisons,
    }
    section_for_name = {
        "indicator_passports": "Indicator passports",
        "indicator_audit": "Indicator audit",
        "qc_issues": "QC issues",
        "failures": "Failure/censoring",
        "pcr_results": "pcr",
        "moduli": "Moduli",
        "group_comparisons": "Group comparison",
    }
    for name, explicit in explicit_values.items():
        entries, used = _result_values(explicit, name=name, result_tables=tables)
        section_entries[section_for_name[name]].extend(entries)
        consumed.update(used)

    plot_entries, used = _result_values(plots, name="plots", result_tables=tables)
    consumed.update(used)
    if plots is None and plot_entries:
        plots = [value for _, value in plot_entries]

    if formulas is not None:
        section_entries["Methodology"].append(
            ("Formulas and ranges", _flatten_mapping(formulas))
        )

    unknown_tables = sorted(set(tables) - consumed)
    for key in unknown_tables:
        section_entries["Prepared data"].append((f"Additional result table: {key}", tables[key]))

    report_title = title
    if report_title is None:
        report_title = str(metadata_dict.get("project") or "Soil Stamp report package")

    review_sources = {
        "metadata": metadata_dict,
        "prepared": prepared,
        "indicator_passports": indicator_passports,
        "indicator_audit": indicator_audit,
        "qc_issues": qc_issues,
        "failures": failures,
        "pcr_results": pcr_results,
        "moduli": moduli,
        "group_comparisons": group_comparisons,
        "audit": _audit_value(audit),
        "provenance": provenance,
        "methodology": methodology,
        "formulas": formulas,
    }
    review_registry = _explicit_review_registry(review_required)
    for source, value in review_sources.items():
        review_registry.extend(_collect_review_required(value, source=source))
    for record in _artifact_records(plots, artifacts):
        if record["status"] != "ok":
            review_registry.append(
                _review_registry_entry(
                    "artifacts", str(record["path"]), record["status"]
                )
            )
    review_registry = sorted(
        review_registry,
        key=lambda row: (
            str(row["source"]),
            str(row["location"]),
            str(row["status"]),
            _machine_text(row.get("detail")),
        ),
    )
    if review_registry:
        section_entries["QC issues"].append(
            ("Review-required registry", review_registry)
        )

    document = [
        "<!doctype html>",
        '<html lang="en">',
        "<head>",
        '<meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        (
            '<meta http-equiv="Content-Security-Policy" content="default-src &#39;none&#39;; '
            "style-src &#39;unsafe-inline&#39;; img-src &#39;self&#39; data:; base-uri &#39;none&#39;; "
            'form-action &#39;none&#39;">'
        ),
        f"<title>{_escape(report_title)}</title>",
        f"<style>{_CSS}</style>",
        "</head>",
        "<body>",
        f"<h1>{_escape(report_title)}</h1>",
        (
            '<p class="report-contract">Standalone UTF-8 report; no JavaScript or remote assets. '
            "Raw source strings are not numerically converted.</p>"
        ),
        _review_banner(review_registry),
        (
            '<div class="legend"><strong>Value display contract</strong>'
            '<span class="display-value">human-readable rounded value</span>'
            '<code class="machine-value">machine precision</code>'
            '<code class="raw-value">lossless source text</code></div>'
        ),
    ]

    for section, section_id in _SECTIONS:
        document.append(f'<section id="{section_id}"><h2>{_escape(section)}</h2>')
        if section == "Plots index":
            document.append(_render_artifact_manifest(plots, artifacts))
        else:
            if section == "Raw data":
                document.append(
                    '<p><strong>Representation notice:</strong> this table preserves the supplied '
                    "cell strings for inspection; the exact source artifact listed in the manifest "
                    "is authoritative.</p>"
                )
            document.append(
                _render_entries(
                    section_entries[section],
                    section=section,
                    raw=section == "Raw data",
                    display_rounding=display_rounding,
                )
            )
        document.append("</section>")
    document.extend(["</body>", "</html>", ""])
    return "\n".join(document).encode("utf-8")


__all__ = ["build_html_report_package"]
