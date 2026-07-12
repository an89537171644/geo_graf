"""Deterministically materialize a synthetic XLSX acceptance input.

This helper is acceptance-only.  It exists so the Excel/manual equivalence case
must pass through the production strict XLSX importer instead of comparing a
CSV surrogate with a manual draft.
"""

from __future__ import annotations

import io
import json
import re
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any

from openpyxl import Workbook


CONTRACT_VERSION = "declarative-xlsx-projection/1.0"
_FIXED_TIMESTAMP = datetime(2000, 1, 1, 0, 0, 0)
_ZIP_TIMESTAMP = (2000, 1, 1, 0, 0, 0)


def _load_projection(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("contract_version") != CONTRACT_VERSION:
        raise ValueError(f"{path}: unsupported XLSX projection contract.")
    if payload.get("binary_xlsx_claimed") is not False:
        raise ValueError(f"{path}: declarative source must not claim a pre-existing binary XLSX.")
    headers = payload.get("headers")
    rows = payload.get("rows")
    if not isinstance(headers, list) or not headers or not all(
        isinstance(value, str) and value for value in headers
    ):
        raise ValueError(f"{path}: headers must be a non-empty string list.")
    if not isinstance(rows, list) or not all(
        isinstance(row, list) and len(row) == len(headers) for row in rows
    ):
        raise ValueError(f"{path}: every row must match the header width.")
    return payload


def _normalise_ooxml(payload: bytes) -> bytes:
    source_buffer = io.BytesIO(payload)
    output = io.BytesIO()
    with zipfile.ZipFile(source_buffer, "r") as source, zipfile.ZipFile(
        output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as target:
        names = source.namelist()
        if len(names) != len(set(names)):
            raise ValueError("Generated XLSX unexpectedly contains duplicate OOXML members.")
        for name in sorted(names):
            member = source.read(name)
            if name == "docProps/core.xml":
                member = re.sub(
                    rb"(<dcterms:modified[^>]*>).*?(</dcterms:modified>)",
                    rb"\g<1>2000-01-01T00:00:00Z\g<2>",
                    member,
                )
            info = zipfile.ZipInfo(name, date_time=_ZIP_TIMESTAMP)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 0
            info.external_attr = 0
            target.writestr(
                info,
                member,
                compress_type=zipfile.ZIP_DEFLATED,
                compresslevel=9,
            )
    return output.getvalue()


def materialize_equivalent_xlsx(projection_path: Path, output_path: Path) -> Path:
    """Create one deterministic XLSX and return its resolved output path."""

    projection_path = Path(projection_path).resolve()
    output_path = Path(output_path).resolve()
    payload = _load_projection(projection_path)
    if output_path.suffix.casefold() != ".xlsx":
        raise ValueError("The materialized equivalence input must have an .xlsx suffix.")

    workbook = Workbook()
    worksheet = workbook.active
    worksheet.title = str(payload.get("sheet_name") or "Protocol")
    for row in [payload["headers"], *payload["rows"]]:
        worksheet.append(row)
    for row in worksheet.iter_rows():
        for cell in row:
            if isinstance(cell.value, str):
                cell.data_type = "s"
    workbook.properties.creator = "Soil Stamp acceptance fixture"
    workbook.properties.lastModifiedBy = "Soil Stamp acceptance fixture"
    workbook.properties.created = _FIXED_TIMESTAMP
    workbook.properties.modified = _FIXED_TIMESTAMP
    workbook.calculation.fullCalcOnLoad = False
    workbook.calculation.forceFullCalc = False
    workbook.calculation.calcMode = "manual"

    buffer = io.BytesIO()
    workbook.save(buffer)
    workbook.close()
    rendered = _normalise_ooxml(buffer.getvalue())
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(rendered)
    return output_path


__all__ = ["materialize_equivalent_xlsx"]
