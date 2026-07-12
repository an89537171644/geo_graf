"""File hashes, software provenance and project-passport completeness."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, BinaryIO, Iterable

from .schema import ProvenanceRecord, ValidationIssue, VERSION


def source_bytes(source: str | Path | bytes | bytearray | BinaryIO | None) -> bytes:
    if source is None:
        return b""
    if isinstance(source, (bytes, bytearray)):
        return bytes(source)
    if isinstance(source, (str, Path)):
        return Path(source).read_bytes()
    position = source.tell() if hasattr(source, "tell") else None
    data = source.read()
    if position is not None and hasattr(source, "seek"):
        source.seek(position)
    return data.encode("utf-8") if isinstance(data, str) else bytes(data)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")


def source_sha256(source: str | Path | bytes | bytearray | BinaryIO | None) -> str:
    return sha256_bytes(source_bytes(source))


def value_sha256(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def current_git_commit(root: str | Path | None = None) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(root) if root else None,
            check=True,
            capture_output=True,
            text=True,
            timeout=3,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    value = completed.stdout.strip()
    return value or None


def current_git_dirty(root: str | Path | None = None) -> bool | None:
    try:
        completed = subprocess.run(
            ["git", "status", "--porcelain=v1", "--untracked-files=normal"],
            cwd=str(root) if root else None,
            check=True,
            capture_output=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    return bool(completed.stdout.strip())


def source_tree_sha256(root: str | Path | None = None) -> str | None:
    if root is None:
        return None
    base = Path(root).resolve()
    try:
        completed = subprocess.run(
            ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
            cwd=str(base),
            check=True,
            capture_output=True,
            timeout=10,
        )
        relative_paths = [os.fsdecode(value) for value in completed.stdout.split(b"\0") if value]
    except (FileNotFoundError, subprocess.SubprocessError):
        ignored_parts = {".git", ".venv", "__pycache__", ".pytest_cache", "work"}
        relative_paths = [
            path.relative_to(base).as_posix()
            for path in base.rglob("*")
            if path.is_file()
            and not ignored_parts.intersection(path.relative_to(base).parts)
            and not any(part.endswith(".egg-info") for part in path.relative_to(base).parts)
        ]
    digest = hashlib.sha256()
    included = 0
    for relative in sorted(relative_paths, key=lambda value: value.replace("\\", "/")):
        path = (base / relative).resolve()
        try:
            path.relative_to(base)
        except ValueError:
            continue
        if not path.is_file():
            continue
        normalized = Path(relative).as_posix().encode("utf-8", errors="surrogateescape")
        digest.update(normalized)
        digest.update(b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
        included += 1
    return digest.hexdigest() if included else None


def dependency_versions() -> dict[str, str]:
    result: dict[str, str] = {}
    for distribution in (
        "numpy",
        "pandas",
        "scipy",
        "matplotlib",
        "streamlit",
        "openpyxl",
        "defusedxml",
    ):
        try:
            result[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            result[distribution] = "not-installed"
    result["platform"] = platform.platform()
    return result


def build_provenance(
    *,
    input_source: str | Path | bytes | bytearray | BinaryIO,
    metadata_source: str | Path | bytes | bytearray | BinaryIO | dict[str, Any] | None,
    config: dict[str, Any],
    project_root: str | Path | None = None,
) -> ProvenanceRecord:
    if isinstance(metadata_source, dict):
        metadata_hash = value_sha256(metadata_source)
    else:
        metadata_hash = source_sha256(metadata_source)
    return ProvenanceRecord(
        input_file_sha256=source_sha256(input_source),
        metadata_sha256=metadata_hash,
        config_sha256=value_sha256(config),
        program_version=VERSION,
        git_commit=current_git_commit(project_root),
        git_dirty=current_git_dirty(project_root),
        source_tree_sha256=source_tree_sha256(project_root),
        python_version=sys.version.split()[0],
        dependency_versions=dependency_versions(),
    )


def load_conversion_formula(metadata: dict[str, Any]) -> str:
    kind = str(metadata.get("load_kind", "force")).casefold()
    unit = metadata.get("load_unit", "kN")
    factor = metadata.get("load_factor", 1.0)
    lever = metadata.get("lever_ratio", 1.0)
    zero = metadata.get("load_zero", 0.0)
    if kind in {"pressure", "p", "давление"}:
        return f"p_kPa = convert((load_raw − {zero}) × {factor}, {unit} → kPa)"
    return (
        f"F_kN = convert((load_raw − {zero}) × {factor} × {lever}, "
        f"{unit} → kN); p_kPa = F_kN / A_m²"
    )


def _present(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, dict, set)):
        return bool(value)
    return True


def _verified_instrument_inventory(value: Any) -> Any | None:
    if isinstance(value, dict):
        records = [value]
    elif isinstance(value, list):
        records = value
    else:
        return None
    if not records or not all(isinstance(item, dict) for item in records):
        return None
    for item in records:
        instrument_id = item.get("instrument_id")
        calibration_reference = (
            item.get("calibration_date")
            or item.get("calibration_certificate")
            or item.get("certificate_id")
        )
        if not _present(instrument_id) or not _present(calibration_reference):
            return None
    return value


def passport_completeness(
    metadata: dict[str, Any], test_ids: Iterable[str] | None = None
) -> dict[str, Any]:
    passport = metadata.get("project_passport") or {}
    if not isinstance(passport, dict):
        passport = {}
    reinforcement_value = metadata.get("reinforcement")
    reinforcement = reinforcement_value if isinstance(reinforcement_value, dict) else {}
    soil_value = metadata.get("soil")
    soil = soil_value if isinstance(soil_value, dict) else {}
    box_value = metadata.get("box")
    box = box_value if isinstance(box_value, dict) else {}
    tests_value = metadata.get("tests")
    tests = tests_value if isinstance(tests_value, dict) else {}
    if test_ids is not None:
        selected_ids = {str(value) for value in test_ids}
        tests = {str(key): value for key, value in tests.items() if str(key) in selected_ids}
    test_reinforcement = {
        str(test_id): values.get("reinforcement")
        for test_id, values in tests.items()
        if isinstance(values, dict) and values.get("reinforcement")
    } if isinstance(tests, dict) else {}
    test_pairs = sorted(
        {
            str(values.get("pair_id"))
            for values in tests.values()
            if isinstance(values, dict) and values.get("pair_id")
        }
    ) if isinstance(tests, dict) else []
    test_baseline_groups = sorted(
        {
            str(values.get("baseline_group"))
            for values in tests.values()
            if isinstance(values, dict) and values.get("baseline_group")
        }
    ) if isinstance(tests, dict) else []
    geometry = metadata.get("stamp_diameter_mm") or metadata.get("stamp_area_m2")
    values = {
        "project_id": passport.get("project_id") or metadata.get("project_id"),
        "series_name": passport.get("series_name") or metadata.get("series_name"),
        "reinforcement_status": passport.get("reinforcement_status")
        or (reinforcement.get("type") if isinstance(reinforcement, dict) else None)
        or (sorted({str(item.get('type')) for item in test_reinforcement.values() if isinstance(item, dict) and item.get('type')}) or None),
        "baseline_group": passport.get("baseline_group")
        or metadata.get("baseline_group")
        or test_baseline_groups
        or None,
        "pair_id": passport.get("pair_id") or metadata.get("pair_id") or test_pairs or None,
        "soil_batch": passport.get("soil_batch") or metadata.get("soil_batch") or soil.get("batch"),
        "experiment_date": passport.get("experiment_date") or metadata.get("experiment_date"),
        "operator": passport.get("operator") or metadata.get("operator"),
        "stamp_geometry": geometry,
        "tray_dimensions_mm": passport.get("tray_dimensions_mm")
        or metadata.get("tray_dimensions_mm")
        or box,
        "dry_density_kg_m3": passport.get("dry_density_kg_m3")
        or metadata.get("dry_density_kg_m3")
        or soil.get("dry_density_kg_m3"),
        "moisture_percent": passport.get("moisture_percent")
        or metadata.get("moisture_percent")
        or soil.get("water_content_percent"),
        "soil_type": passport.get("soil_type")
        or metadata.get("soil_type")
        or soil.get("type")
        or soil.get("gradation"),
        "reinforcement_scheme": reinforcement or test_reinforcement,
        "instruments_and_calibration": _verified_instrument_inventory(
            passport.get("instruments")
            if isinstance(passport.get("instruments"), (list, dict))
            else metadata.get("instruments")
        ),
    }
    optional = {"baseline_group", "pair_id"}
    missing = [
        name
        for name, value in values.items()
        if name not in optional and not _present(value)
    ]
    return {
        "complete": not missing,
        "provided": [name for name, value in values.items() if _present(value)],
        "missing": missing,
        "fields": values,
    }


def effective_conversion_parameters(
    metadata: dict[str, Any], test_ids: Iterable[str]
) -> list[dict[str, Any]]:
    """Return the effective load/pressure formula for every included test."""

    force_to_kn = {
        "n": 0.001,
        "н": 0.001,
        "kn": 1.0,
        "кн": 1.0,
        "mn": 1000.0,
        "мн": 1000.0,
        "kgf": 0.00980665,
        "кгс": 0.00980665,
        "tf": 9.80665,
        "тс": 9.80665,
    }
    pressure_to_kpa = {
        "pa": 0.001,
        "па": 0.001,
        "kpa": 1.0,
        "кпа": 1.0,
        "mpa": 1000.0,
        "мпа": 1000.0,
    }
    tests = metadata.get("tests") if isinstance(metadata.get("tests"), dict) else {}
    rows: list[dict[str, Any]] = []
    for test_id in test_ids:
        effective = dict(metadata)
        override = tests.get(str(test_id)) if isinstance(tests, dict) else None
        if isinstance(override, dict):
            effective.update(override)
        kind_raw = str(effective.get("load_kind", "force")).strip().casefold()
        kind = "pressure" if kind_raw in {"pressure", "p", "давление"} else "force"
        unit = str(effective.get("load_unit", ""))
        unit_key = unit.strip().casefold().replace(" ", "")
        zero = float(effective.get("load_zero", 0.0))
        factor = float(effective.get("load_factor", 1.0))
        lever = float(effective.get("lever_ratio", 1.0))
        diameter = effective.get("stamp_diameter_mm")
        area = effective.get("stamp_area_m2")
        try:
            area_value = float(area) if area is not None else None
        except (TypeError, ValueError):
            area_value = None
        try:
            diameter_value = float(diameter) if diameter is not None else None
        except (TypeError, ValueError):
            diameter_value = None
        if area_value is None and diameter_value and diameter_value > 0:
            area_value = 3.141592653589793 * (diameter_value / 1000.0) ** 2 / 4.0
        if kind == "force":
            unit_factor = force_to_kn.get(unit_key)
            coefficient = unit_factor * factor * lever if unit_factor is not None else None
            formula = (
                f"F_kN=(load_raw−{zero:g})×{coefficient:.12g}; "
                + (
                    f"p_kPa=F_kN/{area_value:.12g}"
                    if area_value
                    else "p_kPa=недоступно без A"
                )
                if coefficient is not None
                else "единица силы не поддерживается"
            )
        else:
            unit_factor = pressure_to_kpa.get(unit_key)
            coefficient = unit_factor * factor if unit_factor is not None else None
            formula = (
                f"p_kPa=(load_raw−{zero:g})×{coefficient:.12g}; "
                + (f"F_kN=p_kPa×{area_value:.12g}" if area_value else "F_kN=недоступно без A")
                if coefficient is not None
                else "единица давления не поддерживается"
            )
        rows.append(
            {
                "test_id": str(test_id),
                "load_kind": kind,
                "load_unit": unit,
                "load_zero": zero,
                "load_factor": factor,
                "lever_ratio": lever,
                "stamp_diameter_mm": diameter_value,
                "stamp_area_m2": area_value,
                "formula": formula,
            }
        )
    return rows


def validate_project_metadata(
    metadata: dict[str, Any], *, strict: bool = True
) -> list[ValidationIssue]:
    """Check explicit import-critical fields without silently relying on defaults."""

    issues: list[ValidationIssue] = []
    # Compatibility mode may relax column discovery, never physical units or
    # calibration inputs. Missing critical metadata remains blocking.
    level = "error"
    required = (
        "load_kind",
        "load_unit",
        "load_factor",
        "lever_ratio",
        "settlement_unit",
    )
    for name in required:
        value = metadata.get(name)
        if value is None or (isinstance(value, str) and not value.strip()):
            issues.append(
                ValidationIssue(
                    level,
                    "missing_explicit_metadata",
                    f"В новом проекте поле metadata.{name} должно быть задано явно.",
                    column=name,
                    suggested_action="Заполните поле в metadata JSON; default не подставляется молча.",
                )
            )
    common_indicator_passport = metadata.get("indicator_passport")
    passport_has_division = bool(
        isinstance(common_indicator_passport, dict)
        and any(
            common_indicator_passport.get(name) is not None
            for name in ("division_mm", "resolution_mm", "scale_division_mm")
        )
    )
    for container_name in ("indicator_passports", "indicator_channels"):
        container = metadata.get(container_name)
        if not isinstance(container, dict):
            continue
        passport_has_division = passport_has_division or any(
            isinstance(item, dict)
            and any(
                item.get(name) is not None
                for name in ("division_mm", "resolution_mm", "scale_division_mm")
            )
            for item in container.values()
        )
    if metadata.get("indicator_resolution_mm") is None and not passport_has_division:
        issues.append(
            ValidationIssue(
                level,
                "missing_explicit_metadata",
                "Задайте metadata.indicator_resolution_mm либо division_mm в поканальном паспорте.",
                column="indicator_resolution_mm",
                suggested_action="Заполните цену деления в metadata JSON; default не подставляется молча.",
            )
        )
    for section, allowed_types in (
        ("project_passport", (dict,)),
        ("soil", (dict,)),
        ("box", (dict,)),
        ("tests", (dict,)),
        ("instruments", (dict, list)),
        ("reinforcement", (dict,)),
    ):
        value = metadata.get(section)
        if value is not None and not isinstance(value, allowed_types):
            issues.append(
                ValidationIssue(
                    "error",
                    "invalid_metadata_section",
                    f"metadata.{section} имеет недопустимый тип {type(value).__name__}.",
                    column=section,
                    raw_value=value,
                    suggested_action="Используйте структуру JSON, описанную в README.",
                )
            )
    kind = str(metadata.get("load_kind", "force")).strip().casefold()
    geometry_present = bool(metadata.get("stamp_diameter_mm") or metadata.get("stamp_area_m2"))
    if not geometry_present:
        geometry_level = level if kind not in {"pressure", "p", "давление"} else "warning"
        issues.append(
            ValidationIssue(
                geometry_level,
                "missing_explicit_geometry",
                "Не задан stamp_diameter_mm или stamp_area_m2.",
                column="stamp_diameter_mm/stamp_area_m2",
                suggested_action="Укажите проверенную геометрию штампа в паспорте проекта.",
            )
        )
    if "load_zero" not in metadata:
        issues.append(
            ValidationIssue(
                "warning",
                "load_zero_not_explicit",
                "load_zero не задан явно; в формуле будет использовано 0.",
                column="load_zero",
                suggested_action="Подтвердите нуль нагрузочного канала в metadata.",
            )
        )
    return issues
