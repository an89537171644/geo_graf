from __future__ import annotations

from soilstamp.provenance import (
    build_provenance,
    load_conversion_formula,
    passport_completeness,
    source_sha256,
    validate_project_metadata,
    value_sha256,
)


def test_hashes_are_repeatable_and_sensitive_to_config() -> None:
    payload = b"same source bytes"
    config = {"import_mode": "strict", "seed": 202604}

    assert source_sha256(payload) == source_sha256(payload)
    assert value_sha256(config) == value_sha256({"seed": 202604, "import_mode": "strict"})
    assert value_sha256(config) != value_sha256({**config, "seed": 1})


def test_provenance_separates_input_metadata_and_config_hashes(tmp_path) -> None:
    source = tmp_path / "protocol.csv"
    metadata = tmp_path / "metadata.json"
    source.write_bytes(b"test_id,stage,load\nT1,1,1\n")
    metadata.write_bytes(b'{"load_unit":"kN"}')

    first = build_provenance(
        input_source=source,
        metadata_source=metadata,
        config={"import_mode": "strict"},
        project_root=tmp_path,
    )
    second = build_provenance(
        input_source=source,
        metadata_source=metadata,
        config={"import_mode": "strict"},
        project_root=tmp_path,
    )

    assert first.input_file_sha256 == second.input_file_sha256
    assert first.metadata_sha256 == second.metadata_sha256
    assert first.config_sha256 == second.config_sha256
    assert first.input_file_sha256 != first.metadata_sha256
    assert first.program_version == "0.4.1a2"
    assert "openpyxl" in first.dependency_versions
    assert "defusedxml" in first.dependency_versions
    assert first.source_tree_sha256 == second.source_tree_sha256
    assert first.source_tree_sha256 is not None


def test_passport_reports_missing_fields_instead_of_filling_them() -> None:
    status = passport_completeness(
        {
            "project_passport": {"project_id": "P-01", "operator": "Иванов"},
            "stamp_diameter_mm": 300,
            "reinforcement": {"type": "none", "layers": 0},
        }
    )

    assert status["complete"] is False
    assert "project_id" in status["provided"]
    assert "operator" in status["provided"]
    assert "stamp_geometry" in status["provided"]
    assert "soil_batch" in status["missing"]
    assert "instruments_and_calibration" in status["missing"]


def test_passport_requires_instrument_id_and_calibration_reference() -> None:
    incomplete = passport_completeness(
        {"project_passport": {"instruments": [{"instrument_id": "IND-1"}]}}
    )
    complete_inventory = passport_completeness(
        {
            "project_passport": {
                "instruments": [
                    {"instrument_id": "IND-1", "calibration_date": "2026-06-01"}
                ]
            }
        }
    )

    assert "instruments_and_calibration" in incomplete["missing"]
    assert "instruments_and_calibration" in complete_inventory["provided"]


def test_load_conversion_formula_exposes_zero_factor_and_lever() -> None:
    formula = load_conversion_formula(
        {
            "load_kind": "force",
            "load_unit": "kgf",
            "load_zero": 2.0,
            "load_factor": 0.5,
            "lever_ratio": 10.0,
        }
    )
    assert "load_raw − 2.0" in formula
    assert "0.5" in formula
    assert "10.0" in formula
    assert "kgf → kN" in formula


def test_strict_project_requires_explicit_units_and_geometry() -> None:
    strict = validate_project_metadata({"load_kind": "force"}, strict=True)
    heuristic = validate_project_metadata({"load_kind": "force"}, strict=False)

    assert any(item.code == "missing_explicit_metadata" and item.blocks_processing for item in strict)
    assert any(item.code == "missing_explicit_geometry" and item.blocks_processing for item in strict)
    assert any(item.blocks_processing for item in heuristic)


def test_channel_passport_division_replaces_legacy_global_resolution() -> None:
    metadata = {
        "load_kind": "force",
        "load_unit": "kN",
        "load_factor": 1.0,
        "lever_ratio": 1.0,
        "settlement_unit": "mm",
        "stamp_area_m2": 0.1,
        "indicator_passports": {"indicator_1": {"division_mm": 0.01}},
    }

    issues = validate_project_metadata(metadata, strict=True)

    assert not any(
        item.code == "missing_explicit_metadata"
        and item.column == "indicator_resolution_mm"
        for item in issues
    )


def test_malformed_nested_metadata_is_reported_without_passport_crash() -> None:
    status = passport_completeness({"soil": "sand", "box": 42, "instruments": "I-1"})
    issues = validate_project_metadata(
        {"soil": "sand", "box": 42, "instruments": "I-1"}, strict=True
    )

    assert status["complete"] is False
    assert sum(item.code == "invalid_metadata_section" for item in issues) == 3
