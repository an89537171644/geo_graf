"""Command-line batch analysis for reproducible, non-interactive runs."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .analysis import calculate_moduli_for_test, fit_segmented_pcr
from .data import (
    AuditTrail,
    apply_settlement_correction,
    failure_analysis_summary,
    failure_summary,
    prepare_measurements,
)
from .indicators import (
    indicator_aggregation_frame,
    indicator_audit_frame,
    indicator_event_frame,
    indicator_passport_frame,
)
from .io import read_metadata_json, read_protocol, validate_import_metadata_consistency
from .methodology import (
    ModulusOverrides,
    modulus_profile_definitions,
    modulus_profile_ids,
    parse_pressure_range,
)
from .plotting import (
    export_figure,
    plot_curves,
    plot_failure_intervals,
    resolve_curve_selections,
)
from .provenance import (
    build_provenance,
    effective_conversion_parameters,
    metrology_evaluations_from_passports,
    passport_completeness,
    validate_project_metadata,
)
from .report_package import (
    build_approval_report_package,
    build_formula_and_range_records,
    build_review_required_registry,
    collect_approval_artifacts,
)
from .reporting import build_markdown_report, reproducibility_bundle
from .schema import ValidationIssue


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="soil-stamp",
        description="Обработка штамповых испытаний и публикационный график Antonov.",
    )
    parser.add_argument("protocol", type=Path, help="long-table CSV/XLSX или legacy XLSX")
    parser.add_argument("metadata", type=Path, help="metadata JSON")
    parser.add_argument("--out", type=Path, default=Path("soil_stamp_results"))
    parser.add_argument(
        "--correction",
        choices=["raw", "zero_shifted", "seating_corrected"],
        default="raw",
    )
    parser.add_argument("--seating-offsets", type=Path, help="JSON {test_id: offset_mm}")
    parser.add_argument(
        "--plot-mode",
        choices=["raw_protocol", "antonov_publication", "group_mean_ci", "diagnostic", "normalized"],
        default="antonov_publication",
    )
    parser.add_argument(
        "--axis",
        choices=[
            "F-s",
            "p-s",
            "p-s/D",
            "p/pu-s/D",
            "F/(gammaD3)-s/D",
            "p/(gammaD)-s/D",
        ],
        default="p-s",
    )
    parser.add_argument(
        "--curve-selections",
        type=Path,
        help=(
            "JSON с явным выбором кривой для каждой серии с повторностями; "
            "если не задан, используется metadata.publication_curve_selection"
        ),
    )
    parser.add_argument(
        "--failure-axis",
        choices=["auto", "force", "pressure"],
        default="auto",
        help="Ось отдельной диаграммы индивидуальных интервалов разрушения",
    )
    parser.add_argument(
        "--failure-summary-method",
        choices=["none"],
        default="none",
        help="Метод сводной оценки цензурированных данных; по умолчанию оценка отсутствует",
    )
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--seed", type=int, default=202604)
    parser.add_argument(
        "--method-profile",
        choices=modulus_profile_ids(),
        help="Версионированный профиль расчёта условного E_stamp_app",
    )
    parser.add_argument(
        "--e-range",
        type=parse_pressure_range,
        metavar="P_MIN:P_MAX",
        help="Явный замкнутый диапазон давления для E, кПа",
    )
    parser.add_argument(
        "--e-range-source",
        choices=["explicit", "accepted_pcr", "project_profile"],
        help="Методический источник диапазона E",
    )
    parser.add_argument("--e-range-author", help="Автор подтверждения диапазона E")
    parser.add_argument("--e-range-reason", help="Обоснование подтверждения диапазона E")
    parser.add_argument("--test-id", help="Испытание для режима diagnostic")
    parser.add_argument(
        "--import-mode",
        choices=["strict", "interactive", "heuristic"],
        default="strict",
        help="XLSX: строгая схема, сохраненный mapping или legacy-совместимость",
    )
    parser.add_argument("--column-map", type=Path, help="JSON {canonical_field: Excel column/header}")
    parser.add_argument("--sheet", help="Имя листа XLSX")
    parser.add_argument("--header-row", type=int, help="Номер строки заголовков XLSX (1-based)")
    return parser


def _curve_selection_records(
    metadata: dict,
    selection_path: Path | None,
) -> list[dict[str, object]]:
    """Load an explicit, versioned publication selection without inference."""

    if selection_path is not None:
        payload = json.loads(selection_path.read_text(encoding="utf-8-sig"))
        source = str(selection_path)
    else:
        payload = metadata.get("publication_curve_selection")
        source = "metadata.publication_curve_selection"
    if payload is None:
        return []
    if isinstance(payload, list):
        records = payload
    elif isinstance(payload, dict):
        version = payload.get("contract_version")
        if version not in {None, "publication-curve-selection/1.0"}:
            raise ValueError(
                f"{source}: неподдерживаемый contract_version={version!r}."
            )
        records = payload.get("decisions")
    else:
        raise ValueError(f"{source} должен быть JSON-объектом или массивом.")
    if not isinstance(records, list):
        raise ValueError(f"{source}.decisions должен быть массивом.")
    normalized: list[dict[str, object]] = []
    for index, record in enumerate(records, start=1):
        if not isinstance(record, dict):
            raise ValueError(f"{source}: decision #{index} должен быть объектом.")
        normalized.append({str(key): value for key, value in record.items()})
    return normalized


def _scope_curve_selection_records(
    frame: pd.DataFrame,
    records: list[dict[str, object]],
    *,
    known_groups: set[str] | None = None,
) -> list[dict[str, object]]:
    """Validate configured groups and keep decisions only for current repeats."""

    active_groups = set(frame["group"].dropna().astype(str))
    supplied_groups = {str(record.get("group") or "") for record in records}
    allowed_groups = active_groups | set(known_groups or set())
    unknown_groups = sorted(supplied_groups - allowed_groups)
    if unknown_groups:
        raise ValueError(
            "Выбор публикационной кривой задан для отсутствующих групп: "
            + ", ".join(repr(value) for value in unknown_groups)
            + "."
        )
    repeated_groups = {
        str(group)
        for group, part in frame.groupby("group", sort=True)
        if part["test_id"].astype(str).nunique() > 1
    }
    return [
        record
        for record in records
        if str(record.get("group") or "") in repeated_groups
    ]


def run(args: argparse.Namespace) -> Path:
    e_range = getattr(args, "e_range", None)
    e_range_source = getattr(args, "e_range_source", None)
    if e_range is not None and e_range_source is None:
        e_range_source = "explicit"
    if e_range_source == "explicit" and e_range is None:
        raise ValueError("Для --e-range-source explicit требуется --e-range P_MIN:P_MAX.")
    range_author = str(getattr(args, "e_range_author", None) or "").strip() or None
    range_reason = str(getattr(args, "e_range_reason", None) or "").strip() or None
    range_approved_at = (
        datetime.now(timezone.utc).isoformat() if range_author and range_reason else None
    )
    modulus_overrides = ModulusOverrides(
        profile_id=getattr(args, "method_profile", None),
        p_range_kpa=e_range,
        p_range_source=e_range_source,
        approval_status="approved" if range_approved_at else None,
        author=range_author,
        timestamp_utc=range_approved_at,
        reason=range_reason,
    )
    protocol_bytes = args.protocol.read_bytes()
    metadata_bytes = args.metadata.read_bytes()
    column_mapping = None
    if args.column_map:
        column_mapping = json.loads(args.column_map.read_text(encoding="utf-8-sig"))
        if not isinstance(column_mapping, dict):
            raise ValueError("--column-map должен содержать JSON-объект.")
    imported = read_protocol(
        protocol_bytes,
        filename=args.protocol.name,
        import_mode=args.import_mode,
        column_mapping=column_mapping,
        sheet_name=args.sheet,
        header_row=args.header_row,
    )
    raw = imported.frame
    metadata = read_metadata_json(metadata_bytes)
    curve_selection_records = (
        _curve_selection_records(metadata, getattr(args, "curve_selections", None))
        if args.plot_mode == "antonov_publication"
        else []
    )
    prepared, measurement_issues = prepare_measurements(
        raw, metadata, strict_metadata=False
    )
    indicator_processing_audit = indicator_audit_frame(prepared)
    indicator_processing_events = indicator_event_frame(prepared)
    indicator_calibration_parameters = indicator_passport_frame(prepared)
    indicator_aggregation_results = indicator_aggregation_frame(prepared)
    metadata_issues = validate_project_metadata(
        metadata, strict=args.import_mode in {"strict", "interactive"}
    )
    consistency_issues = validate_import_metadata_consistency(
        raw,
        metadata,
        imported.info,
        strict=args.import_mode in {"strict", "interactive"},
    )
    issues = [*imported.issues, *metadata_issues, *consistency_issues, *measurement_issues]
    blocking = [issue for issue in issues if bool(issue.blocks_processing)]
    if blocking:
        details = "; ".join(
            f"{issue.sheet or 'данные'}:{issue.row or '—'}:{issue.column or '—'} — {issue.message}"
            for issue in blocking
        )
        raise ValueError(details)
    metadata_tests = metadata.get("tests") if isinstance(metadata, dict) else None
    known_project_groups = {
        str(item.get("group"))
        for item in (metadata_tests or {}).values()
        if isinstance(item, dict) and item.get("group") not in (None, "")
    }
    scoped_curve_selection_records = _scope_curve_selection_records(
        prepared,
        curve_selection_records,
        known_groups=known_project_groups,
    )
    if args.plot_mode == "antonov_publication":
        resolved_curve_selections = resolve_curve_selections(
            prepared, scoped_curve_selection_records
        )
        scoped_groups = {
            str(record.get("group") or "")
            for record in scoped_curve_selection_records
        }
        active_curve_selection_records = [
            asdict(resolved_curve_selections[group]) for group in sorted(scoped_groups)
        ]
    else:
        active_curve_selection_records = []
    offsets = None
    if args.seating_offsets:
        offsets = json.loads(args.seating_offsets.read_text(encoding="utf-8-sig"))
        if not isinstance(offsets, dict):
            raise ValueError("--seating-offsets должен содержать JSON-объект.")
    config_snapshot = {
        key: (str(value) if isinstance(value, Path) else value)
        for key, value in vars(args).items()
        if key != "out"
    }
    config_snapshot["column_mapping"] = column_mapping
    config_snapshot["seating_offsets_content"] = offsets
    config_snapshot["curve_selection_decisions_supplied"] = curve_selection_records
    config_snapshot["curve_selection_decisions_applied"] = active_curve_selection_records
    provenance = build_provenance(
        input_source=protocol_bytes,
        metadata_source=metadata_bytes,
        config=config_snapshot,
        project_root=Path(__file__).resolve().parents[1],
        metrology_evaluations=metrology_evaluations_from_passports(
            indicator_calibration_parameters
        ),
    )
    audit = AuditTrail()
    audit.record(
        "batch_import",
        scope=",".join(prepared["test_id"].unique()),
        reason="Пакетный запуск CLI",
        parameters={
            "protocol": str(args.protocol),
            "metadata": str(args.metadata),
            "import": imported.info,
            "provenance": provenance.to_dict(),
        },
        after=raw,
        method="cli",
    )
    for decision in active_curve_selection_records:
        audit.record(
            "select_publication_curve",
            scope=str(decision.get("group") or ""),
            reason=str(decision.get("reason") or "Явный выбор кривой для публикации CLI"),
            parameters={
                "contract_version": "publication-curve-selection/1.0",
                **decision,
            },
            user=str(decision.get("author") or "cli"),
            method=str(decision.get("method") or ""),
        )
    prepared, correction_issues = apply_settlement_correction(
        prepared,
        args.correction,
        seating_offsets_mm=offsets,
        audit=audit,
        reason=f"Пакетный слой {args.correction}",
    )
    issues.extend(correction_issues)
    failures = failure_summary(prepared)
    requested_failure_axis = str(getattr(args, "failure_axis", "auto"))
    failure_analysis = failure_analysis_summary(
        failures,
        summary_method=str(getattr(args, "failure_summary_method", "none")),
        capacity_axis=requested_failure_axis,
    )
    if requested_failure_axis == "auto":
        capacity_kinds = {
            value
            for value in failures.get("capacity_kind", pd.Series(dtype=str))
            .dropna()
            .astype(str)
            if value != "unknown"
        }
        failure_plot_axis = "pressure" if capacity_kinds == {"pressure"} else "force"
    else:
        failure_plot_axis = requested_failure_axis
    failure_analysis["plot_capacity_axis"] = failure_plot_axis
    audit.record(
        "configure_failure_analysis",
        scope=",".join(failures["test_id"].astype(str).tolist()),
        reason="Безопасный контракт анализа цензурированных разрушений CLI",
        parameters=failure_analysis,
        method=str(failure_analysis["summary_method"]),
    )
    pcr_results = {}
    modulus_tables = []
    analysis_warnings: list[dict[str, str]] = []
    for test_id, part in prepared.groupby("test_id", sort=False):
        try:
            pcr_results[str(test_id)] = fit_segmented_pcr(
                part, bootstrap=args.bootstrap, seed=args.seed
            )
        except ValueError as exc:
            analysis_warnings.append(
                {"test_id": str(test_id), "analysis": "pcr", "message": str(exc)}
            )
            issues.append(
                ValidationIssue("warning", "pcr_not_calculated", str(exc), test_id=str(test_id))
            )
        try:
            table = calculate_moduli_for_test(
                part,
                metadata,
                str(test_id),
                overrides=modulus_overrides,
                pcr_result=pcr_results.get(str(test_id)),
                bootstrap=args.bootstrap,
                seed=args.seed,
            )
            table.insert(0, "test_id", str(test_id))
            modulus_tables.append(table)
            resolved = table.attrs.get("modulus_resolution", {})
            audit.record(
                "resolve_modulus_method",
                scope=str(test_id),
                reason=range_reason or "Разрешение методического контракта CLI",
                parameters=resolved,
                user=range_author or "cli",
                method="methodology_resolver",
            )
            if resolved.get("review_status") == "review_required":
                message = str(resolved.get("methodology_note") or "Требуется проверка методики E.")
                analysis_warnings.append(
                    {"test_id": str(test_id), "analysis": "E_methodology", "message": message}
                )
                issues.append(
                    ValidationIssue(
                        "warning",
                        "modulus_review_required",
                        message,
                        test_id=str(test_id),
                    )
                )
        except ValueError as exc:
            analysis_warnings.append(
                {"test_id": str(test_id), "analysis": "E", "message": str(exc)}
            )
            issues.append(
                ValidationIssue("warning", "modulus_not_calculated", str(exc), test_id=str(test_id))
            )
    moduli = pd.concat(modulus_tables, ignore_index=True) if modulus_tables else pd.DataFrame()
    plot_frame = prepared
    diagnostic_result = None
    if args.plot_mode == "diagnostic":
        if not args.test_id:
            raise ValueError("Для --plot-mode diagnostic укажите --test-id.")
        plot_frame = prepared[prepared["test_id"].astype(str) == str(args.test_id)]
        if plot_frame.empty:
            raise ValueError(f"Испытание {args.test_id} не найдено.")
        diagnostic_result = pcr_results.get(str(args.test_id))
        if diagnostic_result is None:
            raise ValueError(f"Для {args.test_id} pcr не рассчитано; diagnostic недоступен.")
    plot = plot_curves(
        plot_frame,
        mode=args.plot_mode,
        axis_mode=args.axis,
        pcr_result=diagnostic_result,
        bootstrap=args.bootstrap,
        seed=args.seed,
        selections=(
            active_curve_selection_records
            if args.plot_mode == "antonov_publication"
            else None
        ),
    )
    failure_plot = plot_failure_intervals(
        failures,
        capacity_axis=failure_plot_axis,
    )
    report = build_markdown_report(
        metadata=metadata,
        prepared=prepared,
        validation_issues=issues,
        failures=failures,
        pcr_results=pcr_results,
        moduli=moduli,
        figure_caption=plot.caption,
        plot_warnings=plot.warnings,
        audit=audit,
        provenance=provenance,
        passport_status=passport_completeness(metadata),
        import_info=imported.info,
        source_test_ids=raw["test_id"].dropna().astype(str).unique().tolist(),
        source_row_count=len(raw),
        failure_analysis=failure_analysis,
        curve_selections=plot.selection_records,
        plotted_curve_points=plot.plotted_points,
    )

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "prepared.csv").write_text(prepared.to_csv(index=False), encoding="utf-8")
    (args.out / "indicator_processing_audit.csv").write_text(
        indicator_processing_audit.to_csv(index=False), encoding="utf-8-sig"
    )
    (args.out / "indicator_processing_events.csv").write_text(
        indicator_processing_events.to_csv(index=False), encoding="utf-8-sig"
    )
    (args.out / "indicator_calibration_parameters.csv").write_text(
        indicator_calibration_parameters.to_csv(index=False), encoding="utf-8-sig"
    )
    (args.out / "indicator_aggregation_results.csv").write_text(
        indicator_aggregation_results.to_csv(index=False), encoding="utf-8-sig"
    )
    (args.out / "failure_summary.csv").write_text(failures.to_csv(index=False), encoding="utf-8")
    (args.out / "failure_analysis.json").write_text(
        json.dumps(failure_analysis, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.out / "curve_selections.csv").write_text(
        pd.DataFrame(plot.selection_records).to_csv(index=False), encoding="utf-8-sig"
    )
    (args.out / "plotted_curve_points.csv").write_text(
        plot.plotted_points.to_csv(index=False), encoding="utf-8-sig"
    )
    if not moduli.empty:
        (args.out / "moduli.csv").write_text(moduli.to_csv(index=False), encoding="utf-8")
    elif (args.out / "moduli.csv").exists():
        (args.out / "moduli.csv").unlink()
    (args.out / "analysis_warnings.json").write_text(
        json.dumps(analysis_warnings, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.out / "provenance.json").write_text(
        json.dumps(provenance.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.out / "import_issues.csv").write_text(
        pd.DataFrame([item.to_dict() for item in imported.issues]).to_csv(index=False),
        encoding="utf-8-sig",
    )
    (args.out / "validation_issues.csv").write_text(
        pd.DataFrame([item.to_dict() for item in issues]).to_csv(index=False),
        encoding="utf-8-sig",
    )
    raw_cells_path = args.out / "raw_cells.csv"
    if len(imported.raw_cells):
        raw_cells_path.write_text(imported.raw_cells.to_csv(index=False), encoding="utf-8-sig")
    elif raw_cells_path.exists():
        raw_cells_path.unlink()
    (args.out / "pcr.json").write_text(
        json.dumps({key: value.to_dict() for key, value in pcr_results.items()}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (args.out / "modulus_method_profiles.json").write_text(
        json.dumps(modulus_profile_definitions(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (args.out / "report_ru.md").write_text(report, encoding="utf-8")
    figures = {
        "antonov.svg": export_figure(plot.figure, "svg"),
        "antonov.pdf": export_figure(plot.figure, "pdf"),
        "antonov_600dpi.png": export_figure(plot.figure, "png"),
        "failure_intervals.svg": export_figure(failure_plot.figure, "svg"),
        "failure_intervals.pdf": export_figure(failure_plot.figure, "pdf"),
        "failure_intervals_600dpi.png": export_figure(failure_plot.figure, "png"),
    }
    for name, payload in figures.items():
        (args.out / name).write_bytes(payload)
    validation_issue_records = [item.to_dict() for item in issues]
    result_tables = {
        "failure_summary": failures,
        "failure_analysis": failure_analysis,
        "curve_selections": pd.DataFrame(plot.selection_records),
        "plotted_curve_points": plot.plotted_points,
        "moduli": moduli,
        "pcr": {key: value.to_dict() for key, value in pcr_results.items()},
        "analysis_warnings": analysis_warnings,
        "validation_issues": validation_issue_records,
        "modulus_method_profiles": modulus_profile_definitions(),
        "conversion_parameters": pd.DataFrame(
            effective_conversion_parameters(
                metadata,
                prepared["test_id"].dropna().astype(str).unique().tolist(),
            )
        ),
        "indicator_processing_audit": indicator_processing_audit,
        "indicator_processing_events": indicator_processing_events,
        "indicator_calibration_parameters": indicator_calibration_parameters,
        "indicator_aggregation_results": indicator_aggregation_results,
    }
    report_artifacts = collect_approval_artifacts(
        raw=raw,
        prepared=prepared,
        source_file_name=args.protocol.name,
        source_file_bytes=protocol_bytes,
        metadata_file_name=args.metadata.name,
        metadata_file_bytes=metadata_bytes,
        result_tables=result_tables,
        figures=figures,
        audit=audit,
        provenance=provenance,
        report_markdown=report,
        config_snapshot=config_snapshot,
    )
    review_registry = build_review_required_registry(
        passport_status=passport_completeness(metadata),
        qc_issues=issues,
        indicator_passports=indicator_calibration_parameters,
        indicator_audit=indicator_processing_audit,
        indicator_aggregation=indicator_aggregation_results,
        failures=failures,
        moduli=moduli,
    )
    formula_records = build_formula_and_range_records(
        conversion_parameters=result_tables["conversion_parameters"],
        indicator_passports=indicator_calibration_parameters,
        modulus_profiles=modulus_profile_definitions(),
        moduli=moduli,
        pcr_results=result_tables["pcr"],
    )
    report_package = build_approval_report_package(
        artifacts=report_artifacts,
        metadata=metadata,
        raw=raw,
        prepared=prepared,
        indicator_passports=indicator_calibration_parameters,
        indicator_audit=indicator_processing_audit,
        qc_issues=validation_issue_records,
        failures=failures,
        pcr_results={key: value.to_dict() for key, value in pcr_results.items()},
        moduli=moduli,
        audit=audit,
        provenance=provenance,
        methodology={
            "modulus_method_profiles": modulus_profile_definitions(),
            "failure_analysis": failure_analysis,
            "publication_curve_selection_contract": "publication-curve-selection/1.0",
        },
        formulas=formula_records,
        display_rounding={"default": 6},
        review_required=review_registry,
        result_tables=result_tables,
        title=f"Soil Stamp approval report — {metadata.get('project_id', 'project')}",
        scope={
            "source_test_ids": raw["test_id"].dropna().astype(str).unique().tolist(),
            "selected_test_ids": prepared["test_id"].dropna().astype(str).unique().tolist(),
            "source_rows": len(raw),
            "prepared_rows": len(prepared),
        },
    )
    for relative_path, payload in report_artifacts.items():
        artifact_path = args.out.joinpath(*relative_path.split("/"))
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_path.write_bytes(payload)
    report_files = {
        "report.html": report_package.html,
        "report.xlsx": report_package.xlsx,
        "artifact_manifest.json": report_package.artifact_manifest_json,
        "approval_report.zip": report_package.archive,
    }
    embedded_report_files = {
        **{
            f"approval/{relative_path}": payload
            for relative_path, payload in report_artifacts.items()
        },
        **{f"approval/{name}": payload for name, payload in report_files.items()},
    }
    for name, payload in report_files.items():
        (args.out / name).write_bytes(payload)
    (args.out / "audit.json").write_text(audit.to_json(), encoding="utf-8")
    bundle = reproducibility_bundle(
        raw=raw,
        prepared=prepared,
        metadata=metadata,
        audit=audit,
        report_markdown=report,
        result_tables=result_tables,
        figures=figures,
        run_parameters=vars(args),
        provenance=provenance,
        raw_cells=imported.raw_cells,
        import_issues=issues,
        source_file_name=args.protocol.name,
        source_file_bytes=protocol_bytes,
        metadata_file_name=args.metadata.name,
        metadata_file_bytes=metadata_bytes,
        config_snapshot=config_snapshot,
        scope={
            "source_test_ids": raw["test_id"].dropna().astype(str).unique().tolist(),
            "selected_test_ids": prepared["test_id"].dropna().astype(str).unique().tolist(),
            "source_rows": len(raw),
            "prepared_rows": len(prepared),
        },
        additional_files=embedded_report_files,
    )
    bundle_path = args.out / "reproducibility.zip"
    bundle_path.write_bytes(bundle)
    return bundle_path


def main() -> None:
    args = build_parser().parse_args()
    path = run(args)
    print(f"Готово: {path.resolve()}")


if __name__ == "__main__":
    main()
