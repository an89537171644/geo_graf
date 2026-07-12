"""Command-line batch analysis for reproducible, non-interactive runs."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .analysis import calculate_moduli_for_test, fit_segmented_pcr
from .data import AuditTrail, apply_settlement_correction, failure_summary, prepare_measurements
from .indicators import (
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
from .plotting import export_figure, plot_curves
from .provenance import (
    build_provenance,
    effective_conversion_parameters,
    passport_completeness,
    validate_project_metadata,
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
    prepared, measurement_issues = prepare_measurements(
        raw, metadata, strict_metadata=False
    )
    indicator_processing_audit = indicator_audit_frame(prepared)
    indicator_processing_events = indicator_event_frame(prepared)
    indicator_calibration_parameters = indicator_passport_frame(prepared)
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
    provenance = build_provenance(
        input_source=protocol_bytes,
        metadata_source=metadata_bytes,
        config=config_snapshot,
        project_root=Path(__file__).resolve().parents[1],
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
    prepared, correction_issues = apply_settlement_correction(
        prepared,
        args.correction,
        seating_offsets_mm=offsets,
        audit=audit,
        reason=f"Пакетный слой {args.correction}",
    )
    issues.extend(correction_issues)
    failures = failure_summary(prepared)
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
    (args.out / "failure_summary.csv").write_text(failures.to_csv(index=False), encoding="utf-8")
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
    }
    for name, payload in figures.items():
        (args.out / name).write_bytes(payload)
    bundle = reproducibility_bundle(
        raw=raw,
        prepared=prepared,
        metadata=metadata,
        audit=audit,
        report_markdown=report,
        result_tables={
            "failure_summary": failures,
            "moduli": moduli,
            "pcr": {key: value.to_dict() for key, value in pcr_results.items()},
            "analysis_warnings": analysis_warnings,
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
        },
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
