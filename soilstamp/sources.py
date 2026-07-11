"""Source-neutral experiment contract shared by imported and manual data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import pandas as pd

from .schema import Experiment, ExperimentPoint


@runtime_checkable
class ExperimentSource(Protocol):
    """Load primary experiments without performing scientific calculations."""

    def load(self) -> list[Experiment]: ...


def experiments_from_frame(
    frame: pd.DataFrame,
    metadata: dict[str, Any] | None = None,
    *,
    source_type: str,
) -> list[Experiment]:
    """Build the source-neutral model while preserving row order and raw values."""

    metadata = metadata or {}
    experiments: list[Experiment] = []
    if "test_id" not in frame:
        return experiments
    for test_value, part in frame.groupby("test_id", sort=False, dropna=False):
        test_id = str(test_value)
        points: list[ExperimentPoint] = []
        for position, (_, row) in enumerate(part.iterrows()):
            source_sequence = row.get("source_sequence_no", row.get("sequence_no", position))
            try:
                sequence_no = int(source_sequence)
            except (TypeError, ValueError):
                sequence_no = position
            indicators = {
                name: row.get(f"raw_{name}", row.get(name))
                for name in (
                    "indicator_1",
                    "indicator_2",
                    "indicator_3",
                    "indicator_4",
                )
                if name in part
            }
            source_row = row.get("source_row")
            if pd.isna(source_row):
                source_row = None
            elif source_row is not None:
                try:
                    source_row = int(source_row)
                except (TypeError, ValueError):
                    source_row = None
            points.append(
                ExperimentPoint(
                    test_id=test_id,
                    sequence_no=sequence_no,
                    stage=row.get("stage"),
                    stage_raw=row.get("raw_stage", row.get("stage")),
                    branch=row.get("branch"),
                    elapsed_time_s=row.get("elapsed_time_s"),
                    elapsed_time_raw=row.get(
                        "raw_elapsed_time_s", row.get("elapsed_time_s")
                    ),
                    timestamp=row.get("timestamp"),
                    timestamp_raw=row.get("raw_timestamp", row.get("timestamp")),
                    load_raw=row.get("raw_load", row.get("load")),
                    indicator_raws=indicators,
                    row_status=row.get("row_status", row.get("status")),
                    comment=row.get("comment"),
                    source_type=str(row.get("source_type") or source_type),
                    source_row=source_row,
                    manual_row_uuid=row.get("manual_row_uuid"),
                    created_by=row.get("created_by"),
                    created_at=row.get("created_at"),
                    modified_by=row.get("modified_by"),
                    modified_at=row.get("modified_at"),
                )
            )
        tests = metadata.get("tests") if isinstance(metadata.get("tests"), dict) else {}
        test_candidate = tests.get(test_id, {}) if isinstance(tests, dict) else {}
        test_meta = test_candidate if isinstance(test_candidate, dict) else {}
        soil = metadata.get("soil") if isinstance(metadata.get("soil"), dict) else {}
        project_passport = (
            metadata.get("project_passport")
            if isinstance(metadata.get("project_passport"), dict)
            else {}
        )
        experiments.append(
            Experiment(
                test_id=test_id,
                group=test_meta.get("group") or metadata.get("group"),
                pair_id=test_meta.get("pair_id") or metadata.get("pair_id"),
                soil_batch=(
                    test_meta.get("soil_batch")
                    or metadata.get("soil_batch")
                    or soil.get("batch")
                ),
                experiment_date=(
                    test_meta.get("experiment_date")
                    or project_passport.get("experiment_date")
                ),
                operator=test_meta.get("operator") or project_passport.get("operator"),
                metadata=dict(test_meta),
                points=points,
            )
        )
    return experiments


@dataclass(slots=True)
class ExcelExperimentSource:
    """Adapter for the canonical frame produced by strict CSV/XLSX import."""

    frame: pd.DataFrame
    metadata: dict[str, Any]

    def load(self) -> list[Experiment]:
        return experiments_from_frame(self.frame, self.metadata, source_type="excel")
