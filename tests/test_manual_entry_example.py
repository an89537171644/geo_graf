from __future__ import annotations

from pathlib import Path

import pandas as pd

from soilstamp.data import failure_summary
from soilstamp.indicators import indicator_audit_frame, indicator_event_frame
from soilstamp.manual_entry_adapter import adapt_manual_draft
from soilstamp.manual_entry_models import ManualDraft


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "manual_entry_demo.json"


def test_manual_entry_example_roundtrip_and_pipeline() -> None:
    original = ManualDraft.from_json(EXAMPLE.read_bytes())
    restored = ManualDraft.from_json(original.to_json())

    assert original.schema_version == "manual-entry-draft/1.1"
    assert original.passport.baseline_group == "baseline"
    assert original.passport.pair_id is None
    assert restored.to_dict() == original.to_dict()
    assert restored.sha256 == original.sha256

    bundle = adapt_manual_draft(restored)
    prepared, issues = bundle.prepare()

    assert bundle.can_analyze
    assert not [issue for issue in issues if bool(issue.blocks_processing)]
    assert bundle.raw["source_type"].eq("manual").all()
    assert bundle.raw["source_row"].isna().all()
    assert prepared["source_type"].eq("manual").all()
    assert prepared["source_row"].isna().all()
    assert prepared["manual_row_uuid"].tolist() == [
        row.manual_row_uuid for row in restored.rows
    ]
    conversion = indicator_audit_frame(prepared)
    events = indicator_event_frame(prepared)
    assert conversion["manual_row_uuid"].tolist() == [
        row.manual_row_uuid for row in restored.rows
    ]
    assert events["manual_row_uuid"].notna().all()
    assert set(events["manual_row_uuid"]).issubset(
        {row.manual_row_uuid for row in restored.rows}
    )
    assert bundle.raw_cells.loc[
        bundle.raw_cells["canonical_field"].eq("indicator_1"), "raw_value"
    ].tolist() == ["0,00", "0,20", "0,50", None]

    failure = failure_summary(prepared).iloc[0]
    assert bool(failure["failure_reached"])
    assert pd.isna(failure["s_failure"])
