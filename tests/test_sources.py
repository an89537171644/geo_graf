from __future__ import annotations

import pandas as pd

from soilstamp.sources import ExcelExperimentSource, ExperimentSource


def test_excel_source_implements_source_neutral_experiment_contract() -> None:
    frame = pd.DataFrame(
        {
            "test_id": ["T1", "T1"],
            "stage": [1, 2],
            "branch": ["loading", "cyclic"],
            "load": ["0,0", "1,0"],
            "indicator_1": ["9,80", "0,20"],
            "status": ["stable", "failure"],
            "source_row": [2, 3],
        }
    )
    source = ExcelExperimentSource(
        frame,
        {
            "tests": {"T1": {"group": "baseline", "soil_batch": "S-1"}},
            "project_passport": {"operator": "Иванов"},
        },
    )

    assert isinstance(source, ExperimentSource)
    experiment = source.load()[0]
    assert experiment.test_id == "T1"
    assert experiment.group == "baseline"
    assert experiment.soil_batch == "S-1"
    assert experiment.operator == "Иванов"
    assert [point.sequence_no for point in experiment.points] == [0, 1]
    assert experiment.points[1].branch == "cyclic"
    assert experiment.points[1].source_type == "excel"
    assert experiment.points[1].source_row == 3
