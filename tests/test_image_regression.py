from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from soilstamp.data import prepare_measurements as _prepare_measurements
from soilstamp.plotting import plot_curves, plot_failure_intervals


def prepare_measurements(*args, **kwargs):
    kwargs.setdefault("strict_metadata", False)
    return _prepare_measurements(*args, **kwargs)


@pytest.mark.mpl_image_compare(tolerance=8.0, remove_text=True)
def test_antonov_image():
    raw = pd.DataFrame(
        {
            "test_id": ["1"] * 5 + ["2"] * 5,
            "stage": list(range(5)) * 2,
            "load": [0, 5, 10, 15, 20] * 2,
            "settlement": [0.2, 0.5, 1.0, 1.9, 3.2, 0.1, 0.4, 0.8, 1.4, 2.2],
            "branch": ["loading"] * 10,
            "status": ["stable"] * 10,
            "group": ["baseline"] * 5 + ["reinforced"] * 5,
        }
    )
    frame, _ = prepare_measurements(
        raw,
        {
            "stamp_shape": "custom",
            "stamp_area_m2": 0.1,
            "stamp_diameter_mm": 300.0,
            "lever_ratio": 1.0,
        },
    )
    return plot_curves(frame, mode="raw_protocol", axis_mode="p-s").figure


def _publication_frame() -> pd.DataFrame:
    rows = []
    curves = {
        "B2": ("baseline", [0.0, 0.8, 2.6, 4.0]),
        "B1": ("baseline", [0.0, 1.0, 2.2, 3.7]),
        "R2": ("reinforced", [0.0, 0.5, 1.2, 2.0]),
        "R1": ("reinforced", [0.0, 0.6, 1.4, 2.2]),
    }
    for test_id, (group, settlements) in curves.items():
        for sequence_no, (load, settlement) in enumerate(
            zip([0.0, 50.0, 100.0, 150.0], settlements, strict=True),
            1,
        ):
            rows.append(
                {
                    "test_id": test_id,
                    "stage": sequence_no,
                    "load": load,
                    "settlement": settlement,
                    "branch": "loading",
                    "status": "stable",
                    "group": group,
                }
            )
    frame, _ = prepare_measurements(
        pd.DataFrame(rows),
        {
            "stamp_shape": "custom",
            "stamp_area_m2": 1.0,
            "stamp_diameter_mm": 300.0,
            "lever_ratio": 1.0,
        },
    )
    return frame


@pytest.mark.mpl_image_compare(tolerance=8.0, remove_text=True)
def test_publication_image():
    return plot_curves(
        _publication_frame(),
        mode="antonov_publication",
        axis_mode="F-s",
        selections={
            "baseline": "mean_curve",
            "reinforced": "median_curve",
        },
        fixed_axes=(0.0, 165.0, 0.0, 4.5),
        bootstrap=50,
        seed=202604,
    ).figure


@pytest.mark.mpl_image_compare(tolerance=8.0, remove_text=True)
def test_group_ci_image():
    frame = _publication_frame().assign(group="series")
    # One missing real level forces an interpolated contribution at F=50 kN.
    frame = frame[~((frame["test_id"] == "B2") & np.isclose(frame["F_kN"], 50.0))]
    return plot_curves(
        frame,
        mode="group_mean_ci",
        axis_mode="F-s",
        ci_method="t",
        fixed_axes=(0.0, 165.0, 0.0, 4.5),
        bootstrap=50,
        seed=202604,
    ).figure


@pytest.mark.mpl_image_compare(tolerance=8.0, remove_text=True)
def test_failure_interval_image():
    failures = pd.DataFrame(
        {
            "test_id": ["T-03", "T-01", "T-02"],
            "failure_observed": [False, True, True],
            "interval_censored": [False, True, True],
            "right_censored": [True, False, False],
            "Fu_lower": [210.0, 100.0, 125.0],
            "Fu_upper": [np.nan, 145.0, 175.0],
        }
    )
    return plot_failure_intervals(failures, capacity_axis="force").figure
