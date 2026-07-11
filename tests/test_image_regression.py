from __future__ import annotations

import pandas as pd
import pytest

from soilstamp.data import prepare_measurements as _prepare_measurements
from soilstamp.plotting import plot_curves


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
