from __future__ import annotations

from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from soilstamp.data import prepare_measurements as _prepare_measurements
from soilstamp.plotting import export_figure, plot_curves


def prepare_measurements(*args, **kwargs):
    kwargs.setdefault("strict_metadata", False)
    return _prepare_measurements(*args, **kwargs)


def _failure_frame() -> pd.DataFrame:
    raw = pd.DataFrame(
        {
            "test_id": ["T1", "T1", "T1"],
            "stage": [1, 2, 3],
            "load": [100.0, 200.0, 250.0],
            "settlement": [1.0, 2.0, np.nan],
            "status": ["stable", "stable", "failure"],
            "group": ["g", "g", "g"],
            "branch": ["loading", "loading", "loading"],
        }
    )
    frame, _ = prepare_measurements(
        raw,
        {
            "stamp_shape": "custom",
            "stamp_diameter_mm": 300.0,
            "stamp_area_m2": 1.0,
            "lever_ratio": 1.0,
        },
    )
    return frame


def test_antonov_orientation_frame_grid_and_polyline_vertices() -> None:
    frame = _failure_frame()
    output = plot_curves(frame, mode="raw_protocol", axis_mode="F-s")
    ax = output.figure.axes[0]
    assert ax.xaxis.get_label_position() == "top"
    assert not bool(ax.xaxis_inverted())
    assert bool(ax.yaxis_inverted())
    assert all(ax.spines[name].get_visible() for name in ["top", "right", "bottom", "left"])
    assert any(line.get_visible() for line in ax.get_xgridlines())
    raw_line = next(line for line in ax.lines if line.get_gid() == "raw-T1")
    assert np.allclose(raw_line.get_xdata(), [100.0, 200.0])
    assert np.allclose(raw_line.get_ydata(), [1.0, 2.0])
    assert raw_line.get_path().codes is None  # ordinary straight polyline, no spline path codes
    assert ax.get_xlim()[1] > 250.0  # failure load without settlement still defines the axis
    plt.close(output.figure)


def test_mixed_diameters_warn_on_force_axis() -> None:
    frame = pd.concat([_failure_frame(), _failure_frame().assign(test_id="T2", D_mm=500.0)], ignore_index=True)
    output = plot_curves(frame, mode="raw_protocol", axis_mode="F-s")
    assert any("разные диаметры" in warning for warning in output.warnings)
    plt.close(output.figure)


def test_p_over_pu_normalization_requires_confirmed_capacity() -> None:
    frame = _failure_frame().assign(pu_kPa_confirmed=250.0)
    output = plot_curves(frame, mode="raw_protocol", axis_mode="p/pu-s/D")
    raw_line = next(line for line in output.figure.axes[0].lines if line.get_gid() == "raw-T1")
    assert np.allclose(raw_line.get_xdata(), [0.4, 0.8])
    assert output.figure.axes[0].get_xlim()[1] > 1.0
    plt.close(output.figure)


def test_publication_exports_are_valid_and_png_is_600_dpi() -> None:
    output = plot_curves(_failure_frame(), mode="antonov_publication", axis_mode="F-s")
    svg = export_figure(output.figure, "svg")
    pdf = export_figure(output.figure, "pdf")
    png = export_figure(output.figure, "png")
    assert b"<svg" in svg[:1000]
    assert pdf.startswith(b"%PDF")
    assert png.startswith(b"\x89PNG\r\n\x1a\n")
    image = Image.open(BytesIO(png))
    dpi = image.info.get("dpi")
    assert dpi is not None and abs(dpi[0] - 600) < 1
    plt.close(output.figure)


def test_antonov_caption_decodes_numbered_curves() -> None:
    output = plot_curves(_failure_frame(), mode="antonov_publication", axis_mode="F-s")
    assert output.curve_map == {1: "g, репрезентативная T1"}
    assert "1 — g" in output.caption
    assert "без spline" in output.caption
    plt.close(output.figure)
