from __future__ import annotations

import numpy as np

from soilstamp.io import read_protocol, read_protocol_csv


def test_utf16_excel_style_csv_with_decimal_comma() -> None:
    text = "test_id;stage;load;settlement\r\nT1;1;1,5;0,25\r\n"
    frame, info = read_protocol_csv(text.encode("utf-16"))
    assert info["encoding"] == "utf-16"
    assert info["delimiter"] == ";"
    assert np.isclose(frame.loc[0, "load"], 1.5)
    assert np.isclose(frame.loc[0, "settlement"], 0.25)


def test_generic_strict_csv_blocks_unknown_header() -> None:
    payload = b"test_id,stage,load,settlement,mystery\nT1,1,1,0.2,x\n"
    result = read_protocol(payload, filename="protocol.csv", import_mode="strict")

    issue = next(item for item in result.issues if item.code == "unknown_header")
    assert issue.blocks_processing is True
    assert issue.row == 1
    assert issue.column == "mystery"
