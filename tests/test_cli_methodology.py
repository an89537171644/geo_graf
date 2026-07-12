from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from soilstamp.analysis import calculate_moduli_for_test
from soilstamp.cli import build_parser, run
from soilstamp.methodology import ModulusOverrides


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def single_test_demo(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    work = tmp_path_factory.mktemp("cli_methodology_inputs")
    source_lines = (ROOT / "examples" / "demo_protocol.csv").read_text(
        encoding="utf-8"
    ).splitlines()
    protocol = work / "protocol.csv"
    protocol.write_text(
        "\n".join([source_lines[0], *(line for line in source_lines[1:] if line.startswith("B-01,"))])
        + "\n",
        encoding="utf-8",
    )

    metadata_payload = json.loads(
        (ROOT / "examples" / "demo_metadata.json").read_text(encoding="utf-8")
    )
    metadata_payload["tests"] = {"B-01": metadata_payload["tests"]["B-01"]}
    metadata = work / "metadata.json"
    metadata.write_text(
        json.dumps(metadata_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return protocol, metadata


def test_parser_accepts_method_profile_range_and_source_flags() -> None:
    args = build_parser().parse_args(
        [
            "protocol.csv",
            "metadata.json",
            "--method-profile",
            "antonov_round_stamp_v1",
            "--e-range",
            "0:200",
            "--e-range-source",
            "explicit",
        ]
    )

    assert args.method_profile == "antonov_round_stamp_v1"
    assert args.e_range == (0.0, 200.0)
    assert args.e_range_source == "explicit"


@pytest.mark.parametrize("raw_range", ["not-a-range", "200:0", "0:zero"])
def test_parser_rejects_malformed_or_reversed_e_range(raw_range: str) -> None:
    with pytest.raises(SystemExit) as exc_info:
        build_parser().parse_args(
            ["protocol.csv", "metadata.json", "--e-range", raw_range]
        )

    assert exc_info.value.code == 2


def test_cli_approved_antonov_matches_direct_api_with_provenance(
    single_test_demo: tuple[Path, Path], tmp_path: Path
) -> None:
    protocol, metadata_path = single_test_demo
    output = tmp_path / "approved"
    args = build_parser().parse_args(
        [
            str(protocol),
            str(metadata_path),
            "--out",
            str(output),
            "--bootstrap",
            "20",
            "--method-profile",
            "antonov_round_stamp_v1",
            "--e-range",
            "0:200",
            "--e-range-source",
            "explicit",
            "--e-range-author",
            "engineer@example.test",
            "--e-range-reason",
            "Confirmed linear range for CLI/API parity test.",
        ]
    )

    bundle = run(args)
    assert bundle.is_file()
    cli_table = pd.read_csv(output / "moduli.csv")
    cli_regression = cli_table.loc[cli_table["method"].eq("E_regression")].iloc[0]

    prepared = pd.read_csv(output / "prepared.csv")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    direct_table = calculate_moduli_for_test(
        prepared,
        metadata,
        "B-01",
        overrides=ModulusOverrides(
            profile_id="antonov_round_stamp_v1",
            p_range_kpa=(0.0, 200.0),
            p_range_source="explicit",
            approval_status="approved",
            author="engineer@example.test",
            timestamp_utc="2026-07-12T06:00:00+00:00",
            reason="Confirmed linear range for CLI/API parity test.",
        ),
        bootstrap=20,
        seed=args.seed,
    )
    direct_regression = direct_table.loc[
        direct_table["method"].eq("E_regression")
    ].iloc[0]

    assert np.isclose(
        cli_regression["E_stamp_app_kPa"],
        direct_regression["E_stamp_app_kPa"],
        rtol=1e-12,
        atol=0.0,
    )
    for column in (
        "profile_id",
        "p_range_source",
        "nu_source",
        "shape_factor_source",
        "profile_source",
        "p_range_origin",
        "review_status",
    ):
        assert cli_regression[column] == direct_regression[column]
    assert str(cli_regression["profile_version"]) == direct_regression[
        "profile_version"
    ]
    assert ast.literal_eval(cli_regression["used_indices"]) == direct_regression[
        "used_indices"
    ]
    assert bool(cli_regression["is_primary"])
    assert cli_regression["review_status"] == "approved"


def test_default_cli_writes_numeric_nonprimary_diagnostic(
    single_test_demo: tuple[Path, Path], tmp_path: Path
) -> None:
    protocol, metadata = single_test_demo
    output = tmp_path / "diagnostic"
    args = build_parser().parse_args(
        [
            str(protocol),
            str(metadata),
            "--out",
            str(output),
            "--bootstrap",
            "20",
        ]
    )

    run(args)
    table = pd.read_csv(output / "moduli.csv")
    regression = table.loc[table["method"].eq("E_regression")].iloc[0]

    assert np.isfinite(regression["E_stamp_app_kPa"])
    assert regression["profile_id"] == "diagnostic_unapproved_v1"
    assert regression["p_range_source"] == "diagnostic_full_curve"
    assert regression["review_status"] == "review_required"
    assert not bool(regression["is_primary"])
