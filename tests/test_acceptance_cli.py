from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from soilstamp.cli import main


@pytest.mark.parametrize(("exit_code", "failure_count"), [(0, 0), (1, 2)])
def test_acceptance_run_subcommand_returns_runner_exit_code(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    exit_code: int,
    failure_count: int,
) -> None:
    manifest = tmp_path / "manifest.json"
    out = tmp_path / "results"
    observed: dict[str, Path] = {}

    def fake_run(manifest_path: Path, out_dir: Path) -> SimpleNamespace:
        observed["manifest"] = manifest_path
        observed["out"] = out_dir
        return SimpleNamespace(
            json_report=out_dir / "acceptance_report.json",
            markdown_report=out_dir / "acceptance_report.md",
            html_report=out_dir / "acceptance_report.html",
            exit_code=exit_code,
            critical_failure_count=failure_count,
        )

    monkeypatch.setattr("soilstamp.acceptance.run_acceptance_manifest", fake_run)

    actual = main(["acceptance-run", str(manifest), "--out", str(out)])

    assert actual == exit_code
    assert observed == {"manifest": manifest, "out": out}
    captured = capsys.readouterr()
    assert "acceptance_report.json" in captured.out
    if exit_code:
        assert f": {failure_count}" in captured.err
    else:
        assert captured.err == ""


def test_normal_cli_dispatch_remains_backward_compatible(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    protocol = tmp_path / "protocol.csv"
    metadata = tmp_path / "metadata.json"
    output = tmp_path / "normal-results"

    def fake_run(args: object) -> Path:
        assert getattr(args, "protocol") == protocol
        assert getattr(args, "metadata") == metadata
        return output / "reproducibility.zip"

    monkeypatch.setattr("soilstamp.cli.run", fake_run)

    assert main([str(protocol), str(metadata), "--out", str(output)]) == 0
    assert "reproducibility.zip" in capsys.readouterr().out
