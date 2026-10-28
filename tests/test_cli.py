# tests/test_cli.py
import pytest

from stratbt import cli


def test_strategies_lists_builtins(capsys):
    assert cli.main(["strategies"]) == 0
    out = capsys.readouterr().out
    assert "iron_condor" in out and "--param width" in out


def test_run_on_synthetic_prints_summary(capsys):
    rc = cli.main(["run", "short_strangle", "--days", "2", "--underliers", "SPY",
                   "--execution", "realistic", "--param", "width=3"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "total_pnl" in out and "fees paid" in out and "USD" in out


def test_run_writes_pdf(tmp_path, capsys):
    pdf = tmp_path / "t.pdf"
    assert cli.main(["run", "fixed_window", "--days", "2", "--underliers", "SPY", "--pdf", str(pdf)]) == 0
    assert pdf.read_bytes().startswith(b"%PDF")


def test_bad_param_exits():
    with pytest.raises(SystemExit):
        cli.main(["run", "fixed_window", "--days", "1", "--param", "oops"])


def test_unknown_strategy_errors():
    with pytest.raises(KeyError):
        cli.main(["run", "nope", "--days", "1"])
