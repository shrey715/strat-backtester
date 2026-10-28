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


def test_paper_replays_a_session_and_writes_audit_and_state(tmp_path, capsys):
    audit, state = tmp_path / "audit.jsonl", tmp_path / "state.json"
    rc = cli.main(["paper", "iron_condor", "--days", "3", "--underlier", "SPY", "--execution", "realistic",
                   "--audit-log", str(audit), "--state", str(state)])
    out = capsys.readouterr().out
    assert rc == 0 and "paper trading on synthetic data" in out and "8 fills" in out
    events = [__import__("json").loads(l)["event"] for l in audit.read_text().splitlines()]
    assert events[0] == "start" and events[-1] == "end" and events.count("fill") == 8
    assert state.exists()


def test_paper_kill_file_stops_the_session(tmp_path, capsys):
    kill = tmp_path / "STOP"
    kill.touch()                                   # already tripped: nothing should be left open
    cli.main(["paper", "short_strangle", "--days", "3", "--underlier", "SPY", "--kill-file", str(kill)])
    assert "KILLED" in capsys.readouterr().out


def test_yahoo_provider_requires_tickers():
    with pytest.raises(SystemExit, match="--tickers"):
        cli.main(["run", "fixed_window", "--provider", "yahoo"])
