# tests/test_mcp.py
"""The MCP server exercised through the SDK's own client (in memory, and over a real stdio subprocess)."""
import json
import sys

import pytest

pytest.importorskip("mcp.server.mcpserver")
import anyio  # noqa: E402
from mcp.client.client import Client  # noqa: E402
from mcp.client.stdio import StdioServerParameters  # noqa: E402

from stratbt.mcp_server import build_server  # noqa: E402
from tests.test_yahoo_provider import FakeDownloader  # noqa: E402

SYN = {"source": "synthetic", "synthetic_days": 5, "underliers": ["SPY"]}


@pytest.fixture
def run(tmp_path):
    """Run an async client function against a fresh in-memory server."""
    def go(fn):
        async def main():
            async with Client(build_server(tmp_path / "out")) as c:
                return await fn(c)
        return anyio.run(main)
    return go


def _json_safe(payload) -> None:
    json.dumps(payload, allow_nan=False)        # raises on NaN / inf


class TestDiscovery:

    def test_tools_resources_prompts(self, run):
        async def f(c):
            return ([t for t in (await c.list_tools()).tools], (await c.list_resources()).resources,
                    (await c.list_resource_templates()).resource_templates, (await c.list_prompts()).prompts, c.instructions)
        tools, res, tmpl, prompts, instructions = run(f)
        assert {t.name for t in tools} == {"list_strategies", "check_market_data", "run_backtest", "get_run",
                                           "compare_runs", "export_report", "paper_trade"}
        assert {str(r.uri) for r in res} == {"stratbt://strategies", "stratbt://runs"}
        assert [t.uri_template for t in tmpl] == ["stratbt://runs/{run_id}"]
        assert {p.name for p in prompts} == {"evaluate_strategy", "compare_strategies"}
        assert "cannot trade" in instructions and "MODELLED" in instructions

    def test_the_server_offers_no_way_to_trade(self, run):
        async def f(c):
            return {t.name: t for t in (await c.list_tools()).tools}
        tools = run(f)
        assert not [n for n in tools if any(w in n for w in ("place", "order", "buy", "sell", "submit", "live"))]
        assert "no orders" in tools["paper_trade"].description.lower()

    def test_tool_schemas_carry_descriptions_and_annotations(self, run):
        async def f(c):
            return {t.name: t for t in (await c.list_tools()).tools}
        t = run(f)
        props = t["run_backtest"].input_schema["properties"]
        assert "iron_condor" in props["strategy"]["description"]
        assert props["execution"]["enum"] == ["default", "realistic", "instant"]
        assert t["get_run"].annotations.read_only_hint is True
        assert t["export_report"].annotations.read_only_hint is False

    def test_list_strategies(self, run):
        async def f(c):
            return (await c.call_tool("list_strategies", {})).structured_content
        out = run(f)["strategies"]
        by = {s["key"]: s for s in out}
        assert {"iron_condor", "short_strangle", "atm_straddle"} <= set(by)
        assert by["short_strangle"]["unlimited_risk"] is True and by["iron_condor"]["unlimited_risk"] is False
        assert {"name": "width", "default": 2, "min": 1, "max": 20, "step": 1, "description": "Width (strikes from ATM)"} in by["iron_condor"]["parameters"]


class TestRunBacktest:

    def test_returns_structured_json_safe_payload(self, run):
        async def f(c):
            r = await c.call_tool("run_backtest", {"strategy": "iron_condor", **SYN})
            return r.is_error, r.structured_content
        is_error, p = run(f)
        assert not is_error
        _json_safe(p)
        assert p["run_id"] == "run_1" and p["headline"]["unit"] == "USD" and p["headline"]["trading_days"] == 5
        assert p["headline"]["costs"]["total"] > 0                  # 'realistic' is the default
        assert len(p["headline"]["sharpe_ci95"]) == 2 and p["observations"] and "SPY" in p["per_underlier"]
        assert any("Synthetic" in w for w in p["warnings"]) and any("Small sample" in w for w in p["warnings"])
        assert "Iron Condor on SPY" in p["summary"]

    def test_realistic_costs_reduce_pnl_versus_instant(self, run):
        async def f(c):
            a = (await c.call_tool("run_backtest", {"strategy": "fixed_window", "execution": "realistic", **SYN})).structured_content
            b = (await c.call_tool("run_backtest", {"strategy": "fixed_window", "execution": "instant", **SYN})).structured_content
            return a, b
        real, inst = run(f)
        assert real["headline"]["net_pnl"] < inst["headline"]["net_pnl"]
        assert real["headline"]["costs"]["total"] > 0 and inst["headline"]["costs"]["total"] == 0
        assert any("optimistic" in w for w in inst["warnings"])
        assert any("No trading costs" in w for w in inst["warnings"])

    def test_parameters_and_risk_limits_are_applied_and_echoed(self, run):
        async def f(c):
            return (await c.call_tool("run_backtest", {"strategy": "short_strangle", "params": {"width": 4},
                                                       "max_open_contracts": 1, "commission_per_contract": 2.5, **SYN})).structured_content
        p = run(f)
        assert p["config"]["parameters"]["width"] == 4
        assert p["config"]["risk_limits"]["max_open_contracts"] == 1
        assert p["config"]["execution"]["commission_per_contract"] == 2.5
        assert any("rejected" in w for w in p["warnings"])           # 2-leg entry cannot fit in 1 contract

    @pytest.mark.parametrize("args,needle", [
        ({"strategy": "nope"}, "Unknown strategy"),
        ({"strategy": "iron_condor", "params": {"bogus": 1}}, "unknown parameter"),
        ({"strategy": "iron_condor", "params": {"width": 999}}, "outside"),
        ({"strategy": "iron_condor", "source": "yahoo"}, "needs `tickers`"),
        ({"strategy": "iron_condor", "source": "yahoo", "tickers": list("ABCDE")}, "At most 4"),
        ({"strategy": "iron_condor", "source": "synthetic", "underliers": ["XYZ"]}, "Unknown underliers"),
        ({"strategy": "iron_condor", "execution": "bogus", "source": "synthetic"}, "execution"),
        ({"strategy": "iron_condor", "source": "synthetic", "synthetic_days": 9999}, "synthetic_days"),
    ])
    def test_bad_input_is_a_clear_tool_error_not_a_crash(self, run, args, needle):
        async def f(c):
            r = await c.call_tool("run_backtest", args)
            return r.is_error, r.content[0].text
        is_error, text = run(f)
        assert is_error and needle.lower() in text.lower()

    def test_no_matching_days_is_an_error(self, run):
        async def f(c):
            r = await c.call_tool("run_backtest", {"strategy": "fixed_window", "start_date": "1999-01-01", "end_date": "1999-02-01", **SYN})
            return r.is_error, r.content[0].text
        is_error, text = run(f)
        assert is_error and "No results" in text


class TestInspectAndCompare:

    def _two_runs(self, c):
        async def go():
            await c.call_tool("run_backtest", {"strategy": "iron_condor", **SYN})
            await c.call_tool("run_backtest", {"strategy": "short_strangle", **SYN})
        return go()

    def test_get_run_sections(self, run):
        async def f(c):
            await c.call_tool("run_backtest", {"strategy": "short_strangle", **SYN})
            r = await c.call_tool("get_run", {"run_id": "run_1", "max_trades": 3,
                                              "sections": ["statistics", "risk", "costs", "behaviour", "daily", "trades"]})
            return r.structured_content
        o = run(f)
        _json_safe(o)
        assert {"total_pnl", "sharpe", "max_drawdown", "var_95"} <= set(o["statistics"]["SPY"])
        assert {"var_95", "cvar_95", "ulcer_index"} <= set(o["risk"]["metrics"]["SPY"]) and o["risk"]["sharpe_ci95"]
        assert o["costs"]["breakdown"][-1]["index"] == "Total" and o["costs"]["break_even_extra_bps"] is not None
        assert len(o["daily"]) == 5 and "Cumulative" in o["daily"][0]
        assert o["trades"]["returned"] == 3 and o["trades"]["total"] == 20 and "instrument" not in o["trades"]["rows"][0]
        assert o["behaviour"]["by_weekday"]

    def test_unknown_run_id_lists_known_runs(self, run):
        async def f(c):
            await c.call_tool("run_backtest", {"strategy": "fixed_window", **SYN})
            r = await c.call_tool("get_run", {"run_id": "run_99"})
            return r.is_error, r.content[0].text
        is_error, text = run(f)
        assert is_error and "run_99" in text and "run_1" in text

    def test_compare_runs(self, run):
        async def f(c):
            await self._two_runs(c)
            return (await c.call_tool("compare_runs", {"run_ids": ["run_1", "run_2"]})).structured_content
        o = run(f)
        assert [r["strategy"] for r in o["runs"]] == ["Iron Condor", "Short Strangle"]
        assert o["highest_net_pnl"] in ("run_1", "run_2") and "caution" in o and o["notes"] == []

    def test_compare_flags_different_data(self, run):
        async def f(c):
            await c.call_tool("run_backtest", {"strategy": "fixed_window", **SYN})
            await c.call_tool("run_backtest", {"strategy": "fixed_window", **{**SYN, "synthetic_days": 8}})
            return (await c.call_tool("compare_runs", {"run_ids": ["run_1", "run_2"]})).structured_content
        assert any("different data" in n for n in run(f)["notes"])

    def test_compare_needs_two_runs(self, run):
        async def f(c):
            r = await c.call_tool("compare_runs", {"run_ids": ["run_1"]})
            return r.is_error
        assert run(f)


class TestExport:

    def test_pdf_is_written_to_the_output_dir(self, run, tmp_path):
        async def f(c):
            await c.call_tool("run_backtest", {"strategy": "fixed_window", **SYN})
            return (await c.call_tool("export_report", {"run_id": "run_1"})).structured_content
        o = run(f)
        pdf = (tmp_path / "out" / "report_run_1.pdf")
        assert o["path"] == str(pdf) and pdf.read_bytes().startswith(b"%PDF") and o["bytes"] == pdf.stat().st_size

    @pytest.mark.parametrize("name", ["../../evil.pdf", "/etc/passwd", "a b/../c$.txt"])
    def test_filename_cannot_escape_the_output_dir(self, run, tmp_path, name):
        async def f(c):
            await c.call_tool("run_backtest", {"strategy": "fixed_window", **SYN})
            return (await c.call_tool("export_report", {"run_id": "run_1", "filename": name})).structured_content
        o = run(f)
        from pathlib import Path
        assert Path(o["path"]).parent == tmp_path / "out" and o["path"].endswith(".pdf")
        assert not (tmp_path / "evil.pdf").exists()


class TestPaperTrade:

    def test_paper_session_with_audit_trail(self, run):
        async def f(c):
            return (await c.call_tool("paper_trade", {"strategy": "iron_condor", "ticker": "SPY", "source": "synthetic"})).structured_content
        o = run(f)
        _json_safe(o)
        assert o["mode"].startswith("paper") and "no orders" in o["mode"]
        assert o["fills"] == 8 and o["positions_at_end"] == 0
        events = [e["event"] for e in o["audit_log"]]
        assert events[0] == "start" and "fill" in events

    def test_unknown_ticker(self, run):
        async def f(c):
            r = await c.call_tool("paper_trade", {"strategy": "iron_condor", "ticker": "NOPE", "source": "synthetic"})
            return r.is_error, r.content[0].text
        is_error, text = run(f)
        assert is_error and "Unknown ticker" in text


class TestResourcesAndPrompts:

    def test_resources(self, run):
        async def f(c):
            strategies = json.loads((await c.read_resource("stratbt://strategies")).contents[0].text)
            await c.call_tool("run_backtest", {"strategy": "fixed_window", **SYN})
            runs = json.loads((await c.read_resource("stratbt://runs")).contents[0].text)
            one = json.loads((await c.read_resource("stratbt://runs/run_1")).contents[0].text)
            return strategies, runs, one
        strategies, runs, one = run(f)
        assert len(strategies["strategies"]) == 6 and runs["runs"][0]["run_id"] == "run_1" and one["run_id"] == "run_1"

    def test_prompts_steer_towards_honest_evaluation(self, run):
        async def f(c):
            ev = await c.get_prompt("evaluate_strategy", {"strategy": "short_strangle", "ticker": "QQQ"})
            return ev.messages[0].content.text
        text = run(f)
        assert "short_strangle" in text and "QQQ" in text
        assert "execution='realistic'" in text and "execution='instant'" in text and "modelled" in text


class TestYahooPath:

    def test_yahoo_source_through_the_protocol(self, run, monkeypatch):
        import stratbt.mcp_server as ms
        from datetime import date
        real = ms.YahooFinanceProvider
        monkeypatch.setattr(ms, "YahooFinanceProvider", lambda tickers, **kw: real(
            tickers, **{**kw, "end": date(2024, 3, 15), "downloader": FakeDownloader()}))

        async def f(c):
            chk = (await c.call_tool("check_market_data", {"tickers": ["spy"], "lookback_days": 20})).structured_content
            bt = (await c.call_tool("run_backtest", {"strategy": "iron_condor", "tickers": ["spy"], "lookback_days": 20})).structured_content
            return chk, bt
        chk, bt = run(f)
        assert chk["tickers"] == ["SPY"] and chk["session"].startswith("09:30") and chk["contract_multiplier"] == {"SPY": 100.0}
        assert any("modelled" in w for w in bt["warnings"]) and bt["config"]["source"] == "yahoo"
        assert bt["headline"]["trading_days"] == 12
        _json_safe(bt)

    def test_network_failure_is_reported(self, run, monkeypatch):
        import stratbt.mcp_server as ms

        def boom(*a, **k):
            raise ConnectionError("offline")
        monkeypatch.setattr(ms, "YahooFinanceProvider", boom)

        async def f(c):
            r = await c.call_tool("run_backtest", {"strategy": "iron_condor", "tickers": ["SPY"]})
            return r.is_error, r.content[0].text
        is_error, text = run(f)
        assert is_error and "Could not load Yahoo Finance data" in text


@pytest.mark.parametrize("argv", [["-m", "stratbt.mcp_server"], ["-m", "stratbt.cli", "mcp"]], ids=["module", "cli"])
def test_real_stdio_transport_end_to_end(argv, tmp_path):
    """Spawn the server exactly as a desktop agent would and talk to it over stdin/stdout."""
    params = StdioServerParameters(command=sys.executable, args=argv)

    async def main():
        async with Client(params) as c:
            tools = {t.name for t in (await c.list_tools()).tools}
            r = await c.call_tool("run_backtest", {"strategy": "fixed_window", **SYN})
            return tools, r.is_error, r.structured_content
    tools, is_error, payload = anyio.run(main)
    assert "run_backtest" in tools and not is_error and payload["run_id"] == "run_1"


def test_real_streamable_http_transport_end_to_end(tmp_path):
    """The same server for remote clients: start it on a free port, connect over HTTP."""
    import socket
    import subprocess
    import time
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    proc = subprocess.Popen([sys.executable, "-m", "stratbt.cli", "mcp", "--transport", "streamable-http",
                             "--port", str(port), "--output-dir", str(tmp_path)],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(100):
            try:
                socket.create_connection(("127.0.0.1", port), timeout=0.2).close()
                break
            except OSError:
                time.sleep(0.2)
        else:
            pytest.fail("HTTP MCP server did not start")

        async def main():
            async with Client(f"http://127.0.0.1:{port}/mcp") as c:
                r = await c.call_tool("run_backtest", {"strategy": "fixed_window", **SYN})
                e = await c.call_tool("export_report", {"run_id": "run_1"})
                return c.server_info.name, r.is_error, e.structured_content["path"]
        name, is_error, pdf_path = anyio.run(main)
        assert name == "stratbt" and not is_error and (tmp_path / "report_run_1.pdf").exists()
    finally:
        proc.terminate()
        proc.wait(timeout=10)


def test_without_the_mcp_package_the_cli_says_how_to_install(monkeypatch, capsys):
    import builtins
    from stratbt import cli
    real_import = builtins.__import__

    def fake(name, *a, **k):
        if name == "stratbt.mcp_server":
            raise ImportError("The MCP server needs the 'mcp' package (version 2 or newer):  pip install 'strat-backtester[agent]'")
        return real_import(name, *a, **k)
    monkeypatch.setattr(builtins, "__import__", fake)
    assert cli.main(["mcp"]) == 1
    assert "strat-backtester[agent]" in capsys.readouterr().out
