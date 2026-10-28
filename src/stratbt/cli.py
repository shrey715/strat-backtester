# src/stratbt/cli.py
"""Command line: `stratbt strategies | run | app`."""
import argparse
import os
import subprocess
import sys
from datetime import date
from pathlib import Path



def build_provider(args):
    from stratbt.providers import SyntheticProvider, YahooFinanceProvider
    if args.provider == "synthetic":
        return SyntheticProvider(n_days=args.days, seed=args.seed)
    if not args.tickers:
        raise SystemExit("--provider yahoo needs --tickers, e.g. --tickers SPY QQQ  or  --tickers ^NSEI")
    return YahooFinanceProvider(
        args.tickers, interval=args.interval, lookback_days=args.lookback, iv_premium=args.iv_premium,
        vol=args.vol, expiry_weekday=args.expiry_weekday, multiplier=args.multiplier,
    )


def build_execution(name: str):
    from stratbt.execution import ExecutionModel
    return {"instant": ExecutionModel.instant, "default": ExecutionModel,
            "realistic": ExecutionModel.realistic}[name]()


def _parse_params(items: list[str]) -> dict:
    out = {}
    for item in items:
        key, sep, val = item.partition("=")
        if not sep:
            raise SystemExit(f"--param expects KEY=VALUE, got '{item}'")
        out[key] = float(val)
    return out


def cmd_strategies(_args) -> int:
    from stratbt.strategy import registry
    for key, spec in sorted(registry.available().items()):
        print(f"{key:16s} {spec.label}")
        print(f"{'':16s} {spec.description}")
        for p in spec.params:
            print(f"{'':16s}   --param {p.name}=<{p.min}..{p.max}>  (default {p.default})")
    return 0


def cmd_run(args) -> int:
    import logging
    import pandas as pd

    from stratbt import metrics
    from stratbt.api import run_backtest
    from stratbt.risk import RiskLimits

    logging.basicConfig(level=logging.WARNING)
    provider = build_provider(args)
    risk = RiskLimits(max_daily_loss=args.max_daily_loss, max_open_contracts=args.max_contracts)
    execution = _execution_from(args)

    result = run_backtest(
        args.strategy, provider, _parse_params(args.param), execution, risk,
        underliers=args.underliers or None,
        start_date=date.fromisoformat(args.start) if args.start else None,
        end_date=date.fromisoformat(args.end) if args.end else None,
    )
    if not result.day_results:
        print("No results produced.")
        return 1
    pd.set_option("display.width", 200, "display.max_columns", 30)
    note = "  (real underlying bars; options priced by Black-Scholes)" if provider.name == "yahoo" else ""
    print(f"\n{args.strategy} on {provider.name} data — PnL in {result.unit}, execution={args.execution}{note}\n")
    print(metrics.summary_table(result).T.to_string())
    fees = sum(dr.fees for dr in result.day_results)
    rejected = sum(len(dr.rejections) for dr in result.day_results)
    print(f"\nfees paid: {fees:,.2f}   rejected orders: {rejected}")

    if args.pdf:
        from stratbt.report import build_pdf_report
        d = result.daily_pnl.index
        period = f"{d.min():%b %d} – {d.max():%b %d}"
        Path(args.pdf).write_bytes(build_pdf_report(result, args.strategy, period))
        print(f"wrote {args.pdf}")
    return 0


def cmd_app(args) -> int:
    try:
        import streamlit  # noqa: F401
    except ImportError:
        print("The dashboard needs extra dependencies:  pip install 'strat-backtester[app]'")
        return 1
    app = Path(__file__).parent / "dashboard" / "app.py"
    cmd = [sys.executable, "-m", "streamlit", "run", str(app), "--server.port", str(args.port),
           "--browser.gatherUsageStats", "false",
           "--theme.primaryColor", "#2a78d6"]
    return subprocess.call(cmd)


def _execution_from(args):
    from stratbt.execution import ExecutionModel
    execution = build_execution(args.execution)
    overrides = {"delay_s": args.delay_s, "half_spread_bps": args.half_spread_bps,
                 "min_half_spread": args.min_half_spread, "commission_per_contract": args.commission}
    overrides = {k: v for k, v in overrides.items() if v is not None}
    return ExecutionModel(**{**execution.__dict__, **overrides}) if overrides else execution


def cmd_paper(args) -> int:
    """Replay one session through the live runner with a paper broker (rehearsal for a real feed)."""
    from datetime import date as _date
    from stratbt.broker import PaperBroker
    from stratbt.live import AuditLog, KillSwitch, LiveRunner, ReplayFeed
    from stratbt.risk import RiskLimits
    from stratbt.strategy import registry

    provider = build_provider(args)
    day = provider.trading_days[-1] if args.day == "last" else _date.fromisoformat(args.day)
    underlier = args.underlier or provider.underliers[0]
    spec = registry.get(args.strategy)
    feed = ReplayFeed(provider, day, underlier)
    strategy = spec.factory(provider, **_parse_params(args.param))(underlier, feed.expiry, feed.strikes)
    execution = _execution_from(args)
    broker = PaperBroker(provider.multiplier(underlier), execution,
                         RiskLimits(max_daily_loss=args.max_daily_loss, max_open_contracts=args.max_contracts),
                         allow_short=strategy.allow_short)
    runner = LiveRunner(strategy, feed, broker, execution, audit=AuditLog(args.audit_log),
                        kill=KillSwitch(args.kill_file), state_path=args.state, adopt_existing=args.adopt)
    s = runner.run()
    print(f"{spec.label} · {underlier} · {day} · paper trading on {provider.name} data")
    print(f"  net PnL {s.net_pnl:,.2f} {provider.unit} · {s.n_fills} fills · {s.n_rejections} rejections"
          f"{' · HALTED by risk limit' if s.halted else ''}{' · KILLED' if s.killed else ''}")
    if args.audit_log:
        print(f"  audit log: {args.audit_log}")
    return 0


def _add_market_args(r) -> None:
    r.add_argument("--provider", choices=["synthetic", "yahoo"], default="synthetic",
                   help="synthetic: built-in simulator; yahoo: real bars from Yahoo Finance (needs the [data] extra)")
    r.add_argument("--seed", type=int, default=7, help="synthetic market seed")
    r.add_argument("--days", type=int, default=21, help="synthetic market length in trading days")
    r.add_argument("--tickers", nargs="+", metavar="SYMBOL", help="Yahoo symbols, e.g. SPY QQQ  |  ^NSEI ^NSEBANK  |  RELIANCE.NS")
    r.add_argument("--interval", default="5m", choices=["1m", "2m", "5m", "15m", "30m", "60m", "90m"], help="Yahoo bar size")
    r.add_argument("--lookback", type=int, help="calendar days of Yahoo history (default 30, capped by what Yahoo keeps)")
    r.add_argument("--iv-premium", type=float, default=0.10, help="implied vol = trailing realised vol * (1 + this)")
    r.add_argument("--vol", type=float, help="fixed annualised implied vol instead of the trailing estimate")
    r.add_argument("--expiry-weekday", type=int, default=4, help="weekly expiry day, 0=Mon … 4=Fri")
    r.add_argument("--multiplier", type=float, help="contract multiplier (default 100 for USD, 1 otherwise)")


def _add_exec_args(r) -> None:
    r.add_argument("--param", action="append", default=[], metavar="KEY=VALUE", help="strategy parameter")
    r.add_argument("--execution", choices=["instant", "default", "realistic"], default="default",
                   help="fill model: instant (same-second, free), default (next-second, free), realistic (next-second + spread + fees)")
    r.add_argument("--delay-s", type=int, help="override fill delay in seconds (0 = same-second)")
    r.add_argument("--half-spread-bps", type=float, help="override half-spread, basis points of price")
    r.add_argument("--min-half-spread", type=float, help="override minimum half-spread, in price units")
    r.add_argument("--commission", type=float, help="override commission per contract")
    r.add_argument("--max-daily-loss", type=float, help="halt and flatten at this daily loss (currency)")
    r.add_argument("--max-contracts", type=int, help="cap on total contracts held")


def cmd_live(args) -> int:
    """Run a strategy against a real quote feed and broker. Defaults to dry-run: nothing is ever sent."""
    import os
    import signal
    from datetime import datetime
    from types import SimpleNamespace

    from stratbt.adapters import BROKERS, REAL_MONEY_ENV, REAL_MONEY_TOKEN, build
    from stratbt.execution import ExecutionModel
    from stratbt.live import AuditLog, KillSwitch, LiveRunner
    from stratbt.strategy import registry

    mode = args.mode
    if mode != "dry-run" and args.max_daily_loss is None:
        raise SystemExit(f"--max-daily-loss is required in {mode} mode (the hard stop for the day, in account currency).")
    confirm = None
    if mode == "live":
        confirm = os.environ.get(REAL_MONEY_ENV)
        if confirm != REAL_MONEY_TOKEN:
            if not sys.stdin.isatty():
                raise SystemExit(f"Real-money trading needs confirmation: set {REAL_MONEY_ENV}={REAL_MONEY_TOKEN}")
            print(f"\n  !! REAL MONEY: orders will be sent to your live {args.broker} account.\n"
                  f"  Max {args.max_order_qty} contract(s) per order, {args.max_open_contracts} open, daily loss limit {args.max_daily_loss}.")
            confirm = input(f"  Type {REAL_MONEY_TOKEN} to continue: ").strip()

    venue = {}
    if args.broker == "kite":
        venue = {k: v for k, v in dict(exchange=args.kite_exchange, product=args.kite_product, index_key=args.kite_index_key).items() if v}
    if args.broker == "ibkr":
        venue = {k: v for k, v in dict(host=args.ib_host, port=args.ib_port, exchange=args.ib_exchange,
                                       currency=args.ib_currency, multiplier=args.ib_multiplier,
                                       underlying_kind=args.ib_underlying, delayed=args.ib_delayed).items() if v not in (None, False)}
    spec = registry.get(args.strategy)
    feed, broker = build(args.broker, args.ticker, mode=mode, max_daily_loss=args.max_daily_loss,
                         max_open_contracts=args.max_open_contracts, max_order_qty=args.max_order_qty,
                         confirm_real_money=confirm, **venue)
    session = SimpleNamespace(session_open=feed.session_open, session_close=feed.session_close)
    strategy = spec.factory(session, **_parse_params(args.param))(feed.underlier, feed.expiry, feed.strikes)
    broker.position.allow_short = strategy.allow_short

    stamp = datetime.now().strftime("%Y%m%d")
    base = Path(args.log_dir or Path.home() / ".stratbt" / "live") / f"{stamp}_{args.broker}_{feed.underlier}_{mode}"
    kill = KillSwitch(args.kill_file or f"{base}.STOP")
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: kill.kill())          # Ctrl-C flattens positions and stops cleanly

    banner = {"dry-run": "DRY RUN — real quotes and decisions, NO ORDERS ARE SENT",
              "paper": f"PAPER — orders go to the {args.broker} paper account",
              "live": "LIVE — REAL MONEY"}[mode]
    print(f"{spec.label} · {feed.underlier} · expiry {feed.expiry} · {args.broker} · {banner}")
    print(f"  limits: {broker.risk.max_order_qty}/order, {broker.risk.max_open_contracts} open, daily loss {broker.risk.max_daily_loss}")
    print(f"  stop it:  touch {kill.path}   (or Ctrl-C)      audit: {base}.jsonl")
    runner = LiveRunner(strategy, feed, broker, ExecutionModel.instant(), audit=AuditLog(f"{base}.jsonl"), kill=kill,
                        state_path=f"{base}.state.json", adopt_existing=args.adopt)
    s = runner.run()
    print(f"  done: net PnL {s.net_pnl:,.2f} · {s.n_fills} fills · {s.n_rejections} rejections"
          f"{' · HALTED by loss limit' if s.halted else ''}{' · KILLED' if s.killed else ''}")
    if s.open_positions:
        print(f"  !! {s.open_positions} POSITION(S) STILL OPEN — check your broker account now.")
        return 2
    return 0


def cmd_kite_login(args) -> int:
    """Get today's Zerodha access token (it expires daily)."""
    import os
    from stratbt.adapters.kite import exchange_request_token, login_url
    if not os.environ.get("KITE_API_KEY"):
        raise SystemExit("Set KITE_API_KEY (and KITE_API_SECRET) from your Kite Connect app first.")
    print("1. Open this URL, log in to Zerodha, and approve the app:\n\n   " + login_url() + "\n")
    print("2. You are redirected to your app's redirect URL. Paste that whole URL (or just the request_token value) here.")
    raw = input("   > ").strip()
    token = exchange_request_token(raw)
    print("\nToday's access token (valid until the morning; treat it like a password):\n")
    print(f"   export KITE_ACCESS_TOKEN={token}\n")
    return 0


def cmd_mcp(args) -> int:
    try:
        from stratbt.mcp_server import main as serve
    except ImportError as e:
        print(f"{e}")
        return 1
    serve(transport=args.transport, host=args.host, port=args.port, output_dir=args.output_dir)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="stratbt", description="Options strategy backtesting engine.")
    sub = ap.add_subparsers(dest="command", required=True)

    sub.add_parser("strategies", help="list available strategies and their parameters").set_defaults(func=cmd_strategies)

    r = sub.add_parser("run", help="run a backtest and print a summary")
    r.add_argument("strategy", help="strategy key (see `stratbt strategies`)")
    _add_market_args(r)
    r.add_argument("--underliers", nargs="+")
    r.add_argument("--start", help="YYYY-MM-DD")
    r.add_argument("--end", help="YYYY-MM-DD")
    _add_exec_args(r)
    r.add_argument("--pdf", help="write a PDF tearsheet to this path")
    r.set_defaults(func=cmd_run)

    pp = sub.add_parser("paper", help="replay one session through the live runner with a paper broker")
    pp.add_argument("strategy")
    _add_market_args(pp)
    _add_exec_args(pp)
    pp.add_argument("--day", default="last", help="YYYY-MM-DD or 'last' (default)")
    pp.add_argument("--underlier", help="which underlier to trade (default: the first)")
    pp.add_argument("--audit-log", help="write a JSON-lines audit trail here")
    pp.add_argument("--state", help="JSON file holding saved positions (reconciled on start)")
    pp.add_argument("--kill-file", help="create this file to flatten everything and stop")
    pp.add_argument("--adopt", action="store_true", help="start even if the broker holds unknown positions")
    pp.set_defaults(func=cmd_paper)

    lv = sub.add_parser("live", help="trade a strategy on a real feed/broker (default: dry-run, sends nothing)")
    lv.add_argument("strategy")
    lv.add_argument("--broker", choices=["alpaca", "public", "ibkr", "kite"], required=True)
    lv.add_argument("--ticker", required=True, help="underlying, e.g. SPY")
    lv.add_argument("--mode", choices=["dry-run", "paper", "live"], default="dry-run",
                    help="dry-run: decide on real quotes, send nothing (default) · paper: venue's paper account · live: REAL MONEY")
    lv.add_argument("--param", action="append", default=[], metavar="KEY=VALUE", help="strategy parameter")
    lv.add_argument("--max-daily-loss", type=float, help="hard stop for the day (required for paper and live)")
    lv.add_argument("--max-order-qty", type=int, default=1, help="contracts per order (default 1)")
    lv.add_argument("--max-open-contracts", type=int, default=4, help="total contracts held at once (default 4)")
    lv.add_argument("--kill-file", help="create this file to flatten everything and stop")
    lv.add_argument("--log-dir", help="where the audit log and state file go (default ~/.stratbt/live)")
    lv.add_argument("--adopt", action="store_true", help="start even if the account holds positions the saved state doesn't know")
    lv.add_argument("--kite-exchange", choices=["NFO", "BFO"], default=None, help="Kite: NFO (NSE, default) or BFO (BSE)")
    lv.add_argument("--kite-product", choices=["NRML", "MIS"], default=None,
                    help="Kite: NRML (default) or MIS (Zerodha auto-squares MIS around 15:20)")
    lv.add_argument("--kite-index-key", default=None, help="Kite: override the underlying quote key, e.g. 'NSE:NIFTY BANK'")
    lv.add_argument("--ib-host", default=None); lv.add_argument("--ib-port", type=int, default=None)
    lv.add_argument("--ib-exchange", default=None); lv.add_argument("--ib-currency", default=None)
    lv.add_argument("--ib-multiplier", default=None); lv.add_argument("--ib-underlying", choices=["Stock", "Index"], default=None)
    lv.add_argument("--ib-delayed", action="store_true", help="use delayed market data")
    lv.set_defaults(func=cmd_live)

    sub.add_parser("kite-login", help="get today's Zerodha Kite access token").set_defaults(func=cmd_kite_login)

    m = sub.add_parser("mcp", help="serve the backtester to AI agents over the Model Context Protocol")
    m.add_argument("--transport", choices=["stdio", "streamable-http"], default="stdio",
                   help="stdio for desktop agents and IDEs (default); streamable-http for remote clients")
    m.add_argument("--host", default="127.0.0.1", help="bind address for streamable-http")
    m.add_argument("--port", type=int, default=8765, help="port for streamable-http")
    m.add_argument("--output-dir", help="where export_report writes PDFs (default ~/stratbt-reports)")
    m.set_defaults(func=cmd_mcp)

    a = sub.add_parser("app", help="launch the Streamlit dashboard")
    a.add_argument("--port", type=int, default=8501)
    a.set_defaults(func=cmd_app)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
