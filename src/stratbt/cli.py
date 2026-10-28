# src/stratbt/cli.py
"""Command line: `stratbt strategies | run | app`."""
import argparse
import os
import subprocess
import sys
from datetime import date
from pathlib import Path

DEFAULT_DATA_ENV = "STRATBT_DATA_DIR"


def build_provider(kind: str, data_dir: str | None, seed: int, days: int):
    from stratbt.providers import NSETickProvider, SyntheticProvider
    if kind == "synthetic":
        return SyntheticProvider(n_days=days, seed=seed)
    path = data_dir or os.environ.get(DEFAULT_DATA_ENV) or "data/allData"
    return NSETickProvider(path)


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
    from stratbt.execution import ExecutionModel
    from stratbt.risk import RiskLimits

    logging.basicConfig(level=logging.WARNING)
    provider = build_provider(args.provider, args.data_dir, args.seed, args.days)
    risk = RiskLimits(max_daily_loss=args.max_daily_loss, max_open_contracts=args.max_contracts)
    execution = build_execution(args.execution)
    if args.commission is not None:
        execution = ExecutionModel(**{**execution.__dict__, "commission_per_contract": args.commission})

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
    print(f"\n{args.strategy} on {provider.name} data — PnL in {result.unit}, execution={args.execution}\n")
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
           "--browser.gatherUsageStats", "false"]
    if args.data_dir:
        os.environ[DEFAULT_DATA_ENV] = args.data_dir
    return subprocess.call(cmd)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="stratbt", description="Options strategy backtesting engine.")
    sub = ap.add_subparsers(dest="command", required=True)

    sub.add_parser("strategies", help="list available strategies and their parameters").set_defaults(func=cmd_strategies)

    r = sub.add_parser("run", help="run a backtest and print a summary")
    r.add_argument("strategy", help="strategy key (see `stratbt strategies`)")
    r.add_argument("--provider", choices=["synthetic", "nse"], default="synthetic")
    r.add_argument("--data-dir", help=f"NSE tick data folder (or set ${DEFAULT_DATA_ENV})")
    r.add_argument("--seed", type=int, default=7, help="synthetic market seed")
    r.add_argument("--days", type=int, default=21, help="synthetic market length in trading days")
    r.add_argument("--underliers", nargs="+")
    r.add_argument("--start", help="YYYY-MM-DD")
    r.add_argument("--end", help="YYYY-MM-DD")
    r.add_argument("--param", action="append", default=[], metavar="KEY=VALUE", help="strategy parameter")
    r.add_argument("--execution", choices=["instant", "default", "realistic"], default="default",
                   help="fill model: instant (same-second, free), default (next-second, free), realistic (next-second + spread + fees)")
    r.add_argument("--commission", type=float, help="override commission per contract")
    r.add_argument("--max-daily-loss", type=float, help="halt and flatten at this daily loss (currency)")
    r.add_argument("--max-contracts", type=int, help="cap on total contracts held")
    r.add_argument("--pdf", help="write a PDF tearsheet to this path")
    r.set_defaults(func=cmd_run)

    a = sub.add_parser("app", help="launch the Streamlit dashboard")
    a.add_argument("--port", type=int, default=8501)
    a.add_argument("--data-dir", help="NSE tick data folder to offer in the dashboard")
    a.set_defaults(func=cmd_app)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
