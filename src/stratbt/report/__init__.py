"""
Options backtest PDF tearsheet.

Three-stage pipeline:
  data.py    fetch   — every number, table and sentence computed once from a BacktestResult
  pages.py   layout  — composes pages from that data (positions only)
  charts.py / layout.py / theme.py   draw — chart functions, page primitives, design tokens

Public entry point: `build_pdf_report(result, strategy_name, period_label)`.
"""
from io import BytesIO

import matplotlib as mpl
from matplotlib.backends.backend_pdf import PdfPages

from stratbt.engine import BacktestResult
from stratbt.report import theme
from stratbt.report.data import ReportData, build_report_data
from stratbt.report.pages import build_all_pages


def build_pdf_report(result: BacktestResult, strategy_name: str = "", period_label: str = "") -> bytes:
    """
    Build a multi-page PDF tearsheet and return its bytes (for
    `st.download_button` or writing straight to a file).

    The strategy name is taken from `result.meta` when the result came from
    `run_backtest`; `strategy_name` is the fallback.
    """
    data = build_report_data(result, strategy_name, period_label)
    pages = build_all_pages(data)
    buffer = BytesIO()
    with mpl.rc_context(theme.RC), PdfPages(buffer, metadata={
        "Title": f"{data.strategy_name} — backtest report", "Creator": f"stratbt {data.version}",
    }) as pdf:
        for fig, _section in pages:
            pdf.savefig(fig)
    return buffer.getvalue()


__all__ = ["build_pdf_report", "build_report_data", "ReportData"]
