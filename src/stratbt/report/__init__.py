# streamlit_app/report/__init__.py
"""
Options backtest PDF tearsheet package.

Three-stage pipeline (AWS BI pattern):
  1. data.py   — fetch:  pulls all numbers out of BacktestResult into ReportData
  2. pages.py  — merge:  lays out matplotlib figures using a top-down cursor
  3. __init__  — render: stamps chrome and exports each figure as a PDF page

Public entry point: `build_pdf_report(result, strategy_name, period_label)`.
"""

from io import BytesIO

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from stratbt.engine import BacktestResult

from .data import ReportData, build_report_data
from .pages import build_all_pages
from . import widgets as w


def build_pdf_report(
    result: BacktestResult,
    strategy_name: str,
    period_label: str,
) -> bytes:
    """
    Full entry point: build a well-structured multi-page PDF tearsheet.

    Returns raw PDF bytes suitable for ``st.download_button`` or writing
    directly to a file.  All heavy computation happens in build_report_data;
    the pages/ widgets layers are pure layout/drawing with no pandas work.
    """
    data:  ReportData                    = build_report_data(result, strategy_name, period_label)
    pages: list[tuple[plt.Figure, str]]  = build_all_pages(data)

    total_pages = len(pages)
    buffer      = BytesIO()

    with PdfPages(buffer) as pdf:
        for page_num, (fig, section) in enumerate(pages, start=1):
            w.chrome(fig, section, page_num, total_pages,
                     data.generated_at, data.strategy_name)
            pdf.savefig(fig, dpi=150)
            plt.close(fig)

    return buffer.getvalue()


__all__ = ["build_pdf_report", "build_report_data", "ReportData"]
