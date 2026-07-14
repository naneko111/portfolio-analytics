"""Console summary + lightweight HTML report."""

from __future__ import annotations
import base64
from pathlib import Path
import pandas as pd

from . import config


def _img_b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode()


def print_table(df: pd.DataFrame, title: str = ""):
    if title: print(f"\n=== {title} ===")
    print(df.to_string())


def build_html(summary_df: pd.DataFrame, charts: dict, path: str):
    rows = "".join(
        f"<tr><td>{k}</td>" + "".join(f"<td>{v:.4f}</td>" for v in row)
        for k, row in summary_df.iterrows()
    )
    imgs = "".join(
        f'<h3>{name.replace("_"," ").title()}</h3><img src="data:image/png;base64,{_img_b64(Path(p))}">'
        for name, p in charts.items() if Path(p).exists()
    )
    html = f"""
    <html><head><style>
      body{{font-family:system-ui;max-width:1100px;margin:30px auto;padding:0 20px}}
      table{{border-collapse:collapse;width:100%;margin:20px 0}}
      th,td{{border:1px solid #ddd;padding:8px;text-align:right}}
      th{{background:#333;color:#fff}}
      img{{max-width:100%;border:1px solid #ccc;margin:10px 0}}
    </style></head><body>
      <h1>Portfolio Risk &amp; Analytics Report</h1>
      <p>Period: {config.START_DATE.date()} → {config.END_DATE.date()}</p>
      <h2>Summary</h2>
      <table><tr><th>Asset</th>{"".join(f"<th>{c}</th>" for c in summary_df.columns)}</tr>{rows}</table>
      <h2>Charts</h2>{imgs}
    </body></html>"""
    Path(path).write_text(html)
