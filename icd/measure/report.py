from __future__ import annotations

import csv
import json
import os
from typing import Dict


def write_csv_report(out_dir: str, metrics: Dict[str, object]) -> str:
    path = os.path.join(out_dir, "report.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        for k, v in metrics.items():
            w.writerow([k, json.dumps(v)])
    return path


def write_html_report(out_dir: str, metrics: Dict[str, object]) -> str:
    path = os.path.join(out_dir, "report.html")
    # simple HTML table of key-value pairs
    rows = "".join(f"<tr><th>{k}</th><td><pre>{html_escape(str(v))}</pre></td></tr>" for k, v in metrics.items())
    html = f"""
    <html><head><meta charset='utf-8'><title>ICD Report</title>
    <style>body{{font-family:system-ui,Arial,sans-serif}} table{{border-collapse:collapse}} th,td{{border:1px solid #ccc;padding:4px 8px;vertical-align:top}}</style>
    </head><body><h1>ICD Report</h1><table>{rows}</table></body></html>
    """
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    return path


def html_escape(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


__all__ = ["write_csv_report", "write_html_report"]
