#!/usr/bin/env python3
"""
Summarize latest runs into a high-level markdown report.

Reference: theory_validation.md — sections cited per test category.
"""
from __future__ import annotations

import json
import os
import sys
import re
from datetime import datetime
from pathlib import Path

# Ensure local src/ is importable
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
SRC = os.path.join(ROOT, "src")
for p in [ROOT, SRC]:
    if p not in sys.path:
        sys.path.insert(0, p)

from src.ct_cmb_validation.utils.config_loader import ensure_dir
from src.ct_cmb_validation.utils.logging_setup import setup_logging


def _collect_runs(root: Path) -> list[Path]:
    return sorted([p for p in root.glob("*_*") if p.is_dir()])


def _load_json_if(path: Path):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None


def main() -> None:
    logger = setup_logging()
    runs_root = Path("results/runs")
    ensure_dir(str(runs_root))
    runs = _collect_runs(runs_root)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("results/summary_reports")
    ensure_dir(str(out_dir))
    out_path = out_dir / f"report_{ts}.md"

    lines = ["# CT-CMB Validation Summary\n"]

    for r in reversed(runs):
        lines.append(f"\n## Run: {r.name}\n")
        fr = _load_json_if(r / "final_results.json")
        if fr is not None:
            lines.append("```")
            lines.append(json.dumps(fr, indent=2))
            lines.append("```")
        else:
            lines.append("(No final_results.json)")

    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    logger.info("Wrote summary report to %s", out_path)


if __name__ == "__main__":
    main()
