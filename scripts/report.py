"""Render the end-to-end report.

Runnable as `python scripts/report.py` from any working directory: Python puts
`scripts/` (not the repository root) on sys.path for a script invoked by path,
so without the insertion below `import rtforge` only works when the package has
been pip-installed. This is what made the CI "the report runs end to end" step
fail with ModuleNotFoundError on every run.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from rtforge.eval.harness import build_report, render

r = build_report()
txt = render(r)
out = os.path.join(ROOT, "results", "report.txt")
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, "w", encoding="utf-8", newline="\n") as f:
    f.write(txt)
print(txt)
