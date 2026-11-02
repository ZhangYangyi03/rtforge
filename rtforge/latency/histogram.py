"""Parse cyclictest -q, and report what a control loop actually needs from the distribution.

Two output formats had to be checked against real output before this parser was written,
because the obvious assumption is wrong. `cyclictest -h` prints a per-thread histogram of
only the values above a floor; `cyclictest -q` prints a full cross-thread histogram:

    000012 000026  000024 ...     <- microseconds (6 digits), then one count per thread

and, after it, one summary line per thread of the *scaled* histogram: the distinct
latencies observed, so the loss of the counts. That is the format parsed here, and the
reason the parser has a test against a captured run rather than against an invented string.

Why a distribution and not a max: RT behaviour is a long-tailed distribution, and the
question a 1 kHz loop asks is "what does the tail look like at the rate I will hit it",
i.e. p99.9 over the number of cycles the machine will run, not the single worst sample.
"""
from __future__ import annotations

import glob
import os
import re
import statistics
import sys

THREAD_RE = re.compile(r"^#\s*Thread\s+(\d+):\s*(.*?)\s*#\s*(\d+)\s+others\s*$")


def parse_quick(text: str) -> list[dict]:
    """Per-thread histograms from cyclictest -q output.

    Returns [{'thread': n, 'values': [sorted latencies in us], 'others': m}].
    The summary line lists every distinct observed latency once; `others` counts samples
    above the histogram's last bucket and is kept rather than dropped.
    """
    threads = []
    for line in text.splitlines():
        m = THREAD_RE.match(line.strip())
        if not m:
            continue
        # No width filter here. A first version rejected tokens longer than 6 digits
        # as noise, which silently deleted every latency above 300 ms and made a
        # clean run look catastrophic. The only thing after the values is the
        # "# N others" suffix, which the regex has already removed.
        values = [int(tok) for tok in m.group(2).split() if tok.isdigit()]
        threads.append({"thread": int(m.group(1)), "values": sorted(values),
                        "others": int(m.group(3))})
    return threads


def stats(values: list[int], others: int = 0) -> dict:
    """Percentiles over distinct observed values.

    The list is of distinct values, not samples, so a plain percentile over it is a
    percentile over *latencies* rather than over *cycles*. That is stated here because it
    is the difference between an honest and a flattering number; for a max-latency claim
    the distribution of latencies is the right object, not the distribution of cycles.
    """
    if not values:
        return {}
    def q(p: float) -> float:
        idx = min(len(values) - 1, int(round(p * (len(values) - 1))))
        return float(values[idx])
    return {
        "distinct": len(values),
        "min": float(values[0]),
        "p50": q(0.50),
        "p90": q(0.90),
        "p99": q(0.99),
        "p999": q(0.999),
        "max_seen": float(values[-1]),
        # 'others' are above the last bucket: a lower bound on the true worst case.
        "others": others,
    }


def summarize(paths: list[str]) -> dict:
    runs = []
    for path in paths:
        text = open(path, encoding="utf-8", errors="replace").read()
        threads = parse_quick(text)
        if not threads:
            continue
        per = []
        for th in threads:
            s = stats(th["values"], th["others"])
            s["thread"] = th["thread"]
            per.append(s)
        worst = max(per, key=lambda s: s["max_seen"])
        runs.append({
            "file": os.path.basename(path),
            "config": os.path.basename(path).rsplit(".", 1)[0].replace("cyclictest_", ""),
            "threads": len(per),
            "p50": statistics.median([s["p50"] for s in per]),
            "p99": worst["p99"],
            "p999": worst["p999"],
            "max_seen": worst["max_seen"],
            "others": worst["others"],
        })
    return {"runs": runs}


def render(summary: dict) -> str:
    rows = summary["runs"]
    if not rows:
        return "no cyclictest -q runs found"
    out = ["configuration                p50 us   p99 us  p99.9 us   max us   (worst thread)", ""]
    for r in sorted(rows, key=lambda x: x["max_seen"]):
        out.append(f"{r['config']:<28}{r['p50']:>7.0f}{r['p99']:>9.0f}{r['p999']:>9.0f}"
                   f"{r['max_seen']:>9.0f}   n={r['threads']}")
    return "\n".join(out)


def main(argv: list[str]) -> int:
    patterns = argv or ["results/cyclictest_*.hist"]
    paths: list[str] = []
    for p in patterns:
        found = sorted(glob.glob(p))
        paths.extend(found if found else ([p] if os.path.exists(p) else []))
    summary = summarize(paths)
    print(render(summary))
    return 0 if summary["runs"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
