"""One threshold for every category -- the cost of not calibrating per part type.

A line with several part types runs ONE cutoff. This pools the normal-training scores of all
categories, sets the cutoff at a fixed FPR, and reports the recall each category then gets.

Slow (it re-extracts features for every category), so results are cached per category in
results/global_threshold_cache.json and a re-run only does the missing ones.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.bench_threshold import scores   # noqa: E402

ROOT = "/mnt/d/mvtec"
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "results", "global_threshold_cache.json")


def main() -> int:
    cache = json.load(open(OUT, encoding="utf-8")) if os.path.exists(OUT) else {}
    cats = [d for d in sorted(os.listdir(ROOT))
            if os.path.isdir(os.path.join(ROOT, d, "train", "good"))]
    for c in cats:
        if c in cache:
            continue
        tr_s, ts, lab = scores(ROOT, c)
        cache[c] = {"train": tr_s.tolist(), "test": ts.tolist(), "labels": lab.tolist()}
        json.dump(cache, open(OUT, "w", encoding="utf-8"))
        print(f"  cached {c} ({len(tr_s)} train, {len(ts)} test)", flush=True)

    pooled = np.concatenate([np.array(cache[c]["train"]) for c in cats])
    print(f"\n  {len(cats)} categories, {len(pooled)} normal training scores pooled")
    print(f"  {'FPR':>6}{'threshold':>12}{'mean recall':>14}{'worst category':>24}")
    rows = {}
    for fpr in (0.01, 0.05):
        thr = float(np.quantile(pooled, 1.0 - fpr))
        recs = {}
        for c in cats:
            ts = np.array(cache[c]["test"])
            lab = np.array(cache[c]["labels"])
            recs[c] = float((ts[lab == 1] > thr).mean())
        worst = min(recs, key=recs.get)
        print(f"  {fpr * 100:>5.0f}%{thr:>12.4f}{np.mean(list(recs.values())) * 100:>13.1f}%"
              f"   {worst} {recs[worst] * 100:.1f}%")
        rows[f"fpr_{int(fpr * 100)}"] = {"threshold": thr, "recall": recs,
                                         "mean_recall": float(np.mean(list(recs.values())))}
    # for comparison: per-category calibration
    print(f"\n  {'per-category calibration':<40}{'mean recall @1% FPR':>22}")
    per = []
    for c in cats:
        tr_s = np.array(cache[c]["train"])
        ts = np.array(cache[c]["test"])
        lab = np.array(cache[c]["labels"])
        thr = float(np.quantile(tr_s, 0.99))
        per.append(float((ts[lab == 1] > thr).mean()))
    print(f"  {'':40}{np.mean(per) * 100:>21.1f}%")
    json.dump({"global": rows, "per_category_mean_recall_at_1pct": float(np.mean(per)),
               "categories": cats},
              open(os.path.join(os.path.dirname(OUT), "mvtec_global_threshold.json"), "w",
                   encoding="utf-8"), indent=2, ensure_ascii=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
