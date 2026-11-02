"""What a factory actually sees: a fixed false-positive rate, and the recall it buys.

AUROC says how well the scores RANK. It does not say what happens when you set a threshold
once and leave it. This script holds the false-positive rate at 1% and 5% on the normal
training images (the only data a deployed system has seen) and reports the recall on the
defects, per category, plus how many faults a single global cutoff would find.

The second table is the one that matters for a real line: deploy one threshold for every
category, because a line runs more than one part, and see what it costs.

    python scripts/bench_threshold.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rtforge.vision.padim import _batched, _load_model   # noqa: E402


def scores(root: str, cat: str, size: int = 256, n_train: int = 20,
           n_channels: int = 100, seed: int = 0, eps: float = 0.01):
    net = _load_model()
    train_dir = os.path.join(root, cat, "train", "good")
    names = sorted(os.listdir(train_dir))[:n_train]
    tr = _batched(net, [os.path.join(train_dir, n) for n in names], size)
    n_img, c_dim, h, w = tr.shape
    tr = tr.reshape(n_img, c_dim, h * w)
    keep = np.random.default_rng(seed).choice(c_dim, n_channels, replace=False)
    tr = tr[:, keep, :]
    mu, inv = tr.mean(axis=0), 1.0 / (tr.var(axis=0) + eps)

    # normal training scores: the only normals a deployed system can calibrate on
    tr_s = [float(np.quantile(np.sqrt((((x - mu) ** 2) * inv).sum(axis=0)), 0.99)) for x in tr]

    test_dir = os.path.join(root, cat, "test")
    paths, labels = [], []
    for label in sorted(os.listdir(test_dir)):
        d = os.path.join(test_dir, label)
        if not os.path.isdir(d):
            continue
        for n in sorted(os.listdir(d)):
            if n.lower().endswith(".png"):
                paths.append(os.path.join(d, n))
                labels.append(0 if label == "good" else 1)
    feats = _batched(net, paths, size).reshape(len(paths), c_dim, h * w)[:, keep, :]
    ts = [float(np.quantile(np.sqrt((((x - mu) ** 2) * inv).sum(axis=0)), 0.99)) for x in feats]
    return np.array(tr_s), np.array(ts), np.array(labels)


def main(argv: list[str]) -> int:
    root = "/mnt/d/mvtec"
    cats = [d for d in sorted(os.listdir(root))
            if os.path.isdir(os.path.join(root, d, "train", "good"))]
    if "--categories" in argv:
        cats = argv[argv.index("--categories") + 1].split(",")
    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "results", "mvtec_threshold.json")
    prior = {}
    if os.path.exists(out):
        for r_ in json.load(open(out, encoding="utf-8")).get("per_category", []):
            prior[r_["category"]] = r_
    rows = []
    for c in cats:
        tr_s, ts, lab = scores(root, c)
        r = {"category": c}
        for fpr in (0.01, 0.05):
            thr = float(np.quantile(tr_s, 1.0 - fpr))
            hit = (ts[lab == 1] > thr).mean()
            r[f"thr_{int(fpr * 100)}pct"] = thr
            r[f"recall_at_{int(fpr * 100)}pct_fpr"] = float(hit)
        rows.append(r)
        prior[c] = r
        json.dump({"per_category": [prior[k] for k in sorted(prior)],
                   "note": "thresholds calibrated on normal TRAINING images; recall measured "
                           "on the test split"},
                  open(out, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
        print(f"  {c:<12} thr@1%={r['thr_1pct']:.4f} recall={r['recall_at_1pct_fpr'] * 100:5.1f}%   "
              f"thr@5%={r['thr_5pct']:.4f} recall={r['recall_at_5pct_fpr'] * 100:5.1f}%", flush=True)
        if "--no-global" in argv:
            continue

    if "--no-global" in argv:
        print(f"\n  written to {out}")
        return 0

    # one global threshold: calibrate on the pooled normal-training scores, then measure the
    # cost per category. This is what a line with several part types actually runs.
    pooled = []
    per = {}
    for c in cats:
        tr_s, ts, lab = scores(root, c)
        pooled.append(tr_s)
        per[c] = (ts, lab)
    pooled = np.concatenate(pooled)
    print("\n  one global threshold for all categories:")
    glob = {}
    for fpr in (0.01, 0.05):
        thr = float(np.quantile(pooled, 1.0 - fpr))
        worst_c, worst = None, 1.0
        for c in cats:
            ts, lab = per[c]
            rec = float((ts[lab == 1] > thr).mean())
            glob[f"{c}_{int(fpr * 100)}pct"] = rec
            if rec < worst:
                worst, worst_c = rec, c
        mean_rec = float(np.mean([glob[f"{c}_{int(fpr * 100)}pct"] for c in cats]))
        print(f"    @{int(fpr * 100)}% FPR  mean recall {mean_rec * 100:5.1f}%   "
              f"worst: {worst_c} {worst * 100:5.1f}%")

    json.dump({"per_category": [prior[k] for k in sorted(prior)], "global_recall": glob,
               "note": "thresholds calibrated on normal TRAINING images (the only normals a "
                       "deployed system has); recall measured on the test split"},
              open(out, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"\n  written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
