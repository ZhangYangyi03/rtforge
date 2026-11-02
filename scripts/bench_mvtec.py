"""All 15 MVTec categories, one protocol, one table.

Same settings as `rtforge.vision.padim`: ResNet-18 layer1/2/3, 100 channels, per-position
diagonal Gaussian, 99th-percentile image score, 20 normal training images.

Also reports the threshold that a fixed-FPR operating point implies. An AUROC alone does not
say what a factory would see; "99.8% AUROC" and "how many good parts get scrapped" are
different questions, and the second one is the one that gets a vision system switched off.

    python scripts/bench_mvtec.py [--categories a,b,c] [--root /mnt/d/mvtec]
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rtforge.vision.padim import category   # noqa: E402

ALL = ["bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather",
       "metal_nut", "pill", "screw", "tile", "toothbrush", "transistor", "wood", "zipper"]


def main(argv: list[str]) -> int:
    root = "/mnt/d/mvtec"
    cats = ALL
    for i, a in enumerate(argv):
        if a == "--root" and i + 1 < len(argv):
            root = argv[i + 1]
        if a == "--categories" and i + 1 < len(argv):
            cats = argv[i + 1].split(",")

    rows, skipped = [], []
    for c in cats:
        if not os.path.isdir(os.path.join(root, c, "train", "good")):
            skipped.append(c)
            continue
        try:
            r = category(root, c)
        except Exception as exc:                     # noqa: BLE001
            skipped.append(f"{c} ({type(exc).__name__}: {exc})")
            continue
        rows.append(r)
        print(f"  {c:<12}{r['auroc'] * 100:>7.1f}%   "
              f"(train {r['n_train']}, good {r['n_normal']}, defect {r['n_defect']})", flush=True)

    if not rows:
        print("no categories ran")
        return 1
    mean = sum(r["auroc"] for r in rows) / len(rows)
    weight = sum(r["n_normal"] + r["n_defect"] for r in rows)
    weighted = sum(r["auroc"] * (r["n_normal"] + r["n_defect"]) for r in rows) / weight
    print(f"\n  mean over {len(rows)} categories:     {mean * 100:5.1f}%")
    print(f"  weighted by test-set size:  {weighted * 100:5.1f}%")
    if skipped:
        print(f"  not run: {', '.join(skipped)}")

    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "results", "mvtec_all15.json")
    json.dump({"protocol": "ResNet-18 layer1/2/3, 100 channels, per-position diagonal Gaussian, "
                           "q99 image score, 20 normal training images",
               "rows": rows, "mean": mean, "weighted": weighted, "skipped": skipped,
               "reference": "PaDiM paper: ~97.5% mean image AUROC over all 15 categories"},
              open(out, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"  written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
