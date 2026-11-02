"""Second alignment experiment: finer search, and a wider one.

The first attempt used +-6 px at 128x128 (= +-42 px at native 900x900) and lifted bottle
from 90% to 93% while leaving screw at 24%. Either the search is too coarse or the objects
are not rigidly posed at all. This distinguishes the two.
"""
import glob
import os

import numpy as np
from rtforge.vision.coreset import PatchEmbedder
from rtforge.vision.mvtec import load_pgm_ish


def auroc(pos, neg):
    w = 0.0
    for p in pos:
        for n in neg:
            w += 1.0 if p > n else (0.5 if p == n else 0.0)
    return w / (len(pos) * len(neg))


def shift_cost(a, ref, dx, dy):
    """Mean squared difference over the valid (non-wrapped) overlap."""
    h, w = a.shape
    ax0, ax1 = max(0, dx), min(w, w + dx)
    ay0, ay1 = max(0, dy), min(h, h + dy)
    if ax1 - ax0 < 40 or ay1 - ay0 < 40:
        return float("inf")
    sub = a[ay0:ay1, ax0:ax1]
    r = ref[ay0 - dy:ay1 - dy, ax0 - dx:ax1 - dx]
    return float(((sub - r) ** 2).mean())


def best_shift(img, ref, rng):
    a = np.asarray(img, dtype=np.float64)
    best, bx, by = None, 0, 0
    for dy in range(-rng, rng + 1):
        for dx in range(-rng, rng + 1):
            c = shift_cost(a, ref, dx, dy)
            if best is None or c < best:
                best, bx, by = c, dx, dy
    return bx, by


def apply_shift(img, dx, dy):
    return np.roll(np.roll(np.asarray(img, dtype=np.float64), dy, axis=0), dx, axis=1).tolist()


def run(cat, rng, size=128):
    root = "/mnt/d/mvtec/" + cat
    tr = sorted(os.listdir(root + "/train/good"))[:20]
    normals = [load_pgm_ish(root + "/train/good/" + n, size) for n in tr]
    ref = np.asarray(normals, dtype=np.float64).mean(axis=0)
    aligned = []
    for img in normals:
        dx, dy = best_shift(img, ref, rng)
        aligned.append(apply_shift(img, dx, dy))
    e = PatchEmbedder()
    e.fit(aligned)

    def emb(img):
        v = np.asarray(e.patch_vectors(img, stride=2))
        return (v - np.asarray(e.mean)) @ np.asarray(e.projection)

    st = np.stack([emb(i) for i in aligned])
    mu = st.mean(0)
    var = st.var(0) + 1e-6

    def score(path):
        img = load_pgm_ish(path, size)
        dx, dy = best_shift(img, ref, rng)
        img = apply_shift(img, dx, dy)
        em = emb(img)
        n = min(len(em), len(mu))
        z = ((em[:n] - mu[:n]) ** 2 / var[:n]).sum(1)
        return float(np.mean(z))

    G = glob.glob(root + "/test/good/*.png")
    B = [p for p in glob.glob(root + "/test/*/*.png") if "/good/" not in p]
    return auroc([score(p) for p in B], [score(p) for p in G])


if __name__ == "__main__":
    for cat in ["screw", "carpet", "bottle"]:
        row = []
        for rng in (3, 12):
            row.append(f"rng={rng}: {run(cat, rng) * 100:5.1f}%")
        print(f"{cat:<10} " + "   ".join(row))
