"""Is the MVTec failure caused by pose misalignment?

If it is, then aligning each test image to the training mean (a small translation search)
should lift AUROC toward the published numbers. If it is not, alignment will not help and
the cause is the embedding, not the geometry. This is the experiment that decides which
part of the pipeline to rewrite.
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


def best_shift(img, ref, rng=6):
    """Translation in [-rng, rng] that minimises the squared difference to ref."""
    a = np.asarray(img)
    best, bx, by = None, 0, 0
    for dy in range(-rng, rng + 1):
        for dx in range(-rng, rng + 1):
            shifted = np.roll(np.roll(a, dy, axis=0), dx, axis=1)
            # ignore the wrapped border so a shift is not rewarded for it
            core = shifted[rng:-rng, rng:-rng]
            cost = float(((core - ref[rng:-rng, rng:-rng]) ** 2).mean())
            if best is None or cost < best:
                best, bx, by = cost, dx, dy
    return bx, by, best


def run(cat, size=128, rng=6):
    root = "/mnt/d/mvtec/" + cat
    tr = sorted(os.listdir(root + "/train/good"))[:20]
    normals = [load_pgm_ish(root + "/train/good/" + n, size) for n in tr]
    ref = np.asarray(normals).mean(axis=0)

    aligned = []
    for img in normals:
        dx, dy, _ = best_shift(img, ref, rng)
        aligned.append(np.roll(np.roll(np.asarray(img), dy, axis=0), dx, axis=1).tolist())

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
        dx, dy, _ = best_shift(img, ref, rng)
        img = np.roll(np.roll(np.asarray(img), dy, axis=0), dx, axis=1).tolist()
        em = emb(img)
        n = min(len(em), len(mu))
        z = ((em[:n] - mu[:n]) ** 2 / var[:n]).sum(1)
        return float(np.mean(z))

    G = glob.glob(root + "/test/good/*.png")
    B = [p for p in glob.glob(root + "/test/*/*.png") if "/good/" not in p]
    zg = [score(p) for p in G]
    zb = [score(p) for p in B]
    return auroc(zb, zg), len(zg), len(zb)


if __name__ == "__main__":
    for cat in ["bottle", "screw", "carpet"]:
        a, ng, nb = run(cat)
        print(f"{cat:<10} aligned AUROC {a * 100:5.1f}%   ({ng} normal, {nb} defect)")
