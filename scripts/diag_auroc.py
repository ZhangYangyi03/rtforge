import os, glob
import numpy as np
from rtforge.vision.mvtec import load_pgm_ish
from rtforge.vision.coreset import PatchEmbedder

def auroc(pos, neg):
    w = 0.0
    for p in pos:
        for n in neg:
            w += 1.0 if p > n else (0.5 if p == n else 0.0)
    return w / (len(pos) * len(neg))

STATS = [("mean", np.mean), ("med", np.median),
         ("q90", lambda z: np.quantile(z, 0.90)),
         ("q99", lambda z: np.quantile(z, 0.99)),
         ("max", np.max)]

for cat in ["bottle", "screw", "carpet"]:
    root = "/mnt/d/mvtec/" + cat
    tr = sorted(os.listdir(root + "/train/good"))[:20]
    normals = [load_pgm_ish(root + "/train/good/" + n, 128) for n in tr]
    e = PatchEmbedder()
    e.fit(normals)

    def emb(img):
        v = np.asarray(e.patch_vectors(img, stride=2))
        return (v - np.asarray(e.mean)) @ np.asarray(e.projection)

    st = np.stack([emb(i) for i in normals])
    mu = st.mean(0)
    var = st.var(0) + 1e-6

    def z_of(p):
        em = emb(load_pgm_ish(p, 128))
        n = min(len(em), len(mu))
        return ((em[:n] - mu[:n]) ** 2 / var[:n]).sum(1)

    G = glob.glob(root + "/test/good/*.png")
    B = [p for p in glob.glob(root + "/test/*/*.png") if "/good/" not in p]
    zg = [z_of(p) for p in G]
    zb = [z_of(p) for p in B]
    cells = []
    for nm, f in STATS:
        cells.append(nm + "=" + str(round(auroc([float(f(z)) for z in zb],
                                                [float(f(z)) for z in zg]) * 100)) + "%")
    print(cat.ljust(8), " ".join(cells))
