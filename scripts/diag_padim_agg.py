"""Which aggregation over the position grid separates screw's defects from its normals?

Run on screw, carpet and bottle so a statistic that helps one does not get adopted while
hurting another. Statistics: max (the paper's), and several quantiles.
"""
import os

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

from rtforge.vision.padim import _batched, _load_model, auroc

STATS = [("max", np.max), ("q999", lambda z: np.quantile(z, 0.999)),
         ("q99", lambda z: np.quantile(z, 0.99)), ("q95", lambda z: np.quantile(z, 0.95)),
         ("mean", np.mean)]


def run(cat, root="/mnt/d/mvtec", size=256, n_train=20, n_channels=100, seed=0, eps=0.01):
    net = _load_model()
    train_dir = os.path.join(root, cat, "train", "good")
    names = sorted(os.listdir(train_dir))[:n_train]
    tr = _batched(net, [os.path.join(train_dir, n) for n in names], size)
    n_img, c_dim, h, w = tr.shape
    tr = tr.reshape(n_img, c_dim, h * w)
    keep = np.random.default_rng(seed).choice(c_dim, n_channels, replace=False)
    tr = tr[:, keep, :]
    mu = tr.mean(axis=0)
    inv = 1.0 / (tr.var(axis=0) + eps)

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
    maps = [np.sqrt((((feats[i] - mu) ** 2) * inv).sum(axis=0)) for i in range(len(paths))]
    labels = np.asarray(labels)
    out = {}
    for name, f in STATS:
        s = [float(f(m)) for m in maps]
        out[name] = auroc([v for v, l in zip(s, labels) if l == 1],
                          [v for v, l in zip(s, labels) if l == 0])
    return out


if __name__ == "__main__":
    for cat in ["screw", "carpet", "bottle"]:
        r = run(cat)
        print(f"{cat:<8} " + "  ".join(f"{k}={v * 100:5.1f}%" for k, v in r.items()))
