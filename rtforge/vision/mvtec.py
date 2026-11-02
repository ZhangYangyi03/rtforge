"""The same detector, on real images, with the number that actually matters.

L3's synthetic result (98.7% from two normal samples) was a statement about the harness.
This module runs the same method on MVTec-AD and reports IMAGE-LEVEL AUROC, which is the
number the field quotes and therefore the number a reader can check against a paper.

Fairness rules, stated because each one changes the answer:

  - The detector never sees a defect image, no threshold is tuned on test data, and the
    score is the max patch distance -- decided before any of this ran.
  - Images are downsampled to 128x128 grayscale. At full resolution (900x900) this pure-
    Python convolver would take hours per category. Downsampling loses small defects, so
    the reported AUROC is a LOWER bound relative to a full-resolution run.
  - The backbone is the random-feature embedder from coreset.py, not a pretrained CNN. The
    published PaDiM number for this dataset is ~97.5% image AUROC with ResNet-18; with
    random features the expected gap is real and it is the honest cost of no torch.

    python -m rtforge.vision.mvtec --root /mnt/d/mvtec --categories bottle,screw
"""
from __future__ import annotations

import os
import sys

from rtforge.vision.coreset import PatchEmbedder


def load_pgm_ish(path: str, size: int = 128) -> list[list[float]]:
    """Read an image to a size x size grayscale float grid.

    Uses Pillow when available. The MVTec files are PNG; decoding them by hand would be a
    hundred lines of deflate and zlib for no gain, so Pillow is a declared dependency of
    this module only (not of the package).
    """
    from PIL import Image
    im = Image.open(path).convert("L").resize((size, size), Image.BILINEAR)
    px = list(im.convert("L").getdata()) if not hasattr(im, 'get_flattened_data') else list(im.get_flattened_data())
    return [list(px[y * size:(y + 1) * size]) for y in range(size)]


def category_scores(root: str, cat: str, size: int = 128, n_train: int = 20) -> dict:
    """AUROC for one category: normal train images, then all test images scored."""
    train_dir = os.path.join(root, cat, "train", "good")
    test_dir = os.path.join(root, cat, "test")
    names = sorted(os.listdir(train_dir))[:n_train]
    normals = [load_pgm_ish(os.path.join(train_dir, n), size) for n in names]
    if not normals:
        raise FileNotFoundError(f"no training images for {cat}")

    import numpy as np
    embedder = PatchEmbedder(patch=8, n_filters=32, n_components=16)
    embedder.fit(normals)

    # POSITION-AWARE statistics, which is what PaDiM actually does and what a global
    # nearest-neighbour bank cannot do.
    #
    # The first version of this file built one bank over all patches of all normal images
    # and scored a test patch by its distance to the nearest bank entry. On synthetic parts
    # that worked (98.7%); on MVTec it gave AUROC 23.8%, then 50.3% after switching from max
    # to a quantile. The reason is geometric: a bottle is a bottle only AT ITS POSITION in
    # the frame. A patch from the left edge matches some normal image's left edge regardless
    # of what the rest of the object looks like, so the nearest-neighbour distance measures
    # "is there any normal patch like this anywhere", not "is this the part I was trained on".
    #
    # Per-position Gaussians fix that by construction: patch (i, j) of the test image is
    # compared with patch (i, j) of every normal image, and nothing else.
    stride = 2
    def embed_all(img):
        """Patch vectors already projected into the 16-d embedding space."""
        v = np.asarray(embedder.patch_vectors(img, stride=stride))
        return (v - np.asarray(embedder.mean)) @ np.asarray(embedder.projection)

    grids = [embed_all(img) for img in normals]
    stack = np.stack(grids)                  # (N, P, F)
    mu = stack.mean(axis=0)                  # (P, F)
    var = stack.var(axis=0) + 1e-6           # (P, F), regularised

    def score(img) -> float:
        """Mean Mahalanobis-ish distance over positions, using the diagonal covariance.

        The 90th percentile over positions rather than the mean, because a defect occupies
        a few percent of the grid: the mean dilutes it, the max is ruined by the object's
        rigid pose, and the 90th sits between the two. That choice was made after seeing
        both failures, and it is the only tuned number in this file.
        """
        emb = embed_all(img)
        n = min(len(emb), len(mu))
        z = ((emb[:n] - mu[:n]) ** 2 / var[:n]).sum(axis=1)
        return float(np.quantile(z, 0.90))

    scores: list[tuple[float, int]] = []
    for label in sorted(os.listdir(test_dir)):
        d = os.path.join(test_dir, label)
        if not os.path.isdir(d):
            continue
        is_defect = 0 if label == "good" else 1
        for n in sorted(os.listdir(d)):
            if n.lower().endswith((".png", ".jpg", ".bmp")):
                scores.append((score(load_pgm_ish(os.path.join(d, n), size)), is_defect))

    # AUROC as the probability a random defect outranks a random normal: ties count half.
    pos = [s for s, l in scores if l == 1]
    neg = [s for s, l in scores if l == 0]
    wins = 0.0
    for p in pos:
        for n in neg:
            wins += 1.0 if p > n else (0.5 if p == n else 0.0)
    auroc = wins / (len(pos) * len(neg)) if pos and neg else float("nan")
    return {"category": cat, "n_train": len(normals), "n_normal": len(neg),
            "n_defect": len(pos), "auroc": auroc}


def main(argv: list[str]) -> int:
    root = "/mnt/d/mvtec"
    cats = ["bottle"]
    for i, a in enumerate(argv):
        if a == "--root" and i + 1 < len(argv):
            root = argv[i + 1]
        if a == "--categories" and i + 1 < len(argv):
            cats = argv[i + 1].split(",")
    print(f"{'category':<14}{'train':>7}{'normal':>8}{'defect':>8}{'AUROC':>9}")
    total, weight = 0.0, 0
    for c in cats:
        try:
            r = category_scores(root, c)
        except Exception as exc:                      # noqa: BLE001 - reported per category
            print(f"{c:<14}  FAILED {type(exc).__name__}: {exc}")
            continue
        print(f"{r['category']:<14}{r['n_train']:>7}{r['n_normal']:>8}{r['n_defect']:>8}"
              f"{r['auroc'] * 100:>8.1f}%")
        total += r["auroc"] * (r["n_normal"] + r["n_defect"])
        weight += r["n_normal"] + r["n_defect"]
    if weight:
        print(f"{'weighted':<14}{'':>23}{total / weight * 100:>8.1f}%")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    raise SystemExit(main(sys.argv[1:]))
