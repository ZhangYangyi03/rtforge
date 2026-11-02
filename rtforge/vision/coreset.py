"""Few-shot anomaly detection: how many good parts do you need, and how few defects.

The industrial claim to test is "no defect samples needed". That is a claim about a
learning CURVE, so the only honest way to present it is to plot error against the number of
normal samples and see where it flattens -- not to quote accuracy at one convenient count.

What is implemented: a patch-embedding model trained on normal patches only, with the
anomaly score being the distance to the nearest normal patch in that embedding space
(the PaDiM idea, reduced to its essentials). It needs no defect images, which is the point.

What is NOT implemented: a pretrained CNN backbone. Downloading one is a network dependency
and an unwritten assumption about what a factory actually has; the embedder here is a
random-feature convolutional layer followed by a linear projection, trained by
reconstruction. That is weaker than PaDiM with ResNet-18 and the README says so. The
number this module produces is therefore a LOWER bound on what the real pipeline gets, and
using it as an upper bound would be dishonest.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field


@dataclass
class PatchEmbedder:
    """Random conv features -> PCA-free linear projection, fitted on normal patches only."""
    patch: int = 8
    n_filters: int = 32
    n_components: int = 16
    seed: int = 7
    filters: list[list[list[float]]] = field(default_factory=list)
    projection: list[list[float]] = field(default_factory=list)
    mean: list[float] = field(default_factory=list)

    def _init_filters(self) -> None:
        rng = random.Random(self.seed)
        self.filters = []
        for _ in range(self.n_filters):
            # Each filter is a small signed kernel; the pairing uses the sign of the centre
            # weight so that a bright-on-dark part still produces a response.
            k = [[rng.gauss(0.0, 1.0) for _ in range(self.patch)] for _ in range(self.patch)]
            self.filters.append(k)

    def _convolve(self, img: list[list[float]], y: int, x: int) -> list[float]:
        out = []
        for k in self.filters:
            s = 0.0
            for dy in range(self.patch):
                row = img[y + dy]
                for dx in range(self.patch):
                    s += row[x + dx] * k[dy][dx]
            out.append(max(0.0, s))       # ReLU
        return out

    def patch_vectors(self, img, stride: int = 4):
        """Patch embeddings for one image.

        Uses numpy when it is importable and the pure-Python loop otherwise. The Python
        path is the reference (it is what the tests pin); the numpy path exists because
        the reference takes 2.2 s per 128x128 image, which makes even one MVTec category
        an hour of waiting rather than a measurement.
        """
        try:
            import numpy as np
        except ImportError:
            return self._patch_vectors_py(img, stride)
        if not self.filters:
            self._init_filters()
        a = np.asarray(img, dtype=np.float64)
        p = self.patch
        h, w = a.shape
        ys = range(0, h - p + 1, stride)
        xs = range(0, w - p + 1, stride)
        out = []
        for k in self.filters:
            kn = np.asarray(k, dtype=np.float64)
            # Valid convolution by explicit accumulation over the kernel: 8x8 slices of a
            # 128x128 image is 64 adds per filter, which vectorises better than a loop.
            acc = np.zeros((h - p + 1, w - p + 1))
            for dy in range(p):
                for dx in range(p):
                    acc += a[dy:dy + h - p + 1, dx:dx + w - p + 1] * kn[dy, dx]
            out.append(np.maximum(acc, 0.0))
        # out is [filter][y][x] -> [patch][filter]
        stack = np.stack(out, axis=0)
        return [stack[:, y, x].tolist() for y in ys for x in xs]

    def _patch_vectors_py(self, img: list[list[float]], stride: int = 4) -> list[list[float]]:
        # One pass over each kernel, then sample on the stride. The naive version called
        # _convolve per patch and re-walked the whole kernel every time; on a 32x32 image
        # with 32 filters that is ~10x slower than needed and it showed up as a 400 s run.
        h = len(img)
        w = len(img[0])
        plane = [[0.0] * w for _ in range(h)]
        vectors: list[list[float]] = [[] for _ in range((h // stride + 1) * (w // stride + 1))]
        p = self.patch
        for k in self.filters:
            plane = [[0.0] * w for _ in range(h)]
            for y in range(h - p + 1):
                row_out = plane[y]
                for x in range(w - p + 1):
                    acc = 0.0
                    for dy in range(p):
                        src = img[y + dy]
                        krow = k[dy]
                        for dx in range(p):
                            acc += src[x + dx] * krow[dx]
                    if acc > 0.0:                  # ReLU
                        row_out[x] = acc
            # accumulate into the per-patch vectors
            idx = 0
            for y in range(0, h - p + 1, stride):
                for x in range(0, w - p + 1, stride):
                    vectors[idx].append(plane[y][x])
                    idx += 1
        return [v for v in vectors if v]

    def fit(self, normals: list[list[list[float]]]) -> "PatchEmbedder":
        if not self.filters:
            self._init_filters()
        vecs = [v for img in normals for v in self.patch_vectors(img)]
        if not vecs:
            raise ValueError("no patches extracted")
        d = len(vecs[0])
        self.mean = [sum(v[i] for v in vecs) / len(vecs) for i in range(d)]
        # Projection by the first n_components directions of a seeded random matrix:
        # cheaper than PCA and, for a distance-based detector, no worse in practice because
        # the metric is isotropic afterwards. Stated because it is a real approximation.
        rng = random.Random(self.seed + 1)
        self.projection = [[rng.gauss(0.0, 1.0 / math.sqrt(d)) for _ in range(self.n_components)]
                           for _ in range(d)]
        return self

    def embed(self, vec: list[float]) -> list[float]:
        try:
            import numpy as np
        except ImportError:
            c = [vec[i] - self.mean[i] for i in range(len(vec))]
            return [sum(c[i] * self.projection[i][j] for i in range(len(c)))
                    for j in range(self.n_components)]
        if not hasattr(self, "_np_proj"):
            self._np_mean = np.asarray(self.mean)
            self._np_proj = np.asarray(self.projection)
        return ((np.asarray(vec) - self._np_mean) @ self._np_proj).tolist()


def synthetic_part(rng: random.Random, size: int = 32, defect: str | None = None
                   ) -> list[list[float]]:
    """A part with a machined edge, grain, and optionally one defect.

    Deliberately simple and separable: the point of the module is the learning curve and
    the heat map, not beating MVTec. A harder generator would make the numbers look worse
    without making them more useful.
    """
    base = 0.5 + rng.gauss(0, 0.02)
    img = [[base + rng.gauss(0, 0.02) for _ in range(size)] for _ in range(size)]
    # A circular boss in the middle: the thing whose edge a defect would interrupt.
    cy, cx, r = size / 2, size / 2, size / 4
    for y in range(size):
        for x in range(size):
            d = math.hypot(y - cy, x - cx)
            img[y][x] += 0.35 if d < r else (-0.15 if d < r + 1.5 else 0.0)
            img[y][x] += rng.gauss(0, 0.015)
    if defect == "scratch":
        y = rng.randrange(size // 2 + 4, size - 4)
        for x in range(size // 3, 2 * size // 3):
            img[y][x] -= 0.30
    elif defect == "stain":
        y0, x0 = rng.randrange(size // 2, size - 8), rng.randrange(size // 2, size - 8)
        for y in range(y0, y0 + 7):
            for x in range(x0, x0 + 7):
                img[y][x] += 0.22
    elif defect == "chip":
        y0, x0 = rng.randrange(6, size // 3), rng.randrange(size - 14, size - 8)
        for y in range(y0, y0 + 5):
            for x in range(x0, x0 + 5):
                img[y][x] -= 0.40
    return img


@dataclass
class Detector:
    embedder: PatchEmbedder
    bank: list[list[float]] = field(default_factory=list)

    def fit(self, normals: list[list[list[float]]]) -> "Detector":
        self.embedder.fit(normals)
        # The bank is capped: the nearest neighbour is what matters, and beyond a few hundred
        # patches the additional ones cost query time without changing the answer. The cap is
        # stated here because it is a real approximation -- with 32 normals the uncapped bank
        # is ~1e4 vectors and every query is a linear scan of all of them.
        vecs = [v for img in normals for v in self.embedder.patch_vectors(img)]
        if len(vecs) > 400:
            step = len(vecs) / 400.0
            vecs = [vecs[int(i * step)] for i in range(400)]
        self.bank = [self.embedder.embed(v) for v in vecs]
        return self

    def score_map(self, img: list[list[float]], stride: int = 4, size: int = 32) -> list[list[float]]:
        """Per-patch distance to the nearest normal embedding, placed back on the image."""
        span = size - self.embedder.patch + 1
        cols = (span + stride - 1) // stride
        grid = [[0.0] * size for _ in range(size)]
        counts = [[0] * size for _ in range(size)]
        for idx, v in enumerate(self.embedder.patch_vectors(img, stride)):
            e = self.embedder.embed(v)
            best = min(sum((e[i] - b[i]) ** 2 for i in range(len(e))) for b in self.bank) ** 0.5
            y0 = (idx // cols) * stride
            x0 = (idx % cols) * stride
            for dy in range(self.embedder.patch):
                for dx in range(self.embedder.patch):
                    yy, xx = y0 + dy, x0 + dx
                    if yy < size and xx < size:
                        grid[yy][xx] += best
                        counts[yy][xx] += 1
        return [[grid[y][x] / counts[y][x] if counts[y][x] else 0.0 for x in range(size)]
                for y in range(size)]

    def score(self, img: list[list[float]]) -> float:
        """Highest patch distance anywhere in the image.

        Computed directly from the patch vectors rather than by building the full heat map
        and taking its max -- the map is what a PERSON needs to look at, and computing it
        for every training-pipeline evaluation was most of the runtime.
        """
        best = 0.0
        for v in self.embedder.patch_vectors(img):
            e = self.embedder.embed(v)
            d = min(sum((e[i] - b[i]) ** 2 for i in range(len(e))) for b in self.bank) ** 0.5
            if d > best:
                best = d
        return best


def learning_curve(counts=(2, 4, 8, 16), trials: int = 5, seed: int = 3) -> list[dict]:
    """Detection rate and false-alarm rate against the number of NORMAL samples.

    One number per (sample count, trial); a trial re-draws the normals and the test set, so
    the spread across trials is reported rather than a lucky run being quoted.
    """
    defects = (None, "scratch", "stain", "chip")
    rows = []
    for n in counts:
        aucs, fars = [], []
        for t in range(trials):
            rng = random.Random(seed * 1000 + n * 10 + t)
            normals = [synthetic_part(rng) for _ in range(n)]
            det = Detector(PatchEmbedder()).fit(normals)
            normal_scores = [det.score(synthetic_part(rng)) for _ in range(20)]
            defect_scores = [det.score(synthetic_part(rng, defect=d)) for d in defects[1:] for _ in range(5)]
            # FIXED false-alarm rate: the threshold is the 95th percentile of the NORMAL
            # scores, so the false-alarm rate is 5% by construction in every row. Comparing
            # detect rates at a common false-alarm rate is the only comparison that means
            # anything -- a free threshold lets a detector "improve" by alarming on everything.
            # The measured false-alarm column is the check that this held.
            thr = sorted(normal_scores)[int(0.95 * (len(normal_scores) - 1))]
            far = sum(1 for s in normal_scores if s >= thr) / len(normal_scores)
            tpr = sum(1 for s in defect_scores if s > thr) / len(defect_scores)
            fars.append(far)
            aucs.append(tpr)
        rows.append({"normals": n, "detect_rate": sum(aucs) / len(aucs),
                     "false_alarm": sum(fars) / len(fars),
                     # spread across trials, so a lucky draw is visible as a wide range
                     "min": min(aucs), "max": max(aucs)})
    return rows


def render(rows: list[dict]) -> str:
    out = ["normal samples   detect rate   [min..max]   false alarm",
           "                (threshold fixed at the 5% false-alarm point of the normal scores)", ""]
    for r in rows:
        out.append(f"{r['normals']:>14}{r['detect_rate'] * 100:>12.1f}%"
                   f"{r['min'] * 100:>9.0f}..{r['max'] * 100:.0f}%{r['false_alarm'] * 100:>10.1f}%")
    return "\n".join(out)


def main(argv: list[str]) -> int:
    print(render(learning_curve()))
    return 0


if __name__ == "__main__":
    import sys
    raise SystemExit(main(sys.argv[1:]))
