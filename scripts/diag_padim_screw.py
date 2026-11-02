"""Why does screw stay near chance when bottle and carpet do not?

Two candidate causes, tested separately:
  1. the defects are small, so downsampling to 256 destroys them -- tested by raising the
     input resolution;
  2. the score aggregates over a whole 56x56 grid. If the max is dominated by where the
     object sits rather than what it looks like, a quantile will do better.
"""
from rtforge.vision.padim import category

for size in (256, 384):
    r = category("/mnt/d/mvtec", "screw", size=size, n_train=20)
    print(f"screw size={size:4d}  AUROC {r['auroc'] * 100:5.1f}%")
