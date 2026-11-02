import sys
from rtforge.eval.harness import build_report, render
r = build_report()
txt = render(r)
open('results/report.txt','w',encoding='utf-8').write(txt)
print(txt)
