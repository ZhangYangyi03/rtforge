"""The whole stack, one command, with every number traceable to what produced it.

This is the file a reader should run first and the file that decides whether the project is
honest. It executes the layers that CAN be measured on this host, reads the results of the
layers that were measured earlier into results/, and prints -- for each claim in the README
-- whether the number behind it was measured here, modelled, or is not there at all.

The three-way tag is the point. A project like this fails not by having wrong numbers but by
having three kinds of number that look identical in a table:

  measured   produced by running something on this machine in this session
  modelled   produced by a stated model with stated inputs (the PTP link, the vision latency)
  absent     no number, and the README says why

    python -m rtforge.eval.harness            # run and print
    python -m rtforge.eval.harness --json     # machine-readable
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from dataclasses import asdict

from rtforge import budget as budget_mod

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS = os.path.join(HERE, "results")


def _read_latency() -> dict | None:
    """Worst-thread percentiles across every cyclictest run in results/."""
    try:
        sys.path.insert(0, HERE)
        from rtforge.latency.histogram import summarize
        import glob
        paths = sorted(glob.glob(os.path.join(RESULTS, "cyclictest_*.hist")))
        if not paths:
            return None
        s = summarize(paths)
        if not s["runs"]:
            return None
        worst = max(s["runs"], key=lambda r: r["max_seen"])
        return {"runs": s["runs"], "worst": worst}
    except Exception as exc:                      # noqa: BLE001 - reported, not swallowed
        return {"error": f"{type(exc).__name__}: {exc}"}


def _read_build() -> dict:
    """Does the C++ tree compile and does its self-test pass, right now."""
    build = os.path.join(HERE, "cpp", "build")
    info = {"built": False, "selftest": None}
    exe = os.path.join(build, "rtforge_selftest.exe" if os.name == "nt" else "rtforge_selftest")
    if not os.path.exists(exe):
        return info
    info["built"] = True
    try:
        out = subprocess.run([exe], capture_output=True, text=True, timeout=60)
        text = out.stdout + out.stderr
        m = re.search(r"(\d+) failure", text)
        info["selftest"] = {
            "returncode": out.returncode,
            "failures": int(m.group(1)) if m else None,
            "checks": text.count("PASS") + text.count("FAIL"),
        }
    except Exception as exc:                      # noqa: BLE001
        info["selftest"] = {"error": f"{type(exc).__name__}: {exc}"}
    return info


def _read_tracking() -> dict | None:
    """Feedforward sweep, if it was run."""
    path = os.path.join(RESULTS, "tracking.json")
    if not os.path.exists(path):
        return None
    return json.load(open(path, encoding="utf-8"))


def _read_vision() -> dict | None:
    path = os.path.join(RESULTS, "vision_curve.json")
    if not os.path.exists(path):
        return None
    return json.load(open(path, encoding="utf-8"))


def build_report() -> dict:
    b = budget_mod.table()
    return {
        "budget": b,
        "latency": _read_latency(),
        "build": _read_build(),
        "tracking": _read_tracking(),
        "vision": _read_vision(),
        "ptp": _ptp_rows(),
    }


def _ptp_rows() -> list[dict]:
    from rtforge.timebase.ptp import Link, sync_error, contour_error_um
    specs = [
        ("software timestamping, 100 us path", Link(one_way_ns=100_000, asymmetry_ns=5_000, jitter_ns=2_000)),
        ("hardware TS, asymmetric switch", Link(one_way_ns=2_000, asymmetry_ns=50, jitter_ns=20)),
        ("hardware TS, symmetric path", Link(one_way_ns=2_000, asymmetry_ns=0, jitter_ns=20)),
    ]
    rows = []
    for name, link in specs:
        s = sync_error(link)
        rows.append({"link": name, "mean_ns": s["mean_ns"],
                     "residual_p99_ns": s["residual_p99_ns"],
                     "contour_um": contour_error_um(0.1, 60.0, abs(s["residual_p99_ns"]))})
    return rows


def render(r: dict) -> str:
    b = r["budget"]
    cyc = b["cycle"]
    out = []
    out.append("rtforge -- every claim, its number, and where the number came from")
    out.append("")
    out.append("L1 real-time kernel")
    lat = r["latency"]
    if not lat:
        out.append("    absent    no cyclictest runs in results/ (run scripts/collect_latency.sh)")
    elif "error" in lat:
        out.append(f"    broken    {lat['error']}")
    else:
        w = lat["worst"]
        out.append(f"    measured  {len(lat['runs'])} configurations, worst thread of the worst run:")
        out.append(f"              p50 {w['p50']:.0f} us   p99 {w['p99']:.0f} us   "
                   f"p99.9 {w['p999']:.0f} us   max {w['max_seen']:.0f} us   ({w['config']})")
        target = cyc["jitter_target_us"]
        verdict = "MEETS" if w["p999"] <= target else "MISSES"
        out.append(f"    budget    jitter target {target:.1f} us from the {b['kinematics']['control_period_us']:.0f} us "
                   f"cycle -> measured p99.9 {verdict} it")
        out.append(f"              (this host is WSL2, not a PREEMPT_RT kernel; see docs/GAPS.md)")
    out.append("")
    out.append("L2 deterministic network (IEEE 1588 model)")
    for row in r["ptp"]:
        out.append(f"    modelled  {row['link']:<38} residual p99 {row['residual_p99_ns']:>6.0f} ns"
                   f" -> {row['contour_um']:.4f} um contour")
    out.append("")
    out.append("L3 vision: few-shot anomaly detection (synthetic parts)")
    v = r["vision"]
    if not v:
        out.append("    absent    results/vision_curve.json not found")
    else:
        for row in v["rows"]:
            out.append(f"    measured  {row['normals']:>2} normal samples -> detect "
                       f"{row['detect_rate'] * 100:.1f}% at {row['false_alarm'] * 100:.0f}% false alarm")
        if v.get("pending"):
            out.append(f"    pending   {v['pending']}")
        out.append(f"    caveat    {v['caveat']}")
    out.append("")
    out.append("L4 motion control: hand-written dynamics + feedforward")
    t = r["tracking"]
    if t:
        if "scenario" in t["rows"][0]:
            out.append(f"    measured  {'scenario':<24}{'ff=1.0':>10}{'ff=0.0':>10}{'ratio':>8}")
            for row in t["rows"]:
                out.append(f"              {row['scenario']:<24}{row['feedforward_1.0_mm']:>10.3f}"
                           f"{row['feedforward_0.0_mm']:>10.3f}{row['ratio']:>7.1f}x")
            out.append(f"    reading   {t['headline']}")
        else:
            for row in t["rows"]:
                out.append(f"    measured  feedforward {row['feedforward']:.1f} -> worst tracking error "
                           f"{row['worst_error_mm']:.3f} mm")
        out.append(f"    caveat    {t['caveat']}")
    else:
        out.append("    absent    results/tracking.json not found")
    bld = r["build"]
    if bld["selftest"]:
        s = bld["selftest"]
        out.append(f"    measured  C++ self-test: {s.get('checks')} checks, {s.get('failures')} failures")
    out.append("")
    out.append("L5 timing -> error conversion (arithmetic, exact)")
    for label, mm in b["jitter_to_error_mm"].items():
        out.append(f"    derived   {label:<8} of jitter at {b['kinematics']['end_effector_speed_m_s']} m/s "
                   f"= {mm:.4f} mm at the tool")
    vis = b["vision"]
    out.append(f"    derived   the vision chain takes {vis['total_ms']:.2f} ms, during which the object "
               f"moves {vis['object_travel_mm']:.1f} mm")
    return "\n".join(out)


def main(argv: list[str]) -> int:
    r = build_report()
    if "--json" in argv:
        print(json.dumps(r, indent=2, default=str))
    else:
        print(render(r))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
