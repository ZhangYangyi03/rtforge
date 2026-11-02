"""IEEE 1588 PTP: what the timestamp offset is, and what it costs the contour.

This layer exists because "TSN + 1588" is a phrase in every industrial pitch deck and a
number in almost none of them. The number that matters for a multi-axis machine is the
SYNC ERROR between two clocks, because that is what turns into contour error when two axes
are supposed to be somewhere at the same instant.

What is implemented: the four-timestamp offset/delay exchange from IEEE 1588 (delay request
- response), with the two error sources a real link has:
  - asymmetric path delay: the up and down legs differ by real amounts on a switching
    fabric, and no timestamp exchange can see the difference -- it lands directly in the
    offset. This is why hardware timestamping matters and why software timestamping in a
    VM cannot be trusted;
  - oscillator drift and jitter on each clock.

What is NOT implemented: a real PTP stack, servo, BMCA, or any traffic. The simulator takes
a stated path asymmetry and a stated oscillator spec and reports the offset it produces.
That is a model, and it is labelled as one -- but the conversion from sync error to contour
error is exact, and it is the number a machine builder has to budget.
"""
from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class Link:
    """What the network does to timestamps. Every field is something a vendor datasheet
    would quote or that a cable test would measure."""
    one_way_ns: float = 2_000.0          # 2 us: ~300 m of switch hop, plausible for a cell
    asymmetry_ns: float = 50.0           # up vs down; hardware TS gets this to tens of ns
    jitter_ns: float = 20.0              # per-packet, from timestamping granularity
    drift_ppm: float = 10.0              # oscillator, before the servo tracks it out


def offset_estimate(link: Link, rng: random.Random) -> dict:
    """One sync exchange: t1 master send, t2 slave receive, t3 slave send, t4 master receive.

    The standard result is offset = ((t2-t1) - (t4-t3)) / 2 and delay = ((t2-t1)+(t4-t3))/2.
    The asymmetry is invisible to that formula and appears whole in the offset -- which is
    the single most useful thing to know about PTP: it cannot measure its own asymmetry.
    """
    d_up = link.one_way_ns + link.asymmetry_ns / 2 + rng.gauss(0, link.jitter_ns)
    d_down = link.one_way_ns - link.asymmetry_ns / 2 + rng.gauss(0, link.jitter_ns)
    t1 = 0.0
    t2 = t1 + d_up
    t3 = t2 + rng.uniform(1e3, 5e3)          # slave processing, whatever
    t4 = t3 + d_down
    offset = ((t2 - t1) - (t4 - t3)) / 2.0
    delay = ((t2 - t1) + (t4 - t3)) / 2.0
    return {"offset_ns": offset, "delay_ns": delay}


def sync_error(link: Link, samples: int = 1000, seed: int = 11) -> dict:
    """Steady-state offset after the servo has tracked out the fixed part.

    The servo can remove a constant frequency error but not a constant PATH ASYMMETRY, so
    what remains is the asymmetry plus the per-packet jitter, not the raw drift.
    """
    rng = random.Random(seed)
    offsets = [offset_estimate(link, rng)["offset_ns"] for _ in range(samples)]
    offsets.sort()
    n = len(offsets)
    return {
        "mean_ns": sum(offsets) / n,
        "p99_ns": offsets[int(0.99 * (n - 1))],
        "worst_ns": offsets[-1],
        # A servo removes the mean; what is left is the spread about it.
        "residual_p99_ns": offsets[int(0.99 * (n - 1))] - sum(offsets) / n,
    }


def contour_error_um(radius_m: float, rpm: float, sync_ns: float) -> float:
    """Contour error a sync error makes on a coordinated circle.

    The axis is late by dt, so it is behind along the tangent by v*dt with v = r*omega.
    This is the number that decides whether a machine needs PTP at all: at 100 mm radius and
    60 rpm the tangent speed is 0.63 m/s, so 1 ns is 0.63 nm and 1 us is 0.63 um.
    """
    import math
    omega = rpm * 2.0 * math.pi / 60.0
    return radius_m * omega * sync_ns * 1e-9 * 1e6


def report(link_specs: list[tuple[str, Link]], radius_m: float = 0.1, rpm: float = 60.0) -> str:
    out = ["link                                   mean offset  residual p99   contour err",
           "                                                             (um)", ""]
    for name, link in link_specs:
        s = sync_error(link)
        ce = contour_error_um(radius_m, rpm, abs(s["residual_p99_ns"]))
        out.append(f"{name:<36}{s['mean_ns']:>10.0f} ns{s['residual_p99_ns']:>12.0f} ns{ce:>12.4f}")
    return "\n".join(out)


def main(argv: list[str]) -> int:
    specs = [
        ("software timestamping, 100 us path", Link(one_way_ns=100_000, asymmetry_ns=5_000, jitter_ns=2_000)),
        ("hardware TS, asymmetric switch", Link(one_way_ns=2_000, asymmetry_ns=50, jitter_ns=20)),
        ("hardware TS, symmetric path", Link(one_way_ns=2_000, asymmetry_ns=0, jitter_ns=20)),
    ]
    print(report(specs))
    return 0


if __name__ == "__main__":
    import sys
    raise SystemExit(main(sys.argv[1:]))
