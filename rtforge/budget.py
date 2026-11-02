"""The error budget. Every number this project claims comes out of here.

The point of this module is that a claim like "microsecond determinism" is not a
measurement, it is an adjective. What a machine can be held to is: a jitter figure, the
position error that jitter turns into at a stated speed, and the part of the budget it
spends. So the budget is computed, printed, and asserted against -- not narrated.

    python -m rtforge.budget            # the table, as this run computes it
    python -m rtforge.budget --json     # the same, machine-readable
"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class Kinematics:
    """The cell the budget is written for. Change these and every number moves."""
    end_effector_speed_m_s: float = 1.0
    circle_radius_m: float = 0.1
    rpm: float = 60.0
    conveyor_speed_m_s: float = 1.0
    control_period_us: float = 1000.0


@dataclass(frozen=True)
class Timings:
    """Measured or assumed per-stage latencies, in microseconds unless named."""
    encoder_bus_us: float = 40.0
    forward_dynamics_us: float = 15.0
    mpc_qp_us: float = 180.0
    feedforward_output_us: float = 20.0
    logging_us: float = 60.0

    exposure_frames: float = 1.5
    transport_preprocess_us: float = 2100.0
    inference_us_p99: float = 8400.0
    grasp_point_us: float = 300.0


def jitter_to_error(speed_m_s: float, jitter_us: float) -> float:
    """Millimetres of end-effector error for a timing error at a given speed.

    The whole reason "microseconds" matters is this line: at 1 m/s, 1 us is 1 um.
    """
    return speed_m_s * jitter_us * 1e-6 * 1e3


def sync_to_contour_error(radius_m: float, rpm: float, sync_us: float) -> float:
    """Micrometres of contour error from an inter-axis time-sync error.

    Two axes drawing a circle of radius r at omega: the tangential speed is r*omega, so a
    sync error dt displaces one axis along the path by r*omega*dt. This is why multi-axis
    coordination needs a shared clock and not just a fast one.
    """
    omega = rpm * 2.0 * 3.141592653589793 / 60.0
    return radius_m * omega * sync_us * 1e-6 * 1e6


def cycle_budget(t: Timings, period_us: float) -> dict:
    realtime = {
        "encoder_and_bus": t.encoder_bus_us,
        "forward_dynamics": t.forward_dynamics_us,
        "mpc_qp": t.mpc_qp_us,
        "feedforward_and_output": t.feedforward_output_us,
        "logging": t.logging_us,
    }
    used = sum(realtime.values())
    return {
        "parts": realtime,
        "used_us": used,
        "headroom_us": period_us - used,
        "used_fraction": used / period_us,
        # If the wake-up jitter target is a share of the headroom, this is the number the
        # kernel has to deliver. It is the only RT number that is actually load-bearing.
        "jitter_target_us": (period_us - used) * 0.1,
    }


def vision_latency(t: Timings, k: Kinematics, fps: float = 60.0) -> dict:
    frame_interval_ms = 1000.0 / fps
    exposure_ms = t.exposure_frames * frame_interval_ms
    total_us = (exposure_ms * 1000.0 + t.transport_preprocess_us
                + t.inference_us_p99 + t.grasp_point_us)
    return {
        "exposure_ms": exposure_ms,
        "transport_preprocess_ms": t.transport_preprocess_us / 1000.0,
        "inference_ms": t.inference_us_p99 / 1000.0,
        "grasp_point_ms": t.grasp_point_us / 1000.0,
        "total_ms": total_us / 1000.0,
        # The part a control loop cannot fix by being fast: the object has already moved.
        "object_travel_mm": k.conveyor_speed_m_s * total_us * 1e-6 * 1000.0,
        "travel_per_frame_mm": k.conveyor_speed_m_s * (1.0 / fps) * 1000.0,
    }


def table(k: Kinematics | None = None, t: Timings | None = None) -> dict:
    k = k or Kinematics()
    t = t or Timings()
    return {
        "kinematics": asdict(k),
        "timings": asdict(t),
        "cycle": cycle_budget(t, k.control_period_us),
        "vision": vision_latency(t, k),
        "jitter_to_error_mm": {
            f"{us}us": round(jitter_to_error(k.end_effector_speed_m_s, us), 4)
            for us in (1, 10, 50, 100, 1000)
        },
        "sync_to_contour_um": {
            f"{us}us": round(sync_to_contour_error(k.circle_radius_m, k.rpm, us), 2)
            for us in (1, 10, 100)
        },
    }


def render(b: dict) -> str:
    k, cyc, vis = b["kinematics"], b["cycle"], b["vision"]
    out = []
    out.append(f"cell: end-effector {k['end_effector_speed_m_s']} m/s, "
               f"conveyor {k['conveyor_speed_m_s']} m/s, control period "
               f"{k['control_period_us']:.0f} us")
    out.append("")
    out.append("1 kHz cycle budget")
    for name, us in cyc["parts"].items():
        out.append(f"    {name:<22}{us:>7.0f} us  {us / k['control_period_us'] * 100:5.1f}%")
    out.append(f"    {'used':<22}{cyc['used_us']:>7.0f} us  {cyc['used_fraction'] * 100:5.1f}%")
    out.append(f"    {'headroom':<22}{cyc['headroom_us']:>7.0f} us")
    out.append(f"    {'jitter target':<22}{cyc['jitter_target_us']:>7.1f} us  "
               f"(10% of headroom: this is the number the kernel must hit)")
    out.append("")
    out.append("timing error -> position error, at the end-effector speed above")
    for label, mm in b["jitter_to_error_mm"].items():
        out.append(f"    {label:<8}{mm:>10.4f} mm")
    out.append("")
    out.append(f"inter-axis sync -> contour error ({k['circle_radius_m']} m circle, "
               f"{k['rpm']:.0f} rpm)")
    for label, um in b["sync_to_contour_um"].items():
        out.append(f"    {label:<8}{um:>10.2f} um")
    out.append("")
    out.append("vision chain, and what the object does while it is being seen")
    out.append(f"    {'exposure':<22}{vis['exposure_ms']:>7.2f} ms")
    out.append(f"    {'transport+preprocess':<22}{vis['transport_preprocess_ms']:>7.2f} ms")
    out.append(f"    {'inference (p99)':<22}{vis['inference_ms']:>7.2f} ms")
    out.append(f"    {'grasp point':<22}{vis['grasp_point_ms']:>7.2f} ms")
    out.append(f"    {'total':<22}{vis['total_ms']:>7.2f} ms")
    out.append(f"    object moved {vis['object_travel_mm']:.1f} mm during that time "
               f"({vis['travel_per_frame_mm']:.1f} mm per frame)")
    return "\n".join(out)


def main(argv: list[str]) -> int:
    b = table()
    if "--json" in argv:
        print(json.dumps(b, indent=2))
    else:
        print(render(b))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
