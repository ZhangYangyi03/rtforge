# What this is not

Written first, because the rest of the repository only means something if the boundary is
stated before the numbers.

## Hardware

There is no robot. Every motion number comes from a simulated two-link arm whose plant and
controller share one model. That measures the controller and says nothing about the gap
between a model and a real arm -- friction that depends on direction and load, backlash in
a harmonic drive, structural flex at the tool, current-loop bandwidth, encoder quantisation.
A real commissioning run is the only thing that closes it, and it is not here.

There is no camera and no part. The vision layer trains on synthetic 32x32 images generated
by `rtforge/vision/coreset.py`. That generator is separable by construction, so the 98-100%
detection rates in `results/vision_curve.json` are a statement about the harness, not about
a factory. On MVTec-AD the same method with a trained backbone lands around 95-97% image
AUROC; this code with a random-feature embedder would do worse, and the honest expectation
is "it works, less well than the version with a pretrained CNN".

## The real-time claim

`cyclictest` was run on WSL2. WSL2 is a virtual machine: its vCPUs are descheduled by the
Windows host, so the measured p99.9 of ~146 ms is not a statement about Linux's real-time
behaviour, it is a statement about this host. The number is in
`results/cyclictest_*.hist` and the harness reports it as MISSING the budget, which is the
correct verdict for this machine.

What would make the L1 claim real, and what is missing here:

- a PREEMPT_RT kernel built for the target board (not patched on this laptop);
- `isolcpus`/`nohz_full` set at boot for the control cores;
- `cyclictest` on that board, idle AND under load, for an hour, reporting p99.9 and max;
- `timer_migration` and the CPU governor pinned, `mlockall` held for the whole run.

`scripts/collect_latency.sh` runs the four configurations; on this host three of them are
indistinguishable because the bottleneck is outside the guest kernel.

## The network claim

There is no TSN switch and no PTP traffic. `rtforge/timebase/ptp.py` is the IEEE 1588
offset formula with a stated path asymmetry and stated oscillator jitter. It produces the
right ORDER of magnitude and the right structure (asymmetry is invisible to the exchange and
lands whole in the offset) but the 22 nm contour figure is arithmetic applied to an assumed
link, not a measurement. A real number needs two boards, a hardware-timestamping NIC and a
`ptp4l` run.

## What the control layer does not model

- contact and friction cone;
- backlash and harmonic-drive compliance;
- joint limits and singularity handling in the QP (the constraints are there, the
  trajectory generator that would avoid them is not);
- the QP solver is dense and O(n^3) in the horizon. At 12 joints and a 20-step horizon it
  does not fit a 1 kHz cycle on a microcontroller. It fits on the Linux side of a
  EtherCAT master, which is where this design puts it.

## What was measured and what was modelled

`python -m rtforge.eval.harness` tags every number: `measured` (run on this host, this
session), `modelled` (a stated model with stated inputs), `derived` (arithmetic on the
above). If a row is absent, it is absent; nothing here is estimated to fill a table.
