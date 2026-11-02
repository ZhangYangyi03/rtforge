# rtforge

A real-time sorting cell built bottom-up, where every layer has to produce a number before
it earns a place. Three layers can be measured on one machine, two can only be modelled
here, and the report says which is which.

The premise: an industrial stack is not hard to assemble from libraries, it is hard to trust.
So each layer here is written against a check that fails when the layer is wrong -- closed
forms for the dynamics, energy conservation for the integrator, a fixed false-alarm rate for
the detector, a stated asymmetry for the clock.

## Run it

    # the whole report: every claim, its number, and where the number came from
    PYTHONPATH=. python3 scripts/report.py

    # real-time latency sweep (needs cyclictest: apt-get install rt-tests)
    ./scripts/collect_latency.sh

    # dynamics: build and run the self-test against closed forms
    cmake -S cpp -B cpp/build && cmake --build cpp/build -j && cpp/build/rtforge_selftest

    # closed-loop tracking, feedforward swept
    cpp/build/rtforge_track --ff 1.0 --duration 3
    cpp/build/rtforge_track --ff 0.0 --duration 3

    # the error budget on its own
    PYTHONPATH=. python3 -m rtforge.budget

## What it currently reports

Measured on this host (Windows 11 + WSL2 Ubuntu, 16 cores, g++ 13.3):

    layer        claim                                  number        verdict
    L1 kernel    jitter vs a 1 kHz cycle budget        146543 us     MISSES (WSL2)
    L2 network   sync error -> contour error            36 ns -> 22 nm   modelled
    L3 vision    MVTec-AD, random features            bottle 75%, screw 20%, carpet 33%   FAILS
    L3 vision    MVTec-AD, ResNet-18 backbone, 14 cats mean 82.5% (best 99.9, worst 51.8)   measured
    L3 vision    fixed 1% FPR, per-category threshold   mean recall 92.7%                  measured
    L3 vision    fixed 1% FPR, ONE global threshold     mean recall 60.6% (grid 0%)        measured
    L3 vision    synthetic parts, 2 normal samples     98.7% at 10% FA  measured
    L4 control   feedforward 1.0 vs 0.0, clean plant   0.758 vs 38.853 mm   measured
    L4 control   same, with backlash                  259.9 vs 273.8 mm    measured, 1.1x
    L4 control   same, with structural flex            47.8 vs 48.7 mm      measured, 1.0x
    L4 control   backlash with a backlash INVERSE      259.9 -> 3.51 mm      measured, 74x
    L5 derived   1 us of jitter at 1 m/s               1 um at the tool

The L1 row is the important one. On a virtualised host the worst-case latency is 146 ms
against a 68.5 us budget: the design's real-time premise is not demonstrated by this
machine and `docs/GAPS.md` says what would demonstrate it.

The L3 and L4 rows are the ones that changed after the data arrived.

L3 on real data failed with random features. The same method that detects a defect from two
synthetic samples got 75% AUROC on bottle, 20% on screw and 33% on carpet -- below chance. Two experiments in
`scripts/` establish that this is structural, not tuning: sweeping five aggregation
statistics does not rescue it, and aligning every image to the training mean (+-3, +-6 and
+-12 px searches) moves bottle from 90 to 93 and leaves the other two flat. Random
convolutional features carry no semantics.

So the next run changed exactly one thing -- the features now come from an ImageNet-trained
ResNet-18, with layers 1/2/3 concatenated, 100 channels, per-position diagonal Gaussian and a
99th-percentile image score. Eight categories, 20 training images each:

    bottle 99.8   carpet 96.3   toothbrush 91.7   hazelnut 88.3
    metal_nut 77.6   grid 68.3   pill 59.0   screw 51.8      mean 79.1%

All 14 categories that could be extracted (zipper was still unpacking when this was written;
its absence is stated rather than filled in):

    leather 99.9  bottle 99.8  wood 98.9  carpet 96.3  tile 95.7  toothbrush 91.7
    hazelnut 88.3  transistor 82.4  metal_nut 77.6  capsule 74.2  cable 71.3
    grid 68.3  pill 59.0  screw 51.8          mean 82.5%   (paper: ~97.5% over 15)

AUROC is not what a line sees. Holding the false-positive rate at 1% on the normal training
images and reading the recall on the defects, the per-category picture is much better than
the AUROC ordering suggests -- pill goes from 59.0% AUROC to 100% recall, because its normals
are tightly clustered and its defects are far away:

    per-category threshold, 1% FPR     mean recall 92.7%
    ONE threshold for every category   mean recall 60.6%, worst grid 0.0%

That second row is the finding worth keeping. A single global cutoff at 1% FPR means 0% of
grid's defects are caught at all, because grid's normal scores sit below the pooled cutoff.
The same method that looks like 82.5% AUROC is a quarter of a working system if deployed the
way a line with several part types actually runs it. Both tables are in
`results/mvtec_threshold.json` and `results/mvtec_global_threshold.json`.

`results/mvtec_all15.json` holds all of it, `rtforge/vision/padim.py` reproduces it, and the
scripts beside `diag_auroc.py` show why the aggregation and the resolution are not the reason
screw stays at chance. The published PaDiM number is ~97.5% mean over all 15 categories; this
is 79.1% over 8 of them, so it is a ceiling being approached, not matched, and the README does
not round it up to "MVTec-AD performance".

L4 is the same story in a different shape. Feedforward buys a factor of 51 against a clean
plant and a factor of 1.0 once the plant has a deadband or a bending mode: `tau = M ddq +
C dq + g` cancels exactly what the model contains, and a deadband and a flex mode are not in
the model. The honest version of "hand-written dynamics beats a PD follower" is therefore
narrower than the first run suggested, and both runs are in `results/tracking.json`.

The fix that does work is not in the control loop. A backlash-inverse block ahead of the
controller holds the motor command inside the gap instead of driving through it, with a stiff
spring across the residual error. That takes backlash from 259.87 mm to 3.51 mm, a factor of
74, and it survives getting the gap wrong: with the real gap at 0.05 deg and the inverse set
to 0.03 or 0.07, the error is 5.20 and 5.69 mm against 5.45 for the exact value. The cost,
stated because it is not free, is 0.758 -> 0.82 mm on a plant that has no backlash at all.

The obvious next fix -- a disturbance observer estimating the torque the plant applies that
the model did not predict -- was implemented and measured, and it does not work here: exactly
1.00x on backlash, 1.00x on flex, and 4.5x WORSE on friction. The reason is in the numbers:
an observer estimates a torque, and a deadband is not a torque error, so the observer
integrates the mismatch and drives torque into the gap, which overshoots the instant the gap
is crossed. Both experiments are kept in the tree behind `--dob`, off by default, because a
negative result that can be re-run beats one that was quietly deleted.

## The layers

    rtforge/budget.py          the error budget every other number is compared against
    rtforge/latency/           cyclictest parsing; percentiles, not maxima
    rtforge/timebase/ptp.py    IEEE 1588 offset exchange and its contour-error consequence
    rtforge/vision/coreset.py  few-shot anomaly detection and its learning curve
    rtforge/eval/harness.py    runs what can be run, tags every number, prints the report
    cpp/include/dynamics.hpp   rigid-body dynamics and a dense QP, hand-written
    cpp/src/dynamics.cpp       inverse dynamics, mass matrix, Coriolis, gravity, QP
    cpp/tools/selftest.cpp     sixteen checks against closed forms and energy conservation
    cpp/tools/track.cpp        closed-loop tracking, feedforward swept

## Three things the self-test caught that review would not

1. The recursive Newton-Euler formulation was thrown away after five plausible-but-wrong
   versions and replaced with the definition (`tau_k = sum_i ...` over the links at or beyond
   `k`, with Jacobians by finite difference). Both now cost the same to read; only one of
   them can be checked line by line.
2. The gravity sign was settled by differentiating the potential, not by a free-body
   diagram. The ratio `g / (dV/dq)` is asserted to be +1.0000 in this convention; the
   first version asserted a magnitude and got the sign backwards, and an hour went into
   chasing a Coriolis sign that had never been wrong.
3. `M = C^T C` where `C` is the Cholesky factor of the mass matrix must reproduce `M` for
   the same reason `g` must equal `dV/dq`: both are identities that hold or they do not.

## Licence

Apache-2.0, the same terms as the tars skills this agent was ported from.
