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
    L3 vision    MVTec-AD, 20 normal images            bottle 90%, screw 20%, carpet 33%   FAILS
    L3 vision    synthetic parts, 2 normal samples     98.7% at 10% FA  measured
    L4 control   feedforward 1.0 vs 0.0, clean plant   0.758 vs 38.853 mm   measured
    L4 control   same, with backlash                  259.9 vs 273.8 mm    measured, 1.1x
    L4 control   same, with structural flex            47.8 vs 48.7 mm      measured, 1.0x
    L5 derived   1 us of jitter at 1 m/s               1 um at the tool

The L1 row is the important one. On a virtualised host the worst-case latency is 146 ms
against a 68.5 us budget: the design's real-time premise is not demonstrated by this
machine and `docs/GAPS.md` says what would demonstrate it.

The L3 and L4 rows are the ones that changed after the data arrived.

L3 on real data fails. The same method that detects a defect from two synthetic samples gets
90% AUROC on bottle, 20% on screw and 33% on carpet -- below chance. Two experiments in
`scripts/` establish that this is structural, not tuning: sweeping five aggregation
statistics does not rescue it, and aligning every image to the training mean (+-3, +-6 and
+-12 px searches) moves bottle from 90 to 93 and leaves the other two flat. Random
convolutional features carry no semantics; a pretrained backbone is what supplies them, and
not pulling one in was a deliberate choice that turns out to cost the whole layer.

L4 is the same story in a different shape. Feedforward buys a factor of 51 against a clean
plant and a factor of 1.0 once the plant has a deadband or a bending mode: `tau = M ddq +
C dq + g` cancels exactly what the model contains, and a deadband and a flex mode are not in
the model. The honest version of "hand-written dynamics beats a PD follower" is therefore
narrower than the first run suggested, and both runs are in `results/tracking.json`.

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
