# probes

Nine throwaway programs, kept because they are the evidence for the three claims in the top
README and because deleting them would make those claims look like assertions.

    probe1  one-joint arm: does the forward kinematics put the tip where the hand says
    probe2  two-link mass matrix against a closed form
    probe3  mass matrix with the second link's mass switched off, to isolate a term
    probe4  gravity torque split into mass-only / link-1-inertia / link-2-inertia
    probe5  energy drift against integration step, to separate truncation from a wrong model
    probe6  Coriolis against the Christoffel definition (ratio must be +1.0000)
    probe7  free swing: total mechanical energy over 0.05 s at two step sizes
    probe8  Coriolis against Christoffel with the sign printed explicitly
    probe9  gravity against dV/dq with the ratio printed explicitly

None of these are built by CMake on purpose: they are diagnostic, they print, and one of
them is wrong in an instructive way by design (probe3 removes a mass to see what changes).
The seventeen checks that gate the build live in tools/selftest.cpp.
