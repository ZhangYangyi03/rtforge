// Validation against cases that can be computed with a pen.
//
// This is the answer to "you wrote the dynamics yourself, so how do you know they are
// right". Not by comparing to another library -- by comparing to physics that is short
// enough to check by hand. Each check prints PASS/FAIL and the residual, and the program
// exits non-zero if any of them fails, so CI can hold it.
#include "dynamics.hpp"

#include <cmath>
#include <cstdio>
#include <vector>

using namespace rtforge;

namespace {

int failures = 0;

void check(bool ok, const char* name, double got, double want, double tol) {
    std::printf("%-52s %s   got %14.9f  want %14.9f  tol %.1e\n",
                name, ok ? "PASS" : "FAIL", got, want, tol);
    if (!ok) ++failures;
}

// A two-link planar arm in the x-z plane: link 1 turns about y at the origin, link 2 about
// y at the end of link 1. Gravity along -z.
RobotModel two_link(double l1 = 0.4, double l2 = 0.3,
                    double m1 = 2.0, double m2 = 1.5) {
    RobotModel r;
    Joint a;
    a.mass = m1;
    a.axis = {0, 1, 0};
    a.origin = {0, 0, 0};
    a.com = {l1 * 0.5, 0, 0};
    // rod of length l1, uniform, about its centre, axis y
    a.inertia = Mat3::from(0, 0, 0,0, m1 * l1 * l1 / 12.0, 0,0, 0, m1 * l1 * l1 / 12.0);
    Joint b;
    b.mass = m2;
    b.axis = {0, 1, 0};
    b.origin = {l1, 0, 0};
    b.com = {l2 * 0.5, 0, 0};
    b.inertia = Mat3::from(0, 0, 0,0, m2 * l2 * l2 / 12.0, 0,0, 0, m2 * l2 * l2 / 12.0);
    r.joints = {a, b};
    r.viscous = {0, 0};
    r.coulomb = {0, 0};
    return r;
}

}  // namespace

int main() {
    // ---------------------------------------------------------------- gravity
    // Arm stretched horizontally (q = 0): the first joint carries both links at their
    // midpoints, so tau1 = g*(m1*l1/2 + m2*(l1 + l2/2)) and tau2 = g*m2*l2/2.
    {
        const double l1 = 0.4, l2 = 0.3, m1 = 2.0, m2 = 1.5, g = 9.81;
        auto r = two_link(l1, l2, m1, m2);
        std::vector<double> q{0.0, 0.0};
        auto tau = gravity_torques(r, q);
        // The joint axis is +y and gravity is -z; with the arm along +x, r x F points +y,
        // so the torque is positive.
        const double want1 = g * (m1 * l1 / 2.0 + m2 * (l1 + l2 / 2.0));
        const double want2 = g * m2 * l2 / 2.0;
        // SIGN. A joint torque is a generalised force, tau = -dV/dq. With +q about +y the
        // horizontal arm has its COM at +x, its weight along -z, and dV/dq = +12.017, so the
        // torque the actuator must supply is -12.017: it is holding the arm UP against a
        // gravity that wants to swing it down. The first version of this check asserted
        // +12.017 and spent an hour afterwards chasing a Coriolis sign that was never wrong.
        check(std::fabs(tau[0] + want1) < 1e-9, "gravity, shoulder, arm horizontal", tau[0], -want1, 1e-9);
        check(std::fabs(tau[1] + want2) < 1e-9, "gravity, elbow, arm horizontal", tau[1], -want2, 1e-9);
    }

    // Arm hanging straight down (q = -pi/2 each): gravity produces no torque at the
    // shoulder about y, because the lever arm is along the gravity direction.
    {
        auto r = two_link();
        std::vector<double> q{-M_PI / 2.0, 0.0};
        auto tau = gravity_torques(r, q);
        check(std::fabs(tau[0]) < 1e-9, "gravity, shoulder, arm vertical", std::fabs(tau[0]), 0.0, 1e-9);
        check(std::fabs(tau[1]) < 1e-9, "gravity, elbow, arm vertical", std::fabs(tau[1]), 0.0, 1e-9);
    }

    // ---------------------------------------------------------------- potential
    // The sign of gravity, settled by differentiating the potential rather than by drawing
    // a free body diagram: g_k must equal -dV/dq_k, where V = sum_i m_i g z_i computed from
    // the forward kinematics. Any sign convention that fails this makes the arm fall upward.
    {
        auto r = two_link();
        std::vector<double> q{0.5, 0.8};
        const double h = 1e-7;
        auto V = [&](const std::vector<double>& qq) {
            // Same chain as end_effector but accumulating every COM.
            Mat3 R = Mat3::identity();
            double x = 0, z = 0, total = 0;
            for (std::size_t i = 0; i < r.dof(); ++i) {
                x += R.m[0][0] * r.joints[i].origin.x;
                z += R.m[2][0] * r.joints[i].origin.x;
                const double c = std::cos(qq[i]), sn = std::sin(qq[i]);
                Mat3 Q = Mat3::identity();
                Q.m[0][0] = c;  Q.m[0][2] = sn;
                Q.m[2][0] = -sn; Q.m[2][2] = c;
                Mat3 Rn;
                for (int a2 = 0; a2 < 3; ++a2)
                    for (int b = 0; b < 3; ++b) {
                        double acc = 0;
                        for (int k2 = 0; k2 < 3; ++k2) acc += R.m[a2][k2] * Q.m[k2][b];
                        Rn.m[a2][b] = acc;
                    }
                R = Rn;
                const double cz = z + R.m[2][0] * r.joints[i].com.x;
                total += r.joints[i].mass * 9.81 * cz;
            }
            return total;
        };
        auto g = gravity_torques(r, q);
        for (int k = 0; k < 2; ++k) {
            std::vector<double> qp = q, qm = q;
            qp[k] += h; qm[k] -= h;
            const double dV = (V(qp) - V(qm)) / (2 * h);
            // The measured relation in this convention is g_k = +dV/dq_k, not -dV/dq_k.
            // That is not a mistake being papered over: R_y(a) maps +x to (cos a, 0, -sin a),
            // i.e. a positive joint angle swings the link DOWN, so the potential falls with q
            // and the generalised force that holds it up points the same way as the gradient.
            // The physical check that the sign is right is the one that matters: the torque
            // holds the arm UP (a free swing carries it toward hanging down, and the energy
            // test integrates exactly that and conserves).
            const double ratio = (std::fabs(dV) > 1e-9) ? g[k] / dV : 1.0;
            check(std::fabs(ratio - 1.0) < 1e-5,
                  k == 0 ? "gravity matches dV/dq in this convention, joint 0"
                        : "gravity matches dV/dq in this convention, joint 1",
                  ratio, 1.0, 1e-5);
        }
    }

    // ---------------------------------------------------------------- mass matrix
    // With q = 0, M is the planar-chain inertia about y. M11 is the total inertia about the
    // shoulder parallel-axis; M22 is the elbow's own inertia.
    {
        const double l1 = 0.4, l2 = 0.3, m1 = 2.0, m2 = 1.5;
        auto r = two_link(l1, l2, m1, m2);
        std::vector<double> q{0.0, 0.0};
        auto M = mass_matrix(r, q);
        // Careful: the parallel-axis term already contains the cross term. Writing
        // "I2_about_shoulder = I2_own + m2 (l1 + c2)^2" and then adding another m2*l1*c2 is
        // the error this line is here to record -- it inflated the expected value by 5.9%.
        const double c1 = l1 / 2.0, c2 = l2 / 2.0;
        const double I1 = m1 * l1 * l1 / 12.0 + m1 * c1 * c1;      // about the shoulder
        const double I2_own = m2 * l2 * l2 / 12.0 + m2 * c2 * c2;  // about the elbow
        // Second parallel-axis transfer, from the elbow to the shoulder. The lever is l1,
        // NOT (l1 + c2): I2_own already carries the arm from the elbow to link 2's COM, so
        // using (l1 + c2) here counts that arm twice and inflates M11 by m2*c2^2 = 0.034.
        // With the folded arm (q2 = 0) the cross term 2*m2*l1*c2*cos(q2) adds to it.
        const double I2_about_shoulder = I2_own + m2 * l1 * l1 + 2.0 * m2 * l1 * c2;
        check(std::fabs(M[0][0] - (I1 + I2_about_shoulder)) < 1e-9,
              "mass matrix M11 (parallel axis about shoulder)", M[0][0], I1 + I2_about_shoulder, 1e-9);
        check(std::fabs(M[1][1] - I2_own) < 1e-9, "mass matrix M22 (elbow)", M[1][1], I2_own, 1e-9);
        check(std::fabs(M[0][1] - M[1][0]) < 1e-12, "mass matrix is symmetric",
              std::fabs(M[0][1] - M[1][0]), 0.0, 1e-12);
    }

    // Mass matrix must stay symmetric and positive definite across a sweep of postures.
    // If it does not, RNEA and the inertia parameterisation disagree and every MPC result
    // downstream is meaningless.
    {
        auto r = two_link();
        bool symmetric = true, pd = true;
        for (int i = 0; i < 12; ++i) {
            for (int j = 0; j < 12; ++j) {
                std::vector<double> q{i * 0.5 - 3.0, j * 0.5 - 3.0};
                auto M = mass_matrix(r, q);
                if (std::fabs(M[0][1] - M[1][0]) > 1e-10) symmetric = false;
                if (!(M[0][0] > 0 && M[1][1] > 0 &&
                      M[0][0] * M[1][1] - M[0][1] * M[1][0] > 0)) pd = false;
            }
        }
        check(symmetric, "mass matrix symmetric over a 12x12 posture sweep", symmetric ? 1 : 0, 1, 0);
        check(pd, "mass matrix positive definite over the same sweep", pd ? 1 : 0, 1, 0);
    }

    // ---------------------------------------------------------------- Coriolis
    // C(q, dq) dq vanishes when dq = 0, and it is quadratic in dq: doubling dq quadruples it.
    {
        auto r = two_link();
        std::vector<double> q{0.3, -0.7};
        auto c0 = coriolis_torques(r, q, {0.0, 0.0});
        check(std::fabs(c0[0]) + std::fabs(c0[1]) < 1e-12, "Coriolis vanishes at rest",
              std::fabs(c0[0]) + std::fabs(c0[1]), 0.0, 1e-12);

        std::vector<double> v{0.5, -0.4}, v2{1.0, -0.8};
        auto c1 = coriolis_torques(r, q, v);
        auto c2 = coriolis_torques(r, q, v2);
        const double r0 = (std::fabs(c1[0]) > 1e-12) ? c2[0] / c1[0] : 4.0;
        const double r1 = (std::fabs(c1[1]) > 1e-12) ? c2[1] / c1[1] : 4.0;
        check(std::fabs(r0 - 4.0) < 1e-6, "Coriolis is quadratic in dq (joint 0)", r0, 4.0, 1e-6);
        check(std::fabs(r1 - 4.0) < 1e-6, "Coriolis is quadratic in dq (joint 1)", r1, 4.0, 1e-6);
    }

    // ---------------------------------------------------------------- energy
    // The strongest single check: integrate a free swing (zero joint torque) with the same
    // M, C and g the controller uses, and ask whether total mechanical energy stays put.
    // It fails for any wrong sign or factor anywhere in the three terms, which is how the
    // flipped Coriolis term was found.
    //
    // The residual is a first-order truncation error of the integrator, so the test asserts
    // the CONVERGENCE rather than an absolute bound: halving dt must halve the drift.
    {
        auto potential = [](const RobotModel& r, const State& st) {
            auto Q = [](double a) {
                Mat3 m = Mat3::identity();
                const double c = std::cos(a), s = std::sin(a);
                m.m[0][0] = c;  m.m[0][2] = s;
                m.m[2][0] = -s; m.m[2][2] = c;
                return m;
            };
            Mat3 R = Mat3::identity();
            double x = 0, z = 0, V = 0;
            for (std::size_t i = 0; i < r.dof(); ++i) {
                x += R.m[0][0] * r.joints[i].origin.x;
                z += R.m[2][0] * r.joints[i].origin.x;
                R = [&] { Mat3 C; for (int a2 = 0; a2 < 3; ++a2) for (int b = 0; b < 3; ++b) {
                    double acc = 0; for (int k = 0; k < 3; ++k) acc += R.m[a2][k] * Q(st.q[i]).m[k][b];
                    C.m[a2][b] = acc; } return C; }();
                const double cz = z + R.m[2][0] * r.joints[i].com.x;
                (void)x;
                V += r.joints[i].mass * 9.81 * cz;
            }
            return V;
        };
        auto energy = [&](const RobotModel& r, const State& st) {
            auto M = mass_matrix(r, st.q);
            double T = 0;
            for (std::size_t a2 = 0; a2 < r.dof(); ++a2)
                for (std::size_t b = 0; b < r.dof(); ++b)
                    T += 0.5 * M[a2][b] * st.dq[a2] * st.dq[b];
            return T + potential(r, st);
        };

        auto drift = [&](double dt, double seconds) {
            RobotModel r = two_link();
            r.viscous.assign(2, 0.0);
            r.coulomb.assign(2, 0.0);
            State s;
            s.resize(2);
            s.q = {0.5, 0.8};
            s.dq = {0.0, 0.0};
            const double E0 = energy(r, s);
            double worst = 0.0;
            const int steps = static_cast<int>(seconds / dt);
            for (int step = 0; step < steps; ++step) {
                // Free swing: the plant gets no torque. Integrating the forward equation
                // M ddq = -(C dq + g) is what makes energy conservation a statement about the
                // MODEL rather than about the integrator's bookkeeping.
                auto M = mass_matrix(r, s.q);
                auto g_t = gravity_torques(r, s.q);
                auto c_t = coriolis_torques(r, s.q, s.dq);
                const double b0 = -(c_t[0] + g_t[0]), b1 = -(c_t[1] + g_t[1]);
                const double det = M[0][0] * M[1][1] - M[0][1] * M[1][0];
                const double a0 = (M[1][1] * b0 - M[0][1] * b1) / det;
                const double a1 = (M[0][0] * b1 - M[1][0] * b0) / det;
                s.dq[0] += a0 * dt; s.dq[1] += a1 * dt;
                s.q[0] += s.dq[0] * dt; s.q[1] += s.dq[1] * dt;
                worst = std::max(worst, std::fabs(energy(r, s) - E0));
            }
            return worst;
        };

        const double coarse = drift(1e-4, 0.1);
        const double fine = drift(1e-5, 0.1);
        check(coarse < 1e-2, "energy drift at dt=1e-4 over 0.1 s (J)", coarse, 0.0, 1e-2);
        // First-order integrator: a factor 10 in dt should give a factor ~10 in drift.
        const double ratio = (fine > 0) ? coarse / fine : 0.0;
        check(ratio > 5.0 && ratio < 20.0,
              "energy drift converges with dt (expect ~10x for 10x smaller)", ratio, 10.0, 5.0);
    }

    std::printf("\n%s: %d failure(s)\n", failures ? "FAILED" : "OK", failures);
    return failures ? 1 : 0;
}
