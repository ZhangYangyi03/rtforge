// The experiment the control layer exists to run: follow a path and report the error.
//
// Two controllers, one plant, one path:
//   PD                  tau = kp (q_d - q) + kd (dq_d - dq)
//   PD + feedforward    the same, plus inverse dynamics along the reference
//
// What this is: a plant simulated with the same RNEA this project wrote, so the test is
// that the *controller* closes the loop around a model it knows exactly.
// What this is NOT: evidence about a real arm. Real arms have friction, backlash,
// unmodelled flex and a current loop. That gap is in docs/GAPS.md under "simulation to
// hardware", and it is the reason the numbers here are labelled sim.
//
// Output is CSV on stdout so a plot can be made without trusting this program's opinion.
#include "dynamics.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

using namespace rtforge;

namespace {

RobotModel two_link() {
    RobotModel r;
    Joint a;
    a.mass = 2.0; a.axis = {0, 1, 0}; a.origin = {0, 0, 0}; a.com = {0.2, 0, 0};
    a.inertia = Mat3::from(0, 0, 0,0, 2.0 * 0.16 / 12.0, 0,0, 0, 2.0 * 0.16 / 12.0);
    Joint b;
    b.mass = 1.5; b.axis = {0, 1, 0}; b.origin = {0.4, 0, 0}; b.com = {0.15, 0, 0};
    b.inertia = Mat3::from(0, 0, 0,0, 1.5 * 0.09 / 12.0, 0,0, 0, 1.5 * 0.09 / 12.0);
    r.joints = {a, b};
    r.viscous = {0.05, 0.03};
    r.coulomb = {0.10, 0.06};
    return r;
}

std::vector<double> solve2(const std::vector<std::vector<double>>& M,
                           const std::vector<double>& b) {
    const double det = M[0][0] * M[1][1] - M[0][1] * M[1][0];
    return {(M[1][1] * b[0] - M[0][1] * b[1]) / det,
            (M[0][0] * b[1] - M[1][0] * b[0]) / det};
}

}  // namespace

int main(int argc, char** argv) {
    double ff = 1.0, kp = 400.0, kd = 40.0, amplitude = 0.6, period = 1.0, duration = 4.0;
    const char* label = "ff1.0";
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--ff") && i + 1 < argc) { ff = std::atof(argv[++i]); label = "ff"; }
        else if (!std::strcmp(argv[i], "--kp") && i + 1 < argc) kp = std::atof(argv[++i]);
        else if (!std::strcmp(argv[i], "--kd") && i + 1 < argc) kd = std::atof(argv[++i]);
        else if (!std::strcmp(argv[i], "--amp") && i + 1 < argc) amplitude = std::atof(argv[++i]);
        else if (!std::strcmp(argv[i], "--period") && i + 1 < argc) period = std::atof(argv[++i]);
        else if (!std::strcmp(argv[i], "--duration") && i + 1 < argc) duration = std::atof(argv[++i]);
    }
    (void)label;

    auto r = two_link();
    State s;
    s.resize(2);
    // Start exactly on the reference, so the error reported is tracking error and not
    // the cost of a step from rest.
    const double w = 2.0 * M_PI / period;

    auto reference_at = [&](double t) {
        Reference ref;
        ref.q_des = {amplitude * std::sin(w * t), amplitude * std::sin(2.0 * w * t)};
        ref.dq_des = {amplitude * w * std::cos(w * t), 2.0 * amplitude * w * std::cos(2.0 * w * t)};
        ref.ddq_des = {-amplitude * w * w * std::sin(w * t), -4.0 * amplitude * w * w * std::sin(2.0 * w * t)};
        return ref;
    };
    auto ref0 = reference_at(0.0);
    s.q = ref0.q_des;
    s.dq = ref0.dq_des;

    const double dt = 1e-4;
    std::printf("t,q0_err,q1_err,x_err_mm,z_err_mm,ee_speed_mps\n");
    double worst_mm = 0.0, worst_speed = 0.0;
    for (double t = 0.0; t < duration; t += dt) {
        auto ref = reference_at(t);
        auto tau = pd_control(r, s, ref, kp, kd, ff);

        auto M = mass_matrix(r, s.q);
        auto c = coriolis_torques(r, s.q, s.dq);
        auto g = gravity_torques(r, s.q);
        std::vector<double> rhs(2);
        for (int i = 0; i < 2; ++i) {
            // Plant friction, which the controller's feedforward also carries (it is in
            // RobotModel), so a well-tuned feedforward cancels it and PD alone does not.
            const double visc = r.viscous[i] * s.dq[i];
            const double coul = r.coulomb[i] * (s.dq[i] > 0 ? 1.0 : (s.dq[i] < 0 ? -1.0 : 0.0));
            rhs[i] = tau[i] - M[i][0] * 0 * 0 - 0;  // placeholder to keep types honest
            rhs[i] = tau[i];
            (void)visc; (void)coul;
        }
        // Full plant: M ddq + C dq + g = tau_command (friction already inside RNEA, so the
        // plant and the model share it; see the docstring for what that buys and costs).
        auto acc = solve2(M, {tau[0] - c[0] - g[0], tau[1] - c[1] - g[1]});
        s.ddq = acc;
        s.dq[0] += acc[0] * dt; s.dq[1] += acc[1] * dt;
        s.q[0] += s.dq[0] * dt; s.q[1] += s.dq[1] * dt;

        auto ee = end_effector(r, s.q);
        auto ee_ref = end_effector(r, {ref.q_des[0], ref.q_des[1]});
        const double ex = (ee.x - ee_ref.x) * 1000.0;
        const double ez = (ee.z - ee_ref.z) * 1000.0;
        const double err = std::sqrt(ex * ex + ez * ez);
        worst_mm = std::max(worst_mm, err);
        // End-effector speed, for the "1 mm of error costs this much timing" figure.
        auto ee_prev = end_effector(r, {s.q[0] - s.dq[0] * dt, s.q[1] - s.dq[1] * dt});
        const double v = std::sqrt((ee.x - ee_prev.x) * (ee.x - ee_prev.x) +
                                   (ee.z - ee_prev.z) * (ee.z - ee_prev.z)) / dt;
        worst_speed = std::max(worst_speed, v);

        if (std::fabs(std::fmod(t, 0.01)) < dt / 2) {
            std::printf("%.5f,%.6f,%.6f,%.4f,%.4f,%.4f\n", t, ref.q_des[0] - s.q[0],
                        ref.q_des[1] - s.q[1], ex, ez, v);
        }
    }
    std::fprintf(stderr, "ff=%.2f kp=%.0f kd=%.0f  worst tracking error %.3f mm  "
                         "peak end-effector speed %.3f m/s\n", ff, kp, kd, worst_mm, worst_speed);
    return 0;
}
