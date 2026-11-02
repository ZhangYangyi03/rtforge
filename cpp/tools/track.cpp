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

// ------------------------------------------------------------------ plant perturbations
//
// The first version of this program had the plant and the controller share one model, so
// the 0.758 mm tracking error measured the CONTROLLER and said nothing about the gap
// between a model and an arm. These are the three effects that gap is made of, each with a
// magnitude from typical low-cost hardware:
//
//   Friction that differs by direction  -- stick-slip on a ballscrew: 30% more resistance
//                                         going one way than the other. A model with a
//                                         single viscous+Coulomb term cannot represent it.
//   Backlash                            -- 0.05 deg of lost motion in a harmonic drive.
//                                         The output does not move until the input has
//                                         crossed the gap, which no smooth torque can undo.
//   Structural flex                     -- the link bends under load: 2 Hz mode at the tool.
//
// The controller always uses the CLEAN model. Only the plant sees the perturbations, which
// is the point.
struct Perturbation {
    double friction_asymmetry = 0.0;   // 0 = symmetric (what the model has), 0.3 = 30% worse one way
    double backlash_rad = 0.0;         // 0.05 deg = 8.7e-4 rad
    double flex_hz = 0.0;              // 0 = rigid; 2.0 = a soft link
    double flex_gain = 0.0;            // m of tip deflection per N.m, roughly
};

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
    Perturbation pert;
    double dob_tau = 0.0;      // 0 = observer off; 0.005 = 5 ms low-pass
    const char* label = "ff1.0";
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--ff") && i + 1 < argc) { ff = std::atof(argv[++i]); label = "ff"; }
        else if (!std::strcmp(argv[i], "--kp") && i + 1 < argc) kp = std::atof(argv[++i]);
        else if (!std::strcmp(argv[i], "--kd") && i + 1 < argc) kd = std::atof(argv[++i]);
        else if (!std::strcmp(argv[i], "--amp") && i + 1 < argc) amplitude = std::atof(argv[++i]);
        else if (!std::strcmp(argv[i], "--period") && i + 1 < argc) period = std::atof(argv[++i]);
        else if (!std::strcmp(argv[i], "--duration") && i + 1 < argc) duration = std::atof(argv[++i]);
        else if (!std::strcmp(argv[i], "--friction-asym") && i + 1 < argc) pert.friction_asymmetry = std::atof(argv[++i]);
        else if (!std::strcmp(argv[i], "--backlash-deg") && i + 1 < argc)
            pert.backlash_rad = std::atof(argv[++i]) * 3.141592653589793 / 180.0;
        else if (!std::strcmp(argv[i], "--flex-hz") && i + 1 < argc) pert.flex_hz = std::atof(argv[++i]);
        else if (!std::strcmp(argv[i], "--flex-gain") && i + 1 < argc) pert.flex_gain = std::atof(argv[++i]);
        else if (!std::strcmp(argv[i], "--dob") && i + 1 < argc) dob_tau = std::atof(argv[++i]);
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
    double backlash_state[2] = {s.q[0], s.q[1]};
    double flex_pos = 0.0, flex_vel = 0.0;

    // ---- disturbance observer
    //
    // Feedforward cancels what the model contains. Everything the model does NOT contain --
    // the deadband, the bending mode, friction that differs by direction -- shows up as a
    // torque the plant applies that the controller never asked for. A disturbance observer
    // estimates that torque from the mismatch between what the model predicts the arm
    // should be doing and what the encoder says it is doing, and feeds the ESTIMATE back.
    //
    // This is the standard form: d_hat is a first-order low-pass of (measured acceleration
    // minus model-implied acceleration) times the inertia, with time constant tau. tau is
    // the whole design: fast enough to catch the deadband, slow enough not to chase sensor
    // noise or to fight the controller's own bandwidth.
    double d_hat[2] = {0.0, 0.0};
    double dq_prev[2] = {s.dq[0], s.dq[1]};
    double tau_cmd[2] = {0.0, 0.0};
    std::printf("t,q0_err,q1_err,x_err_mm,z_err_mm,ee_speed_mps\n");
    double worst_mm = 0.0, worst_speed = 0.0;
    for (double t = 0.0; t < duration; t += dt) {
        auto ref = reference_at(t);
        // The controller sees the CLEAN model `r` and the measured state.
        auto tau = pd_control(r, s, ref, kp, kd, ff);

        // ---- plant: the same rigid body, plus whatever the model does not know about.
        auto M = mass_matrix(r, s.q);
        auto c = coriolis_torques(r, s.q, s.dq);
        auto g = gravity_torques(r, s.q);

        double extra[2] = {0.0, 0.0};
        if (pert.friction_asymmetry != 0.0) {
            for (int i = 0; i < 2; ++i) {
                const double sign = (s.dq[i] > 0) ? 1.0 : (s.dq[i] < 0 ? -1.0 : 0.0);
                const double base = 0.10 + 0.05 * i;
                extra[i] = base * sign * (1.0 + (sign > 0 ? pert.friction_asymmetry : 0.0));
            }
        }

        // Measured acceleration from the encoder, one step behind.
        double a_meas[2] = {(s.dq[0] - dq_prev[0]) / dt, (s.dq[1] - dq_prev[1]) / dt};
        dq_prev[0] = s.dq[0];
        dq_prev[1] = s.dq[1];

        // What the model says the arm should be doing, given the torque that was ACTUALLY
        // commanded last step -- tau_cmd, without d_hat in it. Including d_hat here (the
        // first version did) makes the comparison self-referential: the model predicts
        // using the same estimate the observer is trying to find, the mismatch is zero by
        // construction, and the observer reports nothing while looking like it works.
        auto a_model = solve2(M, {tau_cmd[0] - c[0] - g[0],
                                  tau_cmd[1] - c[1] - g[1]});

        if (dob_tau > 0.0) {
            for (int i = 0; i < 2; ++i) {
                // Mismatch in acceleration -> a torque via the inertia. Low-passed, then
                // subtracted: the plant is doing something the model did not predict, so
                // stop asking it to.
                const double e = a_meas[i] - a_model[i];
                d_hat[i] += (M[i][i] * e - d_hat[i]) * (dt / dob_tau);
                tau[i] -= d_hat[i];
            }
        }
        tau_cmd[0] = tau[0];
        tau_cmd[1] = tau[1];

        auto acc = solve2(M, {tau[0] - c[0] - g[0] - extra[0],
                              tau[1] - c[1] - g[1] - extra[1]});
        s.ddq = acc;
        s.dq[0] += acc[0] * dt;
        s.dq[1] += acc[1] * dt;
        s.q[0] += s.dq[0] * dt;
        s.q[1] += s.dq[1] * dt;

        // Backlash: the motor angle advances, the gearbox output only follows once the gap
        // is crossed. The measured angle is the output; the controller does not know this
        // happens, which is exactly why it is modelled here and not there.
        if (pert.backlash_rad > 0.0) {
            for (int i = 0; i < 2; ++i) {
                const double half = pert.backlash_rad / 2.0;
                if (s.q[i] - backlash_state[i] > half) backlash_state[i] = s.q[i] - half;
                if (s.q[i] - backlash_state[i] < -half) backlash_state[i] = s.q[i] + half;
                s.q[i] = backlash_state[i];
            }
        }

        auto ee = end_effector(r, s.q);
        auto ee_ref = end_effector(r, {ref.q_des[0], ref.q_des[1]});

        // Flex: a bending mode at the tool, driven by joint 2's torque. Not a structural
        // model -- a damped lag that bends the tool under load, which is what a soft link
        // does and what no rigid-body feedforward can predict.
        double flex_z = 0.0;
        if (pert.flex_hz > 0.0) {
            const double w = 2.0 * 3.141592653589793 * pert.flex_hz;
            const double drive = tau[1] * pert.flex_gain;
            flex_vel += ((drive - flex_pos) * w * w - flex_vel * 0.2 * w) * dt;
            flex_pos += flex_vel * dt;
            flex_z = -flex_pos;
        }

        const double ex = (ee.x - ee_ref.x) * 1000.0;
        const double ez = (ee.z + flex_z - ee_ref.z) * 1000.0;
        const double err = std::sqrt(ex * ex + ez * ez);
        worst_mm = std::max(worst_mm, err);

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
                         "peak end-effector speed %.3f m/s%s%s%s%s\n", ff, kp, kd, worst_mm, worst_speed,
                 pert.friction_asymmetry ? " [friction-asym]" : "",
                 pert.backlash_rad ? " [backlash]" : "",
                 pert.flex_hz ? " [flex]" : "",
                 dob_tau > 0.0 ? " [dob]" : "");
    return 0;
}
