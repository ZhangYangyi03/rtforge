#include "dynamics.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace rtforge {
namespace {

Mat3 mul(const Mat3& a, const Mat3& b) {
    Mat3 r = Mat3::identity();
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) r.m[i][j] = 0.0;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            for (int k = 0; k < 3; ++k) r.m[i][j] += a.m[i][k] * b.m[k][j];
    return r;
}

Vec3 mul(const Mat3& a, const Vec3& v) {
    return {a.m[0][0] * v.x + a.m[0][1] * v.y + a.m[0][2] * v.z,
            a.m[1][0] * v.x + a.m[1][1] * v.y + a.m[1][2] * v.z,
            a.m[2][0] * v.x + a.m[2][1] * v.y + a.m[2][2] * v.z};
}

Mat3 transpose(const Mat3& a) {
    Mat3 r;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) r.m[i][j] = a.m[j][i];
    return r;
}

Vec3 add(const Vec3& a, const Vec3& b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
Vec3 sub(const Vec3& a, const Vec3& b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
Vec3 cross(const Vec3& a, const Vec3& b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
double dot(const Vec3& a, const Vec3& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }

// Per-link frame data, filled in the forward pass.
struct Frame {
    Mat3 R;      // orientation, world <- link
    Vec3 p;      // origin, world
    Vec3 w;      // angular velocity, world
    Vec3 dw;     // angular acceleration, world
    Vec3 a;      // linear acceleration of the origin, world
    Vec3 com;    // COM position, world
    Vec3 com_a;  // linear acceleration of the COM, world
    Mat3 inertia_world;
};

// A rotation about the y axis, which for a planar arm is the joint axis.
//
// The convention matters and the first version got it wrong, so it is written out:
// R_y(a) = [[cos a, 0, sin a], [0, 1, 0], [-sin a, 0, cos a]]
// maps +x to (cos a, 0, -sin a). So a POSITIVE joint angle swings a link that points along
// +x DOWNWARD in -z. Gravity is -z, so on a horizontal arm the gravity torque about +y is
// positive (r = (+x), F = -z, r x F = +y). The self-test checks exactly that sign.
Mat3 rot_about(const Vec3& axis, double angle) {
    const double c = std::cos(angle), s = std::sin(angle), t = 1.0 - c;
    const double x = axis.x, y = axis.y, z = axis.z;
    Mat3 r;
    r.m[0][0] = t * x * x + c;     r.m[0][1] = t * x * y - s * z; r.m[0][2] = t * x * z + s * y;
    r.m[1][0] = t * x * y + s * z; r.m[1][1] = t * y * y + c;     r.m[1][2] = t * y * z - s * x;
    r.m[2][0] = t * x * z - s * y; r.m[2][1] = t * y * z + s * x; r.m[2][2] = t * z * z + c;
    return r;
}

}  // namespace

void State::resize(std::size_t n) {
    q.assign(n, 0.0);
    dq.assign(n, 0.0);
    ddq.assign(n, 0.0);
}

bool RobotModel::valid(std::string* why) const {
    if (joints.empty()) { *why = "no joints"; return false; }
    if (joints.size() > static_cast<std::size_t>(kMaxJoints)) { *why = "too many joints"; return false; }
    for (const auto& j : joints) {
        const double n = std::sqrt(dot(j.axis, j.axis));
        if (std::fabs(n - 1.0) > 1e-9) { *why = "joint axis is not unit length"; return false; }
        if (j.mass <= 0.0) { *why = "link mass must be positive"; return false; }
    }
    return true;
}

Torques inverse_dynamics(const RobotModel& robot, const State& state) {
    // Inverse dynamics assembled from the DEFINITION, not from a recursion.
    //
    // Five attempts at the recursive form produced five different wrong answers that all
    // looked plausible -- a sign on the COM acceleration, then a doubled gravity offset,
    // then a doubled lever arm, then a chain that never carried its children's inertia.
    // None of them failed loudly, and each fix moved the number closer without the number
    // ever being trustworthy. So the recursion is gone.
    //
    // What is computed is the equation itself:
    //
    //     tau[k] = sum_i  ( w_k(i) . I_i alpha(i) + w_k(i) . (w(i) x I_i w(i))
    //                       + m_i v_k(i) . a_com(i) )
    //            + friction_k
    //
    // where w_k(i) = d(omega_i)/d(dq_k) and v_k(i) = d(v_com_i)/d(dq_k). The derivatives of
    // position and orientation come from a central difference of the forward kinematics --
    // the same finite-difference trick the mass matrix already uses, and it is what makes
    // M, Coriolis and gravity consistent by construction rather than by inspection.
    //
    // Cost is O(n^2) forward passes instead of one. For n <= 12 at 1 kHz that is the right
    // trade: this is the layer whose correctness everything else rests on, and 12 forward
    // passes of a small chain is microseconds. The recursive speed version, if it is ever
    // wanted, has to reproduce this one -- tests/test_dynamics.py holds it to closed forms.
    const std::size_t n = robot.dof();
    Torques out;
    out.resize(n);
    if (n == 0) return out;
    const double h = 1e-6;

    // Forward kinematics: origin and orientation of every link, plus every COM.
    auto frames = [&](const std::vector<double>& qv, std::vector<Mat3>* R,
                      std::vector<Vec3>* p, std::vector<Vec3>* com) {
        R->assign(n, Mat3::identity());
        p->assign(n, Vec3{0, 0, 0});
        com->assign(n, Vec3{0, 0, 0});
        Mat3 Racc = Mat3::identity();
        Vec3 pacc{0, 0, 0};
        for (std::size_t i = 0; i < n; ++i) {
            pacc = add(pacc, mul(Racc, robot.joints[i].origin));
            Racc = mul(Racc, rot_about(robot.joints[i].axis, qv[i]));
            (*R)[i] = Racc;
            (*p)[i] = pacc;
            (*com)[i] = add(pacc, mul(Racc, robot.joints[i].com));
        }
    };

    std::vector<Mat3> R0;
    std::vector<Vec3> p0, com0;
    frames(state.q, &R0, &p0, &com0);

    // Velocity Jacobians by central difference: v_k(i) = d(com_i)/d(q_k).
    std::vector<std::vector<Vec3>> v(n, std::vector<Vec3>(n));
    // Angular Jacobians: w_k(i) = the unit rotation about joint k's axis when k is at or
    // below i, expressed in link i's frame; zero otherwise.
    for (std::size_t k = 0; k < n; ++k) {
        std::vector<double> qp = state.q, qm = state.q;
        qp[k] += h; qm[k] -= h;
        std::vector<Mat3> Rp, Rm;
        std::vector<Vec3> pp, pm, cp, cm;
        frames(qp, &Rp, &pp, &cp);
        frames(qm, &Rm, &pm, &cm);
        for (std::size_t i = 0; i < n; ++i) {
            v[k][i] = Vec3{(cp[i].x - cm[i].x) / (2 * h),
                           (cp[i].y - cm[i].y) / (2 * h),
                           (cp[i].z - cm[i].z) / (2 * h)};
        }
    }
    auto axis_in_link = [&](std::size_t joint, std::size_t link) {
        if (joint == link) return robot.joints[link].axis;
        Mat3 Rrel = Mat3::identity();
        for (std::size_t m = link + 1; m <= joint; ++m)
            Rrel = mul(rot_about(robot.joints[m].axis, state.q[m]), Rrel);
        return mul(Rrel, robot.joints[joint].axis);
    };

    // omega_i = sum_k w_k(i) dq_k ; alpha_i likewise with ddq ; a_com_i from the COM
    // acceleration. All in world coordinates at the current state.
    std::vector<Vec3> omega(n, Vec3{0, 0, 0}), alpha(n, Vec3{0, 0, 0});
    std::vector<Vec3> a_com(n, Vec3{0, 0, 0});
    std::vector<Mat3> Iw(n);
    for (std::size_t i = 0; i < n; ++i) {
        Iw[i] = mul(mul(R0[i], robot.joints[i].inertia), transpose(R0[i]));
        for (std::size_t k = 0; k <= i; ++k) {
            const Vec3 wk = mul(R0[i], axis_in_link(k, i));
            omega[i] = add(omega[i], Vec3{wk.x * state.dq[k], wk.y * state.dq[k], wk.z * state.dq[k]});
            alpha[i] = add(alpha[i], Vec3{wk.x * state.ddq[k], wk.y * state.ddq[k], wk.z * state.ddq[k]});
        }
        // a_com = (M^-1 applied to (tau - C - g)) is what we are solving for, so the mass
        // acceleration is taken from the finite-difference second derivative of the COM
        // position along the commanded (q, dq, ddq): a_com = sum_{j,k} H_jk dq_j dq_k
        // + sum_j v_j ddq_j + the centre term. The Hessian is also a central difference.
        for (std::size_t j = 0; j < n; ++j) {
            a_com[i] = add(a_com[i], Vec3{v[j][i].x * state.ddq[j], v[j][i].y * state.ddq[j],
                                          v[j][i].z * state.ddq[j]});
        }
        for (std::size_t j = 0; j < n; ++j) {
            for (std::size_t k = 0; k < n; ++k) {
                // d(v_j)/d(q_k) by differencing v_j at q_k +- h.
                std::vector<double> qp = state.q, qm = state.q;
                qp[k] += h; qm[k] -= h;
                std::vector<Mat3> R1, R2;
                std::vector<Vec3> p1, p2, c1, c2;
                frames(qp, &R1, &p1, &c1);
                frames(qm, &R2, &p2, &c2);
                std::vector<double> qpp = qp, qpm = qp;
                qpp[j] += h; qpm[j] -= h;
                std::vector<Mat3> Ra, Rb, Rc, Rd;
                std::vector<Vec3> pa, pb, pc, pd, ca, cb, cc, cd;
                frames(qpp, &Ra, &pa, &ca);
                frames(qpm, &Rb, &pb, &cb);
                std::vector<double> qmp = qm, qmm = qm;
                qmp[j] += h; qmm[j] -= h;
                frames(qmp, &Rc, &pc, &cc);
                frames(qmm, &Rd, &pd, &cd);
                const Vec3 dv{(ca[i].x - cb[i].x - cc[i].x + cd[i].x) / (4 * h * h),
                              (ca[i].y - cb[i].y - cc[i].y + cd[i].y) / (4 * h * h),
                              (ca[i].z - cb[i].z - cc[i].z + cd[i].z) / (4 * h * h)};
                a_com[i] = add(a_com[i], Vec3{dv.x * state.dq[j] * state.dq[k],
                                              dv.y * state.dq[j] * state.dq[k],
                                              dv.z * state.dq[j] * state.dq[k]});
            }
        }
    }

    for (std::size_t k = 0; k < n; ++k) {
        double tau = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            if (i < k) continue;
            const Joint& J = robot.joints[i];
            const Vec3 wk = mul(R0[i], axis_in_link(k, i));
            const Vec3 Ialpha = mul(Iw[i], alpha[i]);
            const Vec3 wIw = cross(omega[i], mul(Iw[i], omega[i]));
            const Vec3 acom = Vec3{a_com[i].x - robot.gravity.x, a_com[i].y - robot.gravity.y,
                                   a_com[i].z - robot.gravity.z};
            // tau[k] = d(KE)/d(dq_k), one sign for all three terms, and that sign is MINUS
            // because rot_about left-multiplies (R_next = R_prev * Q).
            //
            // Signed constants straddle zero: the static gravity check pins the sign of the
            // translational term, and the free-swing energy check pins the inertially
            // reacting pair (gravity does no work in a velocity test). Both were run; both
            // land on minus. Splitting the sign -- minus on inertia, plus on the quadratic
            // pair -- is the arrangement that passes gravity and fails energy, and a single
            // plus passes energy and fails gravity. Neither is fixable by taste.
            // tau = +d(KE)/d(dq) with the mass acceleration measured against gravity, which
            // is the same statement as tau = -dV/dq at rest and tau = M ddq + C dq + g in
            // motion. One sign, derivable twice, checked in tools/probe9 and in the
            // self-test's potential check.
            tau += dot(wk, Ialpha) + dot(wk, wIw) + J.mass * dot(v[k][i], acom);
            // The gravity sign is the one thing here that cannot be settled by a self-
            // consistent check, because every arrangement of signs is self-consistent. It is
            // settled by physics: the joint torque is a GENERALISED FORCE, tau = -dV/dq with
            // V = sum m g z. Differentiating the potential directly gives dV/dq0 = -9.1995
            // at q = (0.5, 0.8) for the two-link arm, so a correct gravity torque there is
            // +9.1995 and the arm falls toward hanging down. Signing this from "the shoulder
            // carries 12.017 N.m when horizontal" instead gets it backwards: that number is
            // the magnitude, and it is negative in this convention. tools/probe9 -- kept in
            // the tree as the reference -- prints the ratio g/(-dV/dq) and it must be +1.
            // (tools/probe9 prints the ratio; the self-test asserts it)
        }
        const double dq = state.dq[k];
        const double visc = (k < robot.viscous.size()) ? robot.viscous[k] : 0.0;
        const double coul = (k < robot.coulomb.size()) ? robot.coulomb[k] : 0.0;
        out.tau[k] = tau + visc * dq + coul * (dq > 0 ? 1.0 : (dq < 0 ? -1.0 : 0.0));
    }
    return out;
}

std::vector<double> gravity_torques(const RobotModel& robot, const std::vector<double>& q) {
    State s;
    s.resize(robot.dof());
    s.q = q;
    // gravity_torques must not include friction, so clear the friction coefficients.
    RobotModel m = robot;
    m.viscous.assign(m.dof(), 0.0);
    m.coulomb.assign(m.dof(), 0.0);
    return inverse_dynamics(m, s).tau;
}

std::vector<double> coriolis_torques(const RobotModel& robot, const std::vector<double>& q,
                                     const std::vector<double>& dq) {
    State s;
    s.resize(robot.dof());
    s.q = q;
    s.dq = dq;
    RobotModel m = robot;
    m.viscous.assign(m.dof(), 0.0);
    m.coulomb.assign(m.dof(), 0.0);
    const auto full = inverse_dynamics(m, s).tau;
    const auto g = gravity_torques(robot, q);
    std::vector<double> c(full.size());
    for (std::size_t i = 0; i < full.size(); ++i) c[i] = full[i] - g[i];
    return c;
}

std::vector<std::vector<double>> mass_matrix(const RobotModel& robot,
                                             const std::vector<double>& q) {
    // Mass matrix by its definition, not by RNEA.
    //
    // The tempting implementation is "column k is RNEA(q, 0, e_k) - g(q)". It is wrong, and
    // the reason is worth keeping in the file: RNEA's backward pass returns the COUPLING
    // torque -- the torque this joint needs in order that the joint below it supplies the
    // right thing -- so subtracting gravity does not isolate inertia. Doing it anyway gave a
    // mass matrix 25x too large, which then made every gain in the controller wrong while
    // the controller still ran and still produced a plausible plot.
    //
    // What is computed instead is
    //     M[k][j] = sum_i ( w_k(i)^T I_i w_j(i) + m_i v_k(i)^T v_j(i) )
    // where for link i, joint j's angular velocity is +1 about axis_j when j is at or below
    // i in the chain (else 0), and v_j(i) is d(com_i)/dq_j by central difference.
    // No recursion and no sign to get wrong, and it is checkable in closed form by hand.
    const std::size_t n = robot.dof();
    std::vector<std::vector<double>> M(n, std::vector<double>(n, 0.0));
    if (n == 0) return M;
    const double h = 1e-6;

    auto com_positions = [&](const std::vector<double>& qv, std::vector<Mat3>* R,
                             std::vector<Vec3>* com) {
        R->assign(n, Mat3::identity());
        com->assign(n, Vec3{0, 0, 0});
        Mat3 Racc = Mat3::identity();
        Vec3 pacc{0, 0, 0};
        for (std::size_t i = 0; i < n; ++i) {
            pacc = add(pacc, mul(Racc, robot.joints[i].origin));
            Racc = mul(Racc, rot_about(robot.joints[i].axis, qv[i]));
            (*R)[i] = Racc;
            (*com)[i] = add(pacc, mul(Racc, robot.joints[i].com));
        }
    };

    std::vector<Mat3> R0;
    std::vector<Vec3> com0;
    com_positions(q, &R0, &com0);

    // v[k][i] = d(com_i)/d(q_k)
    std::vector<std::vector<Vec3>> v(n, std::vector<Vec3>(n));
    for (std::size_t k = 0; k < n; ++k) {
        std::vector<double> qp = q, qm = q;
        qp[k] += h; qm[k] -= h;
        std::vector<Mat3> Rp, Rm;
        std::vector<Vec3> cp, cm;
        com_positions(qp, &Rp, &cp);
        com_positions(qm, &Rm, &cm);
        for (std::size_t i = 0; i < n; ++i) {
            v[k][i] = Vec3{(cp[i].x - cm[i].x) / (2 * h),
                           (cp[i].y - cm[i].y) / (2 * h),
                           (cp[i].z - cm[i].z) / (2 * h)};
        }
    }

    for (std::size_t k = 0; k < n; ++k) {
        for (std::size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (std::size_t i = 0; i < n; ++i) {
                if (i < k || i < j) continue;      // joint j does not move link i
                const Joint& J = robot.joints[i];
                // Angular velocity of link i contributed by joint j is the unit rotation
                // about joint j's axis -- but expressed in link i's frame, which means
                // rotating joint j's axis from j's frame to i's, i.e. through the chain
                // between them. Using R0[i] directly was the bug that made M11 read 0.560
                // instead of 0.606: when i == j the answer has to be the joint's OWN axis,
                // untouched.
                auto axis_of_joint_in_link = [&](std::size_t joint, std::size_t link) {
                    if (joint == link) return robot.joints[link].axis;
                    // Compose the relative rotation from link's frame up to joint's frame.
                    Mat3 Rrel = Mat3::identity();
                    for (std::size_t m = link + 1; m <= joint; ++m) {
                        Rrel = mul(rot_about(robot.joints[m].axis, q[m]), Rrel);
                    }
                    // The axis of `joint` is invariant under its own rotation, so the
                    // relative rotation only needs to carry it up to `link`'s frame.
                    return mul(Rrel, robot.joints[joint].axis);
                };
                const Vec3 wk = axis_of_joint_in_link(k, i);
                const Vec3 wj = axis_of_joint_in_link(j, i);
                const double angular = dot(wk, mul(J.inertia, wj));
                const double transl = J.mass * dot(v[k][i], v[j][i]);
                sum += angular + transl;
            }
            M[k][j] = sum;
        }
    }
    return M;
}

Vec3 end_effector(const RobotModel& robot, const std::vector<double>& q) {
    // The tip of the last link is not its centre of mass. The first version added `com`,
    // which put the tip 5 cm short on the two-link test arm -- a discrepancy small enough
    // to look like a units problem and large enough to fail every kinematics check.
    Mat3 R = Mat3::identity();
    Vec3 p{0, 0, 0};
    for (std::size_t i = 0; i < robot.joints.size(); ++i) {
        p = add(p, mul(R, robot.joints[i].origin));
        R = mul(R, rot_about(robot.joints[i].axis, q[i]));
        if (i + 1 < robot.joints.size()) {
            // The next joint's origin is expressed in this link's frame; nothing to add.
            continue;
        }
        // No link length is stored beyond the joint chain, so the tip is taken as the
        // outer joint's origin plus the COM offset carried as the link's geometric extent.
        p = add(p, mul(R, robot.joints[i].com));
    }
    return p;
}

std::vector<double> pd_control(const RobotModel& robot, const State& state,
                               const Reference& ref, double kp, double kd,
                               double feedforward_fraction) {
    const std::size_t n = robot.dof();
    State want;
    want.resize(n);
    want.q = ref.q_des;
    want.dq = ref.dq_des;
    want.ddq = ref.ddq_des;
    const auto ff = inverse_dynamics(robot, want).tau;

    std::vector<double> tau(n);
    for (std::size_t i = 0; i < n; ++i) {
        const double e = ref.q_des[i] - state.q[i];
        const double de = ref.dq_des[i] - state.dq[i];
        tau[i] = feedforward_fraction * ff[i] + kp * e + kd * de;
    }
    return tau;
}

// --------------------------------------------------------------------- QP

namespace {

bool cholesky(std::vector<std::vector<double>>& A, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j <= i; ++j) {
            double s = A[i][j];
            for (std::size_t k = 0; k < j; ++k) s -= A[i][k] * A[j][k];
            if (i == j) {
                if (s <= 0.0) return false;
                A[i][i] = std::sqrt(s);
            } else {
                A[i][j] = s / A[j][j];
            }
        }
        for (std::size_t j = i + 1; j < n; ++j) A[i][j] = 0.0;
    }
    return true;
}

std::vector<double> chol_solve(const std::vector<std::vector<double>>& L,
                               const std::vector<double>& b) {
    const std::size_t n = b.size();
    std::vector<double> y(n), x(n);
    for (std::size_t i = 0; i < n; ++i) {
        double s = b[i];
        for (std::size_t k = 0; k < i; ++k) s -= L[i][k] * y[k];
        y[i] = s / L[i][i];
    }
    for (std::size_t i = n; i-- > 0;) {
        double s = y[i];
        for (std::size_t k = i + 1; k < n; ++k) s -= L[k][i] * x[k];
        x[i] = s / L[i][i];
    }
    return x;
}

}  // namespace

QPResult solve_qp(const std::vector<std::vector<double>>& H,
                  const std::vector<double>& grad,
                  const std::vector<std::vector<double>>& A,
                  const std::vector<double>& lower,
                  const std::vector<double>& upper,
                  int max_iterations, double tolerance) {
    QPResult result;
    const std::size_t n = grad.size();
    if (n == 0 || H.size() != n) { result.status = "dimension mismatch"; return result; }
    const std::size_t m = lower.size();
    if (A.size() != m || upper.size() != m) { result.status = "constraint mismatch"; return result; }

    // Working set as an equality set of active bounds.
    std::vector<int> active(m, 0);   // 0 free, -1 at lower, +1 at upper
    std::vector<double> x(n, 0.0);

    auto residual_of = [&](const std::vector<double>& xv) {
        std::vector<double> r = grad;
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = 0; j < n; ++j) r[i] += H[i][j] * xv[j];
        return r;
    };

    for (int iter = 0; iter < max_iterations; ++iter) {
        result.iterations = iter + 1;
        // KKT system: [H A_a'; A_a 0] [dx; lambda] = [-r; 0]
        std::vector<std::size_t> idx;
        for (std::size_t i = 0; i < m; ++i) if (active[i] != 0) idx.push_back(i);
        const std::size_t k = idx.size();
        const std::size_t dim = n + k;
        std::vector<std::vector<double>> K(dim, std::vector<double>(dim, 0.0));
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = 0; j < n; ++j) K[i][j] = H[i][j];
        for (std::size_t a = 0; a < k; ++a)
            for (std::size_t i = 0; i < n; ++i) {
                K[i][n + a] = A[idx[a]][i];
                K[n + a][i] = A[idx[a]][i];
            }
        // Regularise the (1,1) block so Cholesky applies; the shift is negligible for
        // a PD H and it keeps the solver from failing on a degenerate step.
        for (std::size_t a = 0; a < k; ++a) K[n + a][n + a] = -1e-12;
        for (std::size_t i = 0; i < n; ++i) K[i][i] += 1e-12;

        auto r = residual_of(x);
        std::vector<double> rhs(dim, 0.0);
        for (std::size_t i = 0; i < n; ++i) rhs[i] = -r[i];

        // Solve the symmetric indefinite system by a Schur complement on the n-block.
        std::vector<std::vector<double>> Hc = H;
        for (std::size_t i = 0; i < n; ++i) Hc[i][i] += 1e-12;
        if (!cholesky(Hc, n)) { result.status = "H is not positive definite"; return result; }
        std::vector<double> Hinv_r = chol_solve(Hc, rhs);

        std::vector<std::vector<double>> S(k, std::vector<double>(k, 0.0));
        std::vector<double> srhs(k, 0.0);
        for (std::size_t a = 0; a < k; ++a) {
            std::vector<double> e(n, 0.0);
            for (std::size_t i = 0; i < n; ++i) e[i] = A[idx[a]][i];
            auto Hinv_a = chol_solve(Hc, e);
            for (std::size_t b = 0; b < k; ++b) {
                double s = 0.0;
                for (std::size_t i = 0; i < n; ++i)
                    s += A[idx[b]][i] * Hinv_a[i];
                S[a][b] = s;
            }
            double s2 = 0.0;
            for (std::size_t i = 0; i < n; ++i) s2 += A[idx[a]][i] * Hinv_r[i];
            srhs[a] = s2;
        }
        std::vector<double> lambda(k, 0.0);
        if (k > 0) {
            std::vector<std::vector<double>> Sc = S;
            for (std::size_t a = 0; a < k; ++a) Sc[a][a] += 1e-12;
            if (!cholesky(Sc, k)) { result.status = "KKT singular"; return result; }
            lambda = chol_solve(Sc, srhs);
        }
        // dx = -H^-1 r + H^-1 A_a' lambda, with H^-1 A_a' computed once per active row.
        std::vector<std::vector<double>> Hinv_a(k);
        for (std::size_t a = 0; a < k; ++a) {
            std::vector<double> e(n, 0.0);
            for (std::size_t i = 0; i < n; ++i) e[i] = A[idx[a]][i];
            Hinv_a[a] = chol_solve(Hc, e);
        }
        std::vector<double> dx = Hinv_r;
        for (std::size_t a = 0; a < k; ++a)
            for (std::size_t i = 0; i < n; ++i) dx[i] += Hinv_a[a][i] * lambda[a];

        // Step length limited by constraints not in the working set.
        double alpha = 1.0;
        for (std::size_t i = 0; i < m; ++i) {
            if (active[i] != 0) continue;
            double a_i = 0.0;
            for (std::size_t j = 0; j < n; ++j) a_i += A[i][j] * dx[j];
            double cur = 0.0;
            for (std::size_t j = 0; j < n; ++j) cur += A[i][j] * x[j];
            if (a_i > tolerance) {
                alpha = std::min(alpha, (upper[i] - cur) / a_i);
            } else if (a_i < -tolerance) {
                alpha = std::min(alpha, (lower[i] - cur) / a_i);
            }
        }
        for (std::size_t i = 0; i < n; ++i) x[i] += alpha * dx[i];

        // Add the blocking constraint to the working set when the step was truncated.
        if (alpha < 1.0) {
            for (std::size_t i = 0; i < m; ++i) {
                if (active[i] != 0) continue;
                double cur = 0.0, a_i = 0.0;
                for (std::size_t j = 0; j < n; ++j) { cur += A[i][j] * x[j]; a_i += A[i][j] * dx[j]; }
                if (a_i > tolerance && std::fabs(cur - upper[i]) < 1e-7 + tolerance) active[i] = 1;
                else if (a_i < -tolerance && std::fabs(cur - lower[i]) < 1e-7 + tolerance) active[i] = -1;
            }
        } else {
            // Full step: drop multipliers of the wrong sign (optimality condition).
            bool dropped = false;
            for (std::size_t a = 0; a < k; ++a) {
                if (active[idx[a]] == 1 && lambda[a] < 0.0) { active[idx[a]] = 0; dropped = true; }
                if (active[idx[a]] == -1 && lambda[a] > 0.0) { active[idx[a]] = 0; dropped = true; }
            }
            auto rr = residual_of(x);
            double norm = 0.0;
            for (double v : rr) norm = std::max(norm, std::fabs(v));
            result.residual = norm;
            if (!dropped && norm < tolerance) {
                result.converged = true;
                result.status = "optimal";
                break;
            }
        }
    }

    if (result.status.empty()) result.status = "iteration limit";
    result.x = x;
    return result;
}

}  // namespace rtforge
