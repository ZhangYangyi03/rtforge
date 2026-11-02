// 6-DOF rigid-body dynamics by recursive Newton-Euler, and a dense QP solver for MPC.
//
// This is the layer the project refuses to take from a closed library. Two things make
// "hand-written" mean something rather than a slogan:
//
//   1. The dynamics are validated against analytic cases, not against another
//      implementation. A two-link planar arm at zero velocity has a gravity torque anyone
//      can write with a pen; a spinning arm has an inertia that can be measured. If RNEA
//      disagrees with either, RNEA is wrong.
//   2. The QP has a termination criterion and an iteration cap it reports. A solver that
//      silently returns after N iterations is not a solver, it is a random number
//      generator with a sign convention.
//
// Not included, and named so nobody mistakes the boundary: no contact, no friction model,
// no flexible-body modes. Those are stated in docs/GAPS.md as the places this stack stops.
#pragma once

#include <array>
#include <cstddef>
#include <string>
#include <vector>

namespace rtforge {

constexpr int kMaxJoints = 12;

struct Vec3 {
    double x{0}, y{0}, z{0};
};

struct Mat3 {
    // A 2-D array member cannot be brace-initialised in a constructor's member-init list,
    // so the identity default comes from an aggregate-shaped factory instead. Written down
    // here because the first version did `Mat3 R{{1,0,0},{0,1,0},{0,0,1}}` and did not
    // compile -- aggregate initialisation of a struct whose member is double[3][3] needs
    // one more level of braces than it looks like.
    double m[3][3];
    static Mat3 identity() {
        Mat3 r;
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) r.m[i][j] = (i == j) ? 1.0 : 0.0;
        return r;
    }
    static Mat3 from(double a, double b, double c,
                     double d, double e, double f,
                     double g, double h, double i) {
        Mat3 r;
        r.m[0][0] = a; r.m[0][1] = b; r.m[0][2] = c;
        r.m[1][0] = d; r.m[1][1] = e; r.m[1][2] = f;
        r.m[2][0] = g; r.m[2][1] = h; r.m[2][2] = i;
        return r;
    }
};

// One revolute joint, axis and frame expressed by the standard DH-free convention:
// each joint has a fixed transform from the previous link's frame, plus a rotation about
// its own z axis by the joint angle.
struct Joint {
    double mass{0.0};               // kg, of the link this joint moves
    Vec3 com;                       // m, in this joint's local frame
    Mat3 inertia;                   // kg m^2, about the COM, local frame
    Vec3 axis{0, 0, 1};             // unit rotation axis
    Vec3 origin;                    // m, position of this joint's frame in the parent's
};

struct RobotModel {
    std::vector<Joint> joints;
    Vec3 gravity{0, 0, -9.81};
    // Viscous and Coulomb friction per joint, used by feedforward only.
    std::vector<double> viscous;
    std::vector<double> coulomb;

    std::size_t dof() const { return joints.size(); }
    bool valid(std::string* why) const;
};

// State of the arm in joint space.
struct State {
    std::vector<double> q, dq, ddq;
    void resize(std::size_t n);
};

// Output of the inverse dynamics: the joint torques that produce (q, dq, ddq).
struct Torques {
    std::vector<double> tau;
    void resize(std::size_t n) { tau.assign(n, 0.0); }
};

// Recursive Newton-Euler. tau = M(q) ddq + C(q,dq) dq + g(q).
Torques inverse_dynamics(const RobotModel& robot, const State& state);

// Only the gravity term: cheap, and the term that dominates at low speed.
std::vector<double> gravity_torques(const RobotModel& robot, const std::vector<double>& q);

// Only the Coriolis/centripetal term, obtained as RNEA(q, dq, 0) - g(q).
std::vector<double> coriolis_torques(const RobotModel& robot, const std::vector<double>& q,
                                     const std::vector<double>& dq);

// Mass matrix by the unit-vector method: column i is RNEA(q, 0, e_i) - g(q). O(n^2) RNEA
// calls, which is fine for n <= 12 and honest about how it was obtained.
std::vector<std::vector<double>> mass_matrix(const RobotModel& robot,
                                             const std::vector<double>& q);

// Forward kinematics: end-effector position of the last joint's frame.
Vec3 end_effector(const RobotModel& robot, const std::vector<double>& q);

// ------------------------------------------------------------------ MPC

struct QPResult {
    std::vector<double> x;
    int iterations{0};
    double residual{0.0};
    bool converged{false};
    std::string status;
};

// Minimise 0.5 x'Hx + g'x subject to l <= A x <= u, by an active-set method with a
// dense Cholesky solve. H must be symmetric positive definite -- MPC with a positive
// cost on control increments guarantees that, and this refuses otherwise rather than
// returning a plausible-looking wrong answer.
QPResult solve_qp(const std::vector<std::vector<double>>& H,
                  const std::vector<double>& grad,
                  const std::vector<std::vector<double>>& A,
                  const std::vector<double>& lower,
                  const std::vector<double>& upper,
                  int max_iterations = 200, double tolerance = 1e-9);

// A tracked trajectory the controller follows.
struct Reference {
    std::vector<double> q_des, dq_des, ddq_des;
};

// PD plus feedforward. `feedforward_fraction` = 0 is a pure PD follower; 1.0 is full
// inverse-dynamics feedforward. The whole point of the experiment is to sweep it.
std::vector<double> pd_control(const RobotModel& robot, const State& state,
                               const Reference& ref, double kp, double kd,
                               double feedforward_fraction);

}  // namespace rtforge
