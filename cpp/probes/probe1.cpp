#include "dynamics.hpp"
#include <cstdio>
#include <cmath>
using namespace rtforge;
int main(){
  RobotModel r; Joint a; a.mass=1.0; a.axis={0,1,0}; a.origin={0,0,0}; a.com={0.5,0,0};
  a.inertia = Mat3::identity();
  r.joints={a}; r.viscous={0}; r.coulomb={0};
  for(double q : {0.0, M_PI/2, -M_PI/2}){
    auto tau = gravity_torques(r, {q});
    auto ee = end_effector(r, {q});
    std::printf("q=%7.3f  tau_y=%9.4f  com_at=(%.3f, %.3f)  [expected com=(%.3f, %.3f)]\n",
      q, tau[0], ee.x, ee.z, 0.5*std::cos(q), -0.5*std::sin(q));
  }
  return 0;
}
