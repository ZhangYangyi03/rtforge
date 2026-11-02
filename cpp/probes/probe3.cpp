#include "dynamics.hpp"
#include <cstdio>
using namespace rtforge;
int main(){
  for (double m2 : {0.0, 1.5}) {
    RobotModel r;
    Joint a; a.mass=2.0; a.axis={0,1,0}; a.origin={0,0,0}; a.com={0.2,0,0};
    a.inertia = Mat3::from(0,0,0, 0,2.0*0.16/12.0,0, 0,0,2.0*0.16/12.0);
    Joint b; b.mass=(m2>0?m2:1e-9); b.axis={0,1,0}; b.origin={0.4,0,0}; b.com={0.15,0,0};
    b.inertia = Mat3::from(0,0,0, 0,1e-12,0, 0,0,1e-12);
    r.joints={a,b}; r.viscous={0,0}; r.coulomb={0,0};
    State s; s.resize(2); s.q={0,0}; s.ddq={1.0,0.0};
    auto t = inverse_dynamics(r,s).tau;
    std::printf("m2=%.2f  RNEA(ddq0=1) = [%.6f, %.6f]\n", m2, t[0], t[1]);
    s.ddq={0.0,1.0};
    t = inverse_dynamics(r,s).tau;
    std::printf("m2=%.2f  RNEA(ddq1=1) = [%.6f, %.6f]\n", m2, t[0], t[1]);
  }
  std::printf("\n手算: 只有第一节时 M11 = I_c + m1*(0.2)^2 = %.6f\n", 2.0*0.16/12.0 + 2.0*0.04);
  return 0;
}
