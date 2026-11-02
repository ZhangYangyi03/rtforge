#include "dynamics.hpp"
#include <cstdio>
#include <cmath>
using namespace rtforge;
int main(){
  RobotModel r; r.viscous={0,0}; r.coulomb={0,0};
  Joint a; a.mass=2.0; a.axis={0,1,0}; a.origin={0,0,0}; a.com={0.2,0,0};
  a.inertia=Mat3::from(0,0,0,0,0.0266667,0,0,0,0.0266667);
  Joint b; b.mass=1.5; b.axis={0,1,0}; b.origin={0.4,0,0}; b.com={0.15,0,0};
  b.inertia=Mat3::from(0,0,0,0,0.01125,0,0,0,0.01125);
  r.joints={a,b};
  std::vector<double> q={0.5,0.8};
  auto g=gravity_torques(r,q);
  // 势能 V = sum m*g*h, h = com 的 z（在 -z 为重的世界里，h 就是 z）
  // 用手算：每节 com 的 z
  const double z1=-0.2*std::sin(0.5);
  const double z2=-0.4*std::sin(0.5)-0.15*std::sin(1.3);
  const double V = 2.0*9.81*z1 + 1.5*9.81*z2;
  std::printf("手算 V = %.9f  (z1=%.6f z2=%.6f)\n", V, z1, z2);
  // dV/dq_k 应等于 g_k
  const double hh=1e-7;
  auto Vof=[&](std::vector<double> qq){
    const double a1=qq[0], a12=qq[0]+qq[1];
    return 2.0*9.81*(-0.2*std::sin(a1)) + 1.5*9.81*(-0.4*std::sin(a1)-0.15*std::sin(a12));
  };
  for(int k=0;k<2;++k){
    std::vector<double> qp=q,qm=q; qp[k]+=hh; qm[k]-=hh;
    std::printf("dV/dq%d = %+12.6f   g_%d = %+12.6f   ratio=%+.4f\n",
      k,(Vof(qp)-Vof(qm))/(2*hh), k, g[k], g[k]/((Vof(qp)-Vof(qm))/(2*hh)));
  }
  return 0;
}
