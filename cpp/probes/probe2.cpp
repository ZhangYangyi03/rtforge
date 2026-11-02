#include "dynamics.hpp"
#include <cstdio>
#include <cmath>
using namespace rtforge;
static void dump(const char* tag, const RobotModel& r, std::vector<double> q){
  double parent = 0.0;
  for(std::size_t i=0;i<r.joints.size();++i){
    parent += (i==0?0.0:r.joints[i-1].origin.x*0.0);
    auto M = mass_matrix(r, q);
    (void)M;
  }
  auto M2 = mass_matrix(r,q);
  std::printf("%s q=(%.2f,%.2f)  M=[[%9.5f %9.5f][%9.5f %9.5f]]  g=[%8.4f %8.4f]\n",
    tag,q[0],q[1],M2[0][0],M2[0][1],M2[1][0],M2[1][1],
    gravity_torques(r,q)[0],gravity_torques(r,q)[1]);
}
int main(){
  RobotModel r; 
  Joint a; a.mass=2.0; a.axis={0,1,0}; a.origin={0,0,0}; a.com={0.2,0,0};
  a.inertia = Mat3::from(0,0,0, 0,2.0*0.16/12.0,0, 0,0,2.0*0.16/12.0);
  Joint b; b.mass=1.5; b.axis={0,1,0}; b.origin={0.4,0,0}; b.com={0.15,0,0};
  b.inertia = Mat3::from(0,0,0, 0,1.5*0.09/12.0,0, 0,0,1.5*0.09/12.0);
  r.joints={a,b}; r.viscous={0,0}; r.coulomb={0,0};
  dump("two-link", r, {0.0,0.0});
  // 单轴，只在 q=0 给 ddq0=1，看 M[0][0]
  State s; s.resize(2); s.q={0,0};
  s.ddq={1.0,0.0};
  auto t1 = inverse_dynamics(r,s).tau;
  s.ddq={0.0,1.0};
  auto t2 = inverse_dynamics(r,s).tau;
  std::printf("RNEA(ddq=e0) = [%.6f %.6f]\n", t1[0], t1[1]);
  std::printf("RNEA(ddq=e1) = [%.6f %.6f]\n", t2[0], t2[1]);
  std::printf("expect M11=0.605417  M21=0.09  M22=0.045\n");
  return 0;
}
