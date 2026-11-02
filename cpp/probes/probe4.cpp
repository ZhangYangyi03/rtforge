#include "dynamics.hpp"
#include <cstdio>
using namespace rtforge;
static RobotModel build(double m1,double m2,double I1,double I2){
  RobotModel r;
  Joint a; a.mass=m1; a.axis={0,1,0}; a.origin={0,0,0}; a.com={0.2,0,0};
  a.inertia = Mat3::from(0,0,0, 0,I1,0, 0,0,I1);
  Joint b; b.mass=m2; b.axis={0,1,0}; b.origin={0.4,0,0}; b.com={0.15,0,0};
  b.inertia = Mat3::from(0,0,0, 0,I2,0, 0,0,I2);
  r.joints={a,b}; r.viscous={0,0}; r.coulomb={0,0};
  return r;
}
int main(){
  struct C { const char* tag; double m1,m2,I1,I2; };
  C cases[] = {
    {"mass only, no link inertia", 2.0,1.5, 0,0},
    {"link1 inertia only",         2.0,1.5, 2.0*0.16/12.0,0},
    {"link2 inertia only",         2.0,1.5, 0,1.5*0.09/12.0},
    {"both",                       2.0,1.5, 2.0*0.16/12.0,1.5*0.09/12.0},
  };
  for(auto&c:cases){
    auto r=build(c.m1,c.m2,c.I1,c.I2);
    auto t=gravity_torques(r,{0.0,0.0});
    std::printf("%-30s tau1=%9.5f tau2=%9.5f\n", c.tag, t[0], t[1]);
  }
  std::printf("\n手算: 质心力矩 = 9.81*(2*0.2 + 1.5*0.55) = %.5f\n", 9.81*(2*0.2+1.5*0.55));
  std::printf("若惯性项错误地贡献了常值，它会出现在上面的差里\n");
  return 0;
}
