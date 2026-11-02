#include "dynamics.hpp"
#include <cstdio>
#include <cmath>
using namespace rtforge;
static RobotModel two_link(){
  RobotModel r; r.viscous={0,0}; r.coulomb={0,0};
  Joint a; a.mass=2.0; a.axis={0,1,0}; a.origin={0,0,0}; a.com={0.2,0,0};
  a.inertia=Mat3::from(0,0,0,0,2.0*0.16/12.0,0,0,0,2.0*0.16/12.0);
  Joint b; b.mass=1.5; b.axis={0,1,0}; b.origin={0.4,0,0}; b.com={0.15,0,0};
  b.inertia=Mat3::from(0,0,0,0,1.5*0.09/12.0,0,0,0,1.5*0.09/12.0);
  r.joints={a,b}; return r;
}
static double E(const RobotModel& r, const State& s){
  auto M=mass_matrix(r,s.q);
  double T=0.5*(M[0][0]*s.dq[0]*s.dq[0]+2*M[0][1]*s.dq[0]*s.dq[1]+M[1][1]*s.dq[1]*s.dq[1]);
  double V=0, ang=0, x=0,z=0;
  for(std::size_t i=0;i<2;++i){
    ang += s.q[i];
    x += r.joints[i].origin.x*std::cos(ang);
    z += -r.joints[i].origin.x*std::sin(ang);
    const double cx = x + r.joints[i].com.x*std::cos(ang);
    const double cz = z - r.joints[i].com.x*std::sin(ang);
    V += r.joints[i].mass*9.81*cz;
  }
  return T+V;
}
int main(){
  for(double dt : {1e-3,1e-4,1e-5,1e-6}){
    auto r=two_link(); State s; s.resize(2); s.q={0.5,0.8}; s.dq={0,0};
    const double E0=E(r,s); double worst=0;
    const int steps = (int)(0.05/dt);
    for(int k=0;k<steps;++k){
      auto M=mass_matrix(r,s.q); auto g=gravity_torques(r,s.q); auto c=coriolis_torques(r,s.q,s.dq);
      double b0=-(c[0]+g[0]), b1=-(c[1]+g[1]);
      double det=M[0][0]*M[1][1]-M[0][1]*M[1][0];
      double a0=(M[1][1]*b0-M[0][1]*b1)/det, a1=(M[0][0]*b1-M[1][0]*b0)/det;
      s.dq[0]+=a0*dt; s.dq[1]+=a1*dt; s.q[0]+=s.dq[0]*dt; s.q[1]+=s.dq[1]*dt;
      worst=std::max(worst,std::fabs(E(r,s)-E0));
    }
    std::printf("dt=%.0e  steps=%6d  E drift = %.6e J\n", dt, steps, worst);
  }
  return 0;
}
