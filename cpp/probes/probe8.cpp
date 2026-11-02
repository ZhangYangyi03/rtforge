#include "dynamics.hpp"
#include <cstdio>
#include <cmath>
using namespace rtforge;
int main(){
  RobotModel r; r.viscous={0,0}; r.coulomb={0,0};
  Joint a; a.mass=2.0; a.axis={0,1,0}; a.origin={0,0,0}; a.com={0.2,0,0};
  a.inertia=Mat3::from(0,0,0,0,2.0*0.16/12.0,0,0,0,2.0*0.16/12.0);
  Joint b; b.mass=1.5; b.axis={0,1,0}; b.origin={0.4,0,0}; b.com={0.15,0,0};
  b.inertia=Mat3::from(0,0,0,0,1.5*0.09/12.0,0,0,0,1.5*0.09/12.0);
  r.joints={a,b};
  std::vector<double> q={0.5,0.8}, dq={0.7,-0.4};
  auto c = coriolis_torques(r,q,dq);          // = RNEA(q,dq,0) - g，纯 Christoffel
  const double h=1e-6;
  auto Mof=[&](std::vector<double> qq){ return mass_matrix(r,qq); };
  for(int k=0;k<2;++k){
    double acc=0;
    for(int i=0;i<2;++i) for(int j=0;j<2;++j){
      std::vector<double> qi=q; qi[i]+=h; std::vector<double> qim=q; qim[i]-=h;
      std::vector<double> qk=q; qk[k]+=h; std::vector<double> qkm=q; qkm[k]-=h;
      double dMkj = (Mof(qi)[k][j]-Mof(qim)[k][j])/(2*h);
      double dMij = (Mof(qk)[i][j]-Mof(qkm)[i][j])/(2*h);
      acc += (dMkj - 0.5*dMij)*dq[i]*dq[j];
    }
    std::printf("joint %d:  C = %+12.8f   Christoffel = %+12.8f   ratio = %+8.4f\n",
                k, c[k], acc, (std::fabs(acc)>1e-12? c[k]/acc : 0.0));
  }
  std::printf("\n若 ratio == +1 则符号和量级都对；-1 则整体反号；其他则是结构错误\n");
  return 0;
}
