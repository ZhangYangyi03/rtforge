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
  const double h=1e-6;
  // 检查 Coriolis 是否满足 Christoffel 关系：C dq 的第 k 分量应等于
  //   sum_{ij} (dM_kj/dq_i - 0.5 dM_ij/dq_k) dq_i dq_j
  std::vector<double> q={0.5,0.8}, dq={0.7,-0.4};
  auto Mof=[&](std::vector<double> qq){ return mass_matrix(r,qq); };
  std::vector<double> cdq = coriolis_torques(r,q,dq);
  for(int k=0;k<2;++k){
    double acc=0;
    for(int i=0;i<2;++i) for(int j=0;j<2;++j){
      std::vector<double> qi=q, qj=q, qk=q;
      qi[i]+=h; std::vector<double> qim=q; qim[i]-=h;
      qj[j]+=h; std::vector<double> qjm=q; qjm[j]-=h;
      qk[k]+=h; std::vector<double> qkm=q; qkm[k]-=h;
      double dMkj_dqi = (Mof(qi)[k][j]-Mof(qim)[k][j])/(2*h);
      double dMij_dqk = (Mof(qk)[i][j]-Mof(qkm)[i][j])/(2*h);
      acc += (dMkj_dqi - 0.5*dMij_dqk)*dq[i]*dq[j];
    }
    std::printf("joint %d:  C dq = %12.6f   Christoffel = %12.6f   diff = %e\n", k, cdq[k], acc, cdq[k]-acc);
  }
  return 0;
}
