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
// 用库自己的 end_effector 式的 FK：com 的世界坐标
static void coms(const RobotModel& r, const State& s, double* zc){
  const double h=1e-9;
  // 用 mass_matrix 已经验证过的 FK：这里直接重写一次，但方向和库一致（左乘 Q(q)）
  std::vector<double> q=s.q; (void)h; (void)q;
  for(int i=0;i<2;++i){
    // accumulate R with Q(q) left-multiplied (same as rot_about)
    (void)i;
  }
  // 用 end_effector 在 q 和 q 处的差来求 com 的 z：不可行，于是直接复制库内公式
  auto Q=[&](double a){ Mat3 m=Mat3::identity(); m.m[0][0]=cos(a); m.m[0][2]=sin(a);
                        m.m[2][0]=-sin(a); m.m[2][2]=cos(a); return m; };
  auto mulm=[&](Mat3 A, Mat3 B){ Mat3 C; for(int i=0;i<3;++i) for(int j=0;j<3;++j){double s2=0;
              for(int k=0;k<3;++k) s2+=A.m[i][k]*B.m[k][j]; C.m[i][j]=s2;} return C; };
  Mat3 R=Mat3::identity(); double x=0,z=0;
  for(int i=0;i<2;++i){
    // r_i in parent frame
    x += R.m[0][0]*r.joints[i].origin.x;
    z += R.m[2][0]*r.joints[i].origin.x;
    R = mulm(R, Q(s.q[i]));
    const double cx = x + R.m[0][0]*r.joints[i].com.x;
    const double cz = z + R.m[2][0]*r.joints[i].com.x;
    zc[i]=cz; (void)cx;
  }
}
static double E(const RobotModel& r, const State& s){
  auto M=mass_matrix(r,s.q);
  double T=0.5*(M[0][0]*s.dq[0]*s.dq[0]+2*M[0][1]*s.dq[0]*s.dq[1]+M[1][1]*s.dq[1]*s.dq[1]);
  double zc[2]; coms(r,s,zc);
  return T + r.joints[0].mass*9.81*zc[0] + r.joints[1].mass*9.81*zc[1];
}
int main(){
  for(double dt : {1e-4,1e-5}){
    auto r=two_link(); State s; s.resize(2); s.q={0.5,0.8}; s.dq={0,0};
    const double E0=E(r,s); double worst=0; const int steps=(int)(0.05/dt);
    for(int k=0;k<steps;++k){
      auto M=mass_matrix(r,s.q); auto g=gravity_torques(r,s.q); auto c=coriolis_torques(r,s.q,s.dq);
      double b0=-(c[0]+g[0]), b1=-(c[1]+g[1]);
      double det=M[0][0]*M[1][1]-M[0][1]*M[1][0];
      double a0=(M[1][1]*b0-M[0][1]*b1)/det, a1=(M[0][0]*b1-M[1][0]*b0)/det;
      s.dq[0]+=a0*dt; s.dq[1]+=a1*dt; s.q[0]+=s.dq[0]*dt; s.q[1]+=s.dq[1]*dt;
      worst=std::max(worst,std::fabs(E(r,s)-E0));
    }
    std::printf("dt=%.0e  E0=%.6f  drift=%.6e J   q=(%.4f,%.4f)\n", dt, E0, worst, s.q[0], s.q[1]);
  }
  return 0;
}
