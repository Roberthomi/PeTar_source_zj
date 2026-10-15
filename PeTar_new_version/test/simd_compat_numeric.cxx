// Standalone kernel regression; no MPI or FDPS installation required.
#include <cmath>
#include <cstdlib>
#include <iostream>
namespace PS { typedef int S32; }
#include "phantomquad_for_p3t_x86.hpp"

static void check(double got, double expected) {
    if (!std::isfinite(got) || std::abs(got-expected) > 3e-4*(1+std::abs(expected))) {
        std::cerr << "SIMD mismatch: " << got << " expected " << expected << '\n';
        std::exit(1);
    }
}

template<class Kernel> void test() {
    // Static storage preserves the kernel's alignment and avoids stack overflow.
    static Kernel k;
    k.set_eps2(0.125);
    k.set_r_crit2(0.25);
    for (int i=0; i<16; ++i) k.set_xi_one(i, i*0.125, 0.25, -0.5, 1.0);
    // Include the preload sentinel used by the original kernel.
    for (int j=0; j<6; ++j) {
        k.set_epj_one(j, j*0.5, -0.25, 0.125, 1+j*0.25, 1.0);
        k.set_spj_one(j, j*0.5, -0.25, 0.125, 1+j*0.25, 0,0,0,0,0,0);
    }
    // Exercise a full vector and a partial final vector.
    for (int ni : {8, 13}) {
        k.run_epj_for_p3t_with_linear_cutoff(ni, 5);
        for (int i=0; i<ni; ++i) {
            double ref[4]={}, neighbors=0;
            for (int j=0; j<5; ++j) {
                const double dx=j*0.5-i*0.125, dy=-0.5, dz=0.625;
                const double r2=dx*dx+dy*dy+dz*dz+0.125;
                const double inv=1/std::sqrt(r2), m=1+j*0.25;
                ref[0]+=m*inv*inv*inv*dx;
                ref[1]+=m*inv*inv*inv*dy;
                ref[2]+=m*inv*inv*inv*dz;
                ref[3]-=m*inv;
                neighbors += r2<=1.0;
            }
            double ax,ay,az,pot,n;
            k.get_accp_one(i,ax,ay,az,pot,n);
            check(ax,ref[0]); check(ay,ref[1]); check(az,ref[2]); check(pot,ref[3]);
            check(n,neighbors);
        }
        k.run_epj_for_neighbor_count(ni,5);
        for (int i=0; i<ni; ++i) {
            double ax,ay,az,pot,n,ref=0;
            for (int j=0; j<5; ++j) {
                double dx=j*0.5-i*0.125;
                ref += dx*dx+0.25+0.390625<=1.0;
            }
            k.get_accp_one(i,ax,ay,az,pot,n); check(n,ref);
        }
        k.run_spj(ni,5);
        for (int i=0; i<ni; ++i) {
            double ax,ay,az,pot,n,ref=0;
            for (int j=0; j<5; ++j) {
                double dx=j*0.5-i*0.125;
                ref -= (1+j*0.25)/std::sqrt(dx*dx+0.25+0.390625+0.125);
            }
            k.get_accp_one(i,ax,ay,az,pot,n); check(pot,ref);
        }
    }
}

int main() {
    test<PhantomGrapeQuad>();
#if defined(CALC_EP_64bit) || defined(CALC_EP_MIX)
    test<PhantomGrapeQuad64Bit>();
#endif
    std::cout << "SIMD kernel numeric checks passed\n";
}
