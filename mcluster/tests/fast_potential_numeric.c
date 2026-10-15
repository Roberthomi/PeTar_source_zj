#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "../fast_potential.h"

int main(int argc, char **argv)
{
    /* Analytic unequal-mass pair at distance 5: U = -2*3/5 (G=1).
     * The old j<i-1 loop returned zero for this configuration. */
    double a[] = {2, 0, 0, 0}, b[] = {3, 3, 4, 0};
    double *pair[] = {a, b};
    assert(fabs(mcluster_fast_potential(2, pair) + 1.2) < 1e-14);
    int n = argc > 1 ? atoi(argv[1]) : 1025;
    assert(n > 0);
    double **star = malloc((size_t)n*sizeof(*star));
    for (int i=0; i<n; ++i) {
        star[i] = malloc(4*sizeof(double));
        star[i][0] = (1.0 + i%7)/n;
        star[i][1] = sin(i*0.73);
        star[i][2] = cos(i*0.17);
        star[i][3] = i*0.001;
    }
    double t0 = omp_get_wtime(), reference = 0.0;
#pragma omp parallel for schedule(dynamic) reduction(+:reference)
    for (int i=0; i<n; ++i)
        for (int j=0; j<i; ++j) {
            double dx=star[i][1]-star[j][1];
            double dy=star[i][2]-star[j][2];
            double dz=star[i][3]-star[j][3];
            reference -= star[i][0]*star[j][0]/sqrt(dx*dx+dy*dy+dz*dz);
        }
    double t1=omp_get_wtime();
    double result=mcluster_fast_potential(n, star);
    double t2=omp_get_wtime();
    assert(fabs(result-reference) <= 1e-11*fmax(1.0, fabs(reference)));
    printf("N=%d reference_s=%.6f fast_s=%.6f speedup=%.2f relative_error=%.3g\n",
           n, t1-t0, t2-t1, (t1-t0)/(t2-t1),
           fabs(result-reference)/fmax(1e-300, fabs(reference)));
    for (int i=0; i<n; ++i) free(star[i]);
    free(star);
    return 0;
}
