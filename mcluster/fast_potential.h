/* Optional direct Plummer potential: include every unordered pair exactly once (j < i).
 * Only floating-point reduction order changes; no force approximation.
 * Compile with -DFAST_POTENTIAL -O3 -fopenmp -fno-math-errno.
 */
#ifndef MCLUSTER_FAST_POTENTIAL_H
#define MCLUSTER_FAST_POTENTIAL_H
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static double mcluster_fast_potential(int n, double **star)
{
    double *storage = malloc((size_t)n * 4 * sizeof(double));
    if (!storage) {
        fprintf(stderr, "Cannot allocate direct-potential coordinate buffer\n");
        exit(EXIT_FAILURE);
    }
    double *x = storage, *y = x+n, *z = y+n, *m = z+n;
    for (int i=0; i<n; ++i) {
        m[i]=star[i][0]; x[i]=star[i][1];
        y[i]=star[i][2]; z[i]=star[i][3];
    }
    fprintf(stderr, "Starting cache-blocked direct potential: N=%d\n", n);
    double potential = 0.0;
    const int block = 256;
    int completed = 0;
    const int tiles = (n+block-1)/block;
    const int report_every = tiles/20 > 0 ? tiles/20 : 1;
#ifndef NOOMP
#pragma omp parallel for schedule(dynamic, 1) reduction(+:potential)
#endif
    for (int ib=0; ib<n; ib+=block) {
        int end_i = ib+block < n ? ib+block : n;
        double tile_sum = 0.0;
        for (int jb=0; jb<end_i; jb+=block) {
            for (int i=ib; i<end_i; ++i) {
                int end_j = jb+block < i ? jb+block : i;
                double sum = 0.0;
#ifndef NOOMP
#pragma omp simd reduction(+:sum)
#endif
                for (int j=jb; j<end_j; ++j) {
                    double dx=x[i]-x[j], dy=y[i]-y[j], dz=z[i]-z[j];
                    sum += m[i]*m[j]/sqrt(dx*dx+dy*dy+dz*dz);
                }
                tile_sum += sum;
            }
        }
        potential -= tile_sum;
        int done;
#ifndef NOOMP
#pragma omp atomic capture
#endif
        done = ++completed;
        if (done % report_every == 0 || done == tiles) {
#ifndef NOOMP
#pragma omp critical(mcluster_potential_progress)
#endif
            fprintf(stderr, "Direct potential: %d/%d row blocks complete\n", done, tiles);
        }
    }
    free(storage);
    fprintf(stderr, "Direct potential complete: %.17g\n", potential);
    return potential;
}
#endif
