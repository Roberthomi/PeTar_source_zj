# Optional direct Plummer potential acceleration

Build with `make mcluster_fast`, then use `./mcluster_fast` with the usual
McLuster arguments. The cache-blocked implementation is opt-in and applies only to Plummer.
Both original CPU potential loops and the accelerated loop now include all
unordered pairs; this correctness fix also applies to the default build. The optimized executable targets the build
machine's CPU (`-march=native`); rebuild it on other machines.

The CPU Plummer potential uses contiguous mass/coordinate arrays, 256-row
cache blocks and OpenMP SIMD reductions. It still performs direct O(N²)
summation, without a tree, softening or analytical approximation. Additional
storage is 32 bytes per particle (96 MB for three million particles).
The CPU pair range is corrected from `j < i-1` to `j < i`. The old
range omitted N-1 adjacent-index pairs and underestimated the magnitude of
the binding energy. This changes initial-condition normalization; regenerate
models for comparisons using the corrected implementation. Reduction order changes, so results are not bit-identical.
GPU builds retain the existing GPU path.

Progress goes to stderr after roughly every 5% of row blocks; later blocks
contain more pairs, so these counts are not a linear time estimate.

Numerical/kernel timing check:

```sh
gcc -O3 -march=native -fno-math-errno -fopenmp tests/fast_potential_numeric.c -lm -o /tmp/test_fast_potential
OMP_NUM_THREADS=32 /tmp/test_fast_potential 500000
```

Historical measurements before the pair-range fix (not validation of the
corrected physical normalization): local 32-thread test on 2026-10-15: 500,000 synthetic particles took
23.96 s in the reference loop and 16.21 s in the optimized loop (1.48x);
relative potential difference was 3.32e-12. A full 10,000-star Plummer
generation with the benchmark seed had maximum output difference
`abs(new-old)/max(1,abs(old))` of 1.94e-14. These are small-scale checks,
not a measured runtime for the three-million-star model.
