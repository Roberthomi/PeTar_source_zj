# Armv8 SIMD compatibility backend

`src/simd_compat.h` retains the existing x86 intrinsic names and uses
[SIMDe](https://github.com/simd-everywhere/simde) native aliases to implement
the 256-bit kernel on AArch64 NEON. Kernel loops and particle layouts are
unchanged. This is not a new SVE kernel or a complete x86 intrinsic library;
the Arm path selects the existing AVX-width branch, not the AVX-512 branch.
No x86 compiler feature macros are forged. Native x86 builds still include
`immintrin.h`.

## Build on AArch64

Install SIMDe headers or download the tested version:

```sh
git clone --depth 1 --branch v0.8.2 https://github.com/simd-everywhere/simde.git /path/to/simde
cd PeTar_new_version
./configure --with-simd=neon CXXFLAGS="-O3 -march=armv8-a+simd -I/path/to/simde" --with-fdps-prefix=../FDPS --with-sdar-prefix=../SDAR
make
```

Keep any MPI, stellar evolution, and installation options required by your
existing build. `--enable-simd-64` remains available. Leave `--with-arch` at
its default: the legacy `x86` name selects the shared intrinsic kernel API,
while `--with-simd=neon` selects its Arm implementation. Configure rejects
this backend on non-AArch64 targets or if the SIMDe headers are missing.
SIMDe is an external, header-only dependency; it is not vendored here.

NEON uses 128-bit vectors, so a 256-bit operation may require multiple
instructions. SVE is not required. Raw NEON reciprocal-square-root estimates
have lower precision than x86 RSQRTPS; the compatibility header uses vector
sqrt/div for these calls to avoid degrading unrefined kernels. This favors
accuracy over peak throughput. Floating-point results need not be bitwise
identical to x86. Measure force error and simulation energy drift as well as
speed on the target CPU before production use.

## Standalone numeric regression

On AArch64, from this directory:

```sh
c++ -std=c++11 -O2 -march=armv8-a+simd -DPETAR_SIMD_NEON \
  -DCALC_EP_64bit -DRSQRT_NR_EPJ_X2 -DAVX_PRELOAD \
  -I/path/to/simde -Isrc test/simd_compat_numeric.cxx -o /tmp/petar-simd-test
/tmp/petar-simd-test
```

For a host-independent translation-layer test, replace
`-march=armv8-a+simd -DPETAR_SIMD_NEON` with
`-DPETAR_SIMD_PORTABLE -DSIMDE_NO_NATIVE`. For native x86 comparison, replace
it with `-mavx2 -mfma` on a CPU supporting those instructions.

The regression checks float and double kernel EPJ force/potential, neighbor
counts, and zero-quadrupole SPJ potential against scalar formulas, including
a partial final vector. It does not establish Arm performance or cover all
quadrupole inputs. The initial validation was run on x86 using both SIMDe's
portable implementation and native AVX2; an AArch64 run is still needed.
