#ifndef PETAR_SIMD_COMPAT_H
#define PETAR_SIMD_COMPAT_H

// Keep the kernel's x86 intrinsic names and fixed-width data layout.
// PETAR_SIMD_PORTABLE is for testing the translation layer on other hosts.
#if defined(PETAR_SIMD_NEON) || defined(PETAR_SIMD_PORTABLE) || defined(__aarch64__)
#if defined(PETAR_SIMD_NEON) && (!defined(__aarch64__) || !defined(__ARM_NEON))
#error "PETAR_SIMD_NEON requires AArch64 with NEON enabled"
#endif
#define PETAR_SIMD_TRANSLATED 1
#ifndef SIMDE_ENABLE_NATIVE_ALIASES
#define SIMDE_ENABLE_NATIVE_ALIASES
#endif
#include <simde/x86/avx2.h>
#include <simde/x86/fma.h>

// NEON's raw reciprocal-square-root estimate has less precision than x86
// RSQRTPS. Start with sqrt/div so existing kernels without Newton refinement
// (notably SPJ) do not silently lose accuracy. Benchmark before optimizing.
static inline simde__m128 petar_rsqrt_ps(simde__m128 x) {
    return simde_mm_div_ps(simde_mm_set1_ps(1.0f), simde_mm_sqrt_ps(x));
}
static inline simde__m256 petar_rsqrt256_ps(simde__m256 x) {
    return simde_mm256_div_ps(simde_mm256_set1_ps(1.0f), simde_mm256_sqrt_ps(x));
}
#undef _mm_rsqrt_ps
#undef _mm256_rsqrt_ps
#define _mm_rsqrt_ps(x) petar_rsqrt_ps(x)
#define _mm256_rsqrt_ps(x) petar_rsqrt256_ps(x)
#else
#include <immintrin.h>
#endif

#endif
