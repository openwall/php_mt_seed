/*
 * Copyright (c) 2012-2014,2017 Solar Designer <solar at openwall.com>
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted.
 *
 * There's ABSOLUTELY NO WARRANTY, express or implied.
 */

#include <stdint.h>
#include <stdlib.h>
#include <errno.h>
#include <stdio.h>
#include <unistd.h> /* sysconf() */
#include <sys/times.h>
#include <assert.h>

#if defined(__MIC__) || defined(__AVX512F__)
#include <immintrin.h>
typedef __m512i vtype;
/* hack */
#define _mm_set1_epi32(x) _mm512_set1_epi32(x)
#define _mm_add_epi32(x, y) _mm512_add_epi32(x, y)
#define _mm_mullo_epi32(x, y) _mm512_mullo_epi32(x, y)
#ifdef __MIC__
#define _mm_macc_epi32(x, y, z) _mm512_fmadd_epi32(x, y, z)
#endif
#define _mm_slli_epi32(x, i) _mm512_slli_epi32(x, i)
#define _mm_srli_epi32(x, i) _mm512_srli_epi32(x, i)
#define _mm_and_si128(x, y) _mm512_and_epi32(x, y)
#define _mm_or_si128(x, y) _mm512_or_epi32(x, y)
#define _mm_xor_si128(x, y) _mm512_xor_epi32(x, y)
#elif defined(__AVX2__)
#include <x86intrin.h>
typedef __m256i vtype;
/* hack */
#define _mm_set1_epi32(x) _mm256_set1_epi32(x)
#define _mm_add_epi32(x, y) _mm256_add_epi32(x, y)
#define _mm_mullo_epi32(x, y) _mm256_mullo_epi32(x, y)
#define _mm_slli_epi32(x, i) _mm256_slli_epi32(x, i)
#define _mm_srli_epi32(x, i) _mm256_srli_epi32(x, i)
#define _mm_and_si128(x, y) _mm256_and_si256(x, y)
#define _mm_or_si128(x, y) _mm256_or_si256(x, y)
#define _mm_xor_si128(x, y) _mm256_xor_si256(x, y)
#define _mm_cmpeq_epi32(x, y) _mm256_cmpeq_epi32(x, y)
#define _mm_testz_si128(x, y) _mm256_testz_si256(x, y)
#warning AVX-512 not enabled. Try gcc -mavx512f (on Intel Knights Landing, Skylake-X, or some newer).
#elif defined(__SSE2__)
#include <emmintrin.h>
typedef __m128i vtype;
#ifdef __XOP__
#include <x86intrin.h>
#else
#ifdef __SSE4_1__
#include <smmintrin.h>
#else
#ifdef __GNUC__
#define _mm_mullo_epi32(a, b) ({ \
		__m128i _a = (a), _b = (b); \
		_mm_unpacklo_epi32( \
		    _mm_shuffle_epi32(_mm_mul_epu32(_a, _b), 0x08), \
		    _mm_shuffle_epi32(_mm_mul_epu32(_mm_srli_epi64(_a, 32), \
		    _mm_srli_epi64(_b, 32)), 0x08)); \
	})
#else
#define _mm_mullo_epi32(a, b) \
	_mm_unpacklo_epi32( \
	    _mm_shuffle_epi32(_mm_mul_epu32((a), (b)), 0x08), \
	    _mm_shuffle_epi32(_mm_mul_epu32(_mm_srli_epi64((a), 32), \
	    _mm_srli_epi64((b), 32)), 0x08))
#endif
#warning SSE4.1 not enabled, will use only SSE2 doing only 2 multiplies per 4-element vector. Try gcc -msse4 (on capable CPUs).
#endif
#ifdef __AVX__
#warning XOP and AVX2 are not enabled. Try gcc -mxop (on AMD Bulldozer or some newer) or -mavx2 (on Intel Haswell, AMD Zen, or newer).
#elif defined(__SSE4_1__)
#warning AVX* and XOP are not enabled. Try gcc -mxop (on AMD Bulldozer or some newer), -mavx (on Intel Sandy Bridge or newer), or -mavx2 (on Intel Haswell, AMD Zen, or newer).
#endif
#endif
#else
#warning SSE2 not enabled, will use non-vectorized code. Try gcc -msse2 (on non-ancient x86 CPUs).
#endif

#if !defined(__XOP__) && !defined(_mm_macc_epi32)
#define _mm_macc_epi32(x, y, z) \
	_mm_add_epi32(_mm_mullo_epi32(x, y), z)
#endif

#ifndef _OPENMP
#warning OpenMP not enabled, will only use one CPU core. Try gcc -fopenmp.
#endif

#define M 397
#define N 624

#define MATCH_PURE 1
#define MATCH_FULL 2
#define MATCH_SKIP 4
#define MATCH_LAST 8

typedef struct {
	uint32_t flags;
	int32_t mmin, mmax;
	int32_t rmin;
	uint32_t rspan;
} match_t;

typedef enum {
	PHP_LEGACY = 0,
	PHP_MODERN = 1,
	PHP_521 = 1,
	PHP_710 = 2
} version_t;

static const char *flavors[] = {
	"3.0.7 to 5.2.0",
	"5.2.1+"
};

static const char *versions[] = {
	"3.0.7 to 5.2.0",
	"5.2.1 to 7.0.x; HHVM",
	"7.1.0+"
};

#define NEXT_STATE(x, i) \
	(x) = 1812433253U * ((x) ^ ((x) >> 30)) + (i);

#if defined(__SSE2__) || defined(__MIC__)
static inline int diff(uint32_t x, uint32_t xs, uint32_t seed,
    const match_t *match, version_t version)
#else
static inline int diff(uint32_t x, uint32_t x1, uint32_t xs,
    const match_t *match, version_t version)
#endif
{
#if defined(__SSE2__) || defined(__MIC__)
	uint32_t xsi = seed;
#else
	uint32_t xsi = x1;
#endif
	unsigned int i = 1;

	while (1) {
		if (match->flags & MATCH_SKIP) {
			/* nothing */
		} else
		if (match->flags & MATCH_PURE) {
			if (x != match->mmin)
				break;
		} else {
			int32_t xr;
			if (match->flags & MATCH_FULL)
				xr = x;
			else if (version != PHP_710)
				xr = match->rmin +
				    (int32_t)((double)match->rspan * (x >> 1) /
				    (0x7fffffff + 1.0));
			else
				xr = match->rmin + x % match->rspan;
			if (xr < match->mmin || xr > match->mmax)
				break;
		}

		if (match->flags & MATCH_LAST)
			return 0;

		if (version == PHP_LEGACY) {
#if defined(__SSE2__) || defined(__MIC__)
			if (i == 1) {
				xsi = (69069 * 2) * xsi + 69069;
				i = 0;
			}
#endif
			do {
				x = xsi;
				xsi *= 69069;
				xs *= 69069;
			} while ((++match)->flags & MATCH_SKIP);
		} else {
#if defined(__SSE2__) || defined(__MIC__)
			if (i == 1)
				NEXT_STATE(xsi, 1)
#endif
			do {
				x = xsi;
				NEXT_STATE(xsi, i + 1)
				NEXT_STATE(xs, M + i)
				i++;
			} while ((++match)->flags & MATCH_SKIP);
		}

		x = (((x & 0x80000000) | (xsi & 0x7fffffff)) >> 1) ^ xs ^
		    ((((version == PHP_521) ? x : xsi) & 1) * 0x9908b0df);
		x ^= x >> 11;
		x ^= (x << 7) & 0x9d2c5680;
		x ^= (x << 15) & 0xefc60000;
		x ^= x >> 18;

		if (match->flags & MATCH_FULL)
			x >>= 1;
	}

	return -1;
}

static void print_guess(uint32_t seed, uint64_t *found, version_t version)
{
	if (version == PHP_LEGACY)
		seed <<= 1;

#ifdef _OPENMP
#pragma omp critical
#endif
	do {
		if (!*found)
		    putc('\n', stderr);
		printf("seed = 0x%08x = %u (PHP %s)\n",
		    seed, seed, versions[version]);
		(*found)++;
	} while (version == PHP_LEGACY && !(seed++ & 1));
}

#if defined(__SSE2__) || defined(__MIC__)
#define COMPARE(x, xM, seed) \
	if (!diff((x), (xM), (seed), match, version)) \
		print_guess((seed), &found, version);
#else
#define COMPARE(x, x1, xM, seed) \
	if (!diff((x), (x1), (xM), match, version)) \
		print_guess((seed), &found, version);
#endif

#if defined(__MIC__) || defined(__AVX512F__)
#define P 7
#elif defined(__AVX2__)
#define P 6
#else
#define P 5
#endif

static uint64_t crack_range(int32_t start, int32_t end,
    const match_t *match, version_t flavor)
{
	uint64_t found = 0;
	int32_t base; /* signed type for OpenMP 2.5 compatibility */
#if defined(__SSE4_1__) || defined(__MIC__)
	vtype vvalue = _mm_set1_epi32(match->mmin);
#endif
#if defined(__SSE2__) || defined(__MIC__)
	vtype seed_and_0x80000000, seed_shr_30;
#else
	uint32_t seed_and_0x80000000, seed_shr_30;
#endif

	assert((start >> (30 - P)) == ((end - 1) >> (30 - P)));

	{
		uint32_t seed = (uint32_t)start << (P + (flavor == PHP_LEGACY));
#if defined(__SSE2__) || defined(__MIC__)
		vtype vseed = _mm_set1_epi32(seed);
		const vtype c0x80000000 = _mm_set1_epi32(0x80000000);
		seed_and_0x80000000 = _mm_and_si128(vseed, c0x80000000);
		seed_shr_30 = _mm_srli_epi32(vseed, 30);
#else
		seed_and_0x80000000 = seed & 0x80000000;
		seed_shr_30 = seed >> 30;
#endif
	}

#ifdef _OPENMP
#if defined(__SSE4_1__) || defined(__MIC__)
#pragma omp parallel for default(none) private(base) shared(match, flavor, start, end, found, seed_and_0x80000000, seed_shr_30, vvalue)
#elif defined(__SSE2__)
#pragma omp parallel for default(none) private(base) shared(match, flavor, start, end, found, seed_and_0x80000000, seed_shr_30)
#else
#pragma omp parallel for default(none) private(base) shared(match, flavor, start, end, found, seed_and_0x80000000, seed_shr_30)
#endif
#endif
	for (base = start; base < end; base++) {
		uint32_t seed = (uint32_t)base << P;
#if defined(__SSE2__) || defined(__MIC__)
		typedef struct {
			vtype a, b, c, d, e, f, g, h;
		} atype;
		atype xM, x = {}, x710 = {};
		/* Hint to compiler not to waste registers */
		volatile atype x1;
		const vtype cone = _mm_set1_epi32(1);
		vtype vseed = _mm_set1_epi32(seed);
		version_t version;

#define DO(which, add) \
	xM.which = _mm_add_epi32(xM.a, _mm_set1_epi32(add));
#if defined(__MIC__) || defined(__AVX512F__)
		xM.a = _mm512_add_epi32(vseed, _mm512_set_epi32(
		    0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30));
		DO(b, 1) DO(c, 32) DO(d, 33)
		DO(e, 64) DO(f, 65) DO(g, 96) DO(h, 97)
#elif defined(__AVX2__)
		xM.a = _mm256_add_epi32(vseed, _mm256_set_epi32(
		    0, 2, 4, 6, 8, 10, 12, 14));
		DO(b, 1) DO(c, 16) DO(d, 17)
		DO(e, 32) DO(f, 33) DO(g, 48) DO(h, 49)
#else
		xM.a = _mm_add_epi32(vseed, _mm_set_epi32(0, 2, 4, 6));
		DO(b, 1) DO(c, 8) DO(d, 9)
		DO(e, 16) DO(f, 17) DO(g, 24) DO(h, 25)
#endif
#undef DO

#define DO_ALL \
	DO(x.a, x1.a, xM.a) \
	DO(x.b, x1.b, xM.b) \
	DO(x.c, x1.c, xM.c) \
	DO(x.d, x1.d, xM.d) \
	DO(x.e, x1.e, xM.e) \
	DO(x.f, x1.f, xM.f) \
	DO(x.g, x1.g, xM.g) \
	DO(x.h, x1.h, xM.h)

		if (flavor == PHP_LEGACY) {
			const vtype c69069 = _mm_set1_epi32(69069);
			const vtype c69069to396 = _mm_set1_epi32(0x4396a0b1);

#define DO(x, x1, xM) \
	xM = _mm_add_epi32(_mm_add_epi32(xM, xM), cone); \
	x1 = xM = _mm_mullo_epi32(c69069, xM); \
	xM = _mm_mullo_epi32(c69069to396, xM);
			DO_ALL
#undef DO
		} else {
			const vtype cmul = _mm_set1_epi32(1812433253U);
			vtype vi = _mm_add_epi32(cone, cone);
			unsigned int n = (M - 1) / 22;

#define DO(x, x1, xM) \
	x1 = xM = _mm_macc_epi32(cmul, _mm_xor_si128(xM, seed_shr_30), cone);
			DO_ALL
#undef DO

			do {
#define DO(x, x1, xM) \
	xM = _mm_macc_epi32(cmul, _mm_xor_si128(xM, _mm_srli_epi32(xM, 30)), vi);
#define DO_ALLI \
	DO_ALL \
	vi = _mm_add_epi32(vi, cone);
				DO_ALLI DO_ALLI DO_ALLI DO_ALLI DO_ALLI DO_ALLI
				DO_ALLI DO_ALLI DO_ALLI DO_ALLI DO_ALLI DO_ALLI
				DO_ALLI DO_ALLI DO_ALLI DO_ALLI DO_ALLI DO_ALLI
				DO_ALLI DO_ALLI DO_ALLI DO_ALLI
#undef DO_ALLI
#undef DO
			} while (--n);
		}

		version = flavor;

		if (!(match->flags & MATCH_SKIP)) {
			const vtype c0x7fffffff = _mm_set1_epi32(0x7fffffff);
			const vtype c0x9908b0df = _mm_set1_epi32(0x9908b0df);

#define DO(x, x1, xM) \
	x = _mm_xor_si128(xM, _mm_srli_epi32(_mm_or_si128(seed_and_0x80000000, \
	    _mm_and_si128(x1, c0x7fffffff)), 1));
			DO_ALL
#undef DO

#define DO(xout, xin, x1) \
	xout = _mm_xor_si128(xin, _mm_mullo_epi32(c0x9908b0df, \
	    _mm_and_si128(x1, cone)));
			DO(x710.a, x.a, x1.a)
			DO(x710.b, x.b, x1.b)
			DO(x710.c, x.c, x1.c)
			DO(x710.d, x.d, x1.d)
			DO(x710.e, x.e, x1.e)
			DO(x710.f, x.f, x1.f)
			DO(x710.g, x.g, x1.g)
			DO(x710.h, x.h, x1.h)
#undef DO

			if (version == PHP_521) {
#define DO(x) \
	x = _mm_xor_si128(x, c0x9908b0df);
				DO(x.b)
				DO(x.d)
				DO(x.f)
				DO(x.h)
#undef DO
			} else
				x = x710;
		}

		do {
			uint32_t maybe = 1;

			if (!(match->flags & MATCH_SKIP)) {
				const vtype c0x9d2c5680 = _mm_set1_epi32(0x9d2c5680);
				const vtype c0xefc60000 = _mm_set1_epi32(0xefc60000);

#define DO(x, x1, xM) \
	x = _mm_xor_si128(x, _mm_srli_epi32(x, 11));
				DO_ALL
#undef DO

#define DO_SC(x, s, c) \
	x = _mm_xor_si128(x, _mm_and_si128(_mm_slli_epi32(x, s), c));
#define DO(x, x1, xM) \
	DO_SC(x, 7, c0x9d2c5680) \
	DO_SC(x, 15, c0xefc60000)
				DO_ALL
#undef DO
#undef DO_SC

#define DO(x, x1, xM) \
	x = _mm_xor_si128(x, _mm_srli_epi32(x, 18));
				DO_ALL
#undef DO

				if (match->flags & MATCH_FULL) {
#define DO(x, x1, xM) \
	x = _mm_srli_epi32(x, 1);
					DO_ALL
#undef DO
				}
			}

#if defined(__SSE4_1__) || defined(__MIC__)
			if (match->flags & MATCH_PURE) {
#if defined(__MIC__) || defined(__AVX512F__)
				maybe = _mm512_cmpeq_epi32_mask(x.a, vvalue) |
				    _mm512_cmpeq_epi32_mask(x.b, vvalue) |
				    _mm512_cmpeq_epi32_mask(x.c, vvalue) |
				    _mm512_cmpeq_epi32_mask(x.d, vvalue) |
				    _mm512_cmpeq_epi32_mask(x.e, vvalue) |
				    _mm512_cmpeq_epi32_mask(x.f, vvalue) |
				    _mm512_cmpeq_epi32_mask(x.g, vvalue) |
				    _mm512_cmpeq_epi32_mask(x.h, vvalue);
#else
				vtype amask = _mm_cmpeq_epi32(x.a, vvalue);
				vtype bmask = _mm_cmpeq_epi32(x.b, vvalue);
				vtype cmask = _mm_cmpeq_epi32(x.c, vvalue);
				vtype dmask = _mm_cmpeq_epi32(x.d, vvalue);
				vtype emask = _mm_cmpeq_epi32(x.e, vvalue);
				vtype fmask = _mm_cmpeq_epi32(x.f, vvalue);
				vtype gmask = _mm_cmpeq_epi32(x.g, vvalue);
				vtype hmask = _mm_cmpeq_epi32(x.h, vvalue);
				maybe = !(_mm_testz_si128(amask, amask) &&
				    _mm_testz_si128(bmask, bmask) &&
				    _mm_testz_si128(cmask, cmask) &&
				    _mm_testz_si128(dmask, dmask) &&
				    _mm_testz_si128(emask, emask) &&
				    _mm_testz_si128(fmask, fmask) &&
				    _mm_testz_si128(gmask, gmask) &&
				    _mm_testz_si128(hmask, hmask));
#endif
			}
#endif

			if (maybe) {
				unsigned int i;
				uint32_t iseed;
				typedef union {
					atype v;
					uint32_t s[8][sizeof(vtype) / 4];
				} utype;
				utype u;
				/* Hint to compiler not to waste registers */
				volatile utype uM;
				u.v = x;
				uM.v = xM;
#if defined(__MIC__) || defined(__AVX512F__)
				for (i = 0, iseed = seed; i < 8; i++, iseed += 32) {
					unsigned int j, k;
					for (j = 0, k = 30; j < 16; j++, k -= 2) {
						COMPARE(u.s[i][j], uM.s[i][j],
						    iseed + k)
					}
					i++;
					for (j = 0, k = 31; j < 16; j++, k -= 2) {
						COMPARE(u.s[i][j], uM.s[i][j],
						    iseed + k)
					}
				}
#elif defined(__AVX2__)
				for (i = 0, iseed = seed; i < 8; i++, iseed += 16) {
					unsigned int j, k;
					for (j = 0, k = 14; j < 8; j++, k -= 2) {
						COMPARE(u.s[i][j], uM.s[i][j],
						    iseed + k)
					}
					i++;
					for (j = 0, k = 15; j < 8; j++, k -= 2) {
						COMPARE(u.s[i][j], uM.s[i][j],
						    iseed + k)
					}
				}
#else
				for (i = 0, iseed = seed; i < 8; i++, iseed += 8) {
					COMPARE(u.s[i][0], uM.s[i][0], iseed + 6)
					COMPARE(u.s[i][1], uM.s[i][1], iseed + 4)
					COMPARE(u.s[i][2], uM.s[i][2], iseed + 2)
					COMPARE(u.s[i][3], uM.s[i][3], iseed)
					i++;
					COMPARE(u.s[i][0], uM.s[i][0], iseed + 7)
					COMPARE(u.s[i][1], uM.s[i][1], iseed + 5)
					COMPARE(u.s[i][2], uM.s[i][2], iseed + 3)
					COMPARE(u.s[i][3], uM.s[i][3], iseed + 1)
				}
#endif
				/* Hint to compiler not to spill xM above */
				xM = uM.v;
			}

			if (version != PHP_521)
				break;
			version = PHP_710;
			x = x710;
		} while (1);
#else
		typedef struct {
			uint32_t a, b, c, d;
		} atype;
		atype x = {}, x710 = {};
		do {
			atype x1, xM;
			version_t version;
			unsigned int i;

			xM.a = seed;
			xM.b = seed + 1;
			xM.c = seed + 2;
			xM.d = seed + 3;

#define DO_ALL \
	DO(x.a, x1.a, xM.a) \
	DO(x.b, x1.b, xM.b) \
	DO(x.c, x1.c, xM.c) \
	DO(x.d, x1.d, xM.d)

			if (flavor == PHP_LEGACY) {
#define DO(x, x1, xM) \
	xM += xM + 1; \
	x1 = xM *= 69069; \
	xM *= 0x4396a0b1;
				DO_ALL
#undef DO
			} else {
#define DO(x, x1, xM) \
	x1 = xM = 1812433253U * (xM ^ seed_shr_30) + 1;
				DO_ALL
#undef DO

				for (i = 2; i <= M; i++) {
#define DO(x, x1, xM) \
	NEXT_STATE(xM, i)
					DO_ALL
#undef DO
				}
			}

			version = flavor;

			if (!(match->flags & MATCH_SKIP)) {
#define DO(x, x1, xM) \
	x = ((seed_and_0x80000000 | (x1 & 0x7fffffff)) >> 1) ^ xM;
				DO_ALL
#undef DO

#define DO(xout, xin, x1) \
	xout = xin ^ ((x1 & 1) * 0x9908b0df);
				DO(x710.a, x.a, x1.a)
				DO(x710.b, x.b, x1.b)
				DO(x710.c, x.c, x1.c)
				DO(x710.d, x.d, x1.d)
#undef DO

				if (version == PHP_521) {
					x.b ^= 0x9908b0df;
					x.d ^= 0x9908b0df;
				} else
					x = x710;
			}

			do {
				if (!(match->flags & MATCH_SKIP)) {
#define DO(x, x1, xM) \
	x ^= x >> 11; \
	x ^= (x << 7) & 0x9d2c5680; \
	x ^= (x << 15) & 0xefc60000; \
	x ^= x >> 18;
					DO_ALL
#undef DO

					if (match->flags & MATCH_FULL) {
#define DO(x, x1, xM) \
	x >>= 1;
						DO_ALL
#undef DO
					}
				}

				COMPARE(x.a, x1.a, xM.a, seed)
				COMPARE(x.b, x1.b, xM.b, seed + 1)
				COMPARE(x.c, x1.c, xM.c, seed + 2)
				COMPARE(x.d, x1.d, xM.d, seed + 3)

				if (version != PHP_521)
					break;
				version = PHP_710;
				x = x710;
			} while (1);

			seed += 4;
		} while (seed & ((1 << P) - 1));
#endif
	}

	return found;
}

static uint64_t crack(const match_t *match)
{
	uint64_t found = 0, recent = 0;
	uint32_t base, top;
#if defined(__MIC__) || defined(__AVX512F__)
	const uint32_t step = 0x10000000 >> P;
#else
	const uint32_t step = 0x2000000 >> P;
#endif
	version_t flavor;
	long clk_tck;
	clock_t start_time;
	struct tms tms;

	flavor = PHP_LEGACY;
	do {
		unsigned int shift = (flavor == PHP_LEGACY);

		fprintf(stderr, "Version: %s\n", flavors[flavor]);

		clk_tck = sysconf(_SC_CLK_TCK);
		start_time = times(&tms);

		top = 0x40000000 >> (P - 2 + shift);
		for (base = 0; base < top; base += step) {
			uint32_t start = base << (P + shift);
			uint32_t next = (base + step) << (P + shift);
			clock_t running_time = times(&tms) - start_time;
			fprintf(stderr,
			    "\rFound %llu, trying 0x%08x - 0x%08x, "
			    "speed %.1f Mseeds/s ",
			    (unsigned long long)found, start, next - 1,
			    (double)start * clk_tck /
			    (running_time ? running_time * 1e6 : 1e6));

			recent = crack_range(base, base + step, match, flavor);
			found += recent;
		}

		if (flavor == PHP_MODERN)
			break;
		flavor = PHP_MODERN;
		if (!recent)
			putc('\n', stderr);
	} while (1);

	return found;
}

#undef P

static int32_t parse_int(const char *s)
{
	unsigned long ulvalue;
	uint32_t uvalue;
	char *error;

	errno = 0;
	uvalue = ulvalue = strtoul(s, &error, 10);
	if (!errno && !*error &&
	    *s >= '0' && *s <= '9' &&
	    uvalue == ulvalue && uvalue <= 0x7fffffff)
		return uvalue;

	return -1;
}

static void parse(int argc, char **argv, match_t *match, unsigned int nmatch)
{
	const char *prog = argv[0] ? argv[0] : "php_mt_seed";
	int ok = 0;
	match_t *first = match, *last = match;

	argc--;
	argv++;

	while (nmatch && argc > 0) {
		int32_t value = parse_int(argv[0]);
		ok = value >= 0;

		match->flags = MATCH_PURE | MATCH_FULL;
		match->mmin = match->mmax = value;
		match->rmin = 0; match->rspan = 0x80000000U;

		if (argc >= 2) {
			value = parse_int(argv[1]);
			ok &= value >= match->mmin;
			if (value != match->mmin)
				match->flags &= ~MATCH_PURE;
			match->mmax = value;
		}

		if (argc == 3) {
			ok = 0;
			break;
		}

		if (argc >= 4) {
			value = parse_int(argv[2]);
			ok &= value >= 0 && value <= match->mmax;
			if (value != 0)
				match->flags &= ~(MATCH_PURE | MATCH_FULL);
			match->rmin = value;

			value = parse_int(argv[3]);
			ok &= value >= match->rmin && value >= match->mmin;
			if (value != 0x7fffffff)
				match->flags &= ~(MATCH_PURE | MATCH_FULL);
			if (match->mmin == match->rmin &&
			    match->mmax == value)
				match->flags |= MATCH_SKIP;
			match->rspan = (uint32_t)value - match->rmin + 1;
		}

		if (!(match->flags & MATCH_SKIP))
			last = match;

		nmatch--;
		match++;
		if (!ok)
			break;
		if (argc <= 4) {
			argc = 0;
			break;
		}
		argc -= 4;
		argv += 4;
	}

	if (!ok || (!nmatch && argc > 0) || (last->flags & MATCH_SKIP)) {
		printf("Usage: %s VALUE_OR_MATCH_MIN"
		    " [MATCH_MAX [RANGE_MIN RANGE_MAX]] ...\n", prog);
		exit(1);
	}

	last->flags |= MATCH_LAST;

	printf("Pattern:");
	match = first;
	do {
		if (match->flags & MATCH_SKIP)
			printf(" SKIP");
		else if (match->flags & MATCH_PURE)
			printf(" EXACT");
		else if (match->flags & MATCH_FULL)
			printf(" RANGE");
		else if (match->mmin == match->mmax)
			printf(" EXACT-FROM-%u", match->rspan);
		else
			printf(" RANGE-FROM-%u", match->rspan);
	} while (match++ != last);
	putchar('\n');
}

int main(int argc, char **argv)
{
	match_t match[N - M];

	parse(argc, argv, match, sizeof(match) / sizeof(match[0]));

	printf("\nFound %llu\n", (unsigned long long)crack(match));

	return 0;
}
