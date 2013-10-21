/*
 * Copyright (c) 2012,2013 Solar Designer <solar at openwall.com>
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

#ifdef __MIC__
#include <immintrin.h>
typedef __m512i vtype;
/* hack */
#define _mm_set1_epi32(x) _mm512_set1_epi32(x)
#define _mm_add_epi32(x, y) _mm512_add_epi32(x, y)
#define _mm_macc_epi32(x, y, z) _mm512_fmadd_epi32(x, y, z)
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
#define _mm_macc_epi32(x, y, z) \
	_mm256_add_epi32(_mm256_mullo_epi32((x), (y)), (z))
#define _mm_slli_epi32(x, i) _mm256_slli_epi32(x, i)
#define _mm_srli_epi32(x, i) _mm256_srli_epi32(x, i)
#define _mm_and_si128(x, y) _mm256_and_si256(x, y)
#define _mm_or_si128(x, y) _mm256_or_si256(x, y)
#define _mm_xor_si128(x, y) _mm256_xor_si256(x, y)
#define _mm_cmpeq_epi32(x, y) _mm256_cmpeq_epi32(x, y)
#define _mm_testz_si128(x, y) _mm256_testz_si256(x, y)
#elif defined(__SSE4_1__)
#include <emmintrin.h>
#include <smmintrin.h>
typedef __m128i vtype;
#ifdef __XOP__
#include <x86intrin.h>
#else
#define _mm_macc_epi32(a, b, c) \
	_mm_add_epi32(_mm_mullo_epi32((a), (b)), (c))
#ifdef __AVX__
#warning XOP and AVX2 are not enabled. Try gcc -mxop (on AMD Bulldozer or newer) or -mavx2 (on Intel Haswell or newer).
#else
#warning AVX* and XOP are not enabled. Try gcc -mxop (on AMD Bulldozer or newer), -mavx (on Intel Sandy Bridge or newer), or -mavx2 (on Intel Haswell or newer).
#endif
#endif
#else
/*
 * We require at least SSE4.1 for vectorization because we use
 * SSE4.1's _mm_mullo_epi32() or XOP's _mm_macc_epi32(), as well as SSE4.1's
 * _mm_testz_si128().
 */
#warning SSE4.1 not enabled, will use non-vectorized code. Try gcc -msse4 (only on capable CPUs).
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
	double rspan;
} match_t;

#define NEXT_STATE(x, i) \
	(x) = 1812433253U * ((x) ^ ((x) >> 30)) + (i);

#if defined(__SSE4_1__) || defined(__AVX2__) || defined(__MIC__)
static inline int diff(uint32_t x, uint32_t xs, uint32_t seed,
    const match_t *match)
#else
static inline int diff(uint32_t x, uint32_t x1, uint32_t xs,
    const match_t *match)
#endif
{
#if defined(__SSE4_1__) || defined(__AVX2__) || defined(__MIC__)
	uint32_t xsi = seed;
#else
	uint32_t xsi = x1;
#endif
	unsigned int i = 1;

	while (1) {
		if (match->flags & MATCH_PURE) {
			if (x != match->mmin)
				break;
		} else {
			int32_t xr;
			if (match->flags & MATCH_FULL)
				xr = x;
			else
				xr = match->rmin +
				    (int32_t)(match->rspan * (x) /
				    (0x7fffffff + 1.0));
			if (xr < match->mmin || xr > match->mmax)
				break;
		}

		if (match->flags & MATCH_LAST)
			return 0;

#if defined(__SSE4_1__) || defined(__AVX2__) || defined(__MIC__)
		if (i == 1)
			NEXT_STATE(xsi, 1)
#endif
		x = xsi;
		NEXT_STATE(xsi, i + 1)
		NEXT_STATE(xs, M + i)
		i++;
		x = (((x & 0x80000000) | (xsi & 0x7fffffff)) >> 1) ^ xs ^
		    ((x & 1) ? 0x9908b0df : 0);
		x ^= (x >> 11);
		x ^= (x << 7) & 0x9d2c5680;
		x ^= (x << 15) & 0xefc60000;
		x = (x ^ (x >> 18)) >> 1;

		match++;
	}

	return -1;
}

static void print_guess(uint32_t seed, unsigned int *found)
{
#ifdef _OPENMP
#pragma omp critical
#endif
	{
		printf("%sseed = %u\n", *found ? "" : "\n", seed);
		(*found)++;
	}
}

#if defined(__SSE4_1__) || defined(__AVX2__) || defined(__MIC__)
#define COMPARE(x, xM, seed) \
	if (!diff((x), (xM), (seed), match)) \
		print_guess((seed), &found);
#else
#define COMPARE(x, x1, xM, seed) \
	if (!diff((x), (x1), (xM), match)) \
		print_guess((seed), &found);
#endif

#ifdef __MIC__
#define P 7
#elif defined(__AVX2__)
#define P 6
#else
#define P 5
#endif

static unsigned int crack_range(int32_t start, int32_t end,
    const match_t *match)
{
	unsigned int found = 0;
	int32_t base; /* signed type for OpenMP 2.5 compatibility */
#if defined(__SSE4_1__) || defined(__AVX2__) || defined(__MIC__)
	vtype vvalue, v1, seed_and_0x80000000, seed_shr_30;
#else
	uint32_t seed_and_0x80000000, seed_shr_30;
#endif

	assert((start >> (30 - P)) == ((end - 1) >> (30 - P)));

#if defined(__SSE4_1__) || defined(__AVX2__) || defined(__MIC__)
	if (match->flags & MATCH_PURE)
		vvalue = _mm_set1_epi32(match->mmin);
	v1 = _mm_set1_epi32(1);
#endif

	{
		uint32_t seed = (uint32_t)start << P;
#if defined(__SSE4_1__) || defined(__AVX2__) || defined(__MIC__)
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
#if defined(__SSE4_1__) || defined(__AVX2__) || defined(__MIC__)
#pragma omp parallel for default(none) private(base) shared(match, start, end, found, v1, seed_and_0x80000000, seed_shr_30, vvalue)
#else
#pragma omp parallel for default(none) private(base) shared(match, start, end, found, seed_and_0x80000000, seed_shr_30)
#endif
#endif
	for (base = start; base < end; base++) {
		uint32_t seed = (uint32_t)base << P;
#if defined(__SSE4_1__) || defined(__AVX2__) || defined(__MIC__)
		const vtype cmul = _mm_set1_epi32(1812433253U);
		const vtype c0x7fffffff = _mm_set1_epi32(0x7fffffff);
		const vtype c0x9d2c5680 = _mm_set1_epi32(0x9d2c5680);
		const vtype c0xefc60000 = _mm_set1_epi32(0xefc60000);
		const vtype c0x9908b0df = _mm_set1_epi32(0x9908b0df);
		vtype vseed = _mm_set1_epi32(seed);
		vtype a, b, c, d, e, f, g, h;
		vtype a1, b1, c1, d1, e1, f1, g1, h1;
		vtype aM, bM, cM, dM, eM, fM, gM, hM;

#ifdef __MIC__
		aM = _mm512_add_epi32(vseed, _mm512_set_epi32(
		    0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30));
		bM = _mm512_add_epi32(aM, _mm512_set1_epi32(1));
		cM = _mm512_add_epi32(aM, _mm512_set1_epi32(32));
		dM = _mm512_add_epi32(aM, _mm512_set1_epi32(33));
		eM = _mm512_add_epi32(aM, _mm512_set1_epi32(64));
		fM = _mm512_add_epi32(aM, _mm512_set1_epi32(65));
		gM = _mm512_add_epi32(aM, _mm512_set1_epi32(96));
		hM = _mm512_add_epi32(aM, _mm512_set1_epi32(97));
#elif defined(__AVX2__)
		aM = _mm256_add_epi32(vseed, _mm256_set_epi32(
		    0, 2, 4, 6, 8, 10, 12, 14));
		bM = _mm256_add_epi32(aM, _mm256_set1_epi32(1));
		cM = _mm256_add_epi32(aM, _mm256_set1_epi32(16));
		dM = _mm256_add_epi32(aM, _mm256_set1_epi32(17));
		eM = _mm256_add_epi32(aM, _mm256_set1_epi32(32));
		fM = _mm256_add_epi32(aM, _mm256_set1_epi32(33));
		gM = _mm256_add_epi32(aM, _mm256_set1_epi32(48));
		hM = _mm256_add_epi32(aM, _mm256_set1_epi32(49));
#else
		aM = _mm_add_epi32(vseed, _mm_set_epi32(0, 2, 4, 6));
		bM = _mm_add_epi32(vseed, _mm_set_epi32(1, 3, 5, 7));
		cM = _mm_add_epi32(vseed, _mm_set_epi32(8, 10, 12, 14));
		dM = _mm_add_epi32(vseed, _mm_set_epi32(9, 11, 13, 15));
		eM = _mm_add_epi32(vseed, _mm_set_epi32(16, 18, 20, 22));
		fM = _mm_add_epi32(vseed, _mm_set_epi32(17, 19, 21, 23));
		gM = _mm_add_epi32(vseed, _mm_set_epi32(24, 26, 28, 30));
		hM = _mm_add_epi32(vseed, _mm_set_epi32(25, 27, 29, 31));
#endif

		{
			unsigned int n = (M - 1) / (6 * 6);
			vtype vi = _mm_add_epi32(v1, v1);

#define DO(x, x1) \
	x = x1 = _mm_macc_epi32(cmul, _mm_xor_si128(x, seed_shr_30), v1);
			DO(aM, a1)
			DO(bM, b1)
			DO(cM, c1)
			DO(dM, d1)
			DO(eM, e1)
			DO(fM, f1)
			DO(gM, g1)
			DO(hM, h1)
#undef DO

			do {
				vtype vi1;
#define DO(x, vi) \
	x = _mm_macc_epi32(cmul, _mm_xor_si128(x, _mm_srli_epi32(x, 30)), vi);
#define DO_ALL \
		vi1 = _mm_add_epi32(vi, v1); \
		DO(aM, vi) DO(bM, vi) DO(cM, vi) DO(dM, vi) \
		DO(eM, vi) DO(fM, vi) DO(gM, vi) DO(hM, vi) \
		vi = _mm_add_epi32(vi1, v1); \
		DO(aM, vi1) DO(bM, vi1) DO(cM, vi1) DO(dM, vi1) \
		DO(eM, vi1) DO(fM, vi1) DO(gM, vi1) DO(hM, vi1)
				DO_ALL DO_ALL DO_ALL DO_ALL DO_ALL DO_ALL
				DO_ALL DO_ALL DO_ALL DO_ALL DO_ALL DO_ALL
				DO_ALL DO_ALL DO_ALL DO_ALL DO_ALL DO_ALL
#undef DO_ALL
#undef DO
			} while (--n);
		}

#define DO(x, x1, xM) \
	x = _mm_xor_si128(xM, _mm_srli_epi32(_mm_or_si128(seed_and_0x80000000, \
	    _mm_and_si128(x1, c0x7fffffff)), 1));
		DO(a, a1, aM)
		DO(b, b1, bM)
		DO(c, c1, cM)
		DO(d, d1, dM)
		DO(e, e1, eM)
		DO(f, f1, fM)
		DO(g, g1, gM)
		DO(h, h1, hM)
#undef DO

#define DO(x) \
		x = _mm_xor_si128(x, c0x9908b0df);
		DO(b)
		DO(d)
		DO(f)
		DO(h)
#undef DO

#define DO(x) \
	x = _mm_xor_si128(x, _mm_srli_epi32(x, 11));
		DO(a)
		DO(b)
		DO(c)
		DO(d)
		DO(e)
		DO(f)
		DO(g)
		DO(h)
#undef DO
#define DO(x, s, c) \
	x = _mm_xor_si128(x, _mm_and_si128(_mm_slli_epi32(x, s), c));
		DO(a, 7, c0x9d2c5680)
		DO(b, 7, c0x9d2c5680)
		DO(c, 7, c0x9d2c5680)
		DO(d, 7, c0x9d2c5680)
		DO(e, 7, c0x9d2c5680)
		DO(f, 7, c0x9d2c5680)
		DO(g, 7, c0x9d2c5680)
		DO(h, 7, c0x9d2c5680)
		DO(a, 15, c0xefc60000)
		DO(b, 15, c0xefc60000)
		DO(c, 15, c0xefc60000)
		DO(d, 15, c0xefc60000)
		DO(e, 15, c0xefc60000)
		DO(f, 15, c0xefc60000)
		DO(g, 15, c0xefc60000)
		DO(h, 15, c0xefc60000)
#undef DO
#define DO(x) \
	x = _mm_srli_epi32(_mm_xor_si128(x, _mm_srli_epi32(x, 18)), 1);
		DO(a)
		DO(b)
		DO(c)
		DO(d)
		DO(e)
		DO(f)
		DO(g)
		DO(h)
#undef DO

		if (match->flags & MATCH_PURE) {
#ifdef __MIC__
			if ((_mm512_cmpeq_epi32_mask(a, vvalue) |
			    _mm512_cmpeq_epi32_mask(b, vvalue) |
			    _mm512_cmpeq_epi32_mask(c, vvalue) |
			    _mm512_cmpeq_epi32_mask(d, vvalue) |
			    _mm512_cmpeq_epi32_mask(e, vvalue) |
			    _mm512_cmpeq_epi32_mask(f, vvalue) |
			    _mm512_cmpeq_epi32_mask(g, vvalue) |
			    _mm512_cmpeq_epi32_mask(h, vvalue)) == 0)
				continue;
#else
			vtype amask = _mm_cmpeq_epi32(a, vvalue);
			vtype bmask = _mm_cmpeq_epi32(b, vvalue);
			vtype cmask = _mm_cmpeq_epi32(c, vvalue);
			vtype dmask = _mm_cmpeq_epi32(d, vvalue);
			vtype emask = _mm_cmpeq_epi32(e, vvalue);
			vtype fmask = _mm_cmpeq_epi32(f, vvalue);
			vtype gmask = _mm_cmpeq_epi32(g, vvalue);
			vtype hmask = _mm_cmpeq_epi32(h, vvalue);
			if (_mm_testz_si128(amask, amask) &&
			    _mm_testz_si128(bmask, bmask) &&
			    _mm_testz_si128(cmask, cmask) &&
			    _mm_testz_si128(dmask, dmask) &&
			    _mm_testz_si128(emask, emask) &&
			    _mm_testz_si128(fmask, fmask) &&
			    _mm_testz_si128(gmask, gmask) &&
			    _mm_testz_si128(hmask, hmask))
				continue;
#endif
		}

		{
			unsigned int i;
			uint32_t iseed;
#ifdef __ICC
			volatile
#endif
			union {
				vtype v;
				uint32_t s[sizeof(vtype) / 4];
			} u[8], uM[8];
			u[0].v = a;
			u[1].v = b;
			u[2].v = c;
			u[3].v = d;
			u[4].v = e;
			u[5].v = f;
			u[6].v = g;
			u[7].v = h;
			uM[0].v = aM;
			uM[1].v = bM;
			uM[2].v = cM;
			uM[3].v = dM;
			uM[4].v = eM;
			uM[5].v = fM;
			uM[6].v = gM;
			uM[7].v = hM;
#ifdef __MIC__
			for (i = 0, iseed = seed; i < 8; i++, iseed += 32) {
				unsigned int j, k;
				for (j = 0, k = 30; j < 16; j++, k -= 2) {
					COMPARE(u[i].s[j], uM[i].s[j],
					    iseed + k)
				}
				i++;
				for (j = 0, k = 31; j < 16; j++, k -= 2) {
					COMPARE(u[i].s[j], uM[i].s[j],
					    iseed + k)
				}
			}
#elif defined(__AVX2__)
			for (i = 0, iseed = seed; i < 8; i++, iseed += 16) {
				unsigned int j, k;
				for (j = 0, k = 14; j < 8; j++, k -= 2) {
					COMPARE(u[i].s[j], uM[i].s[j],
					    iseed + k)
				}
				i++;
				for (j = 0, k = 15; j < 8; j++, k -= 2) {
					COMPARE(u[i].s[j], uM[i].s[j],
					    iseed + k)
				}
			}
#else
			for (i = 0, iseed = seed; i < 8; i++, iseed += 8) {
				COMPARE(u[i].s[0], uM[i].s[0], iseed + 6)
				COMPARE(u[i].s[1], uM[i].s[1], iseed + 4)
				COMPARE(u[i].s[2], uM[i].s[2], iseed + 2)
				COMPARE(u[i].s[3], uM[i].s[3], iseed)
				i++;
				COMPARE(u[i].s[0], uM[i].s[0], iseed + 7)
				COMPARE(u[i].s[1], uM[i].s[1], iseed + 5)
				COMPARE(u[i].s[2], uM[i].s[2], iseed + 3)
				COMPARE(u[i].s[3], uM[i].s[3], iseed + 1)
			}
#endif
		}
#else
		do {
			uint32_t a, b, c, d, a1, b1, c1, d1, aM, bM, cM, dM;
			unsigned int i;

#define DO(x, x1, seed) \
	x = x1 = 1812433253U * ((seed) ^ seed_shr_30) + 1;
			DO(aM, a1, seed)
			DO(bM, b1, seed + 1)
			DO(cM, c1, seed + 2)
			DO(dM, d1, seed + 3)
#undef DO
			for (i = 2; i <= M; i++) {
#define DO(x) \
	NEXT_STATE(x, i)
				DO(aM)
				DO(bM)
				DO(cM)
				DO(dM)
#undef DO
			}

#define DO(x, x1, xM) \
	x = ((seed_and_0x80000000 | (x1 & 0x7fffffff)) >> 1) ^ xM;
			DO(a, a1, aM)
			DO(b, b1, bM)
			DO(c, c1, cM)
			DO(d, d1, dM)
#undef DO

			b ^= 0x9908b0df;
			d ^= 0x9908b0df;

#define DO(x) \
	x ^= x >> 11;
			DO(a)
			DO(b)
			DO(c)
			DO(d)
#undef DO
#define DO(x, s, c) \
	x ^= (x << s) & c;
			DO(a, 7, 0x9d2c5680)
			DO(b, 7, 0x9d2c5680)
			DO(c, 7, 0x9d2c5680)
			DO(d, 7, 0x9d2c5680)
			DO(a, 15, 0xefc60000)
			DO(b, 15, 0xefc60000)
			DO(c, 15, 0xefc60000)
			DO(d, 15, 0xefc60000)
#undef DO
#define DO(x) \
	x = (x ^ (x >> 18)) >> 1;
			DO(a)
			DO(b)
			DO(c)
			DO(d)
#undef DO

			COMPARE(a, a1, aM, seed)
			COMPARE(b, b1, bM, seed + 1)
			COMPARE(c, c1, cM, seed + 2)
			COMPARE(d, d1, dM, seed + 3)

			seed += 4;
		} while (seed & ((1 << P) - 1));
#endif
	}

	return found;
}

static unsigned int crack(const match_t *match)
{
	unsigned int found = 0;
	uint32_t base;
#ifdef __MIC__
	const uint32_t step = 0x10000000 >> P;
#else
	const uint32_t step = 0x2000000 >> P;
#endif
	long clk_tck;
	clock_t start_time;
	struct tms tms;

	clk_tck = sysconf(_SC_CLK_TCK);
	start_time = times(&tms);

	for (base = 0; base < (0x40000000 >> (P - 2)); base += step) {
		uint32_t start = base << P, next = (base + step) << P;
		clock_t running_time = times(&tms) - start_time;
		fprintf(stderr,
		    "\rFound %u, trying %u - %u, speed %llu seeds per second ",
		    found, start, next - 1,
		    (unsigned long long)start * clk_tck /
		    (running_time ? running_time : 1));

		found += crack_range(base, base + step, match);

#if 0
		if (found)
			break;
#endif
	}

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
		match->rmin = 0; match->rspan = 0x7fffffff + 1.0;

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
			match->rspan = (double)value - match->rmin + 1.0;
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

	if (match - first > 1) {
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
				printf(" EXACT-FROM-%.0f", match->rspan);
			else
				printf(" RANGE-FROM-%.0f", match->rspan);
		} while (match++ != last);
		putchar('\n');
	}
}

int main(int argc, char **argv)
{
	match_t match[N - M + 1];

	parse(argc, argv, match, sizeof(match) / sizeof(match[0]));

	printf("\nFound %u\n", crack(match));

	return 0;
}
