/*
 * Copyright (c) 2012 Solar Designer <solar at openwall.com>
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

#ifdef __SSE4_1__
#include <emmintrin.h>
#include <smmintrin.h>

#ifdef __XOP__
#include <x86intrin.h>
#else
#define _mm_macc_epi32(a, b, c) \
	_mm_add_epi32(_mm_mullo_epi32((a), (b)), (c))
#ifdef __AVX__
#warning XOP not enabled, will not use fused multiply-add. Try gcc -mxop (only on AMD Bulldozer or newer).
#else
#warning AVX and XOP are not enabled, will not use fused multiply-add. Try gcc -mxop (only on AMD Bulldozer or newer) or -mavx (on Intel Sandy Bridge or newer).
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

static void print_guess(uint32_t seed, unsigned int *found)
{
#ifdef _OPENMP
#pragma omp critical
#endif
	{
		printf("\nseed = %u\n", seed);
		(*found)++;
	}
}

#define COMPARE(x, y) \
	if ((x) == value) \
		print_guess((y), &found);

#define P 5

static unsigned int crack_range(int32_t min, int32_t max, uint32_t value)
{
	unsigned int found = 0;
	int32_t base; /* signed type for OpenMP 2.5 compatibility */
#ifdef __SSE4_1__
	__m128i vvalue;
	__m128i vi[M], seed_and_0x80000000, seed_shr_30;
#else
	uint32_t seed_and_0x80000000, seed_shr_30;
#endif

	assert((min >> (30 - P)) == ((max - 1) >> (30 - P)));

#ifdef __SSE4_1__
	assert(P == 5);

	vvalue = _mm_set1_epi32(value);

	{
		unsigned int i;
		for (i = 1; i <= M; i++)
			vi[i - 1] = _mm_set1_epi32(i);
	}
#endif

	{
		uint32_t seed = (uint32_t)min << P;
#ifdef __SSE4_1__
		__m128i vseed = _mm_set1_epi32(seed);
		const __m128i c0x80000000 = _mm_set1_epi32(0x80000000);
		seed_and_0x80000000 = _mm_and_si128(vseed, c0x80000000);
		seed_shr_30 = _mm_srli_epi32(vseed, 30);
#else
		seed_and_0x80000000 = seed & 0x80000000;
		seed_shr_30 = seed >> 30;
#endif
	}

#ifdef _OPENMP
#ifdef __SSE4_1__
#pragma omp parallel for default(none) private(base) shared(value, min, max, found, vi, seed_and_0x80000000, seed_shr_30, vvalue)
#else
#pragma omp parallel for default(none) private(base) shared(value, min, max, found, seed_and_0x80000000, seed_shr_30)
#endif
#endif
	for (base = min; base < max; base++) {
		uint32_t seed = (uint32_t)base << P;
#ifdef __SSE4_1__
		const __m128i cmul = _mm_set1_epi32(1812433253U);
		const __m128i c0x7fffffff = _mm_set1_epi32(0x7fffffff);
		const __m128i c0x9d2c5680 = _mm_set1_epi32(0x9d2c5680);
		const __m128i c0xefc60000 = _mm_set1_epi32(0xefc60000);
		const __m128i c0x9908b0df = _mm_set1_epi32(0x9908b0df);
		__m128i a, b, c, d, e, f, g, h;
		__m128i a1, b1, c1, d1, e1, f1, g1, h1;
		unsigned int i;

		{
			__m128i vseed = _mm_set1_epi32(seed);
			a = _mm_add_epi32(vseed, _mm_set_epi32(0, 2, 4, 6));
			b = _mm_add_epi32(vseed, _mm_set_epi32(1, 3, 5, 7));
			c = _mm_add_epi32(vseed, _mm_set_epi32(8, 10, 12, 14));
			d = _mm_add_epi32(vseed, _mm_set_epi32(9, 11, 13, 15));
			e = _mm_add_epi32(vseed, _mm_set_epi32(16, 18, 20, 22));
			f = _mm_add_epi32(vseed, _mm_set_epi32(17, 19, 21, 23));
			g = _mm_add_epi32(vseed, _mm_set_epi32(24, 26, 28, 30));
			h = _mm_add_epi32(vseed, _mm_set_epi32(25, 27, 29, 31));
		}

		{
			__m128i vi0 = vi[0];
#define DO(x, x1) \
	x = x1 = _mm_macc_epi32(cmul, _mm_xor_si128(x, seed_shr_30), vi0);
			DO(a, a1)
			DO(b, b1)
			DO(c, c1)
			DO(d, d1)
			DO(e, e1)
			DO(f, f1)
			DO(g, g1)
			DO(h, h1)
#undef DO
		}

		for (i = 1; i < M; i++) {
			__m128i vii = vi[i];
#define DO(x) \
	x = _mm_macc_epi32(cmul, _mm_xor_si128(x, _mm_srli_epi32(x, 30)), vii);
			DO(a)
			DO(b)
			DO(c)
			DO(d)
			DO(e)
			DO(f)
			DO(g)
			DO(h)
#undef DO
		}

#define DO(x, x1) \
	x = _mm_xor_si128(x, _mm_srli_epi32(_mm_or_si128(seed_and_0x80000000, \
	    _mm_and_si128(x1, c0x7fffffff)), 1));
		DO(a, a1)
		DO(b, b1)
		DO(c, c1)
		DO(d, d1)
		DO(e, e1)
		DO(f, f1)
		DO(g, g1)
		DO(h, h1)
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

		{
			__m128i amask = _mm_cmpeq_epi32(a, vvalue);
			__m128i bmask = _mm_cmpeq_epi32(b, vvalue);
			__m128i cmask = _mm_cmpeq_epi32(c, vvalue);
			__m128i dmask = _mm_cmpeq_epi32(d, vvalue);
			__m128i emask = _mm_cmpeq_epi32(e, vvalue);
			__m128i fmask = _mm_cmpeq_epi32(f, vvalue);
			__m128i gmask = _mm_cmpeq_epi32(g, vvalue);
			__m128i hmask = _mm_cmpeq_epi32(h, vvalue);
			if (_mm_testz_si128(amask, amask) &&
			    _mm_testz_si128(bmask, bmask) &&
			    _mm_testz_si128(cmask, cmask) &&
			    _mm_testz_si128(dmask, dmask) &&
			    _mm_testz_si128(emask, emask) &&
			    _mm_testz_si128(fmask, fmask) &&
			    _mm_testz_si128(gmask, gmask) &&
			    _mm_testz_si128(hmask, hmask))
				continue;
		}

		{
			uint32_t iseed;
			union {
				__m128i v;
				uint32_t s[4];
			} u[8];
			u[0].v = a;
			u[1].v = b;
			u[2].v = c;
			u[3].v = d;
			u[4].v = e;
			u[5].v = f;
			u[6].v = g;
			u[7].v = h;
			for (i = 0, iseed = seed; i < 8; i++, iseed += 8) {
				COMPARE(u[i].s[0], iseed + 6)
				COMPARE(u[i].s[1], iseed + 4)
				COMPARE(u[i].s[2], iseed + 2)
				COMPARE(u[i++].s[3], iseed)
				COMPARE(u[i].s[0], iseed + 7)
				COMPARE(u[i].s[1], iseed + 5)
				COMPARE(u[i].s[2], iseed + 3)
				COMPARE(u[i].s[3], iseed + 1)
			}
		}
#else
		do {
			uint32_t a, b, c, d, a1, b1, c1, d1;
			unsigned int i;

#define DO(x, x1, seed) \
	x = x1 = 1812433253U * ((seed) ^ seed_shr_30) + 1;
			DO(a, a1, seed)
			DO(b, b1, seed + 1)
			DO(c, c1, seed + 2)
			DO(d, d1, seed + 3)
#undef DO
			for (i = 2; i <= M; i++) {
#define DO(x) \
	x = 1812433253U * (x ^ (x >> 30)) + i;
				DO(a)
				DO(b)
				DO(c)
				DO(d)
#undef DO
			}

#define DO(x, x1) \
	x ^= (seed_and_0x80000000 | (x1 & 0x7fffffffU)) >> 1;
			DO(a, a1)
			DO(b, b1)
			DO(c, c1)
			DO(d, d1)
#undef DO

			b ^= 0x9908b0dfU;
			d ^= 0x9908b0dfU;

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

			COMPARE(a, seed)
			COMPARE(b, seed + 1)
			COMPARE(c, seed + 2)
			COMPARE(d, seed + 3)

			seed += 4;
		} while (seed & ((1 << P) - 1));
#endif
	}

	return found;
}

static unsigned int crack(uint32_t value)
{
	unsigned int found = 0;
	uint32_t base;
	const uint32_t step = 0x2000000 >> P;
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

		found += crack_range(base, base + step, value);

#if 0
		if (found)
			break;
#endif
	}

	return found;
}

#undef P

static uint32_t parse(int argc, char **argv)
{
	int ok = 0;
	uint32_t value;

	if (argc == 2) {
		unsigned long ulvalue;
		char *error;

		errno = 0;
		value = ulvalue = strtoul(argv[1], &error, 10);
		ok = !errno && !*error &&
		    argv[1][0] >= '0' && argv[1][0] <= '9' &&
		    value == ulvalue && value <= 0x7fffffffU;
	}

	if (!ok) {
		printf("Usage: %s MT_RAND_VALUE\n",
		    argv[0] ? argv[0] : "php_mt_seed");
		exit(1);
	}

	return value;
}

int main(int argc, char **argv)
{
	unsigned int found = crack(parse(argc, argv));

	printf("\nFound %u\n", found);

	return 0;
}
