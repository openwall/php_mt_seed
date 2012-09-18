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

	assert((min >> (30 - 3)) == ((max - 1) >> (30 - 3)));

#ifdef __SSE4_1__
	vvalue = _mm_set1_epi32(value);

	{
		unsigned int i;
		for (i = 1; i <= M; i++)
			vi[i - 1] = _mm_set1_epi32(i);
	}
#endif

	{
		uint32_t seed = (uint32_t)min << 3;
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
		uint32_t seed = (uint32_t)base << 3;
#ifdef __SSE4_1__
		__m128i vseed = _mm_set1_epi32(seed);
		const __m128i cadde = _mm_set_epi32(0, 2, 4, 6);
		const __m128i caddo = _mm_set_epi32(1, 3, 5, 7);
		const __m128i cmul = _mm_set1_epi32(1812433253U);
		const __m128i c0x7fffffff = _mm_set1_epi32(0x7fffffff);
		const __m128i c0x9d2c5680 = _mm_set1_epi32(0x9d2c5680);
		const __m128i c0xefc60000 = _mm_set1_epi32(0xefc60000);
		const __m128i c0x9908b0df = _mm_set1_epi32(0x9908b0df);
		__m128i s0e, s0o, s1e, s1o, se, so;
		unsigned int i;

		s0e = _mm_add_epi32(vseed, cadde);
		s0o = _mm_add_epi32(vseed, caddo);

		s1e = se = _mm_macc_epi32(cmul, _mm_xor_si128(
		    s0e, seed_shr_30), vi[0]);
		s1o = so = _mm_macc_epi32(cmul, _mm_xor_si128(
		    s0o, seed_shr_30), vi[0]);

		for (i = 1; i < M; i += 2) {
			se = _mm_macc_epi32(cmul, _mm_xor_si128(
			    se, _mm_srli_epi32(se, 30)), vi[i]);
			so = _mm_macc_epi32(cmul, _mm_xor_si128(
			    so, _mm_srli_epi32(so, 30)), vi[i]);
			se = _mm_macc_epi32(cmul, _mm_xor_si128(
			    se, _mm_srli_epi32(se, 30)), vi[i + 1]);
			so = _mm_macc_epi32(cmul, _mm_xor_si128(
			    so, _mm_srli_epi32(so, 30)), vi[i + 1]);
		}

		se = _mm_xor_si128(se,
		    _mm_srli_epi32(_mm_or_si128(seed_and_0x80000000,
		    _mm_and_si128(s1e, c0x7fffffff)), 1));
		so = _mm_xor_si128(so,
		    _mm_srli_epi32(_mm_or_si128(seed_and_0x80000000,
		    _mm_and_si128(s1o, c0x7fffffff)), 1));

		so = _mm_xor_si128(so, c0x9908b0df);

		se = _mm_xor_si128(se, _mm_srli_epi32(se, 11));
		so = _mm_xor_si128(so, _mm_srli_epi32(so, 11));
		se = _mm_xor_si128(se, _mm_and_si128(_mm_slli_epi32(se, 7),
		    c0x9d2c5680));
		so = _mm_xor_si128(so, _mm_and_si128(_mm_slli_epi32(so, 7),
		    c0x9d2c5680));
		se = _mm_xor_si128(se, _mm_and_si128(_mm_slli_epi32(se, 15),
		    c0xefc60000));
		so = _mm_xor_si128(so, _mm_and_si128(_mm_slli_epi32(so, 15),
		    c0xefc60000));
		se = _mm_xor_si128(se, _mm_srli_epi32(se, 18));
		so = _mm_xor_si128(so, _mm_srli_epi32(so, 18));
		se = _mm_srli_epi32(se, 1);
		so = _mm_srli_epi32(so, 1);

		{
			__m128i semask = _mm_cmpeq_epi32(se, vvalue);
			__m128i somask = _mm_cmpeq_epi32(so, vvalue);
			if (_mm_testz_si128(semask, semask) &&
			    _mm_testz_si128(somask, somask))
				continue;
		}

		{
			union {
				__m128i v;
				uint32_t s[4];
			} u[2];
			u[0].v = se;
			u[1].v = so;
			COMPARE(u[0].s[0], seed + 6)
			COMPARE(u[0].s[1], seed + 4)
			COMPARE(u[0].s[2], seed + 2)
			COMPARE(u[0].s[3], seed)
			COMPARE(u[1].s[0], seed + 7)
			COMPARE(u[1].s[1], seed + 5)
			COMPARE(u[1].s[2], seed + 3)
			COMPARE(u[1].s[3], seed + 1)
		}
#else
		do {
			uint32_t s1e, s1o, se, so;
			unsigned int i;

			s1e = se = 1812433253U * (seed ^ seed_shr_30) + 1;
			s1o = so = 1812433253U * ((seed + 1) ^ seed_shr_30) + 1;
			for (i = 2; i <= M; i++) {
				se = 1812433253U * (se ^ (se >> 30)) + i;
				so = 1812433253U * (so ^ (so >> 30)) + i;
			}

			se ^= (seed_and_0x80000000 | (s1e & 0x7fffffffU)) >> 1;
			so ^= (seed_and_0x80000000 | (s1o & 0x7fffffffU)) >> 1;

			so ^= 0x9908b0dfU;

			se ^= se >> 11;
			so ^= so >> 11;
			se ^= (se << 7) & 0xe9d2c5680U;
			so ^= (so << 7) & 0xe9d2c5680U;
			se ^= (se << 15) & 0xeefc60000U;
			so ^= (so << 15) & 0xeefc60000U;
			se ^= se >> 18;
			so ^= so >> 18;

			COMPARE(se >> 1, seed)
			COMPARE(so >> 1, seed + 1)

			seed += 2;
		} while (seed & 7);
#endif
	}

	return found;
}

static unsigned int crack(uint32_t value)
{
	unsigned int found = 0;
	uint32_t base;
	const uint32_t step = 0x400000;
	long clk_tck;
	clock_t start_time;
	struct tms tms;

	clk_tck = sysconf(_SC_CLK_TCK);
	start_time = times(&tms);

	for (base = 0; base < 0x20000000; base += step) {
		uint32_t start = base << 3, next = (base + step) << 3;
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
