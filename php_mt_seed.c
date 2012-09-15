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
#ifdef DEBUG
#include <assert.h>
#endif

#ifndef _OPENMP
#warning OpenMP not enabled, will only use one CPU core. Try gcc -fopenmp.
#endif

#define M 397

static inline void mt_rand2(int odd, uint32_t seed, uint32_t value,
    unsigned int *found)
{
	uint32_t s1x, s1y, x, y;
	int i;

#ifdef DEBUG
	assert((seed >> 30) == ((seed + 2) >> 30));
#endif

	s1x = x = 1812433253U * (seed ^ (seed >> 30)) + 1;
	s1y = y = 1812433253U * ((seed + 2) ^ (seed >> 30)) + 1;
	for (i = 2; i <= M; i++) {
		x = 1812433253U * (x ^ (x >> 30)) + i;
		y = 1812433253U * (y ^ (y >> 30)) + i;
	}

#ifdef DEBUG
	assert((seed & 0x80000000U) == ((seed + 2) & 0x80000000U));
#endif

	x ^= ((seed & 0x80000000U) | (s1x & 0x7fffffffU)) >> 1;
	y ^= ((seed & 0x80000000U) | (s1y & 0x7fffffffU)) >> 1;
	if (odd) {
		x ^= 0x9908b0dfU;
		y ^= 0x9908b0dfU;
	}

	x ^= x >> 11;
	y ^= y >> 11;
	x ^= (x << 7) & 0x9d2c5680U;
	y ^= (y << 7) & 0x9d2c5680U;
	x ^= (x << 15) & 0xefc60000U;
	y ^= (y << 15) & 0xefc60000U;
	x ^= x >> 18;
	y ^= y >> 18;

	if ((x >> 1) == value)
#ifdef _OPENMP
#pragma omp critical
#endif
	{
		printf("\nseed = %u\n", seed);
		(*found)++;
	}

	if ((y >> 1) == value)
#ifdef _OPENMP
#pragma omp critical
#endif
	{
		printf("\nseed = %u\n", seed + 2);
		(*found)++;
	}
}

static unsigned int crack_range(int32_t min, int32_t max, uint32_t value)
{
	unsigned int found = 0;
	int32_t base; /* signed type for OpenMP 2.5 compatibility */

#ifdef _OPENMP
#pragma omp parallel for default(none) private(base) shared(value, min, max, found)
#endif
	for (base = min; base < max; base++) {
		uint32_t seed = base << 2;
		mt_rand2(0, seed, value, &found);
		mt_rand2(1, seed + 1, value, &found);
	}

	return found;
}

static unsigned int crack(uint32_t value)
{
	unsigned int found = 0;
	int32_t base;
	const int32_t step = 0x400000;
	long clk_tck;
	clock_t start_time;
	struct tms tms;

	clk_tck = sysconf(_SC_CLK_TCK);
	start_time = times(&tms);

	for (base = 0; base < 0x40000000; base += step) {
		uint32_t start = base << 2, next = (base + step) << 2;
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
