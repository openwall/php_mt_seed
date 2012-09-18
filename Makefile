CC = gcc
RM = rm -f
CFLAGS = -Wall -march=native -O2 -fomit-frame-pointer -funroll-loops -fopenmp
PROJ = php_mt_seed

php_mt_seed: php_mt_seed.c
	$(CC) $(CFLAGS) $< -o $@

clean:
	$(RM) $(PROJ)
