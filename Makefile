CC = gcc
RM = rm -f
CFLAGS = -Wall -O2 -fomit-frame-pointer -fopenmp
PROJ = php_mt_seed

php_mt_seed: php_mt_seed.c
	$(CC) $(CFLAGS) $< -o $@

clean:
	$(RM) $(PROJ)
