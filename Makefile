CC = clang
CFLAGS = -O3 -march=native -ffast-math -Wall -Wextra
LDFLAGS = -static -lopenblas -lm -flto

ssm.out: ssm.c
	$(CC) $(CFLAGS) $^ $(LDFLAGS) -o $@

run: ssm.out
	@time ./ssm.out
	
clean:
	rm -f *.out *.csv *.bin
