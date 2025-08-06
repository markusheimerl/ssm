CC = clang
CFLAGS = -O3 -march=native -ffast-math -Wall -Wextra
LDFLAGS = -lopenblas -lm -flto

train.out: ssm.o data.o train.o
	$(CC) ssm.o data.o train.o $(LDFLAGS) -o $@

ssm.o: ssm.c ssm.h
	$(CC) $(CFLAGS) -c ssm.c -o $@

data.o: data.c data.h
	$(CC) $(CFLAGS) -c data.c -o $@

train.o: train.c ssm.h data.h
	$(CC) $(CFLAGS) -c train.c -o $@

run: train.out
	@time ./train.out

clean:
	rm -f *.out *.o *.csv *.bin
	$(MAKE) -C gpu clean