CC = clang
CFLAGS = -O3 -march=native -ffast-math -Wall -Wextra
LDFLAGS = -lm -flto

ARCH ?= sm_87
CUDAFLAGS = --cuda-gpu-arch=$(ARCH) -x cuda -Wno-unknown-cuda-version
CUDALIBS = -L/usr/local/cuda/lib64 -lcudart -lcublas

train.out: ssm.o data.o train.o
	$(CC) ssm.o data.o train.o $(CUDALIBS) $(LDFLAGS) -o $@

ssm.o: ssm.c ssm.h
	$(CC) $(CFLAGS) $(CUDAFLAGS) -c ssm.c -o $@

data.o: ../../data.c ../../data.h
	$(CC) $(CFLAGS) $(CUDAFLAGS) -c ../../data.c -o $@

train.o: train.c ssm.h ../../data.h
	$(CC) $(CFLAGS) $(CUDAFLAGS) -c train.c -o $@

run: train.out
	@time ./train.out

clean:
	rm -f *.out *.o *.csv *.bin