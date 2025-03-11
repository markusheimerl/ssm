CC = clang
CFLAGS = -O3 -march=native -ffast-math -Wall -Wextra
LDFLAGS = -lm -flto
CUDAFLAGS = --cuda-gpu-arch=sm_89 -x cuda -Wno-unknown-cuda-version
CUDALIBS = -L/usr/local/cuda/lib64 -lcudart -lcublas

ssm.out: ssm.c
	$(CC) $(CFLAGS) $(CUDAFLAGS) $^ $(CUDALIBS) $(LDFLAGS) -o $@

run: ssm.out
	@time ./ssm.out

clean:
	rm -f *.out *.csv *.bin
