CC = clang
CFLAGS = -O3 -march=native -ffast-math -Wall -Wextra
LDFLAGS = -lm -flto
CUDAFLAGS = --cuda-gpu-arch=sm_86 \
    -x cuda \
    -fcuda-flush-denormals-to-zero \
    -fcuda-approx-transcendentals \
    -Wno-unknown-cuda-version

CUDALIBS = -L/usr/local/cuda/lib64 -lcudart -lcublas

ssm.out: ssm.c
	$(CC) $(CFLAGS) $(CUDAFLAGS) $^ $(CUDALIBS) $(LDFLAGS) -o $@

run: ssm.out
	@time ./ssm.out

clean:
	rm -f *.out *.csv *.bin