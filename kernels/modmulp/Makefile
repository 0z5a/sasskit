# Build the modmulp-specific test harness
#
# Requires: CUDA driver headers + libcuda.so

CUDA_DIR ?= /usr/local/cuda

CC = gcc
CFLAGS = -O2 -I$(CUDA_DIR)/include
LDFLAGS = -L/usr/lib/x86_64-linux-gnu -lcuda

all: sass_test_modmulp

sass_test_modmulp: sass_test_modmulp.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

stub: modmulp_stub.cu
	$(CUDA_DIR)/bin/nvcc -arch=sm_120 -cubin -o modmulp_stub.cubin $<

clean:
	rm -f sass_test_modmulp *.cubin

.PHONY: all stub clean
