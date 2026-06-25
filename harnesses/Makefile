CC       := gcc
CUDA_INC := /usr/local/cuda/include
CUDA_LIB := /usr/lib/x86_64-linux-gnu

CFLAGS   := -O2 -Wall -I$(CUDA_INC)
LDFLAGS  := -L$(CUDA_LIB) -lcuda

all: sass_test

sass_test: sass_test.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

clean:
	rm -f sass_test

.PHONY: all clean
