/*
 * bench_harness.c — Universal GPU benchmark harness for forge optimization
 *
 * Loads a cubin, launches a kernel with configurable input data, measures
 * ns/op with CUDA event timing, and validates output against a reference.
 *
 * Build:
 *   gcc -O2 -o bench_harness bench_harness.c \
 *       -I/usr/local/cuda/include -lcuda
 *
 * Usage:
 *   ./bench_harness <cubin> <kernel> <iters> [input_hex] [ref_hex]
 *
 *   cubin:      path to .cubin file
 *   kernel:     kernel function name
 *   iters:      number of loop iterations (e.g. 1048576)
 *   input_hex:  hex-encoded input bytes (little-endian, 32 bytes default)
 *   ref_hex:    expected output hex (if provided, validates correctness)
 *
 * Kernel signature expected:
 *   void kernel(void* input_output, void* const_input)
 *   - input_output: 32-byte buffer read+written by kernel (a)
 *   - const_input:  32-byte read-only buffer (b, constant across iters)
 *   Both passed as kernel parameters [0] and [1].
 *
 * Output (stdout):
 *   ns_per_op=XX.XXX result=<hex>
 *
 * Exit codes:
 *   0  — PASS (timing + optional correctness OK)
 *   1  — WRONG RESULT (correctness check failed)
 *   2  — CUDA error / crash
 *   3  — TIMEOUT (kernel took > timeout_ms)
 *   10 — cubin load error
 *   11 — kernel not found
 *   99 — usage error
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda.h>

#define CHK(call) do { \
    CUresult _r = (call); \
    if (_r != CUDA_SUCCESS) { \
        const char *_s = "?"; \
        cuGetErrorString(_r, &_s); \
        fprintf(stderr, "CUDA error %s line %d: %s\n", #call, __LINE__, _s); \
        return 2; \
    } \
} while(0)

#define BUF_BYTES 32
#define WARMUP_ITERS 3
#define TIMEOUT_MS 10000.0f  /* 10 second timeout */

/* Parse hex string into bytes. Returns number of bytes parsed. */
static int parse_hex(const char *hex, unsigned char *buf, int maxbytes) {
    int n = 0;
    while (*hex && n < maxbytes) {
        unsigned int byte;
        if (sscanf(hex, "%02x", &byte) != 1) break;
        buf[n++] = (unsigned char)byte;
        hex += 2;
    }
    return n;
}

/* Default input: a_init from mulmodp benchmark */
static unsigned int DEFAULT_A[8] = {
    0x12345678u, 0xABCDEF01u, 0x22334455u, 0x66778899u,
    0xAABBCCDDu, 0xEEFF0011u, 0x55AA55AAu, 0x00112233u,
};
/* Default const: b_val from mulmodp benchmark */
static unsigned int DEFAULT_B[8] = {
    0xFFFFFAAAu, 0xFFFFFFEFu, 0xFFFFFFFFu, 0xFFFFFFFFu,
    0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu,
};

int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr,
            "Usage: %s <cubin> <kernel> <iters> [input_hex] [ref_hex]\n",
            argv[0]);
        return 99;
    }

    const char *cubin_path  = argv[1];
    const char *kernel_name = argv[2];
    int iters = atoi(argv[3]);
    if (iters <= 0) { fprintf(stderr, "iters must be > 0\n"); return 99; }

    /* Parse optional input data */
    unsigned char a_buf[BUF_BYTES], b_buf[BUF_BYTES];
    memcpy(a_buf, DEFAULT_A, BUF_BYTES);
    memcpy(b_buf, DEFAULT_B, BUF_BYTES);
    if (argc >= 5 && strlen(argv[4]) >= 2) {
        parse_hex(argv[4], a_buf, BUF_BYTES);
    }
    if (argc >= 6 && strlen(argv[5]) >= 2) {
        parse_hex(argv[5], b_buf, BUF_BYTES);
    }

    /* Reference result (optional) */
    const char *ref_hex = (argc >= 7) ? argv[6] : NULL;

    /* CUDA init */
    CHK(cuInit(0));
    CUdevice dev; CHK(cuDeviceGet(&dev, 0));
    CUcontext ctx;
    CUctxCreateParams p = {0};
    CHK(cuCtxCreate(&ctx, &p, 0, dev));

    /* Load cubin */
    CUmodule mod;
    CUresult lr = cuModuleLoad(&mod, cubin_path);
    if (lr != CUDA_SUCCESS) {
        const char *s = "?"; cuGetErrorString(lr, &s);
        fprintf(stderr, "FAIL:LOAD:%s\n", s);
        return 10;
    }

    CUfunction fn;
    CUresult fr = cuModuleGetFunction(&fn, mod, kernel_name);
    if (fr != CUDA_SUCCESS) {
        const char *s = "?"; cuGetErrorString(fr, &s);
        fprintf(stderr, "FAIL:FUNC:%s (kernel='%s')\n", s, kernel_name);
        return 11;
    }

    /* Allocate device memory */
    CUdeviceptr da, db;
    CHK(cuMemAlloc(&da, BUF_BYTES));
    CHK(cuMemAlloc(&db, BUF_BYTES));
    CHK(cuMemcpyHtoD(db, b_buf, BUF_BYTES));

    void *params[] = { &da, &db };

    /* Warmup */
    CHK(cuMemcpyHtoD(da, a_buf, BUF_BYTES));
    for (int w = 0; w < WARMUP_ITERS; w++) {
        CHK(cuLaunchKernel(fn, 1,1,1, 1,1,1, 0, NULL, params, NULL));
    }
    CHK(cuCtxSynchronize());

    /* Timed run */
    CHK(cuMemcpyHtoD(da, a_buf, BUF_BYTES));

    CUevent t0, t1;
    CHK(cuEventCreate(&t0, CU_EVENT_DEFAULT));
    CHK(cuEventCreate(&t1, CU_EVENT_DEFAULT));
    CHK(cuEventRecord(t0, NULL));
    CHK(cuLaunchKernel(fn, 1,1,1, 1,1,1, 0, NULL, params, NULL));
    CHK(cuEventRecord(t1, NULL));
    CHK(cuEventSynchronize(t1));

    float elapsed_ms = 0.0f;
    CHK(cuEventElapsedTime(&elapsed_ms, t0, t1));

    if (elapsed_ms > TIMEOUT_MS) {
        fprintf(stderr, "TIMEOUT: %.1f ms\n", elapsed_ms);
        return 3;
    }

    /* Read result */
    unsigned char result[BUF_BYTES];
    CHK(cuMemcpyDtoH(result, da, BUF_BYTES));

    /* Correctness check — compare using big-endian u32 display format */
    if (ref_hex && strlen(ref_hex) > 0) {
        /* Build our result string in big-endian u32 format */
        char got_str[BUF_BYTES * 2 + 1];
        unsigned int *u32r = (unsigned int *)result;
        int pos = 0;
        for (int i = 7; i >= 0; i--)
            pos += sprintf(got_str + pos, "%08x", u32r[i]);

        if (strncmp(got_str, ref_hex, BUF_BYTES * 2) != 0) {
            fprintf(stderr, "WRONG_RESULT\ngot:      %s\nexpected: %.64s\n",
                    got_str, ref_hex);
            return 1;
        }
    }

    /* Output: ns_per_op=XX.XXX result=<hex>
     * Result is displayed as big-endian u32[8] (result[7]..result[0])
     * matching the convention of the mulmodp/addmodp harnesses.
     */
    double ns_per_op = (double)elapsed_ms * 1e6 / iters;
    printf("ns_per_op=%.3f result=", ns_per_op);
    unsigned int *u32_result = (unsigned int *)result;
    for (int i = 7; i >= 0; i--) printf("%08x", u32_result[i]);
    printf("\n");

    cuMemFree(da);
    cuMemFree(db);
    cuModuleUnload(mod);
    cuCtxDestroy(ctx);
    return 0;
}
