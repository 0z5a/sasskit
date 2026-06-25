/*
 * encoder_harness.c — Simple harness for encoder_test cubin
 *
 * Loads cubin, launches encoder_test kernel, reads back output buffer.
 * Checks canary and any test-specific values.
 *
 * Build:
 *   gcc -O2 -o encoder_harness encoder_harness.c \
 *       -I/usr/local/cuda/include -L/usr/lib/x86_64-linux-gnu -lcuda
 *
 * Usage:
 *   ./encoder_harness <cubin_file> [nthreads=256]
 *
 * Output format (stdout):
 *   PASS canary=0xDEADBEEF out[1]=<val> out[2]=<val> ...
 *   FAIL:<reason>
 *
 * Exit codes:
 *   0 — kernel ran successfully
 *   1 — kernel error (illegal instr, etc.)
 *   10 — load error
 *   20 — CUDA init error
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda.h>

static const char* cu_err_name(CUresult err) {
    const char* s = NULL;
    cuGetErrorName(err, &s);
    return s ? s : "UNKNOWN";
}

static const char* cu_err_str(CUresult err) {
    const char* s = NULL;
    cuGetErrorString(err, &s);
    return s ? s : "";
}

#define CU_CHECK(call, fail_code) do {          \
    CUresult _e = (call);                       \
    if (_e != CUDA_SUCCESS) {                   \
        fprintf(stderr, "FAIL:%s:%s (%s)\n",    \
                #call, cu_err_name(_e),         \
                cu_err_str(_e));                \
        return (fail_code);                     \
    }                                           \
} while (0)

#define OUTPUT_WORDS 64  /* read back 64 uint32s */

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <cubin> [nthreads=256]\n", argv[0]);
        return 99;
    }

    const char* cubin_path = argv[1];
    int nthreads = argc > 2 ? atoi(argv[2]) : 256;

    /* CUDA init */
    CU_CHECK(cuInit(0), 20);
    CUdevice dev;
    CU_CHECK(cuDeviceGet(&dev, 0), 20);
    CUcontext ctx;
    CU_CHECK(cuDevicePrimaryCtxRetain(&ctx, dev), 20);
    CU_CHECK(cuCtxSetCurrent(ctx), 20);

    /* Load cubin */
    CUmodule mod;
    CUresult load_err = cuModuleLoad(&mod, cubin_path);
    if (load_err != CUDA_SUCCESS) {
        fprintf(stderr, "FAIL:LOAD:%s (%s)\n",
                cu_err_name(load_err), cu_err_str(load_err));
        cuDevicePrimaryCtxRelease(dev);
        return 10;
    }

    CUfunction func;
    CUresult func_err = cuModuleGetFunction(&func, mod, "encoder_test");
    if (func_err != CUDA_SUCCESS) {
        fprintf(stderr, "FAIL:FUNC:%s (%s)\n",
                cu_err_name(func_err), cu_err_str(func_err));
        cuModuleUnload(mod);
        cuDevicePrimaryCtxRelease(dev);
        return 11;
    }

    /* Allocate output buffer on GPU */
    CUdeviceptr d_output;
    size_t buf_bytes = OUTPUT_WORDS * sizeof(unsigned int);
    CU_CHECK(cuMemAlloc(&d_output, buf_bytes), 20);
    CU_CHECK(cuMemsetD8(d_output, 0, buf_bytes), 20);

    /* Launch: 1 block, nthreads threads, 0 smem */
    unsigned int n = OUTPUT_WORDS;
    void* args[] = { &d_output, &n };

    CUresult launch_err = cuLaunchKernel(
        func,
        1, 1, 1,              /* grid */
        (unsigned)nthreads, 1, 1,  /* block */
        0,                    /* shared mem */
        NULL,                 /* stream */
        args,                 /* params */
        NULL
    );

    if (launch_err != CUDA_SUCCESS) {
        fprintf(stderr, "FAIL:LAUNCH:%s (%s)\n",
                cu_err_name(launch_err), cu_err_str(launch_err));
        cuMemFree(d_output);
        cuModuleUnload(mod);
        cuDevicePrimaryCtxRelease(dev);
        return 12;
    }

    CUresult sync_err = cuCtxSynchronize();
    if (sync_err != CUDA_SUCCESS) {
        fprintf(stderr, "FAIL:SYNC:%s (%s)\n",
                cu_err_name(sync_err), cu_err_str(sync_err));
        printf("FAIL:%s\n", cu_err_name(sync_err));
        cuMemFree(d_output);
        cuModuleUnload(mod);
        cuDevicePrimaryCtxRelease(dev);
        return 1;
    }

    /* Read back output */
    unsigned int h_output[OUTPUT_WORDS];
    CU_CHECK(cuMemcpyDtoH(h_output, d_output, buf_bytes), 20);

    /* Print results */
    printf("PASS canary=0x%08X", h_output[0]);
    for (int i = 1; i < OUTPUT_WORDS; i++) {
        if (h_output[i] != 0) {
            printf(" out[%d]=0x%08X", i, h_output[i]);
        }
    }
    printf("\n");

    cuMemFree(d_output);
    cuModuleUnload(mod);
    cuDevicePrimaryCtxRelease(dev);
    return 0;
}
