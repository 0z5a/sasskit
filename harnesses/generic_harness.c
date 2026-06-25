/*
 * generic_harness.c — Generic cubin test harness
 *
 * Launches a kernel with signature: void kernel(uint32_t* output, uint32_t n)
 * Compares output with a reference file if provided.
 *
 * Build:
 *   gcc -O2 -o generic_harness generic_harness.c \
 *       -I/usr/local/cuda/include -L/usr/lib/x86_64-linux-gnu -lcuda
 *
 * Usage:
 *   ./generic_harness <cubin> <kernel> [nthreads=256] [ref_file]
 *
 * Output: Prints first 32 output words. If ref_file given, compares.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda.h>

#define CU_CHECK(call, fc) do { CUresult _e=(call); if(_e!=CUDA_SUCCESS){ \
    const char*s=NULL; cuGetErrorString(_e,&s); \
    fprintf(stderr,"FAIL:%s:%d(%s)\n",#call,(int)_e,s?s:"?"); return fc;} } while(0)

#define N_WORDS 1024

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <cubin> <kernel> [nthreads=256] [smem=0] [ref_file]\n", argv[0]);
        return 99;
    }
    const char* cubin_path = argv[1];
    const char* kernel_name = argv[2];
    int nthreads = argc > 3 ? atoi(argv[3]) : 256;
    unsigned smem = argc > 4 ? (unsigned)atoi(argv[4]) : 0;
    const char* ref_file = argc > 5 ? argv[5] : NULL;

    CU_CHECK(cuInit(0), 20);
    CUdevice dev; CU_CHECK(cuDeviceGet(&dev, 0), 20);
    CUcontext ctx; CU_CHECK(cuDevicePrimaryCtxRetain(&ctx, dev), 20);
    CU_CHECK(cuCtxSetCurrent(ctx), 20);

    CUmodule mod;
    CUresult lr = cuModuleLoad(&mod, cubin_path);
    if (lr != CUDA_SUCCESS) {
        const char*s=NULL; cuGetErrorString(lr,&s);
        fprintf(stderr, "FAIL:LOAD:%d(%s)\n", (int)lr, s?s:"?");
        return 10;
    }

    CUfunction func;
    CUresult fr = cuModuleGetFunction(&func, mod, kernel_name);
    if (fr != CUDA_SUCCESS) {
        const char*s=NULL; cuGetErrorString(fr,&s);
        fprintf(stderr, "FAIL:FUNC:%d(%s)\n", (int)fr, s?s:"?");
        return 11;
    }

    if (smem > 0)
        cuFuncSetAttribute(func, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem);

    CUdeviceptr d_out;
    size_t buf = N_WORDS * sizeof(unsigned int);
    CU_CHECK(cuMemAlloc(&d_out, buf), 20);
    CU_CHECK(cuMemsetD8(d_out, 0, buf), 20);

    unsigned int n = N_WORDS;
    void* args[] = { &d_out, &n };

    CUresult launch = cuLaunchKernel(func, 1,1,1, (unsigned)nthreads,1,1,
                                      smem, NULL, args, NULL);
    if (launch != CUDA_SUCCESS) {
        const char*s=NULL; cuGetErrorString(launch,&s);
        fprintf(stderr, "FAIL:LAUNCH:%d(%s)\n", (int)launch, s?s:"?");
        return 12;
    }

    CUresult sync = cuCtxSynchronize();
    if (sync != CUDA_SUCCESS) {
        const char*s=NULL; cuGetErrorString(sync,&s);
        fprintf(stderr, "FAIL:SYNC:%d(%s)\n", (int)sync, s?s:"?");
        printf("FAIL\n");
        return 1;
    }

    unsigned int h_out[N_WORDS];
    CU_CHECK(cuMemcpyDtoH(h_out, d_out, buf), 20);

    /* Print first 16 words */
    printf("OUTPUT:");
    for (int i = 0; i < 16 && i < nthreads; i++)
        printf(" [%d]=0x%08X", i, h_out[i]);
    printf("\n");

    /* Compare with reference if provided */
    if (ref_file) {
        FILE* f = fopen(ref_file, "rb");
        if (!f) { fprintf(stderr, "Can't open ref: %s\n", ref_file); return 2; }
        unsigned int ref[N_WORDS];
        size_t nread = fread(ref, sizeof(unsigned int), N_WORDS, f);
        fclose(f);

        int mismatches = 0;
        for (size_t i = 0; i < nread && (int)i < nthreads; i++) {
            if (h_out[i] != ref[i]) {
                if (mismatches < 10)
                    fprintf(stderr, "MISMATCH [%zu]: got=0x%08X expected=0x%08X\n",
                            i, h_out[i], ref[i]);
                mismatches++;
            }
        }
        if (mismatches == 0) {
            printf("VERIFY: PASS (%zu words match)\n", nread);
        } else {
            printf("VERIFY: FAIL (%d mismatches out of %zu)\n", mismatches, nread);
            cuMemFree(d_out); cuModuleUnload(mod); cuDevicePrimaryCtxRelease(dev);
            return 1;
        }
    } else {
        /* Dump output to binary file for use as reference */
        FILE* f = fopen("/tmp/gpu_output.bin", "wb");
        if (f) {
            fwrite(h_out, sizeof(unsigned int), nthreads < N_WORDS ? nthreads : N_WORDS, f);
            fclose(f);
            printf("REF: saved %d words to /tmp/gpu_output.bin\n",
                   nthreads < N_WORDS ? nthreads : N_WORDS);
        }
        printf("PASS\n");
    }

    cuMemFree(d_out); cuModuleUnload(mod); cuDevicePrimaryCtxRelease(dev);
    return 0;
}
