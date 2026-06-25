/*
 * Driver API harness for modmulp kernel.
 * modmulp(const uint32_t* a, const uint32_t* b, uint32_t* r, int count)
 *
 * Loads cubin directly, runs 1024 modmulps, compares GPU vs CPU.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda.h>
#include <stdint.h>

#define N 1024

#define CU_CHECK(call) do { \
    CUresult _e = (call); \
    if (_e != CUDA_SUCCESS) { \
        const char* s; cuGetErrorString(_e, &s); \
        fprintf(stderr, "FAIL: %s → %s\n", #call, s ? s : "?"); \
        return 1; \
    } \
} while(0)

/* CPU reference modmulp (secp256k1) */
static void cpu_modmulp(const uint32_t a[8], const uint32_t b[8], uint32_t r[8]) {
    uint32_t prod[16] = {0};
    for (int i = 0; i < 8; i++) {
        uint64_t carry = 0;
        for (int j = 0; j < 8; j++) {
            uint64_t w = (uint64_t)a[i] * b[j] + carry + prod[i+j];
            prod[i+j] = (uint32_t)w; carry = w >> 32;
        }
        prod[i+8] += (uint32_t)carry;
    }
    uint64_t acc = 0;
    for (int k = 0; k < 8; k++) {
        acc += (uint64_t)prod[k] + (uint64_t)prod[k+8] * 0x3D1u;
        r[k] = (uint32_t)acc; acc >>= 32;
    }
    uint64_t c2 = 0;
    for (int k = 1; k < 8; k++) {
        c2 += (uint64_t)r[k] + prod[k+7];
        r[k] = (uint32_t)c2; c2 >>= 32;
    }
    uint64_t ov = acc + c2 + prod[15];
    uint64_t ov_lo = ov * 0x3D1u;
    uint64_t c3 = (uint64_t)r[0] + (uint32_t)ov_lo;
    r[0] = (uint32_t)c3; c3 >>= 32;
    c3 += (uint64_t)r[1] + (ov_lo >> 32) + (uint32_t)ov;
    r[1] = (uint32_t)c3; c3 >>= 32;
    c3 += (uint64_t)r[2] + (uint32_t)(ov >> 32);
    r[2] = (uint32_t)c3; c3 >>= 32;
    for (int k = 3; k < 8; k++) { c3 += r[k]; r[k] = (uint32_t)c3; c3 >>= 32; }
    if (c3) {
        uint64_t c4 = (uint64_t)r[0]+0x3D1u; r[0]=(uint32_t)c4; c4>>=32;
        c4+=(uint64_t)r[1]+1u; r[1]=(uint32_t)c4; c4>>=32;
        for(int k=2;k<8;k++){c4+=r[k];r[k]=(uint32_t)c4;c4>>=32;}
    }
    uint64_t t0=(uint64_t)r[0]+0x3D1u; uint32_t tt[8];
    tt[0]=(uint32_t)t0; uint64_t tc=t0>>32;
    tc+=(uint64_t)r[1]+1u; tt[1]=(uint32_t)tc; tc>>=32;
    for(int k=2;k<8;k++){tc+=r[k];tt[k]=(uint32_t)tc;tc>>=32;}
    if(tc) memcpy(r,tt,32);
}

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "Usage: %s <cubin>\n", argv[0]); return 99; }

    CU_CHECK(cuInit(0));
    CUdevice dev; CU_CHECK(cuDeviceGet(&dev, 0));
    CUcontext ctx; CU_CHECK(cuDevicePrimaryCtxRetain(&ctx, dev));
    CU_CHECK(cuCtxSetCurrent(ctx));

    CUmodule mod;
    CUresult lerr = cuModuleLoad(&mod, argv[1]);
    if (lerr != CUDA_SUCCESS) {
        const char* s; cuGetErrorString(lerr, &s);
        printf("FAIL:LOAD:%s\n", s); return 10;
    }
    CUfunction func;
    CUresult ferr = cuModuleGetFunction(&func, mod, "modmulp");
    if (ferr != CUDA_SUCCESS) {
        const char* s; cuGetErrorString(ferr, &s);
        printf("FAIL:FUNC:%s\n", s); return 11;
    }

    size_t sz = N * 8 * 4;
    CUdeviceptr d_a, d_b, d_r;
    CU_CHECK(cuMemAlloc(&d_a, sz));
    CU_CHECK(cuMemAlloc(&d_b, sz));
    CU_CHECK(cuMemAlloc(&d_r, sz));

    uint32_t *h_a = malloc(sz), *h_b = malloc(sz), *h_r = malloc(sz), *h_ref = malloc(sz);
    srand(42);
    for (int i = 0; i < N*8; i++) {
        h_a[i] = rand() ^ (rand() << 16);
        h_b[i] = rand() ^ (rand() << 16);
    }
    /* Special: test 0 = 1×1, test 1 = (P-1)×(P-1) */
    uint32_t P[8] = {0xFFFFFC2F,0xFFFFFFFE,0xFFFFFFFF,0xFFFFFFFF,
                     0xFFFFFFFF,0xFFFFFFFF,0xFFFFFFFF,0xFFFFFFFF};
    memset(&h_a[0],0,32); h_a[0]=1; memset(&h_b[0],0,32); h_b[0]=1;
    memcpy(&h_a[8],P,32); h_a[8]-=1; memcpy(&h_b[8],P,32); h_b[8]-=1;

    for (int i = 0; i < N; i++) cpu_modmulp(&h_a[i*8], &h_b[i*8], &h_ref[i*8]);

    CU_CHECK(cuMemcpyHtoD(d_a, h_a, sz));
    CU_CHECK(cuMemcpyHtoD(d_b, h_b, sz));

    int count = N;
    void* args[] = { &d_a, &d_b, &d_r, &count };
    CUresult launch = cuLaunchKernel(func, (N+255)/256,1,1, 256,1,1, 0,NULL, args,NULL);
    if (launch != CUDA_SUCCESS) {
        const char* s; cuGetErrorString(launch, &s);
        printf("FAIL:LAUNCH:%s\n", s); return 12;
    }

    CUresult sync = cuCtxSynchronize();
    if (sync != CUDA_SUCCESS) {
        const char* s; cuGetErrorString(sync, &s);
        printf("FAIL:SYNC:%s\n", s); return 1;
    }

    CU_CHECK(cuMemcpyDtoH(h_r, d_r, sz));

    int pass=0, fail=0;
    for (int i = 0; i < N; i++) {
        if (memcmp(&h_r[i*8], &h_ref[i*8], 32)==0) { pass++; }
        else {
            fail++;
            if (fail <= 5) {
                printf("FAIL[%d] GPU:", i);
                for(int j=7;j>=0;j--) printf("%08x",h_r[i*8+j]);
                printf("\n        CPU:");
                for(int j=7;j>=0;j--) printf("%08x",h_ref[i*8+j]);
                printf("\n");
            }
        }
    }
    printf("%s: %d/%d\n", pass==N?"PASS":"FAIL", pass, N);

    cuMemFree(d_a); cuMemFree(d_b); cuMemFree(d_r);
    cuModuleUnload(mod); cuDevicePrimaryCtxRelease(dev);
    free(h_a); free(h_b); free(h_r); free(h_ref);
    return pass==N ? 0 : 1;
}
