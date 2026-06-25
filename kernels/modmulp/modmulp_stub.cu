/*
 * Stub modmulp kernel — creates cubin with function "modmulp" and
 * a .text section large enough for ~450 hand-written SASS instructions.
 *
 * Compile: nvcc -arch=sm_120 -cubin -maxrregcount=48 -o modmulp_stub.cubin modmulp_stub.cu
 */
#include <stdint.h>

/*
 * Force a large .text section by doing lots of unrolled dummy work.
 * The actual computation is irrelevant — we'll replace the entire .text.
 */
extern "C"
__global__ void __launch_bounds__(256, 1)
modmulp(const unsigned int* __restrict__ a,
        const unsigned int* __restrict__ b,
        unsigned int* __restrict__ r,
        int count)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= count) return;

    /* Load some values (prevents the compiler from optimizing everything away) */
    unsigned int x0 = a[tid*8], x1 = a[tid*8+1], x2 = a[tid*8+2], x3 = a[tid*8+3];
    unsigned int x4 = a[tid*8+4], x5 = a[tid*8+5], x6 = a[tid*8+6], x7 = a[tid*8+7];
    unsigned int y0 = b[tid*8], y1 = b[tid*8+1], y2 = b[tid*8+2], y3 = b[tid*8+3];
    unsigned int y4 = b[tid*8+4], y5 = b[tid*8+5], y6 = b[tid*8+6], y7 = b[tid*8+7];

    /* Unrolled schoolbook multiply — 8x8 = 64 multiply-add operations.
     * This generates ~300+ SASS instructions. */
    unsigned long long w;
    unsigned int p[16] = {0};

    #pragma unroll
    for (int i = 0; i < 8; i++) {
        unsigned int ai;
        switch (i) {
            case 0: ai = x0; break; case 1: ai = x1; break;
            case 2: ai = x2; break; case 3: ai = x3; break;
            case 4: ai = x4; break; case 5: ai = x5; break;
            case 6: ai = x6; break; case 7: ai = x7; break;
        }
        unsigned int carry = 0;
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            unsigned int bj;
            switch (j) {
                case 0: bj = y0; break; case 1: bj = y1; break;
                case 2: bj = y2; break; case 3: bj = y3; break;
                case 4: bj = y4; break; case 5: bj = y5; break;
                case 6: bj = y6; break; case 7: bj = y7; break;
            }
            w = (unsigned long long)ai * bj + carry + p[i+j];
            p[i+j] = (unsigned int)w;
            carry = (unsigned int)(w >> 32);
        }
        p[i+8] += carry;
    }

    /* Barrett reduction — more dummy work to pad .text */
    unsigned int rr[8];
    unsigned long long acc = 0;
    #pragma unroll
    for (int k = 0; k < 8; k++) {
        acc += (unsigned long long)p[k] + (unsigned long long)p[k+8] * 0x3D1u;
        rr[k] = (unsigned int)acc;
        acc >>= 32;
    }

    /* Extra padding: conditional subtraction + second reduction */
    if (acc) {
        unsigned long long c = (unsigned long long)rr[0] + 0x3D1u;
        rr[0] = (unsigned int)c; c >>= 32;
        c += (unsigned long long)rr[1] + 1; rr[1] = (unsigned int)c; c >>= 32;
        #pragma unroll
        for (int k = 2; k < 8; k++) {
            c += rr[k]; rr[k] = (unsigned int)c; c >>= 32;
        }
    }
    /* Second conditional subtraction for more padding */
    {
        unsigned int tt[8];
        unsigned long long t0 = (unsigned long long)rr[0] + 0x3D1u;
        tt[0] = (unsigned int)t0; unsigned long long tc = t0 >> 32;
        tc += (unsigned long long)rr[1] + 1; tt[1] = (unsigned int)tc; tc >>= 32;
        #pragma unroll
        for (int k = 2; k < 8; k++) {
            tc += rr[k]; tt[k] = (unsigned int)tc; tc >>= 32;
        }
        if (tc) {
            #pragma unroll
            for (int k = 0; k < 8; k++) rr[k] = tt[k];
        }
    }
    /* Third pass: cross multiply for more padding */
    {
        unsigned long long c2 = 0;
        #pragma unroll
        for (int k = 1; k < 8; k++) {
            c2 += (unsigned long long)rr[k] + p[k+7];
            rr[k] = (unsigned int)c2; c2 >>= 32;
        }
        unsigned long long ov = acc + c2 + p[15];
        unsigned long long ov_lo = ov * 0x3D1u;
        unsigned long long c3 = (unsigned long long)rr[0] + (unsigned int)ov_lo;
        rr[0] = (unsigned int)c3; c3 >>= 32;
        c3 += (unsigned long long)rr[1] + (ov_lo >> 32) + (unsigned int)ov;
        rr[1] = (unsigned int)c3; c3 >>= 32;
        c3 += (unsigned long long)rr[2] + (unsigned int)(ov >> 32);
        rr[2] = (unsigned int)c3; c3 >>= 32;
        #pragma unroll
        for (int k = 3; k < 8; k++) {
            c3 += rr[k]; rr[k] = (unsigned int)c3; c3 >>= 32;
        }
    }

    /* Store results */
    r[tid*8]   = rr[0]; r[tid*8+1] = rr[1]; r[tid*8+2] = rr[2]; r[tid*8+3] = rr[3];
    r[tid*8+4] = rr[4]; r[tid*8+5] = rr[5]; r[tid*8+6] = rr[6]; r[tid*8+7] = rr[7];
}
