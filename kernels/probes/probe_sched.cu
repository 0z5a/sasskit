/*
 * Probe kernel to extract nvcc scheduling for opcode types used in modmulp:
 *   MOV, IMAD.WIDE.U32, IADD, IADD.X, SEL
 *
 * Compile:  nvcc -arch=sm_120 -cubin -o probe_sched.cubin probe_sched.cu
 * Dump:     cuobjdump -sass probe_sched.cubin
 */

#include <stdint.h>

extern "C"
__global__ void probe(unsigned int* out,
                      const unsigned int* a,
                      const unsigned int* b,
                      int n)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n) return;

    unsigned int a0 = a[tid*4];
    unsigned int a1 = a[tid*4+1];
    unsigned int b0 = b[tid*4];
    unsigned int b1 = b[tid*4+1];

    /* --- MOV: zero-init via volatile asm to prevent optimization --- */
    unsigned int r0, r1, r2, r3;
    asm volatile("mov.u32 %0, 0;" : "=r"(r0));
    asm volatile("mov.u32 %0, 0;" : "=r"(r1));
    asm volatile("mov.u32 %0, 0;" : "=r"(r2));
    asm volatile("mov.u32 %0, 0;" : "=r"(r3));

    /* --- IMAD.WIDE.U32: 32×32→64 multiply-add via C --- */
    unsigned long long w0 = (unsigned long long)a0 * b0 + r0;
    unsigned int lo0 = (unsigned int)w0;
    unsigned int hi0 = (unsigned int)(w0 >> 32);

    unsigned long long w1 = (unsigned long long)a1 * b1 + r2;
    unsigned int lo1 = (unsigned int)w1;
    unsigned int hi1 = (unsigned int)(w1 >> 32);

    unsigned long long w2 = (unsigned long long)a0 * b1;
    unsigned int lo2 = (unsigned int)w2;
    unsigned int hi2 = (unsigned int)(w2 >> 32);

    unsigned long long w3 = (unsigned long long)a1 * b0;
    unsigned int lo3 = (unsigned int)w3;
    unsigned int hi3 = (unsigned int)(w3 >> 32);

    /* --- IADD with carry-out + IADD.X with carry-in --- */
    /* This 4-wide carry chain generates IADD P0 + IADD.X P0 sequences */
    asm volatile(
        "add.cc.u32  %0, %0, %4;\n\t"
        "addc.cc.u32 %1, %1, %5;\n\t"
        "addc.cc.u32 %2, %2, %6;\n\t"
        "addc.u32    %3, %3, %7;"
        : "+r"(lo0), "+r"(hi0), "+r"(lo1), "+r"(hi1)
        : "r"(lo2), "r"(hi2), "r"(lo3), "r"(hi3)
    );

    /* Another carry chain */
    asm volatile(
        "add.cc.u32  %0, %0, %2;\n\t"
        "addc.cc.u32 %1, %1, 0;\n\t"
        : "+r"(lo0), "+r"(hi0)
        : "r"(a0)
    );

    /* --- SEL: conditional select --- */
    unsigned int sel_r;
    asm volatile(
        "{\n\t"
        ".reg .pred psel;\n\t"
        "setp.ne.u32 psel, %2, 0;\n\t"
        "selp.u32 %0, %1, %2, psel;\n\t"
        "}"
        : "=r"(sel_r)
        : "r"(lo0), "r"(hi1)
    );

    unsigned int sel_r2;
    asm volatile(
        "{\n\t"
        ".reg .pred ps2;\n\t"
        "setp.gt.u32 ps2, %2, %3;\n\t"
        "selp.u32 %0, %1, %2, ps2;\n\t"
        "}"
        : "=r"(sel_r2)
        : "r"(hi0), "r"(lo1), "r"(hi1)
    );

    /* MOV: register copy */
    unsigned int m0;
    asm volatile("mov.u32 %0, %1;" : "=r"(m0) : "r"(sel_r));

    /* Store (prevent DCE) */
    out[tid*4]   = lo0 ^ m0;
    out[tid*4+1] = hi0 ^ sel_r2;
    out[tid*4+2] = lo1;
    out[tid*4+3] = hi1;
}
