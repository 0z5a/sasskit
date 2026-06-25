/*
 * Probe kernel to force MOV instructions in SASS output.
 * Uses volatile asm copies to prevent optimization.
 */
#include <stdint.h>

extern "C"
__global__ void probe_mov(unsigned int* out,
                          const unsigned int* in,
                          int n)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n) return;

    unsigned int x = in[tid];
    unsigned int r0, r1, r2, r3, r4, r5, r6, r7;

    /* Force MOV Rd, RZ (zero init) — volatile prevents opt-out */
    asm volatile("mov.u32 %0, 0;" : "=r"(r0));
    asm volatile("mov.u32 %0, 0;" : "=r"(r1));
    asm volatile("mov.u32 %0, 0;" : "=r"(r2));
    asm volatile("mov.u32 %0, 0;" : "=r"(r3));

    /* Force register-to-register MOVs */
    asm volatile("mov.u32 %0, %1;" : "=r"(r4) : "r"(x));
    asm volatile("mov.u32 %0, %1;" : "=r"(r5) : "r"(r0));
    asm volatile("mov.u32 %0, %1;" : "=r"(r6) : "r"(r1));
    asm volatile("mov.u32 %0, %1;" : "=r"(r7) : "r"(r2));

    /* Use all values to prevent DCE */
    asm volatile("add.u32 %0, %0, %1;" : "+r"(r4) : "r"(r5));
    asm volatile("add.u32 %0, %0, %1;" : "+r"(r6) : "r"(r7));
    asm volatile("add.u32 %0, %0, %1;" : "+r"(r4) : "r"(r3));
    asm volatile("add.u32 %0, %0, %1;" : "+r"(r4) : "r"(r6));

    /* MOV with immediate */
    unsigned int imm;
    asm volatile("mov.u32 %0, 0x3D1;" : "=r"(imm));
    asm volatile("add.u32 %0, %0, %1;" : "+r"(r4) : "r"(imm));

    out[tid] = r4;
}
