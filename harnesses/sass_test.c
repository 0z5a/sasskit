/*
 * sass_test.c — CUDA Driver API test harness for cubin validation
 *
 * Loads a cubin file, launches a kernel with dummy data, reports whether
 * the kernel survives execution.  Designed to be called from a subprocess
 * so that GPU-fatal errors (illegal instruction, OOR register) kill only
 * this process, not the orchestrator.
 *
 * Optional: --bench N  runs the kernel N times with GPU event timing.
 *
 * Build:
 *   gcc -O2 -o sass_test sass_test.c \
 *       -I/usr/local/cuda/include -L/usr/lib/x86_64-linux-gnu -lcuda
 *
 * Usage:
 *   ./sass_test <cubin_file> [kernel_name] [blocks] [threads] [smem_bytes]
 *   ./sass_test <cubin_file> --bench 100  [kernel_name] [blocks] [threads] [smem]
 *
 * Exit codes:
 *   0  — PASS (kernel launched and completed without error)
 *   1  — FAIL at cuCtxSynchronize (kernel error: illegal instr, OOR reg, etc.)
 *   10 — FAIL at cuModuleLoad (bad cubin image)
 *   11 — FAIL at cuModuleGetFunction (kernel not found)
 *   12 — FAIL at cuLaunchKernel (launch config error)
 *   20 — FAIL at CUDA init / device / context
 *   99 — usage error
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda.h>

/* ---------- TKparams layout (must match defs.h from RCKangaroo) ---------- */
typedef unsigned long long u64;
typedef unsigned int       u32;

struct TKparams {
    u64* Kangs;
    u32  KangCnt;
    u32  BlockCnt;
    u32  BlockSize;
    u32  GroupCnt;
    u64* L2;
    u64  DP;
    u32* DPs_out;
    u64* Jumps1;
    u64* Jumps2;
    u64* Jumps3;
    u64* JumpsList;
    u32* DPTable;
    u32* L1S2;
    u64* LastPnts;
    u64* LoopTable;
    u32* dbg_buf;
    u32* LoopedKangs;
    char IsGenMode;   /* bool, 1 byte */
    char UseGLV;      /* bool, 1 byte */
    /* 2 bytes padding here (compiler inserts to align next u32) */
    u32  KernelA_LDS_Size;
    u32  KernelB_LDS_Size;
    u32  KernelC_LDS_Size;
};

/* -------- helpers -------- */

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

/* -------- main -------- */

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr,
            "Usage: %s <cubin> [--bench N] [kernel=KernelA] [blocks=1] [threads=256] [smem=28672]\n",
            argv[0]);
        return 99;
    }

    /* Parse --bench flag (can appear as argv[2]) */
    int bench_iters = 0;
    int arg_shift = 0;
    if (argc >= 3 && strcmp(argv[2], "--bench") == 0) {
        bench_iters = argc >= 4 ? atoi(argv[3]) : 100;
        if (bench_iters < 1) bench_iters = 100;
        arg_shift = 2;  /* shift remaining positional args */
    }

    const char* cubin_path   = argv[1];
    const char* kernel_name  = argc > (2+arg_shift) ? argv[2+arg_shift] : "KernelA";
    int         nblocks      = argc > (3+arg_shift) ? atoi(argv[3+arg_shift]) : 1;
    int         nthreads     = argc > (4+arg_shift) ? atoi(argv[4+arg_shift]) : 256;
    /* Default smem: JMP1_X_STRIDE(6) * 8 * JMP_CNT(512) + 16 * 256 = 28672 */
    unsigned    smem_bytes   = argc > (5+arg_shift) ? (unsigned)atoi(argv[5+arg_shift]) : 28672;

    /* --- CUDA init --- */
    CU_CHECK(cuInit(0), 20);

    CUdevice dev;
    CU_CHECK(cuDeviceGet(&dev, 0), 20);

    CUcontext ctx;
    CU_CHECK(cuDevicePrimaryCtxRetain(&ctx, dev), 20);
    CU_CHECK(cuCtxSetCurrent(ctx), 20);

    /* --- Load cubin --- */
    CUmodule mod;
    CUresult load_err = cuModuleLoad(&mod, cubin_path);
    if (load_err != CUDA_SUCCESS) {
        fprintf(stderr, "FAIL:LOAD:%s (%s)\n",
                cu_err_name(load_err), cu_err_str(load_err));
        cuDevicePrimaryCtxRelease(dev);
        return 10;
    }

    /* --- Get kernel function --- */
    CUfunction func;
    CUresult func_err = cuModuleGetFunction(&func, mod, kernel_name);
    if (func_err != CUDA_SUCCESS) {
        fprintf(stderr, "FAIL:FUNC:%s (%s)\n",
                cu_err_name(func_err), cu_err_str(func_err));
        cuModuleUnload(mod);
        cuDevicePrimaryCtxRelease(dev);
        return 11;
    }

    /* Set max dynamic shared memory for this function */
    cuFuncSetAttribute(func,
        CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem_bytes);

    /* --- Allocate large GPU buffer (256 MB, zeroed) --- */
    CUdeviceptr gpu_buf = 0;
    size_t buf_size = 256ULL * 1024 * 1024;
    CU_CHECK(cuMemAlloc(&gpu_buf, buf_size), 20);
    CU_CHECK(cuMemsetD8(gpu_buf, 0, buf_size), 20);

    /* --- Build TKparams with all pointers → gpu_buf --- */
    struct TKparams params;
    memset(&params, 0, sizeof(params));

    params.Kangs      = (u64*)gpu_buf;
    params.L2         = (u64*)gpu_buf;
    params.DPs_out    = (u32*)gpu_buf;
    params.Jumps1     = (u64*)gpu_buf;
    params.Jumps2     = (u64*)gpu_buf;
    params.Jumps3     = (u64*)gpu_buf;
    params.JumpsList  = (u64*)gpu_buf;
    params.DPTable    = (u32*)gpu_buf;
    params.L1S2       = (u32*)gpu_buf;
    params.LastPnts   = (u64*)gpu_buf;
    params.LoopTable  = (u64*)gpu_buf;
    params.dbg_buf    = (u32*)gpu_buf;
    params.LoopedKangs= (u32*)gpu_buf;

    params.BlockCnt   = nblocks;
    params.BlockSize  = nthreads;
    params.GroupCnt   = 24;
    params.KangCnt    = nblocks * nthreads * 24;
    params.KernelA_LDS_Size = smem_bytes;

    fprintf(stderr, "INFO: cubin=%s kernel=%s grid=%dx%d smem=%u bench=%d sizeof(TKparams)=%zu\n",
            cubin_path, kernel_name, nblocks, nthreads, smem_bytes,
            bench_iters, sizeof(struct TKparams));

    /* --- Launch (single run for crash detection) --- */
    void* kernel_args[] = { &params };

    CUresult launch_err = cuLaunchKernel(
        func,
        (unsigned)nblocks, 1, 1,     /* grid  */
        (unsigned)nthreads, 1, 1,    /* block */
        smem_bytes,                  /* dynamic shared mem */
        NULL,                        /* stream (default) */
        kernel_args,                 /* params */
        NULL                         /* extra */
    );

    if (launch_err != CUDA_SUCCESS) {
        fprintf(stderr, "FAIL:LAUNCH:%s (%s)\n",
                cu_err_name(launch_err), cu_err_str(launch_err));
        cuMemFree(gpu_buf);
        cuModuleUnload(mod);
        cuDevicePrimaryCtxRelease(dev);
        return 12;
    }

    /* --- Synchronize — this is where kernel errors surface --- */
    CUresult sync_err = cuCtxSynchronize();

    if (sync_err != CUDA_SUCCESS) {
        fprintf(stderr, "FAIL:SYNC:%s (%s)\n",
                cu_err_name(sync_err), cu_err_str(sync_err));
        printf("FAIL:%s\n", cu_err_name(sync_err));
        cuDevicePrimaryCtxRelease(dev);
        return 1;
    }

    /* --- Benchmark mode: run N times with GPU event timing --- */
    if (bench_iters > 0) {
        CUevent ev_start, ev_stop;
        CU_CHECK(cuEventCreate(&ev_start, CU_EVENT_DEFAULT), 20);
        CU_CHECK(cuEventCreate(&ev_stop, CU_EVENT_DEFAULT), 20);

        /* Warmup: 3 launches */
        for (int w = 0; w < 3; w++) {
            cuLaunchKernel(func,
                (unsigned)nblocks, 1, 1,
                (unsigned)nthreads, 1, 1,
                smem_bytes, NULL, kernel_args, NULL);
        }
        cuCtxSynchronize();

        /* Timed run */
        CU_CHECK(cuEventRecord(ev_start, NULL), 20);

        for (int b = 0; b < bench_iters; b++) {
            cuLaunchKernel(func,
                (unsigned)nblocks, 1, 1,
                (unsigned)nthreads, 1, 1,
                smem_bytes, NULL, kernel_args, NULL);
        }

        CU_CHECK(cuEventRecord(ev_stop, NULL), 20);
        CU_CHECK(cuEventSynchronize(ev_stop), 20);

        float ms = 0.0f;
        CU_CHECK(cuEventElapsedTime(&ms, ev_start, ev_stop), 20);

        float per_iter = ms / (float)bench_iters;
        printf("PASS bench=%d total=%.3fms per_iter=%.4fms\n",
               bench_iters, ms, per_iter);

        cuEventDestroy(ev_start);
        cuEventDestroy(ev_stop);
    } else {
        printf("PASS\n");
    }

    cuMemFree(gpu_buf);
    cuModuleUnload(mod);
    cuDevicePrimaryCtxRelease(dev);
    return 0;
}
