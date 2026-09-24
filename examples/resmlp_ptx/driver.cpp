#include <cuda.h>
#include <cstdint>

extern "C" int load_kernel(const char* path, const char* name, CUfunction* function) {
    CUmodule module;
    CUresult status = cuModuleLoad(&module, path);
    if (status != CUDA_SUCCESS) return status;
    return cuModuleGetFunction(function, module, name);
}

extern "C" int launch_affine(CUfunction function, CUdeviceptr x, CUdeviceptr y,
                              CUdeviceptr alpha, CUdeviceptr beta, unsigned n,
                              unsigned channels, CUstream stream) {
    void* args[] = {&x, &y, &alpha, &beta, &n, &channels};
    return cuLaunchKernel(function, (n + 1023) / 1024, 1, 1, 256, 1, 1, 0, stream, args, nullptr);
}

extern "C" int launch_chain(CUfunction function, CUdeviceptr x, CUdeviceptr y,
                             CUdeviceptr z, CUdeviceptr gamma, CUdeviceptr branch,
                             CUdeviceptr alpha, CUdeviceptr beta, unsigned n,
                             CUstream stream) {
    void* args[] = {&x, &y, &z, &gamma, &branch, &alpha, &beta, &n};
    return cuLaunchKernel(function, (n + 1023) / 1024, 1, 1, 256, 1, 1, 0, stream, args, nullptr);
}
