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
