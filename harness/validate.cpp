#include <cuda.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>
#define CUDA(call) do { CUresult status = (call); if (status != CUDA_SUCCESS) { \
    const char* text; cuGetErrorString(status, &text); fprintf(stderr, "%s: %s\n", #call, text); return 1; } } while (0)
int main(int argc, char** argv) {
    if (argc != 3) return 2;
    unsigned seed = std::strtoul(argv[2], nullptr, 10);
    CUDA(cuInit(0));
    CUdevice device; CUDA(cuDeviceGet(&device, 0));
    CUuuid uuid; CUDA(cuDeviceGetUuid(&uuid, device));
    CUcontext context; CUDA(cuDevicePrimaryCtxRetain(&context, device));
    CUDA(cuCtxSetCurrent(context));
    CUmodule module; CUDA(cuModuleLoad(&module, argv[1]));
    CUfunction function; CUDA(cuModuleGetFunction(&function, module, "affine"));
    unsigned n = 65536;
    std::vector<uint32_t> input(n), output(n);
    std::mt19937 rng(seed);
    for (auto& value : input) value = seed == 0 ? 0 : rng();
    CUdeviceptr x, y; CUDA(cuMemAlloc(&x, n * 4)); CUDA(cuMemAlloc(&y, n * 4));
    CUDA(cuMemcpyHtoD(x, input.data(), n * 4));
    void* args[] = {&x, &y, &n};
    CUDA(cuLaunchKernel(function, n / 256, 1, 1, 256, 1, 1, 0, nullptr, args, nullptr));
    CUDA(cuCtxSynchronize()); CUDA(cuMemcpyDtoH(output.data(), y, n * 4));
    for (unsigned i = 0; i < n; ++i) if (output[i] != input[i] * 3u + 7u) return 3;
    CUevent start, stop; CUDA(cuEventCreate(&start, 0)); CUDA(cuEventCreate(&stop, 0));
    CUDA(cuEventRecord(start, nullptr));
    for (int i = 0; i < 100; ++i)
        CUDA(cuLaunchKernel(function, n / 256, 1, 1, 256, 1, 1, 0, nullptr, args, nullptr));
    CUDA(cuEventRecord(stop, nullptr)); CUDA(cuEventSynchronize(stop));
    float elapsed; CUDA(cuEventElapsedTime(&elapsed, start, stop));
    printf("{\"numerically_correct\":true,\"seed\":%u,\"elements\":%u,\"event_us\":%.6f,\"uuid_hex\":\"", seed, n, elapsed * 10);
    for (unsigned char byte : uuid.bytes) printf("%02x", byte);
    printf("\"}\n");
    CUDA(cuEventDestroy(start)); CUDA(cuEventDestroy(stop));
    CUDA(cuMemFree(x)); CUDA(cuMemFree(y)); CUDA(cuModuleUnload(module));
    CUDA(cuDevicePrimaryCtxRelease(device));
    return 0;
}
