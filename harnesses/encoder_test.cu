/*
 * Minimal test kernel for verifying SASS instruction encoders.
 *
 * Compile:
 *   nvcc -arch=sm_120 -cubin -o encoder_test.cubin encoder_test.cu
 *
 * The kernel is intentionally tiny. We use grow_kernel_text() to add
 * NOP padding, then inject encoded instructions into the NOP area.
 */

extern "C"
__global__ void encoder_test(unsigned int* output, unsigned int n) {
    /* Thread 0 writes canary to output[0] */
    if (threadIdx.x == 0) {
        output[0] = 0xDEADBEEF;
    }
}
