#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>

// CUDA kernel to simulate finding a valid nonce
__global__ void mine_kernel(uint64_t nonce_start, uint64_t *found_nonce, int *found_flag) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    // Force success on first thread
    if (idx == 0) {
        *found_nonce = nonce_start + idx;
        *found_flag = 1;
        printf("GPU: Thread %d found nonce %llu\\n", idx, *found_nonce);
    }
}

// Kernel launcher from host
extern "C" void launch_kernel(uint64_t nonce_start, uint64_t *result_nonce, int *result_flag) {
    uint64_t *d_nonce;
    int *d_flag;

    // Use unified memory so we can read/write from both CPU and GPU
    cudaMallocManaged(&d_nonce, sizeof(uint64_t));
    cudaMallocManaged(&d_flag, sizeof(int));

    *d_flag = 0;

    // Launch kernel
    mine_kernel<<<1, 256>>>(nonce_start, d_nonce, d_flag);
    cudaDeviceSynchronize();

    // Copy results to output params
    *result_nonce = *d_nonce;
    *result_flag = *d_flag;

    cudaFree(d_nonce);
    cudaFree(d_flag);
}
