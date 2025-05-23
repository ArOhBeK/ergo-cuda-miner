#include "autolykos2_cuda_miner.cuh"
#include "blake2b_cuda.cuh"
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>

// You may need to tweak these
#define THREADS_PER_BLOCK 256

__global__ void mining_kernel(
    const uint8_t* d_header,
    const uint8_t* d_target,
    uint64_t nonceStart,
    uint64_t nonceRange,
    uint64_t* d_foundNonce,
    uint8_t* d_foundHash,
    int* d_found
) {
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nonceRange) return;

    uint64_t nonce = nonceStart + idx;
    uint8_t input[32 + 8];
    for (int i = 0; i < 32; i++) input[i] = d_header[i];
    for (int i = 0; i < 8; i++) input[32 + i] = ((nonce >> (8 * i)) & 0xff);

    uint8_t hash[32];
    blake2b_cuda(hash, input, 40); // Assumes your blake2b_cuda is available

    // Compare hash with target (big endian)
    bool less = false;
    for (int i = 0; i < 32; i++) {
        if (hash[i] < d_target[i]) { less = true; break; }
        if (hash[i] > d_target[i]) { break; }
    }

    if (less) {
        // Only one winner: atomic
        if (atomicExch(d_found, 1) == 0) {
            for (int i = 0; i < 32; i++) d_foundHash[i] = hash[i];
            *d_foundNonce = nonce;
        }
    }
}

bool launchMiningKernel(
    const uint8_t* header,
    const uint8_t* target,
    uint64_t nonceStart,
    uint64_t nonceRange,
    uint64_t& foundNonce,
    uint8_t* foundHash
) {
    uint8_t* d_header;    // 32
    uint8_t* d_target;    // 32
    uint64_t* d_foundNonce;
    uint8_t* d_foundHash;
    int* d_found;

    cudaMalloc(&d_header, 32);
    cudaMalloc(&d_target, 32);
    cudaMalloc(&d_foundNonce, sizeof(uint64_t));
    cudaMalloc(&d_foundHash, 32);
    cudaMalloc(&d_found, sizeof(int));

    cudaMemcpy(d_header, header, 32, cudaMemcpyHostToDevice);
    cudaMemcpy(d_target, target, 32, cudaMemcpyHostToDevice);
    int zero = 0;
    cudaMemcpy(d_found, &zero, sizeof(int), cudaMemcpyHostToDevice);

    int blocks = (nonceRange + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    mining_kernel<<<blocks, THREADS_PER_BLOCK>>>(
        d_header, d_target, nonceStart, nonceRange, d_foundNonce, d_foundHash, d_found
    );
    cudaDeviceSynchronize();

    int h_found;
    cudaMemcpy(&h_found, d_found, sizeof(int), cudaMemcpyDeviceToHost);
    bool found = h_found != 0;

    if (found) {
        cudaMemcpy(&foundNonce, d_foundNonce, sizeof(uint64_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(foundHash, d_foundHash, 32, cudaMemcpyDeviceToHost);
    }

    cudaFree(d_header);
    cudaFree(d_target);
    cudaFree(d_foundNonce);
    cudaFree(d_foundHash);
    cudaFree(d_found);

    return found;
}
