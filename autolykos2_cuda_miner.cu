// autolykos2_cuda_miner.cu
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdint>
#include <vector>
#include <array>
#include <cstdio>
#include <cstring>
#include "autolykos2_cuda_miner.cuh"

__device__ __forceinline__ uint64_t ROTR64(uint64_t x, uint64_t y) {
    return (x >> y) ^ (x << (64 - y));
}

__device__ void blake2b_hash(const uint8_t* input, size_t input_len, uint8_t* output) {
    const uint64_t blake2b_iv[8] = {
        0x6a09e667f3bcc908ULL, 0xbb67ae8584caa73bULL,
        0x3c6ef372fe94f82bULL, 0xa54ff53a5f1d36f1ULL,
        0x510e527fade682d1ULL, 0x9b05688c2b3e6c1fULL,
        0x1f83d9abfb41bd6bULL, 0x5be0cd19137e2179ULL
    };

    uint64_t h[8];
    for (int i = 0; i < 8; ++i) h[i] = blake2b_iv[i];

    uint64_t m[16] = {0};
    for (size_t i = 0; i < input_len && i < 128; ++i) {
        reinterpret_cast<uint8_t*>(m)[i] = input[i];
    }

    const uint8_t sigma[12][16] = {
        { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15 },
        {14,10, 4, 8, 9,15,13, 6, 1,12, 0, 2,11, 7, 5, 3 },
        {11, 8,12, 0, 5, 2,15,13,10,14, 3, 6, 7, 1, 9, 4 },
        { 7, 9, 3, 1,13,12,11,14, 2, 6, 5,10, 4, 0,15, 8 },
        { 9, 0, 5, 7, 2, 4,10,15,14, 1,11,12, 6, 8, 3,13 },
        { 2,12, 6,10, 0,11, 8, 3, 4,13, 7, 5,15,14, 1, 9 },
        {12, 5, 1,15,14,13, 4,10, 0, 7, 6, 3, 9, 2, 8,11 },
        {13,11, 7,14,12, 1, 3, 9, 5, 0,15, 4, 8, 6, 2,10 },
        { 6,15,14, 9,11, 3, 0, 8,12, 2,13, 7, 1, 4,10, 5 },
        {10, 2, 8, 4, 7, 6, 1, 5,15,11, 9,14, 3,12,13, 0 },
        { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15 },
        {14,10, 4, 8, 9,15,13, 6, 1,12, 0, 2,11, 7, 5, 3 }
    };

    auto G = [&](uint64_t& a, uint64_t& b, uint64_t& c, uint64_t& d, uint64_t x, uint64_t y) {
        a += b + x;
        d = ROTR64(d ^ a, 32);
        c += d;
        b = ROTR64(b ^ c, 24);
        a += b + y;
        d = ROTR64(d ^ a, 16);
        c += d;
        b = ROTR64(b ^ c, 63);
    };

    for (int round = 0; round < 12; ++round) {
        uint64_t v[16];
        for (int i = 0; i < 8; ++i) v[i] = h[i];
        for (int i = 0; i < 8; ++i) v[i + 8] = blake2b_iv[i];

        G(v[0], v[4], v[8], v[12], m[sigma[round][0]], m[sigma[round][1]]);
        G(v[1], v[5], v[9], v[13], m[sigma[round][2]], m[sigma[round][3]]);
        G(v[2], v[6], v[10], v[14], m[sigma[round][4]], m[sigma[round][5]]);
        G(v[3], v[7], v[11], v[15], m[sigma[round][6]], m[sigma[round][7]]);
        G(v[0], v[5], v[10], v[15], m[sigma[round][8]], m[sigma[round][9]]);
        G(v[1], v[6], v[11], v[12], m[sigma[round][10]], m[sigma[round][11]]);
        G(v[2], v[7], v[8], v[13], m[sigma[round][12]], m[sigma[round][13]]);
        G(v[3], v[4], v[9], v[14], m[sigma[round][14]], m[sigma[round][15]]);

        for (int i = 0; i < 8; ++i) h[i] ^= v[i] ^ v[i + 8];
    }

    memcpy(output, h, 32);
}

__global__ void miner_kernel(const uint8_t* d_table, const uint8_t* d_header, const uint8_t* d_target, size_t table_size, uint64_t* d_nonce_out, int* d_found) {
    uint64_t nonce = blockIdx.x * blockDim.x + threadIdx.x;

    if (threadIdx.x == 0 && blockIdx.x % 10 == 0) {
        printf("[GPU] Trying nonce: %llu\n", nonce);
    }

    uint8_t local_input[76];
    memcpy(local_input, d_header, 32);
    *((uint64_t*)(local_input + 32)) = nonce;
    for (int i = 0; i < 32; ++i) local_input[40 + i] = d_table[i];

    uint8_t hash[32];
    blake2b_hash(local_input, 72, hash);

    bool valid = true;
    for (int i = 0; i < 32; ++i) {
        if (hash[i] < d_target[i]) break;
        if (hash[i] > d_target[i]) {
            valid = false;
            break;
        }
    }

    if (valid && atomicCAS(d_found, 0, 1) == 0) {
        *d_nonce_out = nonce;
    }
}

uint64_t launch_gpu_miner(
    const std::vector<std::array<uint8_t, 32>>& dag,
    const std::vector<uint8_t>& header,
    const std::vector<uint8_t>& target
) {
    const size_t THREADS_PER_BLOCK = 256;
    const size_t NUM_BLOCKS = 64;

    std::vector<uint8_t> flat_dag(dag.size() * 32);
    for (size_t i = 0; i < dag.size(); ++i) {
        std::copy(dag[i].begin(), dag[i].end(), flat_dag.begin() + i * 32);
    }

    uint8_t *d_table, *d_header, *d_target;
    uint64_t* d_nonce;
    int* d_found;

    cudaMalloc(&d_table, flat_dag.size());
    cudaMalloc(&d_header, header.size());
    cudaMalloc(&d_target, target.size());
    cudaMalloc(&d_nonce, sizeof(uint64_t));
    cudaMalloc(&d_found, sizeof(int));

    cudaMemcpy(d_table, flat_dag.data(), flat_dag.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_header, header.data(), header.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_target, target.data(), target.size(), cudaMemcpyHostToDevice);
    cudaMemset(d_found, 0, sizeof(int));

    miner_kernel<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(d_table, d_header, d_target, dag.size(), d_nonce, d_found);
    cudaDeviceSynchronize();

    int found = 0;
    uint64_t nonce = 0;
    cudaMemcpy(&found, d_found, sizeof(int), cudaMemcpyDeviceToHost);
    if (found) {
        cudaMemcpy(&nonce, d_nonce, sizeof(uint64_t), cudaMemcpyDeviceToHost);
        printf("[+] GPU found valid nonce: %llu\n", (unsigned long long)nonce);
    }

    cudaFree(d_table);
    cudaFree(d_header);
    cudaFree(d_target);
    cudaFree(d_nonce);
    cudaFree(d_found);

    return nonce;
}
