#ifndef AUTOLOKYOS2_CUDA_MINER_H
#define AUTOLOKYOS2_CUDA_MINER_H

#include <cstdint>
#include <vector>
#include <array>

uint64_t launch_gpu_miner(
    const std::vector<std::array<uint8_t, 32>>& dag,
    const std::vector<uint8_t>& header,
    const std::vector<uint8_t>& target
);

#endif // AUTOLOKYOS2_CUDA_MINER_H
