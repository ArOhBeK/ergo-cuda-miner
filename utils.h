#ifndef UTILS_H
#define UTILS_H

#include <stdint.h>
#include <vector>
#include <string>
#include <gmp.h>

// Only in utils.cpp!
std::vector<uint8_t> decimal_to_target_bytes(const std::string& decimal);

inline bool isHashLessThanTarget(const uint8_t* hash, const uint8_t* target) {
    #pragma unroll
    for (int i = 0; i < 32; i++) {
        if (hash[i] < target[i]) return true;
        if (hash[i] > target[i]) return false;
    }
    return false;
}

#endif // UTILS_H
