#ifndef JOB_H
#define JOB_H

#include <string>
#include <vector>
#include <cstdint>

// Pool jobs require: header_hash, block_target, share_target
struct Job {
    std::string job_id;
    std::vector<uint8_t> header_hash;      // block header
    std::vector<uint8_t> block_target;     // true network target
    std::vector<uint8_t> share_target;     // pool share target (can be easier)
};

#endif // JOB_H
