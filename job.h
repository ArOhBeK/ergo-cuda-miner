#ifndef JOB_H
#define JOB_H

#include <string>
#include <vector>
#include <cstdint>

struct Job {
    std::string job_id;
    int height;
    std::vector<uint8_t> header_hash;
    std::vector<uint8_t> target_difficulty;
};

#endif // JOB_H

