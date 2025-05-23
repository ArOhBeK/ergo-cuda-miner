#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <nlohmann/json.hpp>
#include <chrono>
#include <thread>
#include <iomanip>
#include "stratum_client.h"
#include "autolykos2_cuda_miner.cuh"

using json = nlohmann::json;

int main() {
    std::ifstream configFile("config.json");
    if (!configFile.is_open()) {
        std::cerr << "[ERROR] Failed to open config.json" << std::endl;
        return 1;
    }

    json config;
    configFile >> config;
    std::string mode = config["mode"];
    std::string address = config["address"];

    std::cout << "[MAIN] Ergo Miner Starting in " << mode << " mode" << std::endl;

    if (mode == "pool") {
        std::string poolHost = config["pool"]["host"];
        int poolPort = config["pool"]["port"];
        std::cout << "[POOL] Connecting to mining pool at " << poolHost << ":" << poolPort << std::endl;

        StratumClient client(poolHost, poolPort, address);
        if (!client.connect_to_pool()) {
            std::cerr << "[POOL] Failed to connect to pool." << std::endl;
            return 1;
        }

        if (!client.subscribe_and_authorize()) {
            std::cerr << "[POOL] Subscription/authorization failed." << std::endl;
            return 1;
        }

        while (true) {
            Job job;
            if (!client.wait_for_job(job)) {
                std::this_thread::sleep_for(std::chrono::seconds(2));
                continue;
            }

            std::cout << "[JOB] Mining job: " << job.job_id << std::endl;

            uint64_t baseNonce = 0;
            const uint64_t pageSize = 65536;
            const uint64_t maxRange = 1 << 24; // ~16M nonces/job

            bool found = false;
            uint64_t foundNonce = 0;
            uint8_t foundHash[32];

            auto startTime = std::chrono::high_resolution_clock::now();
            uint64_t totalHashes = 0;

            for (uint64_t offset = 0; offset < maxRange; offset += pageSize) {
                uint64_t nonceStart = baseNonce + offset;
                // Use the pool share_target for GPU mining!
                if (launchMiningKernel(
                        job.header_hash.data(),
                        job.share_target.data(),
                        nonceStart,
                        pageSize,
                        foundNonce,
                        foundHash
                    )) {
                    found = true;
                    totalHashes += (foundNonce - nonceStart + 1); // Only count up to the found nonce
                    break;
                }
                totalHashes += pageSize;
            }

            auto endTime = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed = endTime - startTime;
            double hashrate = totalHashes / elapsed.count();
            double mh = hashrate / 1e6;
            double gh = hashrate / 1e9;

            if (found) {
                client.submit_share(job.job_id, foundNonce, job);
            } else {
                std::cout << "[INFO] No valid nonce found." << std::endl;
            }

            std::cout << "[PERF] Hashes: " << totalHashes
                      << " | Time: " << elapsed.count() << "s"
                      << " | Rate: " << std::fixed << std::setprecision(2)
                      << mh << " MH/s (" << gh << " GH/s)" << std::endl;

            std::cout << client.get_gpu_stats();
        }
    } else if (mode == "solo") {
        std::string nodeHost = config["solo"]["host"];
        int nodePort = config["solo"]["port"];
        std::cout << "[SOLO] Connecting to Ergo node at " << nodeHost << ":" << nodePort << std::endl;
        // TODO: Implement solo mining loop
    } else {
        std::cerr << "[ERROR] Unknown mode in config.json" << std::endl;
        return 1;
    }

    return 0;
}
