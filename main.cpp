// main.cpp
#include "stratum_client.h"
#include "dag_generator.h"
#include "autolykos2_cuda_miner.cuh"
#include <iostream>
#include <vector>
#include <array>
#include <csignal>

volatile std::sig_atomic_t stop_flag = 0;

void handle_signal(int) {
    stop_flag = 1;
}

int main() {
    signal(SIGINT, handle_signal);

    const std::string pool_host = "pool.us.woolypooly.com";
    const int pool_port = 3100;
    const std::string miner_address = "9h3dCuaU9BkyriZi2EG4xDagckZ1vGiT8xpXwdvvGtWkH9FnhgZ.Arohbe";

    StratumClient client(pool_host, pool_port, miner_address);

    if (!client.connect_to_pool()) {
        std::cerr << "[-] Failed to connect to pool." << std::endl;
        return 1;
    }

    if (!client.subscribe_and_authorize()) {
        std::cerr << "[-] Failed to subscribe or authorize." << std::endl;
        return 1;
    }

    std::cout << "[*] Connection verified. Generating DAG..." << std::endl;
    std::vector<std::array<uint8_t, 32>> dag_table = generate_full_dag();
    std::cout << "[*] DAG ready. Waiting for job..." << std::endl;

    while (!stop_flag) {
        Job job;
        if (!client.wait_for_job(job)) continue;

        std::cout << "[*] Launching GPU mining job: " << job.job_id << std::endl;
        uint64_t nonce = launch_gpu_miner(dag_table, job.header_hash, job.target_difficulty);

        std::cout << "[*] Nonce found: " << nonce << ". Submitting share..." << std::endl;
        client.submit_share(job.job_id, nonce, job);
    }

    client.close();
    std::cout << "[*] Miner exited." << std::endl;
    return 0;
}
