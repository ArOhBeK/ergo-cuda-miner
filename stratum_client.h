#ifndef STRATUM_CLIENT_H
#define STRATUM_CLIENT_H

#include <string>
#include <vector>
#include <cstdint>
#include <nlohmann/json.hpp>
#include "job.h"

class StratumClient {
public:
    StratumClient(const std::string& host, int port, const std::string& address);
    bool connect_to_pool();
    bool subscribe_and_authorize();
    bool wait_for_job(Job& job);
    void submit_share(const std::string& job_id, uint64_t nonce, const Job& job);
    void close();

    std::string get_gpu_stats();
    int get_accepted_shares() const;
    int get_rejected_shares() const;

private:
    std::string host_;
    int port_;
    std::string miner_wallet;
    int sockfd = -1;
    int accepted_shares = 0;
    int rejected_shares = 0;

    std::vector<uint8_t> hex_to_bytes(const std::string& hex);
    std::string bytes_to_hex(const std::vector<uint8_t>& bytes);
    bool read_line(std::string& out);
    bool send_json(const nlohmann::json& j);
};

#endif // STRATUM_CLIENT_H
