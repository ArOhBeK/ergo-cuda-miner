#ifndef STRATUM_CLIENT_H
#define STRATUM_CLIENT_H

#include <string>
#include <vector>
#include <nlohmann/json.hpp>

struct Job {
    std::string job_id;
    std::vector<uint8_t> header_hash;
    std::vector<uint8_t> target_difficulty;
};

class StratumClient {
public:
    StratumClient(const std::string& host, int port, const std::string& address);

    bool connect_to_pool();
    bool subscribe_and_authorize();
    bool wait_for_job(Job& job);
    void submit_share(const std::string& job_id, uint64_t nonce, const Job& job);
    void close();

    int get_accepted_shares() const;
    int get_rejected_shares() const;
    std::string get_gpu_stats();

private:
    std::string pool_host;
    int pool_port;
    std::string miner_address;
    int sockfd;

    int accepted_shares;
    int rejected_shares;

    bool read_line(std::string& out);
    bool send_json(const nlohmann::json& j);

    std::vector<uint8_t> hex_to_bytes(const std::string& hex);
    std::string bytes_to_hex(const std::vector<uint8_t>& bytes);
};

#endif // STRATUM_CLIENT_H
