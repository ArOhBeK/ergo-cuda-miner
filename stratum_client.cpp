extern "C" {
#include "blake2b.h"
}

#include "stratum_client.h"
#include "utils.h"
#include <iostream>
#include <sstream>
#include <iomanip>
#include <cstring>
#include <netdb.h>
#include <unistd.h>
#include <cstdlib>
#include <gmp.h>
#include <vector>
#include <string>

std::vector<uint8_t> decimal_to_target_bytes(const std::string& decimal) {
    mpz_t target_int;
    mpz_init(target_int);
    mpz_set_str(target_int, decimal.c_str(), 10);
    uint8_t buf[32] = {0};
    size_t count = 0;
    mpz_export(buf, &count, 1, 1, 1, 0, target_int);
    std::vector<uint8_t> out(32, 0);
    if (count > 0) {
        memcpy(out.data() + (32 - count), buf, count);
    }
    mpz_clear(target_int);
    return out;
}

StratumClient::StratumClient(const std::string& host, int port, const std::string& address)
    : pool_host(host), pool_port(port), miner_address(address), sockfd(-1), accepted_shares(0), rejected_shares(0) {}

bool StratumClient::connect_to_pool() {
    struct addrinfo hints{}, *res;
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    std::string port_str = std::to_string(pool_port);

    if (getaddrinfo(pool_host.c_str(), port_str.c_str(), &hints, &res) != 0) {
        std::cerr << "[-] getaddrinfo failed\n";
        return false;
    }
    for (auto ai = res; ai != nullptr; ai = ai->ai_next) {
        sockfd = socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol);
        if (sockfd < 0) continue;
        if (connect(sockfd, ai->ai_addr, ai->ai_addrlen) == 0) break;
        ::close(sockfd);
        sockfd = -1;
    }
    freeaddrinfo(res);
    return sockfd != -1;
}

bool StratumClient::read_line(std::string& out) {
    out.clear();
    char c;
    while (read(sockfd, &c, 1) == 1) {
        if (c == '\n') break;
        out += c;
    }
    return !out.empty();
}

bool StratumClient::send_json(const nlohmann::json& j) {
    std::string msg = j.dump() + "\n";
    return send(sockfd, msg.c_str(), msg.size(), 0) == (ssize_t)msg.size();
}

bool StratumClient::subscribe_and_authorize() {
    nlohmann::json subscribe = {
        {"id", 1}, {"jsonrpc", "2.0"}, {"method", "mining.subscribe"}, {"params", {"ErgoMiner/0.1"}}
    };
    send_json(subscribe);

    std::string response;
    if (!read_line(response)) return false;

    nlohmann::json authorize = {
        {"id", 2}, {"jsonrpc", "2.0"}, {"method", "mining.authorize"}, {"params", {miner_address, "x"}}
    };
    send_json(authorize);

    return read_line(response);
}

bool StratumClient::wait_for_job(Job& job) {
    std::string response;
    if (!read_line(response)) return false;

    nlohmann::json j = nlohmann::json::parse(response);
    if (j.contains("method") && j["method"] == "mining.notify") {
        auto p = j["params"];
        job.job_id = p[0];
        job.header_hash = hex_to_bytes(p[2]);

        // WoolyPooly: [6]=block target, [7]=share difficulty
        std::string block_target_str = p[6];
        std::string share_diff_str = (p.size() > 7) ? p[7].get<std::string>() : "";

        job.block_target = decimal_to_target_bytes(block_target_str);
        if (!share_diff_str.empty() && share_diff_str != "null") {
            job.share_target = decimal_to_target_bytes(share_diff_str);
        } else {
            job.share_target = job.block_target;
        }
        return true;
    }
    return false;
}

void StratumClient::submit_share(const std::string& job_id, uint64_t nonce, const Job& job) {
    char nonce_hex[17];
    snprintf(nonce_hex, sizeof(nonce_hex), "%016llx", (unsigned long long)nonce);

    std::vector<uint8_t> nonce_le(8);
    for (int i = 0; i < 8; ++i)
        nonce_le[i] = (nonce >> (8 * i)) & 0xFF;

    std::vector<uint8_t> input = job.header_hash;
    input.insert(input.end(), nonce_le.begin(), nonce_le.end());

    uint8_t hash_output[32];
    blake2b(hash_output, 32, input.data(), input.size(), nullptr, 0);

    nlohmann::json share = {
        {"id", 4},
        {"jsonrpc", "2.0"},
        {"method", "mining.submit"},
        {"params", {miner_address, job_id, "", bytes_to_hex(job.header_hash), nonce_hex}}
    };

    send_json(share);

    std::string response;
    read_line(response);
    std::cout << "[>] Share response: " << response << std::endl;

    if (response.find("\"error\":null") != std::string::npos ||
        response.find("\"result\":true") != std::string::npos) {
        ++accepted_shares;
    } else {
        ++rejected_shares;
    }
}

std::vector<uint8_t> StratumClient::hex_to_bytes(const std::string& hex) {
    std::vector<uint8_t> bytes;
    for (size_t i = 0; i < hex.size(); i += 2) {
        bytes.push_back(static_cast<uint8_t>(strtol(hex.substr(i, 2).c_str(), nullptr, 16)));
    }
    return bytes;
}

std::string StratumClient::bytes_to_hex(const std::vector<uint8_t>& bytes) {
    std::ostringstream oss;
    for (uint8_t b : bytes) {
        oss << std::hex << std::setw(2) << std::setfill('0') << (int)b;
    }
    return oss.str();
}

std::string StratumClient::get_gpu_stats() {
    FILE* pipe = popen("/usr/bin/nvidia-smi --query-gpu=temperature.gpu,fan.speed,power.draw,utilization.gpu,clocks.gr --format=csv,noheader,nounits", "r");
    if (!pipe) return "[GPU] Failed to get stats.";
    char buffer[256];
    std::string result = "[GPU] ";
    while (fgets(buffer, sizeof(buffer), pipe)) {
        result += buffer;
    }
    pclose(pipe);
    return result;
}

void StratumClient::close() {
    if (sockfd != -1) {
        ::close(sockfd);
        sockfd = -1;
    }
}

int StratumClient::get_accepted_shares() const {
    return accepted_shares;
}

int StratumClient::get_rejected_shares() const {
    return rejected_shares;
}
