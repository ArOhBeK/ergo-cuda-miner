#include "stratum_client.h"
#include "blake2b.h"
#include <iostream>
#include <sstream>
#include <iomanip>
#include <cstring>
#include <netdb.h>
#include <unistd.h>
#include <cstdlib>

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
        job.target_difficulty = hex_to_bytes(p[6]);
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

    std::cout << "[DEBUG] Job ID: " << job_id << "\n";
    std::cout << "[DEBUG] Nonce: " << nonce << "\n";
    std::cout << "[DEBUG] Nonce LE: " << bytes_to_hex(nonce_le) << "\n";
    std::cout << "[DEBUG] Header: " << bytes_to_hex(job.header_hash) << "\n";
    std::cout << "[DEBUG] Hash: ";
    for (int i = 0; i < 32; ++i) std::cout << std::hex << std::setw(2) << std::setfill('0') << (int)hash_output[i];
    std::cout << "\n";
    std::cout << "[DEBUG] Target: " << bytes_to_hex(job.target_difficulty) << "\n";

    if (memcmp(hash_output, job.target_difficulty.data(), 32) >= 0) {
        std::cout << "[-] Rejected by target difficulty\n";
        ++rejected_shares;
        std::cout << get_gpu_stats();
        std::cout << "[SUMMARY] Accepted: " << accepted_shares << " | Rejected: " << rejected_shares << "\n";
        return;
    }

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

    if (response.find("\"error\":null") != std::string::npos || response.find("\"result\":true") != std::string::npos) {
        ++accepted_shares;
    } else {
        ++rejected_shares;
    }

    std::cout << get_gpu_stats();
    std::cout << "[SUMMARY] Accepted: " << accepted_shares << " | Rejected: " << rejected_shares << "\n";
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
    FILE* pipe = popen("/usr/bin/nvidia-smi --query-gpu=temperature.gpu,power.draw,utilization.gpu,clocks.sm --format=csv,noheader,nounits", "r");
    if (!pipe) return "[GPU] Failed to get GPU stats.\n";

    char buffer[256];
    std::string stats = "[GPU] ";
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
        std::istringstream iss(buffer);
        int temp, util, clock;
        float power;
        iss >> temp;
        iss.ignore(); // ,
        iss >> power;
        iss.ignore(); // ,
        iss >> util;
        iss.ignore(); // ,
        iss >> clock;

        stats += "Temp: " + std::to_string(temp) + "°C, ";
        stats += "Power: " + std::to_string(power) + "W, ";
        stats += "Util: " + std::to_string(util) + "%, ";
        stats += "Clock: " + std::to_string(clock) + " MHz\n";
    }
    pclose(pipe);
    return stats;
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
