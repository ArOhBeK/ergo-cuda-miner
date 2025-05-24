#include "stratum_client.h"
#include <nlohmann/json.hpp>
#include <sstream>
#include <iomanip>
#include <iostream>
#include <cstring>
#include <netdb.h>
#include <unistd.h>

using json = nlohmann::json;

StratumClient::StratumClient(const std::string& host, int port, const std::string& address)
    : host_(host), port_(port), miner_wallet(address) {}

bool StratumClient::connect_to_pool() {
    struct addrinfo hints{}, *res;
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    std::string port_str = std::to_string(port_);

    if (getaddrinfo(host_.c_str(), port_str.c_str(), &hints, &res) != 0) {
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
    json subscribe = {
        {"id", 1}, {"jsonrpc", "2.0"}, {"method", "mining.subscribe"}, {"params", {"ErgoMiner/0.1"}}
    };
    send_json(subscribe);

    std::string response;
    if (!read_line(response)) return false;

    json authorize = {
        {"id", 2}, {"jsonrpc", "2.0"}, {"method", "mining.authorize"}, {"params", {miner_wallet, "x"}}
    };
    send_json(authorize);

    return read_line(response);
}

bool StratumClient::wait_for_job(Job& job) {
    std::string response;
    while (true) {
        if (!read_line(response)) return false;
        auto js = json::parse(response, nullptr, false);
        if (js.is_discarded()) continue;
        if (js.contains("method") && js["method"] == "mining.notify") {
            auto params = js["params"];
            if (!params.is_array() || params.size() < 5) continue;
            job.job_id = params[0].get<std::string>();
            std::string header_hex = params[2].get<std::string>();
            job.header_hash = hex_to_bytes(header_hex);
            // Target handling for pool: default to easy target if not provided
            job.share_target.assign(32, 0xFF);
            // Optionally, parse more fields if your pool supports them
            return true;
        }
    }
}

void StratumClient::submit_share(const std::string& job_id, uint64_t nonce, const Job& job) {
    char nonce_hex[17];
    snprintf(nonce_hex, sizeof(nonce_hex), "%016llx", (unsigned long long)nonce);

    nlohmann::json share = {
        {"id", 4},
        {"jsonrpc", "2.0"},
        {"method", "mining.submit"},
        {"params", {miner_wallet, job_id, "", bytes_to_hex(job.header_hash), nonce_hex}}
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
    std::stringstream ss;
    ss << "[STATS] Accepted: " << accepted_shares
       << " | Rejected: " << rejected_shares
       << " | Total: " << (accepted_shares + rejected_shares) << std::endl;
    return ss.str();
}

void StratumClient::close() {
    if (sockfd != -1) {
        ::close(sockfd);
        sockfd = -1;
    }
}

int StratumClient::get_accepted_shares() const { return accepted_shares; }
int StratumClient::get_rejected_shares() const { return rejected_shares; }
