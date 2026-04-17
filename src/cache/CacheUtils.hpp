#pragma once

#include <algorithm>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "../include.hpp"
#include "../storage/KeyFile.hpp"

namespace cam::cache {

inline CachePolicy parse_policy_token(const std::string& value) {
    const std::string token = cam::storage::to_upper(cam::storage::trim(value));
    if (token == "FIFO") return CachePolicy::FIFO;
    if (token == "LRU") return CachePolicy::LRU;
    if (token == "LFU") return CachePolicy::LFU;
    if (token == "NONE") return CachePolicy::NONE;
    throw std::invalid_argument("unknown cache policy: " + value);
}

inline std::vector<CachePolicy> parse_policy_list(const std::string& value) {
    const std::string upper = cam::storage::to_upper(cam::storage::trim(value));
    if (upper == "ALL") {
        return {CachePolicy::FIFO, CachePolicy::LRU, CachePolicy::LFU};
    }

    std::vector<CachePolicy> out;
    std::stringstream ss(value);
    std::string token;
    while (std::getline(ss, token, ',')) {
        token = cam::storage::trim(token);
        if (token.empty()) {
            continue;
        }
        const CachePolicy policy = parse_policy_token(token);
        if (std::find(out.begin(), out.end(), policy) == out.end()) {
            out.push_back(policy);
        }
    }
    if (out.empty()) {
        throw std::invalid_argument("empty policy list");
    }
    return out;
}

inline std::string policy_name(CachePolicy policy) {
    switch (policy) {
        case CachePolicy::NONE: return "NONE";
        case CachePolicy::FIFO: return "FIFO";
        case CachePolicy::LRU: return "LRU";
        case CachePolicy::LFU: return "LFU";
        default: return "UNKNOWN";
    }
}

} // namespace cam::cache
