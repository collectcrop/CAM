#pragma once

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "../include.hpp"

namespace cam::storage {

namespace fs = std::filesystem;

enum class HeaderMode {
    AUTO,
    YES,
    NO
};

struct KeyFileLayout {
    std::string path;
    size_t total_keys = 0;
    size_t header_bytes = 0;
    size_t logical_pages = 0;

    size_t logical_bytes() const {
        return total_keys * sizeof(KeyType);
    }
};

inline std::string trim(std::string s) {
    const auto not_space = [](unsigned char c) { return !std::isspace(c); };
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), not_space));
    s.erase(std::find_if(s.rbegin(), s.rend(), not_space).base(), s.end());
    return s;
}

inline std::string to_upper(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
        return static_cast<char>(std::toupper(c));
    });
    return s;
}

inline std::string resolve_dataset_path(const std::string& value) {
    fs::path path(value);
    if (path.is_absolute()) {
        return path.string();
    }
    if (fs::exists(path)) {
        return fs::absolute(path).string();
    }
    return (fs::path(DATASETS) / path).string();
}

inline HeaderMode parse_header_mode(const std::string& value) {
    const std::string token = to_upper(trim(value));
    if (token == "AUTO") return HeaderMode::AUTO;
    if (token == "YES" || token == "TRUE" || token == "1") return HeaderMode::YES;
    if (token == "NO" || token == "FALSE" || token == "0") return HeaderMode::NO;
    throw std::invalid_argument("unknown header mode: " + value);
}

inline size_t read_u64_at(const std::string& path, size_t offset_bytes) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("failed to open file for header detection: " + path);
    }
    in.seekg(static_cast<std::streamoff>(offset_bytes), std::ios::beg);
    uint64_t value = 0;
    in.read(reinterpret_cast<char*>(&value), sizeof(value));
    if (!in) {
        throw std::runtime_error("failed to read header candidate from: " + path);
    }
    return static_cast<size_t>(value);
}

inline KeyFileLayout make_layout(const std::string& path, size_t keys, size_t header_bytes) {
    return KeyFileLayout{
        path,
        keys,
        header_bytes,
        (keys * sizeof(KeyType) + PAGE_SIZE - 1) / PAGE_SIZE
    };
}

inline KeyFileLayout detect_key_file_layout(
    const std::string& path,
    size_t explicit_keys = 0,
    HeaderMode header_mode = HeaderMode::AUTO)
{
    const size_t file_bytes = static_cast<size_t>(fs::file_size(path));
    const bool can_be_plain = file_bytes % sizeof(KeyType) == 0;
    const bool can_be_header = file_bytes >= sizeof(uint64_t) &&
        (file_bytes - sizeof(uint64_t)) % sizeof(KeyType) == 0;

    if (explicit_keys > 0) {
        if (header_mode == HeaderMode::YES) {
            const size_t expected = sizeof(uint64_t) + explicit_keys * sizeof(KeyType);
            if (file_bytes != expected) {
                throw std::runtime_error("file size does not match explicit key count with header");
            }
            return make_layout(path, explicit_keys, sizeof(uint64_t));
        }
        if (header_mode == HeaderMode::NO) {
            const size_t expected = explicit_keys * sizeof(KeyType);
            if (file_bytes != expected) {
                throw std::runtime_error("file size does not match explicit key count without header");
            }
            return make_layout(path, explicit_keys, 0);
        }

        const size_t plain_bytes = explicit_keys * sizeof(KeyType);
        const size_t header_bytes = sizeof(uint64_t) + plain_bytes;
        if (file_bytes == header_bytes) {
            return make_layout(path, explicit_keys, sizeof(uint64_t));
        }
        if (file_bytes == plain_bytes) {
            return make_layout(path, explicit_keys, 0);
        }
        throw std::runtime_error("file size does not match explicit key count in auto header mode");
    }

    if (header_mode == HeaderMode::YES) {
        if (!can_be_header) {
            throw std::runtime_error("file cannot be parsed as header + keys");
        }
        return make_layout(path, (file_bytes - sizeof(uint64_t)) / sizeof(KeyType), sizeof(uint64_t));
    }
    if (header_mode == HeaderMode::NO) {
        if (!can_be_plain) {
            throw std::runtime_error("file cannot be parsed as plain keys");
        }
        return make_layout(path, file_bytes / sizeof(KeyType), 0);
    }

    if (can_be_header) {
        const size_t header_keys = read_u64_at(path, 0);
        if (header_keys == (file_bytes - sizeof(uint64_t)) / sizeof(KeyType)) {
            return make_layout(path, header_keys, sizeof(uint64_t));
        }
    }

    if (!can_be_plain) {
        throw std::runtime_error("file size is not a multiple of key size");
    }
    return make_layout(path, file_bytes / sizeof(KeyType), 0);
}

inline std::vector<KeyType> load_query_keys(const std::string& path, size_t limit = 0) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("failed to open query file: " + path);
    }

    in.seekg(0, std::ios::end);
    const size_t bytes = static_cast<size_t>(in.tellg());
    in.seekg(0, std::ios::beg);
    if (bytes % sizeof(KeyType) != 0) {
        throw std::runtime_error("query file size is not a multiple of key size");
    }

    size_t query_count = bytes / sizeof(KeyType);
    if (limit > 0) {
        query_count = std::min(query_count, limit);
    }

    std::vector<KeyType> queries(query_count);
    in.read(reinterpret_cast<char*>(queries.data()),
            static_cast<std::streamsize>(query_count * sizeof(KeyType)));
    if (!in) {
        throw std::runtime_error("failed to read query file: " + path);
    }

    constexpr KeyType sentinel = std::numeric_limits<KeyType>::max();
    for (KeyType& key : queries) {
        if (key == sentinel) {
            key = sentinel - 1;
        }
    }
    return queries;
}

inline std::vector<KeyType> load_key_file_keys(
    const KeyFileLayout& layout,
    size_t limit = 0,
    bool fix_sentinel = true)
{
    std::ifstream in(layout.path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("failed to open key file: " + layout.path);
    }

    in.seekg(static_cast<std::streamoff>(layout.header_bytes), std::ios::beg);
    size_t key_count = layout.total_keys;
    if (limit > 0) {
        key_count = std::min(key_count, limit);
    }

    std::vector<KeyType> keys(key_count);
    in.read(reinterpret_cast<char*>(keys.data()),
            static_cast<std::streamsize>(key_count * sizeof(KeyType)));
    if (!in) {
        throw std::runtime_error("failed to read key file: " + layout.path);
    }

    if (fix_sentinel) {
        constexpr KeyType sentinel = std::numeric_limits<KeyType>::max();
        for (KeyType& key : keys) {
            if (key == sentinel) {
                key = sentinel - 1;
            }
        }
    }
    return keys;
}

} // namespace cam::storage
