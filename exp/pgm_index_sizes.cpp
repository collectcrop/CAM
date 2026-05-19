#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "../src/pgm/pgm_index.hpp"
#include "../src/storage/KeyFile.hpp"
#include "../src/utils.hpp"

namespace fs = std::filesystem;
using KeyType = uint64_t;

namespace {

struct Config {
    std::string data_path;
    std::string output_path;
    size_t total_keys = 0;
    std::vector<size_t> epsilons;
};

class RuntimePGMIndex : public pgm::PGMIndex<KeyType, 1, 4, float> {
public:
    explicit RuntimePGMIndex(const std::vector<KeyType>& data, size_t epsilon) {
        if (epsilon == 0) {
            throw std::invalid_argument("epsilon must be > 0");
        }
        this->n = data.size();
        this->first_key = data.empty() ? KeyType(0) : data[0];
        this->build(data.begin(), data.end(), epsilon, 4, this->segments, this->levels_offsets);
    }
};

[[noreturn]] void usage_error(const std::string& msg) {
    throw std::invalid_argument(
        msg +
        "\nUsage: ./pgm_index_sizes --data <file> [--keys <n>] --epsilons <e1,e2,...>"
        " [--output <csv>]");
}

std::string trim(std::string s) {
    const auto not_space = [](unsigned char c) { return !std::isspace(c); };
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), not_space));
    s.erase(std::find_if(s.rbegin(), s.rend(), not_space).base(), s.end());
    return s;
}

std::vector<size_t> parse_size_list(const std::string& value) {
    std::vector<size_t> out;
    std::stringstream ss(value);
    std::string token;
    while (std::getline(ss, token, ',')) {
        token = trim(token);
        if (token.empty()) {
            continue;
        }
        const size_t parsed = std::stoull(token);
        if (parsed == 0) {
            throw std::invalid_argument("epsilon must be > 0");
        }
        if (std::find(out.begin(), out.end(), parsed) == out.end()) {
            out.push_back(parsed);
        }
    }
    if (out.empty()) {
        throw std::invalid_argument("empty epsilon list");
    }
    return out;
}

size_t detect_record_count(const std::string& filename) {
    const auto bytes = fs::file_size(filename);
    if (bytes % sizeof(KeyType) != 0) {
        throw std::runtime_error("data file size is not a multiple of key size");
    }
    return bytes / sizeof(KeyType);
}

void ensure_parent_dir(const std::string& output_path) {
    const fs::path path(output_path);
    const fs::path parent = path.parent_path();
    if (!parent.empty()) {
        fs::create_directories(parent);
    }
}

Config parse_args(int argc, char** argv) {
    Config cfg;
    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);
        auto require_value = [&](const char* flag) -> std::string {
            if (i + 1 >= argc) {
                usage_error(std::string("missing value for ") + flag);
            }
            return argv[++i];
        };

        if (arg == "--data") {
            cfg.data_path = cam::storage::resolve_dataset_path(require_value("--data"));
        } else if (arg == "--keys") {
            cfg.total_keys = std::stoull(require_value("--keys"));
        } else if (arg == "--epsilons") {
            cfg.epsilons = parse_size_list(require_value("--epsilons"));
        } else if (arg == "--output") {
            cfg.output_path = require_value("--output");
        } else if (arg == "-h" || arg == "--help") {
            usage_error("help requested");
        } else {
            usage_error("unknown argument: " + arg);
        }
    }

    if (cfg.data_path.empty()) {
        usage_error("--data is required");
    }
    if (cfg.epsilons.empty()) {
        usage_error("--epsilons is required");
    }
    if (cfg.total_keys == 0) {
        cfg.total_keys = detect_record_count(cfg.data_path);
    }
    return cfg;
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Config cfg = parse_args(argc, argv);
        const auto data = load_data_pgm_safe<KeyType>(cfg.data_path, cfg.total_keys);

        std::ofstream output_file;
        std::ostream* out = &std::cout;
        if (!cfg.output_path.empty()) {
            ensure_parent_dir(cfg.output_path);
            output_file.open(cfg.output_path, std::ios::out | std::ios::trunc);
            if (!output_file) {
                throw std::runtime_error("failed to open output: " + cfg.output_path);
            }
            out = &output_file;
        }

        *out << "epsilon,measured_index_bytes\n";
        for (size_t epsilon : cfg.epsilons) {
            RuntimePGMIndex index(data, epsilon);
            *out << epsilon << ',' << index.size_in_bytes() << '\n';
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[error] " << e.what() << '\n';
        return 1;
    }
}
