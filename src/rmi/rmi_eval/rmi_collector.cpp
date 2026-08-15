#include <algorithm>
#include <cstdint>
#include <fstream>
#include <filesystem>
#include <iomanip>
#include <limits>
#include <unordered_map>
#include <iostream>
#include <stdexcept>
#include <sstream>
#include <string>
#include <vector>

// Auto-generated per configuration:
//   #include "<generated_rmi>.h"
//   namespace rmi_ns = <generated_namespace>;
#include "rmi_wrapper.h"

using KeyType = uint64_t;

struct Args {
    std::string data_file;
    std::string rmi_data_dir;
    std::string query_file;
    std::string out_csv;
    bool has_header = true;   // SOSD-style: first uint64_t is count
    bool use_successor = false; // if true, use lower_bound; else predecessor position
    size_t query_limit = 0;
    size_t branch_factor = 0;
    double max_error_fraction = 0.25;
    double max_dominant_leaf_ratio = 0.99;
};

static Args parse_args(int argc, char** argv) {
    if (argc < 5) {
        throw std::runtime_error(
            "usage: <binary_file> <data_file> <rmi_data_dir> <query_file> <out_csv>"
            " [--no-header] [--successor] [--query-limit <n>] [--branch-factor <n>] [--max-error-fraction <f>] [--max-dominant-leaf-ratio <f>]");
    }
    Args args;
    args.data_file = argv[1];
    args.rmi_data_dir = argv[2];
    args.query_file = argv[3];
    args.out_csv = argv[4];

    for (int i = 5; i < argc; ++i) {
        std::string flag = argv[i];
        if (flag == "--no-header") {
            args.has_header = false;
        } else if (flag == "--successor") {
            args.use_successor = true;
        } else if (flag == "--query-limit") {
            if (i + 1 >= argc) {
                throw std::runtime_error("missing value for --query-limit");
            }
            args.query_limit = std::stoull(argv[++i]);
        } else if (flag == "--branch-factor") {
            if (i + 1 >= argc) throw std::runtime_error("missing value for --branch-factor");
            args.branch_factor = std::stoull(argv[++i]);
        } else if (flag == "--max-error-fraction") {
            if (i + 1 >= argc) throw std::runtime_error("missing value for --max-error-fraction");
            args.max_error_fraction = std::stod(argv[++i]);
        } else if (flag == "--max-dominant-leaf-ratio") {
            if (i + 1 >= argc) throw std::runtime_error("missing value for --max-dominant-leaf-ratio");
            args.max_dominant_leaf_ratio = std::stod(argv[++i]);
        } else {
            throw std::runtime_error("unknown flag: " + flag);
        }
    }
    if (!(args.max_error_fraction > 0.0 && args.max_error_fraction <= 1.0)) {
        throw std::runtime_error("--max-error-fraction must be in (0, 1]");
    }
    if (!(args.max_dominant_leaf_ratio > 0.0 && args.max_dominant_leaf_ratio <= 1.0)) {
        throw std::runtime_error("--max-dominant-leaf-ratio must be in (0, 1]");
    }
    return args;
}

static std::vector<KeyType> load_data(const std::string& path, bool has_header) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("failed to open data file: " + path);

    std::vector<KeyType> data;
    if (has_header) {
        uint64_t n = 0;
        in.read(reinterpret_cast<char*>(&n), sizeof(uint64_t));
        if (!in) throw std::runtime_error("failed to read data header from: " + path);
        data.resize(static_cast<size_t>(n));
        in.read(reinterpret_cast<char*>(data.data()), sizeof(KeyType) * data.size());
        if (!in) throw std::runtime_error("failed to read data payload from: " + path);
    } else {
        in.seekg(0, std::ios::end);
        size_t bytes = static_cast<size_t>(in.tellg());
        in.seekg(0, std::ios::beg);
        if (bytes % sizeof(KeyType) != 0) {
            throw std::runtime_error("data file size is not a multiple of KeyType size: " + path);
        }
        data.resize(bytes / sizeof(KeyType));
        in.read(reinterpret_cast<char*>(data.data()), bytes);
        if (!in) throw std::runtime_error("failed to read data file: " + path);
    }
    return data;
}

static std::vector<KeyType> load_queries(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("failed to open query file: " + path);

    in.seekg(0, std::ios::end);
    size_t bytes = static_cast<size_t>(in.tellg());
    in.seekg(0, std::ios::beg);
    if (bytes % sizeof(KeyType) != 0) {
        throw std::runtime_error("query file size is not a multiple of KeyType size: " + path);
    }

    std::vector<KeyType> queries(bytes / sizeof(KeyType));
    in.read(reinterpret_cast<char*>(queries.data()), bytes);
    if (!in) throw std::runtime_error("failed to read query file: " + path);
    return queries;
}

static size_t true_position_predecessor(const std::vector<KeyType>& data, KeyType key) {
    auto it = std::upper_bound(data.begin(), data.end(), key);
    if (it == data.begin()) return 0;
    return static_cast<size_t>((it - data.begin()) - 1);
}

static size_t true_position_successor(const std::vector<KeyType>& data, KeyType key) {
    auto it = std::lower_bound(data.begin(), data.end(), key);
    if (it == data.end()) return data.size() - 1;
    return static_cast<size_t>(it - data.begin());
}

// -----------------------------------------------------------------------------
// IMPORTANT:
// The generated RMI code already computes the leaf model index internally.
// To collect leaf_id, we expose a helper function in rmi_wrapper.h.
// See the wrapper template shown below the main program.
// -----------------------------------------------------------------------------

int main(int argc, char** argv) {
    try {
        Args args = parse_args(argc, argv);

        auto data = load_data(args.data_file, args.has_header);
        auto queries = load_queries(args.query_file);
        if (args.query_limit > 0 && queries.size() > args.query_limit) {
            queries.resize(args.query_limit);
        }
        if (data.empty()) throw std::runtime_error("data is empty");

        if (!rmi_ns::load(args.rmi_data_dir.c_str())) {
            throw std::runtime_error("rmi_ns::load failed for dir: " + args.rmi_data_dir);
        }

        const std::filesystem::path output_path(args.out_csv);
        const std::filesystem::path body_path(args.out_csv + ".tmp");
        std::error_code remove_error;
        std::filesystem::remove(output_path, remove_error);
        std::filesystem::remove(body_path, remove_error);

        std::ofstream body(body_path);
        if (!body) throw std::runtime_error("failed to open temporary collector output: " + body_path.string());

        const size_t max_allowed_error = static_cast<size_t>(
            args.max_error_fraction * static_cast<double>(data.size()));
        size_t max_error = 0;
        size_t large_error_queries = 0;
        std::unordered_map<size_t, size_t> leaf_counts;

        for (KeyType key : queries) {
            size_t err = 0;
            size_t leaf_id = 0;
            uint64_t pred = rmi_ns::lookup_with_leaf(static_cast<uint64_t>(key), &err, &leaf_id);
            if (args.branch_factor > 0 && leaf_id >= args.branch_factor) {
                throw std::runtime_error("generated RMI returned an out-of-range leaf id");
            }

            size_t pos = args.use_successor
                ? true_position_successor(data, key)
                : true_position_predecessor(data, key);

            max_error = std::max(max_error, err);
            if (err > max_allowed_error) ++large_error_queries;
            ++leaf_counts[leaf_id];
            body << pos << ","
                 << leaf_id << ","
                 << err << ","
                 << pred << ","
                 << key << "\n";
        }
        body.close();
        if (!body) throw std::runtime_error("failed while writing temporary collector output");
        rmi_ns::cleanup();

        size_t dominant_leaf_queries = 0;
        for (const auto& item : leaf_counts) {
            dominant_leaf_queries = std::max(dominant_leaf_queries, item.second);
        }
        const double dominant_leaf_ratio = queries.empty()
            ? 0.0
            : static_cast<double>(dominant_leaf_queries) / static_cast<double>(queries.size());
        const bool excessive_error = max_error > max_allowed_error;
        const bool collapsed_routing =
            args.branch_factor > 1 && dominant_leaf_ratio > args.max_dominant_leaf_ratio;

        if (excessive_error || collapsed_routing) {
            std::filesystem::remove(body_path, remove_error);
            std::ostringstream reason;
            reason << "degenerate RMI records rejected: max_error=" << max_error
                   << " allowed_error=" << max_allowed_error
                   << " large_error_queries=" << large_error_queries
                   << " used_leaves=" << leaf_counts.size()
                   << " dominant_leaf_ratio=" << std::fixed << std::setprecision(6)
                   << dominant_leaf_ratio;
            throw std::runtime_error(reason.str());
        }

        std::ofstream out(output_path);
        if (!out) throw std::runtime_error("failed to open output csv: " + args.out_csv);
        out << "#name," << rmi_ns::NAME << "\n";
        out << "#rmi_size," << rmi_ns::RMI_SIZE << "\n";
        out << "#build_time_ns," << rmi_ns::BUILD_TIME_NS << "\n";
        out << "#num_data," << data.size() << "\n";
        out << "#num_queries," << queries.size() << "\n";
        out << "#branch_factor," << args.branch_factor << "\n";
        out << "#used_leaves," << leaf_counts.size() << "\n";
        out << "#max_error," << max_error << "\n";
        out << "#max_allowed_error," << max_allowed_error << "\n";
        out << "#large_error_queries," << large_error_queries << "\n";
        out << "#dominant_leaf_ratio," << std::setprecision(17) << dominant_leaf_ratio << "\n";
        out << "#degenerate,0\n";
        out << "true_pos,leaf_id,err,pred_pos,key\n";

        std::ifstream body_input(body_path);
        if (!body_input) throw std::runtime_error("failed to reopen temporary collector output");
        out << body_input.rdbuf();
        out.close();
        if (!out) throw std::runtime_error("failed while publishing collector output");
        std::filesystem::remove(body_path, remove_error);

        std::cerr << "[rmi_collector] records accepted: max_error=" << max_error
                  << " used_leaves=" << leaf_counts.size()
                  << " dominant_leaf_ratio=" << std::fixed << std::setprecision(6)
                  << dominant_leaf_ratio << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[rmi_collector] " << e.what() << std::endl;
        return 1;
    }
}

/*
==========================
Suggested rmi_wrapper.h
==========================

This wrapper is auto-generated per configuration. It includes the generated
RMI header and adds a small adapter API with a stable namespace alias.

Example content:

#pragma once
#include "books_rmi_linear_spline_linear_64.h"
namespace rmi_ns = books_rmi_linear_spline_linear_64;

// The generated header normally exposes:
//   bool load(char const* dataPath);
//   void cleanup();
//   uint64_t lookup(uint64_t key, size_t* err);
//   const size_t RMI_SIZE;
//   const uint64_t BUILD_TIME_NS;
//   const char NAME[];
//
// To collect leaf_id, we need one extra function. There are two options:
//
// Option A (recommended): patch the generated .cpp/.h and export:
//   uint64_t lookup_with_leaf(uint64_t key, size_t* err, size_t* leaf);
//
// Option B: if the generated code structure is stable, auto-generate a wrapper
//           .cpp that re-implements the root-stage routing and then calls into
//           the same parameter arrays. This is fragile across model types.
//
// For your current 2-layer linear_spline/linear experiments, Option A is much
// simpler and more reliable.
*/
