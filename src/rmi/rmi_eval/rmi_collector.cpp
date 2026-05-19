#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
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
};

static Args parse_args(int argc, char** argv) {
    if (argc < 5) {
        throw std::runtime_error(
            "usage: <binary_file> <data_file> <rmi_data_dir> <query_file> <out_csv>"
            " [--no-header] [--successor] [--query-limit <n>]");
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
        } else {
            throw std::runtime_error("unknown flag: " + flag);
        }
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

        std::ofstream out(args.out_csv);
        if (!out) throw std::runtime_error("failed to open output csv: " + args.out_csv);

        // Metadata block as commented CSV lines.
        out << "#name," << rmi_ns::NAME << "\n";
        out << "#rmi_size," << rmi_ns::RMI_SIZE << "\n";
        out << "#build_time_ns," << rmi_ns::BUILD_TIME_NS << "\n";
        out << "#num_data," << data.size() << "\n";
        out << "#num_queries," << queries.size() << "\n";
        out << "true_pos,leaf_id,err,pred_pos,key\n";

        for (KeyType key : queries) {
            size_t err = 0;
            size_t leaf_id = 0;
            uint64_t pred = rmi_ns::lookup_with_leaf(static_cast<uint64_t>(key), &err, &leaf_id);

            size_t pos = args.use_successor
                ? true_position_successor(data, key)
                : true_position_predecessor(data, key);

            out << pos << ','
                << leaf_id << ','
                << err << ','
                << pred << ','
                << key << '\n';
        }

        rmi_ns::cleanup();
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
