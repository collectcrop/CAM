#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "../src/cache/CacheUtils.hpp"
#include "../src/pgm/RangeQuery.hpp"
#include "../src/pgm/PointQuery.hpp"
#include "../src/pgm/pgm_index.hpp"
#include "../src/storage/DiskManager.hpp"
#include "../src/storage/KeyFile.hpp"
#include "../src/utils.hpp"

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;
using KeyT = uint64_t;

namespace {

constexpr size_t kEstimatedSegmentBytes = 16;

enum class Mode { POINT, RANGE };

struct Config {
    std::string data_path;
    std::optional<std::string> query_path;
    std::string output_path;
    Mode mode = Mode::POINT;
    size_t epsilon = 16;
    size_t M = 256ULL << 20;
    size_t total_keys = 0;
    size_t num_queries = 0;
    std::vector<size_t> range_page_spans;
    size_t range_repeats = 8;
    bool append_output = false;
    CachePolicy policy = CachePolicy::LRU;
};

[[noreturn]] void usage_error(const std::string& msg) {
    throw std::invalid_argument(
        msg +
        "\nUsage: ./pgm_join_fit --data <file> --mode <point|range>"
        " --output <csv>"
        " [--query <file>]"
        " [--epsilon <n>] [--M <MiB>] [--keys <n>]"
        " [--num-queries <n>]"
        " [--range-page-spans <k1,k2|quoted list>] [--range-repeats <n>]"
        " [--policy <lru|fifo|lfu>] [--append]");
}

size_t detect_record_count(const std::string& filename) {
    const auto bytes = fs::file_size(filename);
    if (bytes % sizeof(KeyT) != 0) {
        throw std::runtime_error("data file size is not a multiple of key size");
    }
    return bytes / sizeof(KeyT);
}

size_t estimate_index_bytes(size_t total_keys, size_t epsilon) {
    return (kEstimatedSegmentBytes * total_keys) / (2 * epsilon);
}

size_t safe_subtract(size_t lhs, size_t rhs) {
    return lhs > rhs ? lhs - rhs : 0;
}

std::vector<size_t> parse_size_list(const std::string& value) {
    std::vector<size_t> out;
    std::string token;
    std::stringstream ss(value);
    while (ss >> token) {
        size_t start = 0;
        while (start < token.size()) {
            const size_t comma = token.find(',', start);
            const std::string part = token.substr(
                start,
                comma == std::string::npos ? std::string::npos : comma - start);
            if (!part.empty()) {
                out.push_back(std::stoull(part));
            }
            if (comma == std::string::npos) {
                break;
            }
            start = comma + 1;
        }
    }
    if (out.empty()) {
        throw std::invalid_argument("empty size list");
    }
    return out;
}

Config parse_args(int argc, char** argv) {
    Config cfg;
    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);
        auto require = [&](const char* flag) -> std::string {
            if (i + 1 >= argc)
                usage_error(std::string("missing value for ") + flag);
            return argv[++i];
        };

        if (arg == "--data") {
            cfg.data_path = cam::storage::resolve_dataset_path(require("--data"));
        } else if (arg == "--query") {
            cfg.query_path = cam::storage::resolve_dataset_path(require("--query"));
        } else if (arg == "--output") {
            cfg.output_path = require("--output");
        } else if (arg == "--mode") {
            const std::string m = require("--mode");
            if (m == "point") cfg.mode = Mode::POINT;
            else if (m == "range") cfg.mode = Mode::RANGE;
            else usage_error("unknown mode: " + m);
        } else if (arg == "--epsilon") {
            cfg.epsilon = std::stoull(require("--epsilon"));
        } else if (arg == "--M") {
            cfg.M = std::stoull(require("--M")) << 20;
        } else if (arg == "--keys") {
            cfg.total_keys = std::stoull(require("--keys"));
        } else if (arg == "--num-queries") {
            cfg.num_queries = std::stoull(require("--num-queries"));
        } else if (arg == "--range-page-spans") {
            cfg.range_page_spans = parse_size_list(require("--range-page-spans"));
        } else if (arg == "--range-repeats") {
            cfg.range_repeats = std::stoull(require("--range-repeats"));
        } else if (arg == "--policy") {
            cfg.policy = cam::cache::parse_policy_token(require("--policy"));
        } else if (arg == "--append") {
            cfg.append_output = true;
        } else if (arg == "-h" || arg == "--help") {
            usage_error("help requested");
        } else {
            usage_error("unknown argument: " + arg);
        }
    }

    if (cfg.data_path.empty()) usage_error("--data is required");
    if (cfg.output_path.empty()) usage_error("--output is required");
    if (cfg.mode == Mode::POINT && !cfg.query_path.has_value())
        usage_error("--query is required");
    if (cfg.mode == Mode::POINT && cfg.num_queries == 0)
        usage_error("--num-queries is required for point mode");
    if (cfg.mode == Mode::RANGE && cfg.range_page_spans.empty())
        usage_error("--range-page-spans is required for range mode");
    if (cfg.mode == Mode::RANGE && cfg.range_repeats == 0)
        usage_error("--range-repeats must be positive");
    if (cfg.total_keys == 0)
        cfg.total_keys = detect_record_count(cfg.data_path);
    return cfg;
}

void ensure_parent_dir(const std::string& path) {
    const fs::path p(path);
    const fs::path parent = p.parent_path();
    if (!parent.empty()) fs::create_directories(parent);
}

// --- Output ---

void write_point_csv_header(std::ostream& out) {
    out << "epsilon,total_wall_time_s,IO_time_s,IOs,"
        << "DAC,cache_hit_ratio,mem_time_s,IO_fraction,num_queries\n";
}

void write_csv_row(std::ostream& out,
                   size_t epsilon,
                   long long total_wall_ns,
                   long long total_io_ns,
                   size_t total_ios,
                   size_t total_dac,
                   size_t total_hits,
                   size_t num_queries)
{
    const double wall_s = total_wall_ns / 1e9;
    const double io_s = total_io_ns / 1e9;
    const double mem_s = wall_s - io_s;
    const double io_frac = wall_s > 0 ? io_s / wall_s : 0.0;
    const double hit = total_dac == 0 ? 0.0
        : static_cast<double>(total_hits) / static_cast<double>(total_dac);

    out << std::fixed << std::setprecision(10)
        << epsilon << ','
        << wall_s << ','
        << io_s << ','
        << total_ios << ','
        << total_dac << ','
        << hit << ','
        << mem_s << ','
        << io_frac << ','
        << num_queries << '\n';
}

void write_range_csv_header(std::ostream& out) {
    out << "epsilon,sample_idx,target_range_pages,query_lo,query_hi,range_pages,"
        << "total_wall_time_s,IO_time_s,IOs,DAC,cache_hit_ratio,"
        << "mem_time_s,IO_fraction,matched_records,"
        << "logical_pages_read,physical_ios,bytes_read\n";
}

void write_range_csv_row(std::ostream& out,
                         size_t epsilon,
                         size_t sample_idx,
                         size_t target_range_pages,
                         KeyT query_lo,
                         KeyT query_hi,
                         size_t range_pages,
                         long long total_wall_ns,
                         const cam::range_query::RangeQueryMetrics& metrics,
                         size_t matched_records)
{
    const double wall_s = total_wall_ns / 1e9;
    const double io_s = metrics.io_ns / 1e9;
    const double mem_s = wall_s - io_s;
    const double io_frac = wall_s > 0 ? io_s / wall_s : 0.0;
    const double hit = metrics.dac == 0 ? 0.0
        : static_cast<double>(metrics.buffer_hits) / static_cast<double>(metrics.dac);

    out << std::fixed << std::setprecision(10)
        << epsilon << ','
        << sample_idx << ','
        << target_range_pages << ','
        << query_lo << ','
        << query_hi << ','
        << range_pages << ','
        << wall_s << ','
        << io_s << ','
        << metrics.cam_io << ','
        << metrics.dac << ','
        << hit << ','
        << mem_s << ','
        << io_frac << ','
        << matched_records << ','
        << metrics.disk_pages_read << ','
        << metrics.device_ios << ','
        << metrics.bytes_read << '\n';
}

// --- Point query benchmark (aggregate) ---

template <size_t Epsilon>
void run_point_bench(
    const std::vector<KeyT>& data,
    const std::vector<KeyT>& queries,
    const Config& cfg,
    std::ostream& out)
{
    using Index = pgm::PGMIndex<KeyT, Epsilon>;
    Index index(data);

    const size_t cache_bytes = safe_subtract(cfg.M,
        estimate_index_bytes(data.size(), Epsilon));

    cam::storage::DiskManager disk(
        cam::storage::detect_key_file_layout(cfg.data_path, cfg.total_keys, cam::storage::HeaderMode::NO),
        cam::storage::make_page_cache(cfg.policy, cache_bytes));

    const auto t0 = Clock::now();
    cam::storage::DiskStats before = disk.stats();

    for (KeyT q : queries) {
        cam::point_query::run_point_query(index, disk, q, ALL_IN_ONCE);
    }

    cam::storage::DiskStats after = disk.stats();
    const auto t1 = Clock::now();

    write_csv_row(out,
        cfg.epsilon,
        std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count(),
        after.io_ns - before.io_ns,
        after.cache_misses - before.cache_misses,
        after.page_requests - before.page_requests,
        after.cache_hits - before.cache_hits,
        queries.size());
}

// --- Range query benchmark (per-query) ---

template <size_t Epsilon>
void run_range_bench(
    const std::vector<KeyT>& data,
    const Config& cfg,
    std::ostream& out)
{
    constexpr size_t items_per_page = PAGE_SIZE / sizeof(KeyT);
    static_assert(items_per_page > 0, "items_per_page must be positive");

    using Index = pgm::PGMIndex<KeyT, Epsilon>;
    Index index(data);

    const size_t cache_bytes = safe_subtract(cfg.M,
        estimate_index_bytes(data.size(), Epsilon));

    const auto layout = cam::storage::detect_key_file_layout(
        cfg.data_path,
        cfg.total_keys,
        cam::storage::HeaderMode::NO);

    size_t sample_idx = 0;
    for (size_t target_pages : cfg.range_page_spans) {
        if (target_pages == 0) {
            continue;
        }
        const size_t range_keys = target_pages * items_per_page;
        if (range_keys == 0 || range_keys > data.size()) {
            continue;
        }

        const size_t max_start = data.size() - range_keys;
        const size_t stride = max_start == 0 ? 0 : max_start / (cfg.range_repeats + 1);

        for (size_t repeat = 0; repeat < cfg.range_repeats; ++repeat) {
            const size_t start = stride == 0 ? 0 : stride * (repeat + 1);
            const size_t end = start + range_keys - 1;
            const KeyT lo = data[start];
            const KeyT hi = data[end];

            cam::storage::DiskManager disk(
                layout,
                cam::storage::make_page_cache(cfg.policy, cache_bytes));

            const auto t0 = Clock::now();
            const auto result = cam::range_query::run_range_query_all_at_once(index, disk, lo, hi);
            const auto t1 = Clock::now();

            write_range_csv_row(out,
                cfg.epsilon,
                sample_idx++,
                target_pages,
                lo,
                hi,
                result.metrics.dac,
                std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count(),
                result.metrics,
                result.matched());
        }
    }
}

// --- Dispatch ---

template <typename Fn>
void dispatch_epsilon(size_t epsilon, Fn&& fn) {
#define CAM_DISPATCH_CASE(E) case E: fn(std::integral_constant<size_t, E>{}); break
    switch (epsilon) {
        CAM_DISPATCH_CASE(2);   CAM_DISPATCH_CASE(4);   CAM_DISPATCH_CASE(6);
        CAM_DISPATCH_CASE(8);   CAM_DISPATCH_CASE(10);  CAM_DISPATCH_CASE(12);
        CAM_DISPATCH_CASE(14);  CAM_DISPATCH_CASE(16);  CAM_DISPATCH_CASE(18);
        CAM_DISPATCH_CASE(20);  CAM_DISPATCH_CASE(22);  CAM_DISPATCH_CASE(24);
        CAM_DISPATCH_CASE(26);  CAM_DISPATCH_CASE(28);  CAM_DISPATCH_CASE(30);
        CAM_DISPATCH_CASE(32);  CAM_DISPATCH_CASE(36);  CAM_DISPATCH_CASE(40);
        CAM_DISPATCH_CASE(48);  CAM_DISPATCH_CASE(56);  CAM_DISPATCH_CASE(64);
        CAM_DISPATCH_CASE(72);  CAM_DISPATCH_CASE(80);  CAM_DISPATCH_CASE(96);
        CAM_DISPATCH_CASE(112); CAM_DISPATCH_CASE(128);
        default:
            throw std::invalid_argument(
                "unsupported epsilon: " + std::to_string(epsilon));
    }
#undef CAM_DISPATCH_CASE
}

} // namespace

int main(int argc, char** argv) {
    try {
        Config cfg = parse_args(argc, argv);

        auto data = load_data_pgm_safe<KeyT>(cfg.data_path, cfg.total_keys);

        std::ofstream out_file;
        std::ostream* out = &std::cout;
        if (cfg.output_path != "-") {
            ensure_parent_dir(cfg.output_path);
            auto mode = std::ios::out;
            mode |= cfg.append_output ? std::ios::app : std::ios::trunc;
            out_file.open(cfg.output_path, mode);
            if (!out_file) {
                throw std::runtime_error("failed to open output: " + cfg.output_path);
            }
            out = &out_file;
        }

        const bool write_header = (cfg.output_path == "-")
            || !fs::exists(cfg.output_path)
            || fs::file_size(cfg.output_path) == 0;
        if (write_header) {
            if (cfg.mode == Mode::POINT) {
                write_point_csv_header(*out);
            } else {
                write_range_csv_header(*out);
            }
        }

        if (cfg.mode == Mode::POINT) {
            auto queries = cam::storage::load_query_keys(*cfg.query_path, cfg.num_queries);

            dispatch_epsilon(cfg.epsilon, [&](auto eps_tag) {
                constexpr size_t Eps = decltype(eps_tag)::value;
                run_point_bench<Eps>(data, queries, cfg, *out);
            });
        } else {
            dispatch_epsilon(cfg.epsilon, [&](auto eps_tag) {
                constexpr size_t Eps = decltype(eps_tag)::value;
                run_range_bench<Eps>(data, cfg, *out);
            });
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[error] " << e.what() << '\n';
        return 1;
    }
}
