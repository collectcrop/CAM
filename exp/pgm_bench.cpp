#include <algorithm>
#include <chrono>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "../src/cache/CacheUtils.hpp"
#include "../src/pgm/pgm_index.hpp"
#include "../src/pgm/PointQuery.hpp"
#include "../src/storage/DiskManager.hpp"
#include "../src/storage/KeyFile.hpp"
#include "../src/utils.hpp"

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;
using KeyType = uint64_t;
using IndexBase = pgm::PGMIndex<KeyType>;

namespace {

struct Config {
    std::string data_path;
    std::string query_path;
    size_t total_keys = 0;
    size_t M = 64ULL << 20;
    std::vector<size_t> epsilons = {4, 8, 10, 12, 14, 16, 18, 20, 24, 32, 64, 128};
    std::vector<SearchStrategy> strategies = {ALL_IN_ONCE};
};

struct RunResult {
    CachePolicy policy = CachePolicy::NONE;
    SearchStrategy strategy = ALL_IN_ONCE;
    size_t epsilon = 0;

    size_t queries = 0;
    size_t found = 0;
    uint64_t checksum = 0;

    size_t page_requests = 0;
    size_t cache_hits = 0;
    size_t cache_misses = 0;

    size_t logical_ios = 0;
    size_t physical_ios = 0;
    uint64_t bytes_read = 0;
    long long io_ns = 0;
    long long wall_ns = 0;
};

[[noreturn]] void usage_error(const std::string& msg) {
    throw std::invalid_argument(
        msg + "\nUsage: ./pgm_bench --data <file> --queries <file> [--keys <n>] [--M <MiB>]"
              " [--epsilons <e1,e2,...>] [--strategies <all_in_once,one_by_one|all>]");
}

std::vector<size_t> parse_size_list(const std::string& value) {
    std::vector<size_t> out;
    std::stringstream ss(value);
    std::string token;
    while (std::getline(ss, token, ',')) {
        token = cam::storage::trim(token);
        if (token.empty()) {
            continue;
        }
        const size_t parsed = std::stoull(token);
        if (std::find(out.begin(), out.end(), parsed) == out.end()) {
            out.push_back(parsed);
        }
    }
    if (out.empty()) {
        throw std::invalid_argument("empty epsilon list");
    }
    return out;
}

Config parse_args(int argc, char** argv) {
    Config cfg;
    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);
        auto require_value = [&](const char* flag) -> std::string {
            if (i + 1 >= argc) usage_error(std::string("missing value for ") + flag);
            return argv[++i];
        };

        if (arg == "--data") {
            cfg.data_path = cam::storage::resolve_dataset_path(require_value("--data"));
        } else if (arg == "--queries") {
            cfg.query_path = cam::storage::resolve_dataset_path(require_value("--queries"));
        } else if (arg == "--keys") {
            cfg.total_keys = std::stoull(require_value("--keys"));
        } else if (arg == "--M") {
            cfg.M = std::stoull(require_value("--M")) << 20;
        } else if (arg == "--epsilons") {
            cfg.epsilons = parse_size_list(require_value("--epsilons"));
        } else if (arg == "--strategies") {
            cfg.strategies = cam::point_query::parse_search_strategy_list(require_value("--strategies"));
        } else if (arg == "-h" || arg == "--help") {
            usage_error("help requested");
        } else {
            usage_error("unknown argument: " + arg);
        }
    }

    if (cfg.data_path.empty() || cfg.query_path.empty()) {
        usage_error("both --data and --queries are required");
    }
    return cfg;
}

template <size_t Eps>
RunResult run_one_policy(
    CachePolicy policy,
    SearchStrategy strategy,
    const cam::storage::KeyFileLayout& data_layout,
    const std::vector<KeyType>& data,
    const std::vector<KeyType>& queries,
    const size_t index_bytes,
    const Config& cfg)
{
    using Index = pgm::PGMIndex<KeyType, Eps>;
    (void)index_bytes;

    RunResult st;
    st.policy = policy;
    st.strategy = strategy;
    st.epsilon = Eps;
    st.queries = queries.size();

    Index index(data);
    const size_t estimated_index_bytes = 16 * data.size() / (2 * Eps);
    const size_t cache_bytes =
        cfg.M > estimated_index_bytes ? cfg.M - estimated_index_bytes : 0;
    cam::storage::DiskManager disk(
        data_layout,
        cam::storage::make_page_cache(policy, cache_bytes));

    auto t0 = Clock::now();
    for (KeyType q : queries) {
        const auto result = cam::point_query::run_point_query(index, disk, q, strategy);
        st.page_requests += result.metrics.dac;
        st.cache_hits += result.metrics.buffer_hits;
        st.cache_misses += result.metrics.cam_io;
        st.logical_ios += result.metrics.disk_pages_read;
        st.physical_ios += result.metrics.device_ios;
        st.bytes_read += result.metrics.bytes_read;
        st.io_ns += result.metrics.io_ns;
        if (result.found) {
            ++st.found;
            st.checksum += result.matched_key;
        }
    }
    auto t1 = Clock::now();
    st.wall_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    return st;
}

void print_header() {
    std::cout
        << "epsilon,policy,strategy,queries,found,page_requests,cache_hits,cache_misses,hit_ratio,"
        << "logical_ios,physical_ios,avg_logical_ios,avg_physical_ios,"
        << "bytes_read,io_ns,wall_ns,throughput_qps,index_bytes,checksum\n";
}

template <size_t Eps>
void print_row(const RunResult& st, size_t index_bytes) {
    const double hit_ratio =
        st.page_requests ? static_cast<double>(st.cache_hits) / st.page_requests : 0.0;
    const double avg_lio =
        st.queries ? static_cast<double>(st.logical_ios) / st.queries : 0.0;
    const double avg_pio =
        st.queries ? static_cast<double>(st.physical_ios) / st.queries : 0.0;
    const double qps =
        st.wall_ns ? static_cast<double>(st.queries) * 1e9 / st.wall_ns : 0.0;

    std::cout
        << Eps << ','
        << cam::cache::policy_name(st.policy) << ','
        << cam::point_query::search_strategy_name(st.strategy) << ','
        << st.queries << ','
        << st.found << ','
        << st.page_requests << ','
        << st.cache_hits << ','
        << st.cache_misses << ','
        << std::fixed << std::setprecision(6) << hit_ratio << ','
        << st.logical_ios << ','
        << st.physical_ios << ','
        << std::fixed << std::setprecision(6) << avg_lio << ','
        << std::fixed << std::setprecision(6) << avg_pio << ','
        << st.bytes_read << ','
        << st.io_ns << ','
        << st.wall_ns << ','
        << std::fixed << std::setprecision(2) << qps << ','
        << index_bytes << ','
        << st.checksum
        << '\n';
}

template <size_t Eps>
void run_epsilon(
    const cam::storage::KeyFileLayout& data_layout,
    const std::vector<KeyType>& data,
    const std::vector<KeyType>& queries,
    const Config& cfg)
{
    using Index = pgm::PGMIndex<KeyType, Eps>;
    Index index(data);
    const size_t index_bytes = index.size_in_bytes();

    for (SearchStrategy strategy : cfg.strategies) {
        for (CachePolicy p : {CachePolicy::FIFO, CachePolicy::LRU, CachePolicy::LFU}) {
            auto st = run_one_policy<Eps>(p, strategy, data_layout, data, queries, index_bytes, cfg);
            print_row<Eps>(st, index_bytes);
        }
    }
}

void run_epsilon_value(
    size_t epsilon,
    const cam::storage::KeyFileLayout& data_layout,
    const std::vector<KeyType>& data,
    const std::vector<KeyType>& queries,
    const Config& cfg)
{
    switch (epsilon) {
        case 4: run_epsilon<4>(data_layout, data, queries, cfg); break;
        case 8: run_epsilon<8>(data_layout, data, queries, cfg); break;
        case 10: run_epsilon<10>(data_layout, data, queries, cfg); break;
        case 12: run_epsilon<12>(data_layout, data, queries, cfg); break;
        case 14: run_epsilon<14>(data_layout, data, queries, cfg); break;
        case 16: run_epsilon<16>(data_layout, data, queries, cfg); break;
        case 18: run_epsilon<18>(data_layout, data, queries, cfg); break;
        case 20: run_epsilon<20>(data_layout, data, queries, cfg); break;
        case 24: run_epsilon<24>(data_layout, data, queries, cfg); break;
        case 28: run_epsilon<28>(data_layout, data, queries, cfg); break;
        case 32: run_epsilon<32>(data_layout, data, queries, cfg); break;
        case 36: run_epsilon<36>(data_layout, data, queries, cfg); break;
        case 40: run_epsilon<40>(data_layout, data, queries, cfg); break;
        case 52: run_epsilon<52>(data_layout, data, queries, cfg); break;
        case 64: run_epsilon<64>(data_layout, data, queries, cfg); break;
        case 96: run_epsilon<96>(data_layout, data, queries, cfg); break;
        case 128: run_epsilon<128>(data_layout, data, queries, cfg); break;
        default:
            throw std::invalid_argument("unsupported epsilon for pgm_bench: " + std::to_string(epsilon));
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        Config cfg = parse_args(argc, argv);
        const auto data_layout = cam::storage::detect_key_file_layout(cfg.data_path, cfg.total_keys);
        cfg.total_keys = data_layout.total_keys;
        auto data = cam::storage::load_key_file_keys(data_layout);
        auto queries = load_queries_pgm_safe<KeyType>(cfg.query_path);

        print_header();
        for (size_t epsilon : cfg.epsilons) {
            run_epsilon_value(epsilon, data_layout, data, queries, cfg);
        }

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[error] " << e.what() << '\n';
        return 1;
    }
}
