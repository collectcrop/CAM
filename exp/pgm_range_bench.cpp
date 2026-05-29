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
#include "../src/pgm/RangeQuery.hpp"
#include "../src/storage/DiskManager.hpp"
#include "../src/storage/KeyFile.hpp"
#include "../src/utils.hpp"

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

namespace {

struct Config {
    std::string data_path;
    std::string query_path;
    size_t total_keys = 0;
    size_t memory_budget_bytes = 64ULL << 20;
    std::vector<CachePolicy> policies = {
        CachePolicy::FIFO,
        CachePolicy::LRU,
        CachePolicy::LFU
    };
};

struct RunResult {
    CachePolicy policy = CachePolicy::NONE;
    size_t cache_bytes = 0;

    size_t ranges = 0;
    size_t matched_records = 0;
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
        msg +
        "\nUsage: ./pgm_range_bench --data <file> --queries <range-file>"
        " [--keys <n>] [--M <MiB>] [--policies <fifo,lru,lfu,none|all>]");
}


size_t estimated_index_bytes(size_t total_keys, size_t epsilon) {
    return (16 * total_keys) / (2 * epsilon);
}

size_t safe_subtract(size_t lhs, size_t rhs) {
    return lhs > rhs ? lhs - rhs : 0;
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
        } else if (arg == "--queries") {
            cfg.query_path = cam::storage::resolve_dataset_path(require_value("--queries"));
        } else if (arg == "--keys") {
            cfg.total_keys = std::stoull(require_value("--keys"));
        } else if (arg == "--M") {
            cfg.memory_budget_bytes = std::stoull(require_value("--M")) << 20;
        } else if (arg == "--policies") {
            cfg.policies = cam::cache::parse_policy_list(require_value("--policies"));
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
    const cam::storage::KeyFileLayout& data_layout,
    const std::vector<KeyType>& data,
    const std::vector<RangeQ>& ranges,
    const Config& cfg)
{
    using Index = pgm::PGMIndex<KeyType, Eps>;

    RunResult st;
    st.policy = policy;
    st.ranges = ranges.size();
    st.cache_bytes = safe_subtract(
        cfg.memory_budget_bytes,
        estimated_index_bytes(data.size(), Eps));

    Index index(data);
    cam::storage::DiskManager disk(
        data_layout,
        cam::storage::make_page_cache(policy, st.cache_bytes));

    const auto t0 = Clock::now();
    for (const RangeQ& range : ranges) {
        const auto result = cam::range_query::run_range_query(index, disk, range);
        st.page_requests += result.metrics.dac;
        st.cache_hits += result.metrics.buffer_hits;
        st.cache_misses += result.metrics.cam_io;
        st.logical_ios += result.metrics.disk_pages_read;
        st.physical_ios += result.metrics.device_ios;
        st.bytes_read += result.metrics.bytes_read;
        st.io_ns += result.metrics.io_ns;
        st.matched_records += result.records.size();
        for (const Record& record : result.records) {
            st.checksum += record.key;
        }
    }
    const auto t1 = Clock::now();
    st.wall_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    return st;
}

void print_header() {
    std::cout
        << "epsilon,policy,strategy,ranges,matched_records,avg_records_per_range,"
        << "page_requests,cache_hits,cache_misses,hit_ratio,"
        << "logical_ios,physical_ios,avg_logical_ios,avg_physical_ios,"
        << "bytes_read,io_ns,wall_ns,throughput_qps,index_bytes,cache_bytes,checksum\n";
}

template <size_t Eps>
void print_row(const RunResult& st, size_t index_bytes) {
    const double hit_ratio =
        st.page_requests ? static_cast<double>(st.cache_hits) / st.page_requests : 0.0;
    const double avg_records =
        st.ranges ? static_cast<double>(st.matched_records) / st.ranges : 0.0;
    const double avg_lio =
        st.ranges ? static_cast<double>(st.logical_ios) / st.ranges : 0.0;
    const double avg_pio =
        st.ranges ? static_cast<double>(st.physical_ios) / st.ranges : 0.0;
    const double qps =
        st.wall_ns ? static_cast<double>(st.ranges) * 1e9 / st.wall_ns : 0.0;

    std::cout
        << Eps << ','
        << cam::cache::policy_name(st.policy) << ','
        << "all_in_once" << ','
        << st.ranges << ','
        << st.matched_records << ','
        << std::fixed << std::setprecision(6) << avg_records << ','
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
        << st.cache_bytes << ','
        << st.checksum
        << '\n';
}

template <size_t Eps>
void run_epsilon(
    const cam::storage::KeyFileLayout& data_layout,
    const std::vector<KeyType>& data,
    const std::vector<RangeQ>& ranges,
    const Config& cfg)
{
    using Index = pgm::PGMIndex<KeyType, Eps>;
    Index index(data);
    const size_t index_bytes = index.size_in_bytes();

    for (CachePolicy policy : cfg.policies) {
        const auto st = run_one_policy<Eps>(policy, data_layout, data, ranges, cfg);
        print_row<Eps>(st, index_bytes);
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Config cfg = parse_args(argc, argv);
        const auto data_layout = cam::storage::detect_key_file_layout(cfg.data_path, cfg.total_keys);
        const auto data = cam::storage::load_key_file_keys(data_layout);
        const auto ranges = load_ranges_pgm_safe(cfg.query_path);

        print_header();
        run_epsilon<4>(data_layout, data, ranges, cfg);
        run_epsilon<8>(data_layout, data, ranges, cfg);
        run_epsilon<10>(data_layout, data, ranges, cfg);
        run_epsilon<12>(data_layout, data, ranges, cfg);
        run_epsilon<14>(data_layout, data, ranges, cfg);
        run_epsilon<16>(data_layout, data, ranges, cfg);
        run_epsilon<20>(data_layout, data, ranges, cfg);
        run_epsilon<24>(data_layout, data, ranges, cfg);
        run_epsilon<28>(data_layout, data, ranges, cfg);
        run_epsilon<32>(data_layout, data, ranges, cfg);
        run_epsilon<48>(data_layout, data, ranges, cfg);
        run_epsilon<64>(data_layout, data, ranges, cfg);
        run_epsilon<96>(data_layout, data, ranges, cfg);
        run_epsilon<128>(data_layout, data, ranges, cfg);

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[error] " << e.what() << '\n';
        return 1;
    }
}
