#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

#include "../src/cache/CacheUtils.hpp"
#include "../src/pgm/pgm_index.hpp"
#include "../src/pgm/PointQuery.hpp"
#include "../src/sort/ExternalMergeSort.hpp"
#include "../src/storage/DiskManager.hpp"
#include "../src/storage/KeyFile.hpp"

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;
using KeyType = uint64_t;

namespace {

constexpr size_t kDefaultEpsilon = 64;

struct Config {
    std::string data_path;
    std::string query_path;
    std::optional<std::string> summary_out;
    std::vector<size_t> sort_budgets_bytes = {0, 8ULL << 20, 16ULL << 20, 32ULL << 20};
    std::vector<CachePolicy> policies = {CachePolicy::LRU};
    std::string work_dir = "build/tmp/sort_cache_tradeoff";
    size_t total_keys = 0;
    size_t total_budget_bytes = 64ULL << 20;
    size_t query_limit = 0;
    size_t epsilon = kDefaultEpsilon;
    cam::storage::HeaderMode header_mode = cam::storage::HeaderMode::AUTO;
    bool keep_sorted_files = false;
};

struct QueryStats {
    size_t queries = 0;
    size_t found = 0;
    uint64_t checksum = 0;
    cam::storage::DiskStats disk;
    long long wall_ns = 0;

    double hit_ratio() const {
        return disk.page_requests == 0
            ? 0.0
            : static_cast<double>(disk.cache_hits) / static_cast<double>(disk.page_requests);
    }

    double avg_page_requests() const {
        return queries == 0
            ? 0.0
            : static_cast<double>(disk.page_requests) / static_cast<double>(queries);
    }

    double avg_physical_reads() const {
        return queries == 0
            ? 0.0
            : static_cast<double>(disk.physical_read_ops) / static_cast<double>(queries);
    }

    double query_throughput_qps() const {
        return wall_ns == 0
            ? 0.0
            : static_cast<double>(queries) * 1e9 / static_cast<double>(wall_ns);
    }
};

struct ExperimentRow {
    std::string mode;
    CachePolicy policy = CachePolicy::NONE;
    size_t epsilon = kDefaultEpsilon;
    size_t total_budget_bytes = 0;
    size_t sort_budget_bytes = 0;
    size_t cache_bytes = 0;
    size_t total_keys = 0;
    cam::sort::SortStats sort;
    QueryStats query;

    long long end_to_end_ns() const {
        return sort.wall_ns + query.wall_ns;
    }

    double end_to_end_throughput_qps() const {
        const long long total_ns = end_to_end_ns();
        return total_ns == 0
            ? 0.0
            : static_cast<double>(query.queries) * 1e9 / static_cast<double>(total_ns);
    }
};

[[noreturn]] void usage_error(const std::string& msg) {
    throw std::invalid_argument(
        msg +
        "\nUsage: ./sort_cache_tradeoff --data <file> --queries <file> [--keys <n>] [--M <MiB>]"
        " [--sort-mibs <m0,m1,...>] [--policies <fifo,lru,lfu,none|all>] [--epsilon <e>]"
        " [--query-limit <n>] [--summary-out <csv>] [--work-dir <dir>]"
        " [--header-mode <auto|yes|no>] [--keep-sorted-files]");
}

std::vector<size_t> parse_mib_list(const std::string& value) {
    std::vector<size_t> out;
    std::stringstream ss(value);
    std::string token;
    while (std::getline(ss, token, ',')) {
        token = cam::storage::trim(token);
        if (token.empty()) {
            continue;
        }
        const size_t mib = std::stoull(token);
        const size_t bytes = mib << 20;
        if (std::find(out.begin(), out.end(), bytes) == out.end()) {
            out.push_back(bytes);
        }
    }
    if (out.empty()) {
        throw std::invalid_argument("empty sort budget list");
    }
    return out;
}

Config parse_args(int argc, char** argv) {
    Config cfg;
    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);
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
            cfg.total_budget_bytes = std::stoull(require_value("--M")) << 20;
        } else if (arg == "--sort-mibs") {
            cfg.sort_budgets_bytes = parse_mib_list(require_value("--sort-mibs"));
        } else if (arg == "--policies") {
            cfg.policies = cam::cache::parse_policy_list(require_value("--policies"));
        } else if (arg == "--epsilon") {
            cfg.epsilon = std::stoull(require_value("--epsilon"));
        } else if (arg == "--query-limit") {
            cfg.query_limit = std::stoull(require_value("--query-limit"));
        } else if (arg == "--summary-out") {
            cfg.summary_out = require_value("--summary-out");
        } else if (arg == "--work-dir") {
            cfg.work_dir = require_value("--work-dir");
        } else if (arg == "--header-mode") {
            cfg.header_mode = cam::storage::parse_header_mode(require_value("--header-mode"));
        } else if (arg == "--keep-sorted-files") {
            cfg.keep_sorted_files = true;
        } else if (arg == "-h" || arg == "--help") {
            usage_error("help requested");
        } else {
            usage_error("unknown argument: " + arg);
        }
    }

    if (cfg.data_path.empty() || cfg.query_path.empty()) {
        usage_error("both --data and --queries are required");
    }

    std::sort(cfg.sort_budgets_bytes.begin(), cfg.sort_budgets_bytes.end());
    cfg.sort_budgets_bytes.erase(
        std::unique(cfg.sort_budgets_bytes.begin(), cfg.sort_budgets_bytes.end()),
        cfg.sort_budgets_bytes.end());

    for (size_t sort_budget_bytes : cfg.sort_budgets_bytes) {
        if (sort_budget_bytes > cfg.total_budget_bytes) {
            throw std::invalid_argument("sort budget exceeds total budget");
        }
    }
    return cfg;
}

void ensure_parent_dir(const std::string& output_path) {
    const fs::path path(output_path);
    if (!path.parent_path().empty()) {
        fs::create_directories(path.parent_path());
    }
}

fs::path materialize_query_prefix_if_needed(
    const cam::storage::KeyFileLayout& query_layout,
    size_t query_limit,
    const fs::path& work_dir)
{
    if (query_limit == 0 || query_limit >= query_layout.total_keys) {
        return query_layout.path;
    }

    fs::create_directories(work_dir);
    const fs::path limited_path = work_dir / ("query_prefix_" + std::to_string(query_limit) + ".bin");
    const std::vector<KeyType> keys = cam::storage::load_key_file_keys(query_layout, query_limit);

    std::ofstream out(limited_path, std::ios::binary | std::ios::trunc);
    if (!out) {
        throw std::runtime_error("failed to create limited query file: " + limited_path.string());
    }
    out.write(reinterpret_cast<const char*>(keys.data()),
              static_cast<std::streamsize>(keys.size() * sizeof(KeyType)));
    if (!out) {
        throw std::runtime_error("failed to write limited query file: " + limited_path.string());
    }
    return limited_path;
}

template <typename IndexT>
QueryStats run_query_workload(
    const IndexT& index,
    CachePolicy policy,
    const cam::storage::KeyFileLayout& data_layout,
    const std::vector<KeyType>& queries,
    size_t cache_bytes)
{
    QueryStats stats;
    stats.queries = queries.size();

    cam::storage::DiskManager disk(data_layout, cam::storage::make_page_cache(policy, cache_bytes));

    const auto t0 = Clock::now();
    for (KeyType key : queries) {
        const auto result = cam::point_query::run_point_query(index, disk, key, ALL_IN_ONCE);
        if (result.found) {
            ++stats.found;
            stats.checksum += result.matched_key;
        }
    }
    const auto t1 = Clock::now();

    stats.wall_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    stats.disk = disk.stats();
    return stats;
}

template <size_t Epsilon>
ExperimentRow make_original_order_row(
    CachePolicy policy,
    const cam::storage::KeyFileLayout& data_layout,
    const std::vector<KeyType>& data_keys,
    const std::vector<KeyType>& queries,
    size_t total_budget_bytes)
{
    using Index = pgm::PGMIndex<KeyType, Epsilon>;

    ExperimentRow row;
    row.mode = "original_query_order";
    row.policy = policy;
    row.epsilon = Epsilon;
    row.total_budget_bytes = total_budget_bytes;
    row.cache_bytes = total_budget_bytes;
    row.total_keys = data_layout.total_keys;

    Index index(data_keys);
    row.query = run_query_workload(index, policy, data_layout, queries, row.cache_bytes);
    return row;
}

template <size_t Epsilon>
ExperimentRow make_sorted_query_row(
    CachePolicy policy,
    const cam::storage::KeyFileLayout& data_layout,
    const std::vector<KeyType>& data_keys,
    const std::vector<KeyType>& sorted_queries,
    size_t total_budget_bytes,
    const cam::sort::SortStats& sort_stats)
{
    using Index = pgm::PGMIndex<KeyType, Epsilon>;

    ExperimentRow row;
    row.mode = "sorted_query_order";
    row.policy = policy;
    row.epsilon = Epsilon;
    row.total_budget_bytes = total_budget_bytes;
    row.sort_budget_bytes = sort_stats.sort_budget_bytes;
    row.cache_bytes = total_budget_bytes - sort_stats.sort_budget_bytes;
    row.total_keys = data_layout.total_keys;
    row.sort = sort_stats;

    Index index(data_keys);
    row.query = run_query_workload(index, policy, data_layout, sorted_queries, row.cache_bytes);
    return row;
}

void write_header(std::ostream& out) {
    out
        << "mode,policy,epsilon,total_budget_bytes,sort_budget_bytes,cache_bytes,total_keys,"
        << "queries,found,checksum,page_requests,cache_hits,cache_misses,hit_ratio,"
        << "avg_page_requests_per_query,logical_page_reads,physical_read_ops,"
        << "avg_physical_reads_per_query,bytes_read,sort_initial_runs,sort_merge_passes,"
        << "sort_runs_written,sort_input_bytes,sort_output_bytes,sort_wall_ns,"
        << "query_wall_ns,end_to_end_ns,query_throughput_qps,end_to_end_throughput_qps\n";
}

void write_row(std::ostream& out, const ExperimentRow& row) {
    out << std::fixed << std::setprecision(10)
        << row.mode << ','
        << cam::cache::policy_name(row.policy) << ','
        << row.epsilon << ','
        << row.total_budget_bytes << ','
        << row.sort_budget_bytes << ','
        << row.cache_bytes << ','
        << row.total_keys << ','
        << row.query.queries << ','
        << row.query.found << ','
        << row.query.checksum << ','
        << row.query.disk.page_requests << ','
        << row.query.disk.cache_hits << ','
        << row.query.disk.cache_misses << ','
        << row.query.hit_ratio() << ','
        << row.query.avg_page_requests() << ','
        << row.query.disk.logical_page_reads << ','
        << row.query.disk.physical_read_ops << ','
        << row.query.avg_physical_reads() << ','
        << row.query.disk.bytes_read << ','
        << row.sort.initial_runs << ','
        << row.sort.merge_passes << ','
        << row.sort.runs_written << ','
        << row.sort.input_bytes << ','
        << row.sort.output_bytes << ','
        << row.sort.wall_ns << ','
        << row.query.wall_ns << ','
        << row.end_to_end_ns() << ','
        << row.query.query_throughput_qps() << ','
        << row.end_to_end_throughput_qps()
        << '\n';
    out.flush();
}

template <typename Fn>
void dispatch_epsilon(size_t epsilon, Fn&& fn) {
    switch (epsilon) {
        case 4: fn(std::integral_constant<size_t, 4>{}); break;
        case 8: fn(std::integral_constant<size_t, 8>{}); break;
        case 10: fn(std::integral_constant<size_t, 10>{}); break;
        case 12: fn(std::integral_constant<size_t, 12>{}); break;
        case 14: fn(std::integral_constant<size_t, 14>{}); break;
        case 16: fn(std::integral_constant<size_t, 16>{}); break;
        case 20: fn(std::integral_constant<size_t, 20>{}); break;
        case 24: fn(std::integral_constant<size_t, 24>{}); break;
        case 32: fn(std::integral_constant<size_t, 32>{}); break;
        case 64: fn(std::integral_constant<size_t, 64>{}); break;
        case 128: fn(std::integral_constant<size_t, 128>{}); break;
        default:
            throw std::invalid_argument(
                "unsupported epsilon: " + std::to_string(epsilon) +
                " (supported: 4,8,10,12,14,16,20,24,32,64,128)");
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Config cfg = parse_args(argc, argv);
        const fs::path work_dir = fs::absolute(cfg.work_dir);
        fs::create_directories(work_dir);

        const auto data_layout =
            cam::storage::detect_key_file_layout(cfg.data_path, cfg.total_keys, cfg.header_mode);
        const auto raw_query_layout =
            cam::storage::detect_key_file_layout(cfg.query_path, 0, cam::storage::HeaderMode::NO);
        const fs::path base_query_path =
            materialize_query_prefix_if_needed(raw_query_layout, cfg.query_limit, work_dir);
        const auto query_layout =
            cam::storage::detect_key_file_layout(base_query_path.string(), 0, cam::storage::HeaderMode::NO);

        const std::vector<KeyType> data_keys = cam::storage::load_key_file_keys(data_layout);
        const std::vector<KeyType> original_queries = cam::storage::load_key_file_keys(query_layout);

        std::ofstream summary_file;
        std::ostream* out = &std::cout;
        if (cfg.summary_out.has_value()) {
            ensure_parent_dir(*cfg.summary_out);
            summary_file.open(*cfg.summary_out, std::ios::out | std::ios::trunc);
            if (!summary_file) {
                throw std::runtime_error("failed to open summary output: " + *cfg.summary_out);
            }
            out = &summary_file;
        }

        write_header(*out);

        dispatch_epsilon(cfg.epsilon, [&](auto eps_tag) {
            constexpr size_t Epsilon = decltype(eps_tag)::value;

            std::cerr << "[sort_cache_tradeoff] epsilon=" << Epsilon
                      << ", queries=" << original_queries.size() << '\n';

            for (CachePolicy policy : cfg.policies) {
                std::cerr << "[sort_cache_tradeoff] running original query order, policy="
                          << cam::cache::policy_name(policy) << '\n';
                write_row(*out, make_original_order_row<Epsilon>(
                    policy,
                    data_layout,
                    data_keys,
                    original_queries,
                    cfg.total_budget_bytes));
            }

            for (size_t sort_budget_bytes : cfg.sort_budgets_bytes) {
                if (sort_budget_bytes == 0) {
                    continue;
                }

                const fs::path per_budget_dir = work_dir / ("query_sort_" + std::to_string(sort_budget_bytes));
                std::error_code ec;
                fs::remove_all(per_budget_dir, ec);
                fs::create_directories(per_budget_dir);

                std::cerr << "[sort_cache_tradeoff] sorting queries with M_s="
                          << (sort_budget_bytes >> 20) << " MiB\n";

                const cam::sort::SortStats sort_stats =
                    cam::sort::external_merge_sort(query_layout, sort_budget_bytes, per_budget_dir);
                const std::vector<KeyType> sorted_queries =
                    cam::storage::load_key_file_keys(sort_stats.output_layout);

                for (CachePolicy policy : cfg.policies) {
                    std::cerr << "[sort_cache_tradeoff] running sorted query order, policy="
                              << cam::cache::policy_name(policy)
                              << ", M_s=" << (sort_budget_bytes >> 20) << " MiB\n";
                    write_row(*out, make_sorted_query_row<Epsilon>(
                        policy,
                        data_layout,
                        data_keys,
                        sorted_queries,
                        cfg.total_budget_bytes,
                        sort_stats));
                }

                if (!cfg.keep_sorted_files) {
                    fs::remove_all(per_budget_dir, ec);
                }
            }
        });

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[error] " << e.what() << '\n';
        return 1;
    }
}
