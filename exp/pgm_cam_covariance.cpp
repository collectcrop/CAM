#include <algorithm>
#include <chrono>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>
#include <unistd.h>

#include "../src/cache/CacheUtils.hpp"
#include "../src/pgm/pgm_index.hpp"
#include "../src/pgm/PointQuery.hpp"
#include "../src/storage/DiskManager.hpp"
#include "../src/storage/KeyFile.hpp"
#include "../src/utils.hpp"

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;
using KeyType = uint64_t;

namespace {

constexpr size_t kEstimatedSegmentBytes = 16;
const std::vector<size_t> kDefaultEpsilons = {4, 8, 10, 12, 14, 16, 20, 24, 32, 64, 128};
const std::vector<CachePolicy> kDefaultPolicies = {
    CachePolicy::FIFO,
    CachePolicy::LRU,
    CachePolicy::LFU
};

enum class BudgetMode {
    RAW,
    ESTIMATED,
    MEASURED,
    FIXED_CACHE
};

struct Config {
    std::string data_path;
    std::string query_path;
    std::optional<std::string> summary_out;
    std::optional<std::string> detail_out;
    std::vector<size_t> epsilons = kDefaultEpsilons;
    std::vector<CachePolicy> policies = kDefaultPolicies;
    std::vector<SearchStrategy> strategies = {ALL_IN_ONCE};
    size_t total_keys = 0;
    size_t query_limit = 0;
    size_t M = 64ULL << 20;
    size_t fixed_cache_bytes = 0;
    BudgetMode budget_mode = BudgetMode::ESTIMATED;
};

struct SummaryStats {
    size_t epsilon = 0;
    CachePolicy policy = CachePolicy::NONE;
    SearchStrategy strategy = ALL_IN_ONCE;
    BudgetMode budget_mode = BudgetMode::ESTIMATED;

    size_t memory_budget_bytes = 0;
    size_t cache_bytes = 0;
    size_t cache_pages = 0;
    size_t estimated_index_bytes = 0;
    size_t measured_index_bytes = 0;
    size_t reserved_index_bytes = 0;

    size_t queries = 0;
    size_t found = 0;
    uint64_t checksum = 0;

    uint64_t total_dac = 0;
    uint64_t total_buffer_hits = 0;
    uint64_t total_cam_io = 0;
    uint64_t total_device_ios = 0;
    uint64_t total_disk_pages_read = 0;

    double mean_h = 0.0;
    double mean_dac = 0.0;
    double m2_h = 0.0;
    double m2_dac = 0.0;
    double c_h_dac = 0.0;

    long long wall_ns = 0;

    void add(const cam::point_query::PointQueryResult& result) {
        const double dac = static_cast<double>(result.metrics.dac);
        const double h = result.metrics.dac == 0
            ? 0.0
            : static_cast<double>(result.metrics.buffer_hits) /
                  static_cast<double>(result.metrics.dac);

        ++queries;

        const double delta_h = h - mean_h;
        mean_h += delta_h / static_cast<double>(queries);

        const double delta_dac = dac - mean_dac;
        mean_dac += delta_dac / static_cast<double>(queries);

        m2_h += delta_h * (h - mean_h);
        m2_dac += delta_dac * (dac - mean_dac);
        c_h_dac += delta_h * (dac - mean_dac);

        total_dac += result.metrics.dac;
        total_buffer_hits += result.metrics.buffer_hits;
        total_cam_io += result.metrics.cam_io;
        total_device_ios += result.metrics.device_ios;
        total_disk_pages_read += result.metrics.disk_pages_read;

        if (result.found) {
            ++found;
            checksum += result.matched_key;
        }
    }

    double global_hit_ratio() const {
        return total_dac == 0
            ? 0.0
            : static_cast<double>(total_buffer_hits) / static_cast<double>(total_dac);
    }

    double mean_cam_io() const {
        return queries == 0
            ? 0.0
            : static_cast<double>(total_cam_io) / static_cast<double>(queries);
    }

    double mean_device_ios() const {
        return queries == 0
            ? 0.0
            : static_cast<double>(total_device_ios) / static_cast<double>(queries);
    }

    double var_h() const {
        return queries == 0 ? 0.0 : m2_h / static_cast<double>(queries);
    }

    double var_dac() const {
        return queries == 0 ? 0.0 : m2_dac / static_cast<double>(queries);
    }

    double cov_h_dac() const {
        return queries == 0 ? 0.0 : c_h_dac / static_cast<double>(queries);
    }

    double cov_over_mean_cam_io() const {
        const double denom = mean_cam_io();
        return denom == 0.0 ? 0.0 : cov_h_dac() / denom;
    }

    double abs_cov_over_mean_cam_io() const {
        const double denom = mean_cam_io();
        return denom == 0.0 ? 0.0 : std::abs(cov_h_dac()) / denom;
    }

    double cauchy_schwarz_bound() const {
        return std::sqrt(std::max(0.0, var_h()) * std::max(0.0, var_dac()));
    }

    double cauchy_schwarz_bound_over_mean_cam_io() const {
        const double denom = mean_cam_io();
        return denom == 0.0 ? 0.0 : cauchy_schwarz_bound() / denom;
    }

    double identity_rhs() const {
        return (1.0 - mean_h) * mean_dac - cov_h_dac();
    }

    double identity_gap() const {
        return mean_cam_io() - identity_rhs();
    }

    double throughput_qps() const {
        return wall_ns == 0
            ? 0.0
            : static_cast<double>(queries) * 1e9 / static_cast<double>(wall_ns);
    }
};

[[noreturn]] void usage_error(const std::string& msg) {
    throw std::invalid_argument(
        msg +
        "\nUsage: ./pgm_cam_covariance --data <file> --queries <file> [--keys <n>] [--M <MiB>]"
        " [--epsilons <e1,e2,...>] [--policies <fifo,lru,lfu,none>]"
        " [--strategies <all_in_once,one_by_one|all>]"
        " [--budget-mode <estimated|measured|raw|fixed-cache>]"
        " [--cache-M <MiB> | --cache-bytes <bytes>]"
        " [--summary-out <csv>] [--detail-out <csv>] [--query-limit <n>]");
}

std::string trim(std::string s) {
    const auto not_space = [](unsigned char c) { return !std::isspace(c); };
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), not_space));
    s.erase(std::find_if(s.rbegin(), s.rend(), not_space).base(), s.end());
    return s;
}

std::string to_upper(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
        return static_cast<char>(std::toupper(c));
    });
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
        if (std::find(out.begin(), out.end(), parsed) == out.end()) {
            out.push_back(parsed);
        }
    }
    if (out.empty()) {
        throw std::invalid_argument("empty epsilon list");
    }
    return out;
}

BudgetMode parse_budget_mode(const std::string& value) {
    const std::string mode = to_upper(trim(value));
    if (mode == "ESTIMATED") return BudgetMode::ESTIMATED;
    if (mode == "MEASURED") return BudgetMode::MEASURED;
    if (mode == "RAW") return BudgetMode::RAW;
    if (mode == "FIXED_CACHE" || mode == "FIXED-CACHE" || mode == "FIXED") {
        return BudgetMode::FIXED_CACHE;
    }
    throw std::invalid_argument("unknown budget mode: " + value);
}

std::string budget_mode_name(BudgetMode mode) {
    switch (mode) {
        case BudgetMode::ESTIMATED: return "estimated";
        case BudgetMode::MEASURED: return "measured";
        case BudgetMode::RAW: return "raw";
        case BudgetMode::FIXED_CACHE: return "fixed_cache";
        default: return "unknown";
    }
}

size_t estimate_index_bytes(size_t total_keys, size_t epsilon) {
    return (kEstimatedSegmentBytes * total_keys) / (2 * epsilon);
}

size_t safe_subtract(size_t lhs, size_t rhs) {
    return lhs > rhs ? lhs - rhs : 0;
}

class RuntimePGMIndex : public pgm::PGMIndex<KeyType, 1, 4, float> {
public:
    explicit RuntimePGMIndex(const std::vector<KeyType>& data, size_t epsilon)
        : epsilon_(epsilon)
    {
        if (epsilon_ == 0) {
            throw std::invalid_argument("epsilon must be > 0");
        }

        this->n = data.size();
        this->first_key = data.empty() ? KeyType(0) : data[0];
        this->build(data.begin(), data.end(), epsilon_, 4, this->segments, this->levels_offsets);
    }

    pgm::ApproxPos search(const KeyType& key) const {
        auto k = std::max(this->first_key, key);
        auto it = this->segment_for_key(k);
        size_t pos = std::min<size_t>((*it)(k), std::next(it)->intercept);
        size_t lo = PGM_SUB_EPS(pos, epsilon_);
        size_t hi = PGM_ADD_EPS(pos, epsilon_, this->n);
        return {pos, lo, hi};
    }

    std::pair<size_t, size_t> estimate_pages_for_key(const KeyType& key) const {
        const auto range = search(key);
        const size_t page_lo = range.lo / ITEM_PER_PAGE;
        size_t page_hi = page_lo;
        if (range.hi > range.lo) {
            page_hi = (range.hi - 1) / ITEM_PER_PAGE;
        }
        if (page_hi < page_lo) {
            page_hi = page_lo;
        }
        return {page_lo, page_hi};
    }

private:
    size_t epsilon_ = 0;
};

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
            cfg.M = std::stoull(require_value("--M")) << 20;
        } else if (arg == "--cache-M") {
            cfg.fixed_cache_bytes = std::stoull(require_value("--cache-M")) << 20;
        } else if (arg == "--cache-bytes") {
            cfg.fixed_cache_bytes = std::stoull(require_value("--cache-bytes"));
        } else if (arg == "--epsilons") {
            cfg.epsilons = parse_size_list(require_value("--epsilons"));
        } else if (arg == "--policies") {
            cfg.policies = cam::cache::parse_policy_list(require_value("--policies"));
        } else if (arg == "--strategies") {
            cfg.strategies = cam::point_query::parse_search_strategy_list(require_value("--strategies"));
        } else if (arg == "--budget-mode") {
            cfg.budget_mode = parse_budget_mode(require_value("--budget-mode"));
        } else if (arg == "--summary-out") {
            cfg.summary_out = require_value("--summary-out");
        } else if (arg == "--detail-out") {
            cfg.detail_out = require_value("--detail-out");
        } else if (arg == "--query-limit") {
            cfg.query_limit = std::stoull(require_value("--query-limit"));
        } else if (arg == "-h" || arg == "--help") {
            usage_error("help requested");
        } else {
            usage_error("unknown argument: " + arg);
        }
    }

    if (cfg.data_path.empty() || cfg.query_path.empty()) {
        usage_error("both --data and --queries are required");
    }
    if (cfg.budget_mode == BudgetMode::FIXED_CACHE) {
        if (cfg.fixed_cache_bytes == 0) {
            usage_error("--budget-mode fixed-cache requires --cache-M or --cache-bytes");
        }
        if (cfg.fixed_cache_bytes > cfg.M) {
            usage_error("--cache-M/--cache-bytes cannot exceed --M");
        }
    }
    return cfg;
}

void ensure_parent_dir(const std::string& output_path) {
    const fs::path path(output_path);
    const fs::path parent = path.parent_path();
    if (!parent.empty()) {
        fs::create_directories(parent);
    }
}

void write_summary_header(std::ostream& out) {
    out
        << "epsilon,policy,strategy,budget_mode,memory_budget_bytes,cache_bytes,cache_pages,"
        << "estimated_index_bytes,measured_index_bytes,reserved_index_bytes,"
        << "queries,found,total_dac,total_buffer_hits,total_cam_io,total_device_ios,"
        << "total_disk_pages_read,mean_h,global_hit_ratio,mean_dac,mean_cam_io,"
        << "mean_device_ios,var_h,var_dac,cov_h_dac,cov_over_mean_cam_io,"
        << "abs_cov_over_mean_cam_io,cauchy_schwarz_bound,"
        << "cauchy_schwarz_bound_over_mean_cam_io,identity_rhs,identity_gap,"
        << "wall_ns,throughput_qps,checksum\n";
}

void write_summary_row(std::ostream& out, const SummaryStats& stats) {
    out << std::fixed << std::setprecision(10)
        << stats.epsilon << ','
        << cam::cache::policy_name(stats.policy) << ','
        << cam::point_query::search_strategy_name(stats.strategy) << ','
        << budget_mode_name(stats.budget_mode) << ','
        << stats.memory_budget_bytes << ','
        << stats.cache_bytes << ','
        << stats.cache_pages << ','
        << stats.estimated_index_bytes << ','
        << stats.measured_index_bytes << ','
        << stats.reserved_index_bytes << ','
        << stats.queries << ','
        << stats.found << ','
        << stats.total_dac << ','
        << stats.total_buffer_hits << ','
        << stats.total_cam_io << ','
        << stats.total_device_ios << ','
        << stats.total_disk_pages_read << ','
        << stats.mean_h << ','
        << stats.global_hit_ratio() << ','
        << stats.mean_dac << ','
        << stats.mean_cam_io() << ','
        << stats.mean_device_ios() << ','
        << stats.var_h() << ','
        << stats.var_dac() << ','
        << stats.cov_h_dac() << ','
        << stats.cov_over_mean_cam_io() << ','
        << stats.abs_cov_over_mean_cam_io() << ','
        << stats.cauchy_schwarz_bound() << ','
        << stats.cauchy_schwarz_bound_over_mean_cam_io() << ','
        << stats.identity_rhs() << ','
        << stats.identity_gap() << ','
        << stats.wall_ns << ','
        << stats.throughput_qps() << ','
        << stats.checksum
        << '\n';
}

void write_detail_header(std::ostream& out) {
    out << "epsilon,policy,strategy,query_idx,query_key,dac,buffer_hits,cam_io,h,device_ios,found\n";
}

void write_detail_row(
    std::ostream& out,
    size_t epsilon,
    CachePolicy policy,
    SearchStrategy strategy,
    size_t query_idx,
    KeyType query_key,
    const cam::point_query::PointQueryResult& result)
{
    const double h = result.metrics.dac == 0
        ? 0.0
        : static_cast<double>(result.metrics.buffer_hits) /
              static_cast<double>(result.metrics.dac);

    out << std::fixed << std::setprecision(10)
        << epsilon << ','
        << cam::cache::policy_name(policy) << ','
        << cam::point_query::search_strategy_name(strategy) << ','
        << query_idx << ','
        << query_key << ','
        << result.metrics.dac << ','
        << result.metrics.buffer_hits << ','
        << result.metrics.cam_io << ','
        << h << ','
        << result.metrics.device_ios << ','
        << (result.found ? 1 : 0)
        << '\n';
}

template <typename IndexT>
SummaryStats run_one_policy(
    const IndexT& index,
    CachePolicy policy,
    SearchStrategy strategy,
    const cam::storage::KeyFileLayout& data_layout,
    const std::vector<KeyType>& queries,
    const Config& cfg,
    size_t epsilon,
    size_t estimated_index_bytes,
    size_t measured_index_bytes,
    std::ostream* detail_out)
{
    SummaryStats stats;
    stats.epsilon = epsilon;
    stats.policy = policy;
    stats.strategy = strategy;
    stats.budget_mode = cfg.budget_mode;
    stats.memory_budget_bytes = cfg.M;
    stats.estimated_index_bytes = estimated_index_bytes;
    stats.measured_index_bytes = measured_index_bytes;
    if (cfg.budget_mode == BudgetMode::ESTIMATED) {
        stats.reserved_index_bytes = estimated_index_bytes;
        stats.cache_bytes = safe_subtract(cfg.M, stats.reserved_index_bytes);
    } else if (cfg.budget_mode == BudgetMode::MEASURED) {
        stats.reserved_index_bytes = measured_index_bytes;
        stats.cache_bytes = safe_subtract(cfg.M, stats.reserved_index_bytes);
    } else if (cfg.budget_mode == BudgetMode::FIXED_CACHE) {
        stats.reserved_index_bytes = safe_subtract(cfg.M, cfg.fixed_cache_bytes);
        stats.cache_bytes = cfg.fixed_cache_bytes;
    } else {
        stats.reserved_index_bytes = measured_index_bytes;
        stats.cache_bytes = cfg.M;
    }
    stats.cache_pages = stats.cache_bytes / PAGE_SIZE;

    cam::storage::DiskManager disk(
        data_layout,
        cam::storage::make_page_cache(policy, stats.cache_bytes));

    const auto t0 = Clock::now();
    for (size_t i = 0; i < queries.size(); ++i) {
        const auto result = cam::point_query::run_point_query(index, disk, queries[i], strategy);
        stats.add(result);

        if (detail_out != nullptr) {
            write_detail_row(*detail_out, epsilon, policy, strategy, i, queries[i], result);
        }
    }
    const auto t1 = Clock::now();
    stats.wall_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    return stats;
}

void run_runtime_epsilon(
    const cam::storage::KeyFileLayout& data_layout,
    const std::vector<KeyType>& data,
    const std::vector<KeyType>& queries,
    const Config& cfg,
    size_t epsilon,
    std::ostream& summary_out,
    std::ostream* detail_out)
{
    RuntimePGMIndex index(data, epsilon);

    const size_t measured_index_bytes = index.size_in_bytes();
    const size_t estimated_index_bytes = estimate_index_bytes(data.size(), epsilon);

    for (SearchStrategy strategy : cfg.strategies) {
        for (CachePolicy policy : cfg.policies) {
            const SummaryStats stats = run_one_policy(
                index,
                policy,
                strategy,
                data_layout,
                queries,
                cfg,
                epsilon,
                estimated_index_bytes,
                measured_index_bytes,
                detail_out);
            write_summary_row(summary_out, stats);
        }
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
        if (cfg.query_limit > 0 && queries.size() > cfg.query_limit) {
            queries.resize(cfg.query_limit);
        }

        std::ofstream summary_file;
        std::ostream* summary_out = &std::cout;
        if (cfg.summary_out.has_value()) {
            ensure_parent_dir(*cfg.summary_out);
            summary_file.open(*cfg.summary_out, std::ios::out | std::ios::trunc);
            if (!summary_file) {
                throw std::runtime_error("failed to open summary output: " + *cfg.summary_out);
            }
            summary_out = &summary_file;
        }

        std::ofstream detail_file;
        std::ostream* detail_out = nullptr;
        if (cfg.detail_out.has_value()) {
            ensure_parent_dir(*cfg.detail_out);
            detail_file.open(*cfg.detail_out, std::ios::out | std::ios::trunc);
            if (!detail_file) {
                throw std::runtime_error("failed to open detail output: " + *cfg.detail_out);
            }
            detail_out = &detail_file;
        }

        write_summary_header(*summary_out);
        if (detail_out != nullptr) {
            write_detail_header(*detail_out);
        }

        for (size_t epsilon : cfg.epsilons) {
            run_runtime_epsilon(data_layout, data, queries, cfg, epsilon, *summary_out, detail_out);
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[error] " << e.what() << '\n';
        return 1;
    }
}
