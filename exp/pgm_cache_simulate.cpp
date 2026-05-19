#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_set>
#include <vector>

#include "../src/cache/CacheInterface.hpp"
#include "../src/cache/CacheUtils.hpp"
#include "../src/pgm/PointQuery.hpp"
#include "../src/pgm/pgm_index.hpp"
#include "../src/storage/KeyFile.hpp"
#include "../src/utils.hpp"

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;
using KeyT = uint64_t;

namespace {

constexpr size_t kEstimatedSegmentBytes = 16;

enum class BudgetMode {
    RAW,
    ESTIMATED,
    MEASURED
};

struct Config {
    std::string data_path;
    std::string query_path;
    std::optional<std::string> summary_out;

    std::vector<size_t> epsilons;
    std::vector<CachePolicy> policies = {CachePolicy::FIFO, CachePolicy::LRU, CachePolicy::LFU};
    std::vector<SearchStrategy> strategies = {ALL_IN_ONCE};

    size_t total_keys = 0;
    size_t query_limit = 0;
    size_t M = 64ULL << 20;
    BudgetMode budget_mode = BudgetMode::RAW;
    size_t warmup_pages = 0;
    uint64_t warmup_seed = 42;

    size_t epsilon_start = 2;
    size_t epsilon_end = 128;
    size_t epsilon_step = 2;
};

struct SummaryRow {
    size_t epsilon = 0;
    CachePolicy policy = CachePolicy::NONE;
    SearchStrategy strategy = ALL_IN_ONCE;
    BudgetMode budget_mode = BudgetMode::RAW;

    size_t memory_budget_bytes = 0;
    size_t cache_bytes = 0;
    size_t cache_pages = 0;
    size_t estimated_index_bytes = 0;
    size_t measured_index_bytes = 0;
    size_t reserved_index_bytes = 0;
    size_t warmup_pages_requested = 0;
    size_t warmup_pages_loaded = 0;
    uint64_t warmup_seed = 0;

    size_t queries = 0;
    uint64_t total_dac = 0;
    uint64_t total_cache_hits = 0;
    uint64_t total_cache_misses = 0;

    long long index_build_ns = 0;
    long long simulate_wall_ns = 0;

    uint64_t query_checksum = 0;

    double global_hit_ratio() const {
        if (total_dac == 0) {
            return 0.0;
        }
        return static_cast<double>(total_cache_hits) / static_cast<double>(total_dac);
    }

    double avg_dac() const {
        if (queries == 0) {
            return 0.0;
        }
        return static_cast<double>(total_dac) / static_cast<double>(queries);
    }

    double avg_cam_io() const {
        if (queries == 0) {
            return 0.0;
        }
        return static_cast<double>(total_cache_misses) / static_cast<double>(queries);
    }

    double throughput_qps() const {
        if (simulate_wall_ns == 0) {
            return 0.0;
        }
        return static_cast<double>(queries) * 1e9 / static_cast<double>(simulate_wall_ns);
    }
};

[[noreturn]] void usage_error(const std::string& msg) {
    throw std::invalid_argument(
        msg +
        "\nUsage: ./pgm_cache_simulate --data <file> --queries <file> [--keys <n>] [--M <MiB>]"
        " [--epsilons <e1,e2,...> | --epsilon-start <s> --epsilon-end <e> --epsilon-step <d>]"
        " [--policies <fifo,lru,lfu,none|all>] [--strategies <all_in_once,one_by_one|all>]"
        " [--budget-mode <estimated|measured>] [--summary-out <csv>] [--query-limit <n>]"
        " [--warmup-pages <n>] [--warmup-seed <seed>]"
    );
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

BudgetMode parse_budget_mode(const std::string& value) {
    const std::string mode = cam::storage::to_upper(cam::storage::trim(value));
    if (mode == "ESTIMATED") return BudgetMode::ESTIMATED;
    if (mode == "MEASURED") return BudgetMode::MEASURED;
    if (mode == "RAW") return BudgetMode::RAW;
    throw std::invalid_argument("unknown budget mode: " + value);
}

std::string budget_mode_name(BudgetMode mode) {
    switch (mode) {
        case BudgetMode::ESTIMATED: return "estimated";
        case BudgetMode::MEASURED: return "measured";
        case BudgetMode::RAW: return "raw";
        default: return "unknown";
    }
}

std::vector<size_t> make_epsilon_range(size_t start, size_t end, size_t step) {
    if (step == 0) {
        throw std::invalid_argument("epsilon step must be > 0");
    }
    if (start > end) {
        throw std::invalid_argument("epsilon start must be <= epsilon end");
    }

    std::vector<size_t> eps;
    for (size_t e = start; e <= end; e += step) {
        eps.push_back(e);
        if (e > end - step) {
            break;
        }
    }
    if (eps.empty()) {
        throw std::invalid_argument("epsilon range is empty");
    }
    return eps;
}

Config parse_args(int argc, char** argv) {
    Config cfg;
    bool eps_explicit = false;

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
        } else if (arg == "--epsilons") {
            cfg.epsilons = parse_size_list(require_value("--epsilons"));
            eps_explicit = true;
        } else if (arg == "--epsilon-start") {
            cfg.epsilon_start = std::stoull(require_value("--epsilon-start"));
        } else if (arg == "--epsilon-end") {
            cfg.epsilon_end = std::stoull(require_value("--epsilon-end"));
        } else if (arg == "--epsilon-step") {
            cfg.epsilon_step = std::stoull(require_value("--epsilon-step"));
        } else if (arg == "--policies") {
            cfg.policies = cam::cache::parse_policy_list(require_value("--policies"));
        } else if (arg == "--strategies") {
            cfg.strategies = cam::point_query::parse_search_strategy_list(require_value("--strategies"));
        } else if (arg == "--budget-mode") {
            cfg.budget_mode = parse_budget_mode(require_value("--budget-mode"));
        } else if (arg == "--summary-out") {
            cfg.summary_out = require_value("--summary-out");
        } else if (arg == "--query-limit") {
            cfg.query_limit = std::stoull(require_value("--query-limit"));
        } else if (arg == "--warmup-pages") {
            cfg.warmup_pages = std::stoull(require_value("--warmup-pages"));
        } else if (arg == "--warmup-seed") {
            cfg.warmup_seed = std::stoull(require_value("--warmup-seed"));
        } else if (arg == "-h" || arg == "--help") {
            usage_error("help requested");
        } else {
            usage_error("unknown argument: " + arg);
        }
    }

    if (cfg.data_path.empty() || cfg.query_path.empty()) {
        usage_error("both --data and --queries are required");
    }

    if (cfg.total_keys == 0) {
        cfg.total_keys = detect_record_count(cfg.data_path);
    }

    if (!eps_explicit) {
        cfg.epsilons = make_epsilon_range(cfg.epsilon_start, cfg.epsilon_end, cfg.epsilon_step);
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
        << "warmup_pages_requested,warmup_pages_loaded,warmup_seed,"
        << "queries,total_dac,total_cache_hits,total_cache_misses,"
        << "global_hit_ratio,avg_dac,avg_cam_io,index_build_ns,simulate_wall_ns,"
        << "throughput_qps,query_checksum\n";
}

void write_summary_row(std::ostream& out, const SummaryRow& row) {
    out << std::fixed << std::setprecision(10)
        << row.epsilon << ','
        << cam::cache::policy_name(row.policy) << ','
        << cam::point_query::search_strategy_name(row.strategy) << ','
        << budget_mode_name(row.budget_mode) << ','
        << row.memory_budget_bytes << ','
        << row.cache_bytes << ','
        << row.cache_pages << ','
        << row.estimated_index_bytes << ','
        << row.measured_index_bytes << ','
        << row.reserved_index_bytes << ','
        << row.warmup_pages_requested << ','
        << row.warmup_pages_loaded << ','
        << row.warmup_seed << ','
        << row.queries << ','
        << row.total_dac << ','
        << row.total_cache_hits << ','
        << row.total_cache_misses << ','
        << row.global_hit_ratio() << ','
        << row.avg_dac() << ','
        << row.avg_cam_io() << ','
        << row.index_build_ns << ','
        << row.simulate_wall_ns << ','
        << row.throughput_qps() << ','
        << row.query_checksum
        << '\n';
}

inline void touch_cache_page(ICache& cache, size_t page_idx, SummaryRow& row) {
    Page out;
    ++row.total_dac;
    if (cache.get(page_idx, out)) {
        ++row.total_cache_hits;
        return;
    }

    ++row.total_cache_misses;
    Page empty;
    cache.put(page_idx, std::move(empty));
}

void warmup_cache_random_pages(
    ICache& cache,
    size_t total_pages,
    size_t requested_pages,
    uint64_t seed,
    SummaryRow& row)
{
    row.warmup_pages_requested = requested_pages;
    row.warmup_seed = seed;
    if (requested_pages == 0 || total_pages == 0 || cache.capacity_pages() == 0) {
        return;
    }

    const size_t target = std::min({requested_pages, cache.capacity_pages(), total_pages});
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<size_t> dist(0, total_pages - 1);
    std::unordered_set<size_t> seen;
    seen.reserve(target * 2);

    while (seen.size() < target) {
        const size_t page_idx = dist(rng);
        if (!seen.insert(page_idx).second) {
            continue;
        }
        Page empty;
        cache.put(page_idx, std::move(empty));
    }
    row.warmup_pages_loaded = seen.size();
}

template <typename IndexT>
inline void simulate_query_all_at_once(
    const IndexT& index,
    ICache& cache,
    KeyT key,
    SummaryRow& row,
    size_t total_pages)
{
    auto [page_lo, page_hi] = index.estimate_pages_for_key(key);
    if (total_pages == 0 || page_lo >= total_pages) {
        return;
    }
    page_hi = std::min(page_hi, total_pages - 1);
    for (size_t page_idx = page_lo; page_idx <= page_hi; ++page_idx) {
        touch_cache_page(cache, page_idx, row);
    }
}

template <typename IndexT>
inline void simulate_query_one_by_one(
    const IndexT& index,
    ICache& cache,
    KeyT key,
    SummaryRow& row,
    const std::vector<KeyT>& page_last_keys)
{
    if (page_last_keys.empty()) {
        return;
    }

    auto [page_lo, page_hi] = index.estimate_pages_for_key(key);
    const size_t total_pages = page_last_keys.size();
    if (page_lo >= total_pages) {
        return;
    }
    page_hi = std::min(page_hi, total_pages - 1);

    for (size_t page_idx = page_lo; page_idx <= page_hi; ++page_idx) {
        touch_cache_page(cache, page_idx, row);
        if (key <= page_last_keys[page_idx]) {
            break;
        }
    }
}

template <typename IndexT>
SummaryRow run_one_policy(
    const IndexT& index,
    const Config& cfg,
    size_t epsilon,
    size_t estimated_index_bytes,
    size_t measured_index_bytes,
    long long index_build_ns,
    CachePolicy policy,
    SearchStrategy strategy,
    const std::vector<KeyT>& queries,
    const std::vector<KeyT>& page_last_keys)
{
    SummaryRow row;
    row.epsilon = epsilon;
    row.policy = policy;
    row.strategy = strategy;
    row.budget_mode = cfg.budget_mode;

    row.memory_budget_bytes = cfg.M;
    row.estimated_index_bytes = estimated_index_bytes;
    row.measured_index_bytes = measured_index_bytes;
    row.reserved_index_bytes =
        cfg.budget_mode == BudgetMode::ESTIMATED ? estimated_index_bytes : measured_index_bytes;
    row.cache_bytes = (cfg.budget_mode == BudgetMode::RAW) ? cfg.M :safe_subtract(cfg.M, row.reserved_index_bytes);
    row.cache_pages = row.cache_bytes / PAGE_SIZE;
    row.index_build_ns = index_build_ns;
    row.queries = queries.size();
    row.warmup_pages_requested = cfg.warmup_pages;
    row.warmup_seed = cfg.warmup_seed;

    auto cache = MakeCache(policy, row.cache_bytes, PAGE_SIZE);
    warmup_cache_random_pages(*cache, page_last_keys.size(), cfg.warmup_pages, cfg.warmup_seed, row);

    const auto t0 = Clock::now();
    for (KeyT q : queries) {
        row.query_checksum += q;
        switch (strategy) {
            case ALL_IN_ONCE:
                simulate_query_all_at_once(index, *cache, q, row, page_last_keys.size());
                break;
            case ONE_BY_ONE:
                simulate_query_one_by_one(index, *cache, q, row, page_last_keys);
                break;
            default:
                throw std::invalid_argument("unsupported search strategy");
        }
    }
    const auto t1 = Clock::now();
    row.simulate_wall_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    return row;
}

std::vector<KeyT> build_page_last_keys(const std::vector<KeyT>& data) {
    if (data.empty()) {
        return {};
    }

    const size_t total_pages = (data.size() + ITEM_PER_PAGE - 1) / ITEM_PER_PAGE;
    std::vector<KeyT> page_last_keys(total_pages);
    for (size_t page = 0; page < total_pages; ++page) {
        const size_t hi = std::min(data.size(), (page + 1) * ITEM_PER_PAGE) - 1;
        page_last_keys[page] = data[hi];
    }
    return page_last_keys;
}

template <size_t Epsilon>
void run_epsilon(
    const Config& cfg,
    const std::vector<KeyT>& data,
    const std::vector<KeyT>& queries,
    const std::vector<KeyT>& page_last_keys,
    std::ostream& summary_out)
{
    using Index = pgm::PGMIndex<KeyT, Epsilon>;

    // const auto t0 = Clock::now(); 
    
    const auto build_t0 = Clock::now();
    Index index(data);
    const auto build_t1 = Clock::now();
    const long long index_build_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(build_t1 - build_t0).count();

    const size_t measured_index_bytes = index.size_in_bytes();
    const size_t estimated_index_bytes = estimate_index_bytes(data.size(), Epsilon);

    for (SearchStrategy strategy : cfg.strategies) {
        for (CachePolicy policy : cfg.policies) {
            const SummaryRow row = run_one_policy(
                index,
                cfg,
                Epsilon,
                estimated_index_bytes,
                measured_index_bytes,
                index_build_ns,
                policy,
                strategy,
                queries,
                page_last_keys);
            write_summary_row(summary_out, row);
        }
    }
}

template <typename Fn>
void dispatch_epsilon(size_t epsilon, Fn&& fn) {
#define CAM_DISPATCH_EPS_CASE(E) case E: fn(std::integral_constant<size_t, E>{}); break
    switch (epsilon) {
        CAM_DISPATCH_EPS_CASE(2);
        CAM_DISPATCH_EPS_CASE(4);
        CAM_DISPATCH_EPS_CASE(6);
        CAM_DISPATCH_EPS_CASE(8);
        CAM_DISPATCH_EPS_CASE(10);
        CAM_DISPATCH_EPS_CASE(12);
        CAM_DISPATCH_EPS_CASE(14);
        CAM_DISPATCH_EPS_CASE(16);
        CAM_DISPATCH_EPS_CASE(18);
        CAM_DISPATCH_EPS_CASE(20);
        CAM_DISPATCH_EPS_CASE(22);
        CAM_DISPATCH_EPS_CASE(24);
        CAM_DISPATCH_EPS_CASE(26);
        CAM_DISPATCH_EPS_CASE(28);
        CAM_DISPATCH_EPS_CASE(30);
        CAM_DISPATCH_EPS_CASE(32);
        CAM_DISPATCH_EPS_CASE(34);
        CAM_DISPATCH_EPS_CASE(36);
        CAM_DISPATCH_EPS_CASE(38);
        CAM_DISPATCH_EPS_CASE(40);
        CAM_DISPATCH_EPS_CASE(42);
        CAM_DISPATCH_EPS_CASE(44);
        CAM_DISPATCH_EPS_CASE(46);
        CAM_DISPATCH_EPS_CASE(48);
        CAM_DISPATCH_EPS_CASE(50);
        CAM_DISPATCH_EPS_CASE(52);
        CAM_DISPATCH_EPS_CASE(54);
        CAM_DISPATCH_EPS_CASE(56);
        CAM_DISPATCH_EPS_CASE(58);
        CAM_DISPATCH_EPS_CASE(60);
        CAM_DISPATCH_EPS_CASE(62);
        CAM_DISPATCH_EPS_CASE(64);
        CAM_DISPATCH_EPS_CASE(66);
        CAM_DISPATCH_EPS_CASE(68);
        CAM_DISPATCH_EPS_CASE(70);
        CAM_DISPATCH_EPS_CASE(72);
        CAM_DISPATCH_EPS_CASE(74);
        CAM_DISPATCH_EPS_CASE(76);
        CAM_DISPATCH_EPS_CASE(78);
        CAM_DISPATCH_EPS_CASE(80);
        CAM_DISPATCH_EPS_CASE(82);
        CAM_DISPATCH_EPS_CASE(84);
        CAM_DISPATCH_EPS_CASE(86);
        CAM_DISPATCH_EPS_CASE(88);
        CAM_DISPATCH_EPS_CASE(90);
        CAM_DISPATCH_EPS_CASE(92);
        CAM_DISPATCH_EPS_CASE(94);
        CAM_DISPATCH_EPS_CASE(96);
        CAM_DISPATCH_EPS_CASE(98);
        CAM_DISPATCH_EPS_CASE(100);
        CAM_DISPATCH_EPS_CASE(102);
        CAM_DISPATCH_EPS_CASE(104);
        CAM_DISPATCH_EPS_CASE(106);
        CAM_DISPATCH_EPS_CASE(108);
        CAM_DISPATCH_EPS_CASE(110);
        CAM_DISPATCH_EPS_CASE(112);
        CAM_DISPATCH_EPS_CASE(114);
        CAM_DISPATCH_EPS_CASE(116);
        CAM_DISPATCH_EPS_CASE(118);
        CAM_DISPATCH_EPS_CASE(120);
        CAM_DISPATCH_EPS_CASE(122);
        CAM_DISPATCH_EPS_CASE(124);
        CAM_DISPATCH_EPS_CASE(126);
        CAM_DISPATCH_EPS_CASE(128);
        default:
            throw std::invalid_argument(
                "unsupported epsilon: " + std::to_string(epsilon) +
                " (supported: even epsilons in [2, 128])");
    }
#undef CAM_DISPATCH_EPS_CASE
}

} // namespace

int main(int argc, char** argv) {
    try {
        Config cfg = parse_args(argc, argv);

        auto data = load_data_pgm_safe<KeyT>(cfg.data_path, cfg.total_keys);
        auto queries = load_queries_pgm_safe<KeyT>(cfg.query_path);
        if (cfg.query_limit > 0 && queries.size() > cfg.query_limit) {
            queries.resize(cfg.query_limit);
        }

        // std::sort(data.begin(), data.end());
        const auto page_last_keys = build_page_last_keys(data);

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

        write_summary_header(*summary_out);
        for (size_t epsilon : cfg.epsilons) {
            dispatch_epsilon(epsilon, [&](auto eps_tag) {
                constexpr size_t Eps = decltype(eps_tag)::value;
                run_epsilon<Eps>(cfg, data, queries, page_last_keys, *summary_out);
            });
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[error] " << e.what() << '\n';
        return 1;
    }
}
