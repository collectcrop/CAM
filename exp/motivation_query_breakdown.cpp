#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "../src/cache/CacheUtils.hpp"
#include "../src/pgm/pgm_index.hpp"
#include "../src/pgm/PointQuery.hpp"
#include "../src/storage/DiskManager.hpp"
#include "../src/storage/KeyFile.hpp"

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

namespace {

#define CAM_RMI_BRANCH_FACTOR_LIST(X) \
    X(64) \
    X(128) \
    X(256) \
    X(512) \
    X(1024) \
    X(2048) \
    X(4096) \
    X(8192) \
    X(16384) \
    X(32768) \
    X(65536) \
    X(131072) \
    X(262144) \
    X(524288) \
    X(1048576) \
    X(2097152)

static const size_t kRmiBranchFactors[] = {
#define CAM_REGISTER_RMI_BRANCH(branch) branch,
    CAM_RMI_BRANCH_FACTOR_LIST(CAM_REGISTER_RMI_BRANCH)
#undef CAM_REGISTER_RMI_BRANCH
};

struct RMIModelSpec {
    std::string name;
    size_t branch_factor = 0;
    size_t index_bytes = 0;
    uint64_t build_time_ns = 0;
    double l0_parameter0 = 0.0;
    double l0_parameter1 = 0.0;
    std::vector<char> l1_parameters;
};

enum class Algorithm {
    PGM,
    RMI
};

struct Config {
    std::string data_path;
    std::string query_path;
    std::string rmi_data_dir = "src/rmi/rmi_data";
    std::string rmi_generated_dir = "src/rmi/rmi_eval/generated";
    std::string rmi_prefix = "books_rmi";
    std::string rmi_model_tag = "linear_spline_linear";
    std::string label = "query";
    std::string summary_out;
    size_t total_keys = 0;
    size_t query_limit = 0;
    size_t cache_bytes = 128ull * 1024ull * 1024ull;
    CachePolicy cache_policy = CachePolicy::LRU;
    bool append = false;
    bool direct_io = false;
    cam::storage::HeaderMode header_mode = cam::storage::HeaderMode::AUTO;
    std::vector<Algorithm> algorithms = {Algorithm::PGM, Algorithm::RMI};
    std::vector<size_t> epsilons = {16};
    std::vector<size_t> rmi_branch_factors;
    std::vector<SearchStrategy> strategies = {ALL_IN_ONCE};
};

struct BreakdownResult {
    std::string label;
    std::string baseline;
    std::string io_mode;
    std::string index_type;
    std::string model_name;
    size_t epsilon = 0;
    size_t branch_factor = 0;
    size_t index_bytes = 0;
    uint64_t build_time_ns = 0;
    SearchStrategy strategy = ALL_IN_ONCE;
    CachePolicy cache_policy = CachePolicy::LRU;
    size_t cache_bytes = 0;

    size_t queries = 0;
    size_t found = 0;
    uint64_t checksum = 0;

    size_t page_requests = 0;
    size_t cache_hits = 0;
    size_t cache_misses = 0;
    size_t logical_ios = 0;
    size_t physical_ios = 0;
    uint64_t bytes_read = 0;
    double io_size_mean_bytes = 0.0;
    double io_size_std_bytes = 0.0;
    uint64_t io_size_min_bytes = 0;
    uint64_t io_size_p50_bytes = 0;
    uint64_t io_size_p75_bytes = 0;
    uint64_t io_size_p90_bytes = 0;
    uint64_t io_size_p95_bytes = 0;
    uint64_t io_size_p99_bytes = 0;
    uint64_t io_size_max_bytes = 0;

    long long index_traversal_ns = 0;
    long long cache_ns = 0;
    long long io_ns = 0;
    long long fetch_wall_ns = 0;
    long long lastmile_search_ns = 0;
    long long wall_ns = 0;
};

std::string resolve_local_path(const std::string& value) {
    fs::path path(value);
    if (path.is_absolute()) {
        return path.string();
    }
    if (fs::exists(path)) {
        return fs::absolute(path).string();
    }
    return value;
}

std::string supported_rmi_list() {
    std::ostringstream oss;
    for (size_t i = 0; i < std::size(kRmiBranchFactors); ++i) {
        if (i != 0) {
            oss << ',';
        }
        oss << kRmiBranchFactors[i];
    }
    return oss.str();
}

[[noreturn]] void usage_error(const std::string& msg) {
    throw std::invalid_argument(
        msg +
        "\nUsage: ./motivation_query_breakdown --data <file> --queries <file>"
        " [--keys <n>] [--header <auto|yes|no>] [--label <name>]"
        " [--algorithms <pgm,rmi|all>] [--epsilons <e1,e2,...>]"
        " [--branch-factors <all|" + supported_rmi_list() + ">]"
        " [--rmi-prefix <prefix>] [--rmi-model-tag <tag>]"
        " [--rmi-data-dir <dir>] [--rmi-generated-dir <dir>]"
        " [--strategies <all_in_once,one_by_one|all>]"
        " [--cache-policy <none|fifo|lru|lfu>] [--cache-bytes <n>]"
        " [--io-mode <buffered|direct>|--direct-io]"
        " [--query-limit <n>] [--summary-out <csv>] [--append]");
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
        throw std::invalid_argument("empty integer list");
    }
    return out;
}

std::vector<Algorithm> parse_algorithm_list(const std::string& value) {
    const std::string upper = cam::storage::to_upper(cam::storage::trim(value));
    if (upper.empty() || upper == "ALL") {
        return {Algorithm::PGM, Algorithm::RMI};
    }

    std::vector<Algorithm> out;
    std::stringstream ss(value);
    std::string token;
    while (std::getline(ss, token, ',')) {
        token = cam::storage::to_upper(cam::storage::trim(token));
        if (token.empty()) {
            continue;
        }
        Algorithm algorithm;
        if (token == "PGM") {
            algorithm = Algorithm::PGM;
        } else if (token == "RMI") {
            algorithm = Algorithm::RMI;
        } else {
            throw std::invalid_argument("unknown algorithm: " + token);
        }
        if (std::find(out.begin(), out.end(), algorithm) == out.end()) {
            out.push_back(algorithm);
        }
    }
    if (out.empty()) {
        throw std::invalid_argument("empty algorithm list");
    }
    return out;
}

size_t find_rmi_branch_factor(const std::string& raw_token) {
    const std::string token = cam::storage::trim(raw_token);
    if (token.empty()) {
        return 0;
    }

    std::string branch_token = token;
    const size_t last_underscore = token.find_last_of('_');
    if (last_underscore != std::string::npos && last_underscore + 1 < token.size()) {
        branch_token = token.substr(last_underscore + 1);
    }

    const bool all_digits = std::all_of(branch_token.begin(), branch_token.end(), [](unsigned char ch) {
        return std::isdigit(ch) != 0;
    });
    if (!all_digits) {
        return 0;
    }

    const size_t branch_factor = std::stoull(branch_token);
    for (size_t supported : kRmiBranchFactors) {
        if (supported == branch_factor) {
            return branch_factor;
        }
    }
    return 0;
}

std::vector<size_t> parse_rmi_list(const std::string& value) {
    const std::string upper = cam::storage::to_upper(cam::storage::trim(value));
    if (upper.empty() || upper == "ALL") {
        return std::vector<size_t>(std::begin(kRmiBranchFactors), std::end(kRmiBranchFactors));
    }

    std::vector<size_t> out;
    std::stringstream ss(value);
    std::string token;
    while (std::getline(ss, token, ',')) {
        token = cam::storage::trim(token);
        if (token.empty()) {
            continue;
        }
        const size_t branch_factor = find_rmi_branch_factor(token);
        if (branch_factor == 0) {
            throw std::invalid_argument(
                "unknown RMI selector: " + token +
                " (supported branch factors: " + supported_rmi_list() + ")");
        }
        if (std::find(out.begin(), out.end(), branch_factor) == out.end()) {
            out.push_back(branch_factor);
        }
    }
    if (out.empty()) {
        throw std::invalid_argument("empty RMI list");
    }
    return out;
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
        } else if (arg == "--header") {
            cfg.header_mode = cam::storage::parse_header_mode(require_value("--header"));
        } else if (arg == "--label") {
            cfg.label = require_value("--label");
        } else if (arg == "--algorithms") {
            cfg.algorithms = parse_algorithm_list(require_value("--algorithms"));
        } else if (arg == "--epsilons") {
            cfg.epsilons = parse_size_list(require_value("--epsilons"));
        } else if (arg == "--branch-factors" || arg == "--rmis") {
            cfg.rmi_branch_factors = parse_rmi_list(require_value(arg.c_str()));
        } else if (arg == "--rmi-prefix") {
            cfg.rmi_prefix = require_value("--rmi-prefix");
        } else if (arg == "--rmi-model-tag") {
            cfg.rmi_model_tag = require_value("--rmi-model-tag");
        } else if (arg == "--rmi-data-dir") {
            cfg.rmi_data_dir = resolve_local_path(require_value("--rmi-data-dir"));
        } else if (arg == "--rmi-generated-dir") {
            cfg.rmi_generated_dir = resolve_local_path(require_value("--rmi-generated-dir"));
        } else if (arg == "--cache-policy") {
            cfg.cache_policy = cam::cache::parse_policy_token(require_value("--cache-policy"));
        } else if (arg == "--cache-bytes") {
            cfg.cache_bytes = std::stoull(require_value("--cache-bytes"));
        } else if (arg == "--direct-io") {
            cfg.direct_io = true;
        } else if (arg == "--io-mode") {
            const std::string mode = cam::storage::to_upper(cam::storage::trim(require_value("--io-mode")));
            if (mode == "DIRECT" || mode == "O_DIRECT") {
                cfg.direct_io = true;
            } else if (mode == "BUFFERED" || mode == "DEFAULT") {
                cfg.direct_io = false;
            } else {
                usage_error("unknown io mode: " + mode);
            }
        } else if (arg == "--strategies") {
            cfg.strategies = cam::point_query::parse_search_strategy_list(require_value("--strategies"));
        } else if (arg == "--query-limit") {
            cfg.query_limit = std::stoull(require_value("--query-limit"));
        } else if (arg == "--summary-out") {
            cfg.summary_out = require_value("--summary-out");
        } else if (arg == "--append") {
            cfg.append = true;
        } else if (arg == "-h" || arg == "--help") {
            usage_error("help requested");
        } else {
            usage_error("unknown argument: " + arg);
        }
    }

    if (cfg.data_path.empty() || cfg.query_path.empty()) {
        usage_error("both --data and --queries are required");
    }
    if (cfg.rmi_branch_factors.empty()) {
        cfg.rmi_branch_factors = parse_rmi_list("all");
    }
    cfg.rmi_data_dir = resolve_local_path(cfg.rmi_data_dir);
    cfg.rmi_generated_dir = resolve_local_path(cfg.rmi_generated_dir);
    return cfg;
}

uint64_t nearest_rank_percentile(const std::vector<uint64_t>& sorted_values, double percentile) {
    if (sorted_values.empty()) {
        return 0;
    }
    if (percentile <= 0.0) {
        return sorted_values.front();
    }
    if (percentile >= 100.0) {
        return sorted_values.back();
    }

    const double rank = percentile / 100.0 * static_cast<double>(sorted_values.size());
    size_t idx = static_cast<size_t>(std::ceil(rank));
    if (idx == 0) {
        idx = 1;
    }
    idx = std::min(idx, sorted_values.size());
    return sorted_values[idx - 1];
}

void fill_io_size_stats(BreakdownResult& st, std::vector<uint64_t>& io_sizes) {
    if (io_sizes.empty()) {
        return;
    }

    double sum = 0.0;
    for (uint64_t bytes : io_sizes) {
        sum += static_cast<double>(bytes);
    }
    st.io_size_mean_bytes = sum / static_cast<double>(io_sizes.size());

    double sum_sq_diff = 0.0;
    for (uint64_t bytes : io_sizes) {
        const double diff = static_cast<double>(bytes) - st.io_size_mean_bytes;
        sum_sq_diff += diff * diff;
    }
    st.io_size_std_bytes = std::sqrt(sum_sq_diff / static_cast<double>(io_sizes.size()));

    std::sort(io_sizes.begin(), io_sizes.end());
    st.io_size_min_bytes = io_sizes.front();
    st.io_size_p50_bytes = nearest_rank_percentile(io_sizes, 50.0);
    st.io_size_p75_bytes = nearest_rank_percentile(io_sizes, 75.0);
    st.io_size_p90_bytes = nearest_rank_percentile(io_sizes, 90.0);
    st.io_size_p95_bytes = nearest_rank_percentile(io_sizes, 95.0);
    st.io_size_p99_bytes = nearest_rank_percentile(io_sizes, 99.0);
    st.io_size_max_bytes = io_sizes.back();
}

template <typename IndexT>
BreakdownResult run_breakdown_queries(
    const IndexT& index,
    const cam::storage::KeyFileLayout& data_layout,
    const std::vector<KeyType>& queries,
    SearchStrategy strategy,
    bool direct_io,
    CachePolicy cache_policy,
    size_t cache_bytes)
{
    BreakdownResult st;
    st.strategy = strategy;
    st.cache_policy = cache_policy;
    st.cache_bytes = cache_bytes;
    st.queries = queries.size();

    cam::storage::DiskManager disk(
        data_layout,
        cam::storage::make_page_cache(cache_policy, cache_bytes),
        direct_io);

    std::vector<Page> scratch_pages;
    std::vector<uint64_t> io_sizes;
    io_sizes.reserve(queries.size());
    const auto run_t0 = Clock::now();
    for (KeyType key : queries) {
        const auto result = cam::point_query::run_point_query_breakdown(
            index, disk, key, strategy, scratch_pages);
        st.page_requests += result.metrics.dac;
        st.cache_hits += result.metrics.buffer_hits;
        st.cache_misses += result.metrics.cam_io;
        st.logical_ios += result.metrics.disk_pages_read;
        st.physical_ios += result.metrics.device_ios;
        st.bytes_read += result.metrics.bytes_read;
        io_sizes.push_back(result.metrics.bytes_read);
        st.index_traversal_ns += result.metrics.index_traversal_ns;
        st.cache_ns += result.metrics.cache_ns;
        st.io_ns += result.metrics.io_ns;
        st.fetch_wall_ns += result.metrics.fetch_wall_ns;
        st.lastmile_search_ns += result.metrics.lastmile_search_ns;
        if (result.found) {
            ++st.found;
            st.checksum += result.matched_key;
        }
    }
    const auto run_t1 = Clock::now();
    st.wall_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(run_t1 - run_t0).count();
    fill_io_size_stats(st, io_sizes);
    return st;
}

std::string rmi_model_name(const Config& cfg, size_t branch_factor) {
    return cfg.rmi_prefix + "_" + cfg.rmi_model_tag + "_" + std::to_string(branch_factor);
}

std::string read_text_file(const fs::path& path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("failed to open RMI metadata file: " + path.string());
    }
    return std::string(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
}

std::vector<char> read_binary_file(const fs::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("failed to open RMI parameter file: " + path.string());
    }
    return std::vector<char>(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
}

double parse_double_constant(const std::string& text, const std::string& name, const fs::path& path) {
    const std::regex pattern("const\\s+double\\s+" + name + "\\s*=\\s*([^;]+);");
    std::smatch match;
    if (!std::regex_search(text, match, pattern)) {
        throw std::runtime_error("missing " + name + " in " + path.string());
    }
    return std::stod(match[1].str());
}

uint64_t parse_uint_constant_or(const std::string& text, const std::string& name, uint64_t fallback) {
    const std::regex pattern("const\\s+(?:size_t|uint64_t)\\s+" + name + "\\s*=\\s*([0-9]+);");
    std::smatch match;
    if (!std::regex_search(text, match, pattern)) {
        return fallback;
    }
    return static_cast<uint64_t>(std::stoull(match[1].str()));
}

template <typename T>
T read_unaligned(const std::vector<char>& bytes, size_t offset) {
    if (offset + sizeof(T) > bytes.size()) {
        throw std::runtime_error("RMI parameter read exceeds buffer size");
    }
    T value{};
    std::memcpy(&value, bytes.data() + offset, sizeof(T));
    return value;
}

RMIModelSpec load_rmi_model_spec(const Config& cfg, size_t branch_factor) {
    RMIModelSpec model;
    model.name = rmi_model_name(cfg, branch_factor);
    model.branch_factor = branch_factor;

    const fs::path header_path = fs::path(cfg.rmi_generated_dir) / (model.name + ".h");
    const fs::path data_header_path = fs::path(cfg.rmi_generated_dir) / (model.name + "_data.h");
    const fs::path l1_path = fs::path(cfg.rmi_data_dir) / (model.name + "_L1_PARAMETERS");

    const std::string header_text = read_text_file(header_path);
    const std::string data_header_text = read_text_file(data_header_path);
    model.l0_parameter0 = parse_double_constant(data_header_text, "L0_PARAMETER0", data_header_path);
    model.l0_parameter1 = parse_double_constant(data_header_text, "L0_PARAMETER1", data_header_path);
    model.l1_parameters = read_binary_file(l1_path);

    constexpr size_t record_size = 24;
    const size_t expected_bytes = branch_factor * record_size;
    if (model.l1_parameters.size() != expected_bytes) {
        std::ostringstream oss;
        oss << "bad RMI L1 parameter size for " << l1_path
            << ": expected " << expected_bytes << " bytes, got " << model.l1_parameters.size();
        throw std::runtime_error(oss.str());
    }

    model.index_bytes = static_cast<size_t>(
        parse_uint_constant_or(header_text, "RMI_SIZE", static_cast<uint64_t>(16 + model.l1_parameters.size())));
    model.build_time_ns = parse_uint_constant_or(header_text, "BUILD_TIME_NS", 0);
    return model;
}

class RMIIndexAdapter {
public:
    RMIIndexAdapter(RMIModelSpec model, size_t total_keys, size_t logical_pages)
        : model_(std::move(model)), total_keys_(total_keys), logical_pages_(logical_pages) {}

    std::pair<size_t, size_t> estimate_pages_for_key(const KeyType& key) const {
        if (total_keys_ == 0 || logical_pages_ == 0) {
            return {0, 0};
        }

        double fpred = std::fma(model_.l0_parameter1, static_cast<double>(key), model_.l0_parameter0);
        const size_t model_index = clamp_prediction(fpred, static_cast<double>(model_.branch_factor - 1));
        const size_t offset = model_index * 24;
        const double leaf_alpha = read_unaligned<double>(model_.l1_parameters, offset);
        const double leaf_beta = read_unaligned<double>(model_.l1_parameters, offset + 8);
        const uint64_t err = read_unaligned<uint64_t>(model_.l1_parameters, offset + 16);
        fpred = std::fma(leaf_beta, static_cast<double>(key), leaf_alpha);
        const uint64_t pred = static_cast<uint64_t>(
            clamp_prediction(fpred, static_cast<double>(total_keys_ - 1)));

        const uint64_t max_pos = static_cast<uint64_t>(total_keys_ - 1);
        const uint64_t lo = pred > err ? pred - err : 0;
        const uint64_t hi = pred >= max_pos - std::min<uint64_t>(err, max_pos)
            ? max_pos
            : pred + err;

        size_t page_lo = static_cast<size_t>(lo / ITEM_PER_PAGE);
        size_t page_hi = static_cast<size_t>(hi / ITEM_PER_PAGE);
        page_lo = std::min(page_lo, logical_pages_ - 1);
        page_hi = std::min(page_hi, logical_pages_ - 1);
        if (page_hi < page_lo) {
            page_hi = page_lo;
        }
        return {page_lo, page_hi};
    }

    const RMIModelSpec& spec() const {
        return model_;
    }

private:
    static size_t clamp_prediction(double value, double bound) {
        if (value < 0.0) {
            return 0;
        }
        return value > bound ? static_cast<size_t>(bound) : static_cast<size_t>(value);
    }

    RMIModelSpec model_;
    size_t total_keys_ = 0;
    size_t logical_pages_ = 0;
};

void print_header(std::ostream& out) {
    out
        << "label,baseline,io_mode,index_type,model,epsilon,branch_factor,index_bytes,build_time_ns,"
        << "policy,cache_bytes,strategy,queries,found,page_requests,cache_hits,cache_misses,"
        << "logical_ios,physical_ios,avg_logical_ios,avg_physical_ios,bytes_read,"
        << "io_size_mean_bytes,io_size_std_bytes,io_size_min_bytes,"
        << "io_size_p50_bytes,io_size_p75_bytes,io_size_p90_bytes,"
        << "io_size_p95_bytes,io_size_p99_bytes,io_size_max_bytes,"
        << "index_traversal_ns,cache_ns,io_ns,fetch_wall_ns,lastmile_search_ns,wall_ns,other_ns,"
        << "avg_index_traversal_ns,avg_cache_ns,avg_io_ns,avg_fetch_wall_ns,avg_lastmile_search_ns,"
        << "avg_wall_ns,throughput_qps,checksum\n";
}

void print_row(std::ostream& out, const BreakdownResult& st) {
    const double queries = static_cast<double>(st.queries);
    const double avg_lio = st.queries ? static_cast<double>(st.logical_ios) / queries : 0.0;
    const double avg_pio = st.queries ? static_cast<double>(st.physical_ios) / queries : 0.0;
    const double avg_index_ns = st.queries ? static_cast<double>(st.index_traversal_ns) / queries : 0.0;
    const double avg_cache_ns = st.queries ? static_cast<double>(st.cache_ns) / queries : 0.0;
    const double avg_io_ns = st.queries ? static_cast<double>(st.io_ns) / queries : 0.0;
    const double avg_fetch_ns = st.queries ? static_cast<double>(st.fetch_wall_ns) / queries : 0.0;
    const double avg_lastmile_ns = st.queries ? static_cast<double>(st.lastmile_search_ns) / queries : 0.0;
    const double avg_wall_ns = st.queries ? static_cast<double>(st.wall_ns) / queries : 0.0;
    const double qps = st.wall_ns ? queries * 1e9 / static_cast<double>(st.wall_ns) : 0.0;
    const long long accounted_ns =
        st.index_traversal_ns + st.cache_ns + st.io_ns + st.lastmile_search_ns;
    const long long other_ns = st.wall_ns > accounted_ns ? st.wall_ns - accounted_ns : 0;

    out
        << st.label << ','
        << st.baseline << ','
        << st.io_mode << ','
        << st.index_type << ','
        << st.model_name << ','
        << st.epsilon << ','
        << st.branch_factor << ','
        << st.index_bytes << ','
        << st.build_time_ns << ','
        << cam::cache::policy_name(st.cache_policy) << ','
        << st.cache_bytes << ','
        << cam::point_query::search_strategy_name(st.strategy) << ','
        << st.queries << ','
        << st.found << ','
        << st.page_requests << ','
        << st.cache_hits << ','
        << st.cache_misses << ','
        << st.logical_ios << ','
        << st.physical_ios << ','
        << std::fixed << std::setprecision(6) << avg_lio << ','
        << std::fixed << std::setprecision(6) << avg_pio << ','
        << st.bytes_read << ','
        << std::fixed << std::setprecision(6) << st.io_size_mean_bytes << ','
        << std::fixed << std::setprecision(6) << st.io_size_std_bytes << ','
        << st.io_size_min_bytes << ','
        << st.io_size_p50_bytes << ','
        << st.io_size_p75_bytes << ','
        << st.io_size_p90_bytes << ','
        << st.io_size_p95_bytes << ','
        << st.io_size_p99_bytes << ','
        << st.io_size_max_bytes << ','
        << st.index_traversal_ns << ','
        << st.cache_ns << ','
        << st.io_ns << ','
        << st.fetch_wall_ns << ','
        << st.lastmile_search_ns << ','
        << st.wall_ns << ','
        << other_ns << ','
        << std::fixed << std::setprecision(2) << avg_index_ns << ','
        << std::fixed << std::setprecision(2) << avg_cache_ns << ','
        << std::fixed << std::setprecision(2) << avg_io_ns << ','
        << std::fixed << std::setprecision(2) << avg_fetch_ns << ','
        << std::fixed << std::setprecision(2) << avg_lastmile_ns << ','
        << std::fixed << std::setprecision(2) << avg_wall_ns << ','
        << std::fixed << std::setprecision(2) << qps << ','
        << st.checksum
        << '\n';
}

template <size_t Eps>
void run_pgm_epsilon(
    std::ostream& out,
    const cam::storage::KeyFileLayout& data_layout,
    const std::vector<KeyType>& data,
    const std::vector<KeyType>& queries,
    const Config& cfg)
{
    using Index = pgm::PGMIndex<KeyType, Eps>;
    const auto build_t0 = Clock::now();
    Index index(data);
    const auto build_t1 = Clock::now();
    const auto build_time_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(build_t1 - build_t0).count();
    const size_t index_bytes = index.size_in_bytes();

    for (SearchStrategy strategy : cfg.strategies) {
        auto st = run_breakdown_queries(
            index,
            data_layout,
            queries,
            strategy,
            cfg.direct_io,
            cfg.cache_policy,
            cfg.cache_bytes);
        st.label = cfg.label;
        st.baseline = cfg.direct_io ? "PGM-DIRECT" : "PGM";
        st.io_mode = cfg.direct_io ? "direct" : "buffered";
        st.index_type = "PGM";
        st.model_name = "pgm_epsilon_" + std::to_string(Eps);
        st.epsilon = Eps;
        st.index_bytes = index_bytes;
        st.build_time_ns = static_cast<uint64_t>(build_time_ns);
        print_row(out, st);
    }
}

void run_pgm_epsilon_value(
    std::ostream& out,
    size_t epsilon,
    const cam::storage::KeyFileLayout& data_layout,
    const std::vector<KeyType>& data,
    const std::vector<KeyType>& queries,
    const Config& cfg)
{
    switch (epsilon) {
        case 4: run_pgm_epsilon<4>(out, data_layout, data, queries, cfg); break;
        case 8: run_pgm_epsilon<8>(out, data_layout, data, queries, cfg); break;
        case 10: run_pgm_epsilon<10>(out, data_layout, data, queries, cfg); break;
        case 12: run_pgm_epsilon<12>(out, data_layout, data, queries, cfg); break;
        case 14: run_pgm_epsilon<14>(out, data_layout, data, queries, cfg); break;
        case 16: run_pgm_epsilon<16>(out, data_layout, data, queries, cfg); break;
        case 18: run_pgm_epsilon<18>(out, data_layout, data, queries, cfg); break;
        case 20: run_pgm_epsilon<20>(out, data_layout, data, queries, cfg); break;
        case 24: run_pgm_epsilon<24>(out, data_layout, data, queries, cfg); break;
        case 28: run_pgm_epsilon<28>(out, data_layout, data, queries, cfg); break;
        case 32: run_pgm_epsilon<32>(out, data_layout, data, queries, cfg); break;
        case 36: run_pgm_epsilon<36>(out, data_layout, data, queries, cfg); break;
        case 40: run_pgm_epsilon<40>(out, data_layout, data, queries, cfg); break;
        case 48: run_pgm_epsilon<48>(out, data_layout, data, queries, cfg); break;
        case 52: run_pgm_epsilon<52>(out, data_layout, data, queries, cfg); break;
        case 64: run_pgm_epsilon<64>(out, data_layout, data, queries, cfg); break;
        case 96: run_pgm_epsilon<96>(out, data_layout, data, queries, cfg); break;
        case 128: run_pgm_epsilon<128>(out, data_layout, data, queries, cfg); break;
        default:
            throw std::invalid_argument(
                "unsupported epsilon for motivation_query_breakdown: " + std::to_string(epsilon));
    }
}

void run_rmi_model(
    std::ostream& out,
    size_t branch_factor,
    const cam::storage::KeyFileLayout& data_layout,
    const std::vector<KeyType>& queries,
    const Config& cfg)
{
    RMIIndexAdapter index(
        load_rmi_model_spec(cfg, branch_factor),
        data_layout.total_keys,
        data_layout.logical_pages);
    const auto& model = index.spec();
    for (SearchStrategy strategy : cfg.strategies) {
        auto st = run_breakdown_queries(
            index,
            data_layout,
            queries,
            strategy,
            cfg.direct_io,
            cfg.cache_policy,
            cfg.cache_bytes);
        st.label = cfg.label;
        st.baseline = cfg.direct_io ? "RMI-DIRECT" : "RMI";
        st.io_mode = cfg.direct_io ? "direct" : "buffered";
        st.index_type = "RMI";
        st.model_name = model.name;
        st.branch_factor = model.branch_factor;
        st.index_bytes = model.index_bytes;
        st.build_time_ns = model.build_time_ns;
        print_row(out, st);
    }
}

bool has_algorithm(const std::vector<Algorithm>& algorithms, Algorithm algorithm) {
    return std::find(algorithms.begin(), algorithms.end(), algorithm) != algorithms.end();
}

} // namespace

int main(int argc, char** argv) {
    try {
        Config cfg = parse_args(argc, argv);
        const auto data_layout = cam::storage::detect_key_file_layout(
            cfg.data_path, cfg.total_keys, cfg.header_mode);
        if (data_layout.total_keys == 0) {
            throw std::runtime_error("data file contains no keys");
        }

        auto queries = cam::storage::load_query_keys(cfg.query_path, cfg.query_limit);
        if (queries.empty()) {
            throw std::runtime_error("query file contains no keys");
        }

        std::ofstream file_out;
        std::ostream* out = &std::cout;
        if (!cfg.summary_out.empty()) {
            const fs::path out_path(cfg.summary_out);
            if (out_path.parent_path() != fs::path()) {
                fs::create_directories(out_path.parent_path());
            }
            file_out.open(cfg.summary_out, cfg.append ? std::ios::app : std::ios::out);
            if (!file_out) {
                throw std::runtime_error("failed to open summary output: " + cfg.summary_out);
            }
            out = &file_out;
        }

        if (!cfg.append) {
            print_header(*out);
        }

        if (has_algorithm(cfg.algorithms, Algorithm::PGM)) {
            auto data = cam::storage::load_key_file_keys(data_layout);
            for (size_t epsilon : cfg.epsilons) {
                run_pgm_epsilon_value(*out, epsilon, data_layout, data, queries, cfg);
            }
        }

        if (has_algorithm(cfg.algorithms, Algorithm::RMI)) {
            for (size_t branch_factor : cfg.rmi_branch_factors) {
                run_rmi_model(*out, branch_factor, data_layout, queries, cfg);
            }
        }

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[error] " << e.what() << '\n';
        return 1;
    }
}
