#include <algorithm>
#include <cctype>
#include <chrono>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "../src/cache/CacheUtils.hpp"
#include "../src/pgm/PointQuery.hpp"
#include "../src/storage/DiskManager.hpp"
#include "../src/storage/KeyFile.hpp"

#include "rmi_bench_registry.h"

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

namespace {

struct RMIModelSpec {
    const char* name = "";
    const char* layers = "";
    size_t branch_factor = 0;
    size_t index_bytes = 0;
    uint64_t build_time_ns = 0;
    bool (*load)(char const*) = nullptr;
    void (*cleanup)() = nullptr;
    uint64_t (*lookup)(uint64_t, size_t*) = nullptr;
};

static const std::vector<RMIModelSpec> kRmiModels = [] {
    std::vector<RMIModelSpec> models;
    models.reserve(16);
#define CAM_REGISTER_RMI(ns, model_layers, branch) \
    models.push_back({ns::NAME, model_layers, branch, ns::RMI_SIZE, ns::BUILD_TIME_NS, ns::load, ns::cleanup, ns::lookup});
    CAM_RMI_MODEL_LIST(CAM_REGISTER_RMI)
#undef CAM_REGISTER_RMI
    return models;
}();

struct Config {
    std::string data_path;
    std::string query_path;
    std::string rmi_data_dir = "src/rmi/rmi_data";
    std::string rmi_layers;
    size_t total_keys = 0;
    size_t total_budget_bytes = 64ULL << 20;
    bool has_fixed_cache_bytes = false;
    size_t fixed_cache_bytes = 0;
    cam::storage::HeaderMode header_mode = cam::storage::HeaderMode::AUTO;
    std::vector<SearchStrategy> strategies = {ALL_IN_ONCE};
    std::vector<CachePolicy> policies = {CachePolicy::FIFO, CachePolicy::LRU, CachePolicy::LFU};
    std::vector<const RMIModelSpec*> models;
    std::string rmi_selectors = "all";
    size_t query_limit = 0;
};

struct RunResult {
    const RMIModelSpec* model = nullptr;
    CachePolicy policy = CachePolicy::NONE;
    SearchStrategy strategy = ALL_IN_ONCE;

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

    size_t total_budget_bytes = 0;
    size_t index_bytes = 0;
    size_t cache_bytes = 0;
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
    if (kRmiModels.empty()) {
        return "<none>";
    }
    std::ostringstream oss;
    for (size_t i = 0; i < kRmiModels.size(); ++i) {
        if (i != 0) {
            oss << ',';
        }
        oss << kRmiModels[i].layers << ':' << kRmiModels[i].branch_factor;
    }
    return oss.str();
}

[[noreturn]] void usage_error(const std::string& msg) {
    throw std::invalid_argument(
        msg +
        "\nUsage: ./rmi_bench --data <file> --queries <file> [--rmi-data-dir <dir>]"
        " [--keys <n>] [--M <MiB>] [--cache-bytes <bytes>]"
        " [--header <auto|yes|no>] [--strategies <all_in_once,one_by_one|all>]"
        " [--policies <fifo,lru,lfu,none|all>]"
        " [--rmi-layers <top,leaf>] [--branch-factors <all|" + supported_rmi_list() + ">]"
        " [--query-limit <n>]");
}

const RMIModelSpec* find_rmi_model(
    const std::string& raw_token,
    const std::string& required_layers)
{
    const std::string token = cam::storage::trim(raw_token);
    if (token.empty()) {
        return nullptr;
    }

    std::vector<const RMIModelSpec*> matches;
    const bool all_digits = std::all_of(token.begin(), token.end(), [](unsigned char ch) {
        return std::isdigit(ch) != 0;
    });
    if (all_digits) {
        const size_t branch_factor = std::stoull(token);
        for (const auto& model : kRmiModels) {
            if (model.branch_factor == branch_factor &&
                (required_layers.empty() || required_layers == model.layers)) {
                matches.push_back(&model);
            }
        }
    } else {
        const std::string upper = cam::storage::to_upper(token);
        for (const auto& model : kRmiModels) {
            if (cam::storage::to_upper(model.name) == upper &&
                (required_layers.empty() || required_layers == model.layers)) {
                matches.push_back(&model);
            }
        }
    }

    if (matches.size() > 1) {
        throw std::invalid_argument(
            "ambiguous RMI selector " + token +
            "; pass --rmi-layers to choose a model family");
    }
    return matches.empty() ? nullptr : matches.front();
}

std::vector<const RMIModelSpec*> parse_rmi_list(
    const std::string& value,
    const std::string& required_layers)
{
    if (kRmiModels.empty()) {
        throw std::invalid_argument(
            "no RMI models were compiled into rmi_bench; run exp/generate_rmi_headers.sh "
            "with the branch factors you want to benchmark, then rebuild rmi_bench");
    }

    const std::string upper = cam::storage::to_upper(cam::storage::trim(value));
    if (upper.empty() || upper == "ALL") {
        std::vector<const RMIModelSpec*> all;
        all.reserve(kRmiModels.size());
        for (const auto& model : kRmiModels) {
            if (required_layers.empty() || required_layers == model.layers) {
                all.push_back(&model);
            }
        }
        if (all.empty()) {
            throw std::invalid_argument(
                "no compiled RMI models use requested layers " + required_layers);
        }
        return all;
    }

    std::vector<const RMIModelSpec*> out;
    std::stringstream ss(value);
    std::string token;
    while (std::getline(ss, token, ',')) {
        token = cam::storage::trim(token);
        if (token.empty()) {
            continue;
        }

        const RMIModelSpec* model = find_rmi_model(token, required_layers);
        if (!model) {
            throw std::invalid_argument(
                "unknown RMI selector: " + token +
                " (supported branch factors: " + supported_rmi_list() + ")");
        }
        if (std::find(out.begin(), out.end(), model) == out.end()) {
            out.push_back(model);
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
        } else if (arg == "--rmi-data-dir") {
            cfg.rmi_data_dir = resolve_local_path(require_value("--rmi-data-dir"));
        } else if (arg == "--keys") {
            cfg.total_keys = std::stoull(require_value("--keys"));
        } else if (arg == "--M") {
            cfg.total_budget_bytes = std::stoull(require_value("--M")) << 20;
        } else if (arg == "--cache-bytes") {
            cfg.fixed_cache_bytes = std::stoull(require_value("--cache-bytes"));
            cfg.has_fixed_cache_bytes = true;
        } else if (arg == "--header") {
            cfg.header_mode = cam::storage::parse_header_mode(require_value("--header"));
        } else if (arg == "--strategies") {
            cfg.strategies =
                cam::point_query::parse_search_strategy_list(require_value("--strategies"));
        } else if (arg == "--policies") {
            cfg.policies = cam::cache::parse_policy_list(require_value("--policies"));
        } else if (arg == "--rmi-layers") {
            cfg.rmi_layers = require_value("--rmi-layers");
        } else if (arg == "--branch-factors" || arg == "--rmis") {
            cfg.rmi_selectors = require_value(arg.c_str());
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
    cfg.models = parse_rmi_list(cfg.rmi_selectors, cfg.rmi_layers);
    cfg.rmi_data_dir = resolve_local_path(cfg.rmi_data_dir);
    return cfg;
}

class RMIIndexAdapter {
public:
    RMIIndexAdapter(const RMIModelSpec& model, size_t total_keys, size_t logical_pages)
        : model_(model), total_keys_(total_keys), logical_pages_(logical_pages) {}

    std::pair<size_t, size_t> estimate_pages_for_key(const KeyType& key) const {
        if (total_keys_ == 0 || logical_pages_ == 0) {
            return {0, 0};
        }

        size_t err = 0;
        const uint64_t pred = model_.lookup(static_cast<uint64_t>(key), &err);
        const uint64_t max_pos = static_cast<uint64_t>(total_keys_ - 1);
        const uint64_t lo = pred > err ? pred - err : 0;
        const uint64_t hi = pred >= max_pos - std::min<uint64_t>(err, max_pos) ? max_pos : pred + err;

        size_t page_lo = static_cast<size_t>(lo / ITEM_PER_PAGE);
        size_t page_hi = static_cast<size_t>(hi / ITEM_PER_PAGE);
        page_lo = std::min(page_lo, logical_pages_ - 1);
        page_hi = std::min(page_hi, logical_pages_ - 1);
        if (page_hi < page_lo) {
            page_hi = page_lo;
        }
        return {page_lo, page_hi};
    }

private:
    const RMIModelSpec& model_;
    size_t total_keys_ = 0;
    size_t logical_pages_ = 0;
};

class LoadedRMI {
public:
    LoadedRMI(const RMIModelSpec& model, const std::string& data_dir) : model_(model) {
        if (!model_.load(data_dir.c_str())) {
            throw std::runtime_error(
                "failed to load RMI parameters for " + std::string(model_.name) +
                " from " + data_dir);
        }
    }

    ~LoadedRMI() {
        model_.cleanup();
    }

    LoadedRMI(const LoadedRMI&) = delete;
    LoadedRMI& operator=(const LoadedRMI&) = delete;

private:
    const RMIModelSpec& model_;
};

RunResult run_one_policy(
    const RMIModelSpec& model,
    CachePolicy policy,
    SearchStrategy strategy,
    const cam::storage::KeyFileLayout& data_layout,
    const std::vector<KeyType>& queries,
    size_t total_budget_bytes,
    bool has_fixed_cache_bytes,
    size_t fixed_cache_bytes)
{
    RunResult st;
    st.model = &model;
    st.policy = policy;
    st.strategy = strategy;
    st.queries = queries.size();
    st.total_budget_bytes = total_budget_bytes;
    st.index_bytes = model.index_bytes;
    if (has_fixed_cache_bytes) {
        st.cache_bytes = fixed_cache_bytes;
        if (st.cache_bytes > total_budget_bytes || st.index_bytes > total_budget_bytes - st.cache_bytes) {
            std::ostringstream oss;
            oss << "fixed cache bytes plus index bytes exceeds total budget for "
                << model.name << ": cache_bytes=" << st.cache_bytes
                << " index_bytes=" << st.index_bytes
                << " total_budget_bytes=" << total_budget_bytes;
            throw std::runtime_error(oss.str());
        }
    } else {
        st.cache_bytes = total_budget_bytes > model.index_bytes
            ? total_budget_bytes - model.index_bytes
            : 0;
    }

    RMIIndexAdapter index(model, data_layout.total_keys, data_layout.logical_pages);
    cam::storage::DiskManager disk(
        data_layout,
        cam::storage::make_page_cache(policy, st.cache_bytes));

    const auto t0 = Clock::now();
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
    const auto t1 = Clock::now();
    st.wall_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    return st;
}

void print_header() {
    std::cout
        << "model,layers,branch_factor,total_budget_bytes,index_bytes,cache_bytes,build_time_ns,"
        << "policy,strategy,queries,found,page_requests,cache_hits,cache_misses,hit_ratio,"
        << "logical_ios,physical_ios,avg_logical_ios,avg_physical_ios,"
        << "bytes_read,io_ns,wall_ns,throughput_qps,checksum\n";
}

void print_row(const RunResult& st) {
    const double hit_ratio =
        st.page_requests ? static_cast<double>(st.cache_hits) / st.page_requests : 0.0;
    const double avg_lio =
        st.queries ? static_cast<double>(st.logical_ios) / st.queries : 0.0;
    const double avg_pio =
        st.queries ? static_cast<double>(st.physical_ios) / st.queries : 0.0;
    const double qps =
        st.wall_ns ? static_cast<double>(st.queries) * 1e9 / st.wall_ns : 0.0;

    std::cout
        << st.model->name << ','
        << "\"" << st.model->layers << "\","
        << st.model->branch_factor << ','
        << st.total_budget_bytes << ','
        << st.index_bytes << ','
        << st.cache_bytes << ','
        << st.model->build_time_ns << ','
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
        << st.checksum
        << '\n';
}

void run_model(
    const RMIModelSpec& model,
    const cam::storage::KeyFileLayout& data_layout,
    const std::vector<KeyType>& queries,
    const Config& cfg)
{
    LoadedRMI loaded(model, cfg.rmi_data_dir);
    for (SearchStrategy strategy : cfg.strategies) {
        for (CachePolicy policy : cfg.policies) {
            const auto st = run_one_policy(
                model, policy, strategy, data_layout, queries,
                cfg.total_budget_bytes, cfg.has_fixed_cache_bytes, cfg.fixed_cache_bytes);
            print_row(st);
        }
    }
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

        auto queries = cam::storage::load_query_keys(cfg.query_path);
        if (cfg.query_limit > 0 && queries.size() > cfg.query_limit) {
            queries.resize(cfg.query_limit);
        }

        print_header();
        for (const RMIModelSpec* model : cfg.models) {
            run_model(*model, data_layout, queries, cfg);
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[error] " << e.what() << '\n';
        return 1;
    }
}
