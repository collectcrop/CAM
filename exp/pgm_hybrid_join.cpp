#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

#include "../src/cache/CacheUtils.hpp"
#include "../src/pgm/PointQuery.hpp"
#include "../src/pgm/RangeQuery.hpp"
#include "../src/pgm/pgm_index.hpp"
#include "../src/sort/ExternalMergeSort.hpp"
#include "../src/storage/DiskManager.hpp"
#include "../src/storage/KeyFile.hpp"
#include "../src/utils.hpp"

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;
using KeyT = uint64_t;

namespace {

struct ProbeSpec {
    uint8_t is_range = 0;
    uint64_t len = 0;
};

enum class ExecutionMode {
    HYBRID,
    POINT,
    RANGE,
    INLJ
};

struct Config {
    std::string data_path;
    std::string query_path;
    std::string par_path;
    std::string bitmap_path;
    std::string output_path;
    std::string label = "hybrid";
    std::string work_dir = "build/tmp/hybrid_join_sort";
    ExecutionMode mode = ExecutionMode::HYBRID;
    size_t epsilon = 16;
    size_t memory_budget_bytes = 256ULL << 20;
    size_t sort_budget_bytes = 32ULL << 20;
    size_t total_keys = 0;
    CachePolicy policy = CachePolicy::LRU;
    bool append_output = false;
    bool sort_queries = true;
    bool keep_sorted_file = false;
};

struct Counter {
    uint64_t page_requests = 0;
    uint64_t cache_hits = 0;
    uint64_t cache_misses = 0;
    uint64_t logical_ios = 0;
    uint64_t physical_ios = 0;
    uint64_t bytes_read = 0;
    long long io_ns = 0;

    void add(const cam::point_query::PointQueryMetrics& metrics) {
        page_requests += metrics.dac;
        cache_hits += metrics.buffer_hits;
        cache_misses += metrics.cam_io;
        logical_ios += metrics.disk_pages_read;
        physical_ios += metrics.device_ios;
        bytes_read += metrics.bytes_read;
        io_ns += metrics.io_ns;
    }

    void add(const cam::range_query::RangeQueryMetrics& metrics) {
        page_requests += metrics.dac;
        cache_hits += metrics.buffer_hits;
        cache_misses += metrics.cam_io;
        logical_ios += metrics.disk_pages_read;
        physical_ios += metrics.device_ios;
        bytes_read += metrics.bytes_read;
        io_ns += metrics.io_ns;
    }
};

struct RunResult {
    std::string label;
    ExecutionMode mode = ExecutionMode::HYBRID;
    CachePolicy policy = CachePolicy::NONE;
    size_t epsilon = 0;
    size_t index_bytes = 0;
    size_t cache_bytes = 0;

    size_t queries = 0;
    size_t partitions = 0;
    size_t point_partitions = 0;
    size_t range_partitions = 0;
    size_t point_queries = 0;
    size_t range_queries = 0;
    size_t matched_records = 0;
    uint64_t checksum = 0;
    cam::sort::SortStats sort;
    long long query_wall_ns = 0;
    long long wall_ns = 0;

    Counter total;
    Counter point;
    Counter range;
};

[[noreturn]] void usage_error(const std::string& msg) {
    throw std::invalid_argument(
        msg +
        "\nUsage: ./pgm_hybrid_join --data <file> --queries <file>"
        " --output <csv> [--mode <hybrid|point|range|inlj>]"
        " [--par <file> --bitmap <file>]"
        " [--label <name>] [--epsilon <n>] [--M <MiB>] [--keys <n>]"
        " [--sort-M <MiB>] [--work-dir <dir>]  # --sort-M accepts decimals, e.g. 0.5"
        " [--policy <lru|fifo|lfu|none>] [--append] [--keep-sorted-file] [--no-sort-queries]");
}

size_t detect_record_count(const std::string& filename) {
    const auto bytes = fs::file_size(filename);
    if (bytes % sizeof(KeyT) != 0) {
        throw std::runtime_error("data file size is not a multiple of key size");
    }
    return bytes / sizeof(KeyT);
}

size_t safe_subtract(size_t lhs, size_t rhs) {
    return lhs > rhs ? lhs - rhs : 0;
}

void ensure_parent_dir(const std::string& path) {
    const fs::path p(path);
    const fs::path parent = p.parent_path();
    if (!parent.empty()) {
        fs::create_directories(parent);
    }
}

bool should_write_header(const Config& cfg) {
    if (!cfg.append_output) {
        return true;
    }
    if (!fs::exists(cfg.output_path)) {
        return true;
    }
    return fs::file_size(cfg.output_path) == 0;
}

ExecutionMode parse_execution_mode(const std::string& value) {
    const std::string token = cam::storage::to_upper(cam::storage::trim(value));
    if (token == "HYBRID") return ExecutionMode::HYBRID;
    if (token == "POINT") return ExecutionMode::POINT;
    if (token == "RANGE") return ExecutionMode::RANGE;
    if (token == "INLJ") return ExecutionMode::INLJ;
    throw std::invalid_argument("unknown execution mode: " + value);
}

std::string execution_mode_name(ExecutionMode mode) {
    switch (mode) {
        case ExecutionMode::HYBRID: return "hybrid";
        case ExecutionMode::POINT: return "point";
        case ExecutionMode::RANGE: return "range";
        case ExecutionMode::INLJ: return "inlj";
        default: return "unknown";
    }
}

bool mode_requires_sorted_workload(ExecutionMode mode) {
    return mode == ExecutionMode::HYBRID ||
           mode == ExecutionMode::POINT ||
           mode == ExecutionMode::RANGE;
}

fs::path make_mode_work_dir(const Config& cfg) {
    fs::path base = fs::absolute(cfg.work_dir);
    base /= cfg.label + "_" + execution_mode_name(cfg.mode);
    return base;
}

size_t parse_mib_bytes(const std::string& value, const std::string& flag) {
    size_t consumed = 0;
    const double mib = std::stod(value, &consumed);
    if (consumed != value.size() || !std::isfinite(mib) || mib <= 0.0) {
        throw std::invalid_argument(flag + " must be a positive MiB value: " + value);
    }

    const double bytes = mib * 1024.0 * 1024.0;
    const double max_size = static_cast<double>(std::numeric_limits<size_t>::max());
    if (bytes > max_size) {
        throw std::overflow_error(flag + " is too large: " + value);
    }
    return static_cast<size_t>(std::llround(bytes));
}

KeyT fix_sentinel(KeyT key) {
    constexpr KeyT sentinel = std::numeric_limits<KeyT>::max();
    return key == sentinel ? sentinel - 1 : key;
}

struct QueryWorkload {
    cam::storage::KeyFileLayout layout;
    cam::sort::SortStats sort;
    fs::path sort_dir;
    bool remove_sort_dir = false;
    size_t chunk_keys = 1;
};

size_t query_chunk_keys(const Config& cfg) {
    return std::max<size_t>(1, cfg.sort_budget_bytes / sizeof(KeyT));
}

class KeyFileCursor {
public:
    KeyFileCursor(
        const cam::storage::KeyFileLayout& layout,
        size_t start_key,
        size_t key_count)
        : in_(layout.path, std::ios::binary),
          remaining_(key_count)
    {
        if (!in_) {
            throw std::runtime_error("failed to open query file: " + layout.path);
        }
        if (start_key > layout.total_keys || key_count > layout.total_keys - start_key) {
            throw std::runtime_error("query cursor range exceeds key file layout");
        }
        const size_t offset = layout.header_bytes + start_key * sizeof(KeyT);
        in_.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
        if (!in_) {
            throw std::runtime_error("failed to seek query file: " + layout.path);
        }
    }

    bool read_next(std::vector<KeyT>& out, size_t max_keys) {
        out.clear();
        if (remaining_ == 0) {
            return false;
        }

        const size_t n = std::min(max_keys, remaining_);
        out.resize(n);
        in_.read(
            reinterpret_cast<char*>(out.data()),
            static_cast<std::streamsize>(n * sizeof(KeyT)));
        if (!in_) {
            throw std::runtime_error("failed to read query chunk");
        }
        for (KeyT& key : out) {
            key = fix_sentinel(key);
        }
        remaining_ -= n;
        return true;
    }

private:
    std::ifstream in_;
    size_t remaining_ = 0;
};

KeyT read_key_at(const cam::storage::KeyFileLayout& layout, size_t key_idx) {
    if (key_idx >= layout.total_keys) {
        throw std::runtime_error("query key index exceeds layout");
    }

    std::ifstream in(layout.path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("failed to open query file: " + layout.path);
    }
    const size_t offset = layout.header_bytes + key_idx * sizeof(KeyT);
    in.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
    KeyT key = 0;
    in.read(reinterpret_cast<char*>(&key), sizeof(KeyT));
    if (!in) {
        throw std::runtime_error("failed to read query key");
    }
    return fix_sentinel(key);
}

QueryWorkload prepare_execution_queries(
    const Config& cfg,
    const cam::storage::KeyFileLayout& query_layout)
{
    QueryWorkload workload;
    workload.layout = query_layout;
    workload.chunk_keys = query_chunk_keys(cfg);

    if (!cfg.sort_queries || !mode_requires_sorted_workload(cfg.mode)) {
        return workload;
    }

    const fs::path sort_dir = make_mode_work_dir(cfg);
    std::error_code ec;
    fs::remove_all(sort_dir, ec);
    fs::create_directories(sort_dir);

    workload.sort = cam::sort::external_merge_sort(
        query_layout,
        cfg.sort_budget_bytes,
        sort_dir);
    workload.layout = workload.sort.output_layout;
    workload.sort_dir = sort_dir;
    workload.remove_sort_dir = !cfg.keep_sorted_file;
    return workload;
}

void cleanup_query_workload(const QueryWorkload& workload) {
    if (workload.remove_sort_dir && !workload.sort_dir.empty()) {
        std::error_code ec;
        fs::remove_all(workload.sort_dir, ec);
    }
}

Config parse_args(int argc, char** argv) {
    Config cfg;
    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);
        auto require = [&](const char* flag) -> std::string {
            if (i + 1 >= argc) {
                usage_error(std::string("missing value for ") + flag);
            }
            return argv[++i];
        };

        if (arg == "--data") {
            cfg.data_path = cam::storage::resolve_dataset_path(require("--data"));
        } else if (arg == "--queries") {
            cfg.query_path = cam::storage::resolve_dataset_path(require("--queries"));
        } else if (arg == "--par") {
            cfg.par_path = cam::storage::resolve_dataset_path(require("--par"));
        } else if (arg == "--bitmap") {
            cfg.bitmap_path = cam::storage::resolve_dataset_path(require("--bitmap"));
        } else if (arg == "--output") {
            cfg.output_path = require("--output");
        } else if (arg == "--label") {
            cfg.label = require("--label");
        } else if (arg == "--mode") {
            cfg.mode = parse_execution_mode(require("--mode"));
        } else if (arg == "--epsilon") {
            cfg.epsilon = std::stoull(require("--epsilon"));
        } else if (arg == "--M") {
            cfg.memory_budget_bytes = std::stoull(require("--M")) << 20;
        } else if (arg == "--sort-M") {
            cfg.sort_budget_bytes = parse_mib_bytes(require("--sort-M"), "--sort-M");
        } else if (arg == "--work-dir") {
            cfg.work_dir = require("--work-dir");
        } else if (arg == "--keys") {
            cfg.total_keys = std::stoull(require("--keys"));
        } else if (arg == "--policy") {
            cfg.policy = cam::cache::parse_policy_token(require("--policy"));
        } else if (arg == "--append") {
            cfg.append_output = true;
        } else if (arg == "--keep-sorted-file") {
            cfg.keep_sorted_file = true;
        } else if (arg == "--no-sort-queries") {
            cfg.sort_queries = false;
        } else if (arg == "-h" || arg == "--help") {
            usage_error("help requested");
        } else {
            usage_error("unknown argument: " + arg);
        }
    }

    if (cfg.data_path.empty()) usage_error("--data is required");
    if (cfg.query_path.empty()) usage_error("--queries is required");
    if (cfg.mode == ExecutionMode::HYBRID && cfg.par_path.empty()) usage_error("--par is required for hybrid mode");
    if (cfg.mode == ExecutionMode::HYBRID && cfg.bitmap_path.empty()) usage_error("--bitmap is required for hybrid mode");
    if (cfg.mode == ExecutionMode::HYBRID && !cfg.sort_queries) {
        usage_error("hybrid mode requires sorted workload because par/bitmap are generated over sorted queries");
    }
    if (cfg.mode == ExecutionMode::RANGE && !cfg.sort_queries) {
        usage_error("range mode requires sorted workload for streaming query processing");
    }
    if (mode_requires_sorted_workload(cfg.mode) && cfg.sort_queries &&
        cfg.sort_budget_bytes < cam::sort::kMinSortBytes) {
        usage_error("--sort-M is too small for external merge sort");
    }
    if (cfg.output_path.empty()) usage_error("--output is required");
    if (cfg.total_keys == 0) {
        cfg.total_keys = detect_record_count(cfg.data_path);
    }
    return cfg;
}

std::vector<ProbeSpec> load_specs(
    const std::string& par_path,
    const std::string& bitmap_path)
{
    const auto lengths = load_binary<uint64_t>(par_path, false);
    const auto bitmap = load_binary<uint8_t>(bitmap_path, false);
    if (lengths.size() != bitmap.size()) {
        throw std::runtime_error("par and bitmap have different partition counts");
    }

    std::vector<ProbeSpec> specs;
    specs.reserve(lengths.size());
    for (size_t i = 0; i < lengths.size(); ++i) {
        if (bitmap[i] != 0 && bitmap[i] != 1) {
            throw std::runtime_error("bitmap contains value other than 0 or 1");
        }
        if (lengths[i] == 0) {
            throw std::runtime_error("par contains zero-length partition");
        }
        specs.push_back(ProbeSpec{bitmap[i], lengths[i]});
    }
    return specs;
}

void validate_specs(
    const std::vector<ProbeSpec>& specs,
    size_t query_count)
{
    uint64_t expected = 0;
    for (const ProbeSpec& spec : specs) {
        expected += spec.len;
    }
    if (expected != query_count) {
        throw std::runtime_error(
            "sum(par lengths) does not match query count: expected " +
            std::to_string(expected) + ", got " + std::to_string(query_count));
    }
}

struct StreamRangeResult {
    cam::range_query::RangeQueryMetrics metrics;
    size_t matched_records = 0;
    uint64_t checksum = 0;
};

template <typename IndexT>
void run_point_keys(
    const IndexT& index,
    cam::storage::DiskManager& disk,
    KeyFileCursor& cursor,
    size_t key_count,
    size_t chunk_keys,
    RunResult& st)
{
    std::vector<KeyT> chunk;
    size_t remaining = key_count;
    while (remaining > 0) {
        const size_t want = std::min(chunk_keys, remaining);
        if (!cursor.read_next(chunk, want)) {
            throw std::runtime_error("unexpected EOF while reading point query chunk");
        }
        remaining -= chunk.size();

        for (KeyT key : chunk) {
            const auto result =
                cam::point_query::run_point_query(index, disk, key, ALL_IN_ONCE);
            st.point.add(result.metrics);
            st.total.add(result.metrics);
            if (result.found) {
                ++st.matched_records;
                st.checksum += result.matched_key;
            }
        }
    }
}

template <typename IndexT>
StreamRangeResult run_range_query_streamed(
    const IndexT& index,
    cam::storage::DiskManager& disk,
    KeyT lo,
    KeyT hi,
    KeyFileCursor& query_cursor,
    size_t query_count,
    size_t chunk_keys)
{
    StreamRangeResult result;
    if (query_count == 0) {
        return result;
    }
    if (hi < lo) {
        std::swap(lo, hi);
    }
    if (disk.page_count() == 0) {
        return result;
    }

    const auto [page_lo, page_hi] =
        cam::range_query::detail::estimate_page_window(index, lo, hi);

    const auto before = disk.stats();
    const std::vector<Page> pages = disk.fetch_window(page_lo, page_hi);
    const auto after = disk.stats();
    result.metrics = cam::range_query::diff_metrics(before, after);

    std::vector<KeyT> query_chunk;
    size_t query_remaining = query_count;
    size_t query_idx = 0;
    KeyT current_query = 0;

    auto load_next_query_chunk = [&]() -> bool {
        if (query_remaining == 0) {
            return false;
        }
        const size_t want = std::min(chunk_keys, query_remaining);
        if (!query_cursor.read_next(query_chunk, want)) {
            throw std::runtime_error("unexpected EOF while reading range query chunk");
        }
        query_remaining -= query_chunk.size();
        query_idx = 0;
        return !query_chunk.empty();
    };

    auto advance_query = [&]() -> bool {
        while (query_idx >= query_chunk.size()) {
            if (!load_next_query_chunk()) {
                return false;
            }
        }
        current_query = query_chunk[query_idx++];
        return true;
    };

    if (!advance_query()) {
        return result;
    }

    const auto record_less_key = [](const Record& record, KeyT key) {
        return record.key < key;
    };

    for (const auto& page : pages) {
        auto [record_it, record_end] =
            cam::range_query::detail::page_record_range(page, lo, hi);
        if (record_it == nullptr || record_it == record_end) {
            continue;
        }

        while (record_it != record_end) {
            while (current_query < record_it->key) {
                if (!advance_query()) {
                    return result;
                }
            }

            if (current_query == record_it->key) {
                ++result.matched_records;
                result.checksum += record_it->key;
                ++record_it;
            } else {
                record_it = std::lower_bound(record_it, record_end, current_query, record_less_key);
            }
        }
    }

    return result;
}

template <typename IndexT>
RunResult run_hybrid_join(
    const IndexT& index,
    const cam::storage::KeyFileLayout& data_layout,
    const QueryWorkload& workload,
    const std::vector<ProbeSpec>& specs,
    const Config& cfg,
    size_t index_bytes)
{
    RunResult st;
    st.label = cfg.label;
    st.mode = cfg.mode;
    st.policy = cfg.policy;
    st.epsilon = cfg.epsilon;
    st.index_bytes = index_bytes;
    st.cache_bytes = safe_subtract(cfg.memory_budget_bytes, index_bytes);
    st.sort = workload.sort;
    st.queries = workload.layout.total_keys;
    st.partitions = specs.size();

    cam::storage::DiskManager disk(
        data_layout,
        cam::storage::make_page_cache(cfg.policy, st.cache_bytes));

    KeyFileCursor cursor(workload.layout, 0, workload.layout.total_keys);
    size_t q_offset = 0;
    const auto t0 = Clock::now();

    for (const ProbeSpec& spec : specs) {
        if (q_offset + spec.len > workload.layout.total_keys) {
            throw std::runtime_error("partition exceeds query file length");
        }

        if (spec.is_range == 0) {
            ++st.point_partitions;
            st.point_queries += spec.len;
            run_point_keys(index, disk, cursor, spec.len, workload.chunk_keys, st);
        } else {
            ++st.range_partitions;
            st.range_queries += spec.len;

            const KeyT lo = read_key_at(workload.layout, q_offset);
            const KeyT hi = read_key_at(
                workload.layout,
                q_offset + static_cast<size_t>(spec.len) - 1);
            const auto range_result = run_range_query_streamed(
                index,
                disk,
                lo,
                hi,
                cursor,
                spec.len,
                workload.chunk_keys);
            st.range.add(range_result.metrics);
            st.total.add(range_result.metrics);
            st.matched_records += range_result.matched_records;
            st.checksum += range_result.checksum;
        }

        q_offset += static_cast<size_t>(spec.len);
    }

    const auto t1 = Clock::now();
    st.query_wall_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    st.wall_ns = st.sort.wall_ns + st.query_wall_ns;
    return st;
}

template <typename IndexT>
RunResult run_point_join(
    const IndexT& index,
    const cam::storage::KeyFileLayout& data_layout,
    const QueryWorkload& workload,
    const Config& cfg,
    size_t index_bytes)
{
    RunResult st;
    st.label = cfg.label;
    st.mode = cfg.mode;
    st.policy = cfg.policy;
    st.epsilon = cfg.epsilon;
    st.index_bytes = index_bytes;
    st.cache_bytes = safe_subtract(cfg.memory_budget_bytes, index_bytes);
    st.sort = workload.sort;
    st.queries = workload.layout.total_keys;
    st.partitions = workload.layout.total_keys;
    st.point_partitions = workload.layout.total_keys;
    st.point_queries = workload.layout.total_keys;

    cam::storage::DiskManager disk(
        data_layout,
        cam::storage::make_page_cache(cfg.policy, st.cache_bytes));

    KeyFileCursor cursor(workload.layout, 0, workload.layout.total_keys);
    const auto t0 = Clock::now();
    run_point_keys(index, disk, cursor, workload.layout.total_keys, workload.chunk_keys, st);
    const auto t1 = Clock::now();
    st.query_wall_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    st.wall_ns = st.sort.wall_ns + st.query_wall_ns;
    return st;
}

template <typename IndexT>
RunResult run_single_range_join(
    const IndexT& index,
    const cam::storage::KeyFileLayout& data_layout,
    const QueryWorkload& workload,
    const Config& cfg,
    size_t index_bytes)
{
    RunResult st;
    st.label = cfg.label;
    st.mode = cfg.mode;
    st.policy = cfg.policy;
    st.epsilon = cfg.epsilon;
    st.index_bytes = index_bytes;
    st.cache_bytes = safe_subtract(cfg.memory_budget_bytes, index_bytes);
    st.sort = workload.sort;
    st.queries = workload.layout.total_keys;
    st.partitions = workload.layout.total_keys == 0 ? 0 : 1;
    st.range_partitions = workload.layout.total_keys == 0 ? 0 : 1;
    st.range_queries = workload.layout.total_keys;

    cam::storage::DiskManager disk(
        data_layout,
        cam::storage::make_page_cache(cfg.policy, st.cache_bytes));

    const auto t0 = Clock::now();
    if (workload.layout.total_keys > 0) {
        const KeyT lo = read_key_at(workload.layout, 0);
        const KeyT hi = read_key_at(workload.layout, workload.layout.total_keys - 1);
        KeyFileCursor cursor(workload.layout, 0, workload.layout.total_keys);
        const auto range_result = run_range_query_streamed(
            index,
            disk,
            lo,
            hi,
            cursor,
            workload.layout.total_keys,
            workload.chunk_keys);
        st.range.add(range_result.metrics);
        st.total.add(range_result.metrics);
        st.matched_records += range_result.matched_records;
        st.checksum += range_result.checksum;
    }
    const auto t1 = Clock::now();
    st.query_wall_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    st.wall_ns = st.sort.wall_ns + st.query_wall_ns;
    return st;
}

void write_header(std::ostream& out) {
    out
        << "label,mode,epsilon,policy,queries,partitions,point_partitions,range_partitions,"
        << "point_queries,range_queries,matched_records,"
        << "page_requests,cache_hits,cache_misses,hit_ratio,"
        << "logical_ios,physical_ios,avg_logical_ios,avg_physical_ios,"
        << "bytes_read,io_ns,sort_wall_ns,query_wall_ns,wall_ns,throughput_qps,"
        << "index_bytes,cache_bytes,sort_budget_bytes,sort_initial_runs,sort_merge_passes,"
        << "sort_runs_written,sort_input_bytes,sort_output_bytes,checksum\n";
}

void write_row(std::ostream& out, const RunResult& st) {
    const double hit_ratio = st.total.page_requests == 0 ? 0.0 :
        static_cast<double>(st.total.cache_hits) /
        static_cast<double>(st.total.page_requests);
    const double avg_lio = st.queries == 0 ? 0.0 :
        static_cast<double>(st.total.logical_ios) /
        static_cast<double>(st.queries);
    const double avg_pio = st.queries == 0 ? 0.0 :
        static_cast<double>(st.total.physical_ios) /
        static_cast<double>(st.queries);
    const double qps = st.wall_ns == 0 ? 0.0 :
        static_cast<double>(st.queries) * 1e9 /
        static_cast<double>(st.wall_ns);

    out << std::fixed << std::setprecision(6)
        << st.label << ','
        << execution_mode_name(st.mode) << ','
        << st.epsilon << ','
        << cam::cache::policy_name(st.policy) << ','
        << st.queries << ','
        << st.partitions << ','
        << st.point_partitions << ','
        << st.range_partitions << ','
        << st.point_queries << ','
        << st.range_queries << ','
        << st.matched_records << ','
        << st.total.page_requests << ','
        << st.total.cache_hits << ','
        << st.total.cache_misses << ','
        << hit_ratio << ','
        << st.total.logical_ios << ','
        << st.total.physical_ios << ','
        << avg_lio << ','
        << avg_pio << ','
        << st.total.bytes_read << ','
        << st.total.io_ns << ','
        << st.sort.wall_ns << ','
        << st.query_wall_ns << ','
        << st.wall_ns << ','
        << qps << ','
        << st.index_bytes << ','
        << st.cache_bytes << ','
        << st.sort.sort_budget_bytes << ','
        << st.sort.initial_runs << ','
        << st.sort.merge_passes << ','
        << st.sort.runs_written << ','
        << st.sort.input_bytes << ','
        << st.sort.output_bytes << ','
        << st.checksum
        << '\n';
}

template <size_t Epsilon>
void run_epsilon(
    const cam::storage::KeyFileLayout& data_layout,
    const std::vector<KeyT>& data,
    const QueryWorkload& workload,
    const std::vector<ProbeSpec>& specs,
    const Config& cfg,
    std::ostream& out)
{
    using Index = pgm::PGMIndex<KeyT, Epsilon>;
    Index index(data);
    const size_t index_bytes = index.size_in_bytes();
    RunResult st;
    switch (cfg.mode) {
        case ExecutionMode::HYBRID:
            st = run_hybrid_join(index, data_layout, workload, specs, cfg, index_bytes);
            break;
        case ExecutionMode::POINT:
            st = run_point_join(index, data_layout, workload, cfg, index_bytes);
            break;
        case ExecutionMode::RANGE:
            st = run_single_range_join(index, data_layout, workload, cfg, index_bytes);
            break;
        case ExecutionMode::INLJ:
            st = run_point_join(index, data_layout, workload, cfg, index_bytes);
            break;
    }
    write_row(out, st);
}

void dispatch_epsilon(
    const cam::storage::KeyFileLayout& data_layout,
    const std::vector<KeyT>& data,
    const QueryWorkload& workload,
    const std::vector<ProbeSpec>& specs,
    const Config& cfg,
    std::ostream& out)
{
    switch (cfg.epsilon) {
        case 4: run_epsilon<4>(data_layout, data, workload, specs, cfg, out); break;
        case 8: run_epsilon<8>(data_layout, data, workload, specs, cfg, out); break;
        case 10: run_epsilon<10>(data_layout, data, workload, specs, cfg, out); break;
        case 12: run_epsilon<12>(data_layout, data, workload, specs, cfg, out); break;
        case 14: run_epsilon<14>(data_layout, data, workload, specs, cfg, out); break;
        case 16: run_epsilon<16>(data_layout, data, workload, specs, cfg, out); break;
        case 20: run_epsilon<20>(data_layout, data, workload, specs, cfg, out); break;
        case 24: run_epsilon<24>(data_layout, data, workload, specs, cfg, out); break;
        case 28: run_epsilon<28>(data_layout, data, workload, specs, cfg, out); break;
        case 32: run_epsilon<32>(data_layout, data, workload, specs, cfg, out); break;
        case 48: run_epsilon<48>(data_layout, data, workload, specs, cfg, out); break;
        case 64: run_epsilon<64>(data_layout, data, workload, specs, cfg, out); break;
        case 96: run_epsilon<96>(data_layout, data, workload, specs, cfg, out); break;
        case 128: run_epsilon<128>(data_layout, data, workload, specs, cfg, out); break;
        default:
            throw std::invalid_argument("unsupported epsilon: " + std::to_string(cfg.epsilon));
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Config cfg = parse_args(argc, argv);
        const auto data_layout = cam::storage::detect_key_file_layout(
            cfg.data_path,
            cfg.total_keys,
            cam::storage::HeaderMode::NO);

        auto data = load_data_pgm_safe<KeyT>(cfg.data_path, cfg.total_keys);
        const auto query_layout = cam::storage::detect_key_file_layout(
            cfg.query_path,
            0,
            cam::storage::HeaderMode::NO);
        QueryWorkload workload = prepare_execution_queries(cfg, query_layout);

        std::vector<ProbeSpec> specs;
        if (cfg.mode == ExecutionMode::HYBRID) {
            specs = load_specs(cfg.par_path, cfg.bitmap_path);
            validate_specs(specs, workload.layout.total_keys);
        }

        ensure_parent_dir(cfg.output_path);
        std::ofstream out(
            cfg.output_path,
            cfg.append_output ? (std::ios::out | std::ios::app) : (std::ios::out | std::ios::trunc));
        if (!out) {
            throw std::runtime_error("failed to open output: " + cfg.output_path);
        }
        if (should_write_header(cfg)) {
            write_header(out);
        }

        dispatch_epsilon(data_layout, data, workload, specs, cfg, out);
        cleanup_query_workload(workload);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[error] " << e.what() << '\n';
        return 1;
    }
}
