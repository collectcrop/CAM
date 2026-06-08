#pragma once

#include <algorithm>
#include <chrono>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "../storage/DiskManager.hpp"
#include "../storage/KeyFile.hpp"
#include "../storage/PageSearch.hpp"

namespace cam::point_query {

struct PointQueryMetrics {
    size_t dac = 0;
    size_t buffer_hits = 0;
    size_t cam_io = 0;
    size_t device_ios = 0;
    size_t disk_pages_read = 0;
    uint64_t bytes_read = 0;
    long long io_ns = 0;
    long long index_traversal_ns = 0;
    long long fetch_wall_ns = 0;
    long long lastmile_search_ns = 0;
};

struct PointQueryResult {
    PointQueryMetrics metrics;
    bool found = false;
    uint64_t matched_key = 0;
};

inline std::string search_strategy_name(SearchStrategy strategy) {
    switch (strategy) {
        case ALL_IN_ONCE: return "all_in_once";
        case ONE_BY_ONE: return "one_by_one";
        default: return "unknown";
    }
}

inline SearchStrategy parse_search_strategy_token(const std::string& value) {
    const std::string token = cam::storage::to_upper(cam::storage::trim(value));
    if (token == "ALL_IN_ONCE" || token == "ALL-AT-ONCE" || token == "ALLATONCE") {
        return ALL_IN_ONCE;
    }
    if (token == "ONE_BY_ONE" || token == "ONE-BY-ONE" || token == "ONEBYONE") {
        return ONE_BY_ONE;
    }
    throw std::invalid_argument("unknown search strategy: " + value);
}

inline std::vector<SearchStrategy> parse_search_strategy_list(const std::string& value) {
    const std::string upper = cam::storage::to_upper(cam::storage::trim(value));
    if (upper == "ALL") {
        return {ALL_IN_ONCE, ONE_BY_ONE};
    }

    std::vector<SearchStrategy> out;
    std::stringstream ss(value);
    std::string token;
    while (std::getline(ss, token, ',')) {
        token = cam::storage::trim(token);
        if (token.empty()) {
            continue;
        }
        const SearchStrategy strategy = parse_search_strategy_token(token);
        if (std::find(out.begin(), out.end(), strategy) == out.end()) {
            out.push_back(strategy);
        }
    }
    if (out.empty()) {
        throw std::invalid_argument("empty search strategy list");
    }
    return out;
}

inline PointQueryMetrics diff_metrics(
    const cam::storage::DiskStats& before,
    const cam::storage::DiskStats& after)
{
    PointQueryMetrics metrics;
    metrics.dac = after.page_requests - before.page_requests;
    metrics.buffer_hits = after.cache_hits - before.cache_hits;
    metrics.cam_io = after.cache_misses - before.cache_misses;
    metrics.device_ios = after.physical_read_ops - before.physical_read_ops;
    metrics.disk_pages_read = after.logical_page_reads - before.logical_page_reads;
    metrics.bytes_read = after.bytes_read - before.bytes_read;
    metrics.io_ns = after.io_ns - before.io_ns;
    return metrics;
}

template <typename TimePointT>
inline long long elapsed_ns(TimePointT begin, TimePointT end) {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
}

template <typename IndexT>
PointQueryResult run_query_all_at_once(
    const IndexT& index,
    cam::storage::DiskManager& disk,
    KeyType key)
{
    PointQueryResult result;
    const auto record_key = [](const Record& record) { return record.key; };

    const auto before = disk.stats();
    auto [page_lo, page_hi] = index.estimate_pages_for_key(key);
    const std::vector<Page> pages = disk.fetch_window(page_lo, page_hi);
    const auto after = disk.stats();
    result.metrics = diff_metrics(before, after);

    for (const auto& page : pages) {
        if (!page.data || page.valid_len < sizeof(Record)) {
            continue;
        }
        const auto [first_key, last_key] = cam::storage::page_bounds<Record>(page, record_key);
        if (key < first_key || key > last_key) {
            continue;
        }
        if (cam::storage::page_binary_contains<Record>(page, key, record_key)) {
            result.found = true;
            result.matched_key = key;
            break;
        }
    }
    return result;
}

template <typename IndexT>
PointQueryResult run_query_one_by_one(
    const IndexT& index,
    cam::storage::DiskManager& disk,
    KeyType key)
{
    PointQueryResult result;
    const auto record_key = [](const Record& record) { return record.key; };

    const auto before = disk.stats();
    auto [page_lo, page_hi] = index.estimate_pages_for_key(key);

    for (size_t page_idx = page_lo; page_idx <= page_hi; ++page_idx) {
        const Page page = disk.fetch(page_idx);
        if (!page.data || page.valid_len < sizeof(Record)) {
            continue;
        }

        const auto [first_key, last_key] = cam::storage::page_bounds<Record>(page, record_key);
        if (key < first_key) {
            break;
        }
        if (key > last_key) {
            if (page_idx == page_hi) {
                break;
            }
            continue;
        }

        result.found = cam::storage::page_binary_contains<Record>(page, key, record_key);
        if (result.found) {
            result.matched_key = key;
        }
        break;
    }

    const auto after = disk.stats();
    result.metrics = diff_metrics(before, after);
    return result;
}

template <typename IndexT>
PointQueryResult run_point_query(
    const IndexT& index,
    cam::storage::DiskManager& disk,
    KeyType key,
    SearchStrategy strategy)
{
    switch (strategy) {
        case ALL_IN_ONCE:
            return run_query_all_at_once(index, disk, key);
        case ONE_BY_ONE:
            return run_query_one_by_one(index, disk, key);
        default:
            throw std::invalid_argument("unsupported search strategy");
    }
}

template <typename IndexT>
PointQueryResult run_query_all_at_once_breakdown(
    const IndexT& index,
    cam::storage::DiskManager& disk,
    KeyType key,
    std::vector<Page>& pages)
{
    PointQueryResult result;
    const auto record_key = [](const Record& record) { return record.key; };

    const auto before = disk.stats();
    const auto index_t0 = std::chrono::steady_clock::now();
    auto [page_lo, page_hi] = index.estimate_pages_for_key(key);
    const auto index_t1 = std::chrono::steady_clock::now();

    const auto fetch_t0 = std::chrono::steady_clock::now();
    disk.fetch_window_into(page_lo, page_hi, pages);
    const auto fetch_t1 = std::chrono::steady_clock::now();
    const auto after = disk.stats();

    result.metrics = diff_metrics(before, after);
    result.metrics.index_traversal_ns = elapsed_ns(index_t0, index_t1);
    result.metrics.fetch_wall_ns = elapsed_ns(fetch_t0, fetch_t1);

    const auto search_t0 = std::chrono::steady_clock::now();
    for (const auto& page : pages) {
        if (!page.data || page.valid_len < sizeof(Record)) {
            continue;
        }
        const auto [first_key, last_key] = cam::storage::page_bounds<Record>(page, record_key);
        if (key < first_key || key > last_key) {
            continue;
        }
        if (cam::storage::page_binary_contains<Record>(page, key, record_key)) {
            result.found = true;
            result.matched_key = key;
            break;
        }
    }
    const auto search_t1 = std::chrono::steady_clock::now();
    result.metrics.lastmile_search_ns = elapsed_ns(search_t0, search_t1);
    return result;
}

template <typename IndexT>
PointQueryResult run_query_all_at_once_breakdown(
    const IndexT& index,
    cam::storage::DiskManager& disk,
    KeyType key)
{
    std::vector<Page> pages;
    return run_query_all_at_once_breakdown(index, disk, key, pages);
}

template <typename IndexT>
PointQueryResult run_query_one_by_one_breakdown(
    const IndexT& index,
    cam::storage::DiskManager& disk,
    KeyType key)
{
    PointQueryResult result;
    const auto record_key = [](const Record& record) { return record.key; };

    const auto before = disk.stats();
    const auto index_t0 = std::chrono::steady_clock::now();
    auto [page_lo, page_hi] = index.estimate_pages_for_key(key);
    const auto index_t1 = std::chrono::steady_clock::now();
    result.metrics.index_traversal_ns = elapsed_ns(index_t0, index_t1);

    for (size_t page_idx = page_lo; page_idx <= page_hi; ++page_idx) {
        const auto fetch_t0 = std::chrono::steady_clock::now();
        const Page page = disk.fetch(page_idx);
        const auto fetch_t1 = std::chrono::steady_clock::now();
        result.metrics.fetch_wall_ns += elapsed_ns(fetch_t0, fetch_t1);

        const auto search_t0 = std::chrono::steady_clock::now();
        if (!page.data || page.valid_len < sizeof(Record)) {
            const auto search_t1 = std::chrono::steady_clock::now();
            result.metrics.lastmile_search_ns += elapsed_ns(search_t0, search_t1);
            continue;
        }

        const auto [first_key, last_key] = cam::storage::page_bounds<Record>(page, record_key);
        if (key < first_key) {
            const auto search_t1 = std::chrono::steady_clock::now();
            result.metrics.lastmile_search_ns += elapsed_ns(search_t0, search_t1);
            break;
        }
        if (key > last_key) {
            const auto search_t1 = std::chrono::steady_clock::now();
            result.metrics.lastmile_search_ns += elapsed_ns(search_t0, search_t1);
            if (page_idx == page_hi) {
                break;
            }
            continue;
        }

        result.found = cam::storage::page_binary_contains<Record>(page, key, record_key);
        if (result.found) {
            result.matched_key = key;
        }
        const auto search_t1 = std::chrono::steady_clock::now();
        result.metrics.lastmile_search_ns += elapsed_ns(search_t0, search_t1);
        break;
    }

    const auto after = disk.stats();
    const auto disk_metrics = diff_metrics(before, after);
    result.metrics.dac = disk_metrics.dac;
    result.metrics.buffer_hits = disk_metrics.buffer_hits;
    result.metrics.cam_io = disk_metrics.cam_io;
    result.metrics.device_ios = disk_metrics.device_ios;
    result.metrics.disk_pages_read = disk_metrics.disk_pages_read;
    result.metrics.bytes_read = disk_metrics.bytes_read;
    result.metrics.io_ns = disk_metrics.io_ns;
    return result;
}

template <typename IndexT>
PointQueryResult run_point_query_breakdown(
    const IndexT& index,
    cam::storage::DiskManager& disk,
    KeyType key,
    SearchStrategy strategy,
    std::vector<Page>& pages)
{
    switch (strategy) {
        case ALL_IN_ONCE:
            return run_query_all_at_once_breakdown(index, disk, key, pages);
        case ONE_BY_ONE:
            return run_query_one_by_one_breakdown(index, disk, key);
        default:
            throw std::invalid_argument("unsupported search strategy");
    }
}

template <typename IndexT>
PointQueryResult run_point_query_breakdown(
    const IndexT& index,
    cam::storage::DiskManager& disk,
    KeyType key,
    SearchStrategy strategy)
{
    std::vector<Page> pages;
    return run_point_query_breakdown(index, disk, key, strategy, pages);
}

} // namespace cam::point_query
