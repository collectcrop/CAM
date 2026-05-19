#pragma once

#include <algorithm>
#include <vector>

#include "../include.hpp"
#include "../storage/DiskManager.hpp"
#include "../storage/KeyFile.hpp"
#include "../storage/PageSearch.hpp"

namespace cam::range_query {

struct RangeQueryMetrics {
    size_t dac = 0;
    size_t buffer_hits = 0;
    size_t cam_io = 0;
    size_t device_ios = 0;
    size_t disk_pages_read = 0;
    uint64_t bytes_read = 0;
    long long io_ns = 0;
};

struct RangeQueryResult {
    RangeQueryMetrics metrics;
    std::vector<Record> records;

    size_t matched() const {
        return records.size();
    }
};

inline RangeQueryMetrics diff_metrics(
    const cam::storage::DiskStats& before,
    const cam::storage::DiskStats& after)
{
    RangeQueryMetrics metrics;
    metrics.dac = after.page_requests - before.page_requests;
    metrics.buffer_hits = after.cache_hits - before.cache_hits;
    metrics.cam_io = after.cache_misses - before.cache_misses;
    metrics.device_ios = after.physical_read_ops - before.physical_read_ops;
    metrics.disk_pages_read = after.logical_page_reads - before.logical_page_reads;
    metrics.bytes_read = after.bytes_read - before.bytes_read;
    metrics.io_ns = after.io_ns - before.io_ns;
    return metrics;
}

namespace detail {

template <typename IndexT>
std::pair<size_t, size_t> estimate_page_window(
    const IndexT& index,
    KeyType lo,
    KeyType hi)
{
    auto [lo_page_lo, lo_page_hi] = index.estimate_pages_for_key(lo);
    auto [hi_page_lo, hi_page_hi] = index.estimate_pages_for_key(hi);

    const size_t page_lo = std::min(lo_page_lo, hi_page_lo);
    const size_t page_hi = std::max(lo_page_hi, hi_page_hi);
    return {page_lo, page_hi};
}

inline std::pair<const Record*, const Record*> page_record_range(
    const Page& page,
    KeyType lo,
    KeyType hi)
{
    if (!page.data || page.valid_len < sizeof(Record)) {
        return {nullptr, nullptr};
    }

    const Record* begin = cam::storage::page_items<Record>(page);
    const Record* end = begin + cam::storage::page_item_count<Record>(page);

    const Record* first = std::lower_bound(
        begin,
        end,
        lo,
        [](const Record& record, KeyType key) {
            return record.key < key;
        });
    const Record* last = std::upper_bound(
        first,
        end,
        hi,
        [](KeyType key, const Record& record) {
            return key < record.key;
        });
    return {first, last};
}

inline void append_records_in_range(
    const std::vector<Page>& pages,
    KeyType lo,
    KeyType hi,
    std::vector<Record>& out)
{
    for (const auto& page : pages) {
        auto [first, last] = page_record_range(page, lo, hi);
        if (first == nullptr || first == last) {
            continue;
        }
        out.insert(out.end(), first, last);
    }
}

inline const std::vector<KeyType>& sorted_query_keys(
    const std::vector<KeyType>& query_keys,
    std::vector<KeyType>& scratch)
{
    if (std::is_sorted(query_keys.begin(), query_keys.end())) {
        return query_keys;
    }

    scratch = query_keys;
    std::sort(scratch.begin(), scratch.end());
    return scratch;
}

inline void append_matching_records_in_range(
    const std::vector<Page>& pages,
    KeyType lo,
    KeyType hi,
    const std::vector<KeyType>& sorted_queries,
    std::vector<Record>& out)
{
    auto query_it = std::lower_bound(sorted_queries.begin(), sorted_queries.end(), lo);
    const auto query_end = std::upper_bound(query_it, sorted_queries.end(), hi);
    if (query_it == query_end) {
        return;
    }

    const auto record_less_key = [](const Record& record, KeyType key) {
        return record.key < key;
    };

    for (const auto& page : pages) {
        auto [record_it, record_end] = page_record_range(page, lo, hi);
        if (record_it == nullptr || record_it == record_end) {
            continue;
        }

        while (record_it != record_end && query_it != query_end) {
            while (query_it != query_end && *query_it < record_it->key) {
                ++query_it;
            }
            if (query_it == query_end) {
                return;
            }

            if (*query_it == record_it->key) {
                out.push_back(*record_it);
                ++record_it;
            } else {
                record_it = std::lower_bound(record_it, record_end, *query_it, record_less_key);
            }
        }
    }
}

} // namespace detail

template <typename IndexT>
RangeQueryResult run_range_query_all_at_once(
    const IndexT& index,
    cam::storage::DiskManager& disk,
    KeyType lo,
    KeyType hi)
{
    RangeQueryResult result;
    if (hi < lo) {
        std::swap(lo, hi);
    }
    if (disk.page_count() == 0) {
        return result;
    }

    const auto [page_lo, page_hi] = detail::estimate_page_window(index, lo, hi);

    const auto before = disk.stats();
    const std::vector<Page> pages = disk.fetch_window(page_lo, page_hi);
    const auto after = disk.stats();
    result.metrics = diff_metrics(before, after);

    detail::append_records_in_range(pages, lo, hi, result.records);
    return result;
}

template <typename IndexT>
RangeQueryResult run_range_query_all_at_once(
    const IndexT& index,
    cam::storage::DiskManager& disk,
    const RangeQ& range)
{
    return run_range_query_all_at_once(index, disk, range.lo, range.hi);
}

template <typename IndexT>
RangeQueryResult run_range_query_all_at_once(
    const IndexT& index,
    cam::storage::DiskManager& disk,
    KeyType lo,
    KeyType hi,
    const std::vector<KeyType>& query_keys)
{
    RangeQueryResult result;
    if (hi < lo) {
        std::swap(lo, hi);
    }
    if (disk.page_count() == 0) {
        return result;
    }

    const auto [page_lo, page_hi] = detail::estimate_page_window(index, lo, hi);

    const auto before = disk.stats();
    const std::vector<Page> pages = disk.fetch_window(page_lo, page_hi);
    const auto after = disk.stats();
    result.metrics = diff_metrics(before, after);

    std::vector<KeyType> sorted_scratch;
    const std::vector<KeyType>& sorted_queries =
        detail::sorted_query_keys(query_keys, sorted_scratch);
    detail::append_matching_records_in_range(
        pages,
        lo,
        hi,
        sorted_queries,
        result.records);
    return result;
}

template <typename IndexT>
RangeQueryResult run_range_query_all_at_once(
    const IndexT& index,
    cam::storage::DiskManager& disk,
    const RangeQ& range,
    const std::vector<KeyType>& query_keys)
{
    return run_range_query_all_at_once(index, disk, range.lo, range.hi, query_keys);
}

template <typename IndexT>
RangeQueryResult run_range_query(
    const IndexT& index,
    cam::storage::DiskManager& disk,
    KeyType lo,
    KeyType hi)
{
    return run_range_query_all_at_once(index, disk, lo, hi);
}

template <typename IndexT>
RangeQueryResult run_range_query(
    const IndexT& index,
    cam::storage::DiskManager& disk,
    const RangeQ& range)
{
    return run_range_query_all_at_once(index, disk, range);
}

template <typename IndexT>
RangeQueryResult run_range_query(
    const IndexT& index,
    cam::storage::DiskManager& disk,
    KeyType lo,
    KeyType hi,
    const std::vector<KeyType>& query_keys)
{
    return run_range_query_all_at_once(index, disk, lo, hi, query_keys);
}

template <typename IndexT>
RangeQueryResult run_range_query(
    const IndexT& index,
    cam::storage::DiskManager& disk,
    const RangeQ& range,
    const std::vector<KeyType>& query_keys)
{
    return run_range_query_all_at_once(index, disk, range, query_keys);
}

} // namespace cam::range_query
