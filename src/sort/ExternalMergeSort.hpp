#pragma once

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <queue>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include "../include.hpp"
#include "../storage/KeyFile.hpp"

namespace cam::sort {

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

constexpr size_t kMinSortBytes = 4 * PAGE_SIZE;
constexpr size_t kKeysPerPage = PAGE_SIZE / sizeof(KeyType);

struct SortStats {
    size_t sort_budget_bytes = 0;
    size_t initial_runs = 0;
    size_t merge_passes = 0;
    uint64_t runs_written = 0;
    uint64_t input_bytes = 0;
    uint64_t output_bytes = 0;
    long long wall_ns = 0;
    cam::storage::KeyFileLayout output_layout;
};

namespace detail {

struct RunFile {
    fs::path path;
    size_t keys = 0;
};

inline void write_run_file(const fs::path& path, const KeyType* data, size_t count) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) {
        throw std::runtime_error("failed to create run file: " + path.string());
    }
    out.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(count * sizeof(KeyType)));
    if (!out) {
        throw std::runtime_error("failed to write run file: " + path.string());
    }
}

inline size_t read_next_chunk(std::ifstream& in, std::vector<KeyType>& buffer, size_t max_keys) {
    buffer.resize(max_keys);
    in.read(reinterpret_cast<char*>(buffer.data()), static_cast<std::streamsize>(max_keys * sizeof(KeyType)));
    const size_t read_keys = static_cast<size_t>(in.gcount()) / sizeof(KeyType);
    buffer.resize(read_keys);
    return read_keys;
}

struct BufferedRunReader {
    explicit BufferedRunReader(const fs::path& path, size_t keys_per_buffer)
        : in(path, std::ios::binary), buffer(keys_per_buffer) {
        if (!in) {
            throw std::runtime_error("failed to open run file: " + path.string());
        }
        refill();
    }

    bool empty() const {
        return exhausted && index >= valid;
    }

    KeyType current() const {
        return buffer[index];
    }

    void pop() {
        ++index;
        if (index >= valid) {
            refill();
        }
    }

private:
    void refill() {
        if (!in.good()) {
            valid = 0;
            index = 0;
            exhausted = true;
            return;
        }

        in.read(reinterpret_cast<char*>(buffer.data()),
                static_cast<std::streamsize>(buffer.size() * sizeof(KeyType)));
        valid = static_cast<size_t>(in.gcount()) / sizeof(KeyType);
        index = 0;
        exhausted = valid == 0;
    }

    std::ifstream in;
    std::vector<KeyType> buffer;
    size_t valid = 0;
    size_t index = 0;
    bool exhausted = false;
};

struct BufferedRunWriter {
    explicit BufferedRunWriter(const fs::path& path, size_t keys_per_buffer)
        : out(path, std::ios::binary | std::ios::trunc), buffer(keys_per_buffer) {
        if (!out) {
            throw std::runtime_error("failed to open output run file: " + path.string());
        }
    }

    void push(KeyType value) {
        buffer[used++] = value;
        if (used == buffer.size()) {
            flush();
        }
    }

    void flush() {
        if (used == 0) {
            return;
        }
        out.write(reinterpret_cast<const char*>(buffer.data()),
                  static_cast<std::streamsize>(used * sizeof(KeyType)));
        if (!out) {
            throw std::runtime_error("failed to flush merged run");
        }
        used = 0;
    }

private:
    std::ofstream out;
    std::vector<KeyType> buffer;
    size_t used = 0;
};

inline RunFile merge_group(
    const std::vector<RunFile>& group,
    const fs::path& out_path,
    size_t keys_per_buffer,
    SortStats& stats)
{
    struct HeapEntry {
        KeyType value = 0;
        size_t reader_idx = 0;

        bool operator>(const HeapEntry& other) const {
            if (value != other.value) {
                return value > other.value;
            }
            return reader_idx > other.reader_idx;
        }
    };

    std::vector<BufferedRunReader> readers;
    readers.reserve(group.size());
    size_t total_keys = 0;
    for (const auto& run : group) {
        readers.emplace_back(run.path, keys_per_buffer);
        total_keys += run.keys;
        stats.input_bytes += run.keys * sizeof(KeyType);
    }

    BufferedRunWriter writer(out_path, keys_per_buffer);
    std::priority_queue<HeapEntry, std::vector<HeapEntry>, std::greater<HeapEntry>> heap;

    for (size_t i = 0; i < readers.size(); ++i) {
        if (!readers[i].empty()) {
            heap.push(HeapEntry{readers[i].current(), i});
        }
    }

    while (!heap.empty()) {
        const HeapEntry entry = heap.top();
        heap.pop();

        writer.push(entry.value);
        readers[entry.reader_idx].pop();
        if (!readers[entry.reader_idx].empty()) {
            heap.push(HeapEntry{readers[entry.reader_idx].current(), entry.reader_idx});
        }
    }

    writer.flush();
    stats.output_bytes += total_keys * sizeof(KeyType);
    ++stats.runs_written;
    return RunFile{out_path, total_keys};
}

} // namespace detail

inline SortStats external_merge_sort(
    const cam::storage::KeyFileLayout& input_layout,
    size_t sort_budget_bytes,
    const fs::path& work_dir)
{
    if (sort_budget_bytes < kMinSortBytes) {
        throw std::invalid_argument(
            "sort budget is too small; require at least " + std::to_string(kMinSortBytes) + " bytes");
    }

    fs::create_directories(work_dir);

    SortStats stats;
    stats.sort_budget_bytes = sort_budget_bytes;

    const auto t0 = Clock::now();
    const size_t chunk_keys = std::max<size_t>(1, sort_budget_bytes / sizeof(KeyType));

    std::ifstream in(input_layout.path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("failed to open input for sorting: " + input_layout.path);
    }
    in.seekg(static_cast<std::streamoff>(input_layout.header_bytes), std::ios::beg);

    std::vector<KeyType> chunk;
    std::vector<detail::RunFile> runs;
    size_t run_id = 0;
    size_t remaining_keys = input_layout.total_keys;

    while (remaining_keys > 0) {
        const size_t want = std::min(chunk_keys, remaining_keys);
        const size_t got = detail::read_next_chunk(in, chunk, want);
        if (got == 0) {
            throw std::runtime_error("unexpected EOF while generating initial runs");
        }
        std::sort(chunk.begin(), chunk.end());
        const fs::path run_path = work_dir / ("run_pass0_" + std::to_string(run_id++) + ".bin");
        detail::write_run_file(run_path, chunk.data(), chunk.size());
        runs.push_back(detail::RunFile{run_path, chunk.size()});
        stats.input_bytes += chunk.size() * sizeof(KeyType);
        stats.output_bytes += chunk.size() * sizeof(KeyType);
        ++stats.runs_written;
        remaining_keys -= chunk.size();
    }
    stats.initial_runs = runs.size();

    const size_t streams_fit = sort_budget_bytes / PAGE_SIZE;
    const size_t max_fanin = streams_fit > 1 ? streams_fit - 1 : 0;
    if (max_fanin < 2 && runs.size() > 1) {
        throw std::invalid_argument(
            "sort budget is too small for merge buffers; need at least 3 pages");
    }

    size_t pass = 1;
    while (runs.size() > 1) {
        ++stats.merge_passes;
        std::vector<detail::RunFile> next_runs;

        for (size_t i = 0; i < runs.size(); i += max_fanin) {
            const size_t end = std::min(runs.size(), i + max_fanin);
            std::vector<detail::RunFile> group(runs.begin() + i, runs.begin() + end);
            const fs::path merged_path =
                work_dir / ("run_pass" + std::to_string(pass) + "_" + std::to_string(i / max_fanin) + ".bin");
            next_runs.push_back(detail::merge_group(group, merged_path, kKeysPerPage, stats));
        }

        for (const auto& run : runs) {
            std::error_code ec;
            fs::remove(run.path, ec);
        }
        runs = std::move(next_runs);
        ++pass;
    }

    const auto t1 = Clock::now();
    stats.wall_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    stats.output_layout = cam::storage::make_layout(
        runs.empty() ? "" : runs.front().path.string(),
        input_layout.total_keys,
        0);
    return stats;
}

} // namespace cam::sort
