#pragma once

#include <algorithm>
#include <chrono>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <fcntl.h>
#include <unistd.h>

#include "../cache/CacheInterface.hpp"
#include "KeyFile.hpp"

namespace cam::storage {

struct DiskStats {
    uint64_t page_requests = 0;
    uint64_t cache_hits = 0;
    uint64_t cache_misses = 0;
    uint64_t logical_page_reads = 0;
    uint64_t physical_read_ops = 0;
    uint64_t bytes_read = 0;
    long long io_ns = 0;
    long long cache_ns = 0;
    bool useDirect = false;
};

inline size_t round_up(size_t value, size_t unit) {
    return ((value + unit - 1) / unit) * unit;
}

class DiskManager {
public:
    DiskManager(const KeyFileLayout& layout, std::unique_ptr<ICache> cache, bool use_direct = false)
        : layout_(layout), cache_(std::move(cache)) {
        stats_.useDirect = use_direct;
        if (stats_.useDirect) fd_ = ::open(layout.path.c_str(), O_RDONLY | O_DIRECT);
        else fd_ = ::open(layout.path.c_str(), O_RDONLY);
        
        if (!cache_) {
            throw std::invalid_argument("DiskManager requires a cache instance");
        }
        no_cache_ = cache_->capacity_pages() == 0;
        if (fd_ < 0) {
            throw std::runtime_error(std::string("open failed: ") + std::strerror(errno));
        }
    }

    ~DiskManager() {
        if (fd_ >= 0) {
            ::close(fd_);
        }
    }

    DiskManager(const DiskManager&) = delete;
    DiskManager& operator=(const DiskManager&) = delete;

    DiskManager(DiskManager&&) = delete;
    DiskManager& operator=(DiskManager&&) = delete;

    size_t page_count() const {
        return layout_.logical_pages;
    }

    const KeyFileLayout& layout() const {
        return layout_;
    }

    const DiskStats& stats() const {
        return stats_;
    }

    Page fetch(size_t page_idx) {
        auto pages = fetch_window(page_idx, page_idx);
        return pages.empty() ? Page{} : std::move(pages.front());
    }

    std::vector<Page> fetch_window(size_t page_lo, size_t page_hi) {
        std::vector<Page> pages;
        fetch_window_into(page_lo, page_hi, pages);
        return pages;
    }

    void fetch_window_into(size_t page_lo, size_t page_hi, std::vector<Page>& pages) {
        pages.clear();
        if (page_lo > page_hi || page_lo >= layout_.logical_pages) {
            return;
        }

        page_hi = std::min(page_hi, layout_.logical_pages - 1);
        const size_t num_pages = page_hi - page_lo + 1;

        if (no_cache_) {
            stats_.page_requests += num_pages;
            stats_.cache_misses += num_pages;
            read_page_run_views_into(page_lo, num_pages, pages);
            return;
        }

        pages.resize(num_pages);

        size_t run_start = 0;
        size_t run_len = 0;
        auto flush_run = [&]() {
            if (run_len == 0) {
                return;
            }

            auto fetched = read_page_run(page_lo + run_start, run_len);
            for (size_t i = 0; i < fetched.size(); ++i) {
                const size_t slot = run_start + i;
                const size_t page_idx = page_lo + slot;
                pages[slot] = fetched[i];
                const auto cache_t0 = std::chrono::steady_clock::now();
                cache_->put(page_idx, Page{pages[slot].data, pages[slot].valid_len});
                const auto cache_t1 = std::chrono::steady_clock::now();
                stats_.cache_ns +=
                    std::chrono::duration_cast<std::chrono::nanoseconds>(cache_t1 - cache_t0).count();
            }
            run_len = 0;
        };

        for (size_t i = 0; i < num_pages; ++i) {
            const size_t page_idx = page_lo + i;
            ++stats_.page_requests;

            Page cached;
            const auto cache_t0 = std::chrono::steady_clock::now();
            const bool hit = cache_->get(page_idx, cached);
            const auto cache_t1 = std::chrono::steady_clock::now();
            stats_.cache_ns +=
                std::chrono::duration_cast<std::chrono::nanoseconds>(cache_t1 - cache_t0).count();
            if (hit) {
                ++stats_.cache_hits;
                flush_run();
                pages[i] = std::move(cached);
            } else {
                ++stats_.cache_misses;
                if (run_len == 0) {
                    run_start = i;
                }
                ++run_len;
            }
        }
        flush_run();
    }

private:
    static Page alloc_page(size_t valid_len) {
        void* raw = nullptr;
        if (posix_memalign(&raw, PAGE_SIZE, PAGE_SIZE) != 0) {
            throw std::runtime_error("posix_memalign failed");
        }

        Page page;
        page.data.reset(reinterpret_cast<char*>(raw), [](char* p) { free(p); });
        page.valid_len = valid_len;
        return page;
    }

    Page alloc_and_copy_page(const char* src, size_t len) const {
        Page page = alloc_page(len);
        std::memcpy(page.data.get(), src, len);
        return page;
    }

    static Page page_view(const std::shared_ptr<char[]>& owner, size_t offset, size_t len) {
        Page page;
        page.data = std::shared_ptr<char[]>(owner, owner.get() + offset);
        page.valid_len = len;
        return page;
    }

    void read_page_run_views_into(
        size_t start_page,
        size_t page_count,
        std::vector<Page>& pages) {
        pages.clear();
        if (page_count == 0 || start_page >= layout_.logical_pages) {
            return;
        }

        const size_t logical_start = layout_.header_bytes + start_page * PAGE_SIZE;
        const size_t logical_bytes_remaining = layout_.logical_bytes() - start_page * PAGE_SIZE;
        const size_t logical_bytes = std::min(page_count * PAGE_SIZE, logical_bytes_remaining);
        const size_t aligned_start = (logical_start / PAGE_SIZE) * PAGE_SIZE;
        const size_t leading_skip = logical_start - aligned_start;
        const size_t aligned_read = round_up(leading_skip + logical_bytes, PAGE_SIZE);

        void* raw = nullptr;
        if (posix_memalign(&raw, PAGE_SIZE, aligned_read) != 0) {
            throw std::runtime_error("posix_memalign failed");
        }
        std::shared_ptr<char[]> buf(reinterpret_cast<char*>(raw), [](char* p) { free(p); });

        const auto t0 = std::chrono::steady_clock::now();
        const ssize_t br = ::pread(fd_, buf.get(), aligned_read, static_cast<off_t>(aligned_start));
        const auto t1 = std::chrono::steady_clock::now();
        if (br < 0) {
            throw std::runtime_error(std::string("pread failed: ") + std::strerror(errno));
        }
        if (static_cast<size_t>(br) < leading_skip + logical_bytes) {
            throw std::runtime_error("short read while fetching logical page run");
        }

        ++stats_.physical_read_ops;
        const size_t pages_read = (logical_bytes + PAGE_SIZE - 1) / PAGE_SIZE;
        stats_.logical_page_reads += pages_read;
        stats_.bytes_read += static_cast<uint64_t>(br);
        stats_.io_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();

        pages.reserve(pages_read);
        for (size_t i = 0; i < pages_read; ++i) {
            const size_t offset = i * PAGE_SIZE;
            const size_t remain =
                logical_bytes > offset ? logical_bytes - offset : 0;
            const size_t len = std::min(PAGE_SIZE, remain);
            pages.push_back(page_view(buf, leading_skip + offset, len));
        }
    }

    std::vector<Page> read_page_run(size_t start_page, size_t page_count) {
        if (page_count == 0 || start_page >= layout_.logical_pages) {
            return {};
        }

        const size_t logical_start = layout_.header_bytes + start_page * PAGE_SIZE;
        const size_t logical_bytes_remaining = layout_.logical_bytes() - start_page * PAGE_SIZE;
        const size_t logical_bytes = std::min(page_count * PAGE_SIZE, logical_bytes_remaining);
        const size_t aligned_start = (logical_start / PAGE_SIZE) * PAGE_SIZE;
        const size_t leading_skip = logical_start - aligned_start;
        const size_t aligned_read = round_up(leading_skip + logical_bytes, PAGE_SIZE);

        void* raw = nullptr;
        if (posix_memalign(&raw, PAGE_SIZE, aligned_read) != 0) {
            throw std::runtime_error("posix_memalign failed");
        }
        std::unique_ptr<char, void(*)(void*)> buf(reinterpret_cast<char*>(raw), free);
        const auto t0 = std::chrono::steady_clock::now();
        const ssize_t br = ::pread(fd_, buf.get(), aligned_read, static_cast<off_t>(aligned_start));
        const auto t1 = std::chrono::steady_clock::now();
        if (br < 0) {
            throw std::runtime_error(std::string("pread failed: ") + std::strerror(errno));
        }
        if (static_cast<size_t>(br) < leading_skip + logical_bytes) {
            throw std::runtime_error("short read while fetching logical page run");
        }

        ++stats_.physical_read_ops;
        const size_t pages_read = (logical_bytes + PAGE_SIZE - 1) / PAGE_SIZE;
        stats_.logical_page_reads += pages_read;
        stats_.bytes_read += static_cast<uint64_t>(br);
        stats_.io_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();

        std::vector<Page> pages;
        pages.reserve(pages_read);
        for (size_t i = 0; i < pages_read; ++i) {
            const size_t offset = i * PAGE_SIZE;
            const size_t remain =
                logical_bytes > offset ? logical_bytes - offset : 0;
            const size_t len = std::min(PAGE_SIZE, remain);
            pages.push_back(alloc_and_copy_page(buf.get() + leading_skip + offset, len));
        }
        return pages;
    }

    Page read_page(size_t page_idx) {
        if (page_idx >= layout_.logical_pages) {
            return {};
        }

        const size_t logical_start = layout_.header_bytes + page_idx * PAGE_SIZE;
        const size_t remaining = layout_.logical_bytes() - page_idx * PAGE_SIZE;
        const size_t logical_len = std::min(PAGE_SIZE, remaining);
        const size_t aligned_start = (logical_start / PAGE_SIZE) * PAGE_SIZE;
        const size_t leading_skip = logical_start - aligned_start;
        const size_t aligned_read = round_up(leading_skip + logical_len, PAGE_SIZE);

        void* raw = nullptr;
        if (posix_memalign(&raw, PAGE_SIZE, aligned_read) != 0) {
            throw std::runtime_error("posix_memalign failed");
        }
        std::unique_ptr<char, void(*)(void*)> buf(reinterpret_cast<char*>(raw), free);
        const auto t0 = std::chrono::steady_clock::now();
        const ssize_t br = ::pread(fd_, buf.get(), aligned_read, static_cast<off_t>(aligned_start));
        const auto t1 = std::chrono::steady_clock::now();
        if (br < 0) {
            throw std::runtime_error(std::string("pread failed: ") + std::strerror(errno));
        }
        if (static_cast<size_t>(br) < leading_skip + logical_len) {
            throw std::runtime_error("short read while fetching logical page");
        }

        ++stats_.physical_read_ops;
        ++stats_.logical_page_reads;
        stats_.bytes_read += static_cast<uint64_t>(br);
        stats_.io_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();

        return alloc_and_copy_page(buf.get() + leading_skip, logical_len);
    }

    KeyFileLayout layout_;
    std::unique_ptr<ICache> cache_;
    int fd_ = -1;
    bool no_cache_ = false;
    DiskStats stats_;
};

inline std::unique_ptr<ICache> make_page_cache(CachePolicy policy, size_t cache_bytes) {
    return MakeCache(policy, cache_bytes, PAGE_SIZE);
}

} // namespace cam::storage
