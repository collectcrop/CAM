#pragma once
#include <list>
#include <mutex>
#include <string>
#include <unordered_map>

#include "CacheInterface.hpp"

class FIFOCache final : public ICache {
public:
    FIFOCache(size_t cap_pages)
      : total_cap_(cap_pages)
    {}

    bool get(size_t pageIndex, Page& out) override {
        std::lock_guard<std::mutex> lg(m_);
        auto it = map_.find(pageIndex);
        if (it == map_.end()) {
            stats_.misses.fetch_add(1, std::memory_order_relaxed);
            return false;
        }
        out = it->second->page;
        stats_.hits.fetch_add(1, std::memory_order_relaxed);
        return true;
    }

    void put(size_t pageIndex, Page&& page) override {
        std::lock_guard<std::mutex> lg(m_);

        if (total_cap_ == 0) {
            stats_.puts.fetch_add(1, std::memory_order_relaxed);
            return;
        }

        auto it = map_.find(pageIndex);
        if (it != map_.end()) {
            it->second->page = std::move(page);
            stats_.puts.fetch_add(1, std::memory_order_relaxed);
            return;
        }

        while (queue_.size() >= total_cap_) {
            auto victim = queue_.begin();
            map_.erase(victim->key);
            queue_.pop_front();
            stats_.evictions.fetch_add(1, std::memory_order_relaxed);
        }

        queue_.push_back(Entry{pageIndex, std::move(page)});
        auto inserted = std::prev(queue_.end());
        map_[pageIndex] = inserted;
        stats_.puts.fetch_add(1, std::memory_order_relaxed);
    }

    void clear() override {
        std::lock_guard<std::mutex> lg(m_);
        map_.clear();
        queue_.clear();
    }

    size_t capacity_pages() const override { return total_cap_; }
    const CacheStats& stats() const override { return stats_; }
    std::string name() const override { return "FIFO"; }

private:
    struct Entry { size_t key; Page page; };

    size_t total_cap_;
    std::mutex m_;
    std::list<Entry> queue_;
    std::unordered_map<size_t, std::list<Entry>::iterator> map_;
    CacheStats stats_;
};
