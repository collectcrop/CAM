#pragma once

#include <list>
#include <mutex>
#include <string>
#include <unordered_map>

#include "CacheInterface.hpp"

class LRUCache final : public ICache {
public:
    explicit LRUCache(size_t cap_pages)
      : total_cap_(cap_pages)
    {}

    bool get(size_t pageIndex, Page& out) override {
        std::lock_guard<std::mutex> lg(m_);
        auto it = map_.find(pageIndex);
        if (it == map_.end()) {
            stats_.misses.fetch_add(1, std::memory_order_relaxed);
            return false;
        }

        lru_.erase(it->second.it);
        lru_.push_front(pageIndex);
        it->second.it = lru_.begin();
        out = it->second.page;
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
            it->second.page = std::move(page);
            lru_.erase(it->second.it);
            lru_.push_front(pageIndex);
            it->second.it = lru_.begin();
            stats_.puts.fetch_add(1, std::memory_order_relaxed);
            return;
        }

        if (map_.size() >= total_cap_) {
            const auto victim = lru_.back();
            lru_.pop_back();
            map_.erase(victim);
            stats_.evictions.fetch_add(1, std::memory_order_relaxed);
        }

        lru_.push_front(pageIndex);
        map_.emplace(pageIndex, Entry{std::move(page), lru_.begin()});
        stats_.puts.fetch_add(1, std::memory_order_relaxed);
    }

    void clear() override {
        std::lock_guard<std::mutex> lg(m_);
        lru_.clear();
        map_.clear();
    }

    size_t capacity_pages() const override { return total_cap_; }
    const CacheStats& stats() const override { return stats_; }
    std::string name() const override { return "LRU"; }

private:
    struct Entry {
        Page page;
        std::list<size_t>::iterator it;
    };

    size_t total_cap_;
    std::mutex m_;
    std::list<size_t> lru_;
    std::unordered_map<size_t, Entry> map_;
    CacheStats stats_;
};
