#pragma once

#include <cassert>
#include <cstdint>
#include <limits>
#include <list>
#include <mutex>
#include <string>
#include <unordered_map>

#include "CacheInterface.hpp"

class LFUCache final : public ICache {
public:
    explicit LFUCache(size_t cap_pages)
      : total_cap_(cap_pages)
    {}

    bool get(size_t pageIndex, Page& out) override {
        std::lock_guard<std::mutex> lg(m_);
        auto it = map_.find(pageIndex);
        if (it == map_.end()) {
            stats_.misses.fetch_add(1, std::memory_order_relaxed);
            return false;
        }

        out = it->second.it->page;
        touch_unlocked(it);
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
            it->second.it->page = std::move(page);
            touch_unlocked(it);
            stats_.puts.fetch_add(1, std::memory_order_relaxed);
            return;
        }

        if (map_.size() >= total_cap_) {
            evict_one_unlocked();
        }

        auto& freq_one = freq_lists_[1];
        freq_one.push_front(Node{pageIndex, std::move(page), 1});
        map_[pageIndex] = Location{1, freq_one.begin()};
        min_freq_ = 1;
        stats_.puts.fetch_add(1, std::memory_order_relaxed);
    }

    void clear() override {
        std::lock_guard<std::mutex> lg(m_);
        freq_lists_.clear();
        map_.clear();
        min_freq_ = 0;
    }

    size_t capacity_pages() const override { return total_cap_; }
    const CacheStats& stats() const override { return stats_; }
    std::string name() const override { return "LFU"; }

private:
    struct Node {
        size_t key;
        Page page;
        uint32_t freq;
    };

    struct Location {
        uint32_t freq;
        std::list<Node>::iterator it;
    };

    using MapIterator = std::unordered_map<size_t, Location>::iterator;
    using FreqLists = std::unordered_map<uint32_t, std::list<Node>>;

    void touch_unlocked(MapIterator it) {
        const uint32_t old_freq = it->second.freq;
        auto list_it = freq_lists_.find(old_freq);
        assert(list_it != freq_lists_.end());

        auto node = std::move(*it->second.it);
        list_it->second.erase(it->second.it);
        if (list_it->second.empty()) {
            freq_lists_.erase(list_it);
            if (min_freq_ == old_freq) {
                min_freq_ = old_freq == std::numeric_limits<uint32_t>::max()
                              ? old_freq
                              : old_freq + 1;
            }
        }

        if (node.freq != std::numeric_limits<uint32_t>::max()) {
            ++node.freq;
        }

        auto& new_list = freq_lists_[node.freq];
        new_list.push_front(std::move(node));
        it->second = Location{new_list.front().freq, new_list.begin()};
    }

    FreqLists::iterator find_min_list_unlocked() {
        auto it = freq_lists_.find(min_freq_);
        if (it != freq_lists_.end() && !it->second.empty()) {
            return it;
        }

        auto best = freq_lists_.end();
        for (auto cur = freq_lists_.begin(); cur != freq_lists_.end(); ++cur) {
            if (cur->second.empty()) {
                continue;
            }
            if (best == freq_lists_.end() || cur->first < best->first) {
                best = cur;
            }
        }
        if (best != freq_lists_.end()) {
            min_freq_ = best->first;
        }
        return best;
    }

    void evict_one_unlocked() {
        auto list_it = find_min_list_unlocked();
        if (list_it == freq_lists_.end()) {
            return;
        }

        auto& nodes = list_it->second;
        auto victim_it = std::prev(nodes.end());
        const size_t victim_key = victim_it->key;
        nodes.erase(victim_it);
        map_.erase(victim_key);
        if (nodes.empty()) {
            freq_lists_.erase(list_it);
        }
        stats_.evictions.fetch_add(1, std::memory_order_relaxed);
    }

    size_t total_cap_;
    std::mutex m_;
    FreqLists freq_lists_;
    std::unordered_map<size_t, Location> map_;
    uint32_t min_freq_{0};
    CacheStats stats_;
};
