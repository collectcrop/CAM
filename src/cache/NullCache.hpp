#pragma once

#include <atomic>
#include <string>

#include "CacheInterface.hpp"

class NullCache final : public ICache {
public:
    NullCache() = default;

    bool get(size_t /*pageIndex*/, Page& /*out*/) override {
        stats_.misses.fetch_add(1, std::memory_order_relaxed);
        return false;
    }

    void put(size_t /*pageIndex*/, Page&& /*page*/) override {
        stats_.puts.fetch_add(1, std::memory_order_relaxed);
    }

    void clear() override {}

    size_t capacity_pages() const override { return 0; }
    const CacheStats& stats() const override { return stats_; }
    std::string name() const override { return "NullCache"; }

private:
    mutable CacheStats stats_{};
};
