#pragma once

#include <atomic>
#include <cstdint>

#include "../include.hpp"


struct CacheStats {
    std::atomic<uint64_t> hits{0};
    std::atomic<uint64_t> misses{0};
    std::atomic<uint64_t> puts{0};
    std::atomic<uint64_t> evictions{0};
};
