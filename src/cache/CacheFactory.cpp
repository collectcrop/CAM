#include "CacheInterface.hpp"
#include "FIFOCache.hpp"
#include "LFUCache.hpp"
#include "LRUCache.hpp"
#include "NullCache.hpp"

#include <algorithm>
#include <iostream>
#include <unistd.h>

std::unique_ptr<ICache>
MakeCache(CachePolicy policy,
          size_t memory_budget_bytes,
          size_t page_size)
{
    const size_t resolved_page_size =
        page_size == 0 ? static_cast<size_t>(std::max(1, ::getpagesize())) : page_size;
    const size_t cap_pages = memory_budget_bytes / resolved_page_size;

    switch (policy) {
        case CachePolicy::FIFO: return std::make_unique<FIFOCache>(cap_pages);
        case CachePolicy::LRU:  return std::make_unique<LRUCache>(cap_pages);
        case CachePolicy::LFU:  return std::make_unique<LFUCache>(cap_pages);
        case CachePolicy::NONE: return std::make_unique<NullCache>();
        default:
            std::cerr << "Invalid cache policy" << std::endl;
            return std::make_unique<NullCache>();
    }
}
