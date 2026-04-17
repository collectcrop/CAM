#pragma once
#include <cstddef>
#include <memory>
#include <string>
#include "defs.h"

class ICache {
public:
    virtual ~ICache() = default;

    // 命中返回 true，并把页面句柄写入 out（共享底层内存）；未命中返回 false
    virtual bool get(size_t pageIndex, Page& out) = 0;

    // 插入/更新；必要时逐出旧页
    virtual void put(size_t pageIndex, Page&& page) = 0;

    // 清空
    virtual void clear() = 0;

    // 统计与容量
    // virtual size_t size_pages() const = 0;          // 总页数
    virtual size_t capacity_pages() const = 0;      // 总容量（页）
    virtual const CacheStats& stats() const = 0;

    // 诊断：返回实现名称
    virtual std::string name() const = 0;
};

// 只做声明，定义放到 CacheFactory.cpp 里
std::unique_ptr<ICache>
MakeCache(CachePolicy policy,
          size_t memory_budget_bytes,
          size_t page_size);
