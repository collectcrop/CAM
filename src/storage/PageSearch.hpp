#pragma once

#include <algorithm>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include "../include.hpp"

namespace cam::storage {

template <typename T>
inline size_t page_item_count(const Page& page) {
    return page.valid_len / sizeof(T);
}

template <typename T>
inline const T* page_items(const Page& page) {
    return reinterpret_cast<const T*>(page.data.get());
}

template <typename T, typename KeyT, typename Accessor>
inline bool page_linear_contains(const Page& page, const KeyT& key, Accessor accessor) {
    if (!page.data || page.valid_len < sizeof(T)) {
        return false;
    }

    const T* begin = page_items<T>(page);
    const T* end = begin + page_item_count<T>(page);
    return std::find_if(begin, end, [&](const T& item) {
        return accessor(item) == key;
    }) != end;
}

template <typename T, typename KeyT, typename Accessor>
inline bool page_binary_contains(const Page& page, const KeyT& key, Accessor accessor) {
    if (!page.data || page.valid_len < sizeof(T)) {
        return false;
    }

    const T* begin = page_items<T>(page);
    const T* end = begin + page_item_count<T>(page);
    const T* it = std::lower_bound(begin, end, key, [&](const T& item, const KeyT& target) {
        return accessor(item) < target;
    });
    if (it == end) {
        return false;
    }
    return accessor(*it) == key;
}

template <typename T, typename Accessor>
inline auto page_bounds(const Page& page, Accessor accessor) {
    using ResultT = std::decay_t<std::invoke_result_t<Accessor, const T&>>;

    if (!page.data || page.valid_len < sizeof(T)) {
        throw std::runtime_error("cannot inspect empty page");
    }

    const T* begin = page_items<T>(page);
    const size_t count = page_item_count<T>(page);
    return std::pair<ResultT, ResultT>{accessor(begin[0]), accessor(begin[count - 1])};
}

inline bool key_page_linear_contains(const Page& page, KeyType key) {
    return page_linear_contains<KeyType>(page, key, [](KeyType value) { return value; });
}

inline bool key_page_binary_contains(const Page& page, KeyType key) {
    return page_binary_contains<KeyType>(page, key, [](KeyType value) { return value; });
}

inline std::pair<KeyType, KeyType> key_page_bounds(const Page& page) {
    return page_bounds<KeyType>(page, [](KeyType value) { return value; });
}

} // namespace cam::storage
