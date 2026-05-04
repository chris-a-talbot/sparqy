// sparqy/src/sparse_count_map.hpp
//
// Open-addressed hash map keyed on uint32_t mutation IDs, valued by uint32_t
// counts.

#pragma once

#include <algorithm>
#include <cstdint>
#include <vector>

struct SparseCountMap {
    static constexpr uint32_t empty_key = UINT32_MAX;

    std::vector<uint32_t> keys;
    std::vector<uint32_t> values;
    std::vector<uint32_t> used_slots;
    uint32_t mask = 0;
    uint32_t size = 0;

    void reserve(size_t expected);
    void clear();
    void add(uint32_t key);

    template<class Fn>
    void for_each(Fn&& fn) const {
        for (uint32_t slot : used_slots)
            fn(keys[slot], values[slot]);
    }

private:
    void rehash(size_t new_capacity);
};

inline void SparseCountMap::reserve(size_t expected) {
    size_t cap = 64u;
    const size_t padded_expected = std::max<size_t>(expected + expected / 2u + 32u, 64u);
    while (cap < padded_expected) cap <<= 1u;
    if (keys.size() < cap || keys.size() > cap * 4u) rehash(cap);
}

inline void SparseCountMap::clear() {
    for (uint32_t slot : used_slots) {
        keys[slot] = empty_key;
        values[slot] = 0u;
    }
    used_slots.clear();
    size = 0u;
}

inline void SparseCountMap::rehash(size_t new_capacity) {
    std::vector<uint32_t> old_keys = std::move(keys);
    std::vector<uint32_t> old_values = std::move(values);
    std::vector<uint32_t> old_used = std::move(used_slots);

    keys.assign(new_capacity, empty_key);
    values.assign(new_capacity, 0u);
    used_slots.clear();
    used_slots.reserve(old_used.empty() ? new_capacity / 4u : old_used.size());
    mask = (uint32_t)new_capacity - 1u;
    size = 0u;

    for (uint32_t slot : old_used) {
        const uint32_t key = old_keys[slot];
        const uint32_t value = old_values[slot];
        uint32_t probe = key * 2654435761u;
        while (true) {
            const uint32_t idx = probe & mask;
            if (keys[idx] == empty_key) {
                keys[idx] = key;
                values[idx] = value;
                used_slots.push_back(idx);
                ++size;
                break;
            }
            ++probe;
        }
    }
}

inline void SparseCountMap::add(uint32_t key) {
    if (keys.empty() || size * 10u >= keys.size() * 7u)
        rehash(keys.empty() ? 64u : keys.size() * 2u);

    uint32_t probe = key * 2654435761u;
    while (true) {
        const uint32_t idx = probe & mask;
        if (keys[idx] == key) {
            ++values[idx];
            return;
        }
        if (keys[idx] == empty_key) {
            keys[idx] = key;
            values[idx] = 1u;
            used_slots.push_back(idx);
            ++size;
            return;
        }
        ++probe;
    }
}
