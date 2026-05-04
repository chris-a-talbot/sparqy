// sparqy/src/alias_sampler.hpp
//
// Walker/Vose alias method for O(1) weighted sampling from a discrete
// distribution. 

#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <numeric>
#include <vector>

#include <omp.h>

#include "rng.hpp"

inline size_t alias_sampler_chunk_begin(size_t size, int part, int parts) {
    return (size * (size_t)part) / (size_t)parts;
}

constexpr double kAliasSamplerBucketEpsilon = 1e-12;

inline void alias_sampler_parallel_prefix_sum(
    const std::vector<int>& indices,
    const std::vector<double>& scaled_weights,
    std::vector<double>& prefix,
    int thread_count,
    std::vector<double>& thread_sums) {
    prefix.resize(indices.size() + 1u);
    prefix[0] = 0.0;
    if (indices.empty()) return;

    const int parts = std::min<int>(thread_count, (int)indices.size());
    thread_sums.assign((size_t)parts + 1u, 0.0);

    #pragma omp parallel num_threads(parts)
    {
        const int thread_index = omp_get_thread_num();
        const size_t begin = alias_sampler_chunk_begin(indices.size(), thread_index, parts);
        const size_t end = alias_sampler_chunk_begin(indices.size(), thread_index + 1, parts);

        double local_sum = 0.0;
        for (size_t i = begin; i < end; ++i) {
            local_sum += scaled_weights[(size_t)indices[i]];
            prefix[i + 1u] = local_sum;
        }
        thread_sums[(size_t)thread_index + 1u] = local_sum;

        #pragma omp barrier
        #pragma omp single
        {
            for (int i = 1; i <= parts; ++i) {
                thread_sums[(size_t)i] += thread_sums[(size_t)i - 1u];
            }
        }

        const double offset = thread_sums[(size_t)thread_index];
        for (size_t i = begin; i < end; ++i) {
            prefix[i + 1u] += offset;
        }
    }
}

class AliasSampler {
public:
    // Build the alias table from the given weights - O(n)
    void build(const std::vector<double>& weights);
    void build_parallel(const std::vector<double>& weights, int thread_count);
    // PSA+ (Hübschle-Schneider & Sanders, §4.2.1): greedy local sweep per
    // thread, then PSA on the leftovers. Result is equivalent to
    // build_parallel but generally faster on inputs with a usable mix of
    // light and heavy items inside each thread's input chunk.
    void build_parallel_psa_plus(const std::vector<double>& weights, int thread_count);

    // Sample one index - O(1)
    int  sample(RNG& rng) const;

private:
    struct SplitState {
        int light_count = 0;
        int heavy_count = 0;
        double spill = 0.0;
    };

    int n_ = 0;
    std::vector<double> prob_;
    std::vector<int>    alias_;
    bool uniform_ = false;

    // Builder scratch
    std::vector<double> norm_scratch_;
    std::vector<int>    small_scratch_;
    std::vector<int>    large_scratch_;

    std::vector<int>    light_items_scratch_;
    std::vector<int>    heavy_items_scratch_;
    std::vector<double> light_prefix_scratch_;
    std::vector<double> heavy_prefix_scratch_;
    std::vector<int>    thread_light_offsets_scratch_;
    std::vector<int>    thread_heavy_offsets_scratch_;
    std::vector<double> thread_prefix_offsets_scratch_;
    std::vector<SplitState> split_scratch_;

    // PSA+ scratch
    std::vector<uint8_t> psa_plus_filled_scratch_;
    std::vector<int>     psa_plus_in_flight_heavy_;
    std::vector<double>  psa_plus_in_flight_residual_scaled_;
};

inline void AliasSampler::build(const std::vector<double>& weights) {
    n_ = (int)weights.size();
    prob_.resize(n_);
    alias_.resize(n_);

    double sum = 0.0;
    for (double w : weights) sum += w;
    if (sum <= 0.0) {
        uniform_ = true;
        return;
    }

    double mn = weights[0], mx = weights[0];
    for (double w : weights) {
        if (w < mn) mn = w;
        if (w > mx) mx = w;
    }
    if (mn == mx) {
        uniform_ = true;
        return;
    }
    uniform_ = false;

    norm_scratch_.resize(n_);
    small_scratch_.clear();
    large_scratch_.clear();

    for (int i = 0; i < n_; i++) norm_scratch_[i] = weights[i] * n_ / sum;

    for (int i = 0; i < n_; i++)
        (norm_scratch_[i] < 1.0 ? small_scratch_ : large_scratch_).push_back(i);

    while (!small_scratch_.empty() && !large_scratch_.empty()) {
        int l = small_scratch_.back();
        small_scratch_.pop_back();
        int g = large_scratch_.back();
        large_scratch_.pop_back();
        prob_[l] = norm_scratch_[l];
        alias_[l] = g;
        norm_scratch_[g] = (norm_scratch_[g] + norm_scratch_[l]) - 1.0;
        (norm_scratch_[g] < 1.0 ? small_scratch_ : large_scratch_).push_back(g);
    }

    for (int i : large_scratch_) prob_[i] = 1.0;
    for (int i : small_scratch_) prob_[i] = 1.0;
}

inline void AliasSampler::build_parallel(const std::vector<double>& weights,
                                         int thread_count) {
    n_ = (int)weights.size();
    prob_.resize((size_t)n_);
    alias_.resize((size_t)n_);
    std::fill(prob_.begin(), prob_.end(), 1.0);
    std::iota(alias_.begin(), alias_.end(), 0);

    if (n_ == 0) {
        uniform_ = true;
        return;
    }
    if (thread_count <= 1 || n_ == 1) {
        build(weights);
        return;
    }

    const int parts = std::min(thread_count, n_);
    double sum = 0.0;
    double mn = std::numeric_limits<double>::max();
    double mx = std::numeric_limits<double>::lowest();

    #pragma omp parallel for reduction(+:sum) reduction(min:mn) reduction(max:mx) num_threads(parts)
    for (int i = 0; i < n_; ++i) {
        const double w = weights[(size_t)i];
        sum += w;
        if (w < mn) mn = w;
        if (w > mx) mx = w;
    }

    if (sum <= 0.0) {
        uniform_ = true;
        return;
    }
    if (mn == mx) {
        uniform_ = true;
        return;
    }
    uniform_ = false;

    norm_scratch_.resize((size_t)n_);
    thread_light_offsets_scratch_.assign((size_t)parts + 1u, 0);
    thread_heavy_offsets_scratch_.assign((size_t)parts + 1u, 0);

    const double scale = (double)n_ / sum;
    #pragma omp parallel num_threads(parts)
    {
        const int thread_index = omp_get_thread_num();
        const size_t begin = alias_sampler_chunk_begin((size_t)n_, thread_index, parts);
        const size_t end = alias_sampler_chunk_begin((size_t)n_, thread_index + 1, parts);

        int local_light_count = 0;
        int local_heavy_count = 0;
        for (size_t i = begin; i < end; ++i) {
            const double scaled = weights[i] * scale;
            norm_scratch_[i] = scaled;
            if (scaled < 1.0) ++local_light_count;
            else if (scaled > 1.0) ++local_heavy_count;
        }
        thread_light_offsets_scratch_[(size_t)thread_index + 1u] = local_light_count;
        thread_heavy_offsets_scratch_[(size_t)thread_index + 1u] = local_heavy_count;
    }

    for (int i = 1; i <= parts; ++i) {
        thread_light_offsets_scratch_[(size_t)i] += thread_light_offsets_scratch_[(size_t)i - 1u];
        thread_heavy_offsets_scratch_[(size_t)i] += thread_heavy_offsets_scratch_[(size_t)i - 1u];
    }

    const int light_count = thread_light_offsets_scratch_[(size_t)parts];
    const int heavy_count = thread_heavy_offsets_scratch_[(size_t)parts];
    const int strict_item_count = light_count + heavy_count;
    if (strict_item_count == 0) {
        // Every scaled weight equaled 1.0 exactly. The pre-fill above set
        // prob_=1.0 / alias_=i so sample() works correctly anyway, but flag
        // it so sample() takes the cheap uniform path and so we don't inherit
        // a stale uniform_ value from a previous build.
        uniform_ = true;
        return;
    }
    if (light_count == 0 || heavy_count == 0) {
        build(weights);
        return;
    }

    light_items_scratch_.resize((size_t)light_count);
    heavy_items_scratch_.resize((size_t)heavy_count);

    #pragma omp parallel num_threads(parts)
    {
        const int thread_index = omp_get_thread_num();
        const size_t begin = alias_sampler_chunk_begin((size_t)n_, thread_index, parts);
        const size_t end = alias_sampler_chunk_begin((size_t)n_, thread_index + 1, parts);

        int light_write = thread_light_offsets_scratch_[(size_t)thread_index];
        int heavy_write = thread_heavy_offsets_scratch_[(size_t)thread_index];
        for (size_t i = begin; i < end; ++i) {
            const double scaled = norm_scratch_[i];
            if (scaled < 1.0) {
                light_items_scratch_[(size_t)light_write++] = (int)i;
            } else if (scaled > 1.0) {
                heavy_items_scratch_[(size_t)heavy_write++] = (int)i;
            }
        }
    }

    alias_sampler_parallel_prefix_sum(light_items_scratch_,
                                      norm_scratch_,
                                      light_prefix_scratch_,
                                      parts,
                                      thread_prefix_offsets_scratch_);
    alias_sampler_parallel_prefix_sum(heavy_items_scratch_,
                                      norm_scratch_,
                                      heavy_prefix_scratch_,
                                      parts,
                                      thread_prefix_offsets_scratch_);

    const int active_parts = std::min(parts, strict_item_count);
    split_scratch_.assign((size_t)active_parts, {});
    split_scratch_[0].light_count = 0;
    split_scratch_[0].heavy_count = 0;
    split_scratch_[0].spill =
        norm_scratch_[(size_t)heavy_items_scratch_[0]];

    auto split_boundary = [&](int target_bucket_count) {
        SplitState split;
        const int search_low = std::max(0, target_bucket_count - light_count);
        const int search_high = std::min(target_bucket_count, heavy_count - 1);
        int low = search_low;
        int high = search_high;
        int best_heavy = search_low;

        while (low <= high) {
            const int used_heavy = (low + high) / 2;
            const int used_light = target_bucket_count - used_heavy;
            const double sigma =
                light_prefix_scratch_[(size_t)used_light]
                + heavy_prefix_scratch_[(size_t)used_heavy];
            // The sweeping order packs a heavy bucket as soon as the current
            // residual drops to one bucket or less, so among all feasible
            // splits we want the maximum number of already-completed heavy
            // buckets on the left.
            if (sigma <= (double)target_bucket_count + kAliasSamplerBucketEpsilon) {
                best_heavy = used_heavy;
                low = used_heavy + 1;
            } else {
                high = used_heavy - 1;
            }
        }

        const int used_heavy = best_heavy;
        const int used_light = target_bucket_count - used_heavy;
        const double sigma =
            light_prefix_scratch_[(size_t)used_light]
            + heavy_prefix_scratch_[(size_t)used_heavy];
        const double current_heavy_weight =
            norm_scratch_[(size_t)heavy_items_scratch_[(size_t)used_heavy]];
        split.light_count = used_light;
        split.heavy_count = used_heavy;
        split.spill = std::max(
            0.0,
            current_heavy_weight + sigma - (double)target_bucket_count);
        return split;
    };

    #pragma omp parallel for schedule(static) num_threads(active_parts)
    for (int part = 1; part < active_parts; ++part) {
        const int target_bucket_count =
            (strict_item_count * part) / active_parts;
        split_scratch_[(size_t)part] = split_boundary(target_bucket_count);
    }

    #pragma omp parallel for schedule(static) num_threads(active_parts)
    for (int part = 0; part < active_parts; ++part) {
        const int bucket_begin = (strict_item_count * part) / active_parts;
        const int bucket_end = (strict_item_count * (part + 1)) / active_parts;
        const int bucket_count = bucket_end - bucket_begin;
        if (bucket_count <= 0) continue;

        int light_index = split_scratch_[(size_t)part].light_count;
        int heavy_index = split_scratch_[(size_t)part].heavy_count;
        double residual = split_scratch_[(size_t)part].spill;

        for (int bucket = 0; bucket < bucket_count; ++bucket) {
            const bool pack_heavy =
                (light_index >= light_count)
                || (residual <= 1.0 + kAliasSamplerBucketEpsilon);
            if (pack_heavy) {
                const int heavy_item = heavy_items_scratch_[(size_t)heavy_index];
                if (heavy_index + 1 < heavy_count) {
                    prob_[(size_t)heavy_item] =
                        std::min(1.0, std::max(0.0, residual));
                    alias_[(size_t)heavy_item] =
                        heavy_items_scratch_[(size_t)heavy_index + 1u];
                    ++heavy_index;
                    residual =
                        residual
                        + norm_scratch_[(size_t)heavy_items_scratch_[(size_t)heavy_index]]
                        - 1.0;
                } else {
                    prob_[(size_t)heavy_item] = 1.0;
                    alias_[(size_t)heavy_item] = heavy_item;
                    ++heavy_index;
                    residual = 0.0;
                }
            } else {
                const int light_item = light_items_scratch_[(size_t)light_index++];
                prob_[(size_t)light_item] = norm_scratch_[(size_t)light_item];
                alias_[(size_t)light_item] =
                    heavy_items_scratch_[(size_t)heavy_index];
                residual += norm_scratch_[(size_t)light_item] - 1.0;
            }
        }
    }
}

inline void AliasSampler::build_parallel_psa_plus(
    const std::vector<double>& weights, int thread_count) {
    n_ = (int)weights.size();
    prob_.resize((size_t)n_);
    alias_.resize((size_t)n_);
    std::fill(prob_.begin(), prob_.end(), 1.0);
    std::iota(alias_.begin(), alias_.end(), 0);

    if (n_ == 0) { uniform_ = true; return; }
    if (thread_count <= 1 || n_ == 1) { build(weights); return; }

    const int parts = std::min(thread_count, n_);

    double sum = 0.0;
    double mn = std::numeric_limits<double>::max();
    double mx = std::numeric_limits<double>::lowest();
    #pragma omp parallel for reduction(+:sum) reduction(min:mn) reduction(max:mx) num_threads(parts)
    for (int i = 0; i < n_; ++i) {
        const double w = weights[(size_t)i];
        sum += w;
        if (w < mn) mn = w;
        if (w > mx) mx = w;
    }

    if (sum <= 0.0) { uniform_ = true; return; }
    if (mn == mx)   { uniform_ = true; return; }
    uniform_ = false;

    const double scale = (double)n_ / sum;

    // ---- Phase 1: per-thread greedy local sweep (PSA+ pre-pass) ----
    // For each thread's input chunk, run the sequential sweeping algorithm
    // (Algorithm 2) using only items in that chunk. Each pack writes prob_
    // and alias_ for the filled bucket and marks the item filled. When the
    // sweep can no longer advance (next light or heavy out of chunk), the
    // current in-flight heavy keeps its residual weight; PSA processes it
    // afterwards using that residual instead of its original weight.
    psa_plus_filled_scratch_.assign((size_t)n_, 0);
    psa_plus_in_flight_heavy_.assign((size_t)parts, -1);
    psa_plus_in_flight_residual_scaled_.assign((size_t)parts, 0.0);

    #pragma omp parallel num_threads(parts)
    {
        const int t = omp_get_thread_num();
        const size_t a = alias_sampler_chunk_begin((size_t)n_, t, parts);
        const size_t b = alias_sampler_chunk_begin((size_t)n_, t + 1, parts);

        // Light = scaled <= 1.0 (matches PSA+ convention; balanced items
        // are treated as light because they pack with prob_=1.0 and don't
        // disturb the residual).
        auto is_heavy = [&](size_t k) { return weights[k] * scale > 1.0; };

        size_t i_idx = a;
        while (i_idx < b && is_heavy(i_idx)) ++i_idx;
        size_t j_idx = a;
        while (j_idx < b && !is_heavy(j_idx)) ++j_idx;
        if (i_idx >= b || j_idx >= b) {
            // No usable light/heavy pair locally; nothing to greedy-pack.
        } else {
            double w_scaled = weights[j_idx] * scale; // residual of in-flight heavy

            for (;;) {
                if (w_scaled > 1.0) {
                    // Pack light bucket i_idx with alias = j_idx.
                    const double light_scaled = weights[i_idx] * scale;
                    prob_[i_idx]  = light_scaled;
                    alias_[i_idx] = (int)j_idx;
                    psa_plus_filled_scratch_[i_idx] = 1;
                    w_scaled = w_scaled + light_scaled - 1.0;
                    // Advance i_idx to next light in chunk.
                    size_t k = i_idx + 1;
                    while (k < b && is_heavy(k)) ++k;
                    if (k >= b) break;
                    i_idx = k;
                } else {
                    // Pack heavy bucket j_idx with alias = next heavy.
                    size_t k = j_idx + 1;
                    while (k < b && !is_heavy(k)) ++k;
                    if (k >= b) break; // no next heavy in chunk; leave j_idx in flight
                    const double clamped = std::min(1.0, std::max(0.0, w_scaled));
                    prob_[j_idx]  = clamped;
                    alias_[j_idx] = (int)k;
                    psa_plus_filled_scratch_[j_idx] = 1;
                    w_scaled = w_scaled + weights[k] * scale - 1.0;
                    j_idx = k;
                }
            }

            psa_plus_in_flight_heavy_[(size_t)t] = (int)j_idx;
            psa_plus_in_flight_residual_scaled_[(size_t)t] = w_scaled;
        }
    }

    // ---- Phase 2: classify + count over remaining items ----
    norm_scratch_.resize((size_t)n_);
    thread_light_offsets_scratch_.assign((size_t)parts + 1u, 0);
    thread_heavy_offsets_scratch_.assign((size_t)parts + 1u, 0);

    #pragma omp parallel num_threads(parts)
    {
        const int t = omp_get_thread_num();
        const size_t a = alias_sampler_chunk_begin((size_t)n_, t, parts);
        const size_t b = alias_sampler_chunk_begin((size_t)n_, t + 1, parts);
        const int    in_flight_idx = psa_plus_in_flight_heavy_[(size_t)t];
        const double in_flight_scaled = psa_plus_in_flight_residual_scaled_[(size_t)t];

        int local_light = 0, local_heavy = 0;
        for (size_t k = a; k < b; ++k) {
            if (psa_plus_filled_scratch_[k]) continue;
            const double scaled = ((int)k == in_flight_idx)
                                ? in_flight_scaled
                                : weights[k] * scale;
            norm_scratch_[k] = scaled;
            if      (scaled < 1.0) ++local_light;
            else if (scaled > 1.0) ++local_heavy;
        }
        thread_light_offsets_scratch_[(size_t)t + 1u] = local_light;
        thread_heavy_offsets_scratch_[(size_t)t + 1u] = local_heavy;
    }

    for (int i = 1; i <= parts; ++i) {
        thread_light_offsets_scratch_[(size_t)i] += thread_light_offsets_scratch_[(size_t)i - 1u];
        thread_heavy_offsets_scratch_[(size_t)i] += thread_heavy_offsets_scratch_[(size_t)i - 1u];
    }

    const int light_count = thread_light_offsets_scratch_[(size_t)parts];
    const int heavy_count = thread_heavy_offsets_scratch_[(size_t)parts];
    const int strict_item_count = light_count + heavy_count;

    if (strict_item_count == 0) return; // greedy filled all strict items
    if (light_count == 0 || heavy_count == 0) {
        // Floating-point edge: should not happen with exact arithmetic.
        // Fall back to plain PSA on the original weights.
        build_parallel(weights, thread_count);
        return;
    }

    // ---- Phase 3: scatter into light/heavy index arrays ----
    light_items_scratch_.resize((size_t)light_count);
    heavy_items_scratch_.resize((size_t)heavy_count);

    #pragma omp parallel num_threads(parts)
    {
        const int t = omp_get_thread_num();
        const size_t a = alias_sampler_chunk_begin((size_t)n_, t, parts);
        const size_t b = alias_sampler_chunk_begin((size_t)n_, t + 1, parts);

        int li = thread_light_offsets_scratch_[(size_t)t];
        int hi = thread_heavy_offsets_scratch_[(size_t)t];
        for (size_t k = a; k < b; ++k) {
            if (psa_plus_filled_scratch_[k]) continue;
            const double scaled = norm_scratch_[k];
            if      (scaled < 1.0) light_items_scratch_[(size_t)li++] = (int)k;
            else if (scaled > 1.0) heavy_items_scratch_[(size_t)hi++] = (int)k;
        }
    }

    // ---- Phase 4: parallel prefix sums over scaled weights ----
    alias_sampler_parallel_prefix_sum(light_items_scratch_, norm_scratch_,
                                      light_prefix_scratch_, parts,
                                      thread_prefix_offsets_scratch_);
    alias_sampler_parallel_prefix_sum(heavy_items_scratch_, norm_scratch_,
                                      heavy_prefix_scratch_, parts,
                                      thread_prefix_offsets_scratch_);

    // ---- Phase 5: split-point binary search per chunk ----
    const int active_parts = std::min(parts, strict_item_count);
    split_scratch_.assign((size_t)active_parts, {});
    split_scratch_[0].light_count = 0;
    split_scratch_[0].heavy_count = 0;
    split_scratch_[0].spill =
        norm_scratch_[(size_t)heavy_items_scratch_[0]];

    auto split_boundary = [&](int target_bucket_count) {
        SplitState split;
        const int search_low  = std::max(0, target_bucket_count - light_count);
        const int search_high = std::min(target_bucket_count, heavy_count - 1);
        int low = search_low, high = search_high;
        int best_heavy = search_low;
        while (low <= high) {
            const int used_heavy = (low + high) / 2;
            const int used_light = target_bucket_count - used_heavy;
            const double sigma =
                light_prefix_scratch_[(size_t)used_light]
                + heavy_prefix_scratch_[(size_t)used_heavy];
            if (sigma <= (double)target_bucket_count + kAliasSamplerBucketEpsilon) {
                best_heavy = used_heavy;
                low = used_heavy + 1;
            } else {
                high = used_heavy - 1;
            }
        }
        const int used_heavy = best_heavy;
        const int used_light = target_bucket_count - used_heavy;
        const double sigma =
            light_prefix_scratch_[(size_t)used_light]
            + heavy_prefix_scratch_[(size_t)used_heavy];
        const double current_heavy_w =
            norm_scratch_[(size_t)heavy_items_scratch_[(size_t)used_heavy]];
        split.light_count = used_light;
        split.heavy_count = used_heavy;
        split.spill = std::max(0.0, current_heavy_w + sigma
                                    - (double)target_bucket_count);
        return split;
    };

    #pragma omp parallel for schedule(static) num_threads(active_parts)
    for (int part = 1; part < active_parts; ++part) {
        const int target_bucket_count = (strict_item_count * part) / active_parts;
        split_scratch_[(size_t)part] = split_boundary(target_bucket_count);
    }

    // ---- Phase 6: per-chunk pack sweep ----
    #pragma omp parallel for schedule(static) num_threads(active_parts)
    for (int part = 0; part < active_parts; ++part) {
        const int bucket_begin = (strict_item_count * part) / active_parts;
        const int bucket_end   = (strict_item_count * (part + 1)) / active_parts;
        const int bucket_count = bucket_end - bucket_begin;
        if (bucket_count <= 0) continue;

        int    light_index = split_scratch_[(size_t)part].light_count;
        int    heavy_index = split_scratch_[(size_t)part].heavy_count;
        double residual    = split_scratch_[(size_t)part].spill;

        for (int bucket = 0; bucket < bucket_count; ++bucket) {
            const bool pack_heavy =
                (light_index >= light_count)
                || (residual <= 1.0 + kAliasSamplerBucketEpsilon);
            if (pack_heavy) {
                const int heavy_item = heavy_items_scratch_[(size_t)heavy_index];
                if (heavy_index + 1 < heavy_count) {
                    prob_[(size_t)heavy_item] =
                        std::min(1.0, std::max(0.0, residual));
                    alias_[(size_t)heavy_item] =
                        heavy_items_scratch_[(size_t)heavy_index + 1u];
                    ++heavy_index;
                    residual =
                        residual
                        + norm_scratch_[(size_t)heavy_items_scratch_[(size_t)heavy_index]]
                        - 1.0;
                } else {
                    prob_[(size_t)heavy_item]  = 1.0;
                    alias_[(size_t)heavy_item] = heavy_item;
                    ++heavy_index;
                    residual = 0.0;
                }
            } else {
                const int light_item = light_items_scratch_[(size_t)light_index++];
                prob_[(size_t)light_item]  = norm_scratch_[(size_t)light_item];
                alias_[(size_t)light_item] =
                    heavy_items_scratch_[(size_t)heavy_index];
                residual += norm_scratch_[(size_t)light_item] - 1.0;
            }
        }
    }
}

inline int AliasSampler::sample(RNG& rng) const {
    if (uniform_) return (int)rng.uniform_int((uint64_t)n_);
    double r = rng.uniform() * n_;
    int i = (int)r;
    if (i >= n_) i = n_ - 1;
    return ((r - i) < prob_[i]) ? i : alias_[i];
}
