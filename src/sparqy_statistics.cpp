// sparqy/src/sparqy_statistics.cpp
//
// Statistics plan compilation, per-generation plan lookup, and the pairwise
// haplotypic similarity reduction. 

#include "sparqy.hpp"

#include <algorithm>
#include <limits>

#include <omp.h>

// ===========================================================================
// Statistics plan compilation
// ===========================================================================

void Simulator::add_request_to_statistics_plan(StatisticsExecutionPlan& plan,
                                               const StatisticRequest* request) const {
    for (const StatisticRequest* existing : plan.active_requests) {
        if (existing->kind == request->kind
            && existing->similarity_metric == request->similarity_metric) {
            return;
        }
    }
    plan.active_requests.push_back(request);
    switch (request->kind) {
        case StatisticKind::mean_fitness:
            plan.need_mean_fitness = true;
            break;
        case StatisticKind::genetic_load:
            plan.need_genetic_load = true;
            break;
        case StatisticKind::realized_masking_bonus:
            plan.need_realized_masking_bonus = true;
            break;
        case StatisticKind::exact_B:
            plan.need_exact_B = true;
            break;
        case StatisticKind::mean_pairwise_haplotypic_similarity:
            switch (request->similarity_metric) {
                case HaplotypeSimilarityMetric::none:
                    break;
                case HaplotypeSimilarityMetric::jaccard:
                    plan.need_jaccard_similarity = true;
                    break;
                case HaplotypeSimilarityMetric::dice:
                    plan.need_dice_similarity = true;
                    break;
                case HaplotypeSimilarityMetric::overlap:
                    plan.need_overlap_similarity = true;
                    break;
            }
            break;
        case StatisticKind::n_seg:
            plan.need_n_seg = true;
            break;
        case StatisticKind::n_fixed:
            plan.need_n_fixed = true;
            break;
        case StatisticKind::genome_words:
            plan.need_genome_words = true;
            break;
        case StatisticKind::mutation_histogram:
            plan.need_histogram = true;
            break;
        case StatisticKind::site_frequency_spectrum:
            plan.need_sfs = true;
            break;
        case StatisticKind::nucleotide_diversity:
            plan.need_nucleotide_diversity = true;
            break;
        case StatisticKind::expected_heterozygosity:
            plan.need_expected_heterozygosity = true;
            break;
    }
}

void Simulator::compile_statistics_plans() {
    struct ScheduledStatisticRequest {
        uint64_t generation = 0;
        const StatisticRequest* request = nullptr;
    };

    statistics_every_generation_plan_ = StatisticsExecutionPlan{};
    statistics_generation_specific_plans_.clear();

    std::vector<ScheduledStatisticRequest> scheduled_requests;
    for (const StatisticRequest& request : p_.statistic_requests) {
        if (request.every_generation) {
            add_request_to_statistics_plan(statistics_every_generation_plan_, &request);
            continue;
        }

        for (uint64_t generation : request.generations) {
            if (generation == 0u || generation > (uint64_t)p_.G) continue;
            scheduled_requests.push_back({generation, &request});
        }
    }

    if (scheduled_requests.empty()) return;

    std::sort(scheduled_requests.begin(),
              scheduled_requests.end(),
              [](const ScheduledStatisticRequest& a,
                 const ScheduledStatisticRequest& b) {
                    return a.generation < b.generation;
              });

    size_t cursor = 0;
    while (cursor < scheduled_requests.size()) {
        const uint64_t generation = scheduled_requests[cursor].generation;
        StatisticsExecutionPlan plan = statistics_every_generation_plan_;
        while (cursor < scheduled_requests.size()
               && scheduled_requests[cursor].generation == generation) {
            add_request_to_statistics_plan(plan, scheduled_requests[cursor].request);
            ++cursor;
        }
        statistics_generation_specific_plans_.push_back(
            {generation, std::move(plan)});
    }
}

const Simulator::StatisticsExecutionPlan&
Simulator::statistics_plan_for_generation(uint64_t generation) const {
    const auto it =
        std::lower_bound(statistics_generation_specific_plans_.begin(),
                         statistics_generation_specific_plans_.end(),
                         generation,
                         [](const std::pair<uint64_t, StatisticsExecutionPlan>& entry,
                            uint64_t value) {
                                return entry.first < value;
                         });
    if (it != statistics_generation_specific_plans_.end()
        && it->first == generation) {
        return it->second;
    }
    return statistics_every_generation_plan_;
}

// ===========================================================================
// Pairwise haplotypic similarity reduction
// ===========================================================================

void Simulator::compute_pairwise_haplotypic_similarity_summaries(
    bool need_jaccard,
    bool need_dice,
    bool need_overlap,
    double& jaccard,
    double& dice,
    double& overlap) const {
    const uint32_t haplotype_count = (uint32_t)(2 * p_.N);
    if (haplotype_count < 2u) {
        jaccard = need_jaccard ? 1.0 : std::numeric_limits<double>::quiet_NaN();
        dice = need_dice ? 1.0 : std::numeric_limits<double>::quiet_NaN();
        overlap = need_overlap ? 1.0 : std::numeric_limits<double>::quiet_NaN();
        return;
    }

    const uint64_t pair_count =
        (uint64_t)haplotype_count * (uint64_t)(haplotype_count - 1u) / 2u;
    const uint64_t fully_segregating_count = haplotype_count;

    double sum_jaccard = 0.0;
    double sum_dice = 0.0;
    double sum_overlap = 0.0;
    #pragma omp parallel for reduction(+:sum_jaccard, sum_dice, sum_overlap) schedule(static) num_threads(nthreads_)
    for (ptrdiff_t hap_a = 0; hap_a < (ptrdiff_t)haplotype_count; hap_a++) {
        for (uint32_t hap_b = (uint32_t)hap_a + 1u; hap_b < haplotype_count; ++hap_b) {
            uint64_t cursor_a = offspring_population_.haplotype_offsets[(size_t)hap_a];
            uint64_t end_a    = offspring_population_.haplotype_offsets[(size_t)hap_a + 1u];
            uint64_t cursor_b = offspring_population_.haplotype_offsets[(size_t)hap_b];
            uint64_t end_b    = offspring_population_.haplotype_offsets[(size_t)hap_b + 1u];

            auto advance_to_segregating =
                [&](uint64_t& cursor, uint64_t end) {
                    while (cursor < end) {
                        const uint32_t mutation_id = offspring_population_.mutation_ids[cursor];
                        const uint32_t count = parent_copy_counts_[mutation_id];
                        if (count > 0u && count < fully_segregating_count) break;
                        ++cursor;
                    }
                };

            uint64_t size_a = 0u;
            uint64_t size_b = 0u;
            uint64_t intersection = 0u;

            while (true) {
                advance_to_segregating(cursor_a, end_a);
                advance_to_segregating(cursor_b, end_b);

                if (cursor_a >= end_a || cursor_b >= end_b) break;

                const uint32_t id_a = offspring_population_.mutation_ids[cursor_a];
                const uint32_t id_b = offspring_population_.mutation_ids[cursor_b];
                if (id_a == id_b) {
                    ++intersection;
                    ++size_a;
                    ++size_b;
                    ++cursor_a;
                    ++cursor_b;
                } else if ((mutation_loci_[id_a] < mutation_loci_[id_b])
                           || (mutation_loci_[id_a] == mutation_loci_[id_b] && id_a < id_b)) {
                    ++size_a;
                    ++cursor_a;
                } else {
                    ++size_b;
                    ++cursor_b;
                }
            }

            while (true) {
                advance_to_segregating(cursor_a, end_a);
                if (cursor_a >= end_a) break;
                ++size_a;
                ++cursor_a;
            }
            while (true) {
                advance_to_segregating(cursor_b, end_b);
                if (cursor_b >= end_b) break;
                ++size_b;
                ++cursor_b;
            }

            const uint64_t union_size = size_a + size_b - intersection;
            const uint64_t min_size = std::min(size_a, size_b);
            if (need_jaccard) {
                sum_jaccard += (union_size == 0u)
                             ? 1.0
                             : (double)intersection / (double)union_size;
            }
            if (need_dice) {
                const uint64_t denom = size_a + size_b;
                sum_dice += (denom == 0u)
                          ? 1.0
                          : (2.0 * (double)intersection) / (double)denom;
            }
            if (need_overlap) {
                sum_overlap += (min_size == 0u)
                             ? ((size_a == 0u && size_b == 0u) ? 1.0 : 0.0)
                             : (double)intersection / (double)min_size;
            }
        }
    }

    jaccard = need_jaccard
            ? sum_jaccard / (double)pair_count
            : std::numeric_limits<double>::quiet_NaN();
    dice = need_dice
         ? sum_dice / (double)pair_count
         : std::numeric_limits<double>::quiet_NaN();
    overlap = need_overlap
            ? sum_overlap / (double)pair_count
            : std::numeric_limits<double>::quiet_NaN();
}
