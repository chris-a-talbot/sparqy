#pragma once

#include <array>
#include <cstddef>
#include <string_view>

#include "sparqy.hpp"

namespace sparqy_names {

template <class Enum>
struct NamedValue {
    const char* name;
    Enum value;
};

constexpr std::array<NamedValue<StatisticKind>, 12> kStatisticKinds = {{
    {"mean_fitness", StatisticKind::mean_fitness},
    {"genetic_load", StatisticKind::genetic_load},
    {"realized_masking_bonus", StatisticKind::realized_masking_bonus},
    {"exact_B", StatisticKind::exact_B},
    {"pairwise_similarity", StatisticKind::mean_pairwise_haplotypic_similarity},
    {"n_seg", StatisticKind::n_seg},
    {"n_fixed", StatisticKind::n_fixed},
    {"genome_words", StatisticKind::genome_words},
    {"mutation_histogram", StatisticKind::mutation_histogram},
    {"site_frequency_spectrum", StatisticKind::site_frequency_spectrum},
    {"nucleotide_diversity", StatisticKind::nucleotide_diversity},
    {"expected_heterozygosity", StatisticKind::expected_heterozygosity},
}};

constexpr std::array<NamedValue<HaplotypeSimilarityMetric>, 3> kSimilarityMetrics = {{
    {"jaccard", HaplotypeSimilarityMetric::jaccard},
    {"dice", HaplotypeSimilarityMetric::dice},
    {"overlap", HaplotypeSimilarityMetric::overlap},
}};

constexpr std::array<NamedValue<ParentSamplerBuildMode>, 4> kParentSamplerBuildModes = {{
    {"auto", ParentSamplerBuildMode::automatic},
    {"sequential", ParentSamplerBuildMode::sequential},
    {"parallel", ParentSamplerBuildMode::parallel},
    {"parallel_psa_plus", ParentSamplerBuildMode::parallel_psa_plus},
}};

constexpr size_t kStatisticKindCount = kStatisticKinds.size();

static_assert(static_cast<size_t>(StatisticKind::expected_heterozygosity) + 1u
                  == kStatisticKindCount,
              "StatisticKind metadata assumes contiguous enum values");

template <class Enum, size_t N>
inline const char* name_for_value(const std::array<NamedValue<Enum>, N>& entries,
                                  Enum value,
                                  const char* fallback) {
    for (const NamedValue<Enum>& entry : entries) {
        if (entry.value == value) return entry.name;
    }
    return fallback;
}

template <class Enum, size_t N>
inline bool try_parse_named_value(const std::array<NamedValue<Enum>, N>& entries,
                                  std::string_view name,
                                  Enum& out_value) {
    for (const NamedValue<Enum>& entry : entries) {
        if (name == entry.name) {
            out_value = entry.value;
            return true;
        }
    }
    return false;
}

inline const char* statistic_kind_name(StatisticKind kind) {
    return name_for_value(kStatisticKinds, kind, "unknown");
}

inline bool try_parse_statistic_kind_name(std::string_view name,
                                          StatisticKind& out_kind) {
    return try_parse_named_value(kStatisticKinds, name, out_kind);
}

inline const char* similarity_metric_name(HaplotypeSimilarityMetric metric) {
    return name_for_value(kSimilarityMetrics, metric, "");
}

inline bool try_parse_similarity_metric_name(
    std::string_view name,
    HaplotypeSimilarityMetric& out_metric) {
    return try_parse_named_value(kSimilarityMetrics, name, out_metric);
}

inline const char* parent_sampler_build_mode_name(ParentSamplerBuildMode mode) {
    return name_for_value(kParentSamplerBuildModes, mode, "unknown");
}

inline bool try_parse_parent_sampler_build_mode_name(
    std::string_view name,
    ParentSamplerBuildMode& out_mode) {
    return try_parse_named_value(kParentSamplerBuildModes, name, out_mode);
}

inline constexpr const char* kParentSamplerBuildModeChoicesForError =
    "'auto', 'sequential', 'parallel', or 'parallel_psa_plus'";

}  // namespace sparqy_names
