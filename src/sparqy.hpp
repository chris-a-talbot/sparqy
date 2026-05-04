// sparqy/src/sparqy.hpp
//
// Public model types and the Simulator runtime state.

#pragma once

#include <atomic>
#include <cstdint>
#include <limits>
#include <variant>
#include <vector>

#include "alias_sampler.hpp"
#include "dist_spec.hpp"
#include "dominance_spec.hpp"
#include "rng.hpp"
#include "sparse_count_map.hpp"

// Model specification: mutation types, genome regions, and recombination maps.
struct MutationTypeSpec {
    DistSpec      selection = {DistKind::constant, -0.01, 0.0, -0.999999999, 1.0};
    DominanceSpec dominance;
};

struct MutationTypeWeight {
    uint32_t type_index = 0;
    double   weight     = 1.0;
};

// Reusable mutation-behavior annotation, such as "exon" or "intron".
struct RegionTypeSpec {
    double mutation_scale = 1.0;
    std::vector<MutationTypeWeight> mutation_types;
};

// Placement of a reusable region type on a specific chromosome.
struct ChromosomeRegionSpec {
    uint32_t region_type_index = 0;
    uint32_t start             = 0;
    uint32_t end               = 0;
};

struct RecIntervalSpec {
    uint32_t start      = 0;
    uint32_t end        = 0;
    double   rate_scale = 1.0;
};

struct ChromosomeSpec {
    uint32_t length              = 0;
    std::vector<RecIntervalSpec> recombination_map;
    std::vector<ChromosomeRegionSpec> regions;
};

enum class HaplotypeSimilarityMetric : uint8_t {
    none,
    jaccard,
    dice,
    overlap
};

enum class StatisticKind : uint8_t {
    mean_fitness,
    genetic_load,
    realized_masking_bonus,
    exact_B,
    mean_pairwise_haplotypic_similarity,
    n_seg,
    n_fixed,
    genome_words,
    mutation_histogram,
    site_frequency_spectrum,
    nucleotide_diversity,       // Sigma 2p(1-p) / total_genome_length
    expected_heterozygosity     // Sigma 2p(1-p) / n_seg
};

enum class ParentSamplerBuildMode : uint8_t {
    automatic,
    sequential,
    parallel,           // PSA (Hübschle-Schneider & Sanders, Algorithm 3)
    parallel_psa_plus   // PSA+ (§4.2.1: greedy local pre-pass + PSA on remainder)
};

constexpr int kAutoParentSamplerBuildModePopulationThreshold = 10000;

int resolve_simulation_thread_count(int configured_threads);
ParentSamplerBuildMode resolve_parent_sampler_build_mode(
    ParentSamplerBuildMode requested_mode,
    int population_size,
    int effective_threads);

struct StatisticRequest {
    StatisticKind kind = StatisticKind::mean_fitness;
    std::vector<uint64_t> generations;
    HaplotypeSimilarityMetric similarity_metric = HaplotypeSimilarityMetric::jaccard;
};

struct SimParams {
    int      N       = 1000;
    int      G       = 100;
    double   mu      = 1e-7;
    double   rho     = 0.01;
    int      threads = 0;    // 0 = omp_get_max_threads()
    uint64_t seed    = 42;
    bool     enable_profiling = false;
    ParentSamplerBuildMode parent_sampler_build_mode =
        ParentSamplerBuildMode::automatic;

    std::vector<MutationTypeSpec> mutation_types;
    std::vector<RegionTypeSpec>   mutation_region_types;
    std::vector<ChromosomeSpec>   chromosomes;
    std::vector<StatisticRequest> statistic_requests;
};

struct MutationHistogram {
    std::vector<uint64_t> by_type;
    std::vector<uint64_t> by_chromosome;
};

struct SiteFrequencySpectrum {
    std::vector<uint64_t> unfolded_by_copy_number;
    std::vector<uint64_t> folded_by_minor_allele_count;
};

using StatisticPayload = std::variant<
    double,
    uint64_t,
    MutationHistogram,
    SiteFrequencySpectrum>;

struct ComputedStatistic {
    StatisticKind kind = StatisticKind::mean_fitness;
    HaplotypeSimilarityMetric similarity_metric = HaplotypeSimilarityMetric::none;
    StatisticPayload value = 0.0;
};

struct StatisticsSnapshot {
    uint64_t generation = 0;
    std::vector<ComputedStatistic> statistics;
};

struct GenerationProfileSnapshot {
    uint64_t generation = 0;
    double reserve_mutation_metadata_sec = 0.0;
    double build_parent_sampler_sec = 0.0;
    double distribute_recyclable_ids_sec = 0.0;
    double parallel_reproduction_sec = 0.0;
    double reclaim_recyclable_ids_sec = 0.0;
    double collect_created_mutations_sec = 0.0;
    double offspring_prefix_sum_sec = 0.0;
    double offspring_offset_adjust_sec = 0.0;
    double offspring_copy_and_count_sec = 0.0;
    double zero_offspring_counts_sec = 0.0;
    double merge_thread_counts_sec = 0.0;
    double classify_mutations_sec = 0.0;
    double collect_finalize_buffers_sec = 0.0;
    double adjust_fixed_fitness_sec = 0.0;
    double statistics_sec = 0.0;
    double swap_populations_sec = 0.0;
    double total_sec = 0.0;
};

// alignas(64) keeps adjacent scratch objects off the same cache line.
struct alignas(64) OffspringBlockScratch {
    std::vector<uint32_t> offspring_block_mutation_ids;
    std::vector<uint32_t> inherited_gamete_mutation_ids;
    std::vector<uint32_t> new_gamete_mutation_ids;
    std::vector<uint32_t> mutation_ids_created_this_thread;
    std::vector<uint32_t> crossover_breakpoints;
    std::vector<uint32_t> recyclable_mutation_ids;
    SparseCountMap        offspring_block_counts;

    std::vector<uint32_t> surviving_mutation_ids;
    std::vector<uint32_t> recyclable_mutation_ids_after_finalize;
    size_t                offspring_block_count_reserve_hint = 1024u;
};

class Simulator {
public:
    explicit Simulator(const SimParams& p);
    void step();

    const StatisticsSnapshot& latest_statistics() const { return latest_statistics_; }
    const GenerationProfileSnapshot& latest_profile() const { return latest_profile_; }

private:
    struct StatisticsExecutionPlan {
        std::vector<const StatisticRequest*> active_requests;
        bool need_mean_fitness = false;
        bool need_genetic_load = false;
        bool need_realized_masking_bonus = false;
        bool need_exact_B = false;
        bool need_n_seg = false;
        bool need_n_fixed = false;
        bool need_genome_words = false;
        bool need_histogram = false;
        bool need_sfs = false;
        bool need_jaccard_similarity = false;
        bool need_dice_similarity = false;
        bool need_overlap_similarity = false;
        bool need_nucleotide_diversity = false;
        bool need_expected_heterozygosity = false;
    };

    struct CompiledMutationType {
        DistSpec      selection;
        DominanceSpec dominance;
        bool          is_additive_dominance = false;
    };

    struct CompiledMutationRegion {
        uint32_t chromosome = 0;
        uint32_t start = 0;
        uint32_t end = 0;
        double   mutation_mass = 0.0;
        std::vector<uint32_t> mutation_type_indices;
        std::vector<double>   mutation_type_cdf;
    };

    struct CompiledChromosome {
        uint32_t length = 0;
        double   recombination_mass = 0.0;
        double   crossover_lambda = 0.0;
        double   exp_neg_crossover_lambda = 1.0;
        bool     use_fast_poisson = false;
        std::vector<RecIntervalSpec> recombination_intervals;
        std::vector<double>          recombination_interval_cdf;
    };

    // Each haplotype is a slice in a flat mutation-id array.
    struct PackedPopulation {
        std::vector<uint64_t> haplotype_offsets;
        std::vector<uint32_t> mutation_ids;
    };

    SimParams p_;

    int       nthreads_ = 1;

    std::vector<CompiledMutationType>   compiled_mutation_types_;
    std::vector<CompiledMutationRegion> compiled_mutation_regions_;
    std::vector<CompiledChromosome>     compiled_chromosomes_;
    std::vector<double>                 mutation_region_cdf_;
    double                              total_mutation_mass_ = 0.0;
    uint64_t                            total_genome_length_ = 0;
    double                              mutation_lambda_ = 0.0;
    double                              exp_neg_mutation_lambda_ = 1.0;
    bool                                use_fast_mutation_poisson_ = false;
    bool                                all_mutation_types_are_additive_ = true;

    PackedPopulation parent_population_;
    PackedPopulation offspring_population_;

    std::vector<uint64_t> mutation_loci_;
    std::vector<double>   mutation_homozygous_fitness_factor_;
    std::vector<double>   mutation_heterozygous_fitness_factor_;
    std::vector<double>   mutation_masking_coefficient_;
    std::vector<uint32_t> mutation_type_index_;

    std::vector<uint32_t> parent_copy_counts_;
    std::vector<uint32_t> offspring_copy_counts_;

    std::vector<uint32_t> segregating_mutation_ids_;
    std::vector<uint32_t> recyclable_mutation_ids_;
    std::vector<uint32_t> mutation_ids_created_this_generation_;
    std::vector<uint32_t> next_generation_segregating_mutation_ids_;

    std::vector<double> individual_fitness_;
    std::vector<double> homozygous_genome_fitness_;

    AliasSampler        parent_sampler_;

    uint64_t            completed_generations_ = 0;
    uint64_t            total_fixed_mutations_ = 0;
    double              fixed_mutation_fitness_factor_ = 1.0;
    uint32_t next_unused_mutation_id_ = 0;

    StatisticsSnapshot  latest_statistics_;
    GenerationProfileSnapshot latest_profile_;
    StatisticsExecutionPlan statistics_every_generation_plan_;
    std::vector<std::pair<uint64_t, StatisticsExecutionPlan>>
                        statistics_generation_specific_plans_;

    std::vector<RNG>                 thread_rngs_;
    std::vector<OffspringBlockScratch> thread_scratch_;
    std::vector<uint64_t>            thread_offspring_word_totals_;
    std::vector<uint64_t>            thread_offspring_word_offsets_;

    void compile_model();
    void compile_statistics_plans();
    const StatisticsExecutionPlan& statistics_plan_for_generation(
        uint64_t generation) const;
    void add_request_to_statistics_plan(StatisticsExecutionPlan& plan,
        const StatisticRequest* request) const;

    void reserve_mutation_metadata_capacity(uint32_t expected_new_mutations);
    void distribute_recyclable_mutation_ids_to_threads();
    void reclaim_unused_recyclable_mutation_ids_from_threads();
    uint32_t allocate_mutation_id(OffspringBlockScratch& scratch,
                                  std::atomic<uint32_t>& next_mutation_id);

    void     sample_crossover_breakpoints(RNG& rng,
                                          OffspringBlockScratch& scratch,
                                          const CompiledChromosome& chromosome,
                                          uint32_t n_crossovers);
    uint32_t sample_mutation_region_index(RNG& rng);
    uint32_t sample_mutation_type_index(const CompiledMutationRegion& region, RNG& rng);
    uint64_t sample_mutation_locus(const CompiledMutationRegion& region, RNG& rng);

    void     create_new_gamete_mutations(RNG& rng,
                                         OffspringBlockScratch& scratch,
                                         std::atomic<uint32_t>& next_mutation_id);
    void     build_recombined_gamete(RNG& rng,
                                     OffspringBlockScratch& scratch,
                                     uint32_t first_parent_haplotype,
                                     uint32_t second_parent_haplotype,
                                     std::atomic<uint32_t>& next_mutation_id);

    void build_offspring_generation_and_compute_fitness(
        bool compute_homozygous_genome_fitness);
    void finalize_offspring_generation();
    void compute_pairwise_haplotypic_similarity_summaries(
        bool need_jaccard,
        bool need_dice,
        bool need_overlap,
        double& jaccard,
        double& dice,
        double& overlap) const;
};

inline uint64_t pack_locus(uint32_t chrom, uint32_t pos) {
    return ((uint64_t)chrom << 32) | (uint64_t)pos;
}

inline uint32_t locus_chrom(uint64_t loc) { return (uint32_t)(loc >> 32); }
inline uint32_t locus_pos(uint64_t loc)   { return (uint32_t)loc; }
