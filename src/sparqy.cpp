// sparqy/src/sparqy.cpp
//
// The implementation here is written
// around a packed mutation representation and thread-local offspring blocks:
//
//   1. Reserve metadata capacity for this generation's new mutations.
//   2. Build a weighted parent sampler from current individual fitness.
//   3. Hand each thread a local stash of recyclable mutation IDs.
//   4. In parallel, each thread builds a contiguous block of offspring:
//        - choose parents
//        - build two recombinant gametes
//        - add new mutations to each gamete
//        - compute offspring fitness immediately while the data is hot
//   5. Reclaim any recyclable IDs the threads did not consume.
//   6. Convert per-haplotype lengths into global offsets.
//   7. Copy thread-local offspring blocks into the packed offspring population.
//   8. Count mutation copies in the offspring generation.
//   9. Classify mutations as lost, fixed, or still segregating.
//  10. Reduce summary statistics and promote offspring to parents.

#include "sparqy.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

#include <omp.h>

namespace {

inline double profile_seconds_now() {
    return omp_get_wtime();
}

}  // namespace

int resolve_simulation_thread_count(int configured_threads) {
    int effective_threads =
        (configured_threads > 0) ? configured_threads : omp_get_max_threads();
    if (effective_threads < 1) effective_threads = 1;
    return effective_threads;
}

ParentSamplerBuildMode resolve_parent_sampler_build_mode(
    ParentSamplerBuildMode requested_mode,
    int population_size,
    int effective_threads) {
    if (requested_mode != ParentSamplerBuildMode::automatic) {
        return requested_mode;
    }
    if (effective_threads <= 1) {
        return ParentSamplerBuildMode::sequential;
    }
    return (population_size < kAutoParentSamplerBuildModePopulationThreshold)
               ? ParentSamplerBuildMode::sequential
               : ParentSamplerBuildMode::parallel_psa_plus;
}

// ===========================================================================
// Construction
// ===========================================================================

Simulator::Simulator(const SimParams& p)
    : p_(p),
      individual_fitness_(p.N, 1.0)
{
    if (p_.N <= 0) throw std::invalid_argument("N must be positive");
    if (p_.G <= 0) throw std::invalid_argument("G must be positive");

    nthreads_ = resolve_simulation_thread_count(p_.threads);
    p_.parent_sampler_build_mode = resolve_parent_sampler_build_mode(
        p_.parent_sampler_build_mode, p_.N, nthreads_);

    // Each thread gets an independent RNG stream to avoid overlap
    thread_rngs_.resize((size_t)nthreads_);
    thread_scratch_.resize((size_t)nthreads_);
    thread_offspring_word_totals_.resize((size_t)nthreads_, 0u);
    thread_offspring_word_offsets_.resize((size_t)nthreads_, 0u);
    thread_rngs_[0].seed(p_.seed);
    for (int t = 1; t < nthreads_; t++) {
        thread_rngs_[(size_t)t] = thread_rngs_[(size_t)t - 1];
        thread_rngs_[(size_t)t].long_jump();
    }

    // Sort the statistic requests by generation, remove duplicates, and compile the statistics plans
    for (StatisticRequest& request : p_.statistic_requests) {
        std::sort(request.generations.begin(), request.generations.end());
        request.generations.erase(
            std::unique(request.generations.begin(), request.generations.end()),
            request.generations.end());
    }
    compile_statistics_plans();

    // Initialize the haplotype offsets for the parent and offspring populations
    parent_population_.haplotype_offsets.assign((size_t)(2 * p_.N + 1), 0);
    offspring_population_.haplotype_offsets.assign((size_t)(2 * p_.N + 1), 0);

    // Compile the model (chromosomes, mutation types, regions)
    compile_model();

    // Reserve space for the mutation metadata tables
    const size_t initial_capacity = std::max<size_t>(
        4096,
        (size_t)(4.0 * p_.N * p_.mu * total_mutation_mass_) + 1024);
    mutation_loci_.resize(initial_capacity, 0);
    mutation_homozygous_fitness_factor_.resize(initial_capacity, 1.0);
    mutation_heterozygous_fitness_factor_.resize(initial_capacity, 1.0);
    mutation_masking_coefficient_.resize(initial_capacity, 0.0);
    mutation_type_index_.resize(initial_capacity, 0);
    mutation_origin_generation_.resize(initial_capacity, 0u);
    parent_copy_counts_.resize(initial_capacity, 0);
    offspring_copy_counts_.resize(initial_capacity, 0);
}

// ===========================================================================
// One generation
// ===========================================================================

void Simulator::step() {
    // Get upcoming generation number and the statistics plan if needed
    const uint64_t current_generation = completed_generations_ + 1u;
    const StatisticsExecutionPlan& statistics_plan =
        statistics_plan_for_generation(current_generation);
    const bool profiling_enabled = p_.enable_profiling;

    latest_profile_ = {};
    latest_profile_.generation = current_generation;
    const double step_begin = profiling_enabled ? profile_seconds_now() : 0.0;
    double phase_begin = step_begin;

    // Calculate the expected number of new mutations and reserve space for them (with padding)
    const double expected_new_mutations_mean = 2.0 * p_.N * mutation_lambda_;
    const uint32_t expected_new_mutations = (uint32_t)std::ceil(
        expected_new_mutations_mean
        + 8.0 * std::sqrt(std::max(1.0, expected_new_mutations_mean))
        + 4096.0);
    reserve_mutation_metadata_capacity(expected_new_mutations);
    if (profiling_enabled) {
        const double phase_end = profile_seconds_now();
        latest_profile_.reserve_mutation_metadata_sec = phase_end - phase_begin;
        phase_begin = phase_end;
    }

    // Build the parent sampler from the individual fitness
    switch (p_.parent_sampler_build_mode) {
        case ParentSamplerBuildMode::parallel:
            parent_sampler_.build_parallel(individual_fitness_, nthreads_);
            break;
        case ParentSamplerBuildMode::parallel_psa_plus:
            parent_sampler_.build_parallel_psa_plus(individual_fitness_, nthreads_);
            break;
        case ParentSamplerBuildMode::sequential:
        default:
            parent_sampler_.build(individual_fitness_);
            break;
    }
    if (profiling_enabled) {
        const double phase_end = profile_seconds_now();
        latest_profile_.build_parent_sampler_sec = phase_end - phase_begin;
        phase_begin = phase_end;
    }

    // Build the offspring generation and compute the fitness; compute masked bonus if needed
    build_offspring_generation_and_compute_fitness(
        statistics_plan.need_realized_masking_bonus,
        current_generation);

    // Finalize the offspring generation and update the mutation counts
    finalize_offspring_generation();
    if (profiling_enabled) {
        phase_begin = profile_seconds_now();
    }

    // Swap the parent and offspring populations before incrementing
    std::swap(parent_population_.mutation_ids, offspring_population_.mutation_ids);
    std::swap(parent_population_.haplotype_offsets,
              offspring_population_.haplotype_offsets);
    if (profiling_enabled) {
        const double phase_end = profile_seconds_now();
        latest_profile_.swap_populations_sec = phase_end - phase_begin;
        latest_profile_.total_sec = phase_end - step_begin;
    }
    completed_generations_++;
}

// ===========================================================================
// Capacity management
// ===========================================================================

// Reserve space for the mutation metadata tables
void Simulator::reserve_mutation_metadata_capacity(uint32_t expected_new_mutations) {
    const size_t needed = (size_t)next_unused_mutation_id_ + expected_new_mutations + 1024u;
    if (mutation_loci_.size() >= needed) return;

    const size_t new_capacity = std::max<size_t>(needed, mutation_loci_.size() * 2);
    mutation_loci_.resize(new_capacity, 0);
    mutation_homozygous_fitness_factor_.resize(new_capacity, 1.0);
    mutation_heterozygous_fitness_factor_.resize(new_capacity, 1.0);
    mutation_masking_coefficient_.resize(new_capacity, 0.0);
    mutation_type_index_.resize(new_capacity, 0);
    mutation_origin_generation_.resize(new_capacity, 0u);
    parent_copy_counts_.resize(new_capacity, 0);
    offspring_copy_counts_.resize(new_capacity, 0);
}

// ===========================================================================
// Recyclable mutation IDs
// ===========================================================================

void Simulator::distribute_recyclable_mutation_ids_to_threads() {
    const double per_gamete_mutation_lambda = mutation_lambda_;
    const size_t per_thread = (size_t)std::ceil(
        (2.0 * per_gamete_mutation_lambda) * ((double)p_.N / nthreads_) + 256.0);

    for (int thread_index = 0; thread_index < nthreads_; thread_index++) {
        auto& local_recyclable_ids = thread_scratch_[(size_t)thread_index].recyclable_mutation_ids;
        local_recyclable_ids.clear();

        const size_t take = std::min(per_thread, recyclable_mutation_ids_.size());
        if (take > 0u) {
            local_recyclable_ids.insert(local_recyclable_ids.end(),
                                        recyclable_mutation_ids_.end() - (ptrdiff_t)take,
                                        recyclable_mutation_ids_.end());
            recyclable_mutation_ids_.erase(recyclable_mutation_ids_.end() - (ptrdiff_t)take,
                                           recyclable_mutation_ids_.end());
        }
    }
}

void Simulator::reclaim_unused_recyclable_mutation_ids_from_threads() {
    for (int thread_index = 0; thread_index < nthreads_; thread_index++) {
        auto& local_recyclable_ids = thread_scratch_[(size_t)thread_index].recyclable_mutation_ids;
        recyclable_mutation_ids_.insert(recyclable_mutation_ids_.end(),
                                        local_recyclable_ids.begin(),
                                        local_recyclable_ids.end());
        local_recyclable_ids.clear();
    }
}

uint32_t Simulator::allocate_mutation_id(OffspringBlockScratch& scratch,
                                         std::atomic<uint32_t>& next_mutation_id) {
    if (!scratch.recyclable_mutation_ids.empty()) {
        uint32_t id = scratch.recyclable_mutation_ids.back();
        scratch.recyclable_mutation_ids.pop_back();
        return id;
    }
    return next_mutation_id.fetch_add(1u, std::memory_order_relaxed);
}

// ===========================================================================
// Sampling helpers
// ===========================================================================

uint32_t Simulator::sample_mutation_region_index(RNG& rng) {
    if (compiled_mutation_regions_.size() == 1) return 0;
    double u = rng.uniform() * total_mutation_mass_;
    size_t idx = (size_t)(std::lower_bound(mutation_region_cdf_.begin(),
                                           mutation_region_cdf_.end(),
                                           u)
                          - mutation_region_cdf_.begin());
    if (idx >= compiled_mutation_regions_.size()) idx = compiled_mutation_regions_.size() - 1;
    return (uint32_t)idx;
}

uint32_t Simulator::sample_mutation_type_index(const CompiledMutationRegion& region, RNG& rng) {
    if (region.mutation_type_indices.size() == 1) return region.mutation_type_indices[0];
    double u = rng.uniform() * region.mutation_type_cdf.back();
    size_t idx = (size_t)(std::lower_bound(region.mutation_type_cdf.begin(),
                                           region.mutation_type_cdf.end(),
                                           u)
                          - region.mutation_type_cdf.begin());
    if (idx >= region.mutation_type_indices.size()) idx = region.mutation_type_indices.size() - 1;
    return region.mutation_type_indices[idx];
}

uint64_t Simulator::sample_mutation_locus(const CompiledMutationRegion& region, RNG& rng) {
    uint32_t width = region.end - region.start;
    uint32_t pos = region.start
                 + (width > 0 ? (uint32_t)rng.uniform_int((uint64_t)width) : 0u);
    return pack_locus(region.chromosome, pos);
}

void Simulator::sample_crossover_breakpoints(RNG& rng,
                                             OffspringBlockScratch& scratch,
                                             const CompiledChromosome& chromosome,
                                             uint32_t n_crossovers) {
    scratch.crossover_breakpoints.clear();
    if (n_crossovers == 0u || chromosome.recombination_mass <= 0.0) return;

    scratch.crossover_breakpoints.reserve(n_crossovers);
    if (chromosome.recombination_intervals.size() == 1u) {
        const RecIntervalSpec& interval = chromosome.recombination_intervals[0];
        uint32_t width = interval.end - interval.start;
        for (uint32_t i = 0; i < n_crossovers; ++i) {
            scratch.crossover_breakpoints.push_back(
                interval.start
                + (width > 0 ? (uint32_t)rng.uniform_int((uint64_t)width) : 0u));
        }
    } else {
        for (uint32_t i = 0; i < n_crossovers; ++i) {
            double u = rng.uniform() * chromosome.recombination_mass;
            size_t idx = (size_t)(std::lower_bound(chromosome.recombination_interval_cdf.begin(),
                                                   chromosome.recombination_interval_cdf.end(),
                                                   u)
                                  - chromosome.recombination_interval_cdf.begin());
            if (idx >= chromosome.recombination_intervals.size())
                idx = chromosome.recombination_intervals.size() - 1;
            const RecIntervalSpec& interval = chromosome.recombination_intervals[idx];
            uint32_t width = interval.end - interval.start;
            scratch.crossover_breakpoints.push_back(
                interval.start
                + (width > 0 ? (uint32_t)rng.uniform_int((uint64_t)width) : 0u));
        }
    }

    if (n_crossovers > 1) {
        std::sort(scratch.crossover_breakpoints.begin(), scratch.crossover_breakpoints.end());
    }
}

// ===========================================================================
// New mutation creation
// ===========================================================================

void Simulator::create_new_gamete_mutations(RNG& rng,
                                            OffspringBlockScratch& scratch,
                                            std::atomic<uint32_t>& next_mutation_id,
                                            uint64_t current_generation) {
    uint32_t n_new_mutations = use_fast_mutation_poisson_
                             ? rng.poisson_precomputed(exp_neg_mutation_lambda_)
                             : rng.poisson(mutation_lambda_);
    scratch.new_gamete_mutation_ids.clear();
    if (n_new_mutations == 0) return;

    scratch.new_gamete_mutation_ids.reserve(n_new_mutations);

    for (uint32_t i = 0; i < n_new_mutations; ++i) {
        uint32_t region_index = sample_mutation_region_index(rng);
        const CompiledMutationRegion& region = compiled_mutation_regions_[region_index];
        uint32_t mutation_type_index = sample_mutation_type_index(region, rng);
        const CompiledMutationType& mutation_type = compiled_mutation_types_[mutation_type_index];

        double s = mutation_type.selection.sample(rng);
        if (s <= -1.0) s = -0.999999999;
        const bool additive_dominance = mutation_type.is_additive_dominance;
        double h = additive_dominance
                 ? AdditiveDominanceSpec::additive_h
                 : mutation_type.dominance.sample(rng, s);

        uint32_t mutation_id = allocate_mutation_id(scratch, next_mutation_id);
        mutation_loci_[mutation_id] = sample_mutation_locus(region, rng);
        mutation_homozygous_fitness_factor_[mutation_id] = 1.0 + s;
        mutation_heterozygous_fitness_factor_[mutation_id] = 1.0 + h * s;
        mutation_masking_coefficient_[mutation_id] =
            additive_dominance ? 0.0 : s * (0.5 - h);
        mutation_type_index_[mutation_id] = mutation_type_index;
        mutation_origin_generation_[mutation_id] = current_generation;
        parent_copy_counts_[mutation_id] = 0;
        offspring_copy_counts_[mutation_id] = 0;

        scratch.new_gamete_mutation_ids.push_back(mutation_id);
        scratch.mutation_ids_created_this_thread.push_back(mutation_id);
    }

    std::sort(scratch.new_gamete_mutation_ids.begin(),
              scratch.new_gamete_mutation_ids.end(),
              [this](uint32_t a, uint32_t b) {
                  return (mutation_loci_[a] < mutation_loci_[b])
                      || (mutation_loci_[a] == mutation_loci_[b] && a < b);
              });
}

// ===========================================================================
// Gamete construction by recombination
// ===========================================================================

void Simulator::build_recombined_gamete(RNG& rng,
                                        OffspringBlockScratch& scratch,
                                        uint32_t first_parent_haplotype,
                                        uint32_t second_parent_haplotype,
                                        std::atomic<uint32_t>& next_mutation_id,
                                        uint64_t current_generation) {
    const uint32_t haplotype_count = (uint32_t)(2 * p_.N);
    const uint64_t first_parent_end =
        parent_population_.haplotype_offsets[first_parent_haplotype + 1];
    const uint64_t second_parent_end =
        parent_population_.haplotype_offsets[second_parent_haplotype + 1];

    uint64_t first_parent_cursor =
        parent_population_.haplotype_offsets[first_parent_haplotype];
    uint64_t second_parent_cursor =
        parent_population_.haplotype_offsets[second_parent_haplotype];

    // Lost and fixed mutations are omitted from explicit haplotypes; their
    // effects are already represented elsewhere.
    auto advance_to_segregating_mutation =
        [&](uint64_t& idx, uint64_t end) {
            while (idx < end) {
                uint32_t mutation_id = parent_population_.mutation_ids[idx];
                uint32_t count = parent_copy_counts_[mutation_id];
                if (count > 0u && count < haplotype_count) break;
                ++idx;
            }
        };

    scratch.inherited_gamete_mutation_ids.clear();

    for (uint32_t chromosome_index = 0;
         chromosome_index < compiled_chromosomes_.size();
         ++chromosome_index) {
        uint64_t first_parent_chromosome_end = first_parent_cursor;
        while (first_parent_chromosome_end < first_parent_end
               && locus_chrom(mutation_loci_[
                      parent_population_.mutation_ids[first_parent_chromosome_end]])
                      == chromosome_index) {
            ++first_parent_chromosome_end;
        }

        uint64_t second_parent_chromosome_end = second_parent_cursor;
        while (second_parent_chromosome_end < second_parent_end
               && locus_chrom(mutation_loci_[
                      parent_population_.mutation_ids[second_parent_chromosome_end]])
                      == chromosome_index) {
            ++second_parent_chromosome_end;
        }

        const CompiledChromosome& chromosome = compiled_chromosomes_[chromosome_index];
        uint32_t n_crossovers = chromosome.use_fast_poisson
                              ? rng.poisson_precomputed(chromosome.exp_neg_crossover_lambda)
                              : (chromosome.crossover_lambda > 0.0
                                     ? rng.poisson(chromosome.crossover_lambda)
                                     : 0u);
        bool start_on_second_haplotype = (rng.uniform_int(2) != 0);
        sample_crossover_breakpoints(rng, scratch, chromosome, n_crossovers);

        uint64_t first_stream = first_parent_cursor;
        uint64_t second_stream = second_parent_cursor;

        if (n_crossovers == 0) {
            if (!start_on_second_haplotype) {
                while (first_stream < first_parent_chromosome_end) {
                    advance_to_segregating_mutation(first_stream, first_parent_chromosome_end);
                    if (first_stream >= first_parent_chromosome_end) break;
                    scratch.inherited_gamete_mutation_ids.push_back(
                        parent_population_.mutation_ids[first_stream++]);
                }
            } else {
                while (second_stream < second_parent_chromosome_end) {
                    advance_to_segregating_mutation(second_stream, second_parent_chromosome_end);
                    if (second_stream >= second_parent_chromosome_end) break;
                    scratch.inherited_gamete_mutation_ids.push_back(
                        parent_population_.mutation_ids[second_stream++]);
                }
            }
        } else if (n_crossovers == 1) {
            uint32_t breakpoint = scratch.crossover_breakpoints[0];
            while (true) {
                advance_to_segregating_mutation(first_stream, first_parent_chromosome_end);
                advance_to_segregating_mutation(second_stream, second_parent_chromosome_end);
                if (first_stream >= first_parent_chromosome_end
                    && second_stream >= second_parent_chromosome_end) {
                    break;
                }

                bool taking_from_first_parent;
                if (second_stream >= second_parent_chromosome_end) {
                    taking_from_first_parent = true;
                } else if (first_stream >= first_parent_chromosome_end) {
                    taking_from_first_parent = false;
                } else {
                    uint32_t first_id = parent_population_.mutation_ids[first_stream];
                    uint32_t second_id = parent_population_.mutation_ids[second_stream];
                    if (first_id == second_id) {
                        scratch.inherited_gamete_mutation_ids.push_back(first_id);
                        ++first_stream;
                        ++second_stream;
                        continue;
                    }
                    taking_from_first_parent =
                        (mutation_loci_[first_id] < mutation_loci_[second_id])
                     || (mutation_loci_[first_id] == mutation_loci_[second_id]
                         && first_id < second_id);
                }

                uint32_t mutation_id = taking_from_first_parent
                                     ? parent_population_.mutation_ids[first_stream++]
                                     : parent_population_.mutation_ids[second_stream++];
                uint32_t position = locus_pos(mutation_loci_[mutation_id]);
                bool before_breakpoint = (position < breakpoint);
                bool taking_from_starting_haplotype =
                    (before_breakpoint != start_on_second_haplotype);
                if (taking_from_first_parent == taking_from_starting_haplotype)
                    scratch.inherited_gamete_mutation_ids.push_back(mutation_id);
            }
        } else {
            size_t crossover_index = 0;
            while (true) {
                advance_to_segregating_mutation(first_stream, first_parent_chromosome_end);
                advance_to_segregating_mutation(second_stream, second_parent_chromosome_end);
                if (first_stream >= first_parent_chromosome_end
                    && second_stream >= second_parent_chromosome_end) {
                    break;
                }

                bool taking_from_first_parent;
                if (second_stream >= second_parent_chromosome_end) {
                    taking_from_first_parent = true;
                } else if (first_stream >= first_parent_chromosome_end) {
                    taking_from_first_parent = false;
                } else {
                    uint32_t first_id = parent_population_.mutation_ids[first_stream];
                    uint32_t second_id = parent_population_.mutation_ids[second_stream];
                    if (first_id == second_id) {
                        scratch.inherited_gamete_mutation_ids.push_back(first_id);
                        ++first_stream;
                        ++second_stream;
                        continue;
                    }
                    taking_from_first_parent =
                        (mutation_loci_[first_id] < mutation_loci_[second_id])
                     || (mutation_loci_[first_id] == mutation_loci_[second_id]
                         && first_id < second_id);
                }

                uint32_t mutation_id = taking_from_first_parent
                                     ? parent_population_.mutation_ids[first_stream++]
                                     : parent_population_.mutation_ids[second_stream++];
                uint32_t position = locus_pos(mutation_loci_[mutation_id]);
                while (crossover_index < scratch.crossover_breakpoints.size()
                       && scratch.crossover_breakpoints[crossover_index] < position) {
                    ++crossover_index;
                }
                bool currently_on_second_haplotype =
                    start_on_second_haplotype ^ ((crossover_index & 1u) != 0u);
                if (taking_from_first_parent == !currently_on_second_haplotype)
                    scratch.inherited_gamete_mutation_ids.push_back(mutation_id);
            }
        }

        first_parent_cursor = first_parent_chromosome_end;
        second_parent_cursor = second_parent_chromosome_end;
    }

    create_new_gamete_mutations(rng, scratch, next_mutation_id, current_generation);

    auto& offspring_block = scratch.offspring_block_mutation_ids;
    const auto& inherited_mutations = scratch.inherited_gamete_mutation_ids;
    const auto& new_mutations = scratch.new_gamete_mutation_ids;

    size_t inherited_index = 0;
    size_t new_index = 0;
    while (inherited_index < inherited_mutations.size()
           && new_index < new_mutations.size()) {
        uint32_t inherited_id = inherited_mutations[inherited_index];
        uint32_t new_id = new_mutations[new_index];
        bool inherited_first =
            (mutation_loci_[inherited_id] < mutation_loci_[new_id])
         || (mutation_loci_[inherited_id] == mutation_loci_[new_id]
             && inherited_id < new_id);
        offspring_block.push_back(
            inherited_first ? inherited_mutations[inherited_index++]
                            : new_mutations[new_index++]);
    }
    while (inherited_index < inherited_mutations.size())
        offspring_block.push_back(inherited_mutations[inherited_index++]);
    while (new_index < new_mutations.size())
        offspring_block.push_back(new_mutations[new_index++]);
}

// ===========================================================================
// Offspring generation and fitness evaluation
// ===========================================================================

void Simulator::build_offspring_generation_and_compute_fitness(
    bool compute_homozygous_genome_fitness,
    uint64_t current_generation) {
    const bool profiling_enabled = p_.enable_profiling;
    std::atomic<uint32_t> next_mutation_id(next_unused_mutation_id_);
    mutation_ids_created_this_generation_.clear();

    if (compute_homozygous_genome_fitness) {
        homozygous_genome_fitness_.assign((size_t)p_.N, 1.0);
    } else {
        homozygous_genome_fitness_.clear();
    }

    // Each thread overwrites the starts for its own haplotype block, so there
    // is no need to zero-fill the array every generation.
    const size_t haplotype_count = (size_t)(2 * p_.N);
    if (offspring_population_.haplotype_offsets.size() != haplotype_count + 1u) {
        offspring_population_.haplotype_offsets.resize(haplotype_count + 1u);
    }

    double phase_begin = profiling_enabled ? profile_seconds_now() : 0.0;
    distribute_recyclable_mutation_ids_to_threads();
    if (profiling_enabled) {
        const double phase_end = profile_seconds_now();
        latest_profile_.distribute_recyclable_ids_sec = phase_end - phase_begin;
        phase_begin = phase_end;
    }

    #pragma omp parallel num_threads(nthreads_)
    {
        const int thread_index = omp_get_thread_num();
        RNG& rng = thread_rngs_[(size_t)thread_index];
        OffspringBlockScratch& scratch = thread_scratch_[(size_t)thread_index];
        auto& offspring_block = scratch.offspring_block_mutation_ids;

        const int individual_begin = (thread_index * p_.N) / nthreads_;
        const int individual_end = ((thread_index + 1) * p_.N) / nthreads_;

        offspring_block.clear();
        scratch.mutation_ids_created_this_thread.clear();

        const double* const hom_factor = mutation_homozygous_fitness_factor_.data();
        const double* const het_factor = mutation_heterozygous_fitness_factor_.data();
        const uint64_t* const mutation_loci = mutation_loci_.data();

        for (int individual_index = individual_begin;
             individual_index < individual_end;
             individual_index++) {
            const int first_parent = parent_sampler_.sample(rng);
            const int second_parent = parent_sampler_.sample(rng);
            const int first_flip = (int)rng.uniform_int(2);
            const int second_flip = (int)rng.uniform_int(2);

            const uint32_t first_parent_haplotype_a =
                (uint32_t)(2 * first_parent + first_flip);
            const uint32_t first_parent_haplotype_b =
                (uint32_t)(2 * first_parent + (1 - first_flip));
            const uint32_t second_parent_haplotype_a =
                (uint32_t)(2 * second_parent + second_flip);
            const uint32_t second_parent_haplotype_b =
                (uint32_t)(2 * second_parent + (1 - second_flip));

            const size_t offspring_haplotype0_start = offspring_block.size();
            offspring_population_.haplotype_offsets[(size_t)(2 * individual_index)] =
                (uint64_t)offspring_haplotype0_start;
            build_recombined_gamete(rng,
                                    scratch,
                                    first_parent_haplotype_a,
                                    first_parent_haplotype_b,
                                    next_mutation_id,
                                    current_generation);

            const size_t offspring_haplotype1_start = offspring_block.size();
            offspring_population_.haplotype_offsets[(size_t)(2 * individual_index + 1)] =
                (uint64_t)offspring_haplotype1_start;
            build_recombined_gamete(rng,
                                    scratch,
                                    second_parent_haplotype_a,
                                    second_parent_haplotype_b,
                                    next_mutation_id,
                                    current_generation);

            const size_t offspring_haplotype1_end = offspring_block.size();
            const uint32_t* const offspring_mutation_ids = offspring_block.data();

            double realized_fitness = 1.0;
            double fully_homozygous_fitness = 1.0;
            size_t hap0_index = offspring_haplotype0_start;
            size_t hap1_index = offspring_haplotype1_start;

            while (hap0_index < offspring_haplotype1_start
                   && hap1_index < offspring_haplotype1_end) {
                const uint32_t hap0_mutation = offspring_mutation_ids[hap0_index];
                const uint32_t hap1_mutation = offspring_mutation_ids[hap1_index];
                if (hap0_mutation == hap1_mutation) {
                    const double factor = hom_factor[hap0_mutation];
                    realized_fitness *= factor;
                    if (compute_homozygous_genome_fitness)
                        fully_homozygous_fitness *= factor;
                    ++hap0_index;
                    ++hap1_index;
                } else if (mutation_loci[hap0_mutation] < mutation_loci[hap1_mutation]
                           || (mutation_loci[hap0_mutation] == mutation_loci[hap1_mutation]
                               && hap0_mutation < hap1_mutation)) {
                    realized_fitness *= het_factor[hap0_mutation];
                    if (compute_homozygous_genome_fitness)
                        fully_homozygous_fitness *= hom_factor[hap0_mutation];
                    ++hap0_index;
                } else {
                    realized_fitness *= het_factor[hap1_mutation];
                    if (compute_homozygous_genome_fitness)
                        fully_homozygous_fitness *= hom_factor[hap1_mutation];
                    ++hap1_index;
                }
            }

            while (hap0_index < offspring_haplotype1_start) {
                const uint32_t mutation_id = offspring_mutation_ids[hap0_index++];
                realized_fitness *= het_factor[mutation_id];
                if (compute_homozygous_genome_fitness)
                    fully_homozygous_fitness *= hom_factor[mutation_id];
            }
            while (hap1_index < offspring_haplotype1_end) {
                const uint32_t mutation_id = offspring_mutation_ids[hap1_index++];
                realized_fitness *= het_factor[mutation_id];
                if (compute_homozygous_genome_fitness)
                    fully_homozygous_fitness *= hom_factor[mutation_id];
            }

            individual_fitness_[(size_t)individual_index] = realized_fitness;
            if (compute_homozygous_genome_fitness)
                homozygous_genome_fitness_[(size_t)individual_index] = fully_homozygous_fitness;
        }

        thread_offspring_word_totals_[(size_t)thread_index] =
            (uint64_t)offspring_block.size();
    }
    if (profiling_enabled) {
        const double phase_end = profile_seconds_now();
        latest_profile_.parallel_reproduction_sec = phase_end - phase_begin;
        phase_begin = phase_end;
    }

    next_unused_mutation_id_ = next_mutation_id.load(std::memory_order_relaxed);

    reclaim_unused_recyclable_mutation_ids_from_threads();
    if (profiling_enabled) {
        const double phase_end = profile_seconds_now();
        latest_profile_.reclaim_recyclable_ids_sec = phase_end - phase_begin;
        phase_begin = phase_end;
    }

    for (int thread_index = 0; thread_index < nthreads_; thread_index++) {
        mutation_ids_created_this_generation_.insert(
            mutation_ids_created_this_generation_.end(),
            thread_scratch_[(size_t)thread_index].mutation_ids_created_this_thread.begin(),
            thread_scratch_[(size_t)thread_index].mutation_ids_created_this_thread.end());
    }
    if (profiling_enabled) {
        const double phase_end = profile_seconds_now();
        latest_profile_.collect_created_mutations_sec = phase_end - phase_begin;
        phase_begin = phase_end;
    }

    // Each thread already wrote local haplotype starts while building its
    // contiguous offspring block. Only the per-thread base offsets remain.
    uint64_t total_mutation_words = 0;
    for (int thread_index = 0; thread_index < nthreads_; thread_index++) {
        thread_offspring_word_offsets_[(size_t)thread_index] = total_mutation_words;
        total_mutation_words += thread_offspring_word_totals_[(size_t)thread_index];
    }
    if (profiling_enabled) {
        const double phase_end = profile_seconds_now();
        latest_profile_.offspring_prefix_sum_sec = phase_end - phase_begin;
        phase_begin = phase_end;
    }

    #pragma omp parallel for schedule(static) num_threads(nthreads_)
    for (int thread_index = 0; thread_index < nthreads_; thread_index++) {
        const int individual_begin = (thread_index * p_.N) / nthreads_;
        const int individual_end = ((thread_index + 1) * p_.N) / nthreads_;
        const size_t haplotype_begin = (size_t)(2 * individual_begin);
        const size_t haplotype_end = (size_t)(2 * individual_end);
        const uint64_t block_offset =
            thread_offspring_word_offsets_[(size_t)thread_index];

        for (size_t haplotype_index = haplotype_begin;
             haplotype_index < haplotype_end;
             haplotype_index++) {
            offspring_population_.haplotype_offsets[haplotype_index] += block_offset;
        }
    }
    if (profiling_enabled) {
        const double phase_end = profile_seconds_now();
        latest_profile_.offspring_offset_adjust_sec = phase_end - phase_begin;
        phase_begin = phase_end;
    }
    offspring_population_.haplotype_offsets[haplotype_count] = total_mutation_words;
    offspring_population_.mutation_ids.resize((size_t)total_mutation_words);

    // Copy each thread's contiguous offspring block into its global position,
    // then count mutation copies within that block.
    #pragma omp parallel num_threads(nthreads_)
    {
        const int thread_index = omp_get_thread_num();
        const int individual_begin = (thread_index * p_.N) / nthreads_;
        const uint32_t first_haplotype_index = (uint32_t)(2 * individual_begin);
        const auto& offspring_block =
            thread_scratch_[(size_t)thread_index].offspring_block_mutation_ids;
        if (!offspring_block.empty()) {
            std::memcpy(offspring_population_.mutation_ids.data()
                            + offspring_population_.haplotype_offsets[first_haplotype_index],
                        offspring_block.data(),
                        offspring_block.size() * sizeof(uint32_t));
        }

        auto& block_counts = thread_scratch_[(size_t)thread_index].offspring_block_counts;
        const size_t observed_words = offspring_block.size();
        const size_t reserve_hint = thread_scratch_[(size_t)thread_index]
                                        .offspring_block_count_reserve_hint;
        const size_t capped_expected_unique = std::max<size_t>(
            64u,
            std::min(observed_words, reserve_hint));
        block_counts.clear();
        block_counts.reserve(capped_expected_unique);
        for (uint32_t mutation_id : offspring_block) block_counts.add(mutation_id);
        thread_scratch_[(size_t)thread_index].offspring_block_count_reserve_hint =
            std::max<size_t>(64u, (size_t)block_counts.size + (size_t)block_counts.size / 4u + 32u);
    }
    if (profiling_enabled) {
        latest_profile_.offspring_copy_and_count_sec = profile_seconds_now() - phase_begin;
    }
}

// ===========================================================================
// Finalize offspring generation
// ===========================================================================

void Simulator::finalize_offspring_generation() {
    const bool profiling_enabled = p_.enable_profiling;
    const uint32_t haplotype_count = (uint32_t)(2 * p_.N);

    const size_t n_parent_segregating = segregating_mutation_ids_.size();
    double phase_begin = profiling_enabled ? profile_seconds_now() : 0.0;
    #pragma omp parallel for schedule(static) num_threads(nthreads_)
    for (ptrdiff_t i = 0; i < (ptrdiff_t)n_parent_segregating; i++) {
        offspring_copy_counts_[segregating_mutation_ids_[(size_t)i]] = 0;
    }
    for (uint32_t mutation_id : mutation_ids_created_this_generation_)
        offspring_copy_counts_[mutation_id] = 0;
    if (profiling_enabled) {
        const double phase_end = profile_seconds_now();
        latest_profile_.zero_offspring_counts_sec = phase_end - phase_begin;
        phase_begin = phase_end;
    }

    for (int thread_index = 0; thread_index < nthreads_; thread_index++) {
        thread_scratch_[(size_t)thread_index].offspring_block_counts.for_each(
            [&](uint32_t mutation_id, uint32_t count) {
                offspring_copy_counts_[mutation_id] += count;
            });
    }
    if (profiling_enabled) {
        const double phase_end = profile_seconds_now();
        latest_profile_.merge_thread_counts_sec = phase_end - phase_begin;
        phase_begin = phase_end;
    }

    const size_t n_new_mutations = mutation_ids_created_this_generation_.size();
    const size_t n_mutations_to_classify = n_parent_segregating + n_new_mutations;

    double newly_fixed_fitness_factor = 1.0;
    uint64_t newly_fixed_mutation_count = 0;

    for (int thread_index = 0; thread_index < nthreads_; thread_index++) {
        thread_scratch_[(size_t)thread_index].surviving_mutation_ids.clear();
        thread_scratch_[(size_t)thread_index].recyclable_mutation_ids_after_finalize.clear();
    }

    #pragma omp parallel num_threads(nthreads_)
    {
        const int thread_index = omp_get_thread_num();
        auto& surviving_mutation_ids =
            thread_scratch_[(size_t)thread_index].surviving_mutation_ids;
        auto& recyclable_after_finalize =
            thread_scratch_[(size_t)thread_index].recyclable_mutation_ids_after_finalize;
        double local_fixed_fitness_factor = 1.0;
        uint64_t local_fixed_mutation_count = 0;

        #pragma omp for schedule(static)
        for (ptrdiff_t i = 0; i < (ptrdiff_t)n_mutations_to_classify; i++) {
            const uint32_t mutation_id =
                ((size_t)i < n_parent_segregating)
                    ? segregating_mutation_ids_[(size_t)i]
                    : mutation_ids_created_this_generation_[(size_t)i - n_parent_segregating];
            const uint32_t offspring_count = offspring_copy_counts_[mutation_id];
            parent_copy_counts_[mutation_id] = offspring_count;

            if (offspring_count == 0) {
                recyclable_after_finalize.push_back(mutation_id);
            } else if (offspring_count == haplotype_count) {
                local_fixed_fitness_factor *=
                    mutation_homozygous_fitness_factor_[mutation_id];
                local_fixed_mutation_count++;
                recyclable_after_finalize.push_back(mutation_id);
            } else {
                surviving_mutation_ids.push_back(mutation_id);
            }
        }

        #pragma omp critical(finalize_reduce)
        {
            newly_fixed_fitness_factor *= local_fixed_fitness_factor;
            newly_fixed_mutation_count += local_fixed_mutation_count;
        }
    }
    if (profiling_enabled) {
        const double phase_end = profile_seconds_now();
        latest_profile_.classify_mutations_sec = phase_end - phase_begin;
        phase_begin = phase_end;
    }

    next_generation_segregating_mutation_ids_.clear();
    next_generation_segregating_mutation_ids_.reserve(n_mutations_to_classify);
    for (int thread_index = 0; thread_index < nthreads_; thread_index++) {
        next_generation_segregating_mutation_ids_.insert(
            next_generation_segregating_mutation_ids_.end(),
            thread_scratch_[(size_t)thread_index].surviving_mutation_ids.begin(),
            thread_scratch_[(size_t)thread_index].surviving_mutation_ids.end());
        recyclable_mutation_ids_.insert(
            recyclable_mutation_ids_.end(),
            thread_scratch_[(size_t)thread_index]
                .recyclable_mutation_ids_after_finalize.begin(),
            thread_scratch_[(size_t)thread_index]
                .recyclable_mutation_ids_after_finalize.end());
    }
    if (profiling_enabled) {
        const double phase_end = profile_seconds_now();
        latest_profile_.collect_finalize_buffers_sec = phase_end - phase_begin;
        phase_begin = phase_end;
    }

    segregating_mutation_ids_.swap(next_generation_segregating_mutation_ids_);
    total_fixed_mutations_ += newly_fixed_mutation_count;
    fixed_mutation_fitness_factor_ *= newly_fixed_fitness_factor;

    if (newly_fixed_fitness_factor != 1.0) {
        double inverse_fixed_factor = 1.0 / newly_fixed_fitness_factor;
        #pragma omp parallel for schedule(static) num_threads(nthreads_)
        for (int individual_index = 0; individual_index < p_.N; individual_index++) {
            individual_fitness_[(size_t)individual_index] *= inverse_fixed_factor;
            if (!homozygous_genome_fitness_.empty()) {
                homozygous_genome_fitness_[(size_t)individual_index] *= inverse_fixed_factor;
            }
        }
    }
    if (profiling_enabled) {
        const double phase_end = profile_seconds_now();
        latest_profile_.adjust_fixed_fitness_sec = phase_end - phase_begin;
        phase_begin = phase_end;
    }

    const uint64_t current_generation = completed_generations_ + 1u;
    const StatisticsExecutionPlan& statistics_plan =
        statistics_plan_for_generation(current_generation);
    latest_statistics_.generation = current_generation;
    latest_statistics_.statistics.clear();

    double mean_fitness = std::numeric_limits<double>::quiet_NaN();
    double genetic_load = std::numeric_limits<double>::quiet_NaN();
    double realized_masking_bonus = std::numeric_limits<double>::quiet_NaN();
    double exact_masking_bonus = std::numeric_limits<double>::quiet_NaN();
    double nucleotide_diversity = std::numeric_limits<double>::quiet_NaN();
    double expected_heterozygosity = std::numeric_limits<double>::quiet_NaN();
    double mean_pairwise_jaccard_similarity = std::numeric_limits<double>::quiet_NaN();
    double mean_pairwise_dice_similarity = std::numeric_limits<double>::quiet_NaN();
    double mean_pairwise_overlap_similarity = std::numeric_limits<double>::quiet_NaN();

    if (statistics_plan.need_mean_fitness
        || statistics_plan.need_genetic_load
        || statistics_plan.need_realized_masking_bonus) {
        double sum_realized_fitness = 0.0;
        double sum_homozygous_fitness = 0.0;
        #pragma omp parallel for reduction(+:sum_realized_fitness, sum_homozygous_fitness) schedule(static) num_threads(nthreads_)
        for (int individual_index = 0; individual_index < p_.N; individual_index++) {
            sum_realized_fitness += individual_fitness_[(size_t)individual_index];
            if (statistics_plan.need_realized_masking_bonus) {
                sum_homozygous_fitness += homozygous_genome_fitness_[(size_t)individual_index];
            }
        }

        mean_fitness = (sum_realized_fitness / p_.N) * fixed_mutation_fitness_factor_;
        if (statistics_plan.need_genetic_load) {
            genetic_load = 1.0 - mean_fitness;
        }
        if (statistics_plan.need_realized_masking_bonus) {
            realized_masking_bonus =
                ((sum_realized_fitness - sum_homozygous_fitness) / p_.N)
                * fixed_mutation_fitness_factor_;
        }
    }

    // Shared per-segregating-mutation pass for the heterozygosity family:
    //   exact_B            uses Sigma mask_coef * 2p(1-p)
    //   nucleotide_diversity uses Sigma 2p(1-p) / total_genome_length
    //   expected_heterozygosity uses Sigma 2p(1-p) / n_seg
    // One pass, one reduction, both sums accumulated together. The extra
    // multiply-accumulate for sum_mask_times_2pq is negligible next to the
    // memory load of parent_copy_counts_/mutation_masking_coefficient_.
    const bool need_mask_sum =
        statistics_plan.need_exact_B && !all_mutation_types_are_additive_;
    const bool need_pq_sum =
        statistics_plan.need_nucleotide_diversity
        || statistics_plan.need_expected_heterozygosity;
    const bool need_segregating_pass = need_mask_sum || need_pq_sum;

    double sum_2pq = 0.0;
    double sum_mask_times_2pq = 0.0;
    if (need_segregating_pass) {
        const double inv_haplotype_count = 1.0 / (double)haplotype_count;
        const size_t n_segregating = segregating_mutation_ids_.size();
        #pragma omp parallel for reduction(+:sum_2pq, sum_mask_times_2pq) schedule(static) num_threads(nthreads_)
        for (ptrdiff_t i = 0; i < (ptrdiff_t)n_segregating; i++) {
            const uint32_t mutation_id = segregating_mutation_ids_[(size_t)i];
            const double q = parent_copy_counts_[mutation_id] * inv_haplotype_count;
            const double heterozygosity = 2.0 * q * (1.0 - q);
            sum_2pq += heterozygosity;
            sum_mask_times_2pq +=
                mutation_masking_coefficient_[mutation_id] * heterozygosity;
        }
    }

    if (statistics_plan.need_exact_B) {
        // All-additive short-circuit: every masking coefficient is 0, so the
        // sum is known to be 0 without running the pass above.
        exact_masking_bonus =
            all_mutation_types_are_additive_ ? 0.0 : sum_mask_times_2pq;
    }
    if (statistics_plan.need_nucleotide_diversity) {
        nucleotide_diversity = (total_genome_length_ > 0u)
                             ? sum_2pq / (double)total_genome_length_
                             : 0.0;
    }
    if (statistics_plan.need_expected_heterozygosity) {
        const size_t n_seg = segregating_mutation_ids_.size();
        expected_heterozygosity = (n_seg > 0u) ? sum_2pq / (double)n_seg : 0.0;
    }

    if (statistics_plan.need_jaccard_similarity
        || statistics_plan.need_dice_similarity
        || statistics_plan.need_overlap_similarity) {
        compute_pairwise_haplotypic_similarity_summaries(
            statistics_plan.need_jaccard_similarity,
            statistics_plan.need_dice_similarity,
            statistics_plan.need_overlap_similarity,
            mean_pairwise_jaccard_similarity,
            mean_pairwise_dice_similarity,
            mean_pairwise_overlap_similarity);
    }

    MutationHistogram histogram_result;
    SiteFrequencySpectrum sfs_result;
    if (statistics_plan.need_histogram || statistics_plan.need_sfs) {
        if (statistics_plan.need_histogram) {
            histogram_result.by_type.assign(compiled_mutation_types_.size(), 0);
            histogram_result.by_chromosome.assign(compiled_chromosomes_.size(), 0);
        }
        if (statistics_plan.need_sfs) {
            sfs_result.unfolded_by_copy_number.assign((size_t)haplotype_count + 1u, 0);
            sfs_result.folded_by_minor_allele_count.assign((size_t)p_.N + 1u, 0);
        }

        for (uint32_t mutation_id : segregating_mutation_ids_) {
            if (statistics_plan.need_histogram) {
                histogram_result.by_type[mutation_type_index_[mutation_id]]++;
                histogram_result.by_chromosome[locus_chrom(mutation_loci_[mutation_id])]++;
            }
            if (statistics_plan.need_sfs) {
                const uint32_t derived_count = parent_copy_counts_[mutation_id];
                if (derived_count == 0u || derived_count >= haplotype_count) continue;
                sfs_result.unfolded_by_copy_number[derived_count]++;
                const uint32_t minor_count =
                    std::min<uint32_t>(derived_count, haplotype_count - derived_count);
                sfs_result.folded_by_minor_allele_count[minor_count]++;
            }
        }
    }

    for (const StatisticRequest* request : statistics_plan.active_requests) {
        ComputedStatistic computed;
        computed.kind = request->kind;
        switch (request->kind) {
            case StatisticKind::mean_fitness:
                computed.value = mean_fitness;
                break;
            case StatisticKind::genetic_load:
                computed.value = genetic_load;
                break;
            case StatisticKind::realized_masking_bonus:
                computed.value = realized_masking_bonus;
                break;
            case StatisticKind::exact_B:
                computed.value = exact_masking_bonus;
                break;
            case StatisticKind::mean_pairwise_haplotypic_similarity:
                computed.similarity_metric = request->similarity_metric;
                switch (request->similarity_metric) {
                    case HaplotypeSimilarityMetric::none:
                        computed.value = 0.0;
                        break;
                    case HaplotypeSimilarityMetric::jaccard:
                        computed.value = mean_pairwise_jaccard_similarity;
                        break;
                    case HaplotypeSimilarityMetric::dice:
                        computed.value = mean_pairwise_dice_similarity;
                        break;
                    case HaplotypeSimilarityMetric::overlap:
                        computed.value = mean_pairwise_overlap_similarity;
                        break;
                }
                break;
            case StatisticKind::n_seg:
                computed.value = (uint64_t)segregating_mutation_ids_.size();
                break;
            case StatisticKind::n_fixed:
                computed.value = total_fixed_mutations_;
                break;
            case StatisticKind::genome_words:
                computed.value = (uint64_t)offspring_population_.mutation_ids.size();
                break;
            case StatisticKind::mutation_histogram:
                computed.value = histogram_result;
                break;
            case StatisticKind::site_frequency_spectrum:
                computed.value = sfs_result;
                break;
            case StatisticKind::nucleotide_diversity:
                computed.value = nucleotide_diversity;
                break;
            case StatisticKind::expected_heterozygosity:
                computed.value = expected_heterozygosity;
                break;
        }
        latest_statistics_.statistics.push_back(std::move(computed));
    }

    mutation_ids_created_this_generation_.clear();
    if (profiling_enabled) {
        latest_profile_.statistics_sec = profile_seconds_now() - phase_begin;
    }
}
