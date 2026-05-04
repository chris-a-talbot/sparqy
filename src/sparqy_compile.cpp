// sparqy/src/sparqy_compile.cpp
//
// Model compilation: flatten user-facing specs (chromosomes, mutation types,
// region types + placements) into the compiled structures the hot loop reads.
// Runs once in the Simulator constructor; not on the per-generation path.

#include "sparqy.hpp"

#include <cmath>
#include <stdexcept>
#include <utility>

void Simulator::compile_model() {
    // ---- Chromosomes ----
    if (p_.chromosomes.empty())
        throw std::invalid_argument("at least one chromosome must be specified");

    // Reserve space for the compiled chromosomes
    compiled_chromosomes_.reserve(p_.chromosomes.size());
    for (const ChromosomeSpec& spec : p_.chromosomes) {
        if (spec.length == 0u)
            throw std::invalid_argument("chromosome length must be positive");
        CompiledChromosome chromosome;
        chromosome.length = spec.length;
        if (spec.recombination_map.empty())
            chromosome.recombination_intervals.push_back({0u, chromosome.length, 1.0});
        else
            chromosome.recombination_intervals = spec.recombination_map;
        compiled_chromosomes_.push_back(std::move(chromosome));
    }

    // Cache total genome length in bp (used by per-bp heterozygosity stats).
    total_genome_length_ = 0u;
    for (const CompiledChromosome& chromosome : compiled_chromosomes_) {
        total_genome_length_ += (uint64_t)chromosome.length;
    }

    // Compute the total recombination mass for each chromosome
    double total_recombination_mass = 0.0;
    for (size_t chromosome_index = 0; chromosome_index < compiled_chromosomes_.size(); ++chromosome_index) {
        CompiledChromosome& chromosome = compiled_chromosomes_[chromosome_index];
        chromosome.recombination_mass = 0.0;
        chromosome.recombination_interval_cdf.clear();
        for (const RecIntervalSpec& interval : chromosome.recombination_intervals) {
            if (interval.end <= interval.start || interval.end > chromosome.length)
                throw std::invalid_argument("invalid recombination interval");
            chromosome.recombination_mass +=
                interval.rate_scale * (double)(interval.end - interval.start);
            chromosome.recombination_interval_cdf.push_back(chromosome.recombination_mass);
        }
        total_recombination_mass += chromosome.recombination_mass;
    }
    if (total_recombination_mass > 0.0) {
        for (CompiledChromosome& chromosome : compiled_chromosomes_) {
            chromosome.crossover_lambda =
                p_.rho * chromosome.recombination_mass / total_recombination_mass;
        }
    }

    // ---- Mutation types ----
    all_mutation_types_are_additive_ = true;

    if (p_.mutation_types.empty())
        throw std::invalid_argument("at least one mutation type must be specified");

    // Reserve space for the compiled mutation types
    compiled_mutation_types_.reserve(p_.mutation_types.size());
    for (const MutationTypeSpec& spec : p_.mutation_types) {
        compiled_mutation_types_.push_back({
            spec.selection,
            spec.dominance,
            spec.dominance.is_additive()
        });
    }

    // Check if all mutation types are additive
    for (const CompiledMutationType& mutation_type : compiled_mutation_types_) {
        all_mutation_types_are_additive_ =
            all_mutation_types_are_additive_ && mutation_type.is_additive_dominance;
    }

    // ---- Mutable regions ----
    std::vector<RegionTypeSpec> resolved_region_types = p_.mutation_region_types;
    if (resolved_region_types.empty()) {
        resolved_region_types.push_back(RegionTypeSpec{});
    }

    // Count the total number of mutation regions across all chromosomes
    size_t mutation_region_count = 0;
    for (const ChromosomeSpec& chromosome : p_.chromosomes) {
        mutation_region_count += chromosome.regions.size();
    }
    const bool has_mutation_regions = (mutation_region_count > 0u);

    // Append a compiled region to the list
    auto append_compiled_region =
        [&](uint32_t chromosome_index,
            uint32_t start,
            uint32_t end,
            double mutation_scale,
            const std::vector<MutationTypeWeight>& mutation_type_weights) {
            if (chromosome_index >= compiled_chromosomes_.size())
                throw std::invalid_argument("region chromosome out of range");
            if (end <= start || end > compiled_chromosomes_[chromosome_index].length)
                throw std::invalid_argument("invalid region");

            CompiledMutationRegion region;
            region.chromosome = chromosome_index;
            region.start = start;
            region.end = end;
            region.mutation_mass = mutation_scale * (double)(end - start);

            if (mutation_type_weights.empty()) {
                region.mutation_type_indices = {0u};
                region.mutation_type_cdf = {1.0};
            } else {
                double sum = 0.0;
                for (const MutationTypeWeight& weight : mutation_type_weights) {
                    if (weight.type_index >= compiled_mutation_types_.size())
                        throw std::invalid_argument("mutation type index out of range");
                    sum += weight.weight;
                    region.mutation_type_indices.push_back(weight.type_index);
                    region.mutation_type_cdf.push_back(sum);
                }
                if (sum <= 0.0)
                    throw std::invalid_argument("region type weights must sum to > 0");
            }

            compiled_mutation_regions_.push_back(std::move(region));
        };

    // Compile the mutation regions
    if (has_mutation_regions) {
        compiled_mutation_regions_.reserve(mutation_region_count);
        for (size_t chromosome_index = 0; chromosome_index < p_.chromosomes.size(); ++chromosome_index) {
            for (const ChromosomeRegionSpec& placement : p_.chromosomes[chromosome_index].regions) {
                if (placement.region_type_index >= resolved_region_types.size())
                    throw std::invalid_argument("region type index out of range");
                const RegionTypeSpec& region_type =
                    resolved_region_types[placement.region_type_index];
                append_compiled_region((uint32_t)chromosome_index,
                                       placement.start,
                                       placement.end,
                                       region_type.mutation_scale,
                                       region_type.mutation_types);
            }
        }
    } else {
        // If no mutation regions are specified, use the default region type
        if (p_.mutation_region_types.size() > 1u) {
            throw std::invalid_argument(
                "multiple mutation_region_types require chromosome-local regions");
        }

        compiled_mutation_regions_.reserve(compiled_chromosomes_.size());
        const RegionTypeSpec& default_region_type = resolved_region_types[0];
        for (size_t chromosome_index = 0;
             chromosome_index < compiled_chromosomes_.size();
             ++chromosome_index) {
            append_compiled_region((uint32_t)chromosome_index,
                                   0u,
                                   compiled_chromosomes_[chromosome_index].length,
                                   default_region_type.mutation_scale,
                                   default_region_type.mutation_types);
        }
    }

    // Compute the total mutation mass and the mutation region CDF
    total_mutation_mass_ = 0.0;
    mutation_region_cdf_.clear();
    for (const CompiledMutationRegion& region : compiled_mutation_regions_) {
        total_mutation_mass_ += region.mutation_mass;
        mutation_region_cdf_.push_back(total_mutation_mass_);
    }

    // Compute the mutation lambda and the exponential of the negative mutation lambda
    mutation_lambda_ = p_.mu * total_mutation_mass_;
    use_fast_mutation_poisson_ = (mutation_lambda_ > 0.0 && mutation_lambda_ < 30.0);
    exp_neg_mutation_lambda_ = use_fast_mutation_poisson_ ? std::exp(-mutation_lambda_) : 1.0;

    // Compute the crossover lambda and the exponential of the negative crossover lambda for each chromosome
    for (CompiledChromosome& chromosome : compiled_chromosomes_) {
        chromosome.use_fast_poisson =
            (chromosome.crossover_lambda > 0.0 && chromosome.crossover_lambda < 30.0);
        chromosome.exp_neg_crossover_lambda =
            chromosome.use_fast_poisson ? std::exp(-chromosome.crossover_lambda) : 1.0;
    }
}
