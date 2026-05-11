#include "sparqy.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

struct ImportOnlyMutationType {
    uint32_t    slim_type_id = 0;
    uint32_t    base_type_index = 0;
    std::string dominance_text;
};

struct StandingMutationExportRecord {
    uint32_t    mutation_id = 0;
    uint32_t    chromosome_index = 0;
    uint32_t    position = 0;
    uint32_t    slim_type_id = 0;
    uint32_t    copy_count = 0;
    uint64_t    origin_generation = 0;
    std::string selection_text;
    std::string dominance_text;
};

struct SlimExportPreparation {
    std::vector<std::vector<uint32_t>> chromosome_mutation_ids;
    std::vector<std::unordered_map<uint32_t, uint32_t>> polymorphism_ids_by_mutation;
    std::unordered_map<uint32_t, StandingMutationExportRecord> records_by_mutation_id;
    std::vector<ImportOnlyMutationType> import_only_types;
    uint32_t import_only_mutation_type_count = 0;
    uint32_t zero_selection_dominance_fallback_count = 0;
};

bool is_default_min_bound(double value) {
    return value <= -std::numeric_limits<double>::max() / 2.0;
}

bool is_default_max_bound(double value) {
    return value >= std::numeric_limits<double>::max() / 2.0;
}

std::string format_slim_number(double value) {
    if (std::abs(value) < 1e-15) value = 0.0;
    char buffer[64];
    std::snprintf(buffer, sizeof(buffer), "%.17g", value);
    return std::string(buffer);
}

std::string format_slim_integer(uint64_t value) {
    return std::to_string((unsigned long long)value);
}

std::string escape_eidos_string(const std::string& value) {
    std::string escaped;
    escaped.reserve(value.size());
    for (char c : value) {
        if (c == '\\' || c == '"') escaped.push_back('\\');
        escaped.push_back(c);
    }
    return escaped;
}

std::string maybe_clamp_expression(const std::string& expr,
                                   double min_value,
                                   double max_value) {
    const bool has_min = !is_default_min_bound(min_value);
    const bool has_max = !is_default_max_bound(max_value);
    if (has_min && has_max) {
        return "pmax(" + format_slim_number(min_value)
             + ", pmin(" + format_slim_number(max_value)
             + ", " + expr + "))";
    }
    if (has_min) {
        return "pmax(" + format_slim_number(min_value) + ", " + expr + ")";
    }
    if (has_max) {
        return "pmin(" + format_slim_number(max_value) + ", " + expr + ")";
    }
    return expr;
}

std::string dist_draw_expression(const DistSpec& dist) {
    switch (dist.kind) {
        case DistKind::constant:
            return format_slim_number(dist.p1);
        case DistKind::uniform:
            return maybe_clamp_expression(
                "runif(1, " + format_slim_number(dist.p1)
                    + ", " + format_slim_number(dist.p2) + ")",
                dist.min_value,
                dist.max_value);
        case DistKind::normal:
            return maybe_clamp_expression(
                "rnorm(1, " + format_slim_number(dist.p1)
                    + ", " + format_slim_number(dist.p2) + ")",
                dist.min_value,
                dist.max_value);
        case DistKind::exponential:
            return maybe_clamp_expression(
                "rexp(1, " + format_slim_number(dist.p1) + ")",
                dist.min_value,
                dist.max_value);
        case DistKind::gamma: {
            std::string expr = "0.0";
            if (dist.p2 > 0.0 && dist.p1 != 0.0) {
                expr = std::string(dist.p1 < 0.0 ? "-" : "")
                     + "rgamma(1, " + format_slim_number(std::abs(dist.p1))
                     + ", " + format_slim_number(dist.p2) + ")";
            }
            return maybe_clamp_expression(expr, dist.min_value, dist.max_value);
        }
        case DistKind::beta:
            return maybe_clamp_expression(
                "rbeta(1, " + format_slim_number(dist.p1)
                    + ", " + format_slim_number(dist.p2) + ")",
                dist.min_value,
                dist.max_value);
    }
    return format_slim_number(dist.p1);
}

double dist_expected_value(const DistSpec& dist) {
    double value = dist.p1;
    switch (dist.kind) {
        case DistKind::constant:
            value = dist.p1;
            break;
        case DistKind::uniform:
            value = 0.5 * (dist.p1 + dist.p2);
            break;
        case DistKind::normal:
            value = dist.p1;
            break;
        case DistKind::exponential:
            value = dist.p1;
            break;
        case DistKind::gamma:
            value = dist.p1;
            break;
        case DistKind::beta:
            value = (dist.p1 + dist.p2 > 0.0)
                        ? (dist.p1 / (dist.p1 + dist.p2))
                        : 0.5;
            break;
    }
    return clamp_value(value, dist.min_value, dist.max_value);
}

double base_mutation_type_dominance(const DominanceSpec& dominance) {
    if (std::holds_alternative<AdditiveDominanceSpec>(dominance.model)) {
        return AdditiveDominanceSpec::additive_h;
    }
    if (const auto* fixed = std::get_if<FixedDominanceSpec>(&dominance.model)) {
        return fixed->h;
    }
    return AdditiveDominanceSpec::additive_h;
}

bool requires_import_only_type(const DominanceSpec& dominance) {
    return std::holds_alternative<DistributedDominanceSpec>(dominance.model)
        || std::holds_alternative<LinearDominanceFromSelectionSpec>(dominance.model);
}

double exported_dominance_for_mutation(const MutationTypeSpec& mutation_type,
                                       double selection_coeff,
                                       double heterozygous_fitness_factor,
                                       uint32_t& fallback_counter) {
    if (std::holds_alternative<AdditiveDominanceSpec>(mutation_type.dominance.model)) {
        return AdditiveDominanceSpec::additive_h;
    }
    if (const auto* fixed = std::get_if<FixedDominanceSpec>(&mutation_type.dominance.model)) {
        return fixed->h;
    }
    if (const auto* linear =
            std::get_if<LinearDominanceFromSelectionSpec>(&mutation_type.dominance.model)) {
        return clamp_value(linear->intercept + linear->slope * selection_coeff,
                           linear->min_h,
                           linear->max_h);
    }

    if (std::abs(selection_coeff) > 1e-15) {
        return (heterozygous_fitness_factor - 1.0) / selection_coeff;
    }

    ++fallback_counter;
    const auto& distributed =
        std::get<DistributedDominanceSpec>(mutation_type.dominance.model);
    return dist_expected_value(distributed.distribution);
}

std::string selection_dfe_argument(const DistSpec& selection) {
    if (selection.kind == DistKind::constant) {
        return format_slim_number(selection.p1);
    }
    return "\"" + escape_eidos_string("return " + dist_draw_expression(selection) + ";") + "\"";
}

const char* selection_dfe_mode(const DistSpec& selection) {
    return (selection.kind == DistKind::constant) ? "f" : "s";
}

std::string linear_dominance_expression(const LinearDominanceFromSelectionSpec& linear) {
    return maybe_clamp_expression(
        format_slim_number(linear.intercept)
            + " + " + format_slim_number(linear.slope) + " * mut.selectionCoeff",
        linear.min_h,
        linear.max_h);
}

std::string join_string_vector(const std::vector<std::string>& values) {
    std::string joined;
    for (size_t i = 0; i < values.size(); ++i) {
        if (i != 0u) joined += ", ";
        joined += values[i];
    }
    return joined;
}

std::filesystem::path absolute_prefix_path(const std::string& output_prefix) {
    if (output_prefix.empty()) {
        throw std::runtime_error("sparqy: --export-slim requires a non-empty prefix");
    }
    return std::filesystem::absolute(std::filesystem::path(output_prefix));
}

}  // namespace

SlimExportResult Simulator::export_state_for_slim(const std::string& output_prefix) const {
    const std::filesystem::path prefix_path = absolute_prefix_path(output_prefix);
    const std::filesystem::path parent_path = prefix_path.parent_path();
    if (!parent_path.empty()) {
        std::filesystem::create_directories(parent_path);
    }

    const std::string population_path = prefix_path.string() + ".txt";
    const std::string loader_script_path = prefix_path.string() + ".slim";

    SlimExportPreparation prep;
    prep.chromosome_mutation_ids.resize(compiled_chromosomes_.size());
    prep.polymorphism_ids_by_mutation.resize(compiled_chromosomes_.size());
    prep.records_by_mutation_id.reserve(segregating_mutation_ids_.size());

    for (uint32_t mutation_id : segregating_mutation_ids_) {
        const uint32_t chromosome_index = locus_chrom(mutation_loci_[mutation_id]);
        prep.chromosome_mutation_ids[chromosome_index].push_back(mutation_id);
    }

    for (std::vector<uint32_t>& chromosome_mutations : prep.chromosome_mutation_ids) {
        std::sort(chromosome_mutations.begin(),
                  chromosome_mutations.end(),
                  [this](uint32_t lhs, uint32_t rhs) {
                      return (mutation_loci_[lhs] < mutation_loci_[rhs])
                          || (mutation_loci_[lhs] == mutation_loci_[rhs] && lhs < rhs);
                  });
    }

    std::unordered_map<std::string, uint32_t> import_type_ids;
    import_type_ids.reserve(segregating_mutation_ids_.size());
    uint32_t next_import_type_id = (uint32_t)p_.mutation_types.size() + 1u;

    for (size_t chromosome_index = 0;
         chromosome_index < prep.chromosome_mutation_ids.size();
         ++chromosome_index) {
        auto& polymorphism_ids = prep.polymorphism_ids_by_mutation[chromosome_index];
        const auto& chromosome_mutations = prep.chromosome_mutation_ids[chromosome_index];
        polymorphism_ids.reserve(chromosome_mutations.size());

        for (size_t polymorphism_id = 0; polymorphism_id < chromosome_mutations.size(); ++polymorphism_id) {
            const uint32_t mutation_id = chromosome_mutations[polymorphism_id];
            const uint32_t base_type_index = mutation_type_index_[mutation_id];
            const MutationTypeSpec& mutation_type = p_.mutation_types[base_type_index];

            const double selection_coeff =
                mutation_homozygous_fitness_factor_[mutation_id] - 1.0;
            double dominance_coeff = base_mutation_type_dominance(mutation_type.dominance);
            uint32_t slim_type_id = base_type_index + 1u;

            if (requires_import_only_type(mutation_type.dominance)) {
                dominance_coeff = exported_dominance_for_mutation(
                    mutation_type,
                    selection_coeff,
                    mutation_heterozygous_fitness_factor_[mutation_id],
                    prep.zero_selection_dominance_fallback_count);
                const std::string dominance_key = format_slim_number(dominance_coeff);
                const std::string import_key =
                    std::to_string((unsigned long long)base_type_index) + ":" + dominance_key;
                auto found = import_type_ids.find(import_key);
                if (found == import_type_ids.end()) {
                    slim_type_id = next_import_type_id++;
                    import_type_ids.emplace(import_key, slim_type_id);
                    prep.import_only_types.push_back(
                        {slim_type_id, base_type_index, dominance_key});
                } else {
                    slim_type_id = found->second;
                }
            }

            polymorphism_ids.emplace(mutation_id, (uint32_t)polymorphism_id);
            prep.records_by_mutation_id.emplace(
                mutation_id,
                StandingMutationExportRecord{
                    mutation_id,
                    (uint32_t)chromosome_index,
                    locus_pos(mutation_loci_[mutation_id]),
                    slim_type_id,
                    parent_copy_counts_[mutation_id],
                    mutation_origin_generation_[mutation_id],
                    format_slim_number(selection_coeff),
                    format_slim_number(dominance_coeff)
                });
        }
    }

    prep.import_only_mutation_type_count = (uint32_t)prep.import_only_types.size();

    {
        std::ofstream out(population_path);
        if (!out.is_open()) {
            throw std::runtime_error(
                "sparqy: could not open '" + population_path + "' for SLiM population export");
        }

        out << "#OUT: " << format_slim_integer(completed_generations_)
            << " " << format_slim_integer(completed_generations_) << " A\n";
        out << "Version: 8\n";
        out << "Flags:\n";
        out << "Populations:\n";
        out << "p1 " << p_.N << " H\n";
        out << "Individuals:\n";
        for (int individual_index = 0; individual_index < p_.N; ++individual_index) {
            out << "p1:i" << individual_index << " H\n";
        }

        for (size_t chromosome_index = 0;
             chromosome_index < compiled_chromosomes_.size();
             ++chromosome_index) {
            const CompiledChromosome& chromosome = compiled_chromosomes_[chromosome_index];
            const uint32_t chromosome_id = (uint32_t)chromosome_index + 1u;
            out << "Chromosome: " << chromosome_index << " A " << chromosome_id
                << " " << (chromosome.length - 1u)
                << " \"" << chromosome_id << "\"\n";
            out << "Mutations:\n";

            const auto& chromosome_mutations = prep.chromosome_mutation_ids[chromosome_index];
            for (size_t polymorphism_id = 0; polymorphism_id < chromosome_mutations.size(); ++polymorphism_id) {
                const uint32_t mutation_id = chromosome_mutations[polymorphism_id];
                const StandingMutationExportRecord& record =
                    prep.records_by_mutation_id.at(mutation_id);
                out << polymorphism_id
                    << " " << record.mutation_id
                    << " m" << record.slim_type_id
                    << " " << record.position
                    << " " << record.selection_text
                    << " " << record.dominance_text
                    << " p1 " << format_slim_integer(record.origin_generation)
                    << " " << record.copy_count << "\n";
            }

            out << "Haplosomes:\n";
            const auto& polymorphism_ids = prep.polymorphism_ids_by_mutation[chromosome_index];
            for (int individual_index = 0; individual_index < p_.N; ++individual_index) {
                for (int hap_index = 0; hap_index < 2; ++hap_index) {
                    const uint32_t haplotype_index = (uint32_t)(2 * individual_index + hap_index);
                    const uint64_t start = parent_population_.haplotype_offsets[haplotype_index];
                    const uint64_t end = parent_population_.haplotype_offsets[haplotype_index + 1u];

                    out << "p1:i" << individual_index;
                    for (uint64_t cursor = start; cursor < end; ++cursor) {
                        const uint32_t mutation_id = parent_population_.mutation_ids[cursor];
                        if (locus_chrom(mutation_loci_[mutation_id]) != chromosome_index) continue;
                        out << " " << polymorphism_ids.at(mutation_id);
                    }
                    out << "\n";
                }
            }
        }
    }

    {
        std::ofstream out(loader_script_path);
        if (!out.is_open()) {
            throw std::runtime_error(
                "sparqy: could not open '" + loader_script_path + "' for SLiM loader export");
        }

        out << "// Generated by sparqy --export-slim.\n";
        out << "// This bootstrap recreates the sparqy model structure expected by\n";
        out << "// SLiM's readFromPopulationFile() and then loads the exported state.\n";
        out << "// Remove sim.simulationFinished() if you want to continue from the\n";
        out << "// imported state inside SLiM.\n\n";

        for (size_t type_index = 0; type_index < p_.mutation_types.size(); ++type_index) {
            const MutationTypeSpec& mutation_type = p_.mutation_types[type_index];
            const uint32_t slim_type_id = (uint32_t)type_index + 1u;

            if (const auto* distributed =
                    std::get_if<DistributedDominanceSpec>(&mutation_type.dominance.model)) {
                out << "mutation(m" << slim_type_id << ") {\n";
                out << "    mut.setValue(\"dom\", "
                    << dist_draw_expression(distributed->distribution) << ");\n";
                out << "    return T;\n";
                out << "}\n\n";
                out << "mutationEffect(m" << slim_type_id << ") {\n";
                out << "    if (homozygous)\n";
                out << "        return 1.0 + mut.selectionCoeff;\n";
                out << "    return 1.0 + mut.getValue(\"dom\") * mut.selectionCoeff;\n";
                out << "}\n\n";
            } else if (const auto* linear =
                           std::get_if<LinearDominanceFromSelectionSpec>(
                               &mutation_type.dominance.model)) {
                out << "mutationEffect(m" << slim_type_id << ") {\n";
                out << "    if (homozygous)\n";
                out << "        return 1.0 + mut.selectionCoeff;\n";
                out << "    return 1.0 + (" << linear_dominance_expression(*linear)
                    << ") * mut.selectionCoeff;\n";
                out << "}\n\n";
            }
        }

        out << "initialize() {\n";
        for (size_t type_index = 0; type_index < p_.mutation_types.size(); ++type_index) {
            const MutationTypeSpec& mutation_type = p_.mutation_types[type_index];
            const uint32_t slim_type_id = (uint32_t)type_index + 1u;
            out << "    initializeMutationType(\"m" << slim_type_id << "\", "
                << format_slim_number(base_mutation_type_dominance(mutation_type.dominance))
                << ", \"" << selection_dfe_mode(mutation_type.selection) << "\", "
                << selection_dfe_argument(mutation_type.selection) << ");\n";
        }

        if (!prep.import_only_types.empty()) {
            out << "\n";
            for (const ImportOnlyMutationType& import_type : prep.import_only_types) {
                out << "    initializeMutationType(\"m" << import_type.slim_type_id
                    << "\", " << import_type.dominance_text
                    << ", \"f\", 0.0);  // import-only standing variation for base m"
                    << (import_type.base_type_index + 1u) << "\n";
            }
        }

        out << "\n";
        std::vector<RegionTypeSpec> region_types = p_.mutation_region_types;
        if (region_types.empty()) region_types.push_back(RegionTypeSpec{});

        for (size_t region_type_index = 0; region_type_index < region_types.size(); ++region_type_index) {
            const RegionTypeSpec& region_type = region_types[region_type_index];
            std::vector<std::string> mutation_type_refs;
            std::vector<std::string> weights;

            if (region_type.mutation_types.empty()) {
                mutation_type_refs.push_back("m1");
                weights.push_back("1.0");
            } else {
                mutation_type_refs.reserve(region_type.mutation_types.size());
                weights.reserve(region_type.mutation_types.size());
                for (const MutationTypeWeight& weight : region_type.mutation_types) {
                    mutation_type_refs.push_back(
                        "m" + std::to_string((unsigned long long)weight.type_index + 1u));
                    weights.push_back(format_slim_number(weight.weight));
                }
            }

            out << "    initializeGenomicElementType(\"g" << (region_type_index + 1u) << "\", ";
            if (mutation_type_refs.size() == 1u) {
                out << mutation_type_refs[0] << ", " << weights[0] << ");\n";
            } else {
                out << "c(" << join_string_vector(mutation_type_refs) << "), c("
                    << join_string_vector(weights) << "));\n";
            }
        }

        double total_recombination_mass = 0.0;
        for (const CompiledChromosome& chromosome : compiled_chromosomes_) {
            total_recombination_mass += chromosome.recombination_mass;
        }
        const double recombination_denominator =
            (total_recombination_mass > 0.0) ? total_recombination_mass : 1.0;

        for (size_t chromosome_index = 0;
             chromosome_index < p_.chromosomes.size();
             ++chromosome_index) {
            const ChromosomeSpec& chromosome = p_.chromosomes[chromosome_index];
            out << "\n";
            out << "    initializeChromosome(" << (chromosome_index + 1u) << ", "
                << chromosome.length << ", type=\"A\");\n";

            std::vector<ChromosomeRegionSpec> regions = chromosome.regions;
            if (regions.empty()) {
                regions.push_back({0u, 0u, chromosome.length});
            }

            std::vector<std::string> mutation_rates;
            std::vector<std::string> mutation_ends;
            mutation_rates.reserve(regions.size());
            mutation_ends.reserve(regions.size());
            for (const ChromosomeRegionSpec& region : regions) {
                const RegionTypeSpec& region_type = region_types[region.region_type_index];
                mutation_rates.push_back(
                    format_slim_number(p_.mu * region_type.mutation_scale));
                mutation_ends.push_back(format_slim_integer((uint64_t)region.end - 1u));
            }

            if (mutation_rates.size() == 1u) {
                out << "    initializeMutationRate(" << mutation_rates[0] << ");\n";
            } else {
                out << "    initializeMutationRate(c(" << join_string_vector(mutation_rates)
                    << "), c(" << join_string_vector(mutation_ends) << "));\n";
            }

            for (const ChromosomeRegionSpec& region : regions) {
                out << "    initializeGenomicElement(g" << (region.region_type_index + 1u)
                    << ", " << region.start << ", " << (region.end - 1u) << ");\n";
            }

            std::vector<RecIntervalSpec> recombination_intervals = chromosome.recombination_map;
            if (recombination_intervals.empty()) {
                recombination_intervals.push_back({0u, chromosome.length, 1.0});
            }

            std::vector<std::string> recombination_rates;
            std::vector<std::string> recombination_ends;
            recombination_rates.reserve(recombination_intervals.size());
            recombination_ends.reserve(recombination_intervals.size());
            for (const RecIntervalSpec& interval : recombination_intervals) {
                const double rate = (total_recombination_mass > 0.0)
                                        ? (p_.rho * interval.rate_scale / recombination_denominator)
                                        : 0.0;
                recombination_rates.push_back(format_slim_number(rate));
                recombination_ends.push_back(format_slim_integer((uint64_t)interval.end - 1u));
            }

            if (recombination_rates.size() == 1u) {
                out << "    initializeRecombinationRate(" << recombination_rates[0] << ");\n";
            } else {
                out << "    initializeRecombinationRate(c("
                    << join_string_vector(recombination_rates) << "), c("
                    << join_string_vector(recombination_ends) << "));\n";
            }
        }
        out << "}\n\n";

        out << "1 late() {\n";
        out << "    sim.readFromPopulationFile(\""
            << escape_eidos_string(population_path) << "\");\n";
        out << "    totalIndividuals = 0;\n";
        out << "    for (subpop in sim.subpopulations)\n";
        out << "        totalIndividuals = totalIndividuals + size(subpop.individuals);\n";
        out << "    catn(\"sparqy_import_ok\\tgeneration=\" + sim.cycle\n";
        out << "         + \"\\tmutations=\" + size(sim.mutations)\n";
        out << "         + \"\\tindividuals=\" + totalIndividuals);\n";
        out << "    sim.simulationFinished();\n";
        out << "}\n";
    }

    SlimExportResult result;
    result.population_path = population_path;
    result.loader_script_path = loader_script_path;
    result.import_only_mutation_type_count = prep.import_only_mutation_type_count;
    result.zero_selection_dominance_fallback_count =
        prep.zero_selection_dominance_fallback_count;
    return result;
}
