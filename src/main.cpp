// sparqy — CLI entry points for the packed mutation-selection-recombination model.
//
// Modes:
//   1. Legacy positional mode for the simple one-chromosome benchmark model:
//      sparqy [N] [L] [mu] [rho] [s] [G] [out_interval] [h] [seed] [threads] [--stats=LIST]
//
//   2. Full config mode for the complete SimParams surface:
//      sparqy --config path/to/model.sparqy

#include "config_loader.hpp"
#include "sparqy.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace {

struct StatName {
    const char* name;
    StatisticKind kind;
};
const StatName kStatNames[] = {
    {"mean_fitness",             StatisticKind::mean_fitness},
    {"genetic_load",             StatisticKind::genetic_load},
    {"realized_masking_bonus",   StatisticKind::realized_masking_bonus},
    {"exact_B",                  StatisticKind::exact_B},
    {"pairwise_similarity",      StatisticKind::mean_pairwise_haplotypic_similarity},
    {"n_seg",                    StatisticKind::n_seg},
    {"n_fixed",                  StatisticKind::n_fixed},
    {"genome_words",             StatisticKind::genome_words},
    {"mutation_histogram",       StatisticKind::mutation_histogram},
    {"site_frequency_spectrum",  StatisticKind::site_frequency_spectrum},
    {"nucleotide_diversity",     StatisticKind::nucleotide_diversity},
    {"expected_heterozygosity",  StatisticKind::expected_heterozygosity},
};

struct ProfileField {
    const char* name;
    double GenerationProfileSnapshot::*member;
};

const ProfileField kProfileFields[] = {
    {"reserve_mutation_metadata", &GenerationProfileSnapshot::reserve_mutation_metadata_sec},
    {"build_parent_sampler", &GenerationProfileSnapshot::build_parent_sampler_sec},
    {"distribute_recyclable_ids", &GenerationProfileSnapshot::distribute_recyclable_ids_sec},
    {"parallel_reproduction", &GenerationProfileSnapshot::parallel_reproduction_sec},
    {"reclaim_recyclable_ids", &GenerationProfileSnapshot::reclaim_recyclable_ids_sec},
    {"collect_created_mutations", &GenerationProfileSnapshot::collect_created_mutations_sec},
    {"offspring_prefix_sum", &GenerationProfileSnapshot::offspring_prefix_sum_sec},
    {"offspring_offset_adjust", &GenerationProfileSnapshot::offspring_offset_adjust_sec},
    {"offspring_copy_and_count", &GenerationProfileSnapshot::offspring_copy_and_count_sec},
    {"zero_offspring_counts", &GenerationProfileSnapshot::zero_offspring_counts_sec},
    {"merge_thread_counts", &GenerationProfileSnapshot::merge_thread_counts_sec},
    {"classify_mutations", &GenerationProfileSnapshot::classify_mutations_sec},
    {"collect_finalize_buffers", &GenerationProfileSnapshot::collect_finalize_buffers_sec},
    {"adjust_fixed_fitness", &GenerationProfileSnapshot::adjust_fixed_fitness_sec},
    {"statistics", &GenerationProfileSnapshot::statistics_sec},
    {"swap_populations", &GenerationProfileSnapshot::swap_populations_sec},
    {"total", &GenerationProfileSnapshot::total_sec},
};

struct ProfileAccumulator {
    uint64_t samples = 0;
    GenerationProfileSnapshot totals;

    void add(const GenerationProfileSnapshot& snapshot) {
        ++samples;
        for (const ProfileField& field : kProfileFields) {
            totals.*(field.member) += snapshot.*(field.member);
        }
    }
};

const char* statistic_kind_name(StatisticKind kind) {
    for (const StatName& sn : kStatNames) {
        if (sn.kind == kind) return sn.name;
    }
    return "unknown";
}

const char* similarity_metric_name(HaplotypeSimilarityMetric metric) {
    switch (metric) {
        case HaplotypeSimilarityMetric::none:    return "";
        case HaplotypeSimilarityMetric::jaccard: return "jaccard";
        case HaplotypeSimilarityMetric::dice:    return "dice";
        case HaplotypeSimilarityMetric::overlap: return "overlap";
    }
    return "";
}

const char* parent_sampler_build_mode_name(ParentSamplerBuildMode mode) {
    switch (mode) {
        case ParentSamplerBuildMode::automatic:         return "auto";
        case ParentSamplerBuildMode::sequential:        return "sequential";
        case ParentSamplerBuildMode::parallel:          return "parallel";
        case ParentSamplerBuildMode::parallel_psa_plus: return "parallel_psa_plus";
    }
    return "unknown";
}

ParentSamplerBuildMode parse_parent_sampler_build_mode_value(
    const std::string& value) {
    if (value == "auto")              return ParentSamplerBuildMode::automatic;
    if (value == "sequential")        return ParentSamplerBuildMode::sequential;
    if (value == "parallel")          return ParentSamplerBuildMode::parallel;
    if (value == "parallel_psa_plus") return ParentSamplerBuildMode::parallel_psa_plus;
    throw std::runtime_error(
        "sparqy: --alias-builder must be 'auto', 'sequential', 'parallel', or 'parallel_psa_plus'");
}

bool has_parent_sampler_build_mode_flag(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--alias-builder") return true;
        if (arg.rfind("--alias-builder=", 0) == 0) return true;
    }
    return false;
}

ParentSamplerBuildMode find_parent_sampler_build_mode(int argc, char** argv) {
    ParentSamplerBuildMode mode = ParentSamplerBuildMode::automatic;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--alias-builder") {
            if (i + 1 >= argc) {
                throw std::runtime_error(
                    "sparqy: --alias-builder requires a value");
            }
            mode = parse_parent_sampler_build_mode_value(argv[++i]);
            continue;
        }
        const std::string prefix = "--alias-builder=";
        if (arg.rfind(prefix, 0) == 0) {
            mode = parse_parent_sampler_build_mode_value(
                arg.substr(prefix.size()));
        }
    }
    return mode;
}

// Parse a --stats=LIST argument into a set of requested kinds.
// Returns false on unknown stat name.
bool parse_stats_flag(const std::string& arg,
                      std::unordered_set<int>& out_kinds) {
    const std::string prefix = "--stats=";
    if (arg.rfind(prefix, 0) != 0) return true;  // not a stats flag, ignore
    const std::string list = arg.substr(prefix.size());

    size_t pos = 0;
    while (pos <= list.size()) {
        size_t comma = list.find(',', pos);
        if (comma == std::string::npos) comma = list.size();
        std::string name = list.substr(pos, comma - pos);
        pos = comma + 1;
        if (name.empty()) continue;

        if (name == "all") {
            for (const StatName& sn : kStatNames)
                out_kinds.insert((int)sn.kind);
            continue;
        }
        bool matched = false;
        for (const StatName& sn : kStatNames) {
            if (name == sn.name) {
                out_kinds.insert((int)sn.kind);
                matched = true;
                break;
            }
        }
        if (!matched) {
            std::fprintf(stderr, "sparqy: unknown stat name '%s'\n", name.c_str());
            return false;
        }
    }
    return true;
}

std::string join_uint64_vector(const std::vector<uint64_t>& values) {
    std::string out;
    for (size_t i = 0; i < values.size(); ++i) {
        if (i != 0u) out.push_back(',');
        out += std::to_string(values[i]);
    }
    return out;
}

std::string format_double(double value) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.10f", value);
    return std::string(buf);
}

std::string format_uint64(uint64_t value) {
    return std::to_string((unsigned long long)value);
}

bool has_profile_flag(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--profile") == 0) return true;
    }
    return false;
}

bool has_stats_flag(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg.rfind("--stats=", 0) == 0) return true;
    }
    return false;
}

void print_profile_summary(const ProfileAccumulator& accumulator,
                           FILE* stream) {
    if (accumulator.samples == 0u) return;

    const double total = accumulator.totals.total_sec;
    std::fprintf(stream,
                 "Profiler summary across %llu generations:\n",
                 (unsigned long long)accumulator.samples);
    for (const ProfileField& field : kProfileFields) {
        const double total_sec = accumulator.totals.*(field.member);
        const double avg_sec = total_sec / (double)accumulator.samples;
        const double pct = (total > 0.0) ? (100.0 * total_sec / total) : 0.0;
        std::fprintf(stream,
                     "  %-26s total=%10.6f  avg/gen=%10.6f  pct=%6.2f%%\n",
                     field.name,
                     total_sec,
                     avg_sec,
                     pct);
    }
}

void print_usage() {
    std::fputs(
        "Usage:\n"
        "  sparqy [N] [L] [mu] [rho] [s] [G] [out_interval] [h] [seed] [threads] [--stats=LIST] [--profile] [--alias-builder=MODE]\n"
        "  sparqy --config MODEL.sparqy [--profile] [--alias-builder=MODE]\n"
        "\n"
        "Legacy positional mode builds a single chromosome with one mutation type.\n"
        "Config mode exposes the full SimParams model surface through an R-style config file.\n"
        "Legacy mode emits scalar and pairwise-similarity summaries; use config mode\n"
        "for histogram and site-frequency-spectrum outputs.\n"
        "Config mode does not accept --stats=LIST; declare stats in the config file.\n"
        "Profiling summary is printed to stderr so stdout stays machine-readable.\n"
        "Alias builder modes: auto (default), sequential, parallel, parallel_psa_plus.\n"
        "Auto chooses sequential for single-thread or N < 10000 runs, and\n"
        "parallel_psa_plus otherwise.\n"
        "\n"
        "Config structure:\n"
        "  constants <- list(name = value, ...)\n"
        "  config <- list(\n"
        "      N = ..., G = ..., mu = ..., rho = ..., seed = ..., threads = ...,\n"
        "      runtime = list(alias_builder = ..., profile = ...),\n"
        "      mutation_types = list(name = list(selection = gamma(...), dominance = additive()), ...),\n"
        "      region_types = list(name = list(mutation_scale = ..., weights = c(mt = weight, ...)), ...),\n"
        "      chromosomes = list(name = list(length = ..., recombination_intervals = list(interval(...), ...),\n"
        "                                     regions = list(region(...), ...)), ...))\n"
        "  stats <- list(always(stat, ...), every(step, stat, ...), at(c(g1, g2), stat, ...),\n"
        "                up_to(end, stat, ...), at_after(start, stat, ...),\n"
        "                every_up_to(step, end, stat, ...), every_at_after(step, start, stat, ...),\n"
        "                range(start, end, stat, ...), every_range(step, start, end, stat, ...))\n"
        "\n"
        "Builder calls:\n"
        "  constant(value), uniform(min, max), normal(mean, sd, min, max),\n"
        "  exponential(mean, min, max), gamma(mean, shape, min, max), beta(alpha, beta, min, max)\n"
        "  additive(), fixed(h), distributed(beta(...)), linear_from_s(intercept, slope, min_h, max_h)\n"
        "  interval(start, end, rate_scale), region(region_type, start, end)\n"
        "See CONFIG_REFERENCE.md for the complete config reference.\n",
        stdout);
}

bool find_config_path(int argc, char** argv, std::string& config_path) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--config") {
            if (i + 1 >= argc)
                throw std::runtime_error("sparqy: --config requires a path");
            config_path = argv[i + 1];
            return true;
        }
        const std::string prefix = "--config=";
        if (arg.rfind(prefix, 0) == 0) {
            config_path = arg.substr(prefix.size());
            if (config_path.empty())
                throw std::runtime_error("sparqy: --config requires a path");
            return true;
        }
    }
    return false;
}

int run_legacy_mode(int argc,
                    char** argv,
                    bool enable_profile,
                    ParentSamplerBuildMode requested_parent_sampler_build_mode) {
    SimParams p;
    p.enable_profiling = enable_profile;
    p.parent_sampler_build_mode = requested_parent_sampler_build_mode;

    std::vector<std::string> positional_args;

    std::unordered_set<int> requested_kinds;
    for (int i = 1; i < argc; i++) {
        const std::string arg = argv[i];
        if (arg == "--profile") continue;
        if (arg == "--alias-builder") {
            ++i;
            continue;
        }
        if (arg.rfind("--alias-builder=", 0) == 0) continue;
        if (arg.rfind("--stats=", 0) == 0) {
            if (!parse_stats_flag(arg, requested_kinds)) return 1;
            continue;
        }
        positional_args.push_back(arg);
    }

    p.N       = (positional_args.size() > 0u) ? std::atoi(positional_args[0].c_str()) : 1000;
    const int L = (positional_args.size() > 1u) ? std::atoi(positional_args[1].c_str()) : 100000;
    p.mu      = (positional_args.size() > 2u) ? std::atof(positional_args[2].c_str()) : 1e-7;
    p.rho     = (positional_args.size() > 3u) ? std::atof(positional_args[3].c_str()) : 0.01;
    const double s = (positional_args.size() > 4u) ? std::atof(positional_args[4].c_str()) : -0.01;
    p.G       = (positional_args.size() > 5u) ? std::atoi(positional_args[5].c_str()) : 100;
    int out_interval = (positional_args.size() > 6u) ? std::atoi(positional_args[6].c_str()) : 10;
    const double h = (positional_args.size() > 7u) ? std::atof(positional_args[7].c_str()) : 0.5;
    p.seed    = (positional_args.size() > 8u) ? std::atoll(positional_args[8].c_str()) : 42;
    p.threads = (positional_args.size() > 9u) ? std::atoi(positional_args[9].c_str()) : 0;
    const bool any_stats = !requested_kinds.empty();
    const int effective_threads = resolve_simulation_thread_count(p.threads);
    p.parent_sampler_build_mode = resolve_parent_sampler_build_mode(
        p.parent_sampler_build_mode, p.N, effective_threads);

    for (int kind_int : requested_kinds) {
        if (kind_int == (int)StatisticKind::mutation_histogram
            || kind_int == (int)StatisticKind::site_frequency_spectrum) {
            std::fputs(
                "sparqy: mutation_histogram and site_frequency_spectrum require --config mode\n",
                stderr);
            return 1;
        }
    }

    MutationTypeSpec mutation_type;
    mutation_type.selection = {DistKind::constant, s, 0.0, -0.999999999, 1.0};
    mutation_type.dominance =
        (h == AdditiveDominanceSpec::additive_h)
            ? DominanceSpec::additive()
            : DominanceSpec::fixed(h);
    p.mutation_types.push_back(mutation_type);

    p.mutation_region_types.push_back(RegionTypeSpec{});

    ChromosomeSpec chromosome;
    chromosome.length = (uint32_t)L;
    chromosome.regions.push_back({0u, 0u, (uint32_t)L});
    p.chromosomes.push_back(std::move(chromosome));

    std::vector<uint64_t> report_generations;
    if (any_stats) {
        report_generations.reserve(
            (size_t)std::max(1, p.G / std::max(1, out_interval) + 1));
        for (int generation = 1; generation <= p.G; generation++) {
            if ((generation % out_interval) == 0 || generation == p.G) {
                report_generations.push_back((uint64_t)generation);
            }
        }
        for (int kind_int : requested_kinds) {
            StatisticRequest request;
            request.kind = (StatisticKind)kind_int;
            request.generations = report_generations;
            if (request.kind == StatisticKind::mean_pairwise_haplotypic_similarity) {
                request.similarity_metric = HaplotypeSimilarityMetric::jaccard;
            }
            p.statistic_requests.push_back(std::move(request));
        }
    }

    std::fprintf(stderr,
        "sparqy: N=%d L=%d mu=%.2e rho=%.4f s=%.4f h=%.2f G=%d seed=%llu threads=%d stats=%s profile=%s alias_builder=%s\n",
        p.N, L, p.mu, p.rho, s, h, p.G, (unsigned long long)p.seed, p.threads,
        any_stats ? "on" : "off",
        p.enable_profiling ? "on" : "off",
        parent_sampler_build_mode_name(p.parent_sampler_build_mode));

    Simulator sim(p);
    ProfileAccumulator profile_accumulator;

    if (any_stats) {
        std::puts("gen,dt_sec,cumul_sec,meanFitness,geneticLoad,realizedMaskingBonus,"
                  "exactB,meanPairwiseHaplotypicSimilarity,numSeg,numFixed,genomeWords,"
                  "nucleotideDiversity,expectedHeterozygosity");
    }

    auto t0 = std::chrono::steady_clock::now();
    auto t_prev = t0;

    for (int g = 0; g < p.G; g++) {
        sim.step();
        if (p.enable_profiling) profile_accumulator.add(sim.latest_profile());

        if (!any_stats) continue;
        if (((g + 1) % out_interval) != 0 && (g + 1) != p.G) continue;

        auto now = std::chrono::steady_clock::now();
        double dt    = std::chrono::duration<double>(now - t_prev).count();
        double cumul = std::chrono::duration<double>(now - t0).count();
        t_prev = now;
        const StatisticsSnapshot& snapshot = sim.latest_statistics();

        constexpr size_t kNumKinds = 12;
        double  dvals[kNumKinds];
        uint64_t uvals[kNumKinds];
        for (size_t i = 0; i < kNumKinds; i++) { dvals[i] = 0.0; uvals[i] = 0; }
        double jaccard_sim = 0.0;
        const size_t n = snapshot.statistics.size();
        for (size_t i = 0; i < n; i++) {
            const ComputedStatistic& stat = snapshot.statistics[i];
            const size_t k = (size_t)(uint8_t)stat.kind;
            if (k >= kNumKinds) continue;
            if (stat.kind == StatisticKind::mean_pairwise_haplotypic_similarity) {
                if (stat.similarity_metric == HaplotypeSimilarityMetric::jaccard) {
                    jaccard_sim = std::get<double>(stat.value);
                }
                continue;
            }
            if (stat.kind == StatisticKind::mutation_histogram) continue;
            if (stat.kind == StatisticKind::site_frequency_spectrum) continue;
            if (stat.value.index() == 0)      dvals[k] = std::get<double>(stat.value);
            else if (stat.value.index() == 1) uvals[k] = std::get<uint64_t>(stat.value);
        }

        std::printf("%d,%.6f,%.6f,%.8f,%.8f,%.10f,%.10f,%.10f,"
                    "%llu,%llu,%llu,%.10f,%.10f\n",
                    (int)snapshot.generation, dt, cumul,
                    dvals[(size_t)StatisticKind::mean_fitness],
                    dvals[(size_t)StatisticKind::genetic_load],
                    dvals[(size_t)StatisticKind::realized_masking_bonus],
                    dvals[(size_t)StatisticKind::exact_B],
                    jaccard_sim,
                    (unsigned long long)uvals[(size_t)StatisticKind::n_seg],
                    (unsigned long long)uvals[(size_t)StatisticKind::n_fixed],
                    (unsigned long long)uvals[(size_t)StatisticKind::genome_words],
                    dvals[(size_t)StatisticKind::nucleotide_diversity],
                    dvals[(size_t)StatisticKind::expected_heterozygosity]);
    }

    std::fflush(stdout);
    auto t1 = std::chrono::steady_clock::now();
    double total = std::chrono::duration<double>(t1 - t0).count();
    std::fprintf(stderr, "Total wall-clock: %.6f sec\nAvg sec/gen: %.6f\n",
                 total, total / p.G);
    if (p.enable_profiling) {
        print_profile_summary(profile_accumulator, stderr);
    }
    return 0;
}

int run_config_mode(const std::string& config_path,
                    bool enable_profile,
                    bool cli_has_alias_builder,
                    ParentSamplerBuildMode requested_parent_sampler_build_mode) {
    const LoadedConfig loaded = load_config_file(config_path);
    SimParams p = loaded.params;
    p.enable_profiling = p.enable_profiling || enable_profile;
    if (cli_has_alias_builder) {
        p.parent_sampler_build_mode = requested_parent_sampler_build_mode;
    }
    const int effective_threads = resolve_simulation_thread_count(p.threads);
    p.parent_sampler_build_mode = resolve_parent_sampler_build_mode(
        p.parent_sampler_build_mode, p.N, effective_threads);

    for (const std::string& warning : loaded.warnings) {
        std::fprintf(stderr, "sparqy: warning: %s\n", warning.c_str());
    }

    std::fprintf(stderr,
                 "sparqy: config=%s N=%d G=%d mu=%.2e rho=%.4f seed=%llu threads=%d"
                 " mutation_types=%zu region_types=%zu chromosomes=%zu stats=%s profile=%s alias_builder=%s\n",
                 config_path.c_str(),
                 p.N,
                 p.G,
                 p.mu,
                 p.rho,
                 (unsigned long long)p.seed,
                 p.threads,
                 p.mutation_types.size(),
                 p.mutation_region_types.size(),
                 p.chromosomes.size(),
                 loaded.has_statistics ? "on" : "off",
                 p.enable_profiling ? "on" : "off",
                 parent_sampler_build_mode_name(p.parent_sampler_build_mode));

    Simulator sim(p);
    ProfileAccumulator profile_accumulator;
    if (loaded.has_statistics) {
        std::puts("generation\tdt_sec\tcumul_sec\tstatistic\tmetric\tscalar_value\tby_type\tby_chromosome\tunfolded_sfs\tfolded_sfs");
    }

    auto t0 = std::chrono::steady_clock::now();
    auto t_prev = t0;
    for (int g = 0; g < p.G; ++g) {
        sim.step();
        if (p.enable_profiling) profile_accumulator.add(sim.latest_profile());

        if (!loaded.has_statistics) continue;

        auto now = std::chrono::steady_clock::now();
        const StatisticsSnapshot& snapshot = sim.latest_statistics();
        if (snapshot.statistics.empty()) continue;

        double dt = std::chrono::duration<double>(now - t_prev).count();
        double cumul = std::chrono::duration<double>(now - t0).count();
        t_prev = now;

        for (const ComputedStatistic& stat : snapshot.statistics) {
            std::string scalar_value;
            std::string by_type;
            std::string by_chromosome;
            std::string unfolded_sfs;
            std::string folded_sfs;

            if (stat.value.index() == 0u) {
                scalar_value = format_double(std::get<double>(stat.value));
            } else if (stat.value.index() == 1u) {
                scalar_value = format_uint64(std::get<uint64_t>(stat.value));
            } else if (stat.kind == StatisticKind::mutation_histogram) {
                const MutationHistogram& hist = std::get<MutationHistogram>(stat.value);
                by_type = join_uint64_vector(hist.by_type);
                by_chromosome = join_uint64_vector(hist.by_chromosome);
            } else if (stat.kind == StatisticKind::site_frequency_spectrum) {
                const SiteFrequencySpectrum& sfs = std::get<SiteFrequencySpectrum>(stat.value);
                unfolded_sfs = join_uint64_vector(sfs.unfolded_by_copy_number);
                folded_sfs = join_uint64_vector(sfs.folded_by_minor_allele_count);
            }

            std::printf("%llu\t%.6f\t%.6f\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n",
                        (unsigned long long)snapshot.generation,
                        dt,
                        cumul,
                        statistic_kind_name(stat.kind),
                        similarity_metric_name(stat.similarity_metric),
                        scalar_value.c_str(),
                        by_type.c_str(),
                        by_chromosome.c_str(),
                        unfolded_sfs.c_str(),
                        folded_sfs.c_str());
        }
    }

    std::fflush(stdout);
    auto t1 = std::chrono::steady_clock::now();
    double total = std::chrono::duration<double>(t1 - t0).count();
    std::fprintf(stderr, "Total wall-clock: %.6f sec\nAvg sec/gen: %.6f\n",
                 total, total / p.G);
    if (p.enable_profiling) {
        print_profile_summary(profile_accumulator, stderr);
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc > 1 && (std::strcmp(argv[1], "--help") == 0
                         || std::strcmp(argv[1], "-h") == 0)) {
            print_usage();
            return 0;
        }

        const bool enable_profile = has_profile_flag(argc, argv);
        const bool cli_has_alias_builder = has_parent_sampler_build_mode_flag(argc, argv);
        const ParentSamplerBuildMode parent_sampler_build_mode =
            find_parent_sampler_build_mode(argc, argv);
        std::string config_path;
        if (find_config_path(argc, argv, config_path)) {
            if (has_stats_flag(argc, argv)) {
                throw std::runtime_error(
                    "sparqy: --stats=LIST is only supported in legacy positional mode; declare stats in the config file");
            }
            return run_config_mode(
                config_path, enable_profile, cli_has_alias_builder, parent_sampler_build_mode);
        }

        return run_legacy_mode(
            argc, argv, enable_profile, parent_sampler_build_mode);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
}
