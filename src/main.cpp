// sparqy — CLI entry points for the packed mutation-selection-recombination model.
//
// Modes:
//   1. Legacy positional mode for the simple one-chromosome benchmark model:
//      sparqy [N] [L] [mu] [rho] [s] [G] [out_interval] [h] [seed] [threads] [--stats=LIST]
//
//   2. Full config mode for the complete SimParams surface:
//      sparqy --config path/to/model.sparqy

#include "config_loader.hpp"
#include "sparqy_names.hpp"
#include "sparqy.hpp"

#include <algorithm>
#include <cmath>
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

ParentSamplerBuildMode parse_parent_sampler_build_mode_value(
    const std::string& value) {
    ParentSamplerBuildMode mode = ParentSamplerBuildMode::automatic;
    if (sparqy_names::try_parse_parent_sampler_build_mode_name(value, mode)) {
        return mode;
    }
    throw std::runtime_error(
        "sparqy: --alias-builder must be "
        + std::string(sparqy_names::kParentSamplerBuildModeChoicesForError));
}

void parse_stats_flag(const std::string& arg,
                      std::unordered_set<int>& out_kinds) {
    const std::string prefix = "--stats=";
    const std::string list = arg.substr(prefix.size());

    size_t pos = 0;
    while (pos <= list.size()) {
        size_t comma = list.find(',', pos);
        if (comma == std::string::npos) comma = list.size();
        std::string name = list.substr(pos, comma - pos);
        pos = comma + 1;
        if (name.empty()) continue;

        if (name == "all") {
            for (const auto& entry : sparqy_names::kStatisticKinds) {
                out_kinds.insert((int)entry.value);
            }
            continue;
        }
        StatisticKind kind = StatisticKind::mean_fitness;
        if (!sparqy_names::try_parse_statistic_kind_name(name, kind)) {
            throw std::runtime_error("sparqy: unknown stat name '" + name + "'");
        }
        out_kinds.insert((int)kind);
    }
}

struct ParsedCli {
    bool show_help = false;
    bool enable_profile = false;
    bool cli_has_alias_builder = false;
    ParentSamplerBuildMode parent_sampler_build_mode =
        ParentSamplerBuildMode::automatic;
    bool has_config = false;
    bool has_stats_flag = false;
    std::string config_path;
    std::string slim_export_prefix;
    std::vector<std::string> positional_args;
    std::vector<std::string> stats_flags;
};

ParsedCli parse_cli_args(int argc, char** argv) {
    ParsedCli parsed;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            parsed.show_help = true;
            continue;
        }
        if (arg == "--profile") {
            parsed.enable_profile = true;
            continue;
        }
        if (arg == "--config") {
            if (i + 1 >= argc) {
                throw std::runtime_error("sparqy: --config requires a path");
            }
            parsed.has_config = true;
            parsed.config_path = argv[++i];
            if (parsed.config_path.empty()) {
                throw std::runtime_error("sparqy: --config requires a path");
            }
            continue;
        }
        const std::string config_prefix = "--config=";
        if (arg.rfind(config_prefix, 0) == 0) {
            parsed.has_config = true;
            parsed.config_path = arg.substr(config_prefix.size());
            if (parsed.config_path.empty()) {
                throw std::runtime_error("sparqy: --config requires a path");
            }
            continue;
        }
        if (arg == "--alias-builder") {
            if (i + 1 >= argc) {
                throw std::runtime_error(
                    "sparqy: --alias-builder requires a value");
            }
            parsed.cli_has_alias_builder = true;
            parsed.parent_sampler_build_mode =
                parse_parent_sampler_build_mode_value(argv[++i]);
            continue;
        }
        const std::string alias_builder_prefix = "--alias-builder=";
        if (arg.rfind(alias_builder_prefix, 0) == 0) {
            parsed.cli_has_alias_builder = true;
            parsed.parent_sampler_build_mode =
                parse_parent_sampler_build_mode_value(
                    arg.substr(alias_builder_prefix.size()));
            continue;
        }
        if (arg == "--export-slim") {
            if (i + 1 >= argc) {
                throw std::runtime_error(
                    "sparqy: --export-slim requires a prefix");
            }
            parsed.slim_export_prefix = argv[++i];
            if (parsed.slim_export_prefix.empty()) {
                throw std::runtime_error(
                    "sparqy: --export-slim requires a prefix");
            }
            continue;
        }
        const std::string export_prefix = "--export-slim=";
        if (arg.rfind(export_prefix, 0) == 0) {
            parsed.slim_export_prefix = arg.substr(export_prefix.size());
            if (parsed.slim_export_prefix.empty()) {
                throw std::runtime_error(
                    "sparqy: --export-slim requires a prefix");
            }
            continue;
        }
        if (arg.rfind("--stats=", 0) == 0) {
            parsed.has_stats_flag = true;
            parsed.stats_flags.push_back(arg);
            continue;
        }
        parsed.positional_args.push_back(arg);
    }
    return parsed;
}

void validate_config_mode_args(const ParsedCli& cli) {
    if (cli.has_stats_flag) {
        throw std::runtime_error(
            "sparqy: --stats=LIST is only supported in legacy positional mode; declare stats in the config file");
    }
    if (!cli.positional_args.empty()) {
        throw std::runtime_error(
            "sparqy: unexpected argument in --config mode: '"
            + cli.positional_args.front() + "'");
    }
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

int parse_int_argument(const std::string& text, const char* name) {
    try {
        size_t consumed = 0u;
        const long long value = std::stoll(text, &consumed, 10);
        if (consumed != text.size()) {
            throw std::runtime_error("");
        }
        if (value < (long long)std::numeric_limits<int>::min()
            || value > (long long)std::numeric_limits<int>::max()) {
            throw std::runtime_error("");
        }
        return (int)value;
    } catch (const std::exception&) {
        throw std::runtime_error(
            "sparqy: invalid " + std::string(name) + " value '" + text + "'");
    }
}

double parse_double_argument(const std::string& text, const char* name) {
    try {
        size_t consumed = 0u;
        const double value = std::stod(text, &consumed);
        if (consumed != text.size() || !std::isfinite(value)) {
            throw std::runtime_error("");
        }
        return value;
    } catch (const std::exception&) {
        throw std::runtime_error(
            "sparqy: invalid " + std::string(name) + " value '" + text + "'");
    }
}

uint64_t parse_uint64_argument(const std::string& text, const char* name) {
    if (!text.empty() && text[0] == '-') {
        throw std::runtime_error(
            "sparqy: invalid " + std::string(name) + " value '" + text + "'");
    }

    try {
        size_t consumed = 0u;
        const unsigned long long value = std::stoull(text, &consumed, 10);
        if (consumed != text.size()
            || value > (unsigned long long)std::numeric_limits<uint64_t>::max()) {
            throw std::runtime_error("");
        }
        return (uint64_t)value;
    } catch (const std::exception&) {
        throw std::runtime_error(
            "sparqy: invalid " + std::string(name) + " value '" + text + "'");
    }
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
        "  sparqy [N] [L] [mu] [rho] [s] [G] [out_interval] [h] [seed] [threads] [--stats=LIST] [--profile] [--alias-builder=MODE] [--export-slim PREFIX]\n"
        "  sparqy --config MODEL.sparqy [--profile] [--alias-builder=MODE] [--export-slim PREFIX]\n"
        "\n"
        "Legacy positional mode builds a single chromosome with one mutation type.\n"
        "Config mode exposes the full SimParams model surface through an R-style config file.\n"
        "Legacy mode emits scalar and pairwise-similarity summaries; use config mode\n"
        "for histogram and site-frequency-spectrum outputs.\n"
        "Config mode does not accept --stats=LIST; declare stats in the config file.\n"
        "Profiling summary is printed to stderr so stdout stays machine-readable.\n"
        "--export-slim PREFIX writes PREFIX.txt in SLiM's text population format\n"
        "plus a matching PREFIX.slim bootstrap script that imports it.\n"
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

int run_legacy_mode(const ParsedCli& cli) {
    SimParams p;
    p.enable_profiling = cli.enable_profile;
    p.parent_sampler_build_mode = cli.parent_sampler_build_mode;

    std::unordered_set<int> requested_kinds;
    for (const std::string& arg : cli.stats_flags) {
        parse_stats_flag(arg, requested_kinds);
    }

    if (cli.positional_args.size() > 10u) {
        throw std::runtime_error(
            "sparqy: legacy positional mode accepts at most 10 positional arguments");
    }

    p.N       = (cli.positional_args.size() > 0u)
                  ? parse_int_argument(cli.positional_args[0], "N")
                  : 1000;
    const int L = (cli.positional_args.size() > 1u)
                    ? parse_int_argument(cli.positional_args[1], "L")
                    : 100000;
    p.mu      = (cli.positional_args.size() > 2u)
                  ? parse_double_argument(cli.positional_args[2], "mu")
                  : 1e-7;
    p.rho     = (cli.positional_args.size() > 3u)
                  ? parse_double_argument(cli.positional_args[3], "rho")
                  : 0.01;
    const double s = (cli.positional_args.size() > 4u)
                       ? parse_double_argument(cli.positional_args[4], "s")
                       : -0.01;
    p.G       = (cli.positional_args.size() > 5u)
                  ? parse_int_argument(cli.positional_args[5], "G")
                  : 100;
    int out_interval = (cli.positional_args.size() > 6u)
                         ? parse_int_argument(cli.positional_args[6], "out_interval")
                         : 10;
    const double h = (cli.positional_args.size() > 7u)
                       ? parse_double_argument(cli.positional_args[7], "h")
                       : 0.5;
    p.seed    = (cli.positional_args.size() > 8u)
                  ? parse_uint64_argument(cli.positional_args[8], "seed")
                  : 42u;
    p.threads = (cli.positional_args.size() > 9u)
                  ? parse_int_argument(cli.positional_args[9], "threads")
                  : 0;

    if (p.N <= 0) throw std::runtime_error("sparqy: N must be positive");
    if (L <= 0) throw std::runtime_error("sparqy: L must be positive");
    if (p.mu < 0.0) throw std::runtime_error("sparqy: mu must be non-negative");
    if (p.rho < 0.0) throw std::runtime_error("sparqy: rho must be non-negative");
    if (p.G <= 0) throw std::runtime_error("sparqy: G must be positive");
    if (out_interval <= 0) {
        throw std::runtime_error("sparqy: out_interval must be positive");
    }
    if (p.threads < 0) {
        throw std::runtime_error("sparqy: threads must be non-negative");
    }

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
        sparqy_names::parent_sampler_build_mode_name(
            p.parent_sampler_build_mode));

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

        constexpr size_t kNumKinds = sparqy_names::kStatisticKindCount;
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
    if (!cli.slim_export_prefix.empty()) {
        const SlimExportResult export_result =
            sim.export_state_for_slim(cli.slim_export_prefix);
        std::fprintf(stderr,
                     "sparqy: wrote SLiM export files: %s and %s\n",
                     export_result.population_path.c_str(),
                     export_result.loader_script_path.c_str());
        if (export_result.import_only_mutation_type_count > 0u) {
            std::fprintf(stderr,
                         "sparqy: generated %u import-only mutation types for standing variation with mutation-specific dominance\n",
                         export_result.import_only_mutation_type_count);
        }
        if (export_result.zero_selection_dominance_fallback_count > 0u) {
            std::fprintf(stderr,
                         "sparqy: warning: %u zero-selection mutations from distributed-dominance types used fallback dominance means in the SLiM export\n",
                         export_result.zero_selection_dominance_fallback_count);
        }
    }
    return 0;
}

int run_config_mode(const ParsedCli& cli) {
    const LoadedConfig loaded = load_config_file(cli.config_path);
    SimParams p = loaded.params;
    p.enable_profiling = p.enable_profiling || cli.enable_profile;
    if (cli.cli_has_alias_builder) {
        p.parent_sampler_build_mode = cli.parent_sampler_build_mode;
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
                 cli.config_path.c_str(),
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
                 sparqy_names::parent_sampler_build_mode_name(
                     p.parent_sampler_build_mode));

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
                        sparqy_names::statistic_kind_name(stat.kind),
                        sparqy_names::similarity_metric_name(
                            stat.similarity_metric),
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
    if (!cli.slim_export_prefix.empty()) {
        const SlimExportResult export_result =
            sim.export_state_for_slim(cli.slim_export_prefix);
        std::fprintf(stderr,
                     "sparqy: wrote SLiM export files: %s and %s\n",
                     export_result.population_path.c_str(),
                     export_result.loader_script_path.c_str());
        if (export_result.import_only_mutation_type_count > 0u) {
            std::fprintf(stderr,
                         "sparqy: generated %u import-only mutation types for standing variation with mutation-specific dominance\n",
                         export_result.import_only_mutation_type_count);
        }
        if (export_result.zero_selection_dominance_fallback_count > 0u) {
            std::fprintf(stderr,
                         "sparqy: warning: %u zero-selection mutations from distributed-dominance types used fallback dominance means in the SLiM export\n",
                         export_result.zero_selection_dominance_fallback_count);
        }
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const ParsedCli cli = parse_cli_args(argc, argv);
        if (cli.show_help) {
            print_usage();
            return 0;
        }

        if (cli.has_config) {
            validate_config_mode_args(cli);
            return run_config_mode(cli);
        }

        return run_legacy_mode(cli);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
}
