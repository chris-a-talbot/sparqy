#include "config_loader.hpp"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <unistd.h>

namespace {

std::string make_temp_config_path() {
    char path[] = "/tmp/sparqy_config_test_XXXXXX";
    const int fd = mkstemp(path);
    if (fd < 0) {
        throw std::runtime_error("mkstemp failed");
    }
    std::fclose(fdopen(fd, "w"));
    return std::string(path);
}

struct TempConfigFile {
    explicit TempConfigFile(const std::string& contents) : path(make_temp_config_path()) {
        std::ofstream out(path);
        if (!out.is_open()) {
            throw std::runtime_error("unable to open temp config file for writing");
        }
        out << contents;
    }

    ~TempConfigFile() {
        std::remove(path.c_str());
    }

    std::string path;
};

std::string valid_config_text(bool include_unused_constant = false,
                              bool include_runtime = false) {
    std::string text =
        "constants <- list(\n"
        "  pop_size = 100,\n"
        "  generation_count = 5,\n"
        "  mutation_rate = 1e-7,\n"
        "  recombination_rate = 0.5,\n"
        "  run_seed = 42,\n"
        "  thread_count = 1";
    if (include_runtime) {
        text +=
            ",\n"
            "  profile_default = TRUE,\n"
            "  builder_name = \"parallel_psa_plus\"";
    }
    if (include_unused_constant) {
        text += ",\n  note = \"unused\"";
    }
    text +=
        "\n)\n\n"
        "config <- list(\n"
        "  N = pop_size,\n"
        "  G = generation_count,\n"
        "  mu = mutation_rate,\n"
        "  rho = recombination_rate,\n"
        "  seed = run_seed,\n"
        "  threads = thread_count,\n";
    if (include_runtime) {
        text += "  runtime = list(alias_builder = builder_name, profile = profile_default),\n";
    }
    text +=
        "  mutation_types = list(\n"
        "    deleterious = list(\n"
        "      selection = gamma(mean = -0.02, 0.3, min = -0.999999999, 0.0),\n"
        "      dominance = fixed(0.25)\n"
        "    ),\n"
        "    neutral = list(\n"
        "      selection = constant(0.0),\n"
        "      dominance = additive()\n"
        "    )\n"
        "  ),\n"
        "  region_types = list(\n"
        "    whole_genome = list(\n"
        "      mutation_scale = 1.0,\n"
        "      weights = c(deleterious = 1, neutral = 1)\n"
        "    )\n"
        "  ),\n"
        "  chromosomes = list(\n"
        "    chr1 = list(\n"
        "      length = 1000,\n"
        "      recombination_intervals = list(\n"
        "        interval(0, 1000, 1.0)\n"
        "      ),\n"
        "      regions = list(\n"
        "        region(whole_genome, 0, 1000)\n"
        "      )\n"
        "    )\n"
        "  )\n"
        ")\n\n"
        "stats <- list(\n"
        "  every(2, mean_fitness, genetic_load),\n"
        "  at(c(3, 5), pairwise_similarity(jaccard))\n"
        ")\n";
    return text;
}

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void require_generations(const std::vector<uint64_t>& actual,
                         const std::vector<uint64_t>& expected,
                         const std::string& label) {
    require(actual == expected, "unexpected generations for " + label);
}

void test_valid_config_parses() {
    TempConfigFile file(valid_config_text());
    const LoadedConfig loaded = load_config_file(file.path);

    require(loaded.params.N == 100, "expected N to parse from constants reference");
    require(loaded.params.G == 5, "expected G to parse from constants reference");
    require(loaded.params.mu == 1e-7, "expected mu to parse");
    require(loaded.params.rho == 0.5, "expected rho to parse");
    require(loaded.params.seed == 42u, "expected seed to parse");
    require(loaded.params.threads == 1, "expected threads to parse");
    require(loaded.params.mutation_types.size() == 2u, "expected two mutation types");
    require(loaded.params.mutation_region_types.size() == 1u, "expected one region type");
    require(loaded.params.chromosomes.size() == 1u, "expected one chromosome");
    require(loaded.has_statistics, "expected statistics to be enabled");
    require(loaded.params.statistic_requests.size() == 3u, "expected three statistic requests");
    require(loaded.params.parent_sampler_build_mode == ParentSamplerBuildMode::automatic,
            "expected omitted alias_builder to default to auto");
    require_generations(loaded.params.statistic_requests[0].generations, {2u, 4u}, "every()");
    require_generations(loaded.params.statistic_requests[2].generations, {3u, 5u}, "at(c(...))");
    require(loaded.warnings.empty(), "did not expect warnings for fully used constants");
}

void test_unused_constant_warns() {
    TempConfigFile file(valid_config_text(true));
    const LoadedConfig loaded = load_config_file(file.path);
    require(loaded.warnings.size() == 1u, "expected one unused-constant warning");
    require(loaded.warnings[0].find("unused constant 'note'") != std::string::npos,
            "expected unused constant warning text");
}

void test_runtime_config_parses() {
    TempConfigFile file(valid_config_text(false, true));
    const LoadedConfig loaded = load_config_file(file.path);
    require(loaded.params.parent_sampler_build_mode == ParentSamplerBuildMode::parallel_psa_plus,
            "expected runtime alias_builder to parse");
    require(loaded.params.enable_profiling, "expected runtime profile to parse");
}

void test_runtime_auto_alias_builder_parses() {
    const std::string text =
        "constants <- list(builder_name = \"auto\")\n"
        "config <- list(\n"
        "  N = 10,\n"
        "  G = 2,\n"
        "  mu = 1e-7,\n"
        "  rho = 0.5,\n"
        "  seed = 1,\n"
        "  threads = 4,\n"
        "  runtime = list(alias_builder = builder_name),\n"
        "  mutation_types = list(m = list(selection = constant(0.0), dominance = additive())),\n"
        "  region_types = list(r = list(mutation_scale = 1.0, weights = c(m = 1))),\n"
        "  chromosomes = list(chr1 = list(\n"
        "    length = 10,\n"
        "    recombination_intervals = list(interval(0, 10, 1.0)),\n"
        "    regions = list(region(r, 0, 10))\n"
        "  ))\n"
        ")\n"
        "stats <- list()\n";
    TempConfigFile file(text);
    const LoadedConfig loaded = load_config_file(file.path);
    require(loaded.params.parent_sampler_build_mode == ParentSamplerBuildMode::automatic,
            "expected runtime alias_builder='auto' to parse");
}

void test_auto_alias_builder_resolution() {
    require(resolve_parent_sampler_build_mode(
                ParentSamplerBuildMode::automatic, 5000, 4)
                == ParentSamplerBuildMode::sequential,
            "expected auto alias builder to stay sequential for small-N jobs");
    require(resolve_parent_sampler_build_mode(
                ParentSamplerBuildMode::automatic, 10000, 4)
                == ParentSamplerBuildMode::parallel_psa_plus,
            "expected auto alias builder to use PSA+ at the large-N threshold");
    require(resolve_parent_sampler_build_mode(
                ParentSamplerBuildMode::automatic, 50000, 1)
                == ParentSamplerBuildMode::sequential,
            "expected auto alias builder to stay sequential for single-thread jobs");
    require(resolve_parent_sampler_build_mode(
                ParentSamplerBuildMode::parallel, 5000, 4)
                == ParentSamplerBuildMode::parallel,
            "expected explicit alias-builder selections to pass through unchanged");
}

void test_extended_schedule_builders_parse() {
    const std::string text =
        "constants <- list()\n"
        "config <- list(\n"
        "  N = 10,\n"
        "  G = 8,\n"
        "  mu = 1e-7,\n"
        "  rho = 0.5,\n"
        "  seed = 1,\n"
        "  threads = 1,\n"
        "  mutation_types = list(m = list(selection = constant(0.0), dominance = additive())),\n"
        "  region_types = list(r = list(mutation_scale = 1.0, weights = c(m = 1))),\n"
        "  chromosomes = list(chr1 = list(\n"
        "    length = 10,\n"
        "    recombination_intervals = list(interval(0, 10, 1.0)),\n"
        "    regions = list(region(r, 0, 10))\n"
        "  ))\n"
        ")\n"
        "stats <- list(\n"
        "  up_to(end = 3, mean_fitness),\n"
        "  at_after(start = 6, genetic_load),\n"
        "  every_up_to(step = 3, end = 8, n_seg),\n"
        "  every_at_after(step = 2, start = 5, n_fixed),\n"
        "  range(start = 2, end = 4, exact_B),\n"
        "  every_range(step = 2, start = 2, end = 7, genome_words)\n"
        ")\n";
    TempConfigFile file(text);
    const LoadedConfig loaded = load_config_file(file.path);
    require(loaded.params.statistic_requests.size() == 6u, "expected six statistic requests");
    require_generations(loaded.params.statistic_requests[0].generations, {1u, 2u, 3u}, "up_to()");
    require_generations(loaded.params.statistic_requests[1].generations, {6u, 7u, 8u}, "at_after()");
    require_generations(loaded.params.statistic_requests[2].generations, {3u, 6u}, "every_up_to()");
    require_generations(loaded.params.statistic_requests[3].generations, {5u, 7u}, "every_at_after()");
    require_generations(loaded.params.statistic_requests[4].generations, {2u, 3u, 4u}, "range()");
    require_generations(loaded.params.statistic_requests[5].generations, {2u, 4u, 6u}, "every_range()");
}

void expect_parse_error(const std::string& text, const std::string& expected_substring) {
    TempConfigFile file(text);
    try {
        (void)load_config_file(file.path);
        throw std::runtime_error("expected config parse failure");
    } catch (const std::exception& e) {
        const std::string message = e.what();
        require(message.find(expected_substring) != std::string::npos,
                "unexpected error text: " + message);
    }
}

void test_missing_stats_section_fails() {
    const std::string text =
        "constants <- list()\n"
        "config <- list(\n"
        "  N = 10,\n"
        "  G = 1,\n"
        "  mu = 1e-7,\n"
        "  rho = 0.5,\n"
        "  seed = 1,\n"
        "  threads = 1,\n"
        "  mutation_types = list(m = list(selection = constant(0.0), dominance = additive())),\n"
        "  region_types = list(r = list(mutation_scale = 1.0, weights = c(m = 1))),\n"
        "  chromosomes = list(chr1 = list(\n"
        "    length = 10,\n"
        "    recombination_intervals = list(interval(0, 10, 1.0)),\n"
        "    regions = list(region(r, 0, 10))\n"
        "  ))\n"
        ")\n";
    expect_parse_error(text, "missing required top-level section 'stats'");
}

void test_negative_mu_fails() {
    std::string text = valid_config_text();
    const std::string needle = "  mu = mutation_rate,\n";
    const size_t pos = text.find(needle);
    text.replace(pos, needle.size(), "  mu = -1e-7,\n");
    expect_parse_error(text, "config$mu must be non-negative");
}

void test_gappy_regions_fail() {
    std::string text = valid_config_text();
    const std::string needle = "        region(whole_genome, 0, 1000)\n";
    const size_t pos = text.find(needle);
    text.replace(pos, needle.size(),
                 "        region(whole_genome, 0, 400),\n"
                 "        region(whole_genome, 600, 1000)\n");
    expect_parse_error(text, "must be contiguous from 0 without gaps or overlaps");
}

void test_invalid_range_fails() {
    std::string text = valid_config_text();
    const std::string needle =
        "stats <- list(\n"
        "  every(2, mean_fitness, genetic_load),\n"
        "  at(c(3, 5), pairwise_similarity(jaccard))\n"
        ")\n";
    text.replace(text.find(needle), needle.size(),
                 "stats <- list(\n"
                 "  range(5, 3, mean_fitness)\n"
                 ")\n");
    expect_parse_error(text, "requires end >= start");
}

void test_invalid_distribution_parameters_fail() {
    {
        std::string text = valid_config_text();
        const std::string needle =
            "      selection = gamma(mean = -0.02, 0.3, min = -0.999999999, 0.0),\n";
        text.replace(text.find(needle), needle.size(),
                     "      selection = gamma(-0.02, 0.0, -0.999999999, 0.0),\n");
        expect_parse_error(text, "gamma shape must be > 0");
    }

    {
        std::string text = valid_config_text();
        const std::string needle =
            "      selection = constant(0.0),\n";
        text.replace(text.find(needle), needle.size(),
                     "      selection = constant(5.0, 0.0, 1.0),\n");
        expect_parse_error(text, "constant(value, min, max) requires value to lie within [min, max]");
    }
}

void test_overlapping_schedules_coalesce_per_generation() {
    const std::string text =
        "constants <- list()\n"
        "config <- list(\n"
        "  N = 10,\n"
        "  G = 3,\n"
        "  mu = 1e-7,\n"
        "  rho = 0.5,\n"
        "  seed = 1,\n"
        "  threads = 1,\n"
        "  mutation_types = list(m = list(selection = constant(0.0), dominance = additive())),\n"
        "  region_types = list(r = list(mutation_scale = 1.0, weights = c(m = 1))),\n"
        "  chromosomes = list(chr1 = list(\n"
        "    length = 10,\n"
        "    recombination_intervals = list(interval(0, 10, 1.0)),\n"
        "    regions = list(region(r, 0, 10))\n"
        "  ))\n"
        ")\n"
        "stats <- list(\n"
        "  always(mean_fitness),\n"
        "  at(c(2, 2), mean_fitness),\n"
        "  at_after(start = 2, mean_fitness)\n"
        ")\n";
    TempConfigFile file(text);
    const LoadedConfig loaded = load_config_file(file.path);
    Simulator simulator(loaded.params);

    simulator.step();
    require(simulator.latest_statistics().statistics.size() == 1u,
            "expected one statistic at generation 1");

    simulator.step();
    require(simulator.latest_statistics().statistics.size() == 1u,
            "expected overlapping schedules to coalesce at generation 2");

    simulator.step();
    require(simulator.latest_statistics().statistics.size() == 1u,
            "expected overlapping schedules to coalesce at generation 3");
}

}  // namespace

int main() {
    try {
        test_valid_config_parses();
        test_unused_constant_warns();
        test_runtime_config_parses();
        test_runtime_auto_alias_builder_parses();
        test_auto_alias_builder_resolution();
        test_extended_schedule_builders_parse();
        test_missing_stats_section_fails();
        test_negative_mu_fails();
        test_gappy_regions_fail();
        test_invalid_range_fails();
        test_invalid_distribution_parameters_fail();
        test_overlapping_schedules_coalesce_per_generation();
    } catch (const std::exception& e) {
        std::cerr << "config_loader_test failed: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
