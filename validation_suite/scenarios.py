from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from typing import Any


ALIAS_BUILDERS = ("sequential", "parallel", "parallel_psa_plus")


def format_float(value: float) -> str:
    return f"{value:.12g}"


@dataclass(frozen=True)
class SelectionSpec:
    kind: str
    p1: float
    p2: float = 0.0
    min_value: float = -0.999999999
    max_value: float = 1.0

    def to_sparqy_expr(self) -> str:
        if self.kind == "constant":
            if (
                abs(self.min_value - self.p1) < 1e-12
                and abs(self.max_value - self.p1) < 1e-12
            ):
                return f"constant({format_float(self.p1)})"
            return (
                "constant("
                f"{format_float(self.p1)}, {format_float(self.min_value)}, "
                f"{format_float(self.max_value)})"
            )
        if self.kind == "uniform":
            if (
                abs(self.min_value - self.p1) < 1e-12
                and abs(self.max_value - self.p2) < 1e-12
            ):
                return f"uniform({format_float(self.p1)}, {format_float(self.p2)})"
            return (
                "uniform("
                f"{format_float(self.p1)}, {format_float(self.p2)}, "
                f"{format_float(self.min_value)}, {format_float(self.max_value)})"
            )
        if self.kind == "normal":
            return (
                "normal("
                f"{format_float(self.p1)}, {format_float(self.p2)}, "
                f"{format_float(self.min_value)}, {format_float(self.max_value)})"
            )
        if self.kind == "exponential":
            return (
                "exponential("
                f"{format_float(self.p1)}, {format_float(self.min_value)}, "
                f"{format_float(self.max_value)})"
            )
        if self.kind == "gamma":
            return (
                "gamma("
                f"{format_float(self.p1)}, {format_float(self.p2)}, "
                f"{format_float(self.min_value)}, {format_float(self.max_value)})"
            )
        if self.kind == "beta":
            return (
                "beta("
                f"{format_float(self.p1)}, {format_float(self.p2)}, "
                f"{format_float(self.min_value)}, {format_float(self.max_value)})"
            )
        raise ValueError(f"Unsupported sparqy distribution '{self.kind}'")

    def to_slim_tokens(self) -> list[str]:
        if self.kind == "constant":
            return ['"f"', format_float(self.p1)]
        if self.kind == "gamma":
            return ['"g"', format_float(self.p1), format_float(self.p2)]
        if self.kind == "normal":
            return ['"n"', format_float(self.p1), format_float(self.p2)]
        if self.kind == "exponential":
            return ['"e"', format_float(self.p1)]
        raise ValueError(
            f"Selection distribution '{self.kind}' is not supported by the "
            "generic SLiM renderer. Use constant, gamma, normal, or exponential."
        )


@dataclass(frozen=True)
class DominanceSpec:
    mode: str
    params: tuple[float, ...] = ()

    @staticmethod
    def additive() -> "DominanceSpec":
        return DominanceSpec("additive", ())

    @staticmethod
    def fixed(h: float) -> "DominanceSpec":
        return DominanceSpec("fixed", (h,))

    def to_sparqy_expr(self) -> str:
        if self.mode == "additive":
            return "additive()"
        if self.mode == "fixed":
            return f"fixed({format_float(self.params[0])})"
        raise ValueError(
            f"Dominance mode '{self.mode}' is not supported by the generic suite."
        )

    def slim_h(self) -> float:
        if self.mode == "additive":
            return 0.5
        if self.mode == "fixed":
            return self.params[0]
        raise ValueError(
            f"Dominance mode '{self.mode}' is not supported by the generic SLiM renderer."
        )


@dataclass(frozen=True)
class MutationTypeSpec:
    name: str
    selection: SelectionSpec
    dominance: DominanceSpec


@dataclass(frozen=True)
class RegionTypeSpec:
    name: str
    mutation_scale: float = 1.0
    weights: tuple[tuple[str, float], ...] = ()


@dataclass(frozen=True)
class RecIntervalSpec:
    start: int
    end: int
    rate_scale: float


@dataclass(frozen=True)
class ChromosomeRegionSpec:
    region_type_name: str
    start: int
    end: int


@dataclass(frozen=True)
class ChromosomeSpec:
    name: str
    length: int
    recombination_intervals: tuple[RecIntervalSpec, ...]
    regions: tuple[ChromosomeRegionSpec, ...]


@dataclass(frozen=True)
class Scenario:
    name: str
    description: str
    N: int
    G: int
    mu: float
    rho: float
    mutation_types: tuple[MutationTypeSpec, ...]
    region_types: tuple[RegionTypeSpec, ...]
    chromosomes: tuple[ChromosomeSpec, ...]

    @property
    def total_genome_length(self) -> int:
        return sum(chrom.length for chrom in self.chromosomes)

    @property
    def total_recombination_mass(self) -> float:
        return sum(
            interval.rate_scale * (interval.end - interval.start)
            for chrom in self.chromosomes
            for interval in normalized_recombination_intervals(chrom)
        )

    def to_manifest(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AccuracyVariant:
    alias_builder: str
    threads: int

    @property
    def label(self) -> str:
        return f"{self.alias_builder}@T{self.threads}"


@dataclass(frozen=True)
class SuiteCase:
    label: str
    scenario: Scenario
    replicates: int


@dataclass(frozen=True)
class SuitePreset:
    name: str
    description: str
    accuracy_cases: tuple[SuiteCase, ...]
    speed_cases: tuple[SuiteCase, ...]
    scaling_cases: tuple[SuiteCase, ...]
    profile_cases: tuple[SuiteCase, ...]
    accuracy_variants: tuple[AccuracyVariant, ...]
    speed_threads: tuple[int, ...]
    scaling_threads: tuple[int, ...]
    profile_threads: tuple[int, ...]
    speed_alias_builders: tuple[str, ...] = ALIAS_BUILDERS
    scaling_alias_builders: tuple[str, ...] = ALIAS_BUILDERS
    profile_alias_builders: tuple[str, ...] = ALIAS_BUILDERS
    scaling_threads_by_case: dict[str, tuple[int, ...]] = field(default_factory=dict)
    profile_threads_by_case: dict[str, tuple[int, ...]] = field(default_factory=dict)

    def to_manifest(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "accuracy_cases": [case.label for case in self.accuracy_cases],
            "speed_cases": [case.label for case in self.speed_cases],
            "scaling_cases": [case.label for case in self.scaling_cases],
            "profile_cases": [case.label for case in self.profile_cases],
            "accuracy_variants": [variant.label for variant in self.accuracy_variants],
            "speed_threads": list(self.speed_threads),
            "scaling_threads": list(self.scaling_threads),
            "profile_threads": list(self.profile_threads),
            "speed_alias_builders": list(self.speed_alias_builders),
            "scaling_alias_builders": list(self.scaling_alias_builders),
            "profile_alias_builders": list(self.profile_alias_builders),
            "scaling_threads_by_case": {
                case_label: list(threads)
                for case_label, threads in self.scaling_threads_by_case.items()
            },
            "profile_threads_by_case": {
                case_label: list(threads)
                for case_label, threads in self.profile_threads_by_case.items()
            },
        }

    def scaling_threads_for(self, case_label: str) -> tuple[int, ...]:
        return self.scaling_threads_by_case.get(case_label, self.scaling_threads)

    def profile_threads_for(self, case_label: str) -> tuple[int, ...]:
        return self.profile_threads_by_case.get(case_label, self.profile_threads)


def fixed_type(name: str, s: float, h: float) -> MutationTypeSpec:
    dominance = (
        DominanceSpec.additive() if abs(h - 0.5) < 1e-12 else DominanceSpec.fixed(h)
    )
    return MutationTypeSpec(
        name=name,
        selection=SelectionSpec("constant", s, 0.0, s, s),
        dominance=dominance,
    )


def gamma_type(name: str, mean_s: float, shape: float, h: float) -> MutationTypeSpec:
    return MutationTypeSpec(
        name=name,
        selection=SelectionSpec("gamma", mean_s, shape, -0.999999999, 0.0),
        dominance=DominanceSpec.fixed(h) if abs(h - 0.5) >= 1e-12 else DominanceSpec.additive(),
    )


def normal_type(name: str, mean_s: float, sd: float, h: float) -> MutationTypeSpec:
    return MutationTypeSpec(
        name=name,
        selection=SelectionSpec("normal", mean_s, sd, -0.999999999, 1.0),
        dominance=DominanceSpec.fixed(h) if abs(h - 0.5) >= 1e-12 else DominanceSpec.additive(),
    )


def exponential_type(name: str, mean_s: float, h: float) -> MutationTypeSpec:
    return MutationTypeSpec(
        name=name,
        selection=SelectionSpec("exponential", mean_s, 0.0, -0.999999999, 0.0),
        dominance=DominanceSpec.fixed(h) if abs(h - 0.5) >= 1e-12 else DominanceSpec.additive(),
    )


def normalized_recombination_intervals(
    chromosome: ChromosomeSpec,
) -> tuple[RecIntervalSpec, ...]:
    if chromosome.recombination_intervals:
        return chromosome.recombination_intervals
    return (RecIntervalSpec(0, chromosome.length, 1.0),)


def normalized_regions(
    scenario: Scenario,
    chromosome: ChromosomeSpec,
) -> tuple[ChromosomeRegionSpec, ...]:
    if chromosome.regions:
        return chromosome.regions
    default_region_name = scenario.region_types[0].name
    return (ChromosomeRegionSpec(default_region_name, 0, chromosome.length),)


def validate_scenario(scenario: Scenario) -> None:
    if scenario.N <= 0 or scenario.G <= 0:
        raise ValueError(f"{scenario.name}: N and G must be positive")
    if scenario.mu < 0.0 or scenario.rho < 0.0:
        raise ValueError(f"{scenario.name}: mu and rho must be non-negative")
    if not scenario.mutation_types:
        raise ValueError(f"{scenario.name}: at least one mutation type is required")
    if not scenario.region_types:
        raise ValueError(f"{scenario.name}: at least one region type is required")
    if not scenario.chromosomes:
        raise ValueError(f"{scenario.name}: at least one chromosome is required")

    mutation_type_names = {mutation_type.name for mutation_type in scenario.mutation_types}
    region_type_names = {region_type.name for region_type in scenario.region_types}

    for region_type in scenario.region_types:
        for mutation_type_name, weight in region_type.weights:
            if mutation_type_name not in mutation_type_names:
                raise ValueError(
                    f"{scenario.name}: region type '{region_type.name}' references "
                    f"unknown mutation type '{mutation_type_name}'"
                )
            if weight <= 0.0:
                raise ValueError(
                    f"{scenario.name}: region type '{region_type.name}' has non-positive weight"
                )

    for chromosome in scenario.chromosomes:
        if chromosome.length <= 0:
            raise ValueError(f"{scenario.name}: chromosome '{chromosome.name}' has non-positive length")

        last_end = 0
        for interval in normalized_recombination_intervals(chromosome):
            if interval.start != last_end:
                raise ValueError(
                    f"{scenario.name}: recombination intervals on chromosome "
                    f"'{chromosome.name}' must be contiguous from 0"
                )
            if interval.end <= interval.start or interval.end > chromosome.length:
                raise ValueError(
                    f"{scenario.name}: chromosome '{chromosome.name}' has an invalid recombination interval"
                )
            last_end = interval.end
        if last_end != chromosome.length:
            raise ValueError(
                f"{scenario.name}: recombination intervals on chromosome '{chromosome.name}' "
                "must cover the whole chromosome"
            )

        last_end = 0
        for region in normalized_regions(scenario, chromosome):
            if region.region_type_name not in region_type_names:
                raise ValueError(
                    f"{scenario.name}: chromosome '{chromosome.name}' references "
                    f"unknown region type '{region.region_type_name}'"
                )
            if region.start != last_end:
                raise ValueError(
                    f"{scenario.name}: regions on chromosome '{chromosome.name}' "
                    "must be contiguous from 0"
                )
            if region.end <= region.start or region.end > chromosome.length:
                raise ValueError(
                    f"{scenario.name}: chromosome '{chromosome.name}' has an invalid region"
                )
            last_end = region.end
        if last_end != chromosome.length:
            raise ValueError(
                f"{scenario.name}: regions on chromosome '{chromosome.name}' must cover the whole chromosome"
            )


def render_sparqy_config(
    scenario: Scenario,
    *,
    seed: int,
    threads: int,
    include_stats: bool,
    include_pairwise: bool = False,
) -> str:
    validate_scenario(scenario)

    lines = [
        f"# Auto-generated by validation_suite for scenario '{scenario.name}'",
        f"# {scenario.description}",
        "",
        "constants <- list(",
        f"  pop_size = {scenario.N},",
        f"  generation_count = {scenario.G},",
        f"  mutation_rate = {format_float(scenario.mu)},",
        f"  recombination_rate = {format_float(scenario.rho)},",
        f"  run_seed = {seed},",
        f"  thread_count = {threads}",
        ")",
        "",
        "config <- list(",
        "  N = pop_size,",
        "  G = generation_count,",
        "  mu = mutation_rate,",
        "  rho = recombination_rate,",
        "  seed = run_seed,",
        "  threads = thread_count,",
        "  mutation_types = list(",
    ]

    for index, mutation_type in enumerate(scenario.mutation_types):
        suffix = "," if index + 1 < len(scenario.mutation_types) else ""
        lines.extend(
            [
                f"    {mutation_type.name} = list(",
                f"      selection = {mutation_type.selection.to_sparqy_expr()},",
                f"      dominance = {mutation_type.dominance.to_sparqy_expr()}",
                f"    ){suffix}",
            ]
        )
    lines.extend(
        [
            "  ),",
            "  region_types = list(",
        ]
    )

    for index, region_type in enumerate(scenario.region_types):
        suffix = "," if index + 1 < len(scenario.region_types) else ""
        weight_text = ", ".join(
            f"{mutation_type_name} = {format_float(weight)}"
            for mutation_type_name, weight in region_type.weights
        )
        lines.extend(
            [
                f"    {region_type.name} = list(",
                f"      mutation_scale = {format_float(region_type.mutation_scale)},",
                f"      weights = c({weight_text})",
                f"    ){suffix}",
            ]
        )
    lines.extend(
        [
            "  ),",
            "  chromosomes = list(",
        ]
    )

    for chrom_index, chromosome in enumerate(scenario.chromosomes):
        chrom_suffix = "," if chrom_index + 1 < len(scenario.chromosomes) else ""
        intervals = normalized_recombination_intervals(chromosome)
        regions = normalized_regions(scenario, chromosome)
        lines.extend(
            [
                f"    {chromosome.name} = list(",
                f"      length = {chromosome.length},",
                "      recombination_intervals = list(",
            ]
        )
        for interval_index, interval in enumerate(intervals):
            interval_suffix = "," if interval_index + 1 < len(intervals) else ""
            lines.append(
                "        interval("
                f"{interval.start}, {interval.end}, {format_float(interval.rate_scale)})"
                f"{interval_suffix}"
            )
        lines.extend(
            [
                "      ),",
                "      regions = list(",
            ]
        )
        for region_index, region in enumerate(regions):
            region_suffix = "," if region_index + 1 < len(regions) else ""
            lines.append(
                "        region("
                f"{region.region_type_name}, {region.start}, {region.end})"
                f"{region_suffix}"
            )
        lines.extend(
            [
                "      )",
                f"    ){chrom_suffix}",
            ]
        )
    lines.extend(
        [
            "  )",
            ")",
            "",
        ]
    )

    if include_stats:
        g = scenario.G
        lines.extend(
            [
                "stats <- list(",
                "  at("
                f"{g}, mean_fitness, genetic_load, realized_masking_bonus, exact_B, "
                "n_seg, n_fixed, mutation_histogram, site_frequency_spectrum, "
                "nucleotide_diversity, expected_heterozygosity"
                ")",
            ]
        )
        if include_pairwise:
            lines[-1] = lines[-1] + ","
            lines.extend(
                [
                    "  at("
                    f"{g}, pairwise_similarity(jaccard), "
                    "pairwise_similarity(dice), pairwise_similarity(overlap)"
                    ")",
                ]
            )
        lines.append(")")
    else:
        lines.append("stats <- list()")

    lines.append("")
    return "\n".join(lines)


SLIM_RUNTIME_EMIT = r"""
function (void)emit_runtime(float$ wall_seconds) {
    catn("scalar\ttotal_wall_sec\t" + wall_seconds);
}
"""


SLIM_STATS_EMIT = r"""
function (void)emit_stats(float$ wall_seconds) {
    inds = p1.individuals;
    meanW = mean(p1.cachedFitness(NULL));
    gl = 1.0 - meanW;
    muts = sim.mutations;
    nseg = size(muts);
    nfix = size(sim.substitutions);
    totalHaps = 2 * N_POP;

    sum2pq = 0.0;
    exactB = 0.0;
    typeCounts = rep(0, NUM_MUTATION_TYPES);
    chromCounts = rep(0, NUM_CHROMOSOMES);
    unfolded = rep(0, totalHaps + 1);
    folded = rep(0, N_POP + 1);

    if (nseg > 0) {
        freqs = sim.mutationFrequencies(NULL, muts);
        H = 2.0 * freqs * (1.0 - freqs);
        sum2pq = sum(H);
        exactB = sum(muts.selectionCoeff * (0.5 - muts.mutationType.dominanceCoeff) * H);

        mtids = muts.mutationType.id;
        chids = muts.chromosome.id;
        copyNums = asInteger(round(freqs * totalHaps));
        for (k in 0:(nseg - 1)) {
            typeCounts[mtids[k] - 1] = typeCounts[mtids[k] - 1] + 1;
            chromCounts[chids[k] - 1] = chromCounts[chids[k] - 1] + 1;

            copy = copyNums[k];
            if ((copy > 0) & (copy < totalHaps)) {
                unfolded[copy] = unfolded[copy] + 1;
                minor = copy;
                if ((totalHaps - copy) < minor)
                    minor = totalHaps - copy;
                folded[minor] = folded[minor] + 1;
            }
        }
    }

    wHom = rep(1.0, size(inds));
    if (nseg > 0) {
        for (i in 0:(size(inds) - 1)) {
            uniqueMuts = unique(inds[i].haplosomes.mutations);
            if (size(uniqueMuts) > 0)
                wHom[i] = product(1.0 + uniqueMuts.selectionCoeff);
        }
    }
    realizedMasking = meanW - mean(wHom);
    pi = 0.0;
    if (GENOME_BP > 0)
        pi = sum2pq / GENOME_BP;
    eh = 0.0;
    if (nseg > 0)
        eh = sum2pq / nseg;

    catn("scalar\tmean_fitness\t" + meanW);
    catn("scalar\tgenetic_load\t" + gl);
    catn("scalar\trealized_masking_bonus\t" + realizedMasking);
    catn("scalar\texact_B\t" + exactB);
    catn("scalar\tn_seg\t" + nseg);
    catn("scalar\tn_fixed\t" + nfix);
    catn("scalar\tnucleotide_diversity\t" + pi);
    catn("scalar\texpected_heterozygosity\t" + eh);
    catn("scalar\ttotal_wall_sec\t" + wall_seconds);
    catn("vector\tmutation_histogram_by_type\t" + paste(typeCounts, sep=","));
    catn("vector\tmutation_histogram_by_chromosome\t" + paste(chromCounts, sep=","));
    catn("vector\tsfs_unfolded\t" + paste(unfolded, sep=","));
    catn("vector\tsfs_folded\t" + paste(folded, sep=","));
}
"""


def _slim_mutation_type_lines(
    scenario: Scenario,
    mutation_type_var_names: dict[str, str],
) -> list[str]:
    lines: list[str] = []
    for index, mutation_type in enumerate(scenario.mutation_types, start=1):
        variable_name = mutation_type_var_names[mutation_type.name]
        h = mutation_type.dominance.slim_h()
        dist_tokens = ", ".join(mutation_type.selection.to_slim_tokens())
        lines.append(
            f'    initializeMutationType("{variable_name}", {format_float(h)}, {dist_tokens});'
        )
        lines.append(f"    {variable_name}.convertToSubstitution = T;")
        if index != len(scenario.mutation_types):
            lines.append("")
    return lines


def _slim_genomic_element_type_lines(
    scenario: Scenario,
    mutation_type_var_names: dict[str, str],
    region_type_var_names: dict[str, str],
) -> list[str]:
    lines: list[str] = []
    for index, region_type in enumerate(scenario.region_types, start=1):
        variable_name = region_type_var_names[region_type.name]
        weights = region_type.weights or ((scenario.mutation_types[0].name, 1.0),)
        if len(weights) == 1:
            lines.append(
                f'    initializeGenomicElementType("{variable_name}", '
                f'{mutation_type_var_names[weights[0][0]]}, 1.0);'
            )
        else:
            slim_mutation_refs = ", ".join(
                mutation_type_var_names[mutation_type_name]
                for mutation_type_name, _weight in weights
            )
            slim_weight_values = ", ".join(
                format_float(weight) for _mutation_type_name, weight in weights
            )
            lines.append(
                f'    initializeGenomicElementType("{variable_name}", '
                f"c({slim_mutation_refs}), c({slim_weight_values}));"
            )
        if index != len(scenario.region_types):
            lines.append("")
    return lines


def _slim_mutation_rate_lines(scenario: Scenario, chromosome: ChromosomeSpec) -> list[str]:
    regions = normalized_regions(scenario, chromosome)
    rate_values = [
        format_float(
            scenario.mu * next(
                region_type.mutation_scale
                for region_type in scenario.region_types
                if region_type.name == region.region_type_name
            )
        )
        for region in regions
    ]
    end_values = [str(region.end - 1) for region in regions]
    if len(rate_values) == 1:
        return [f"    initializeMutationRate({rate_values[0]});"]
    return [
        f"    initializeMutationRate(c({', '.join(rate_values)}), "
        f"c({', '.join(end_values)}));"
    ]


def _slim_region_lines(
    scenario: Scenario,
    chromosome: ChromosomeSpec,
    region_type_var_names: dict[str, str],
) -> list[str]:
    lines: list[str] = []
    for region in normalized_regions(scenario, chromosome):
        lines.append(
            f"    initializeGenomicElement("
            f"{region_type_var_names[region.region_type_name]}, "
            f"{region.start}, {region.end - 1});"
        )
    return lines


def _slim_recombination_rate_lines(
    scenario: Scenario,
    chromosome: ChromosomeSpec,
) -> list[str]:
    intervals = normalized_recombination_intervals(chromosome)
    total_mass = scenario.total_recombination_mass
    if total_mass <= 0.0:
        rate_values = ["0.0" for _interval in intervals]
    else:
        rate_values = [
            format_float((scenario.rho * interval.rate_scale) / total_mass)
            for interval in intervals
        ]
    end_values = [str(interval.end - 1) for interval in intervals]
    if len(rate_values) == 1:
        return [f"    initializeRecombinationRate({rate_values[0]});"]
    return [
        f"    initializeRecombinationRate(c({', '.join(rate_values)}), "
        f"c({', '.join(end_values)}));"
    ]


def render_slim_script(scenario: Scenario, *, include_stats: bool) -> str:
    validate_scenario(scenario)

    mutation_type_var_names = {
        mutation_type.name: f"m{index}"
        for index, mutation_type in enumerate(scenario.mutation_types, start=1)
    }
    region_type_var_names = {
        region_type.name: f"g{index}"
        for index, region_type in enumerate(scenario.region_types, start=1)
    }

    body_lines: list[str] = [
        f"// Auto-generated by validation_suite for scenario '{scenario.name}'",
        f"// {scenario.description}",
        "",
        SLIM_STATS_EMIT.strip() if include_stats else SLIM_RUNTIME_EMIT.strip(),
        "",
        "initialize() {",
        f'    defineConstant("N_POP", {scenario.N});',
        f'    defineConstant("NGEN", {scenario.G});',
        f'    defineConstant("GENOME_BP", {scenario.total_genome_length});',
        f'    defineConstant("NUM_MUTATION_TYPES", {len(scenario.mutation_types)});',
        f'    defineConstant("NUM_CHROMOSOMES", {len(scenario.chromosomes)});',
        "",
    ]

    body_lines.extend(_slim_mutation_type_lines(scenario, mutation_type_var_names))
    body_lines.append("")
    body_lines.extend(
        _slim_genomic_element_type_lines(
            scenario,
            mutation_type_var_names,
            region_type_var_names,
        )
    )
    body_lines.append("")

    for chromosome_index, chromosome in enumerate(scenario.chromosomes, start=1):
        body_lines.append(
            f'    initializeChromosome({chromosome_index}, {chromosome.length}, type="A");'
        )
        body_lines.extend(_slim_mutation_rate_lines(scenario, chromosome))
        body_lines.extend(_slim_region_lines(scenario, chromosome, region_type_var_names))
        body_lines.extend(_slim_recombination_rate_lines(scenario, chromosome))
        if chromosome_index != len(scenario.chromosomes):
            body_lines.append("")
    body_lines.extend(
        [
            "}",
            "",
            "1 early() {",
            '    sim.addSubpop("p1", N_POP);',
            '    defineGlobal("t_start", clock("mono"));',
            "    community.rescheduleScriptBlock(s1, NGEN + 1, NGEN + 1);",
            "}",
            "",
            "s1 2 early() {",
            '    wall = clock("mono") - t_start;',
            "    " + ("emit_stats(wall);" if include_stats else "emit_runtime(wall);"),
            "    sim.simulationFinished();",
            "}",
            "",
        ]
    )
    return "\n".join(body_lines)


def single_chrom_scenario(
    *,
    name: str,
    description: str,
    N: int,
    G: int,
    L: int,
    mu: float,
    rho: float,
    mutation_type: MutationTypeSpec,
    mutation_scale: float = 1.0,
    recombination_intervals: tuple[RecIntervalSpec, ...] = (),
) -> Scenario:
    region_type = RegionTypeSpec("whole_genome", mutation_scale, ((mutation_type.name, 1.0),))
    chromosome = ChromosomeSpec(
        name="chr1",
        length=L,
        recombination_intervals=recombination_intervals,
        regions=(ChromosomeRegionSpec(region_type.name, 0, L),),
    )
    return Scenario(
        name=name,
        description=description,
        N=N,
        G=G,
        mu=mu,
        rho=rho,
        mutation_types=(mutation_type,),
        region_types=(region_type,),
        chromosomes=(chromosome,),
    )


def multitype_region_scenario(
    *,
    name: str,
    description: str,
    N: int,
    G: int,
    L: int,
    mu: float,
    rho: float,
    deleterious_type: MutationTypeSpec,
) -> Scenario:
    neutral_type = fixed_type("neutral", 0.0, 0.5)
    exon = RegionTypeSpec(
        "exon",
        mutation_scale=2.0,
        weights=((deleterious_type.name, 0.8), (neutral_type.name, 0.2)),
    )
    intron = RegionTypeSpec(
        "intron",
        mutation_scale=0.6,
        weights=((deleterious_type.name, 0.25), (neutral_type.name, 0.75)),
    )
    hotspot = RegionTypeSpec(
        "hotspot",
        mutation_scale=3.5,
        weights=((deleterious_type.name, 0.9), (neutral_type.name, 0.1)),
    )
    chromosome = ChromosomeSpec(
        name="chr1",
        length=L,
        recombination_intervals=(),
        regions=(
            ChromosomeRegionSpec("exon", 0, L // 5),
            ChromosomeRegionSpec("intron", L // 5, (4 * L) // 5),
            ChromosomeRegionSpec("hotspot", (4 * L) // 5, L),
        ),
    )
    return Scenario(
        name=name,
        description=description,
        N=N,
        G=G,
        mu=mu,
        rho=rho,
        mutation_types=(deleterious_type, neutral_type),
        region_types=(exon, intron, hotspot),
        chromosomes=(chromosome,),
    )


def comprehensive_two_chrom_scenario(
    *,
    name: str,
    description: str,
    N: int,
    G: int,
    mu: float,
    rho: float,
) -> Scenario:
    neutral_type = fixed_type("neutral", 0.0, 0.5)
    deleterious_type = gamma_type("deleterious", -0.015, 0.3, 0.2)

    neutral_region = RegionTypeSpec(
        "neutral_region",
        mutation_scale=0.6,
        weights=((neutral_type.name, 1.0),),
    )
    mixed_region = RegionTypeSpec(
        "mixed_region",
        mutation_scale=1.5,
        weights=((deleterious_type.name, 0.85), (neutral_type.name, 0.15)),
    )

    chr1 = ChromosomeSpec(
        name="chr1",
        length=60000,
        recombination_intervals=(
            RecIntervalSpec(0, 20000, 0.5),
            RecIntervalSpec(20000, 40000, 2.0),
            RecIntervalSpec(40000, 60000, 0.5),
        ),
        regions=(
            ChromosomeRegionSpec("neutral_region", 0, 10000),
            ChromosomeRegionSpec("mixed_region", 10000, 50000),
            ChromosomeRegionSpec("neutral_region", 50000, 60000),
        ),
    )
    chr2 = ChromosomeSpec(
        name="chr2",
        length=40000,
        recombination_intervals=(RecIntervalSpec(0, 40000, 1.0),),
        regions=(
            ChromosomeRegionSpec("neutral_region", 0, 10000),
            ChromosomeRegionSpec("mixed_region", 10000, 40000),
        ),
    )
    return Scenario(
        name=name,
        description=description,
        N=N,
        G=G,
        mu=mu,
        rho=rho,
        mutation_types=(neutral_type, deleterious_type),
        region_types=(neutral_region, mixed_region),
        chromosomes=(chr1, chr2),
    )


FULL_ACCURACY_CASES: tuple[SuiteCase, ...] = (
    SuiteCase(
        "baseline_fixed",
        single_chrom_scenario(
            name="baseline_fixed",
            description="Baseline fixed-effect deleterious model.",
            N=500,
            G=100,
            L=100000,
            mu=1e-6,
            rho=0.1,
            mutation_type=fixed_type("deleterious", -0.01, 0.5),
        ),
        replicates=4,
    ),
    SuiteCase(
        "larger_population",
        single_chrom_scenario(
            name="larger_population",
            description="Same biology as baseline, but with larger population size.",
            N=800,
            G=80,
            L=100000,
            mu=1e-6,
            rho=0.1,
            mutation_type=fixed_type("deleterious", -0.01, 0.5),
        ),
        replicates=4,
    ),
    SuiteCase(
        "longer_genome",
        single_chrom_scenario(
            name="longer_genome",
            description="Longer mutational target with matched simple DFE.",
            N=400,
            G=80,
            L=1000000,
            mu=2e-7,
            rho=0.8,
            mutation_type=fixed_type("deleterious", -0.01, 0.5),
        ),
        replicates=4,
    ),
    SuiteCase(
        "higher_mutation_rate",
        single_chrom_scenario(
            name="higher_mutation_rate",
            description="Higher mutation rate regime.",
            N=400,
            G=80,
            L=100000,
            mu=3e-6,
            rho=0.1,
            mutation_type=fixed_type("deleterious", -0.01, 0.5),
        ),
        replicates=4,
    ),
    SuiteCase(
        "higher_recombination_rate",
        single_chrom_scenario(
            name="higher_recombination_rate",
            description="Higher recombination intensity regime.",
            N=500,
            G=100,
            L=100000,
            mu=1e-6,
            rho=2.0,
            mutation_type=fixed_type("deleterious", -0.01, 0.5),
        ),
        replicates=4,
    ),
    SuiteCase(
        "gamma_dfe",
        single_chrom_scenario(
            name="gamma_dfe",
            description="Gamma-distributed DFE with partial recessivity.",
            N=500,
            G=100,
            L=100000,
            mu=1e-6,
            rho=0.1,
            mutation_type=gamma_type("deleterious", -0.01, 0.3, 0.25),
        ),
        replicates=4,
    ),
    SuiteCase(
        "normal_dfe",
        single_chrom_scenario(
            name="normal_dfe",
            description="Normal DFE.",
            N=500,
            G=100,
            L=100000,
            mu=1e-6,
            rho=0.1,
            mutation_type=normal_type("deleterious", -0.01, 0.005, 0.5),
        ),
        replicates=4,
    ),
    SuiteCase(
        "exponential_dfe",
        single_chrom_scenario(
            name="exponential_dfe",
            description="Exponential DFE.",
            N=500,
            G=100,
            L=100000,
            mu=1e-6,
            rho=0.1,
            mutation_type=exponential_type("deleterious", -0.01, 0.5),
        ),
        replicates=4,
    ),
    SuiteCase(
        "partial_recessive",
        single_chrom_scenario(
            name="partial_recessive",
            description="Fixed-effect deleterious mutations with h=0.1.",
            N=500,
            G=100,
            L=100000,
            mu=1e-6,
            rho=0.1,
            mutation_type=fixed_type("deleterious", -0.01, 0.1),
        ),
        replicates=4,
    ),
    SuiteCase(
        "multi_type_regions",
        multitype_region_scenario(
            name="multi_type_regions",
            description="Multiple mutation types and region-specific mutation scales.",
            N=500,
            G=100,
            L=100000,
            mu=1e-6,
            rho=0.15,
            deleterious_type=fixed_type("deleterious", -0.02, 0.25),
        ),
        replicates=4,
    ),
    SuiteCase(
        "heterogeneous_recombination",
        single_chrom_scenario(
            name="heterogeneous_recombination",
            description="Single chromosome with heterogeneous recombination map.",
            N=500,
            G=100,
            L=100000,
            mu=1e-6,
            rho=0.2,
            mutation_type=fixed_type("deleterious", -0.01, 0.5),
            recombination_intervals=(
                RecIntervalSpec(0, 50000, 0.1),
                RecIntervalSpec(50000, 100000, 1.9),
            ),
        ),
        replicates=4,
    ),
    SuiteCase(
        "comprehensive_multichrom",
        comprehensive_two_chrom_scenario(
            name="comprehensive_multichrom",
            description="Two chromosomes, mixed regions, and heterogeneous recombination.",
            N=500,
            G=100,
            mu=1e-6,
            rho=0.2,
        ),
        replicates=4,
    ),
)


FULL_SPEED_CASES: tuple[SuiteCase, ...] = (
    SuiteCase(
        "small_fixed",
        single_chrom_scenario(
            name="small_fixed",
            description="Small fixed-effect speed case.",
            N=2000,
            G=150,
            L=100000,
            mu=1e-7,
            rho=0.1,
            mutation_type=fixed_type("deleterious", -0.01, 0.5),
        ),
        replicates=3,
    ),
    SuiteCase(
        "pop_10k",
        single_chrom_scenario(
            name="pop_10k",
            description="Medium population benchmark.",
            N=10000,
            G=60,
            L=1000000,
            mu=1e-8,
            rho=1.0,
            mutation_type=fixed_type("deleterious", -0.01, 0.5),
        ),
        replicates=3,
    ),
    SuiteCase(
        "pop_50k",
        single_chrom_scenario(
            name="pop_50k",
            description="Larger population benchmark.",
            N=50000,
            G=40,
            L=1000000,
            mu=1e-8,
            rho=1.0,
            mutation_type=fixed_type("deleterious", -0.01, 0.5),
        ),
        replicates=2,
    ),
    SuiteCase(
        "pop_100k",
        single_chrom_scenario(
            name="pop_100k",
            description="Heavy population-size benchmark.",
            N=100000,
            G=30,
            L=1000000,
            mu=1e-8,
            rho=1.0,
            mutation_type=fixed_type("deleterious", -0.01, 0.5),
        ),
        replicates=2,
    ),
    SuiteCase(
        "long_genome",
        single_chrom_scenario(
            name="long_genome",
            description="Long-genome benchmark.",
            N=20000,
            G=30,
            L=5000000,
            mu=5e-9,
            rho=2.0,
            mutation_type=fixed_type("deleterious", -0.01, 0.5),
        ),
        replicates=2,
    ),
    SuiteCase(
        "high_mutation_rate",
        single_chrom_scenario(
            name="speed_high_mutation_rate",
            description="Higher mutation-rate speed benchmark.",
            N=20000,
            G=40,
            L=1000000,
            mu=5e-8,
            rho=1.0,
            mutation_type=fixed_type("deleterious", -0.01, 0.5),
        ),
        replicates=2,
    ),
    SuiteCase(
        "high_recombination_rate",
        single_chrom_scenario(
            name="speed_high_recombination_rate",
            description="Higher recombination-rate speed benchmark.",
            N=20000,
            G=40,
            L=1000000,
            mu=1e-8,
            rho=5.0,
            mutation_type=fixed_type("deleterious", -0.01, 0.5),
        ),
        replicates=2,
    ),
    SuiteCase(
        "gamma_mix",
        multitype_region_scenario(
            name="speed_gamma_mix",
            description="Mixed-region gamma-DFE benchmark.",
            N=20000,
            G=40,
            L=1000000,
            mu=1e-8,
            rho=1.0,
            deleterious_type=gamma_type("deleterious", -0.015, 0.3, 0.25),
        ),
        replicates=2,
    ),
    SuiteCase(
        "pop_250k",
        single_chrom_scenario(
            name="speed_pop_250k",
            description="Large-population SLiM comparison probe.",
            N=250000,
            G=24,
            L=2000000,
            mu=8e-9,
            rho=1.0,
            mutation_type=fixed_type("deleterious", -0.01, 0.5),
        ),
        replicates=2,
    ),
    SuiteCase(
        "pop_500k",
        single_chrom_scenario(
            name="speed_pop_500k",
            description="Very large-population SLiM comparison probe.",
            N=500000,
            G=18,
            L=2000000,
            mu=8e-9,
            rho=1.0,
            mutation_type=fixed_type("deleterious", -0.01, 0.5),
        ),
        replicates=2,
    ),
)


FULL_SCALING_CASES: tuple[SuiteCase, ...] = (
    SuiteCase("medium_fixed", FULL_SPEED_CASES[1].scenario, 2),
    SuiteCase("large_fixed", FULL_SPEED_CASES[3].scenario, 2),
    SuiteCase("large_high_rho", FULL_SPEED_CASES[6].scenario, 2),
    SuiteCase(
        "xlarge_long_genome",
        single_chrom_scenario(
            name="xlarge_long_genome",
            description="Scaling-focused case beyond practical SLiM benchmarking.",
            N=1000000,
            G=12,
            L=20000000,
            mu=2e-9,
            rho=3.0,
            mutation_type=fixed_type("deleterious", -0.01, 0.5),
        ),
        2,
    ),
    SuiteCase(
        "xlarge_gamma_mix",
        multitype_region_scenario(
            name="xlarge_gamma_mix",
            description="Million-scale mixed-region gamma DFE scaling case.",
            N=1000000,
            G=12,
            L=5000000,
            mu=1e-8,
            rho=1.0,
            deleterious_type=gamma_type("deleterious", -0.015, 0.3, 0.25),
        ),
        2,
    ),
    SuiteCase(
        "ultra_population",
        single_chrom_scenario(
            name="ultra_population",
            description="Two-million diploid scaling case emphasizing population size.",
            N=2000000,
            G=8,
            L=2000000,
            mu=3e-9,
            rho=1.0,
            mutation_type=fixed_type("deleterious", -0.01, 0.5),
        ),
        2,
    ),
)


FULL_PROFILE_CASES: tuple[SuiteCase, ...] = (
    SuiteCase("profile_large_high_rho", FULL_SPEED_CASES[6].scenario, 2),
    SuiteCase("profile_xlarge_long_genome", FULL_SCALING_CASES[3].scenario, 2),
    SuiteCase("profile_xlarge_gamma_mix", FULL_SCALING_CASES[4].scenario, 2),
    SuiteCase("profile_ultra_population", FULL_SCALING_CASES[5].scenario, 2),
)


EXTREME_SCALING_CASES: tuple[SuiteCase, ...] = (
    SuiteCase("ultra_population", FULL_SCALING_CASES[5].scenario, 1),
    SuiteCase(
        "pop_5m",
        single_chrom_scenario(
            name="extreme_pop_5m",
            description="Five-million diploid sparqy-only population scaling case.",
            N=5000000,
            G=6,
            L=1000000,
            mu=1e-9,
            rho=1.0,
            mutation_type=fixed_type("deleterious", -0.01, 0.5),
        ),
        1,
    ),
    SuiteCase(
        "pop_10m",
        single_chrom_scenario(
            name="extreme_pop_10m",
            description="Ten-million diploid sparqy-only population scaling case.",
            N=10000000,
            G=6,
            L=1000000,
            mu=1e-9,
            rho=1.0,
            mutation_type=fixed_type("deleterious", -0.01, 0.5),
        ),
        1,
    ),
    SuiteCase(
        "pop_25m",
        single_chrom_scenario(
            name="extreme_pop_25m",
            description="Twenty-five-million diploid sparqy-only population scaling case.",
            N=25000000,
            G=4,
            L=1000000,
            mu=1e-9,
            rho=1.0,
            mutation_type=fixed_type("deleterious", -0.01, 0.5),
        ),
        1,
    ),
    SuiteCase(
        "pop_50m",
        single_chrom_scenario(
            name="extreme_pop_50m",
            description="Fifty-million diploid sparqy-only population scaling case.",
            N=50000000,
            G=4,
            L=1000000,
            mu=1e-9,
            rho=1.0,
            mutation_type=fixed_type("deleterious", -0.01, 0.5),
        ),
        1,
    ),
)


EXTREME_PROFILE_CASES: tuple[SuiteCase, ...] = (
    SuiteCase("profile_ultra_population", FULL_SCALING_CASES[5].scenario, 1),
    SuiteCase("profile_pop_5m", EXTREME_SCALING_CASES[1].scenario, 1),
    SuiteCase("profile_pop_10m", EXTREME_SCALING_CASES[2].scenario, 1),
    SuiteCase("profile_pop_25m", EXTREME_SCALING_CASES[3].scenario, 1),
    SuiteCase("profile_pop_50m", EXTREME_SCALING_CASES[4].scenario, 1),
)


SMOKE_ACCURACY_CASES: tuple[SuiteCase, ...] = (
    SuiteCase(
        "smoke_baseline",
        single_chrom_scenario(
            name="smoke_baseline",
            description="Tiny baseline smoke-test scenario.",
            N=80,
            G=15,
            L=20000,
            mu=2e-6,
            rho=0.2,
            mutation_type=fixed_type("deleterious", -0.01, 0.5),
        ),
        replicates=1,
    ),
    SuiteCase(
        "smoke_multitype",
        multitype_region_scenario(
            name="smoke_multitype",
            description="Tiny mixed-region smoke-test scenario.",
            N=80,
            G=15,
            L=20000,
            mu=2e-6,
            rho=0.2,
            deleterious_type=fixed_type("deleterious", -0.02, 0.25),
        ),
        replicates=1,
    ),
)


SMOKE_SPEED_CASES: tuple[SuiteCase, ...] = (
    SuiteCase(
        "smoke_speed",
        single_chrom_scenario(
            name="smoke_speed",
            description="Small speed smoke-test case.",
            N=500,
            G=20,
            L=50000,
            mu=5e-7,
            rho=0.3,
            mutation_type=fixed_type("deleterious", -0.01, 0.5),
        ),
        replicates=1,
    ),
)


SMOKE_SCALING_CASES: tuple[SuiteCase, ...] = (
    SuiteCase("smoke_scaling", SMOKE_SPEED_CASES[0].scenario, 1),
)


SMOKE_PROFILE_CASES: tuple[SuiteCase, ...] = (
    SuiteCase("smoke_profile", SMOKE_SPEED_CASES[0].scenario, 1),
)


FULL_PRESET = SuitePreset(
    name="full",
    description=(
        "Research-grade Perlmutter preset covering statistical agreement, "
        "SLiM speed comparisons, sparqy scaling, and sparqy profiling, "
        "including million-scale sparqy-only workloads."
    ),
    accuracy_cases=FULL_ACCURACY_CASES,
    speed_cases=FULL_SPEED_CASES,
    scaling_cases=FULL_SCALING_CASES,
    profile_cases=FULL_PROFILE_CASES,
    accuracy_variants=(
        AccuracyVariant("sequential", 1),
        AccuracyVariant("parallel", 128),
        AccuracyVariant("parallel_psa_plus", 128),
    ),
    speed_threads=(1, 32, 128),
    scaling_threads=(1, 2, 4, 8, 16, 32, 64, 128),
    profile_threads=(1, 32, 128),
)


SMOKE_PRESET = SuitePreset(
    name="smoke",
    description="Tiny local smoke-test preset exercising all pipeline stages quickly.",
    accuracy_cases=SMOKE_ACCURACY_CASES,
    speed_cases=SMOKE_SPEED_CASES,
    scaling_cases=SMOKE_SCALING_CASES,
    profile_cases=SMOKE_PROFILE_CASES,
    accuracy_variants=(
        AccuracyVariant("sequential", 1),
        AccuracyVariant("parallel_psa_plus", 2),
    ),
    speed_threads=(1, 2),
    scaling_threads=(1, 2, 4),
    profile_threads=(1, 2, 4),
)


EXTREME_PRESET = SuitePreset(
    name="extreme",
    description=(
        "sparqy-only Perlmutter stress preset for very large scaling and profiling "
        "runs up to N=50,000,000."
    ),
    accuracy_cases=(),
    speed_cases=(),
    scaling_cases=EXTREME_SCALING_CASES,
    profile_cases=EXTREME_PROFILE_CASES,
    accuracy_variants=(),
    speed_threads=(1,),
    scaling_threads=(1, 8, 16, 32, 64, 128),
    profile_threads=(128,),
    scaling_threads_by_case={
        "pop_25m": (1, 8, 16, 32, 64),
        "pop_50m": (1, 8, 16, 32, 64),
    },
    profile_threads_by_case={
        "profile_pop_25m": (64,),
        "profile_pop_50m": (64,),
    },
)


PRESETS = {
    "extreme": EXTREME_PRESET,
    "full": FULL_PRESET,
    "smoke": SMOKE_PRESET,
}


def get_preset(name: str) -> SuitePreset:
    try:
        return PRESETS[name]
    except KeyError as exc:
        raise KeyError(
            f"Unknown preset '{name}'. Available presets: {', '.join(sorted(PRESETS))}"
        ) from exc


def all_unique_scenarios(preset: SuitePreset) -> dict[str, Scenario]:
    scenarios: dict[str, Scenario] = {}
    for collection in (
        preset.accuracy_cases,
        preset.speed_cases,
        preset.scaling_cases,
        preset.profile_cases,
    ):
        for case in collection:
            scenarios[case.scenario.name] = case.scenario
    return scenarios


def scenario_with_overrides(scenario: Scenario, **overrides: Any) -> Scenario:
    updated = replace(scenario, **overrides)
    validate_scenario(updated)
    return updated
