# sparqy

**sparqy** (SPArse Recombination, QuicklY) is an OpenMP-parallel forward-time Wright–Fisher simulator for
population genetics. It models mutation, selection, dominance, and
recombination on (multi-chromosome) diploid genomes and is designed to scale
from a laptop to many-core compute nodes without changing the model spec.

The simulator is written in C++17, has no runtime dependencies beyond
OpenMP, and can be driven from the command line either through a positional
benchmark interface or through a declarative configuration file that
exposes the full feature set.

## Features

### Biological model

- **Multiple mutation types**, each with a custom distribution of fitness
  effects drawn from any of six parametric families
  (`constant`, `uniform`, `normal`, `exponential`, `gamma`, `beta`),
  optionally clamped to a user-specified `[min, max]` range, and one of
  four dominance models: additive, fixed coefficient `h`,
  distribution-sampled, or linear-from-`s`.
- **Multiple chromosomes** with independent, heterogeneous recombination
  maps specified as piecewise-constant `(start, end, rate)` intervals.
- **Reusable region types** (e.g. `exon`, `intron`, `hotspot`) that mix
  mutation types under arbitrary weights and apply per-region mutation-rate
  scaling. Region types are defined once and placed at chromosome-local
  coordinates.
- **Compiled CDFs** for O(log R) weighted sampling of mutation locations
  across regions.
- **Configurable per-generation statistics**, including mean fitness,
  genetic load, realized masking bonus, inbreeding load, segregating- and
  fixed-site counts, mutation histograms by type and chromosome, site
  frequency spectra (folded and unfolded), nucleotide diversity, expected
  heterozygosity, and mean pairwise haplotypic similarity under three
  metrics (Jaccard, Dice, overlap-coefficient) computable in the same
  generation with shared intermediates.

### Performance

- **Offspring-block parallel decomposition.** Each thread independently
  produces a contiguous block of `N/P` offspring into thread-local scratch
  buffers; the parent population is read-only during reproduction, so the
  hot loop never takes a lock.
- **Cache-line-aligned per-thread scratch** (`alignas(64)`) prevents
  false sharing on the vector headers each thread mutates.
- **Lock-free mutation ID allocation.** Recyclable IDs are pre-distributed
  to per-thread stacks before the parallel region; threads fall back to a
  shared atomic cursor only after exhausting their local pool. Unused
  pre-distributed IDs are returned to the global pool after the parallel
  region so they cannot leak across generations.
- **Three alias-table builders** for parent sampling, plus a smart
  default `auto` mode selectable via `--alias-builder=`:
    - `auto` — chooses `sequential` for single-thread runs and for
      `N < 10000`, and `parallel_psa_plus` otherwise.
    - `sequential` — single-threaded Vose construction.
    - `parallel` — the parallel split-and-pack (PSA) algorithm of
      Hübschle-Schneider & Sanders (2022, Algorithm 3).
    - `parallel_psa_plus` — the PSA+ refinement (op. cit. §4.2.1):
      a per-thread greedy local pre-pass that resolves balanced items
      inside each thread's input chunk before invoking PSA on the
      remainder. Typically 1.2–1.4× faster than PSA on realistic fitness
      distributions; can be slower than PSA on adversarial inputs (e.g.
      one extreme outlier among uniform weights), which is why `auto`
      only selects it once the job is large enough to amortize the extra
      setup cost.
- **Deferred sparse counting.** Each thread inserts mutation IDs into a
  thread-local open-addressed `SparseCountMap` in a separate pass *after*
  the recombination merge, so the hash-map probe traffic never pollutes
  L1 during the recombination hot loop.
- **Fused fitness evaluation** keeps offspring mutation data hot in L1
  during fitness computation.
- **Per-thread `xoshiro256**` RNG streams** seeded by `long_jump()`
  advances of 2^192 draws, guaranteeing non-overlapping sequences for any
  realistic simulation length.

## Requirements

- A C++17 compiler with OpenMP support (GCC 9+, Clang 10+, or Apple Clang
  with libomp).
- CMake 3.16 or newer.
- **macOS**: libomp installed via Homebrew at `/opt/homebrew/opt/libomp`.
  CMake locates it automatically through the `if(APPLE)` branch in
  `CMakeLists.txt`.
- **Validation suite (optional)**: Python 3.9+, NumPy, Matplotlib, and a
  SLiM 5.x binary for cross-simulator comparison.

## Installation

```bash
git clone https://github.com/chris-a-talbot/sparqy sparqy
cd sparqy
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

This produces three main binaries under `build/`:

- `build/sparqy` — the simulator.
- `build/alias_bench` — a standalone microbenchmark and correctness
  harness for the three alias-table builders.
- `build/config_loader_bench` — a microbenchmark for config-file loading
  overhead.

The default release flags are
`-O3 -march=native -mtune=native -ffast-math -funroll-loops`. For
deployment to a homogeneous cluster, override `-march` to the target
microarchitecture (for example `-march=znver3` for AMD EPYC Milan).

## Quick start

```bash
# Single-chromosome benchmark model: N=10000, L=1Mbp, mu=1e-8, rho=1.0,
# s=-0.01, G=100 generations, output every 50 generations, h=0.5,
# seed=42, 4 threads.
./build/sparqy 10000 1000000 1e-8 1.0 -0.01 100 50 0.5 42 4

# Same run, with the segregating-site count and mean fitness reported
# every generation.
./build/sparqy 10000 1000000 1e-8 1.0 -0.01 100 50 0.5 42 4 \
    --stats=mean_fitness,n_seg

# Full biological model from a configuration file.
./build/sparqy --config examples/full_power_model.sparqy
```

Setting `threads=0` uses `omp_get_max_threads()`, which respects the
`OMP_NUM_THREADS` environment variable.

## Usage

The executable provides two command-line modes.

### 1. Positional mode

For the single-chromosome, single-mutation-type benchmark model:

```text
sparqy N L mu rho s G out_interval h seed threads [flags]
```

| Argument        | Meaning                                                  |
|-----------------|----------------------------------------------------------|
| `N`             | Diploid population size                                  |
| `L`             | Chromosome length in base pairs                          |
| `mu`            | Per-base, per-generation mutation rate                   |
| `rho`           | Mean number of crossovers per chromosome per meiosis     |
| `s`             | Selection coefficient (negative for deleterious)         |
| `G`             | Number of generations to simulate                        |
| `out_interval`  | Generations between summary outputs                      |
| `h`             | Dominance coefficient                                    |
| `seed`          | RNG seed (a 64-bit integer)                              |
| `threads`       | OpenMP thread count; `0` means `omp_get_max_threads()`   |

### 2. Config mode

For the full biological model:

```text
sparqy --config path/to/model.sparqy [--profile] [--alias-builder=MODE]
```

The configuration language is described below.

### Flags

- `--stats=LIST` — positional-mode only. It enables selected statistics
  for the legacy benchmark model (or `--stats=all`). Config mode does
  **not** accept this flag; declare statistics in `stats <- list(...)`
  inside the config file. Recognized names: `mean_fitness`,
  `genetic_load`, `realized_masking_bonus`, `exact_B`,
  `pairwise_similarity`, `n_seg`, `n_fixed`, `genome_words`,
  `mutation_histogram`, `site_frequency_spectrum`,
  `nucleotide_diversity`, `expected_heterozygosity`.
- `--profile` — record per-phase wall-clock time for every generation and
  print a summary to `stderr` after the run completes. The standard output
  remains machine-readable.
- `--alias-builder=MODE` — selects the parent-sampling alias-table
  construction algorithm. `MODE` is one of `auto` (default),
  `sequential` (single-threaded Vose), `parallel` (PSA), or
  `parallel_psa_plus` (PSA+). `auto` chooses `sequential` for
  single-thread runs and for `N < 10000`, and `parallel_psa_plus`
  otherwise.

## Configuration file format

The config language is an R-style subset with exactly three top-level
sections:

- `constants <- list(...)`
- `config <- list(...)`
- `stats <- list(...)`

Comments start with `#`. Whitespace and line breaks are insignificant
inside `list(...)`, `c(...)`, and builder calls. The loader validates the
full script before the simulation starts; invalid configs do not run.
The complete, exhaustive reference lives in
[`CONFIG_REFERENCE.md`](CONFIG_REFERENCE.md).

Constants may be integers, floats, booleans (`TRUE` / `FALSE`), or
strings, and config fields may reference them by name. There are no
implicit defaults for required model fields in this language: they must
be written explicitly, and chromosome intervals and regions must cover
the full chromosome without gaps or overlaps. The optional
`config$runtime <- list(...)` block controls how the simulator runs, such
as alias-builder mode and profiling defaults.

```text
constants <- list(
  pop_size = 1000,
  generation_count = 100,
  mutation_rate = 1e-7,
  recombination_rate = 0.5,
  run_seed = 42,
  thread_count = 0,
  default_alias_builder = "auto",
  default_profile = FALSE
)

config <- list(
  N = pop_size,
  G = generation_count,
  mu = mutation_rate,
  rho = recombination_rate,
  seed = run_seed,
  threads = thread_count,
  runtime = list(
    alias_builder = default_alias_builder,
    profile = default_profile
  ),
  mutation_types = list(
    deleterious = list(
      selection = gamma(-0.02, 0.3, -0.999999999, 0.0),
      dominance = linear_from_s(0.5, -5.0, 0.0, 0.5)
    ),
    neutral = list(
      selection = constant(0.0),
      dominance = additive()
    )
  ),
  region_types = list(
    exon = list(
      mutation_scale = 2.5,
      weights = c(deleterious = 8, neutral = 1)
    ),
    intron = list(
      mutation_scale = 0.5,
      weights = c(deleterious = 1, neutral = 4)
    )
  ),
  chromosomes = list(
    chr1 = list(
      length = 100000,
      recombination_intervals = list(
        interval(0, 60000, 1.0),
        interval(60000, 100000, 0.4)
      ),
      regions = list(
        region(exon, 0, 20000),
        region(intron, 20000, 100000)
      )
    )
  )
)

stats <- list(
  up_to(end = 5, mean_fitness),
  every(10, genetic_load),
  at(c(50, 100), mutation_histogram),
  at_after(start = 90, pairwise_similarity(jaccard))
)
```

A complete worked example is provided at `examples/full_power_model.sparqy`.
Every checked-in config example also has a paired `SLiM 5.x` script under
`examples/slim/`; see `examples/slim/README.md` for the mapping and
comparison workflow.

### Required config fields

- `constants` may be empty, but it must be present as `list(...)`.
- `config` must define `N`, `G`, `mu`, `rho`, `seed`, `threads`,
  `mutation_types`, `region_types`, and `chromosomes`.
- `config` may also define an optional `runtime` block with runtime-only
  controls such as `alias_builder` and `profile`.
- Every mutation type must define `selection` and `dominance`.
- Every region type must define `mutation_scale` and `weights`.
- Every chromosome must define `length`, `recombination_intervals`, and
  `regions`.

### Builder calls

- Distribution builders:
  `constant(value[, min, max])`, `uniform(min, max[, clamp_min, clamp_max])`,
  `normal(mean, sd, min, max)`, `exponential(mean, min, max)`,
  `gamma(mean, shape, min, max)`, `beta(alpha, beta, min, max)`.
- Dominance builders:
  `additive()`, `fixed(h)`, `distributed(<distribution>)`,
  `linear_from_s(intercept, slope, min_h, max_h)`.
- Genome builders:
  `interval(start, end, rate_scale)`, `region(region_type, start, end)`.
- Stats schedules:
  `always(stat, ...)`, `every(step, stat, ...)`,
  `at(generation_or_c_vector, stat, ...)`, `up_to(end, stat, ...)`,
  `at_after(start, stat, ...)`, `every_up_to(step, end, stat, ...)`,
  `every_at_after(step, start, stat, ...)`, `range(start, end, stat, ...)`,
  `every_range(step, start, end, stat, ...)`.
- Pairwise similarity uses
  `pairwise_similarity(jaccard)`, `pairwise_similarity(dice)`, or
  `pairwise_similarity(overlap)`.

Config-mode output is tab-separated with one row per computed statistic:

```text
generation  dt_sec  cumul_sec  statistic  metric  scalar_value  by_type  by_chromosome  unfolded_sfs  folded_sfs
```

Scalar statistics populate the `scalar_value` column; histogram and
site-frequency-spectrum statistics populate their respective vector
columns as comma-separated integer lists.

## Paired SLiM examples

For each checked-in `examples/*.sparqy` config, the repository now
includes a paired `examples/slim/*.slim` script that mirrors the same
model as closely as practical in `SLiM 5.x`. The paired scripts emit the
same statistic names and output columns as `sparqy`, which makes quick
side-by-side inspection straightforward.

For a repo-local comparison across all example pairs:

```bash
python3 validation_suite/compare_example_pairs.py \
    --sparqy-bin ./build/sparqy \
    --slim-bin slim
```

`examples/slim/full_power_model.slim` uses sampled whole-genome
pairwise-similarity estimates rather than exhaustive all-pairs
calculation so it remains practical as an example script; the smaller
paired examples use exact pairwise calculations.

## Validation

A repository-local validation suite lives under `validation_suite/` and is
driven by `run_suite.py`. It covers four tracks:

- `accuracy` — biological agreement against [SLiM](https://messerlab.org/slim/) on matched scenarios.
- `speed` — matched wall-clock benchmarking against SLiM across all three
  alias builders.
- `scaling` — sparqy-only thread-scaling curves, including population
  sizes too large for SLiM comparison.
- `profile` — per-phase runtime breakdowns from `--profile` runs.

Larger presets (`full`, `extreme`) are intended for HPC nodes; wrapper
scripts for SLURM-based clusters are provided as
`validation_suite/perlmutter_full_suite.sh` and
`validation_suite/perlmutter_extreme_suite.sh`. Per-process peak resident
memory is recorded automatically on Linux via `/proc/<pid>/status`.

## Reproducibility

- **Single-thread runs are deterministic** for a given `seed`, model
  specification, and binary.
- **Multi-thread runs are statistically equivalent but not bit-identical
  across thread counts**, because the order in which threads consume the
  RNG streams depends on offspring-block assignment. This is intrinsic to
  the parallel decomposition; do not file it as a bug.
- Because each thread RNG is reached by a `long_jump()` of 2^192 draws,
  thread streams are guaranteed not to overlap for any realistic
  simulation length.

## Architecture overview

The implementation is built around three data representations.

1. **Per-mutation metadata tables.** Every distinct mutation has a single
   entry in arrays indexed by mutation ID: locus, homozygous and
   heterozygous fitness factors, masking coefficient, and mutation type.
   These tables are grown serially before each parallel region and are
   read-only during reproduction.
2. **Packed populations.** A population is a flat `mutation_ids` array
   plus a `haplotype_offsets` index. Haplotype `h` occupies the sorted
   slice `[offsets[h], offsets[h+1])`. Recombination is implemented as a
   merge over two such slices.
3. **Per-thread offspring-block scratch.** Every mutable buffer touched
   in the hot loop — recombination scratch, new-ID lists, count maps,
   classification buffers — lives in a thread-local struct. No thread
   ever writes to another thread's scratch.

Each generation cycles through a fixed sequence of serial and parallel
phases: metadata reservation, alias-table construction, recyclable-ID
distribution, parallel reproduction with fused fitness, ID reclamation,
prefix-sum over offsets, parallel block copy, thread-local counting,
serial count merge, parallel classification of mutations as lost / fixed
/ surviving, and parallel reductions.

## Acknowledgments

The parallel parent-sampler implementations (`--alias-builder=parallel`
and `--alias-builder=parallel_psa_plus`) are direct adaptations of the
parallel split-and-pack alias-construction algorithm (PSA) and its
greedy-pre-pass refinement (PSA+) introduced by **Lorenz
Hübschle-Schneider and Peter Sanders** in
*Parallel Weighted Random Sampling* (ACM TOMS, 2022). Algorithm 3 of
that paper is implemented in `AliasSampler::build_parallel`, and
§4.2.1 is implemented in `AliasSampler::build_parallel_psa_plus`. Any
work that benchmarks or builds on the parallel parent-sampler in
sparqy should cite that paper in addition to citing sparqy itself.

The sequential alias-construction code follows Vose's (1991) linear-time
refinement of Walker's (1977) original alias method. The per-thread
random-number streams use the `xoshiro256**` generator of Blackman &
Vigna, advanced by `long_jump()` to guarantee non-overlapping
sub-sequences.

## License

sparqy is released under the MIT License. See [`LICENSE`](LICENSE) for the
full text.

## References

- Walker, A. J. (1977). An efficient method for generating discrete
  random variables with general distributions. *ACM Transactions on
  Mathematical Software*, 3(3), 253–256.
- Vose, M. D. (1991). A linear algorithm for generating random numbers
  with a given distribution. *IEEE Transactions on Software
  Engineering*, 17(9), 972–975.
- **Hübschle-Schneider, L., & Sanders, P. (2022).** Parallel weighted
  random sampling. *ACM Transactions on Mathematical Software*, 48(3),
  1–40. doi:10.1145/3549934.
- Blackman, D., & Vigna, S. (2021). Scrambled linear pseudorandom number
  generators. *ACM Transactions on Mathematical Software*, 47(4), 1–32.
  See also <https://prng.di.unimi.it/>.
- Haller, B. C., & Messer, P. W. (2019). SLiM 3: Forward genetic
  simulations beyond the Wright–Fisher model. *Molecular Biology and
  Evolution*, 36(3), 632–637.
