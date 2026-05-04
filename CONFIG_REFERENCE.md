# sparqy Config Reference

This file is the complete reference for `sparqy --config`.

The config language is a strict, R-flavored subset. It is meant to feel
like writing structured R data, but it is not a general R interpreter.
There is no arbitrary expression evaluation, no loops, no user-defined
functions, and no code execution.

## Scope

What the loader supports:

- Three top-level assignments: `constants`, `config`, and `stats`
- `list(...)` for structured objects
- `c(...)` for specific vector-like contexts
- Builder calls such as `gamma(...)`, `linear_from_s(...)`,
  `interval(...)`, `region(...)`, `every(...)`, and so on
- Scalar constants of type integer, float, boolean, or string
- Positional arguments, named arguments, or a mix of the two inside
  builder calls

What the loader does not support:

- Arbitrary R expressions such as `2 * x`, `x + y`, `ifelse(...)`, or
  `seq(...)`
- Constant-to-constant expressions or references inside `constants`
- Extra top-level sections
- Undeclared fields inside structured blocks
- Silent schema defaults other than the explicitly documented defaults in
  this reference

## File Structure

Every config file must define exactly these three top-level sections:

```r
constants <- list(...)
config <- list(...)
stats <- list(...)
```

Top-level assignment may use either `<-` or `=`:

```r
constants <- list()
config = list(...)
stats <- list()
```

No other top-level section names are allowed.

## Lexical Rules

- Comments start with `#` and continue to the end of the line.
- Whitespace is ignored except inside string literals.
- Strings may use single quotes or double quotes.
- Supported string escapes are `\\`, `\'`, `\"`, `\n`, `\t`, and `\r`.
- Booleans are `TRUE` and `FALSE`.
- Numeric literals may be integers or floats, and may use scientific
  notation such as `1e-7`.
- A leading `+` or `-` is allowed on numeric literals.
- Identifiers follow this shape:
  first character is a letter, `_`, or `.`
  later characters may also include digits

Examples of valid identifiers:

- `deleterious`
- `chr1`
- `_tmp`
- `.hidden_style`

## Argument Matching

Inside builder calls, arguments may be provided positionally, by name, or
with a mix of both:

```r
gamma(-0.02, 0.3, -0.999999999, 0.0)
gamma(mean = -0.02, shape = 0.3, min = -0.999999999, max = 0.0)
gamma(mean = -0.02, 0.3, min = -0.999999999, 0.0)
```

Duplicate named arguments are rejected.

For stats schedules, the control arguments may be named or positional, but
the statistics themselves must remain unnamed positional entries:

```r
every(10, mean_fitness)
every(step = 10, mean_fitness)
every_range(step = 5, start = 10, end = 50, n_seg)
```

## `constants`

`constants` must be a `list(...)`. It may be empty.

```r
constants <- list(
  pop_size = 1000,
  mutation_rate = 1e-7,
  use_profile = FALSE,
  builder_name = "parallel_psa_plus"
)
```

Rules:

- Each entry must be named.
- Duplicate constant names are rejected.
- Allowed constant value types are:
  integer
  float
  boolean
  string
- Constants are literal-only. A constant value cannot be another symbol,
  a builder call, or a nested `list(...)`.

Valid:

```r
constants <- list(x = 10, y = TRUE, label = "chr1")
```

Invalid:

```r
constants <- list(y = x)
constants <- list(z = 2 * x)
constants <- list(obj = list(a = 1))
```

Unused constants produce warnings before the simulation starts.

## `config`

`config` must be a `list(...)`.

Required fields:

| Field | Type | Meaning | Validation |
|---|---|---|---|
| `N` | integer | Diploid population size | must be positive |
| `G` | integer | Number of generations to simulate | must be positive |
| `mu` | numeric | Per-base mutation rate | must be non-negative |
| `rho` | numeric | Mean crossovers per chromosome per meiosis | must be non-negative |
| `seed` | integer | RNG seed | must be a non-negative integer |
| `threads` | integer | OpenMP thread count | must be a non-negative integer |
| `mutation_types` | `list(...)` | Named mutation-type definitions | must be non-empty |
| `region_types` | `list(...)` | Named region-type definitions | must be non-empty |
| `chromosomes` | `list(...)` | Named chromosome definitions | must be non-empty |

Optional field:

| Field | Type | Meaning |
|---|---|---|
| `runtime` | `list(...)` | Runtime-only controls such as alias-builder mode and profiling default |

Unknown `config` fields are rejected.

## `config$runtime`

`runtime` is optional. When present, it must be a `list(...)`.

Allowed fields:

| Field | Type | Meaning | Default if omitted |
|---|---|---|---|
| `alias_builder` | name or string | Parent-sampler alias-table builder | `auto` |
| `profile` | boolean | Whether profiling is enabled by default | `FALSE` |

Accepted `alias_builder` values:

- `auto`
- `sequential`
- `parallel`
- `parallel_psa_plus`

Examples:

```r
runtime = list(alias_builder = auto)
runtime = list(alias_builder = parallel)
runtime = list(alias_builder = "parallel_psa_plus", profile = TRUE)
runtime = list(alias_builder = builder_name, profile = use_profile)
```

`auto` resolves as follows:

- if the effective thread count is `1`, use `sequential`
- else if `N < 10000`, use `sequential`
- else use `parallel_psa_plus`

CLI precedence:

- `--alias-builder=...` overrides `config$runtime$alias_builder`
- `--profile` enables profiling even if `config$runtime$profile` is
  `FALSE`
- There is no CLI flag to force profiling off when the config already set
  `profile = TRUE`

## `config$mutation_types`

`mutation_types` must be a non-empty named `list(...)`.

Each mutation type must define exactly:

- `selection`
- `dominance`

Example:

```r
mutation_types = list(
  deleterious = list(
    selection = gamma(-0.02, 0.3, -0.999999999, 0.0),
    dominance = linear_from_s(0.5, -5.0, 0.0, 0.5)
  ),
  neutral = list(
    selection = constant(0.0),
    dominance = additive()
  )
)
```

### Selection Distribution Builders

All selection builders produce a `DistSpec`.

`constant(value[, min, max])`

- `value` is required
- `min` defaults to `value`
- `max` defaults to `value`
- `value` must lie within `[min, max]`

`uniform(min, max[, clamp_min, clamp_max])`

- `min` and `max` are required
- `clamp_min` defaults to `min`
- `clamp_max` defaults to `max`
- `max` must be greater than or equal to `min`

`normal(mean, sd, min, max)`

- all four arguments are required
- `sd` must be non-negative

`exponential(mean, min, max)`

- all three arguments are required
- `mean` must be non-negative

`gamma(mean, shape, min, max)`

- all four arguments are required
- `shape` must be greater than `0`
- `mean` may be negative, zero, or positive

`beta(alpha, beta, min, max)`

- all four arguments are required
- `alpha` must be greater than `0`
- `beta` must be greater than `0`

For every distribution builder:

- the final clamp range must satisfy `max >= min`
- draws are sampled first and clamped afterward

### Dominance Builders

`additive()`

- additive dominance

`fixed(h)`

- fixed dominance coefficient

`distributed(distribution)`

- dominance sampled from a distribution builder

`linear_from_s(intercept, slope, min_h, max_h)`

- computes dominance from selection coefficient
- `max_h` must be greater than or equal to `min_h`

The config loader does not impose any additional range restriction on
dominance values beyond the checks above. If you want dominance to stay in
`[0, 1]`, specify that explicitly through your chosen builder.

## `config$region_types`

`region_types` must be a non-empty named `list(...)`.

Each region type must define exactly:

- `mutation_scale`
- `weights`

Example:

```r
region_types = list(
  exon = list(
    mutation_scale = 2.5,
    weights = c(deleterious = 8, neutral = 1)
  ),
  intron = list(
    mutation_scale = 0.5,
    weights = c(deleterious = 1, neutral = 4)
  )
)
```

Rules:

- `mutation_scale` must be non-negative
- `weights` must be a non-empty named `c(...)`
- each weight name must reference an existing mutation type
- each weight value must be numeric and greater than `0`
- the total weight must be greater than `0`

There is no implicit default mutation type and no implicit default weight
vector. `weights` must always be written explicitly.

## `config$chromosomes`

`chromosomes` must be a non-empty named `list(...)`.

Each chromosome must define exactly:

- `length`
- `recombination_intervals`
- `regions`

Example:

```r
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
```

### `length`

- must be a positive integer

### `recombination_intervals`

`recombination_intervals` must be a `list(...)` of unnamed `interval(...)`
calls.

`interval(start, end, rate_scale)`

- `start` must be a non-negative integer
- `end` must be a non-negative integer
- `end` must be strictly greater than `start`
- `rate_scale` must be numeric and non-negative

Coverage rules:

- intervals must begin at `0`
- intervals must be contiguous
- intervals must not overlap
- intervals must not extend past chromosome length
- intervals must cover the full chromosome exactly

### `regions`

`regions` must be a `list(...)` of unnamed `region(...)` calls.

`region(region_type, start, end)`

- `region_type` must reference an existing region type
- `start` must be a non-negative integer
- `end` must be a non-negative integer
- `end` must be strictly greater than `start`

Coverage rules:

- regions must begin at `0`
- regions must be contiguous
- regions must not overlap
- regions must not extend past chromosome length
- regions must cover the full chromosome exactly

## `stats`

`stats` must be a `list(...)`. It may be empty.

Each entry in `stats` must be an unnamed schedule builder call. Allowed
schedules are:

- `always(...)`
- `every(...)`
- `at(...)`
- `up_to(...)`
- `at_after(...)`
- `every_up_to(...)`
- `every_at_after(...)`
- `range(...)`
- `every_range(...)`

If `stats <- list()` is empty, config mode runs without statistic output.

### Statistic Names

Accepted statistics:

- `mean_fitness`
- `genetic_load`
- `realized_masking_bonus`
- `exact_B`
- `pairwise_similarity(...)`
- `n_seg`
- `n_fixed`
- `genome_words`
- `mutation_histogram`
- `site_frequency_spectrum`
- `nucleotide_diversity`
- `expected_heterozygosity`

Most statistics may be written either as a bare symbol or as a zero-arg
call:

```r
mean_fitness
mean_fitness()
```

`pairwise_similarity` is the one statistic builder that takes an argument:

```r
pairwise_similarity(jaccard)
pairwise_similarity("dice")
pairwise_similarity(metric_name)
```

Allowed pairwise-similarity metrics:

- `jaccard`
- `dice`
- `overlap`

### Schedule Builders

`always(stat, ...)`

- collect every listed statistic at every generation from `1` through `G`

`every(step, stat, ...)`

- collect every `step` generations starting at `step`
- `step` must be a positive integer
- if `step > G`, the schedule is accepted but a warning is emitted because
  it never fires

`at(generation_or_c_vector, stat, ...)`

- collect at exactly the specified generation or generations
- the first argument may be a single generation or `c(g1, g2, ...)`
- every generation must lie in `[1, G]`

`up_to(end, stat, ...)`

- collect every generation from `1` through `end`, inclusive
- `end` must lie in `[1, G]`

`at_after(start, stat, ...)`

- collect every generation from `start` through `G`, inclusive
- `start` must lie in `[1, G]`

`every_up_to(step, end, stat, ...)`

- collect `step`, `2 * step`, `3 * step`, and so on up to `end`,
  inclusive when divisible
- `step` must be positive
- `end` must lie in `[1, G]`
- if `step > end`, the schedule is accepted but a warning is emitted
  because it never fires

`every_at_after(step, start, stat, ...)`

- collect `start`, `start + step`, `start + 2 * step`, and so on through
  `G`
- `step` must be positive
- `start` must lie in `[1, G]`

`range(start, end, stat, ...)`

- collect every generation from `start` through `end`, inclusive
- `start` and `end` must both lie in `[1, G]`
- `end` must be greater than or equal to `start`

`every_range(step, start, end, stat, ...)`

- collect `start`, `start + step`, `start + 2 * step`, and so on through
  `end`
- `step` must be positive
- `start` and `end` must both lie in `[1, G]`
- `end` must be greater than or equal to `start`

If overlapping schedules request the same statistic for the same
generation, `sparqy` emits that statistic once for that generation.

### `c(...)` in Stats

`c(...)` is supported in two places:

- `at(c(...), ...)` for generation vectors
- `weights = c(name = value, ...)` inside region types

For `at(c(...), ...)`:

- entries must be unnamed
- each entry must resolve to a non-negative integer

For `weights = c(...)`:

- entries must be named
- names must be existing mutation-type names

## Config-Mode Output

When statistics are enabled, config mode prints tab-separated rows with
this header:

```text
generation	dt_sec	cumul_sec	statistic	metric	scalar_value	by_type	by_chromosome	unfolded_sfs	folded_sfs
```

Column behavior:

- scalar statistics populate `scalar_value`
- `pairwise_similarity(...)` sets both `statistic` and `metric`
- `mutation_histogram` populates `by_type` and `by_chromosome`
- `site_frequency_spectrum` populates `unfolded_sfs` and `folded_sfs`
- unused columns in a given row are emitted empty

If `stats <- list()` is empty, config mode emits no statistic rows.

## Warnings

Current non-fatal warnings:

- unused constant
- `every(step, ...)` where `step > G`
- `every_up_to(step, end, ...)` where `step > end`

Warnings are reported before simulation output begins.

## Errors

The loader rejects invalid configs before simulation starts. Errors include
source locations whenever possible.

Examples of rejected input:

- missing `constants`, `config`, or `stats`
- duplicate top-level section
- unknown top-level section
- unknown field inside any structured block
- missing required field
- invalid scalar type in `constants`
- unknown constant reference
- unknown mutation type, region type, statistic, metric, or
  `alias_builder`
- negative `mu`, `rho`, or `threads`
- non-positive `N`, `G`, or chromosome length
- invalid distribution parameterization
- empty `weights`
- non-contiguous or incomplete interval coverage
- non-contiguous or incomplete region coverage
- out-of-range schedule generations

## CLI Interactions

Config mode intentionally rejects `--stats=LIST`.

Use:

```bash
./build/sparqy --config model.sparqy
```

Do not use:

```bash
./build/sparqy --config model.sparqy --stats=mean_fitness
```

That flag is only for the legacy positional interface.

## Complete Example

```r
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
  every(step = 10, genetic_load, n_seg),
  every_range(step = 20, start = 20, end = 80, genome_words),
  at(100, mutation_histogram, site_frequency_spectrum),
  at_after(start = 90, pairwise_similarity(jaccard))
)
```

The repository also includes paired `SLiM 5.x` versions of the checked-in
example configs under [`examples/slim/`](examples/slim/); the full worked
example above corresponds to
[`examples/slim/full_power_model.slim`](examples/slim/full_power_model.slim).
