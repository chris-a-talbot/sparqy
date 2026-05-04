#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path


SUITE_ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(SUITE_ROOT / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(SUITE_ROOT / ".cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch


BLACK = "#333333"
BLUE = "#333399"
RED = "#FF3333"
YELLOW = "#FFFF00"
WHITE = "#FFFFFF"
PALETTE = {
    "sequential": RED,
    "parallel": YELLOW,
    "parallel_psa_plus": BLUE,
}
ALIAS_ORDER = ["sequential", "parallel", "parallel_psa_plus"]
BIOLOGICAL_METRICS = [
    "mean_fitness",
    "genetic_load",
    "realized_masking_bonus",
    "exact_B",
    "n_seg",
    "n_fixed",
    "nucleotide_diversity",
    "expected_heterozygosity",
]


def configure_matplotlib() -> None:
    font_name = "Atkinson Hyperlegible Next"
    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    if font_name not in available_fonts:
        raise RuntimeError(
            "Atkinson Hyperlegible Next is not available to matplotlib on this machine."
        )

    plt.rcParams.update(
        {
            "figure.facecolor": WHITE,
            "axes.facecolor": WHITE,
            "savefig.facecolor": WHITE,
            "font.family": font_name,
            "text.color": BLACK,
            "axes.labelcolor": BLACK,
            "axes.edgecolor": BLACK,
            "axes.linewidth": 2.8,
            "xtick.color": BLACK,
            "ytick.color": BLACK,
            "xtick.major.width": 2.4,
            "ytick.major.width": 2.4,
            "grid.color": BLACK,
            "grid.linewidth": 1.6,
        }
    )


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def label_alias(alias_builder: str) -> str:
    return {
        "sequential": "Serial alias control",
        "parallel": "PSA alias",
        "parallel_psa_plus": "PSA+ alias",
    }.get(alias_builder, alias_builder)


def pretty_case_label(case_label: str) -> str:
    mapping = {
        "baseline_fixed": "Baseline fixed",
        "larger_population": "Larger population",
        "longer_genome": "Longer genome",
        "higher_mutation_rate": "Higher mutation rate",
        "higher_recombination_rate": "Higher recombination rate",
        "gamma_dfe": "Gamma DFE",
        "normal_dfe": "Normal DFE",
        "exponential_dfe": "Exponential DFE",
        "partial_recessive": "Partial recessive",
        "multi_type_regions": "Multi-type regions",
        "heterogeneous_recombination": "Heterogeneous recombination",
        "comprehensive_multichrom": "Comprehensive multichrom",
        "small_fixed": "Small fixed",
        "pop_10k": "Population 10k",
        "pop_50k": "Population 50k",
        "pop_100k": "Population 100k",
        "pop_250k": "Population 250k",
        "pop_500k": "Population 500k",
        "pop_5m": "Population 5M",
        "pop_10m": "Population 10M",
        "pop_25m": "Population 25M",
        "pop_50m": "Population 50M",
        "long_genome": "Long genome",
        "ultra_long_genome": "Ultra long genome",
        "high_mutation_rate": "High mutation rate",
        "high_recombination_rate": "High recombination rate",
        "gamma_mix": "Gamma-region mix",
        "medium_fixed": "Medium fixed",
        "large_fixed": "Large fixed",
        "large_high_rho": "Large high rho",
        "xlarge_long_genome": "XL long genome",
        "xlarge_gamma_mix": "XL gamma mix",
        "ultra_population": "Ultra population",
        "profile_medium_fixed": "Profile: medium fixed",
        "profile_large_high_rho": "Profile: large high rho",
        "profile_xlarge_long_genome": "Profile: XL long genome",
        "profile_xlarge_gamma_mix": "Profile: XL gamma mix",
        "profile_ultra_population": "Profile: ultra population",
        "profile_pop_5m": "Profile: population 5M",
        "profile_pop_10m": "Profile: population 10M",
        "profile_pop_25m": "Profile: population 25M",
        "profile_pop_50m": "Profile: population 50M",
    }
    return mapping.get(case_label, case_label.replace("_", " "))


def pretty_metric_label(metric: str) -> str:
    mapping = {
        "mean_fitness": "Mean fitness",
        "genetic_load": "Genetic load",
        "realized_masking_bonus": "Realized masking\nbonus",
        "exact_B": "Exact B",
        "n_seg": "Segregating\nmutations",
        "n_fixed": "Fixed\nmutations",
        "nucleotide_diversity": "Nucleotide\ndiversity",
        "expected_heterozygosity": "Expected\nheterozygosity",
    }
    return mapping.get(metric, metric.replace("_", " "))


def lighten_grid(ax: plt.Axes, axis: str = "y") -> None:
    if axis in ("x", "both"):
        ax.xaxis.grid(True, linewidth=1.6, alpha=0.9)
    if axis in ("y", "both"):
        ax.yaxis.grid(True, linewidth=1.6, alpha=0.9)


def save_figure(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    ensure_dir(out_dir)
    fig.savefig(out_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)


def thicken_axes(ax: plt.Axes) -> None:
    for spine in ax.spines.values():
        spine.set_linewidth(2.8)
        spine.set_color(BLACK)
    ax.tick_params(width=2.2, length=6, labelsize=12)


def plot_accuracy_heatmap(results_dir: Path, out_dir: Path) -> None:
    path = results_dir / "accuracy_scalar_comparison.csv"
    if not path.exists() or path.stat().st_size == 0:
        return

    df = pd.read_csv(path)
    production = df[df["variant"].astype(str).str.contains("parallel_psa_plus")]
    if production.empty:
        production = df[df["variant"] != "slim"]
    production = production[production["metric"].isin(BIOLOGICAL_METRICS)]
    if production.empty:
        return

    case_order = list(dict.fromkeys(production["case_label"].tolist()))
    metric_order = [metric for metric in BIOLOGICAL_METRICS if metric in set(production["metric"])]

    pivot = (
        production.pivot_table(
            index="case_label",
            columns="metric",
            values="rel_diff",
            aggfunc="mean",
        )
        .reindex(index=case_order, columns=metric_order)
    )

    values = pivot.fillna(0.0).to_numpy()
    binned = np.zeros_like(values, dtype=int)
    binned[values >= 0.01] = 1
    binned[values >= 0.05] = 2
    binned[values >= 0.10] = 3
    cmap = ListedColormap([BLUE, WHITE, YELLOW, RED])

    fig, ax = plt.subplots(figsize=(16, 9.5))
    ax.imshow(binned, cmap=cmap, aspect="auto", vmin=0, vmax=3)
    ax.set_title(
        "Accuracy Agreement Grid:\nsparqy PSA+ @ high thread count vs SLiM",
        fontsize=24,
        pad=18,
    )
    ax.set_xlabel("Statistic", fontsize=18)
    ax.set_ylabel("Scenario", fontsize=18)
    ax.set_xticks(np.arange(len(metric_order)))
    ax.set_xticklabels([pretty_metric_label(metric) for metric in metric_order], rotation=0, ha="center", fontsize=13)
    ax.set_yticks(np.arange(len(case_order)))
    ax.set_yticklabels([pretty_case_label(case) for case in case_order], fontsize=13)

    ax.set_xticks(np.arange(-0.5, len(metric_order), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(case_order), 1), minor=True)
    ax.grid(which="minor", color=BLACK, linewidth=2.8)
    ax.tick_params(which="minor", bottom=False, left=False)
    thicken_axes(ax)

    for row_index in range(values.shape[0]):
        for col_index in range(values.shape[1]):
            text = f"{100 * values[row_index, col_index]:.1f}%"
            cell_color = binned[row_index, col_index]
            text_color = WHITE if cell_color in (0, 3) else BLACK
            ax.text(
                col_index,
                row_index,
                text,
                ha="center",
                va="center",
                fontsize=11,
                color=text_color,
                fontweight="bold",
            )

    fig.text(
        0.01,
        0.01,
        "Blue <1% relative error, white 1–5%, yellow 5–10%, red >=10%",
        fontsize=13,
        color=BLACK,
    )
    fig.tight_layout(rect=(0.02, 0.04, 0.98, 0.96))
    save_figure(fig, out_dir, "mondrian_accuracy_heatmap")


def plot_accuracy_variant_overview(results_dir: Path, out_dir: Path) -> None:
    path = results_dir / "accuracy_variant_overview.csv"
    if not path.exists() or path.stat().st_size == 0:
        return

    df = pd.read_csv(path)
    if df.empty:
        return
    df = df.sort_values(["threads", "alias_builder"])

    x = np.arange(len(df))
    width = 0.34
    fig, ax = plt.subplots(figsize=(11, 7))
    scalar_bars = ax.bar(
        x - width / 2,
        100.0 * df["max_scalar_rel_diff"],
        width,
        color=RED,
        edgecolor=BLACK,
        linewidth=2.4,
        label="Worst scalar relative error",
    )
    vector_bars = ax.bar(
        x + width / 2,
        100.0 * df["max_vector_tvd"],
        width,
        color=BLUE,
        edgecolor=BLACK,
        linewidth=2.4,
        label="Worst SFS TVD",
    )
    ax.set_ylabel("Percent", fontsize=18)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{label_alias(alias)}\nT{threads}" for alias, threads in zip(df["alias_builder"], df["threads"])],
        fontsize=13,
    )
    ax.set_axisbelow(True)
    lighten_grid(ax, "y")
    thicken_axes(ax)

    for bar in list(scalar_bars) + list(vector_bars):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.4,
            f"{bar.get_height():.1f}",
            ha="center",
            va="bottom",
            fontsize=12,
            color=BLACK,
            fontweight="bold",
        )
    handles, labels = ax.get_legend_handles_labels()
    fig.suptitle("Accuracy Envelope Across Validation Variants", fontsize=24, y=0.992)
    fig.legend(
        handles,
        labels,
        frameon=False,
        fontsize=14,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.97),
        ncol=2,
    )
    fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.91))
    save_figure(fig, out_dir, "mondrian_accuracy_variant_overview")


def plot_speedup(results_dir: Path, out_dir: Path) -> None:
    path = results_dir / "speed_best_vs_slim.csv"
    if not path.exists() or path.stat().st_size == 0:
        return

    df = pd.read_csv(path)
    if df.empty:
        return
    case_order = list(dict.fromkeys(df.sort_values("speedup_vs_slim", ascending=False)["case_label"].tolist()))
    y = np.arange(len(case_order))
    height = 0.23
    offsets = {
        "sequential": -height,
        "parallel": 0.0,
        "parallel_psa_plus": height,
    }

    fig_height = max(7.2, 0.82 * len(case_order) + 2.8)
    fig, ax = plt.subplots(figsize=(15.5, fig_height))
    for alias_builder in ALIAS_ORDER:
        sub = df[df["alias_builder"] == alias_builder].set_index("case_label").reindex(case_order)
        if sub.empty:
            continue
        ypos = y + offsets[alias_builder]
        bars = ax.barh(
            ypos,
            sub["speedup_vs_slim"],
            height=height,
            color=PALETTE[alias_builder],
            edgecolor=BLACK,
            linewidth=2.6,
            label=label_alias(alias_builder),
        )
        for bar, threads in zip(bars, sub["best_threads"].fillna(0).astype(int)):
            ax.text(
                bar.get_width() + 0.14,
                bar.get_y() + bar.get_height() / 2,
                f"T{threads}",
                ha="left",
                va="center",
                fontsize=11,
                fontweight="bold",
                color=BLACK,
            )

    fig.suptitle("Best Measured sparqy Speedup Over SLiM", fontsize=24, y=0.992)
    ax.set_xlabel("Speedup (SLiM / sparqy)", fontsize=18)
    ax.set_yticks(y)
    ax.set_yticklabels([pretty_case_label(case) for case in case_order], fontsize=13)
    ax.set_axisbelow(True)
    lighten_grid(ax, "x")
    thicken_axes(ax)
    ax.set_xlim(0.0, max(df["speedup_vs_slim"]) * 1.16)
    ax.invert_yaxis()
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        fontsize=14,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.968),
    )
    fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.91))
    save_figure(fig, out_dir, "mondrian_speedup_vs_slim")


def plot_scaling(results_dir: Path, out_dir: Path) -> None:
    path = results_dir / "scaling_summary.csv"
    if not path.exists() or path.stat().st_size == 0:
        return

    df = pd.read_csv(path)
    if df.empty:
        return
    cases = list(dict.fromkeys(df["case_label"].tolist()))
    ncols = 2 if len(cases) > 3 else len(cases)
    nrows = math.ceil(len(cases) / ncols)
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(15.5, 5.1 * nrows),
        squeeze=False,
        sharex=False,
        sharey=False,
    )

    for axis, case_label in zip(axes.flat, cases):
        sub = df[df["case_label"] == case_label]
        for alias_builder in ALIAS_ORDER:
            alias_sub = sub[sub["alias_builder"] == alias_builder].sort_values("threads")
            if alias_sub.empty:
                continue
            axis.plot(
                alias_sub["threads"],
                alias_sub["speedup_vs_t1"],
                marker="o",
                markersize=8,
                linewidth=3.2,
                color=PALETTE[alias_builder],
                markeredgecolor=BLACK,
                label=label_alias(alias_builder),
            )
        ideal = np.array(sorted(sub["threads"].unique()), dtype=float)
        axis.plot(
            ideal,
            ideal,
            linestyle=(0, (6, 4)),
            color=BLACK,
            linewidth=2.0,
            alpha=0.55,
            label="Ideal" if case_label == cases[0] else None,
        )
        axis.set_title(pretty_case_label(case_label), fontsize=18, pad=12)
        axis.set_xscale("log", base=2)
        axis.set_xticks(sorted(sub["threads"].unique()))
        axis.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        axis.set_xlabel("Threads", fontsize=14)
        axis.set_ylabel("Speedup vs T1", fontsize=14)
        axis.set_ylim(0.8, max(sub["speedup_vs_t1"].max() * 1.12, 2.0))
        lighten_grid(axis, "both")
        thicken_axes(axis)

    for axis in axes.flat[len(cases):]:
        axis.axis("off")

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.suptitle("sparqy Thread Scaling Across Alias Builders", fontsize=24, y=0.992)
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.962),
        ncol=4,
        frameon=False,
        fontsize=13,
    )
    fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.88))
    save_figure(fig, out_dir, "mondrian_thread_scaling")


def plot_memory_scaling_metric(
    results_dir: Path,
    out_dir: Path,
    *,
    column: str,
    ylabel: str,
    title: str,
    stem: str,
) -> None:
    path = results_dir / "scaling_summary.csv"
    if not path.exists() or path.stat().st_size == 0:
        return

    df = pd.read_csv(path)
    if df.empty or column not in df.columns:
        return

    df = df.copy()
    df[column] = pd.to_numeric(df[column], errors="coerce") / (1024.0 * 1024.0)
    df = df[np.isfinite(df[column])]
    if df.empty:
        return

    cases = list(dict.fromkeys(df["case_label"].tolist()))
    ncols = 2 if len(cases) > 3 else len(cases)
    nrows = math.ceil(len(cases) / ncols)
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(15.5, 5.1 * nrows),
        squeeze=False,
        sharex=False,
        sharey=False,
    )

    for axis, case_label in zip(axes.flat, cases):
        sub = df[df["case_label"] == case_label]
        for alias_builder in ALIAS_ORDER:
            alias_sub = sub[sub["alias_builder"] == alias_builder].sort_values("threads")
            if alias_sub.empty:
                continue
            axis.plot(
                alias_sub["threads"],
                alias_sub[column],
                marker="o",
                markersize=8,
                linewidth=3.2,
                color=PALETTE[alias_builder],
                markeredgecolor=BLACK,
                label=label_alias(alias_builder),
            )
        axis.set_title(pretty_case_label(case_label), fontsize=18, pad=12)
        axis.set_xscale("log", base=2)
        axis.set_xticks(sorted(sub["threads"].unique()))
        axis.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        axis.set_xlabel("Threads", fontsize=14)
        axis.set_ylabel(ylabel, fontsize=14)
        axis.set_ylim(0.0, max(sub[column].max() * 1.12, 0.25))
        lighten_grid(axis, "both")
        thicken_axes(axis)

    for axis in axes.flat[len(cases):]:
        axis.axis("off")

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.suptitle(title, fontsize=24, y=0.992)
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.962),
        ncol=3,
        frameon=False,
        fontsize=13,
    )
    fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.88))
    save_figure(fig, out_dir, stem)


def plot_memory_scaling(results_dir: Path, out_dir: Path) -> None:
    plot_memory_scaling_metric(
        results_dir,
        out_dir,
        column="mean_peak_rss_kb",
        ylabel="Peak RSS (GiB)",
        title="sparqy Peak RSS Across Alias Builders",
        stem="mondrian_peak_rss_scaling",
    )
    plot_memory_scaling_metric(
        results_dir,
        out_dir,
        column="mean_peak_vmsize_kb",
        ylabel="Peak virtual memory (GiB)",
        title="sparqy Peak Virtual Memory Across Alias Builders",
        stem="mondrian_peak_vmsize_scaling",
    )


def plot_profile(results_dir: Path, out_dir: Path) -> None:
    path = results_dir / "profile_phase_summary.csv"
    if not path.exists() or path.stat().st_size == 0:
        return

    df = pd.read_csv(path)
    if df.empty:
        return

    max_threads = df["threads"].max()
    df = df[(df["threads"] == max_threads) & (df["phase"] != "total")].copy()
    if df.empty:
        return

    phase_display_order = [
        "parallel_reproduction",
        "build_parent_sampler",
        "merge_thread_counts",
        "classify_mutations",
        "offspring_copy_and_count",
        "zero_offspring_counts",
    ]
    major_phase_map = {
        "parallel_reproduction": "Reproduction",
        "build_parent_sampler": "Alias build",
        "merge_thread_counts": "Merge thread counts",
        "classify_mutations": "Classify mutations",
        "offspring_copy_and_count": "Copy and count",
        "zero_offspring_counts": "Zero counts",
    }
    df["phase_group"] = df["phase"].map(major_phase_map).fillna("Minor phases")

    grouped = (
        df.groupby(["case_label", "alias_builder", "phase_group"], as_index=False)["mean_pct_total"]
        .sum()
    )
    stack_totals = grouped.groupby(["case_label", "alias_builder"], as_index=False)["mean_pct_total"].sum()
    invalid = stack_totals[
        (stack_totals["mean_pct_total"] < 95.0) | (stack_totals["mean_pct_total"] > 100.5)
    ]
    if not invalid.empty:
        sample = invalid.head(5).to_dict("records")
        raise ValueError(
            "Profile phase shares do not sum to ~100%; a total row likely leaked into the "
            f"stacked plot. Sample bad groups: {sample}"
        )
    cases = list(dict.fromkeys(grouped["case_label"].tolist()))
    phase_order = [
        "Reproduction",
        "Alias build",
        "Merge thread counts",
        "Classify mutations",
        "Copy and count",
        "Zero counts",
        "Minor phases",
    ]
    phase_styles = {
        "Reproduction": {"facecolor": BLUE, "hatch": None},
        "Alias build": {"facecolor": YELLOW, "hatch": None},
        "Merge thread counts": {"facecolor": RED, "hatch": None},
        "Classify mutations": {"facecolor": BLUE, "hatch": "//"},
        "Copy and count": {"facecolor": YELLOW, "hatch": "//"},
        "Zero counts": {"facecolor": RED, "hatch": "//"},
        "Minor phases": {"facecolor": BLACK, "hatch": None},
    }

    fig_height = max(8.5, 0.72 * len(cases) * len(ALIAS_ORDER) + 2.4)
    fig, ax = plt.subplots(figsize=(15.5, fig_height))
    labels: list[str] = []
    y_positions: list[float] = []
    current_y = 0.0
    for case_label in cases:
        for alias_builder in ALIAS_ORDER:
            labels.append(f"{pretty_case_label(case_label)} | {label_alias(alias_builder)}")
            y_positions.append(current_y)
            current_y += 1.0
        current_y += 0.35

    lookup = {
        (row["case_label"], row["alias_builder"], row["phase_group"]): row["mean_pct_total"]
        for _, row in grouped.iterrows()
    }

    lefts = np.zeros(len(labels))
    for phase in phase_order:
        widths = []
        for case_label in cases:
            for alias_builder in ALIAS_ORDER:
                widths.append(lookup.get((case_label, alias_builder, phase), 0.0))
        style = phase_styles[phase]
        ax.barh(
            y_positions,
            widths,
            left=lefts,
            color=style["facecolor"],
            edgecolor=BLACK,
            linewidth=2.4,
            height=0.82,
            hatch=style["hatch"] or "",
            label=phase,
        )
        lefts += np.array(widths)

    fig.suptitle(f"Perlmutter Phase Shares at T{int(max_threads)}", fontsize=24, y=0.992)
    ax.set_xlabel("Percent of total runtime", fontsize=18)
    ax.set_xlim(0.0, 100.0)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, fontsize=11)
    ax.invert_yaxis()
    legend_handles = [
        Patch(
            facecolor=phase_styles[phase]["facecolor"],
            edgecolor=BLACK,
            linewidth=2.0,
            hatch=phase_styles[phase]["hatch"] or "",
            label=phase,
        )
        for phase in phase_order
    ]
    fig.legend(
        handles=legend_handles,
        frameon=False,
        ncol=4,
        fontsize=12,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.97),
    )
    ax.set_axisbelow(True)
    lighten_grid(ax, "x")
    thicken_axes(ax)
    fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.91))
    save_figure(fig, out_dir, "mondrian_profile_phase_shares")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate Mondrian-style poster figures from validation_suite results."
    )
    parser.add_argument(
        "results_dir",
        help="Results directory produced by validation_suite/run_suite.py run",
    )
    parser.add_argument(
        "--out-dir",
        default="",
        help="Optional explicit output directory for figures. Defaults to <results_dir>/figures.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    results_dir = Path(args.results_dir).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else results_dir / "figures"

    configure_matplotlib()
    ensure_dir(out_dir)
    plot_accuracy_heatmap(results_dir, out_dir)
    plot_accuracy_variant_overview(results_dir, out_dir)
    plot_speedup(results_dir, out_dir)
    plot_scaling(results_dir, out_dir)
    plot_memory_scaling(results_dir, out_dir)
    plot_profile(results_dir, out_dir)
    print(f"Figures written to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
