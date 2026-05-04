#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path


SUITE_ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(SUITE_ROOT / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(SUITE_ROOT / ".cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import patheffects
from matplotlib.colors import LinearSegmentedColormap

if __package__ in (None, ""):
    sys.path.insert(0, str(SUITE_ROOT))
    import plot_results as base  # type: ignore
else:
    from . import plot_results as base


FULL_DEFAULT = SUITE_ROOT / "results" / "perlmutter_full_20260503_121006"
EXTREME_DEFAULT = SUITE_ROOT / "results" / "perlmutter_extreme_20260503_123713"
OUT_DEFAULT = SUITE_ROOT / "results" / "poster_figures_20260503"

BLACK = base.BLACK
BLUE = base.BLUE
RED = base.RED
YELLOW = base.YELLOW
WHITE = base.WHITE
PALETTE = base.PALETTE
ALIAS_ORDER = base.ALIAS_ORDER

CASE_LABELS = {
    "high_mutation_rate": "High mutation",
    "high_recombination_rate": "High recombination",
    "long_genome": "Long genome",
    "pop_10k": "Population 10k",
    "pop_50k": "Population 50k",
    "pop_100k": "Population 100k",
    "pop_250k": "Population 250k",
    "pop_500k": "Population 500k",
    "small_fixed": "Small fixed",
    "ultra_population": "Population 2M",
    "pop_5m": "Population 5M",
    "pop_10m": "Population 10M",
    "pop_25m": "Population 25M",
    "pop_50m": "Population 50M",
    "xlarge_gamma_mix": "Gamma mix 1M",
    "xlarge_long_genome": "Long genome 1M",
    "baseline_fixed": "Baseline fixed",
    "larger_population": "Larger population",
    "longer_genome": "Long genome",
    "partial_recessive": "Partial recessive",
    "exponential_dfe": "Exponential DFE",
    "higher_mutation_rate": "High mutation",
}

SHORT_POP_LABELS = {
    "ultra_population": "2M",
    "pop_5m": "5M",
    "pop_10m": "10M",
    "pop_25m": "25M",
    "pop_50m": "50M*",
}

METRIC_LABELS = {
    "mean_fitness": "Mean\nfitness",
    "n_seg": "Segregating\nmuts",
    "nucleotide_diversity": "Nucleotide\ndiversity",
}

ACCURACY_ROW_LABELS = {
    "higher_mutation_rate": "High\nmutation",
    "longer_genome": "Long\ngenome",
    "partial_recessive": "Partial\nrecessive",
    "exponential_dfe": "Exponential\nDFE",
}

SPEEDUP_CASE_LABELS = {
    "pop_500k": "N=500k",
    "pop_250k": "N=250k",
    "pop_50k": "N=50k",
    "long_genome": "N=20k\nlong genome",
    "high_mutation_rate": "N=20k\nhigh mutation",
    "small_fixed": "N=2k\nadditive",
}


def configure_poster_matplotlib() -> None:
    base.configure_matplotlib()
    plt.rcParams.update(
        {
            "axes.titlesize": 14,
            "axes.labelsize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 10,
        }
    )


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_figure(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    ensure_dir(out_dir)
    fig.savefig(
        out_dir / f"{stem}.png",
        dpi=300,
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
        edgecolor=fig.get_edgecolor(),
    )
    fig.savefig(
        out_dir / f"{stem}.svg",
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
        edgecolor=fig.get_edgecolor(),
    )
    fig.savefig(
        out_dir / f"{stem}_transparent.png",
        dpi=300,
        bbox_inches="tight",
        transparent=True,
    )
    fig.savefig(
        out_dir / f"{stem}_transparent.svg",
        bbox_inches="tight",
        transparent=True,
    )
    plt.close(fig)


def thicken_axes(ax: plt.Axes) -> None:
    base.thicken_axes(ax)


def add_figure_header(
    fig: plt.Figure,
    title: str,
    subtitle: str | None = None,
    *,
    color: str = BLACK,
) -> None:
    fig.suptitle(title, fontsize=18, fontweight="bold", y=0.975, color=color)
    if subtitle:
        fig.text(0.5, 0.935, subtitle, ha="center", va="top", fontsize=9.5, color=color)


def pretty_case_label(case_label: str) -> str:
    return CASE_LABELS.get(case_label, base.pretty_case_label(case_label))


def label_alias(alias_builder: str) -> str:
    return base.label_alias(alias_builder)


def speedup_case_label(case_label: str) -> str:
    return SPEEDUP_CASE_LABELS.get(case_label, pretty_case_label(case_label))


def bar_text_color(alias_builder: str | None = None, fill_color: str | None = None) -> str:
    if fill_color is None and alias_builder is not None:
        fill_color = PALETTE.get(alias_builder, WHITE)
    return WHITE if fill_color in {BLUE, RED} else BLACK


def style_red_background(ax: plt.Axes) -> None:
    ax.set_facecolor(RED)
    for spine in ax.spines.values():
        spine.set_color(WHITE)
    ax.tick_params(colors=WHITE)
    ax.xaxis.label.set_color(WHITE)
    ax.yaxis.label.set_color(WHITE)
    ax.title.set_color(WHITE)


def add_text_outline(text_artist: plt.Text, outline_color: str) -> None:
    text_artist.set_path_effects(
        [
            patheffects.Stroke(linewidth=2.6, foreground=outline_color),
            patheffects.Normal(),
        ]
    )


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path)


def giB_from_kb(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce") / (1024.0 * 1024.0)


def build_n_lookup(extreme_dir: Path) -> dict[str, int]:
    manifest = json.loads((extreme_dir / "manifest.json").read_text())
    scenario_key_by_case = {
        "ultra_population": "ultra_population",
        "pop_5m": "extreme_pop_5m",
        "pop_10m": "extreme_pop_10m",
        "pop_25m": "extreme_pop_25m",
        "pop_50m": "extreme_pop_50m",
    }
    lookup: dict[str, int] = {}
    for case_label, scenario_name in scenario_key_by_case.items():
        if scenario_name in manifest["scenarios"]:
            lookup[case_label] = int(manifest["scenarios"][scenario_name]["N"])
    return lookup


def plot_speedup_vs_slim(full_dir: Path, out_dir: Path, *, red_background: bool = False) -> None:
    df = load_csv(full_dir / "speed_best_vs_slim.csv")
    if df.empty:
        return

    selected_cases = [
        "pop_500k",
        "pop_250k",
        "pop_50k",
        "long_genome",
        "high_mutation_rate",
        "small_fixed",
    ]
    df = df[df["case_label"].isin(selected_cases)].copy()
    if df.empty:
        return

    best = (
        df.sort_values("speedup_vs_slim", ascending=False)
        .groupby("case_label", as_index=False)
        .first()
    )
    case_order = [
        case for case in selected_cases if case in set(best["case_label"])
    ]
    best = best.set_index("case_label").loc[case_order].reset_index()

    fig_height = 0.78 * len(best) + 2.4
    fig, ax = plt.subplots(figsize=(14.5, fig_height))
    if red_background:
        fig.patch.set_facecolor(RED)
    y = np.arange(len(best))
    bars = ax.barh(
        y,
        best["speedup_vs_slim"],
        color=[PALETTE[str(alias)] for alias in best["alias_builder"]],
        edgecolor=BLACK,
        linewidth=2.8,
        height=0.68,
    )
    for bar, speedup, alias, threads in zip(
        bars,
        best["speedup_vs_slim"],
        best["alias_builder"],
        best["best_threads"],
    ):
        alias_short = "PSA+" if str(alias) == "parallel_psa_plus" else "PSA"
        fontsize = 9.5 if float(speedup) >= 3.0 else 8.0
        text_artist = ax.text(
            bar.get_width() - 0.18,
            bar.get_y() + bar.get_height() / 2,
            f"{speedup:.1f}x\n{alias_short} T{int(threads)}",
            ha="right",
            va="center",
            fontsize=fontsize,
            fontweight="bold",
            color=bar_text_color(alias_builder=str(alias)),
        )
        if red_background:
            outline_color = BLACK if bar_text_color(alias_builder=str(alias)) == WHITE else WHITE
            add_text_outline(text_artist, outline_color)

    ax.set_yticks(y)
    ax.set_yticklabels([speedup_case_label(str(case)) for case in best["case_label"]], fontsize=12.5)
    ax.set_xlabel("Best measured speedup over SLiM", fontsize=13)
    ax.set_xlim(0.0, max(best["speedup_vs_slim"]) * 1.22)
    ax.grid(True, axis="x", linewidth=1.8, alpha=0.9, color=WHITE if red_background else BLACK)
    ax.set_axisbelow(True)
    ax.invert_yaxis()
    thicken_axes(ax)
    if red_background:
        style_red_background(ax)
    fig.subplots_adjust(left=0.19, right=0.96, top=0.9, bottom=0.12)
    save_figure(
        fig,
        out_dir,
        "poster_speedup_vs_slim_red_bg" if red_background else "poster_speedup_vs_slim",
    )


def plot_psa_specific_gains(full_dir: Path, out_dir: Path) -> None:
    df = load_csv(full_dir / "speed_summary.csv")
    if df.empty:
        return

    df = df[df["simulator"] == "sparqy"].copy()
    selected_cases = [
        "pop_500k",
        "long_genome",
        "high_mutation_rate",
    ]
    records: list[dict[str, object]] = []
    for case_label in selected_cases:
        case_df = df[df["case_label"] == case_label]
        if case_df.empty:
            continue
        for alias_builder in ("parallel", "parallel_psa_plus"):
            best_gain = 0.0
            best_threads = None
            for threads in sorted(set(case_df["threads"])):
                if int(threads) <= 1:
                    continue
                seq = case_df[
                    (case_df["alias_builder"] == "sequential")
                    & (case_df["threads"] == threads)
                ]
                alt = case_df[
                    (case_df["alias_builder"] == alias_builder)
                    & (case_df["threads"] == threads)
                ]
                if seq.empty or alt.empty:
                    continue
                gain = float(seq.iloc[0]["mean_ms_per_gen"]) / float(alt.iloc[0]["mean_ms_per_gen"])
                if gain > best_gain:
                    best_gain = gain
                    best_threads = int(threads)
            if best_threads is None:
                continue
            records.append(
                {
                    "case_label": case_label,
                    "alias_builder": alias_builder,
                    "gain_vs_serial": best_gain,
                    "threads": best_threads,
                }
            )

    gain_df = pd.DataFrame.from_records(records)
    if gain_df.empty:
        return

    fig, ax = plt.subplots(figsize=(10.4, 7.0))
    x = np.arange(len(selected_cases))
    width = 0.34
    offsets = {"parallel": -width / 2.0, "parallel_psa_plus": width / 2.0}

    for alias_builder in ("parallel", "parallel_psa_plus"):
        sub = (
            gain_df[gain_df["alias_builder"] == alias_builder]
            .set_index("case_label")
            .reindex(selected_cases)
        )
        bars = ax.bar(
            x + offsets[alias_builder],
            sub["gain_vs_serial"],
            width=width,
            color=PALETTE[alias_builder],
            edgecolor=BLACK,
            linewidth=2.6,
            label=(
                "Parallel split-and-pack\n(PSA)"
                if alias_builder == "parallel"
                else "Greedy pre-pass refinement\n(PSA+)"
            ),
        )
        for bar, gain, threads in zip(
            bars,
            sub["gain_vs_serial"],
            sub["threads"],
        ):
            if pd.isna(gain):
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                float(gain) - 0.035,
                f"{float(gain):.2f}x\nT{int(threads)}",
                ha="center",
                va="top",
                fontsize=12.2,
                fontweight="bold",
                color=bar_text_color(alias_builder=alias_builder),
            )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [speedup_case_label(case) for case in selected_cases],
        fontsize=15,
    )
    ax.set_ylabel("Same-thread speedup over\nserial alias control", fontsize=15)
    ax.set_ylim(0.9, max(gain_df["gain_vs_serial"]) * 1.22)
    ax.grid(True, axis="y", linewidth=1.8, alpha=0.9)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", labelsize=13.5)
    thicken_axes(ax)
    ax.legend(
        loc="lower left",
        bbox_to_anchor=(0.0, 1.01),
        frameon=False,
        fontsize=14.0,
        ncol=1,
        handlelength=1.8,
        borderaxespad=0.0,
    )
    fig.subplots_adjust(left=0.16, right=0.97, top=0.83, bottom=0.22)
    save_figure(fig, out_dir, "poster_psa_specific_speedups")


def plot_thread_population_scaling(full_dir: Path, extreme_dir: Path, out_dir: Path) -> None:
    speed_df = load_csv(full_dir / "speed_runs.csv")
    throughput_df = load_csv(extreme_dir / "scaling_summary.csv")
    if speed_df.empty and throughput_df.empty:
        return

    n_lookup = build_n_lookup(extreme_dir)
    scaling_cases = ["pop_50k", "pop_100k", "pop_500k", "ultra_population"]
    throughput_cases = ["ultra_population", "pop_5m", "pop_10m", "pop_25m", "pop_50m"]
    line_colors = {
        "pop_50k": RED,
        "pop_100k": BLUE,
        "pop_500k": YELLOW,
        "ultra_population": BLACK,
    }
    line_styles = {
        "pop_50k": "-",
        "pop_100k": "-",
        "pop_500k": (0, (7, 3)),
        "ultra_population": "-",
    }
    marker_facecolors = {
        "pop_50k": RED,
        "pop_100k": BLUE,
        "pop_500k": YELLOW,
        "ultra_population": BLACK,
    }
    marker_map = {
        "pop_50k": "s",
        "pop_100k": "D",
        "pop_500k": "^",
        "ultra_population": "o",
    }
    scaling_labels = {
        "pop_50k": "N=50k",
        "pop_100k": "N=100k",
        "pop_500k": "N=500k",
        "ultra_population": "N=2M",
    }

    fig, axes = plt.subplots(1, 2, figsize=(16.6, 6.8), gridspec_kw={"width_ratios": [1.2, 1.0]})

    ax = axes[0]
    for case_label in scaling_cases:
        if case_label == "ultra_population":
            sub = throughput_df[throughput_df["case_label"] == case_label].copy()
            if sub.empty:
                continue
            thread_records = []
            for threads in sorted(set(int(value) for value in sub["threads"])):
                thread_sub = sub[sub["threads"] == threads]
                best_row = thread_sub.loc[thread_sub["mean_ms_per_gen"].astype(float).idxmin()]
                thread_records.append((threads, float(best_row["mean_ms_per_gen"])))
        else:
            sub = speed_df[(speed_df["simulator"] == "sparqy") & (speed_df["case_label"] == case_label)].copy()
            if sub.empty:
                continue
            grouped = (
                sub.groupby(["alias_builder", "threads"], as_index=False)["ms_per_gen"]
                .mean()
            )
            thread_records = []
            for threads in sorted(set(int(value) for value in grouped["threads"])):
                thread_sub = grouped[grouped["threads"] == threads]
                best_row = thread_sub.loc[thread_sub["ms_per_gen"].astype(float).idxmin()]
                thread_records.append((threads, float(best_row["ms_per_gen"])))
        if sub.empty:
            continue
        if not thread_records:
            continue
        baseline = next(ms for threads, ms in thread_records if threads == 1)
        xs = [threads for threads, _ms in thread_records]
        ys = [baseline / ms for _threads, ms in thread_records]
        ax.plot(
            xs,
            ys,
            color=line_colors[case_label],
            linestyle=line_styles[case_label],
            marker=marker_map[case_label],
            markersize=8.2,
            markerfacecolor=marker_facecolors[case_label],
            markeredgecolor=BLACK,
            linewidth=2.9,
            label=scaling_labels[case_label],
        )

    ideal = np.array([1, 8, 16, 32, 64, 128], dtype=float)
    ax.plot(
        ideal,
        ideal,
        linestyle=(0, (6, 4)),
        color=BLACK,
        linewidth=2.0,
        alpha=0.5,
        label="Ideal",
    )
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xticks([1, 8, 16, 32, 64, 128])
    ax.set_yticks([1, 2, 4, 8, 16])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_ylim(0.9, 12.5)
    ax.set_xlabel("Threads", fontsize=15)
    ax.set_ylabel("Best measured speedup vs T1", fontsize=15)
    ax.set_title("Log-log strong scaling from 50k to 2M diploids", fontsize=16, pad=7)
    ax.grid(True, axis="both", linewidth=1.7, alpha=0.9)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", labelsize=13.5)
    thicken_axes(ax)
    legend = ax.legend(
        loc="lower right",
        frameon=True,
        fontsize=13.2,
        ncol=1,
        handlelength=2.1,
    )
    legend.get_frame().set_facecolor(WHITE)
    legend.get_frame().set_edgecolor(BLACK)
    legend.get_frame().set_linewidth(1.6)

    ax = axes[1]
    throughput_rows: list[dict[str, object]] = []
    for case_label in throughput_cases:
        sub = throughput_df[throughput_df["case_label"] == case_label]
        if sub.empty:
            continue
        best_row = sub.loc[sub["mean_ms_per_gen"].astype(float).idxmin()]
        n_value = n_lookup.get(case_label)
        if n_value is None:
            continue
        throughput_mdip_s = n_value / (float(best_row["mean_ms_per_gen"]) / 1000.0) / 1.0e6
        throughput_rows.append(
            {
                "case_label": case_label,
                "throughput": throughput_mdip_s,
                "alias_builder": str(best_row["alias_builder"]),
                "threads": int(best_row["threads"]),
                "partial": case_label == "pop_50m",
            }
        )
    throughput_df = pd.DataFrame.from_records(throughput_rows)
    x = np.arange(len(throughput_df))
    bars = ax.bar(
        x,
        throughput_df["throughput"],
        color=[PALETTE.get(str(alias), WHITE) for alias in throughput_df["alias_builder"]],
        edgecolor=BLACK,
        linewidth=2.8,
        width=0.72,
    )
    for bar, row in zip(bars, throughput_rows):
        if bool(row["partial"]):
            bar.set_hatch("//")
        label_text = (
            f"{row['throughput']:.1f}M/s\n"
            f"{label_alias(str(row['alias_builder'])).replace(' alias','')} T{row['threads']}"
        )
        if bool(row["partial"]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.35,
                label_text,
                ha="center",
                va="bottom",
                fontsize=11.2,
                fontweight="bold",
                color=BLACK,
                bbox={"facecolor": WHITE, "edgecolor": BLACK, "linewidth": 1.0, "pad": 0.2},
            )
        else:
            text_color = WHITE if str(row["alias_builder"]) == "parallel_psa_plus" else BLACK
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() - 0.45,
                label_text,
                ha="center",
                va="top",
                fontsize=11.5,
                fontweight="bold",
                color=text_color,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [SHORT_POP_LABELS.get(str(case), pretty_case_label(str(case))) for case in throughput_df["case_label"]],
        fontsize=14,
    )
    ax.set_ylabel("Best measured throughput\n(million diploids / sec)", fontsize=15)
    ax.set_title("Best measured throughput by population size", fontsize=16, pad=10)
    ax.set_ylim(0.0, max(30.0, float(throughput_df["throughput"].max()) * 1.12))
    ax.grid(True, axis="y", linewidth=1.7, alpha=0.9)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", labelsize=13.5)
    thicken_axes(ax)
    ax.text(
        0.97,
        0.03,
        "* partial run",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=12.2,
        color=BLACK,
        bbox={"facecolor": WHITE, "edgecolor": BLACK, "linewidth": 1.2, "pad": 0.25},
    )
    fig.subplots_adjust(left=0.09, right=0.985, top=0.91, bottom=0.16, wspace=0.24)
    save_figure(fig, out_dir, "poster_thread_population_scaling")


def plot_accuracy_best_cases(full_dir: Path, out_dir: Path, *, red_background: bool = False) -> None:
    scalar_df = load_csv(full_dir / "accuracy_scalar_comparison.csv")
    if scalar_df.empty:
        return

    scalar_threads = pd.to_numeric(scalar_df["threads"], errors="coerce")
    scalar_df = scalar_df[
        (scalar_df["alias_builder"] == "parallel_psa_plus")
        & (scalar_threads == 128)
    ].copy()
    if scalar_df.empty:
        return

    metrics = [
        "mean_fitness",
        "n_seg",
        "nucleotide_diversity",
    ]
    selected_cases = [
        "higher_mutation_rate",
        "longer_genome",
        "partial_recessive",
        "exponential_dfe",
    ]

    heat_df = (
        scalar_df[
            scalar_df["case_label"].isin(selected_cases)
            & scalar_df["metric"].isin(metrics)
        ]
        .pivot_table(index="case_label", columns="metric", values="rel_diff", aggfunc="mean")
        .reindex(index=selected_cases, columns=metrics)
        * 100.0
    )
    if heat_df.isnull().any().any():
        return

    fig, ax = plt.subplots(figsize=(7.9, 5.8))
    if red_background:
        fig.patch.set_facecolor(RED)
        ax.set_facecolor(RED)

    cmap = LinearSegmentedColormap.from_list("poster_accuracy", [WHITE, YELLOW, RED])
    values = heat_df.to_numpy()
    vmax = max(4.0, float(np.nanmax(values)) * 1.05)
    ax.imshow(values, cmap=cmap, aspect="auto", vmin=0.0, vmax=vmax)
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_xticklabels([METRIC_LABELS[metric] for metric in metrics], fontsize=13)
    ax.set_yticks(np.arange(len(selected_cases)))
    ax.set_yticklabels([ACCURACY_ROW_LABELS[case] for case in selected_cases], fontsize=13.5)
    for i in range(len(selected_cases)):
        for j in range(len(metrics)):
            value = values[i, j]
            text_color = WHITE if value >= 0.62 * vmax else BLACK
            text_artist = ax.text(
                j,
                i,
                f"{value:.1f}",
                ha="center",
                va="center",
                fontsize=12,
                fontweight="bold",
                color=text_color,
            )
            if red_background:
                outline_color = BLACK if text_color == WHITE else WHITE
                add_text_outline(text_artist, outline_color)
    ax.set_xticks(np.arange(-0.5, len(metrics), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(selected_cases), 1), minor=True)
    ax.grid(which="minor", color=BLACK, linewidth=2.6)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.set_title("Scalar relative error (%)", fontsize=15, pad=8)
    thicken_axes(ax)
    if red_background:
        ax.tick_params(colors=WHITE)
        ax.title.set_color(WHITE)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_color(WHITE)

    add_figure_header(
        fig,
        "Selected matched validation cases",
        "PSA+ at 128 threads against matched SLiM runs.",
        color=WHITE if red_background else BLACK,
    )
    fig.subplots_adjust(left=0.175, right=0.96, top=0.8, bottom=0.16)
    save_figure(
        fig,
        out_dir,
        "poster_accuracy_best_cases_red_bg" if red_background else "poster_accuracy_best_cases",
    )


def plot_offspring_only(extreme_dir: Path, out_dir: Path) -> None:
    df = load_csv(extreme_dir / "scaling_summary.csv")
    if df.empty:
        return

    selected_cases = ["ultra_population", "pop_5m", "pop_10m", "pop_25m", "pop_50m"]
    seq_df = df[
        (df["alias_builder"] == "sequential")
        & (df["case_label"].isin(selected_cases))
    ].copy()
    if seq_df.empty:
        return

    best_rows = []
    for case_label in selected_cases:
        sub = seq_df[seq_df["case_label"] == case_label]
        if sub.empty:
            continue
        best = sub.loc[sub["mean_ms_per_gen"].astype(float).idxmin()]
        best_rows.append(
            {
                "case_label": case_label,
                "speedup": float(best["speedup_vs_t1"]),
                "threads": int(best["threads"]),
                "partial": case_label == "pop_50m",
            }
        )
    best_df = pd.DataFrame.from_records(best_rows)

    fig, ax = plt.subplots(figsize=(13.5, 7.0))
    x = np.arange(len(best_df))
    bars = ax.bar(
        x,
        best_df["speedup"],
        color=RED,
        edgecolor=BLACK,
        linewidth=2.8,
        width=0.72,
    )
    for bar, row in zip(bars, best_rows):
        if bool(row["partial"]):
            bar.set_hatch("//")
        if bool(row["partial"]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.14,
                f"{row['speedup']:.1f}x\nT{row['threads']}",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
                color=BLACK,
                bbox={"facecolor": WHITE, "edgecolor": BLACK, "linewidth": 1.0, "pad": 0.2},
            )
        else:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() - 0.14,
                f"{row['speedup']:.1f}x\nT{row['threads']}",
                ha="center",
                va="top",
                fontsize=10.5,
                fontweight="bold",
                color=bar_text_color(fill_color=RED),
            )

    ax.set_xticks(x)
    ax.set_xticklabels([SHORT_POP_LABELS.get(case, pretty_case_label(case)) for case in best_df["case_label"]], fontsize=12)
    ax.set_ylabel("Best measured speedup vs T1", fontsize=13)
    ax.set_ylim(0.0, max(best_df["speedup"]) * 1.23)
    ax.set_title("Embarrassingly parallel offspring generation alone", fontsize=15, pad=8)
    ax.grid(True, axis="y", linewidth=1.8, alpha=0.9)
    ax.set_axisbelow(True)
    thicken_axes(ax)
    ax.text(
        0.02,
        0.98,
        "Serial alias control only: parent-sampler build stays serial, so these gains isolate threaded offspring generation.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9.5,
        color=BLACK,
    )
    ax.text(
        0.98,
        0.03,
        "* partial run",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.5,
        color=BLACK,
    )
    fig.subplots_adjust(left=0.12, right=0.98, top=0.9, bottom=0.12)
    save_figure(fig, out_dir, "poster_offspring_parallel_only")


def plot_memory_efficiency(full_dir: Path, out_dir: Path) -> None:
    df = load_csv(full_dir / "scaling_summary.csv")
    if df.empty:
        return

    cases = ["ultra_population", "xlarge_long_genome", "xlarge_gamma_mix"]
    large_df = df[df["case_label"].isin(cases)].copy()
    if large_df.empty:
        return

    best_rows: list[dict[str, object]] = []
    for case_label in cases:
        for alias_builder in ALIAS_ORDER:
            sub = large_df[
                (large_df["case_label"] == case_label)
                & (large_df["alias_builder"] == alias_builder)
            ]
            if sub.empty:
                continue
            best = sub.loc[sub["mean_ms_per_gen"].astype(float).idxmin()]
            best_rows.append(
                {
                    "case_label": case_label,
                    "alias_builder": alias_builder,
                    "threads": int(best["threads"]),
                    "rss_gib": float(best["mean_peak_rss_kb"]) / (1024.0 * 1024.0),
                }
            )
    best_df = pd.DataFrame.from_records(best_rows)

    ultra_df = df[df["case_label"] == "ultra_population"].copy()
    ultra_df["vm_gib"] = giB_from_kb(ultra_df["mean_peak_vmsize_kb"])

    fig, axes = plt.subplots(1, 2, figsize=(16.0, 6.0), gridspec_kw={"width_ratios": [1.0, 1.0]})

    ax = axes[0]
    x = np.arange(len(cases))
    width = 0.24
    offsets = {"sequential": -width, "parallel": 0.0, "parallel_psa_plus": width}
    for alias_builder in ALIAS_ORDER:
        sub = best_df[best_df["alias_builder"] == alias_builder].set_index("case_label").reindex(cases)
        bars = ax.bar(
            x + offsets[alias_builder],
            sub["rss_gib"],
            width=width,
            color=PALETTE[alias_builder],
            edgecolor=BLACK,
            linewidth=2.6,
            label=label_alias(alias_builder),
        )
        for bar, threads in zip(bars, sub["threads"]):
            if pd.isna(threads):
                continue
            height = float(bar.get_height())
            label = f"T{int(threads)}"
            if height >= 0.04:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    height - 0.008,
                    label,
                    ha="center",
                    va="top",
                    fontsize=8.5,
                    fontweight="bold",
                    color=bar_text_color(alias_builder=alias_builder),
                )
            else:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    height + 0.003,
                    label,
                    ha="center",
                    va="bottom",
                    fontsize=8.5,
                    fontweight="bold",
                    color=BLACK,
                )
    ax.set_xticks(x)
    ax.set_xticklabels([pretty_case_label(case) for case in cases], fontsize=11)
    ax.set_ylabel("Peak RSS at best measured point (GiB)", fontsize=13)
    ax.set_ylim(0.0, max(best_df["rss_gib"]) * 1.28)
    ax.set_title("Resident memory stays compact", fontsize=14, pad=6)
    ax.grid(True, axis="y", linewidth=1.7, alpha=0.9)
    ax.set_axisbelow(True)
    thicken_axes(ax)

    ax = axes[1]
    for alias_builder in ALIAS_ORDER:
        sub = ultra_df[ultra_df["alias_builder"] == alias_builder].sort_values("threads")
        if sub.empty:
            continue
        ax.plot(
            sub["threads"],
            sub["vm_gib"],
            marker="o",
            markersize=6,
            linewidth=2.6,
            color=PALETTE[alias_builder],
            markeredgecolor=BLACK,
            label=label_alias(alias_builder),
        )
    ax.set_xscale("log", base=2)
    ax.set_xticks(sorted(set(int(value) for value in ultra_df["threads"])))
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("Threads", fontsize=13)
    ax.set_ylabel("Peak virtual memory (GiB)\nUltra population case", fontsize=13)
    ax.set_title("Virtual address space grows with thread count", fontsize=14, pad=6)
    ax.grid(True, axis="both", linewidth=1.7, alpha=0.9)
    ax.set_axisbelow(True)
    thicken_axes(ax)

    add_figure_header(
        fig,
        "Memory efficiency at scale",
        "Memory data from the completed full run.",
    )
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=PALETTE[alias], edgecolor=BLACK, linewidth=2.2)
        for alias in ALIAS_ORDER
    ]
    fig.legend(
        legend_handles,
        [label_alias(alias) for alias in ALIAS_ORDER],
        loc="upper center",
        bbox_to_anchor=(0.25, 0.885),
        frameon=False,
        ncol=3,
        fontsize=9,
        handlelength=2.2,
        columnspacing=1.0,
    )
    fig.subplots_adjust(left=0.08, right=0.985, top=0.82, bottom=0.14, wspace=0.22)
    save_figure(fig, out_dir, "poster_memory_efficiency")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate cut-down poster-ready figures from validation suite results."
    )
    parser.add_argument(
        "--full-dir",
        default=str(FULL_DEFAULT),
        help="Completed full-suite results directory.",
    )
    parser.add_argument(
        "--extreme-dir",
        default=str(EXTREME_DEFAULT),
        help="Extreme scaling results directory.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(OUT_DEFAULT),
        help="Directory for poster-ready figures.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    full_dir = Path(args.full_dir)
    extreme_dir = Path(args.extreme_dir)
    out_dir = Path(args.output_dir)

    configure_poster_matplotlib()
    plot_speedup_vs_slim(full_dir, out_dir)
    plot_speedup_vs_slim(full_dir, out_dir, red_background=True)
    plot_psa_specific_gains(full_dir, out_dir)
    plot_thread_population_scaling(full_dir, extreme_dir, out_dir)
    plot_accuracy_best_cases(full_dir, out_dir)
    plot_accuracy_best_cases(full_dir, out_dir, red_background=True)
    plot_offspring_only(extreme_dir, out_dir)
    plot_memory_efficiency(full_dir, out_dir)
    print(f"Poster figures written to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
