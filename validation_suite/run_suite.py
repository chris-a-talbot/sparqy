#!/usr/bin/env python3
# For local testing
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from io import StringIO
from pathlib import Path
from statistics import mean, stdev
from typing import Iterable


if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import scenarios as suite_scenarios  # type: ignore
else:
    from . import scenarios as suite_scenarios


REPO_ROOT = Path(__file__).resolve().parents[1]
SUITE_ROOT = Path(__file__).resolve().parent
DEFAULT_RESULTS_ROOT = SUITE_ROOT / "results"
DEFAULT_SPARQY_BIN = REPO_ROOT / "build" / "sparqy"
DEFAULT_SLIM_BIN = shutil.which("slim") or "/usr/local/bin/slim"

TOTAL_WALL_RE = re.compile(r"Total wall-clock:\s*([0-9.]+)\s+sec")
AVG_SEC_RE = re.compile(r"Avg sec/gen:\s*([0-9.]+)")
PROFILE_LINE_RE = re.compile(
    r"^\s+([a-zA-Z0-9_]+)\s+total=\s*([0-9.]+)\s+avg/gen=\s*([0-9.]+)\s+pct=\s*([0-9.]+)%$"
)


def log(message: str) -> None:
    print(message, flush=True)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def timestamp_label() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def parse_csv_list(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def numeric_summary(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return values[0], 0.0
    return mean(values), stdev(values)


def optional_numeric_summary(values: list[float]) -> tuple[float, float]:
    if not values:
        return math.nan, math.nan
    return numeric_summary(values)


def parse_number(text: str) -> float:
    return float(text.strip())


def parse_optional_number(value: object) -> float | None:
    text = str(value).strip()
    if not text:
        return None
    return float(text)


def parse_int_vector(text: str) -> list[int]:
    if not text.strip():
        return []
    return [int(part) for part in text.split(",") if part != ""]


def pad_vectors(vectors: list[list[int]]) -> list[list[float]]:
    width = max((len(vector) for vector in vectors), default=0)
    padded: list[list[float]] = []
    for vector in vectors:
        padded.append([float(value) for value in vector] + [0.0] * (width - len(vector)))
    return padded


def mean_vector(vectors: list[list[int]]) -> list[float]:
    padded = pad_vectors(vectors)
    if not padded:
        return []
    width = len(padded[0])
    return [
        sum(vector[index] for vector in padded) / len(padded)
        for index in range(width)
    ]


def normalized_distribution(values: list[float]) -> list[float]:
    total = sum(values)
    if total <= 0.0:
        return [0.0 for _value in values]
    return [value / total for value in values]


def total_variation_distance(a: list[float], b: list[float]) -> float:
    width = max(len(a), len(b))
    aa = a + [0.0] * (width - len(a))
    bb = b + [0.0] * (width - len(b))
    return 0.5 * sum(abs(x - y) for x, y in zip(aa, bb))


def max_abs_bin_diff(a: list[float], b: list[float]) -> float:
    width = max(len(a), len(b))
    aa = a + [0.0] * (width - len(a))
    bb = b + [0.0] * (width - len(b))
    return max((abs(x - y) for x, y in zip(aa, bb)), default=0.0)


def make_seed_list(replicates: int) -> list[int]:
    base = [101, 202, 303, 404, 505, 606, 707, 808]
    if replicates <= len(base):
        return base[:replicates]
    return base + [909 + 101 * index for index in range(replicates - len(base))]


def write_text(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    path.write_text(text)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    ensure_dir(path.parent)
    if not rows:
        path.write_text("")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key in seen:
                continue
            seen.add(key)
            fieldnames.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open() as handle:
        return list(csv.DictReader(handle))


@dataclass(frozen=True)
class CommandResult:
    stdout: str
    stderr: str
    peak_rss_kb: int | None
    peak_vmsize_kb: int | None


class ProcStatusTracker:
    def __init__(self, pid: int):
        self.status_path = Path(f"/proc/{pid}/status")
        self.peak_rss_kb = 0
        self.peak_hwm_kb = 0
        self.peak_vmsize_kb = 0
        self.peak_vmpeak_kb = 0
        self.supported = self.status_path.exists()

    def sample(self) -> None:
        if not self.supported:
            return
        try:
            status_text = self.status_path.read_text()
        except FileNotFoundError:
            return

        for line in status_text.splitlines():
            if line.startswith("VmRSS:"):
                self.peak_rss_kb = max(self.peak_rss_kb, _parse_status_kb(line))
            elif line.startswith("VmHWM:"):
                self.peak_hwm_kb = max(self.peak_hwm_kb, _parse_status_kb(line))
            elif line.startswith("VmSize:"):
                self.peak_vmsize_kb = max(self.peak_vmsize_kb, _parse_status_kb(line))
            elif line.startswith("VmPeak:"):
                self.peak_vmpeak_kb = max(self.peak_vmpeak_kb, _parse_status_kb(line))

    def command_metrics(self) -> tuple[int | None, int | None]:
        peak_rss_kb = self.peak_hwm_kb or self.peak_rss_kb or None
        peak_vmsize_kb = self.peak_vmpeak_kb or self.peak_vmsize_kb or None
        return peak_rss_kb, peak_vmsize_kb


def _parse_status_kb(line: str) -> int:
    match = re.search(r"(\d+)\s+kB", line)
    if not match:
        return 0
    return int(match.group(1))


def run_command(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    stdout_path: Path,
    stderr_path: Path,
) -> CommandResult:
    ensure_dir(stdout_path.parent)
    ensure_dir(stderr_path.parent)
    with stdout_path.open("w") as stdout_handle, stderr_path.open("w") as stderr_handle:
        process = subprocess.Popen(
            command,
            cwd=str(cwd),
            env=env,
            stdout=stdout_handle,
            stderr=stderr_handle,
            text=True,
        )
        tracker = ProcStatusTracker(process.pid)
        while True:
            tracker.sample()
            returncode = process.poll()
            if returncode is not None:
                break
            time.sleep(0.05)
        tracker.sample()

    stdout_text = stdout_path.read_text()
    stderr_text = stderr_path.read_text()
    peak_rss_kb, peak_vmsize_kb = tracker.command_metrics()
    if returncode != 0:
        memory_suffix = ""
        if peak_rss_kb is not None or peak_vmsize_kb is not None:
            memory_suffix = (
                f"\npeak_rss_kb: {peak_rss_kb if peak_rss_kb is not None else 'NA'}"
                f"\npeak_vmsize_kb: {peak_vmsize_kb if peak_vmsize_kb is not None else 'NA'}"
            )
        raise RuntimeError(
            f"Command failed with exit code {returncode}: {' '.join(command)}\n"
            f"stdout: {stdout_path}\nstderr: {stderr_path}{memory_suffix}"
        )
    return CommandResult(
        stdout=stdout_text,
        stderr=stderr_text,
        peak_rss_kb=peak_rss_kb,
        peak_vmsize_kb=peak_vmsize_kb,
    )


def build_sparqy(binary_path: Path) -> None:
    log("Configuring sparqy build...")
    subprocess.run(
        ["cmake", "-S", str(REPO_ROOT), "-B", str(REPO_ROOT / "build"), "-DCMAKE_BUILD_TYPE=Release"],
        cwd=str(REPO_ROOT),
        check=True,
    )
    log("Building sparqy...")
    subprocess.run(
        ["cmake", "--build", str(REPO_ROOT / "build"), "-j16"],
        cwd=str(REPO_ROOT),
        check=True,
    )
    if not binary_path.exists():
        raise FileNotFoundError(f"sparqy binary not found at {binary_path}")


def generate_suite_artifacts(results_dir: Path, preset: suite_scenarios.SuitePreset) -> None:
    ensure_dir(results_dir)
    generated_dir = results_dir / "generated"
    ensure_dir(generated_dir / "slim" / "runtime")
    ensure_dir(generated_dir / "slim" / "stats")

    manifest = {
        "generated_at": datetime.now().isoformat(),
        "repo_root": str(REPO_ROOT),
        "preset": preset.to_manifest(),
        "scenarios": {
            name: scenario.to_manifest()
            for name, scenario in suite_scenarios.all_unique_scenarios(preset).items()
        },
    }
    write_text(results_dir / "manifest.json", json.dumps(manifest, indent=2))

    for name, scenario in suite_scenarios.all_unique_scenarios(preset).items():
        runtime_script = suite_scenarios.render_slim_script(scenario, include_stats=False)
        stats_script = suite_scenarios.render_slim_script(scenario, include_stats=True)
        write_text(generated_dir / "slim" / "runtime" / f"{name}.slim", runtime_script)
        write_text(generated_dir / "slim" / "stats" / f"{name}.slim", stats_script)


def parse_sparqy_wall_metrics(stderr_text: str) -> tuple[float, float]:
    total_match = TOTAL_WALL_RE.search(stderr_text)
    avg_match = AVG_SEC_RE.search(stderr_text)
    if not total_match or not avg_match:
        raise ValueError("Unable to parse sparqy wall-clock summary from stderr")
    return float(total_match.group(1)), float(avg_match.group(1))


def parse_sparqy_accuracy(stdout_text: str) -> tuple[int, dict[str, float], dict[str, list[int]]]:
    reader = csv.DictReader(StringIO(stdout_text), delimiter="\t")
    generation = 0
    scalars: dict[str, float] = {}
    vectors: dict[str, list[int]] = {}
    for row in reader:
        if not row:
            continue
        generation = int(row["generation"])
        statistic = row["statistic"]
        scalar_value = row["scalar_value"].strip()
        if scalar_value:
            scalars[statistic] = float(scalar_value)
            continue
        if statistic == "mutation_histogram":
            vectors["mutation_histogram_by_type"] = parse_int_vector(row["by_type"])
            vectors["mutation_histogram_by_chromosome"] = parse_int_vector(row["by_chromosome"])
        elif statistic == "site_frequency_spectrum":
            vectors["sfs_unfolded"] = parse_int_vector(row["unfolded_sfs"])
            vectors["sfs_folded"] = parse_int_vector(row["folded_sfs"])
    return generation, scalars, vectors


def parse_slim_metrics(stdout_text: str) -> tuple[dict[str, float], dict[str, list[int]]]:
    scalars: dict[str, float] = {}
    vectors: dict[str, list[int]] = {}
    for line in stdout_text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split("\t", 2)
        if len(parts) != 3:
            continue
        kind, name, value = parts
        if kind == "scalar":
            scalars[name] = float(value)
        elif kind == "vector":
            vectors[name] = parse_int_vector(value)
    return scalars, vectors


def parse_profile_summary(stderr_text: str) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for line in stderr_text.splitlines():
        match = PROFILE_LINE_RE.match(line.rstrip())
        if not match:
            continue
        rows.append(
            {
                "phase": match.group(1),
                "total_sec": float(match.group(2)),
                "avg_sec_per_gen": float(match.group(3)),
                "pct_total": float(match.group(4)),
            }
        )
    if not rows:
        raise ValueError("Unable to parse sparqy profiler summary from stderr")
    return rows


def sparqy_env(threads: int) -> dict[str, str]:
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(threads)
    env.setdefault("OMP_PLACES", "cores")
    env.setdefault("OMP_PROC_BIND", "close")
    env.setdefault("OMP_STACKSIZE", "16M")
    return env


def render_sparqy_config_file(
    path: Path,
    *,
    scenario: suite_scenarios.Scenario,
    seed: int,
    threads: int,
    include_stats: bool,
) -> None:
    config_text = suite_scenarios.render_sparqy_config(
        scenario,
        seed=seed,
        threads=threads,
        include_stats=include_stats,
        include_pairwise=False,
    )
    write_text(path, config_text)


def variant_label(alias_builder: str, threads: int) -> str:
    return f"{alias_builder}@T{threads}"


def maybe_check_slim(phases: set[str], simulators: set[str], slim_bin: Path) -> None:
    needs_slim = bool({"accuracy", "speed"} & phases & {"accuracy", "speed"}) and "slim" in simulators
    if needs_slim and not slim_bin.exists():
        raise FileNotFoundError(
            f"SLiM binary not found at {slim_bin}. Pass --slim-bin or exclude slim runs."
        )


def run_accuracy_phase(
    *,
    preset: suite_scenarios.SuitePreset,
    results_dir: Path,
    sparqy_bin: Path,
    slim_bin: Path,
    simulators: set[str],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    scalar_rows: list[dict[str, object]] = []
    vector_rows: list[dict[str, object]] = []
    raw_dir = results_dir / "raw" / "accuracy"
    generated_dir = results_dir / "generated"

    if "slim" not in simulators and "sparqy" not in simulators:
        return scalar_rows, vector_rows

    for case in preset.accuracy_cases:
        seeds = make_seed_list(case.replicates)
        log(f"[accuracy] {case.label}: {case.replicates} replicate(s)")

        for replicate_index, seed in enumerate(seeds, start=1):
            if "slim" in simulators:
                slim_script = generated_dir / "slim" / "stats" / f"{case.scenario.name}.slim"
                slim_result = run_command(
                    [str(slim_bin), "-s", str(seed), str(slim_script)],
                    cwd=REPO_ROOT,
                    env=os.environ.copy(),
                    stdout_path=raw_dir / case.label / f"slim_seed{seed}.stdout",
                    stderr_path=raw_dir / case.label / f"slim_seed{seed}.stderr",
                )
                slim_stdout = slim_result.stdout
                slim_stderr = slim_result.stderr
                if slim_stderr.strip():
                    log(f"[accuracy] slim stderr for {case.label} seed {seed}: {slim_stderr.strip()}")
                slim_scalars, slim_vectors = parse_slim_metrics(slim_stdout)
                for metric, value in slim_scalars.items():
                    scalar_rows.append(
                        {
                            "case_label": case.label,
                            "scenario_name": case.scenario.name,
                            "scenario_description": case.scenario.description,
                            "simulator": "slim",
                            "variant": "slim",
                            "alias_builder": "",
                            "threads": 1,
                            "replicate": replicate_index,
                            "seed": seed,
                            "generation": case.scenario.G,
                            "metric": metric,
                            "value": value,
                        }
                    )
                for statistic, values in slim_vectors.items():
                    vector_rows.append(
                        {
                            "case_label": case.label,
                            "scenario_name": case.scenario.name,
                            "scenario_description": case.scenario.description,
                            "simulator": "slim",
                            "variant": "slim",
                            "alias_builder": "",
                            "threads": 1,
                            "replicate": replicate_index,
                            "seed": seed,
                            "generation": case.scenario.G,
                            "statistic": statistic,
                            "values_csv": ",".join(str(value) for value in values),
                        }
                    )

            if "sparqy" not in simulators:
                continue

            for variant in preset.accuracy_variants:
                config_path = (
                    generated_dir
                    / "sparqy"
                    / "accuracy"
                    / case.label
                    / f"{variant.alias_builder}_T{variant.threads}_seed{seed}.sparqy"
                )
                render_sparqy_config_file(
                    config_path,
                    scenario=case.scenario,
                    seed=seed,
                    threads=variant.threads,
                    include_stats=True,
                )
                sparqy_result = run_command(
                    [
                        str(sparqy_bin),
                        "--config",
                        str(config_path),
                        f"--alias-builder={variant.alias_builder}",
                    ],
                    cwd=REPO_ROOT,
                    env=sparqy_env(variant.threads),
                    stdout_path=raw_dir
                    / case.label
                    / f"sparqy_{variant.alias_builder}_T{variant.threads}_seed{seed}.stdout",
                    stderr_path=raw_dir
                    / case.label
                    / f"sparqy_{variant.alias_builder}_T{variant.threads}_seed{seed}.stderr",
                )
                stdout_text = sparqy_result.stdout
                stderr_text = sparqy_result.stderr
                generation, scalars, vectors = parse_sparqy_accuracy(stdout_text)
                total_sec, avg_sec = parse_sparqy_wall_metrics(stderr_text)
                scalars = dict(scalars)
                scalars["total_wall_sec"] = total_sec
                scalars["avg_sec_per_gen"] = avg_sec
                for metric, value in scalars.items():
                    scalar_rows.append(
                        {
                            "case_label": case.label,
                            "scenario_name": case.scenario.name,
                            "scenario_description": case.scenario.description,
                            "simulator": "sparqy",
                            "variant": variant.label,
                            "alias_builder": variant.alias_builder,
                            "threads": variant.threads,
                            "replicate": replicate_index,
                            "seed": seed,
                            "generation": generation,
                            "metric": metric,
                            "value": value,
                        }
                    )
                for statistic, values in vectors.items():
                    vector_rows.append(
                        {
                            "case_label": case.label,
                            "scenario_name": case.scenario.name,
                            "scenario_description": case.scenario.description,
                            "simulator": "sparqy",
                            "variant": variant.label,
                            "alias_builder": variant.alias_builder,
                            "threads": variant.threads,
                            "replicate": replicate_index,
                            "seed": seed,
                            "generation": generation,
                            "statistic": statistic,
                            "values_csv": ",".join(str(value) for value in values),
                        }
                    )

    return scalar_rows, vector_rows


def run_speed_phase(
    *,
    preset: suite_scenarios.SuitePreset,
    results_dir: Path,
    sparqy_bin: Path,
    slim_bin: Path,
    simulators: set[str],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    raw_dir = results_dir / "raw" / "speed"
    generated_dir = results_dir / "generated"

    for case in preset.speed_cases:
        seeds = make_seed_list(case.replicates)
        log(f"[speed] {case.label}: {case.replicates} replicate(s)")
        for replicate_index, seed in enumerate(seeds, start=1):
            if "slim" in simulators:
                slim_script = generated_dir / "slim" / "runtime" / f"{case.scenario.name}.slim"
                slim_result = run_command(
                    [str(slim_bin), "-s", str(seed), str(slim_script)],
                    cwd=REPO_ROOT,
                    env=os.environ.copy(),
                    stdout_path=raw_dir / case.label / f"slim_seed{seed}.stdout",
                    stderr_path=raw_dir / case.label / f"slim_seed{seed}.stderr",
                )
                slim_stdout = slim_result.stdout
                slim_stderr = slim_result.stderr
                if slim_stderr.strip():
                    log(f"[speed] slim stderr for {case.label} seed {seed}: {slim_stderr.strip()}")
                slim_scalars, _slim_vectors = parse_slim_metrics(slim_stdout)
                total_sec = slim_scalars["total_wall_sec"]
                avg_sec = total_sec / case.scenario.G
                rows.append(
                    {
                        "case_label": case.label,
                        "scenario_name": case.scenario.name,
                        "scenario_description": case.scenario.description,
                        "simulator": "slim",
                        "alias_builder": "",
                        "threads": 1,
                        "replicate": replicate_index,
                        "seed": seed,
                        "total_sec": total_sec,
                        "avg_sec_per_gen": avg_sec,
                        "ms_per_gen": avg_sec * 1000.0,
                        "peak_rss_kb": slim_result.peak_rss_kb,
                        "peak_vmsize_kb": slim_result.peak_vmsize_kb,
                    }
                )

            if "sparqy" not in simulators:
                continue

            for alias_builder in preset.speed_alias_builders:
                for threads in preset.speed_threads:
                    config_path = (
                        generated_dir
                        / "sparqy"
                        / "speed"
                        / case.label
                        / f"{alias_builder}_T{threads}_seed{seed}.sparqy"
                    )
                    render_sparqy_config_file(
                        config_path,
                        scenario=case.scenario,
                        seed=seed,
                        threads=threads,
                        include_stats=False,
                    )
                    sparqy_result = run_command(
                        [
                            str(sparqy_bin),
                            "--config",
                            str(config_path),
                            f"--alias-builder={alias_builder}",
                        ],
                        cwd=REPO_ROOT,
                        env=sparqy_env(threads),
                        stdout_path=raw_dir / case.label / f"sparqy_{alias_builder}_T{threads}_seed{seed}.stdout",
                        stderr_path=raw_dir / case.label / f"sparqy_{alias_builder}_T{threads}_seed{seed}.stderr",
                    )
                    stderr_text = sparqy_result.stderr
                    total_sec, avg_sec = parse_sparqy_wall_metrics(stderr_text)
                    rows.append(
                        {
                            "case_label": case.label,
                            "scenario_name": case.scenario.name,
                            "scenario_description": case.scenario.description,
                            "simulator": "sparqy",
                            "alias_builder": alias_builder,
                            "threads": threads,
                            "replicate": replicate_index,
                            "seed": seed,
                            "total_sec": total_sec,
                            "avg_sec_per_gen": avg_sec,
                            "ms_per_gen": avg_sec * 1000.0,
                            "peak_rss_kb": sparqy_result.peak_rss_kb,
                            "peak_vmsize_kb": sparqy_result.peak_vmsize_kb,
                        }
                    )

    return rows


def run_scaling_phase(
    *,
    preset: suite_scenarios.SuitePreset,
    results_dir: Path,
    sparqy_bin: Path,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    raw_dir = results_dir / "raw" / "scaling"
    generated_dir = results_dir / "generated"

    for case in preset.scaling_cases:
        seeds = make_seed_list(case.replicates)
        log(f"[scaling] {case.label}: {case.replicates} replicate(s)")
        for alias_builder in preset.scaling_alias_builders:
            for threads in preset.scaling_threads_for(case.label):
                for replicate_index, seed in enumerate(seeds, start=1):
                    config_path = (
                        generated_dir
                        / "sparqy"
                        / "scaling"
                        / case.label
                        / f"{alias_builder}_T{threads}_seed{seed}.sparqy"
                    )
                    render_sparqy_config_file(
                        config_path,
                        scenario=case.scenario,
                        seed=seed,
                        threads=threads,
                        include_stats=False,
                    )
                    sparqy_result = run_command(
                        [
                            str(sparqy_bin),
                            "--config",
                            str(config_path),
                            f"--alias-builder={alias_builder}",
                        ],
                        cwd=REPO_ROOT,
                        env=sparqy_env(threads),
                        stdout_path=raw_dir / case.label / f"{alias_builder}_T{threads}_seed{seed}.stdout",
                        stderr_path=raw_dir / case.label / f"{alias_builder}_T{threads}_seed{seed}.stderr",
                    )
                    stderr_text = sparqy_result.stderr
                    total_sec, avg_sec = parse_sparqy_wall_metrics(stderr_text)
                    rows.append(
                        {
                            "case_label": case.label,
                            "scenario_name": case.scenario.name,
                            "scenario_description": case.scenario.description,
                            "alias_builder": alias_builder,
                            "threads": threads,
                            "replicate": replicate_index,
                            "seed": seed,
                            "total_sec": total_sec,
                            "avg_sec_per_gen": avg_sec,
                            "ms_per_gen": avg_sec * 1000.0,
                            "peak_rss_kb": sparqy_result.peak_rss_kb,
                            "peak_vmsize_kb": sparqy_result.peak_vmsize_kb,
                        }
                    )
    return rows


def run_profile_phase(
    *,
    preset: suite_scenarios.SuitePreset,
    results_dir: Path,
    sparqy_bin: Path,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    run_rows: list[dict[str, object]] = []
    phase_rows: list[dict[str, object]] = []
    raw_dir = results_dir / "raw" / "profile"
    generated_dir = results_dir / "generated"

    for case in preset.profile_cases:
        seeds = make_seed_list(case.replicates)
        log(f"[profile] {case.label}: {case.replicates} replicate(s)")
        for alias_builder in preset.profile_alias_builders:
            for threads in preset.profile_threads_for(case.label):
                for replicate_index, seed in enumerate(seeds, start=1):
                    config_path = (
                        generated_dir
                        / "sparqy"
                        / "profile"
                        / case.label
                        / f"{alias_builder}_T{threads}_seed{seed}.sparqy"
                    )
                    render_sparqy_config_file(
                        config_path,
                        scenario=case.scenario,
                        seed=seed,
                        threads=threads,
                        include_stats=False,
                    )
                    sparqy_result = run_command(
                        [
                            str(sparqy_bin),
                            "--config",
                            str(config_path),
                            "--profile",
                            f"--alias-builder={alias_builder}",
                        ],
                        cwd=REPO_ROOT,
                        env=sparqy_env(threads),
                        stdout_path=raw_dir / case.label / f"{alias_builder}_T{threads}_seed{seed}.stdout",
                        stderr_path=raw_dir / case.label / f"{alias_builder}_T{threads}_seed{seed}.stderr",
                    )
                    stderr_text = sparqy_result.stderr
                    total_sec, avg_sec = parse_sparqy_wall_metrics(stderr_text)
                    run_rows.append(
                        {
                            "case_label": case.label,
                            "scenario_name": case.scenario.name,
                            "scenario_description": case.scenario.description,
                            "alias_builder": alias_builder,
                            "threads": threads,
                            "replicate": replicate_index,
                            "seed": seed,
                            "total_sec": total_sec,
                            "avg_sec_per_gen": avg_sec,
                            "ms_per_gen": avg_sec * 1000.0,
                            "peak_rss_kb": sparqy_result.peak_rss_kb,
                            "peak_vmsize_kb": sparqy_result.peak_vmsize_kb,
                        }
                    )

                    for phase_row in parse_profile_summary(stderr_text):
                        phase_rows.append(
                            {
                                "case_label": case.label,
                                "scenario_name": case.scenario.name,
                                "scenario_description": case.scenario.description,
                                "alias_builder": alias_builder,
                                "threads": threads,
                                "replicate": replicate_index,
                                "seed": seed,
                                **phase_row,
                            }
                        )
    return run_rows, phase_rows


def summarize_accuracy(
    scalar_rows: list[dict[str, object]],
    vector_rows: list[dict[str, object]],
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    scalar_groups: dict[tuple[str, str, str, str], list[float]] = defaultdict(list)
    scalar_meta: dict[tuple[str, str, str, str], dict[str, object]] = {}
    for row in scalar_rows:
        key = (
            str(row["case_label"]),
            str(row["scenario_name"]),
            str(row["simulator"]),
            str(row["metric"]),
        )
        if row["simulator"] == "sparqy":
            key = key + (str(row["variant"]),)
        else:
            key = key + ("slim",)
        scalar_groups[key].append(float(row["value"]))
        scalar_meta[key] = row

    scalar_summary: list[dict[str, object]] = []
    comparisons: list[dict[str, object]] = []

    slim_lookup: dict[tuple[str, str], tuple[float, float, int]] = {}
    for key, values in scalar_groups.items():
        case_label, scenario_name, simulator, metric, variant = key
        value_mean, value_sd = numeric_summary(values)
        summary_row = {
            "case_label": case_label,
            "scenario_name": scenario_name,
            "simulator": simulator,
            "variant": variant,
            "metric": metric,
            "n_reps": len(values),
            "mean": value_mean,
            "sd": value_sd,
        }
        scalar_summary.append(summary_row)
        if simulator == "slim":
            slim_lookup[(case_label, metric)] = (value_mean, value_sd, len(values))

    for key, values in scalar_groups.items():
        case_label, scenario_name, simulator, metric, variant = key
        if simulator != "sparqy":
            continue
        slim_stats = slim_lookup.get((case_label, metric))
        if not slim_stats:
            continue
        sparqy_mean, sparqy_sd = numeric_summary(values)
        slim_mean, slim_sd, slim_n = slim_stats
        abs_diff = abs(sparqy_mean - slim_mean)
        denom = abs(slim_mean) if abs(slim_mean) > 1e-12 else 1.0
        rel_diff = abs_diff / denom
        meta_row = scalar_meta[key]
        comparisons.append(
            {
                "case_label": case_label,
                "scenario_name": scenario_name,
                "scenario_description": meta_row["scenario_description"],
                "variant": variant,
                "alias_builder": meta_row["alias_builder"],
                "threads": meta_row["threads"],
                "metric": metric,
                "sparqy_mean": sparqy_mean,
                "sparqy_sd": sparqy_sd,
                "sparqy_n": len(values),
                "slim_mean": slim_mean,
                "slim_sd": slim_sd,
                "slim_n": slim_n,
                "abs_diff": abs_diff,
                "rel_diff": rel_diff,
            }
        )

    vector_groups: dict[tuple[str, str, str, str], list[list[int]]] = defaultdict(list)
    vector_meta: dict[tuple[str, str, str, str], dict[str, object]] = {}
    for row in vector_rows:
        key = (
            str(row["case_label"]),
            str(row["scenario_name"]),
            str(row["simulator"]),
            str(row["statistic"]),
        )
        if row["simulator"] == "sparqy":
            key = key + (str(row["variant"]),)
        else:
            key = key + ("slim",)
        vector_groups[key].append(parse_int_vector(str(row["values_csv"])))
        vector_meta[key] = row

    vector_summary: list[dict[str, object]] = []
    slim_vector_lookup: dict[tuple[str, str], tuple[list[float], float]] = {}
    for key, vectors in vector_groups.items():
        case_label, scenario_name, simulator, statistic, variant = key
        mean_values = mean_vector(vectors)
        total_mean = sum(mean_values)
        summary_row = {
            "case_label": case_label,
            "scenario_name": scenario_name,
            "simulator": simulator,
            "variant": variant,
            "statistic": statistic,
            "n_reps": len(vectors),
            "mean_values_csv": ",".join(f"{value:.10f}" for value in mean_values),
            "mean_total": total_mean,
        }
        vector_summary.append(summary_row)
        if simulator == "slim":
            slim_vector_lookup[(case_label, statistic)] = (mean_values, total_mean)

    overview_by_variant: dict[str, dict[str, float | str]] = {}
    for key, vectors in vector_groups.items():
        case_label, scenario_name, simulator, statistic, variant = key
        if simulator != "sparqy":
            continue
        slim_data = slim_vector_lookup.get((case_label, statistic))
        if not slim_data:
            continue
        meta_row = vector_meta[key]
        sparqy_mean_values = mean_vector(vectors)
        slim_mean_values, slim_total = slim_data
        sparqy_dist = normalized_distribution(sparqy_mean_values)
        slim_dist = normalized_distribution(slim_mean_values)
        tvd = total_variation_distance(sparqy_dist, slim_dist)
        max_bin = max_abs_bin_diff(sparqy_dist, slim_dist)
        vector_summary.append(
            {
                "case_label": case_label,
                "scenario_name": scenario_name,
                "simulator": "comparison",
                "variant": variant,
                "statistic": statistic,
                "n_reps": len(vectors),
                "mean_values_csv": "",
                "mean_total": sum(sparqy_mean_values),
                "slim_mean_total": slim_total,
                "tvd": tvd,
                "max_bin_abs_diff": max_bin,
                "alias_builder": meta_row["alias_builder"],
                "threads": meta_row["threads"],
                "scenario_description": meta_row["scenario_description"],
            }
        )

    biological_accuracy_metrics = {
        "mean_fitness",
        "genetic_load",
        "realized_masking_bonus",
        "exact_B",
        "n_seg",
        "n_fixed",
        "nucleotide_diversity",
        "expected_heterozygosity",
    }

    scalar_max_by_variant: dict[str, float] = defaultdict(float)
    vector_max_by_variant: dict[str, float] = defaultdict(float)
    variant_meta: dict[str, tuple[str, int]] = {}
    for row in comparisons:
        variant = str(row["variant"])
        if str(row["metric"]) in biological_accuracy_metrics:
            scalar_max_by_variant[variant] = max(
                scalar_max_by_variant[variant],
                float(row["rel_diff"]),
            )
        variant_meta[variant] = (str(row["alias_builder"]), int(row["threads"]))
    for row in vector_summary:
        if row.get("simulator") != "comparison":
            continue
        variant = str(row["variant"])
        vector_max_by_variant[variant] = max(
            vector_max_by_variant[variant],
            float(row["tvd"]),
        )
        variant_meta[variant] = (str(row["alias_builder"]), int(row["threads"]))

    overview_rows: list[dict[str, object]] = []
    for variant in sorted(variant_meta):
        alias_builder, threads = variant_meta[variant]
        overview_rows.append(
            {
                "variant": variant,
                "alias_builder": alias_builder,
                "threads": threads,
                "max_scalar_rel_diff": scalar_max_by_variant.get(variant, 0.0),
                "max_vector_tvd": vector_max_by_variant.get(variant, 0.0),
            }
        )

    return scalar_summary, comparisons, vector_summary, overview_rows


def summarize_speed(speed_rows: list[dict[str, object]]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    groups: dict[tuple[str, str, str, int], list[float]] = defaultdict(list)
    totals: dict[tuple[str, str, str, int], list[float]] = defaultdict(list)
    rss_values: dict[tuple[str, str, str, int], list[float]] = defaultdict(list)
    vmsize_values: dict[tuple[str, str, str, int], list[float]] = defaultdict(list)
    meta: dict[tuple[str, str, str, int], dict[str, object]] = {}
    for row in speed_rows:
        key = (
            str(row["case_label"]),
            str(row["simulator"]),
            str(row["alias_builder"]),
            int(row["threads"]),
        )
        groups[key].append(float(row["ms_per_gen"]))
        totals[key].append(float(row["total_sec"]))
        peak_rss_kb = parse_optional_number(row.get("peak_rss_kb", ""))
        if peak_rss_kb is not None:
            rss_values[key].append(peak_rss_kb)
        peak_vmsize_kb = parse_optional_number(row.get("peak_vmsize_kb", ""))
        if peak_vmsize_kb is not None:
            vmsize_values[key].append(peak_vmsize_kb)
        meta[key] = row

    summary_rows: list[dict[str, object]] = []
    slim_mean_by_case: dict[str, float] = {}
    for key, values in groups.items():
        case_label, simulator, alias_builder, threads = key
        ms_mean, ms_sd = numeric_summary(values)
        total_mean, total_sd = numeric_summary(totals[key])
        rss_mean, rss_sd = optional_numeric_summary(rss_values[key])
        vmsize_mean, vmsize_sd = optional_numeric_summary(vmsize_values[key])
        row = meta[key]
        summary_rows.append(
            {
                "case_label": case_label,
                "scenario_name": row["scenario_name"],
                "scenario_description": row["scenario_description"],
                "simulator": simulator,
                "alias_builder": alias_builder,
                "threads": threads,
                "n_reps": len(values),
                "mean_ms_per_gen": ms_mean,
                "sd_ms_per_gen": ms_sd,
                "mean_total_sec": total_mean,
                "sd_total_sec": total_sd,
                "mean_peak_rss_kb": rss_mean,
                "sd_peak_rss_kb": rss_sd,
                "mean_peak_vmsize_kb": vmsize_mean,
                "sd_peak_vmsize_kb": vmsize_sd,
            }
        )
        if simulator == "slim":
            slim_mean_by_case[case_label] = ms_mean

    best_rows: list[dict[str, object]] = []
    best_by_case_builder: dict[tuple[str, str], dict[str, object]] = {}
    for row in summary_rows:
        if row["simulator"] != "sparqy":
            continue
        key = (str(row["case_label"]), str(row["alias_builder"]))
        current = best_by_case_builder.get(key)
        if current is None or float(row["mean_ms_per_gen"]) < float(current["best_mean_ms_per_gen"]):
            best_by_case_builder[key] = {
                "case_label": row["case_label"],
                "scenario_name": row["scenario_name"],
                "scenario_description": row["scenario_description"],
                "alias_builder": row["alias_builder"],
                "best_threads": row["threads"],
                "best_mean_ms_per_gen": row["mean_ms_per_gen"],
                "best_sd_ms_per_gen": row["sd_ms_per_gen"],
            }

    for key, row in sorted(best_by_case_builder.items()):
        slim_mean = slim_mean_by_case.get(str(row["case_label"]))
        speedup = (slim_mean / float(row["best_mean_ms_per_gen"])) if slim_mean else math.nan
        best_rows.append(
            {
                **row,
                "slim_mean_ms_per_gen": slim_mean,
                "speedup_vs_slim": speedup,
            }
        )

    return summary_rows, best_rows


def summarize_scaling(scaling_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    totals: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    rss_values: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    vmsize_values: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    meta: dict[tuple[str, str, int], dict[str, object]] = {}
    for row in scaling_rows:
        key = (str(row["case_label"]), str(row["alias_builder"]), int(row["threads"]))
        groups[key].append(float(row["ms_per_gen"]))
        totals[key].append(float(row["total_sec"]))
        peak_rss_kb = parse_optional_number(row.get("peak_rss_kb", ""))
        if peak_rss_kb is not None:
            rss_values[key].append(peak_rss_kb)
        peak_vmsize_kb = parse_optional_number(row.get("peak_vmsize_kb", ""))
        if peak_vmsize_kb is not None:
            vmsize_values[key].append(peak_vmsize_kb)
        meta[key] = row

    base_ms: dict[tuple[str, str], float] = {}
    for key, values in groups.items():
        case_label, alias_builder, threads = key
        if threads == 1:
            base_ms[(case_label, alias_builder)] = mean(values)

    summary_rows: list[dict[str, object]] = []
    for key, values in sorted(groups.items()):
        case_label, alias_builder, threads = key
        row = meta[key]
        ms_mean, ms_sd = numeric_summary(values)
        total_mean, total_sd = numeric_summary(totals[key])
        rss_mean, rss_sd = optional_numeric_summary(rss_values[key])
        vmsize_mean, vmsize_sd = optional_numeric_summary(vmsize_values[key])
        baseline = base_ms.get((case_label, alias_builder), ms_mean)
        speedup = baseline / ms_mean if ms_mean > 0.0 else math.nan
        efficiency = speedup / threads if threads > 0 else math.nan
        summary_rows.append(
            {
                "case_label": case_label,
                "scenario_name": row["scenario_name"],
                "scenario_description": row["scenario_description"],
                "alias_builder": alias_builder,
                "threads": threads,
                "n_reps": len(values),
                "mean_ms_per_gen": ms_mean,
                "sd_ms_per_gen": ms_sd,
                "mean_total_sec": total_mean,
                "sd_total_sec": total_sd,
                "mean_peak_rss_kb": rss_mean,
                "sd_peak_rss_kb": rss_sd,
                "mean_peak_vmsize_kb": vmsize_mean,
                "sd_peak_vmsize_kb": vmsize_sd,
                "speedup_vs_t1": speedup,
                "parallel_efficiency": efficiency,
            }
        )
    return summary_rows


def summarize_profile(
    profile_run_rows: list[dict[str, object]],
    profile_phase_rows: list[dict[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    run_groups: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    run_totals: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    run_rss_values: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    run_vmsize_values: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    run_meta: dict[tuple[str, str, int], dict[str, object]] = {}
    for row in profile_run_rows:
        key = (str(row["case_label"]), str(row["alias_builder"]), int(row["threads"]))
        run_groups[key].append(float(row["ms_per_gen"]))
        run_totals[key].append(float(row["total_sec"]))
        peak_rss_kb = parse_optional_number(row.get("peak_rss_kb", ""))
        if peak_rss_kb is not None:
            run_rss_values[key].append(peak_rss_kb)
        peak_vmsize_kb = parse_optional_number(row.get("peak_vmsize_kb", ""))
        if peak_vmsize_kb is not None:
            run_vmsize_values[key].append(peak_vmsize_kb)
        run_meta[key] = row

    run_summary: list[dict[str, object]] = []
    for key, values in sorted(run_groups.items()):
        case_label, alias_builder, threads = key
        row = run_meta[key]
        ms_mean, ms_sd = numeric_summary(values)
        total_mean, total_sd = numeric_summary(run_totals[key])
        rss_mean, rss_sd = optional_numeric_summary(run_rss_values[key])
        vmsize_mean, vmsize_sd = optional_numeric_summary(run_vmsize_values[key])
        run_summary.append(
            {
                "case_label": case_label,
                "scenario_name": row["scenario_name"],
                "scenario_description": row["scenario_description"],
                "alias_builder": alias_builder,
                "threads": threads,
                "n_reps": len(values),
                "mean_ms_per_gen": ms_mean,
                "sd_ms_per_gen": ms_sd,
                "mean_total_sec": total_mean,
                "sd_total_sec": total_sd,
                "mean_peak_rss_kb": rss_mean,
                "sd_peak_rss_kb": rss_sd,
                "mean_peak_vmsize_kb": vmsize_mean,
                "sd_peak_vmsize_kb": vmsize_sd,
            }
        )

    phase_groups: dict[tuple[str, str, int, str], list[tuple[float, float, float]]] = defaultdict(list)
    phase_meta: dict[tuple[str, str, int, str], dict[str, object]] = {}
    for row in profile_phase_rows:
        key = (
            str(row["case_label"]),
            str(row["alias_builder"]),
            int(row["threads"]),
            str(row["phase"]),
        )
        phase_groups[key].append(
            (
                float(row["total_sec"]),
                float(row["avg_sec_per_gen"]),
                float(row["pct_total"]),
            )
        )
        phase_meta[key] = row

    phase_summary: list[dict[str, object]] = []
    for key, triplets in sorted(phase_groups.items()):
        case_label, alias_builder, threads, phase = key
        row = phase_meta[key]
        total_mean, total_sd = numeric_summary([triplet[0] for triplet in triplets])
        avg_mean, avg_sd = numeric_summary([triplet[1] for triplet in triplets])
        pct_mean, pct_sd = numeric_summary([triplet[2] for triplet in triplets])
        phase_summary.append(
            {
                "case_label": case_label,
                "scenario_name": row["scenario_name"],
                "scenario_description": row["scenario_description"],
                "alias_builder": alias_builder,
                "threads": threads,
                "phase": phase,
                "n_reps": len(triplets),
                "mean_total_sec": total_mean,
                "sd_total_sec": total_sd,
                "mean_avg_sec_per_gen": avg_mean,
                "sd_avg_sec_per_gen": avg_sd,
                "mean_pct_total": pct_mean,
                "sd_pct_total": pct_sd,
            }
        )

    return run_summary, phase_summary


def summarize_results(results_dir: Path) -> None:
    accuracy_scalar_rows = load_csv_rows(results_dir / "accuracy_scalars.csv")
    accuracy_vector_rows = load_csv_rows(results_dir / "accuracy_vectors.csv")
    speed_rows = load_csv_rows(results_dir / "speed_runs.csv")
    scaling_rows = load_csv_rows(results_dir / "scaling_runs.csv")
    profile_run_rows = load_csv_rows(results_dir / "profile_runs.csv")
    profile_phase_rows = load_csv_rows(results_dir / "profile_phases.csv")

    if accuracy_scalar_rows or accuracy_vector_rows:
        scalar_summary, scalar_comparisons, vector_summary, overview_rows = summarize_accuracy(
            [dict(row) for row in accuracy_scalar_rows],
            [dict(row) for row in accuracy_vector_rows],
        )
        write_csv(results_dir / "accuracy_scalar_summary.csv", scalar_summary)
        write_csv(results_dir / "accuracy_scalar_comparison.csv", scalar_comparisons)
        write_csv(results_dir / "accuracy_vector_summary.csv", vector_summary)
        write_csv(results_dir / "accuracy_variant_overview.csv", overview_rows)

    if speed_rows:
        speed_summary, best_rows = summarize_speed([dict(row) for row in speed_rows])
        write_csv(results_dir / "speed_summary.csv", speed_summary)
        write_csv(results_dir / "speed_best_vs_slim.csv", best_rows)

    if scaling_rows:
        scaling_summary = summarize_scaling([dict(row) for row in scaling_rows])
        write_csv(results_dir / "scaling_summary.csv", scaling_summary)

    if profile_run_rows or profile_phase_rows:
        run_summary, phase_summary = summarize_profile(
            [dict(row) for row in profile_run_rows],
            [dict(row) for row in profile_phase_rows],
        )
        write_csv(results_dir / "profile_run_summary.csv", run_summary)
        write_csv(results_dir / "profile_phase_summary.csv", phase_summary)


def command_generate(args: argparse.Namespace) -> int:
    preset = suite_scenarios.get_preset(args.preset)
    results_dir = Path(args.results_dir)
    generate_suite_artifacts(results_dir, preset)
    log(f"Generated validation artifacts in {results_dir}")
    return 0


def command_run(args: argparse.Namespace) -> int:
    preset = suite_scenarios.get_preset(args.preset)
    results_dir = Path(args.results_dir)
    sparqy_bin = Path(args.sparqy_bin)
    slim_bin = Path(args.slim_bin)
    phases = set(parse_csv_list(args.phases))
    simulators = set(parse_csv_list(args.simulators))

    ensure_dir(results_dir)
    generate_suite_artifacts(results_dir, preset)
    maybe_check_slim(phases, simulators, slim_bin)

    if args.build:
        build_sparqy(sparqy_bin)
    elif not sparqy_bin.exists():
        raise FileNotFoundError(
            f"sparqy binary not found at {sparqy_bin}. Build it first or pass --build."
        )

    accuracy_scalar_rows: list[dict[str, object]] = []
    accuracy_vector_rows: list[dict[str, object]] = []
    speed_rows: list[dict[str, object]] = []
    scaling_rows: list[dict[str, object]] = []
    profile_run_rows: list[dict[str, object]] = []
    profile_phase_rows: list[dict[str, object]] = []

    if "accuracy" in phases:
        accuracy_scalar_rows, accuracy_vector_rows = run_accuracy_phase(
            preset=preset,
            results_dir=results_dir,
            sparqy_bin=sparqy_bin,
            slim_bin=slim_bin,
            simulators=simulators,
        )
        write_csv(results_dir / "accuracy_scalars.csv", accuracy_scalar_rows)
        write_csv(results_dir / "accuracy_vectors.csv", accuracy_vector_rows)

    if "speed" in phases:
        speed_rows = run_speed_phase(
            preset=preset,
            results_dir=results_dir,
            sparqy_bin=sparqy_bin,
            slim_bin=slim_bin,
            simulators=simulators,
        )
        write_csv(results_dir / "speed_runs.csv", speed_rows)

    if "scaling" in phases:
        scaling_rows = run_scaling_phase(
            preset=preset,
            results_dir=results_dir,
            sparqy_bin=sparqy_bin,
        )
        write_csv(results_dir / "scaling_runs.csv", scaling_rows)

    if "profile" in phases:
        profile_run_rows, profile_phase_rows = run_profile_phase(
            preset=preset,
            results_dir=results_dir,
            sparqy_bin=sparqy_bin,
        )
        write_csv(results_dir / "profile_runs.csv", profile_run_rows)
        write_csv(results_dir / "profile_phases.csv", profile_phase_rows)

    summarize_results(results_dir)

    metadata = {
        "completed_at": datetime.now().isoformat(),
        "results_dir": str(results_dir),
        "preset": preset.name,
        "phases": sorted(phases),
        "simulators": sorted(simulators),
        "sparqy_bin": str(sparqy_bin),
        "slim_bin": str(slim_bin),
        "memory_sampling": "linux_proc_status_peak_rss_and_vmsize",
    }
    write_text(results_dir / "run_metadata.json", json.dumps(metadata, indent=2))

    log(f"Run complete. Results written to {results_dir}")
    return 0


def command_summarize(args: argparse.Namespace) -> int:
    summarize_results(Path(args.results_dir))
    log(f"Summaries updated in {args.results_dir}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    default_results_dir = DEFAULT_RESULTS_ROOT / f"run_{timestamp_label()}"

    parser = argparse.ArgumentParser(
        description=(
            "Fresh validation, benchmarking, scaling, and profiling suite for sparqy."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate_parser = subparsers.add_parser(
        "generate",
        help="Generate matched sparqy/SLiM artifacts for a preset without running them.",
    )
    generate_parser.add_argument(
        "--preset",
        default="full",
        choices=sorted(suite_scenarios.PRESETS),
        help="Scenario preset to materialize.",
    )
    generate_parser.add_argument(
        "--results-dir",
        default=str(default_results_dir),
        help="Directory to place generated artifacts and manifests in.",
    )
    generate_parser.set_defaults(func=command_generate)

    run_parser = subparsers.add_parser(
        "run",
        help="Run one or more validation/benchmark/profile phases and summarize the results.",
    )
    run_parser.add_argument(
        "--preset",
        default="full",
        choices=sorted(suite_scenarios.PRESETS),
        help="Scenario preset to run.",
    )
    run_parser.add_argument(
        "--results-dir",
        default=str(default_results_dir),
        help="Directory for generated inputs, raw logs, CSVs, and summaries.",
    )
    run_parser.add_argument(
        "--phases",
        default="accuracy,speed,scaling,profile",
        help="Comma-separated subset of phases to run.",
    )
    run_parser.add_argument(
        "--simulators",
        default="sparqy,slim",
        help="Comma-separated subset of simulators to run where applicable.",
    )
    run_parser.add_argument(
        "--sparqy-bin",
        default=str(DEFAULT_SPARQY_BIN),
        help="Path to the sparqy executable.",
    )
    run_parser.add_argument(
        "--slim-bin",
        default=str(DEFAULT_SLIM_BIN),
        help="Path to the SLiM executable.",
    )
    run_parser.add_argument(
        "--build",
        action="store_true",
        help="Build sparqy before running the suite.",
    )
    run_parser.set_defaults(func=command_run)

    summarize_parser = subparsers.add_parser(
        "summarize",
        help="Recompute summary CSVs from existing raw run CSVs.",
    )
    summarize_parser.add_argument(
        "results_dir",
        help="Existing results directory produced by the run command.",
    )
    summarize_parser.set_defaults(func=command_summarize)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
