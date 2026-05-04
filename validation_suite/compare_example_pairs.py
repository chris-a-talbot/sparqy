#!/usr/bin/env python3
"""Compare checked-in sparqy example outputs against paired SLiM example outputs."""

from __future__ import annotations

import argparse
import math
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


HEADER = [
    "generation",
    "dt_sec",
    "cumul_sec",
    "statistic",
    "metric",
    "scalar_value",
    "by_type",
    "by_chromosome",
    "unfolded_sfs",
    "folded_sfs",
]

VECTOR_FIELDS = ("by_type", "by_chromosome", "unfolded_sfs", "folded_sfs")


def build_parser() -> argparse.ArgumentParser:
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sparqy-bin",
        default=str(repo_root / "build" / "sparqy"),
        help="Path to the sparqy executable.",
    )
    parser.add_argument(
        "--slim-bin",
        default="slim",
        help="Path to the SLiM executable.",
    )
    parser.add_argument(
        "--examples-dir",
        default=str(repo_root / "examples"),
        help="Directory containing the .sparqy examples.",
    )
    return parser


def parse_seed(config_text: str) -> int:
    import re

    patterns = [
        r"\brun_seed\s*=\s*(\d+)",
        r"^\s*seed\s*=\s*(\d+)\s*$",
    ]
    for pattern in patterns:
        match = re.search(pattern, config_text, re.MULTILINE)
        if match:
            return int(match.group(1))
    raise ValueError("could not find a numeric seed in the config example")


def run_capture(command: List[str]) -> str:
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout


def parse_rows(text: str) -> Dict[Tuple[int, str, str], Dict[str, object]]:
    rows: Dict[Tuple[int, str, str], Dict[str, object]] = {}
    for raw_line in text.splitlines():
        if not raw_line or raw_line.startswith("//"):
            continue
        if raw_line.startswith("generation\t"):
            continue
        if not raw_line[0].isdigit():
            continue

        fields = raw_line.split("\t")
        if len(fields) != len(HEADER):
            raise ValueError(f"unexpected output row shape: {raw_line!r}")

        row = dict(zip(HEADER, fields))
        key = (int(row["generation"]), row["statistic"], row["metric"])
        parsed: Dict[str, object] = {
            "scalar_value": None,
            "by_type": None,
            "by_chromosome": None,
            "unfolded_sfs": None,
            "folded_sfs": None,
        }
        if row["scalar_value"]:
            parsed["scalar_value"] = float(row["scalar_value"])
        for field in VECTOR_FIELDS:
            value = row[field]
            if value:
                parsed[field] = [int(token) for token in value.split(",") if token]
        rows[key] = parsed
    return rows


def mean_or_nan(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        return math.nan
    return statistics.mean(values)


def compare_rows(
    sparqy_rows: Dict[Tuple[int, str, str], Dict[str, object]],
    slim_rows: Dict[Tuple[int, str, str], Dict[str, object]],
) -> Dict[str, object]:
    scalar_diffs: List[Tuple[Tuple[int, str, str], float, float]] = []
    vector_diffs: List[Tuple[Tuple[int, str, str], str, int, float]] = []
    missing_in_slim: List[Tuple[int, str, str]] = []
    missing_in_sparqy: List[Tuple[int, str, str]] = []

    for key in sorted(set(sparqy_rows) | set(slim_rows)):
        sparqy = sparqy_rows.get(key)
        slim = slim_rows.get(key)
        if sparqy is None:
            missing_in_sparqy.append(key)
            continue
        if slim is None:
            missing_in_slim.append(key)
            continue

        sparqy_scalar = sparqy["scalar_value"]
        slim_scalar = slim["scalar_value"]
        if sparqy_scalar is not None or slim_scalar is not None:
            if sparqy_scalar is None or slim_scalar is None:
                missing_in_slim.append(key)
                continue
            abs_diff = abs(float(sparqy_scalar) - float(slim_scalar))
            rel_diff = abs_diff / max(
                abs(float(sparqy_scalar)),
                abs(float(slim_scalar)),
                1.0,
            )
            scalar_diffs.append((key, abs_diff, rel_diff))

        for field in VECTOR_FIELDS:
            sparqy_vec = sparqy[field]
            slim_vec = slim[field]
            if sparqy_vec is None and slim_vec is None:
                continue
            if sparqy_vec is None or slim_vec is None:
                missing_in_slim.append(key)
                continue
            sparqy_list = list(sparqy_vec)  # type: ignore[arg-type]
            slim_list = list(slim_vec)  # type: ignore[arg-type]
            if len(sparqy_list) != len(slim_list):
                raise ValueError(f"vector length mismatch for {key} field {field}")
            l1 = sum(abs(a - b) for a, b in zip(sparqy_list, slim_list))
            denom = max(
                sum(abs(a) for a in sparqy_list),
                sum(abs(b) for b in slim_list),
                1,
            )
            vector_diffs.append((key, field, l1, l1 / denom))

    worst_scalar = max(scalar_diffs, key=lambda entry: entry[2], default=None)
    worst_vector = max(vector_diffs, key=lambda entry: entry[3], default=None)
    return {
        "scalar_count": len(scalar_diffs),
        "vector_count": len(vector_diffs),
        "scalar_mean_rel_diff": mean_or_nan(entry[2] for entry in scalar_diffs),
        "scalar_max_rel_diff": worst_scalar[2] if worst_scalar else math.nan,
        "scalar_worst_key": worst_scalar[0] if worst_scalar else None,
        "vector_mean_norm_l1": mean_or_nan(entry[3] for entry in vector_diffs),
        "vector_max_norm_l1": worst_vector[3] if worst_vector else math.nan,
        "vector_worst_key": (worst_vector[0], worst_vector[1]) if worst_vector else None,
        "missing_in_slim": missing_in_slim,
        "missing_in_sparqy": missing_in_sparqy,
    }


def format_key(key: Tuple[int, str, str] | None) -> str:
    if key is None:
        return "-"
    generation, statistic, metric = key
    if metric:
        return f"g{generation}:{statistic}[{metric}]"
    return f"g{generation}:{statistic}"


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    examples_dir = Path(args.examples_dir)
    slim_dir = examples_dir / "slim"
    example_paths = sorted(examples_dir.glob("*.sparqy"))
    if not example_paths:
        raise SystemExit(f"no .sparqy examples found in {examples_dir}")

    print("# Example Pair Comparison")
    print()
    print(
        "These comparisons use each checked-in example's own seed. "
        "SLiM and sparqy are stochastic simulators with different RNGs and implementations, "
        "so the goal here is structural agreement and similar magnitudes, not row-for-row identity."
    )
    print()

    for config_path in example_paths:
        base = config_path.stem
        slim_path = slim_dir / f"{base}.slim"
        if not slim_path.exists():
            print(f"- `{base}`: missing paired SLiM script at `{slim_path}`")
            continue

        seed = parse_seed(config_path.read_text())
        sparqy_stdout = run_capture([args.sparqy_bin, "--config", str(config_path)])
        slim_stdout = run_capture([args.slim_bin, "-l", "0", "-s", str(seed), str(slim_path)])

        sparqy_rows = parse_rows(sparqy_stdout)
        slim_rows = parse_rows(slim_stdout)
        summary = compare_rows(sparqy_rows, slim_rows)

        print(f"## {base}")
        print()
        print(f"- Scalar rows compared: {summary['scalar_count']}")
        print(f"- Vector rows compared: {summary['vector_count']}")
        print(
            f"- Mean scalar relative diff: {summary['scalar_mean_rel_diff']:.4f}"
            if not math.isnan(summary["scalar_mean_rel_diff"])
            else "- Mean scalar relative diff: -"
        )
        print(
            f"- Worst scalar relative diff: {summary['scalar_max_rel_diff']:.4f} at "
            f"{format_key(summary['scalar_worst_key'])}"
            if not math.isnan(summary["scalar_max_rel_diff"])
            else "- Worst scalar relative diff: -"
        )
        print(
            f"- Mean vector normalized L1: {summary['vector_mean_norm_l1']:.4f}"
            if not math.isnan(summary["vector_mean_norm_l1"])
            else "- Mean vector normalized L1: -"
        )
        if not math.isnan(summary["vector_max_norm_l1"]):
            worst_key, worst_field = summary["vector_worst_key"]
            print(
                f"- Worst vector normalized L1: {summary['vector_max_norm_l1']:.4f} at "
                f"{format_key(worst_key)}::{worst_field}"
            )
        else:
            print("- Worst vector normalized L1: -")
        if summary["missing_in_slim"]:
            print("- Missing in SLiM:")
            for key in summary["missing_in_slim"]:
                print(f"  - {format_key(key)}")
        if summary["missing_in_sparqy"]:
            print("- Missing in sparqy:")
            for key in summary["missing_in_sparqy"]:
                print(f"  - {format_key(key)}")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
