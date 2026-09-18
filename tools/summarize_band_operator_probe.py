#!/usr/bin/env python
"""Summarise the fixed-RF57 single-band operator specificity probe (3 seeds).

For each band, one operator at a time replaces path 1 while every other path
stays on ``dilated_rf57``; all cells share RF 57, split, budget and screening
protocol, and the same seed is paired across cells.  Positive ``dNLL`` means
the replacement is worse than the all-dilated baseline.

The interaction question is answered per seed: does the per-band argmin operator
differ across bands, and does each band keep its argmin across all seeds?  With
three seeds this is a screening-level verdict, not a significance test; the
per-cell mean ± std is reported so a difference smaller than seed noise is
visible as such.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from run_layout import run_leaf, subject_id  # noqa: E402

BANDS = ("Low", "Mid", "High")
OPERATORS = ("dilated", "normal", "dwsep", "lkdw")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("run/outputs"))
    parser.add_argument("--dataset", default="bci42a")
    parser.add_argument("--arm", default="band_op", help="root of the band-wise probe runs")
    parser.add_argument("--reference-arm", default="opsep_rf57", help="all-bands-operator pilot root, or empty to skip")
    parser.add_argument("--subject", default="003")
    parser.add_argument("--seeds", nargs="+", default=["20250901", "20250902", "20250903"])
    parser.add_argument("--json", type=Path, default=None)
    return parser.parse_args()


def _load(path: Path) -> dict | None:
    if not path.is_file():
        print(f"!! missing: {path}", file=sys.stderr)
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    validation = payload.get("validation_best")
    if validation is None:
        return None
    return {
        "nll": validation["nll"],
        "acc": validation["acc"],
        "kappa": validation["kappa"],
        "macro_f1": validation["macro_f1"],
        "parameters": payload["parameters"],
        "macs": payload["macs"],
    }


def load_arm(args: argparse.Namespace, arm: str, seed: str, root_dir: Path | None = None) -> dict | None:
    root_dir = root_dir if root_dir is not None else args.output_root / args.arm
    leaf = run_leaf("train", subject_id(args.subject), seed, arm)
    return _load(root_dir / args.dataset / leaf / "final_summary.json")


def _mean_std(values: list[float]) -> str:
    if not values:
        return "n/a"
    if len(values) == 1:
        return f"{values[0]:.4f}"
    return f"{statistics.mean(values):.4f} ± {statistics.stdev(values):.4f}"


def main() -> int:
    args = parse_args()
    baseline = {seed: load_arm(args, "baseline_dd", seed) for seed in args.seeds}
    # A seed whose baseline is still running contributes NLL rows but no paired
    # deltas; do not abort the whole summary for it.
    missing_baselines = [seed for seed, row in baseline.items() if row is None]
    if missing_baselines:
        print(f"!! baseline_dd not finished for seeds {missing_baselines}; deltas skipped for them", file=sys.stderr)
    baseline_nll = {seed: row["nll"] for seed, row in baseline.items() if row is not None}

    cells: dict[tuple[str, str], dict[str, dict]] = {}
    for band in BANDS:
        for operator in OPERATORS:
            for seed in args.seeds:
                arm = "baseline_dd" if operator == "dilated" else f"{band.lower()}_{operator}"
                row = load_arm(args, arm, seed)
                if row is None:
                    continue
                cells.setdefault((band, operator), {})[seed] = row

    print(f"fixed-RF57 single-band operator probe, subject {subject_id(args.subject)}, seeds {args.seeds}")
    print("dNLL per seed; negative = that band's operator replacement beats the all-dilated baseline")
    header = f"{'band':<6}{'operator':<10}" + "".join(f"{'dNLL@' + seed[-4:]:>12}" for seed in args.seeds) + f"{'dNLL mean±std':>22}{'NLL mean±std':>22}"
    print(header)
    print("-" * len(header))
    for band in BANDS:
        for operator in OPERATORS:
            per_seed = cells.get((band, operator), {})
            deltas = [
                per_seed[seed]["nll"] - baseline_nll[seed]
                for seed in args.seeds
                if seed in per_seed and seed in baseline_nll
            ]
            nlls = [per_seed[seed]["nll"] for seed in args.seeds if seed in per_seed]
            cells_text = "".join(
                f"{per_seed[seed]['nll'] - baseline_nll[seed]:>+12.4f}"
                if seed in per_seed and seed in baseline_nll
                else f"{'--':>12}"
                for seed in args.seeds
            )
            print(f"{band:<6}{operator:<10}{cells_text}{_mean_std(deltas):>22}{_mean_std(nlls):>22}")

    print("\nper-band ranking by mean NLL (lower is better)")
    best_by_band: dict[str, str] = {}
    for band in BANDS:
        means = {
            operator: statistics.mean(row["nll"] for row in cells.get((band, operator), {}).values())
            for operator in OPERATORS
            if cells.get((band, operator))
        }
        ranked = sorted(means, key=means.get)
        best_by_band[band] = ranked[0]
        # Seed-level argmin consistency for the winning operator.
        wins = sum(
            1
            for seed in args.seeds
            if all(cells.get((band, op), {}).get(seed) for op in OPERATORS)
            and min(
                OPERATORS,
                key=lambda op: cells[(band, op)][seed]["nll"],
            )
            == ranked[0]
        )
        print(
            f"  {band:<5} best {ranked[0]:<8} mean {means[ranked[0]]:.4f}   "
            + " < ".join(f"{op}({means[op]:.4f})" for op in ranked)
            + f"   seed wins {wins}/{len(args.seeds)}"
        )

    if len(set(best_by_band.values())) == 1:
        print(
            f"\nverdict: every band's best operator is '{next(iter(set(best_by_band.values())))}' "
            "-- no band x operator interaction at this seed budget"
        )
    else:
        print(
            "\nverdict: bands disagree on the best operator "
            f"({', '.join(f'{band}={op}' for band, op in best_by_band.items())}); "
            "if the per-band winners also win per seed, this is a candidate interaction"
        )

    reference_root = args.output_root / args.reference_arm if args.reference_arm else None
    if reference_root is not None and reference_root.is_dir():
        print("\nreference: all-bands operator pilot (path1 = one operator in every band)")
        print(f"{'operator':<10}{'NLL mean±std':>22}{'seeds':>7}")
        for operator in OPERATORS:
            nlls = []
            for seed in args.seeds:
                row = load_arm(args, operator, seed, root_dir=reference_root)
                if row is not None:
                    nlls.append(row["nll"])
            print(f"{operator:<10}{_mean_std(nlls):>22}{len(nlls):>7}")

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(
                {
                    "baseline_nll": baseline_nll,
                    "cells": {
                        f"{band}/{op}": {seed: row["nll"] for seed, row in per_seed.items()}
                        for (band, op), per_seed in cells.items()
                    },
                    "best_by_band": best_by_band,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
