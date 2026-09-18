#!/usr/bin/env python
"""Summarise the pre-registered random RF baseline against the majority vote.

Round 1 (no ``--arms``): twelve sampled no-duplicate RF-only genotypes and the
majority-vote genotype, one training seed, identical protocol.  The question:
does the majority vote sit near the top of the random distribution or in the
middle?  Primary metric is validation NLL; accuracy/kappa/macro-F1 are
secondary; all genotypes in this space share one parameter/MAC count, which the
tool asserts rather than assumes.

Round 2 (``--arms`` given): re-run the chosen arms (majority plus the random
top-3) under the remaining seeds and report per-seed values, mean ± std and the
same-seed paired difference against the reference arm, so a round-1 lead is only
believed if it survives seed variation.
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

SECONDARY = (("acc", "valAcc"), ("kappa", "kappa"), ("macro_f1", "macroF1"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("run/outputs"))
    parser.add_argument("--arm", default="rf_random")
    parser.add_argument("--dataset", default="bci42a")
    parser.add_argument("--subject", default="003")
    parser.add_argument("--seeds", nargs="+", default=["20250901"], help="training seeds to read")
    parser.add_argument("--count", type=int, default=12, help="round-1 random sample size")
    parser.add_argument("--arms", nargs="+", default=None, help="round-2 stability mode: arms to compare across seeds; the first is the reference")
    parser.add_argument("--genotype-dir", type=Path, default=None, help="optional: annotate rows with their RF pairs")
    parser.add_argument("--json", type=Path, default=None)
    return parser.parse_args()


def load_row(args: argparse.Namespace, arm: str, seed: str) -> dict | None:
    leaf = run_leaf("train", subject_id(args.subject), seed, arm)
    path = args.output_root / args.arm / args.dataset / leaf / "final_summary.json"
    if not path.is_file():
        print(f"!! missing: {path}", file=sys.stderr)
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    validation = payload.get("validation_best")
    if validation is None:
        print(f"!! {path} has no validation_best (was it run with --screening-only?)", file=sys.stderr)
        return None
    return {
        "arm": arm,
        "seed": seed,
        "nll": validation["nll"],
        "acc": validation["acc"],
        "kappa": validation["kappa"],
        "macro_f1": validation["macro_f1"],
        "parameters": payload["parameters"],
        "macs": payload["macs"],
    }


def pairs_for(genotype_dir: Path | None, arm: str) -> str:
    if genotype_dir is None:
        return ""
    path = genotype_dir / f"{arm}.json"
    if not path.is_file():
        return ""
    payload = json.loads(path.read_text(encoding="utf-8"))
    by_band: dict[str, list[int]] = {}
    for gene in payload["genes"]:
        by_band.setdefault(gene["band"], []).append(int(gene["candidate"].rsplit("_rf", 1)[1]))
    return " ".join(f"{band}:{sorted(by_band[band])}" for band in ("Low", "Mid", "High"))


def _mean_std(values: list[float]) -> str:
    if len(values) == 1:
        return f"{values[0]:.4f}"
    return f"{statistics.mean(values):.4f} ± {statistics.stdev(values):.4f}"


def round1(args: argparse.Namespace) -> int:
    seed = args.seeds[0]
    random_rows = [row for gid in range(args.count) if (row := load_row(args, f"g{gid:03d}", seed)) is not None]
    majority = load_row(args, "majority", seed)
    if not random_rows or majority is None:
        return 1

    all_rows = random_rows + [majority]
    params = {row["parameters"] for row in all_rows}
    macs = {row["macs"] for row in all_rows}
    if len(params) != 1 or len(macs) != 1:
        print(f"!! baselines are not cost-identical: params={params} macs={macs}", file=sys.stderr)

    print(f"round 1: per-genotype read-out (training seed {seed}, 50 epochs, screening-only)")
    header = f"{'arm':<10}{'valNLL':>9}{'valAcc':>9}{'kappa':>8}{'macroF1':>9}{'params':>8}{'MACs':>12}  pairs"
    print(header)
    print("-" * len(header))
    for row in sorted(all_rows, key=lambda r: r["nll"]):
        print(
            f"{row['arm']:<10}{row['nll']:>9.4f}{row['acc']:>9.4f}{row['kappa']:>8.4f}"
            f"{row['macro_f1']:>9.4f}{row['parameters']:>8}{row['macs']:>12}  {pairs_for(args.genotype_dir, row['arm'])}"
        )

    nlls = [row["nll"] for row in random_rows]
    print(f"\nrandom baseline distribution (N={len(nlls)}), primary metric valNLL")
    print(f"  mean {statistics.mean(nlls):.4f}  std {statistics.stdev(nlls):.4f}  min {min(nlls):.4f}  max {max(nlls):.4f}")
    worse = sum(1 for value in nlls if value >= majority["nll"])
    better = sum(1 for value in nlls if value < majority["nll"])
    percentile = 100.0 * worse / len(nlls)
    print(
        f"\nmajority vote  valNLL {majority['nll']:.4f}  "
        f"beats {better}/{len(nlls)} random genotypes  ->  percentile {percentile:.1f}"
    )
    if better == 0:
        print("  verdict: majority is at least as good as every sampled random genotype")
    elif percentile >= 75:
        print("  verdict: majority sits in the top quartile of the random distribution")
    else:
        print("  verdict: majority does NOT stand out from the random distribution")

    print("\nsecondary metrics (random mean ± std vs majority)")
    for key, label in SECONDARY:
        values = [row[key] for row in random_rows]
        print(
            f"  {label:<8} random {statistics.mean(values):.4f} ± {statistics.stdev(values):.4f}   "
            f"majority {majority[key]:.4f}"
        )

    top3 = sorted(random_rows, key=lambda r: r["nll"])[:3]
    print("\nrandom top-3 by NLL (round-2 candidates)")
    for row in top3:
        print(f"  {row['arm']}  valNLL {row['nll']:.4f}  {pairs_for(args.genotype_dir, row['arm'])}")

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(
                {
                    "random": sorted(random_rows, key=lambda r: r["arm"]),
                    "majority": majority,
                    "random_nll_mean": statistics.mean(nlls),
                    "random_nll_std": statistics.stdev(nlls),
                    "majority_percentile": percentile,
                    "round2_candidates": [row["arm"] for row in top3],
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"\nwrote {args.json}")
    return 0


def round2(args: argparse.Namespace) -> int:
    reference = args.arms[0]
    rows: dict[str, dict[str, dict]] = {}
    for arm in args.arms:
        rows[arm] = {}
        for seed in args.seeds:
            row = load_row(args, arm, seed)
            if row is not None:
                rows[arm][seed] = row

    print(f"round 2: stability across seeds {args.seeds} (reference arm: {reference})")
    header = f"{'arm':<10}" + "".join(f"{'s' + seed[-4:]:>11}" for seed in args.seeds) + f"{'mean ± std':>20}"
    print(header)
    print("-" * len(header))
    for arm in args.arms:
        values = [rows[arm][seed]["nll"] for seed in args.seeds if seed in rows[arm]]
        cells = "".join(
            f"{rows[arm][seed]['nll']:>11.4f}" if seed in rows[arm] else f"{'--':>11}" for seed in args.seeds
        )
        print(f"{arm:<10}{cells}{_mean_std(values):>20}")

    if reference in rows and len(args.seeds) > 1:
        print(f"\npaired same-seed ΔNLL vs {reference} (negative = the arm has lower/better NLL)")
        for arm in args.arms[1:]:
            deltas = [
                rows[arm][seed]["nll"] - rows[reference][seed]["nll"]
                for seed in args.seeds
                if seed in rows[arm] and seed in rows[reference]
            ]
            if deltas:
                detail = ", ".join(f"{seed[-4:]}:{delta:+.4f}" for seed, delta in zip(args.seeds, deltas))
                print(f"  {arm:<10} mean {statistics.mean(deltas):+.4f}  std {statistics.stdev(deltas):.4f}  [{detail}]")

    print("\nsecondary metrics, mean ± std over seeds")
    for arm in args.arms:
        present = [rows[arm][seed] for seed in args.seeds if seed in rows[arm]]
        if not present:
            continue
        cells = "  ".join(
            f"{label} {_mean_std([row[key] for row in present])}" for key, label in SECONDARY
        )
        print(f"  {arm:<10} {cells}")

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(
                {arm: {seed: row for seed, row in rows[arm].items()} for arm in args.arms},
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"\nwrote {args.json}")
    return 0


def main() -> int:
    args = parse_args()
    if args.arms:
        return round2(args)
    return round1(args)


if __name__ == "__main__":
    raise SystemExit(main())
