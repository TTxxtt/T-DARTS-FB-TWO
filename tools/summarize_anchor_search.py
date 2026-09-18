#!/usr/bin/env python
"""Summarise the full two-phase anchored search and its per-seed retrains.

Each seed runs its own complete chain (Phase A operator search -> Phase B RF
search -> genotype -> from-scratch retrain); no cross-seed majority vote is
formed.  This tool reports, per seed:

* Phase A's operator choice per band with its probability margin/entropy;
* Phase B's decoded RF pair per band (sorted: the paths are interchangeable);
* the search's soft/hard validation accuracy and the soft-to-hard gap;
* the retrain read-out (validation accuracy / NLL / kappa / macro-F1).

Cross-seed counts of operators and RF pairs are shown for information only --
they are not combined into an architecture.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from run_layout import run_leaf, subject_id  # noqa: E402

BANDS = ("Low", "Mid", "High")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("run/outputs"))
    parser.add_argument("--dataset", default="bci42a")
    parser.add_argument("--search-arm", default="anchored")
    parser.add_argument("--retrain-arm", default="anchor_retrain")
    parser.add_argument("--subjects", nargs="+", default=["003"])
    parser.add_argument("--seeds", nargs="+", default=["20250901", "20250902", "20250903"])
    parser.add_argument("--json", type=Path, default=None)
    return parser.parse_args()


def _load(path: Path) -> dict | None:
    if not path.is_file():
        print(f"!! missing: {path}", file=sys.stderr)
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_seed(args: argparse.Namespace, subject: str, seed: str) -> dict | None:
    leaf = run_leaf("anchor_search", subject_id(subject), seed, None)
    base = args.output_root / args.search_arm / args.dataset / leaf
    final = _load(base / "final_summary.json")
    if final is None:
        return None
    phase_a = _load(base / "phase_a_result.json") or {}
    genes = final["genotype"]["genes"]
    pairs = {}
    for band in BANDS:
        pairs[band] = sorted(
            int(gene["candidate"].rsplit("_rf", 1)[1])
            for gene in genes
            if gene["band"] == band
        )
    ops = {band: values["selected"] for band, values in phase_a.get("operators", {}).items()}
    margins = {
        band: (values["margin"], values["normalized_entropy"])
        for band, values in phase_a.get("operators", {}).items()
    }
    return {
        "subject": subject_id(subject),
        "seed": seed,
        "operators": ops,
        "operator_margins": margins,
        "rf_pairs": pairs,
        "soft_acc": final["final_val_soft"]["acc"],
        "soft_nll": final["final_val_soft"]["nll"],
        "hard_acc": final["final_val_hard"]["acc"],
        # The final summary stores the two evaluations but not their gap;
        # metrics.jsonl stores the gap, so recompute it here from the same two
        # accuracies rather than reading a key that does not exist.
        "gap": final["final_val_soft"]["acc"] - final["final_val_hard"]["acc"],
        "no_duplicate_paths": final["no_duplicate_paths"],
    }


def load_retrain(args: argparse.Namespace, subject: str, seed: str) -> dict | None:
    leaf = run_leaf("train", subject_id(subject), seed, "anchor")
    payload = _load(args.output_root / args.retrain_arm / args.dataset / leaf / "final_summary.json")
    if payload is None:
        return None
    validation = payload.get("validation_best")
    if validation is None:
        return None
    return {
        "subject": subject_id(subject),
        "seed": seed,
        "val_acc": validation["acc"],
        "val_nll": validation["nll"],
        "kappa": validation["kappa"],
        "macro_f1": validation["macro_f1"],
        "parameters": payload["parameters"],
        "macs": payload["macs"],
        "duplicate_structure_bands": payload["duplicate_structure_bands"],
    }


def _mean_std(values: list[float]) -> tuple[float, float | None]:
    if len(values) == 1:
        return values[0], None
    return statistics.mean(values), statistics.stdev(values)


def _fmt(mean: float, std: float | None) -> str:
    return f"{mean:.4f}" if std is None else f"{mean:.4f} ± {std:.4f}"


def main() -> int:
    args = parse_args()
    seeds = [
        row
        for subject in args.subjects
        for seed in args.seeds
        if (row := load_seed(args, subject, seed)) is not None
    ]
    if not seeds:
        return 1

    print("per-seed complete chain (Phase A operators -> Phase B RF pairs -> search read-out)")
    header = f"{'subj':<6}{'seed':<11}{'PhaseA L/M/H':<28}{'RF pairs L/M/H':<30}{'softAcc':>9}{'hardAcc':>9}{'gap':>8}"
    print(header)
    print("-" * len(header))
    for row in seeds:
        ops = "/".join(row["operators"].get(band, "?") for band in BANDS)
        pairs = " ".join(str(row["rf_pairs"].get(band)) for band in BANDS)
        print(
            f"{row['subject']:<6}{row['seed']:<11}{ops:<28}{pairs:<30}"
            f"{row['soft_acc']:>9.4f}{row['hard_acc']:>9.4f}{row['gap']:>+8.4f}"
        )

    print("\nPhase A operator probability margin / normalized entropy per band")
    for row in seeds:
        cells = []
        for band in BANDS:
            margin = row["operator_margins"].get(band)
            cells.append("n/a" if margin is None else f"{band} {margin[0]:.3f}/{margin[1]:.3f}")
        print(f"  {row['seed']}  " + "  ".join(cells))

    print("\ncross-seed counts (information only; not combined into an architecture)")
    for band in BANDS:
        op_counter = Counter(row["operators"].get(band) for row in seeds)
        pair_counter = Counter(tuple(row["rf_pairs"].get(band, ())) for row in seeds)
        ops = ", ".join(f"{name}x{count}" for name, count in op_counter.most_common())
        pairs = ", ".join(f"{list(pair)}x{count}" for pair, count in pair_counter.most_common())
        print(f"  {band:<5} operators: {ops:<40} RF pairs: {pairs}")

    retrains = [
        row
        for subject in args.subjects
        for seed in args.seeds
        if (row := load_retrain(args, subject, seed)) is not None
    ]
    if retrains:
        print("\nper-seed from-scratch retrain (validation-best, Session 1 closed)")
        header = f"{'seed':<11}{'valAcc':>9}{'valNLL':>9}{'kappa':>8}{'macroF1':>9}{'params':>8}{'MACs':>12}{'dups':>5}"
        print(header)
        print("-" * len(header))
        for row in retrains:
            print(
                f"{row['seed']:<11}{row['val_acc']:>9.4f}{row['val_nll']:>9.4f}"
                f"{row['kappa']:>8.4f}{row['macro_f1']:>9.4f}{row['parameters']:>8}{row['macs']:>12}"
                f"{len(row['duplicate_structure_bands']):>5}"
            )
        for key, label in (("val_acc", "valAcc"), ("val_nll", "valNLL"), ("kappa", "kappa"), ("macro_f1", "macroF1")):
            mean, std = _mean_std([row[key] for row in retrains])
            print(f"  mean ± std {label:<8}: {_fmt(mean, std)}")
    else:
        print("\nno retrain results found (run submit_anchor_retrain.sh once the searches finish)")

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps({"searches": seeds, "retrains": retrains}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
