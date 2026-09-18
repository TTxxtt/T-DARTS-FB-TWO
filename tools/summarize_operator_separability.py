#!/usr/bin/env python
"""Summarise the fixed-genotype operator-separability runs.

Reads each arm's ``final_summary.json`` (written by
``train_retrain.py --screening-only``) and applies the pre-registered analysis:

* primary metric: validation NLL (the search optimises NLL, and it is more
  sensitive than a 57-trial accuracy);
* secondary metrics: kappa, macro-F1, validation accuracy;
* per operator: mean and standard deviation over seeds;
* same-seed paired differences between operators, so seed-to-seed variation
  cancels instead of being read as an operator effect.

The verdict is deliberately conservative -- with three seeds it can only ever
say "a leader is consistent with the data", never "the operators differ
significantly".  Operator A is called a stable leader on NLL only if it is
rank 1 in *every* run and its mean paired advantage over every other operator
exceeds the seed-to-seed standard deviation of that paired difference.
"""

from __future__ import annotations

import argparse
import itertools
import json
import statistics
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from run_layout import run_leaf, subject_id  # noqa: E402

#: Primary first; all four are reported for every operator.
METRICS = (
    ("val_nll", "valNLL"),
    ("kappa", "kappa"),
    ("macro_f1", "macroF1"),
    ("val_acc", "valAcc"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("run/outputs/opsep_rf57"))
    parser.add_argument("--dataset", default="bci42a")
    parser.add_argument("--subjects", nargs="+", default=["003"])
    parser.add_argument("--seeds", nargs="+", default=["20250901", "20250902", "20250903"])
    parser.add_argument("--ops", nargs="+", default=["dilated", "normal", "dwsep", "lkdw"])
    parser.add_argument("--json", type=Path, default=None, help="also write the raw rows and stats here")
    return parser.parse_args()


def load_row(root: Path, dataset: str, subject: str, seed: str, op: str) -> dict | None:
    leaf = run_leaf("train", subject_id(subject), seed, op)
    path = root / dataset / leaf / "final_summary.json"
    if not path.is_file():
        print(f"!! missing: {path}", file=sys.stderr)
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    validation = payload.get("validation_best")
    if validation is None:
        print(f"!! {path} has no validation_best; was it run with --screening-only?", file=sys.stderr)
        return None
    return {
        "subject": subject_id(subject),
        "seed": seed,
        "op": op,
        "best_epoch": payload["stage1"]["best_epoch"],
        "val_acc": validation["acc"],
        "val_nll": validation["nll"],
        "kappa": validation["kappa"],
        "macro_f1": validation["macro_f1"],
        "parameters": payload["parameters"],
        "macs": payload["macs"],
    }


def _mean_std(values: list[float]) -> tuple[float, float | None]:
    if len(values) == 1:
        return values[0], None
    return statistics.mean(values), statistics.stdev(values)


def _fmt(mean: float, std: float | None) -> str:
    return f"{mean:.4f}" if std is None else f"{mean:.4f} ± {std:.4f}"


def _pick(rows: list[dict], subject: str, seed: str, op: str) -> dict | None:
    return next(
        (row for row in rows if row["subject"] == subject and row["seed"] == seed and row["op"] == op),
        None,
    )


def _delta(paired: dict[tuple[str, str], dict], left: str, right: str) -> dict | None:
    """Paired ``left - right`` stats, flipping whichever direction was stored."""

    if (left, right) in paired:
        return paired[(left, right)]
    if (right, left) in paired:
        stored = paired[(right, left)]
        return {
            "n": stored["n"],
            "mean": -stored["mean"],
            "std": stored["std"],
            "deltas": [-value for value in stored["deltas"]],
        }
    return None


def main() -> int:
    args = parse_args()
    subjects = [subject_id(subject) for subject in args.subjects]
    rows = [
        row
        for subject in subjects
        for seed in args.seeds
        for op in args.ops
        if (row := load_row(args.output_root, args.dataset, subject, seed, op)) is not None
    ]
    if not rows:
        return 1

    # ---- per-run table ---------------------------------------------------
    header = f"{'op':<10}{'subj':<6}{'seed':<11}{'bestEp':>7}{'valAcc':>9}{'valNLL':>9}{'kappa':>8}{'macroF1':>9}{'params':>8}{'MACs':>12}"
    print(header)
    print("-" * len(header))
    for row in sorted(rows, key=lambda r: (r["op"], r["seed"])):
        print(
            f"{row['op']:<10}{row['subject']:<6}{row['seed']:<11}{row['best_epoch']:>7}"
            f"{row['val_acc']:>9.4f}{row['val_nll']:>9.4f}{row['kappa']:>8.4f}"
            f"{row['macro_f1']:>9.4f}{row['parameters']:>8}{row['macs']:>12}"
        )

    # ---- per-operator mean / std ----------------------------------------
    print("\nper-operator mean ± std over runs")
    print(f"{'op':<10}{'n':>3}  " + "  ".join(f"{label:>17}" for _, label in METRICS))
    stats: dict[str, dict] = {}
    for op in args.ops:
        group = [row for row in rows if row["op"] == op]
        if not group:
            continue
        stats[op] = {"n": len(group)}
        cells = []
        for key, _ in METRICS:
            mean, std = _mean_std([row[key] for row in group])
            stats[op][key] = {"mean": mean, "std": std}
            cells.append(f"{_fmt(mean, std):>17}")
        print(f"{op:<10}{len(group):>3}  " + "  ".join(cells))

    # ---- paired same-seed differences (all pairs, primary metric) -------
    print("\npaired same-seed differences on the primary metric (NLL, lower is better)")
    print(f"{'pair':<24}{'subj/seed':<18}{'delta':>9}   {'mean':>9}  {'std':>9}")
    paired: dict[tuple[str, str], dict] = {}
    for left, right in itertools.combinations(args.ops, 2):
        deltas = []
        for subject in subjects:
            for seed in args.seeds:
                a = _pick(rows, subject, seed, left)
                b = _pick(rows, subject, seed, right)
                if a is None or b is None:
                    continue
                deltas.append(a["val_nll"] - b["val_nll"])
                print(f"{left + ' - ' + right:<24}{subject + '/' + seed:<18}{deltas[-1]:>9.4f}")
        if deltas:
            mean, std = _mean_std(deltas)
            paired[(left, right)] = {"n": len(deltas), "mean": mean, "std": std, "deltas": deltas}
            print(f"{'':<24}{'':<18}{'':>9}   {mean:>9.4f}  {'n/a' if std is None else f'{std:>9.4f}'}")

    # ---- rank stability --------------------------------------------------
    print("\nrank-1 counts per metric (rank 1 = best; NLL: lowest, others: highest)")
    best_counts = {op: {} for op in args.ops}
    total_runs = 0
    for subject in subjects:
        for seed in args.seeds:
            if any(_pick(rows, subject, seed, op) is not None for op in args.ops):
                total_runs += 1
    for key, label in METRICS:
        counts = {op: 0 for op in args.ops}
        for subject in subjects:
            for seed in args.seeds:
                group = [
                    row
                    for op in args.ops
                    if (row := _pick(rows, subject, seed, op)) is not None
                ]
                if not group:
                    continue
                winner = (
                    min(group, key=lambda r: r["val_nll"])
                    if key == "val_nll"
                    else max(group, key=lambda r: r[key])
                )
                counts[winner["op"]] += 1
        for op in args.ops:
            best_counts[op][key] = counts[op]
        print(f"  {label:<8}" + "  ".join(f"{op}={counts[op]}/{total_runs}" for op in args.ops))

    # ---- verdict (conservative, pre-registered rule) --------------------
    leaders = []
    for op in args.ops:
        if best_counts.get(op, {}).get("val_nll", 0) < total_runs:
            continue
        advantages = []
        for other in args.ops:
            if other == op:
                continue
            diff = _delta(paired, op, other)
            if diff is not None:
                advantages.append(diff["mean"] < 0 and (diff["std"] is None or abs(diff["mean"]) > diff["std"]))
        if advantages and all(advantages):
            leaders.append(op)
    nll_means = {op: stats[op]["val_nll"]["mean"] for op in stats}
    if len(nll_means) > 1:
        spread = max(nll_means.values()) - min(nll_means.values())
        within = [stats[op]["val_nll"]["std"] for op in stats if stats[op]["val_nll"]["std"] is not None]
        print(f"\nbetween-operator spread of NLL means : {spread:.4f}")
        print(
            "mean within-operator NLL std       : "
            + (f"{statistics.mean(within):.4f}" if within else "n/a (one seed per arm)")
        )
    print(
        "verdict: "
        + (
            f"stable NLL leader {leaders[0]} (rank 1 in every run, paired advantage larger than seed noise)"
            if len(leaders) == 1
            else "no operator is a stable NLL leader -- ranking is not reliable at this seed budget"
        )
    )

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(
                {
                    "rows": sorted(rows, key=lambda r: (r["op"], r["seed"])),
                    "per_operator": stats,
                    "paired_nll": {f"{a} - {b}": value for (a, b), value in paired.items()},
                    "rank1_counts": best_counts,
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
