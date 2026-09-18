#!/usr/bin/env python
"""Paired comparison: full operator->RF chain vs the RF-only ablation.

Both arms are searched and retrained with the same seed, split, epoch budgets
and screening protocol, so the same-numeric seed gives one paired observation:

    Delta NLL = NLL(operator->RF chain) - NLL(RF-only chain)

Positive means the operator search HURT.  The tool reports per-subject paired
means, the overall mean across subjects, and the same paired deltas for
accuracy / kappa / macro-F1 plus the parameter/MAC cost, all from
``final_summary.json`` files.
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("run/outputs"))
    parser.add_argument("--dataset", default="bci42a")
    # Defaults describe the frozen hierarchical (full operator->RF chain) vs the
    # RF-only control, over all nine subjects.  These used to point at the
    # anchored arm with only three subjects, which silently overwrote the pooled
    # JSON with a partial, wrong comparison -- so the defaults must match the
    # run_remaining_subjects.sh invocation to be safe with no flags.
    parser.add_argument("--subjects", nargs="+", default=["001", "002", "003", "004", "005", "006", "007", "008", "009"])
    parser.add_argument("--seeds", nargs="+", default=["20250901", "20250902", "20250903"])
    parser.add_argument("--anchor-search-arm", default="hier")
    parser.add_argument("--anchor-search-phase", default="hier_search", help="run phase string of the full-method search")
    parser.add_argument("--anchor-retrain-arm", default="hier_retrain")
    parser.add_argument("--anchor-leaf-arm", default="hier", help="arm suffix of the full-method retrain leaf")
    parser.add_argument("--rf-search-arm", default="rfsearch")
    parser.add_argument("--rf-search-phase", default="search")
    parser.add_argument("--rf-retrain-arm", default="rf_retrain_nll")
    parser.add_argument("--rf-leaf-arm", default="rf")
    parser.add_argument("--json", type=Path, default=None)
    return parser.parse_args()


def _load(path: Path) -> dict | None:
    if not path.is_file():
        print(f"!! missing: {path}", file=sys.stderr)
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_retrain(root: Path, dataset: str, subject: str, seed: str, arm: str) -> dict | None:
    leaf = run_leaf("train", subject_id(subject), seed, arm)
    payload = _load(root / dataset / leaf / "final_summary.json")
    if payload is None:
        return None
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
        "genotype": [gene["candidate"] for gene in payload["genotype"]["genes"]],
    }


def load_search_gap(root: Path, dataset: str, subject: str, seed: str, phase: str, arm: str | None) -> dict | None:
    leaf = run_leaf(phase, subject_id(subject), seed, arm)
    payload = _load(root / dataset / leaf / "final_summary.json")
    if payload is None:
        return None
    # Both searches store the two final evaluations under the same keys.
    soft, hard = payload["final_val_soft"]["acc"], payload["final_val_hard"]["acc"]
    return {"soft_acc": soft, "hard_acc": hard, "gap": soft - hard}


def _stats(values: list[float]) -> str:
    if not values:
        return "n/a"
    if len(values) == 1:
        return f"{values[0]:+.4f}"
    return f"{statistics.mean(values):+.4f} ± {statistics.stdev(values):.4f}"


def main() -> int:
    args = parse_args()
    rows = []
    for subject in args.subjects:
        for seed in args.seeds:
            anchor = load_retrain(args.output_root / args.anchor_retrain_arm, args.dataset, subject, seed, args.anchor_leaf_arm)
            rf = load_retrain(args.output_root / args.rf_retrain_arm, args.dataset, subject, seed, args.rf_leaf_arm)
            if anchor is None or rf is None:
                continue
            anchor_gap = load_search_gap(args.output_root / args.anchor_search_arm, args.dataset, subject, seed, args.anchor_search_phase, None)
            rf_gap = load_search_gap(args.output_root / args.rf_search_arm, args.dataset, subject, seed, args.rf_search_phase, None)
            rows.append(
                {
                    "subject": subject_id(subject),
                    "seed": seed,
                    "delta_nll": anchor["nll"] - rf["nll"],
                    "delta_acc": anchor["acc"] - rf["acc"],
                    "delta_kappa": anchor["kappa"] - rf["kappa"],
                    "delta_macro_f1": anchor["macro_f1"] - rf["macro_f1"],
                    "delta_macs": anchor["macs"] - rf["macs"],
                    "anchor": anchor,
                    "rf": rf,
                    "anchor_gap": None if anchor_gap is None else anchor_gap["gap"],
                    "rf_gap": None if rf_gap is None else rf_gap["gap"],
                }
            )
    if not rows:
        return 1

    header = (
        f"{'subj':<6}{'seed':<11}{'NLL(op->RF)':>12}{'NLL(RF-only)':>13}{'dNLL':>9}"
        f"{'dAcc':>9}{'dKappa':>9}{'dMacroF1':>10}{'dMACs':>12}"
    )
    print("paired chains (same seed, split, budgets, screening protocol)")
    print(header)
    print("-" * len(header))
    for row in sorted(rows, key=lambda r: (r["subject"], r["seed"])):
        print(
            f"{row['subject']:<6}{row['seed']:<11}{row['anchor']['nll']:>12.4f}{row['rf']['nll']:>13.4f}"
            f"{row['delta_nll']:>+9.4f}{row['delta_acc']:>+9.4f}{row['delta_kappa']:>+9.4f}"
            f"{row['delta_macro_f1']:>+10.4f}{row['delta_macs']:>+12}"
        )

    print("\npaired Delta NLL per subject (positive = operator search hurts)")
    subject_means = {}
    for subject in args.subjects:
        subject = subject_id(subject)
        deltas = [row["delta_nll"] for row in rows if row["subject"] == subject]
        if deltas:
            subject_means[subject] = statistics.mean(deltas)
            print(f"  {subject}: {_stats(deltas)}  (n={len(deltas)})")
    all_deltas = [row["delta_nll"] for row in rows]
    print(f"  ALL    : {_stats(all_deltas)}  (n={len(all_deltas)}, {len(subject_means)} subjects)")

    print("\nsecondary paired deltas, all subjects (positive = operator search better)")
    for key, label in (("delta_acc", "valAcc"), ("delta_kappa", "kappa"), ("delta_macro_f1", "macroF1")):
        print(f"  {label:<8} {_stats([row[key] for row in rows])}")
    print(f"  {'MACs':<8} {_stats([float(row['delta_macs']) for row in rows])}")

    gaps = [(row["anchor_gap"], row["rf_gap"]) for row in rows if row["anchor_gap"] is not None and row["rf_gap"] is not None]
    if gaps:
        print("\nsearch-side soft-hard gap reduction (anchor gap - RF-only gap)")
        print(f"  anchor gap mean {statistics.mean(g[0] for g in gaps):.4f}  RF-only gap mean {statistics.mean(g[1] for g in gaps):.4f}")
        print(f"  paired reduction {_stats([g[0] - g[1] for g in gaps])}")

    mean_delta = statistics.mean(all_deltas)
    if mean_delta > 0:
        print(
            "\nverdict: on these subjects the operator->RF chain is worse on average -- "
            "RF-only should be the main method and operator search a negative result/ablation."
        )
    else:
        print(
            "\nverdict: the operator->RF chain is not worse on average -- operator choice may be "
            "subject-specific; extend to the remaining subjects before deciding."
        )

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(
                {
                    "rows": [{k: v for k, v in row.items() if k not in ("anchor", "rf")} for row in rows],
                    "subject_mean_delta_nll": subject_means,
                    "overall_mean_delta_nll": mean_delta,
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
