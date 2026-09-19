#!/usr/bin/env python
"""Paired Session-2 comparison: Arm A (dilated-only FBNAS) vs Arm B (four mechanisms).

The question this file exists to answer:

    Does searching over temporal mechanisms improve Session-1 -> Session-2
    generalisation, compared with searching only the receptive field of a
    dilated convolution?

Both arms are evaluated on the dataset's **second** recording session (code
``session=1``), by a strict per-subject pairing.  Each arm contributes exactly
one architecture per subject, frozen before that session was read:

* Arm A -- the vendored upstream FBNAS chain.  Its architecture comes from
  ``opt_choice.csv`` and its Session-2 number from ``results.csv``; both were
  produced by ``run/py/run_fbnas_subject.py`` and are read, never recomputed.
* Arm B -- ``train_arm_b_search.py`` freezes the architecture, then
  ``train_retrain.py`` runs the same two-stage FBNAS protocol and evaluates
  Session 2 once.

Every metric is a **paired** difference ``Arm B - Arm A`` at the same subject.
No seed is ever selected on: both arms run the single seed 20190821, because
Arm A's is hardcoded inside the frozen baseline.  The best subject is not
reported as a headline; the mean over all nine is.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

#: ``(label, Arm A key, Arm B key)``.  The two producers disagree on one name:
#: upstream's ``results.csv`` calls macro-F1 ``f1``, ``train_retrain.py`` calls
#: it ``macro_f1``.  Same quantity, so the mapping is explicit rather than a
#: rename that would silently drop a column.
METRICS = (
    ("acc", "acc", "acc"),
    ("f1", "f1", "macro_f1"),
    ("kappa", "kappa", "kappa"),
)
LABELS = tuple(label for label, _, _ in METRICS)

#: Arm A's Session-2 evaluation lives in the `test` row of results.csv.
ARM_A_DEFAULT = PROJECT_ROOT / "run/outputs/operator_armB_comparison/arm_a_session2.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm-a", type=Path, default=ARM_A_DEFAULT)
    parser.add_argument(
        "--arm-b-root", type=Path,
        default=PROJECT_ROOT / "run/outputs/operator_armB_retrain/bci42a",
    )
    parser.add_argument("--arm-b-arm", default="armB", help="run-leaf arm suffix for Arm B")
    parser.add_argument("--seed", default="20190821")
    parser.add_argument(
        "--json", type=Path,
        default=PROJECT_ROOT / "run/outputs/operator_armB_comparison/session2_paired.json",
    )
    return parser.parse_args()


def _load(path: Path) -> dict | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_arm_b(root: Path, arm: str, subject: str, seed: str) -> dict | None:
    leaf = f"train_s{subject}_seed{seed}_{arm}"
    payload = _load(root / leaf / "final_summary.json")
    if payload is None:
        return None
    test = payload.get("test")
    if not test:
        return None
    return {
        "leaf": leaf,
        "test": {label: float(test[b_key]) for label, _, b_key in METRICS},
        "nll": float(test.get("nll", float("nan"))),
        "genotype": payload.get("genotype"),
        "parameters": payload.get("parameters"),
        "macs": payload.get("macs"),
        "screening_only": payload.get("screening_only"),
    }


def _fmt(value: float) -> str:
    return "   --   " if value is None else f"{value:8.4f}"


def main() -> int:
    args = parse_args()
    arm_a = _load(args.arm_a)
    if arm_a is None:
        print(
            f"!! no Arm A record at {args.arm_a}\n"
            f"   run: python tools/extract_arm_a_session2.py",
            file=sys.stderr,
        )
        return 1

    subjects = sorted(arm_a["subjects"])
    rows = []
    missing = []
    for subject in subjects:
        b = load_arm_b(args.arm_b_root, args.arm_b_arm, subject, args.seed)
        if b is None:
            missing.append(subject)
            continue
        a_test = arm_a["subjects"][subject]["session2"]
        rows.append(
            {
                "subject": subject,
                "seed": args.seed,
                "arm_a": {label: float(a_test[a_key]) for label, a_key, _ in METRICS},
                "arm_a_rf_values": arm_a["subjects"][subject]["rf_values"],
                "arm_b": b["test"],
                "arm_b_nll": b["nll"],
                "arm_b_genotype": b["genotype"],
                "arm_b_parameters": b["parameters"],
                "arm_b_macs": b["macs"],
                "delta": {
                    label: b["test"][label] - float(a_test[a_key])
                    for label, a_key, _ in METRICS
                },
                "delta_nll": b["nll"] - float(a_test["loss"]),
            }
        )

    if not rows:
        print("!! no paired subjects found -- has the Arm B retrain finished?", file=sys.stderr)
        return 1

    header = (
        f"{'subj':<6}{'A acc':>9}{'B acc':>9}{'dAcc':>9}   "
        f"{'A f1':>8}{'B f1':>8}{'dF1':>9}   {'A kap':>8}{'B kap':>8}{'dKap':>9}"
    )
    print("paired Session-2 comparison (dataset's second recording session, code session=1)")
    print(header)
    print("-" * len(header))
    for row in rows:
        a, b, d = row["arm_a"], row["arm_b"], row["delta"]
        print(
            f"{row['subject']:<6}{a['acc']:>9.4f}{b['acc']:>9.4f}{d['acc']:>+9.4f}   "
            f"{a['f1']:>8.4f}{b['f1']:>8.4f}{d['f1']:>+9.4f}   "
            f"{a['kappa']:>8.4f}{b['kappa']:>8.4f}{d['kappa']:>+9.4f}"
        )

    summary = {}
    print()
    print(f"{'metric':<8}{'Arm A mean':>13}{'Arm A sd':>11}{'Arm B mean':>13}{'Arm B sd':>11}"
          f"{'mean delta':>13}{'sd':>9}{'B wins':>9}")
    for metric in LABELS:
        a_values = [row["arm_a"][metric] for row in rows]
        b_values = [row["arm_b"][metric] for row in rows]
        deltas = [row["delta"][metric] for row in rows]
        wins = sum(1 for value in deltas if value > 0)
        ties = sum(1 for value in deltas if value == 0)
        summary[metric] = {
            "arm_a_mean": statistics.mean(a_values),
            "arm_a_sd": statistics.stdev(a_values) if len(a_values) > 1 else None,
            "arm_b_mean": statistics.mean(b_values),
            "arm_b_sd": statistics.stdev(b_values) if len(b_values) > 1 else None,
            "paired_mean_delta": statistics.mean(deltas),
            "paired_sd_delta": statistics.stdev(deltas) if len(deltas) > 1 else None,
            "arm_b_wins": wins,
            "ties": ties,
            "n": len(deltas),
        }
        print(
            f"{metric:<8}{summary[metric]['arm_a_mean']:>13.4f}"
            f"{summary[metric]['arm_a_sd'] or 0:>11.4f}"
            f"{summary[metric]['arm_b_mean']:>13.4f}"
            f"{summary[metric]['arm_b_sd'] or 0:>11.4f}"
            f"{summary[metric]['paired_mean_delta']:>+13.4f}"
            f"{summary[metric]['paired_sd_delta'] or 0:>9.4f}"
            f"{f'{wins}/{len(deltas)}':>9}"
        )

    dacc = summary["acc"]
    print()
    print("reading")
    print(
        f"  the paired mean delta on accuracy is {dacc['paired_mean_delta']:+.4f} "
        f"(sd {dacc['paired_sd_delta']:.4f}) over {dacc['n']} subjects;"
    )
    mean_sd = (
        dacc["paired_sd_delta"] / (dacc["n"] ** 0.5) if dacc["paired_sd_delta"] else float("nan")
    )
    print(f"  its standard error is {mean_sd:.4f}, so the interval is roughly "
          f"{dacc['paired_mean_delta'] - 1.96 * mean_sd:+.4f} .. {dacc['paired_mean_delta'] + 1.96 * mean_sd:+.4f}.")
    if dacc["paired_mean_delta"] - 1.96 * mean_sd <= 0 <= dacc['paired_mean_delta'] + 1.96 * mean_sd:
        print("  That interval spans zero: this design does not separate the two arms.")
    else:
        print("  That interval excludes zero, but n=9 with one seed is a small basis for a claim.")

    print()
    print("searched architectures (are they actually different?)")
    for row in rows:
        rf = " / ".join(str(row["arm_a_rf_values"][b]) for b in ("Low", "Mid", "High"))
        genotype = row["arm_b_genotype"] or {}
        bands = genotype.get("bands", {})
        b_families = " / ".join(str(bands.get(b, {}).get("family", "?")) for b in ("Low", "Mid", "High"))
        b_rfs = " / ".join(str(bands.get(b, {}).get("rfs", "?")) for b in ("Low", "Mid", "High"))
        print(f"  {row['subject']}  A RF {rf}")
        print(f"        B {b_families}   RF {b_rfs}")

    distinct_a = {tuple(sorted(tuple(row["arm_a_rf_values"][b]) for b in ("Low", "Mid", "High"))) for row in rows}
    distinct_b = {
        tuple(sorted(tuple((row["arm_b_genotype"] or {}).get("bands", {}).get(b, {}).get("rfs", [])) for b in ("Low", "Mid", "High")))
        for row in rows
    }
    print(f"\n  distinct Arm A architectures across {len(rows)} subjects: {len(distinct_a)}")
    print(f"  distinct Arm B architectures across {len(rows)} subjects: {len(distinct_b)}")

    payload = {
        "arm_a_description": arm_a.get("arm_description"),
        "arm_b_description": "four-mechanism hierarchical search (family then RF)",
        "seed": args.seed,
        "session_note": "Session 2 = the dataset's second recording session = code session=1",
        "metrics": list(METRICS),
        "rows": rows,
        "summary": summary,
        "distinct_architectures": {"arm_a": len(distinct_a), "arm_b": len(distinct_b)},
        "subjects_missing_arm_b": missing,
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    if missing:
        print(f"\n!! Arm B is missing for subject(s) {missing}; the table above is partial")
    print(f"\nwritten: {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
