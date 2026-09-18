#!/usr/bin/env python
"""Summarise the Phase B RF-only searches: unordered RF-pair stability.

Each searched band exports two RFs.  The two paths are interchangeable at the
architecture level, so ``(29, 57)`` and ``(57, 29)`` are one architecture;
counting raw path labels would call a label swap an architecture change.  This
tool therefore counts each band's *sorted* RF pair across seeds and reports the
mode agreement, alongside the search's soft/hard validation metrics and, when
present, the from-scratch screening retrain of each searched genotype.
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
    parser.add_argument("--search-arm", default="rfsearch")
    parser.add_argument("--retrain-arm", default="rfretrain")
    parser.add_argument("--subjects", nargs="+", default=["003"])
    parser.add_argument("--seeds", nargs="+", default=["20250901", "20250902", "20250903"])
    parser.add_argument("--json", type=Path, default=None, help="also write the raw rows here")
    return parser.parse_args()


def _load(path: Path) -> dict | None:
    if not path.is_file():
        print(f"!! missing: {path}", file=sys.stderr)
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_search(args: argparse.Namespace, subject: str, seed: str) -> dict | None:
    leaf = run_leaf("search", subject_id(subject), seed, None)
    payload = _load(args.output_root / args.search_arm / args.dataset / leaf / "final_summary.json")
    if payload is None:
        return None
    pairs = payload["unordered_rf_pairs"]
    return {
        "subject": subject_id(subject),
        "seed": seed,
        "pairs": {band: sorted(int(rf) for rf in pairs[band]) for band in BANDS},
        "no_duplicate_paths": payload["no_duplicate_paths"],
        "soft_acc": payload["final_val_soft"]["acc"],
        "hard_acc": payload["final_val_hard"]["acc"],
        "soft_nll": payload["final_val_soft"]["nll"],
        "hard_nll": payload["final_val_hard"]["nll"],
        "gap": payload["soft_hard_acc_gap"],
        "mixtures": payload["rf_mixtures"],
    }


def load_retrain(args: argparse.Namespace, subject: str, seed: str) -> dict | None:
    leaf = run_leaf("train", subject_id(subject), seed, "rf")
    payload = _load(args.output_root / args.retrain_arm / args.dataset / leaf / "final_summary.json")
    if payload is None:
        return None
    validation = payload.get("validation_best")
    if validation is None:
        print(f"!! retrain {leaf} has no validation_best; was it run with --screening-only?", file=sys.stderr)
        return None
    pairs = {}
    for band in BANDS:
        pairs[band] = sorted(
            int(gene["candidate"].rsplit("_rf", 1)[1])
            for gene in payload["genotype"]["genes"]
            if gene["band"] == band
        )
    return {
        "subject": subject_id(subject),
        "seed": seed,
        "pairs": pairs,
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
    searches = [
        row
        for subject in args.subjects
        for seed in args.seeds
        if (row := load_search(args, subject, seed)) is not None
    ]
    if not searches:
        return 1

    header = f"{'subj':<6}{'seed':<11}{'Low':<12}{'Mid':<12}{'High':<12}{'softAcc':>9}{'hardAcc':>9}{'gap':>8}"
    print("per-seed RF pairs (sorted; (29,57) == (57,29)) and search validation metrics")
    print(header)
    print("-" * len(header))
    for row in searches:
        cells = "".join(f"{str(row['pairs'][band]):<12}" for band in BANDS)
        print(
            f"{row['subject']:<6}{row['seed']:<11}{cells}"
            f"{row['soft_acc']:>9.4f}{row['hard_acc']:>9.4f}{row['gap']:>+8.4f}"
        )

    print("\nper-band unordered RF-pair counts across seeds")
    for band in BANDS:
        counter = Counter(tuple(row["pairs"][band]) for row in searches)
        total = sum(counter.values())
        mode, count = counter.most_common(1)[0]
        share = ", ".join(f"{list(pair)}x{n}" for pair, n in counter.most_common())
        print(f"  {band:<5} {share}   mode={list(mode)} agreement={count}/{total}")

    retrains = [
        row
        for subject in args.subjects
        for seed in args.seeds
        if (row := load_retrain(args, subject, seed)) is not None
    ]
    if retrains:
        print("\nfrom-scratch screening retrain of each searched genotype (validation-best)")
        header = f"{'seed':<11}{'Low':<12}{'Mid':<12}{'High':<12}{'valAcc':>9}{'valNLL':>9}{'kappa':>8}{'macroF1':>9}{'params':>8}{'MACs':>12}{'dups':>5}"
        print(header)
        print("-" * len(header))
        for row in retrains:
            cells = "".join(f"{str(row['pairs'][band]):<12}" for band in BANDS)
            print(
                f"{row['seed']:<11}{cells}{row['val_acc']:>9.4f}{row['val_nll']:>9.4f}"
                f"{row['kappa']:>8.4f}{row['macro_f1']:>9.4f}{row['parameters']:>8}{row['macs']:>12}"
                f"{len(row['duplicate_structure_bands']):>5}"
            )
        for key, label in (("val_acc", "valAcc"), ("val_nll", "valNLL"), ("kappa", "kappa"), ("macro_f1", "macroF1")):
            mean, std = _mean_std([row[key] for row in retrains])
            print(f"  mean ± std {label:<8}: {_fmt(mean, std)}")
    else:
        print("\nno retrain results found (run submit_rf_retrain.sh once the searches finish)")
        return 0

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(
                {
                    "searches": [
                        {key: value for key, value in row.items() if key != "mixtures"}
                        for row in searches
                    ],
                    "rf_mixtures": {f"{row['subject']}/{row['seed']}": row["mixtures"] for row in searches},
                    "retrains": retrains,
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
