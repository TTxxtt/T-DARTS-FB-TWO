#!/usr/bin/env python
"""Normalise Arm A's archived official-FBNAS runs into one JSON.

Arm A is the vendored upstream chain driven by ``run/py/run_fbnas_subject.py``.
Its results are already on disk from 2026-09-15 -- nine subjects, seed 20190821,
each with the searched architecture (``opt_choice.csv``), the calibrated
1000-candidate traversal (``cali_bn_acc.npy``), and a Session-2 evaluation
(``results.csv``).  Nothing here retrains or re-evaluates anything.

This tool is strictly **read-only** with respect to
``run/outputs/fbnas/``: that tree is the archive of a frozen arm, and a report
that rewrote its own input would be indefensible.  The normalised file is
written next to Arm B's results instead.

Session numbering, stated once here because it is the thing most easily
confused: ``sub0``..``sub8`` are the nine BCI-IV-2a subjects 001..009 in sorted
order.  The `test` row of ``results.csv`` is the dataset's **second** recording
session (code ``session=1``), i.e. the held-out session the whole comparison is
about.

The RF index mapping is upstream's: ``opt_choice.csv`` lists indices into the
four ``ConvBn`` branches, which the frozen ``Cell`` builds at dilations
1/2/4/8 with kernel 15 -- effective receptive fields 15/29/57/113.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent

#: Index -> effective receptive field, from `1 + (kernel - 1) * dilation`.
RF_BY_INDEX = (15, 29, 57, 113)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive", type=Path,
        default=PROJECT_ROOT / "run/outputs/fbnas/bci42a/ses2Test",
        help="directory holding the per-subject official FBNAS runs",
    )
    parser.add_argument(
        "--json", type=Path,
        default=PROJECT_ROOT / "run/outputs/operator_armB_comparison/arm_a_session2.json",
        help="where to write the normalised record (deliberately NOT inside the archive)",
    )
    return parser.parse_args()


def _read_choice(path: Path) -> dict[str, list[int]]:
    """``opt_choice.csv`` is CRLF, with quoted multi-index values."""

    with path.open(newline="", encoding="utf-8") as handle:
        return {
            row[0]: [int(index) for index in re.findall(r"-?\d+", row[1])]
            for row in csv.reader(handle)
            if len(row) >= 2 and row[0] in {"Low", "Mid", "High"}
        }


def _read_results(path: Path) -> dict[str, dict]:
    """Pull the three metric rows out of ``results.csv``.

    The file stores a Python repr whose confusion matrix spans several lines, so
    each row is sliced from its own marker to the end of the file and scraped --
    the first occurrence of each key is the one belonging to that row.
    """

    text = path.read_text(encoding="utf-8")
    rows: dict[str, dict] = {}
    for marker in ("train:", "val: ", "test,"):
        if marker not in text:
            continue
        tail = text[text.index(marker):]
        values = dict(re.findall(r"'(\w+)': ([0-9.]+)", tail))
        if not {"acc", "f1", "kappa", "loss"} <= set(values):
            raise ValueError(f"{path}: could not parse the {marker!r} row")
        confusion = re.search(r"array\(\[(\[.*?\])\]\)", tail, re.S)
        rows[marker.strip(":, ")] = {
            "acc": float(values["acc"]),
            "f1": float(values["f1"]),
            "kappa": float(values["kappa"]),
            "loss": float(values["loss"]),
            "confusion_matrix": (
                [[int(n) for n in re.findall(r"\d+", line)]
                 for line in confusion.group(1).splitlines() if line.strip()]
                if confusion else None
            ),
        }
    return rows


def main() -> int:
    args = parse_args()
    if not args.archive.is_dir():
        print(f"!! no archive at {args.archive}", file=sys.stderr)
        return 1

    subjects: dict[str, dict] = {}
    for run_dir in sorted(args.archive.iterdir()):
        for sub_dir in sorted(run_dir.glob("sub*")):
            index = int(sub_dir.name[3:])
            subject = f"{index + 1:03d}"
            choice = _read_choice(sub_dir / "opt_choice.csv")
            calibration = np.load(sub_dir / "cali_bn_acc.npy")
            results = _read_results(sub_dir / "results.csv")

            entry = {
                "subject": subject,
                "sub_index": index,
                "run_dir": str(sub_dir.relative_to(PROJECT_ROOT)),
                "rf_indices": choice,
                "rf_values": {band: [RF_BY_INDEX[i] for i in choice[band]] for band in choice},
                "calibrated_traversal": {
                    "candidates": int(calibration.size),
                    "argmax_index": int(calibration.argmax()),
                    "argmax_accuracy": float(calibration.max()),
                    "min": float(calibration.min()),
                    "max": float(calibration.max()),
                    "mean": float(calibration.mean()),
                    "std": float(calibration.std()),
                },
                "session2": results.get("test"),
                "session1_split": {"train": results.get("train"), "val": results.get("val")},
            }
            if subject in subjects:
                raise ValueError(
                    f"{subject} appears in more than one archive run "
                    f"({subjects[subject]['run_dir']} and {entry['run_dir']}); "
                    f"the pairing is ambiguous"
                )
            subjects[subject] = entry

    missing = [f"{i:03d}" for i in range(1, 10) if f"{i:03d}" not in subjects]
    if missing:
        print(f"!! archive is missing subject(s) {missing}", file=sys.stderr)
        return 1

    accuracies = [subjects[s]["session2"]["acc"] for s in sorted(subjects)]
    payload = {
        "arm": "A",
        "arm_description": "dilated-only FBNAS search (vendored upstream chain)",
        "source": str(args.archive.relative_to(PROJECT_ROOT)),
        "seed": 20190821,
        "session_note": (
            "the 'session2' block is the dataset's second recording session, "
            "code session=1; 'session1_split' is the dataset's first session "
            "(code session=0), split 231/57"
        ),
        "selection_metric": "calibrated validation accuracy (NAS.nas_phase)",
        "rf_index_to_value": {str(i): rf for i, rf in enumerate(RF_BY_INDEX)},
        "session2_accuracy_mean": float(np.mean(accuracies)),
        "session2_accuracy_std": float(np.std(accuracies, ddof=1)),
        "subjects": {s: subjects[s] for s in sorted(subjects)},
    }

    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print(f"{'subj':<6}{'acc':>9}{'f1':>9}{'kappa':>9}{'loss':>9}   Low / Mid / High")
    for subject in sorted(subjects):
        entry = subjects[subject]
        s2 = entry["session2"]
        bands = " / ".join(str(entry["rf_values"][b]) for b in ("Low", "Mid", "High"))
        print(
            f"{subject:<6}{s2['acc']:>9.4f}{s2['f1']:>9.4f}{s2['kappa']:>9.4f}{s2['loss']:>9.4f}   {bands}"
        )
    print(
        f"\nmean Session-2 accuracy over {len(subjects)} subjects: "
        f"{payload['session2_accuracy_mean']:.6f} +/- {payload['session2_accuracy_std']:.6f}"
    )
    print(f"written: {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
