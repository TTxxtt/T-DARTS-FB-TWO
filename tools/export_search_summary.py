#!/usr/bin/env python
"""Export final architecture weights from completed Stage-3 search runs.

The CSV is deliberately long-form: each row is one candidate in one of the
six architecture nodes (Low/Mid/High x path 1/2).  ``alpha`` is the learned
logit and ``probability`` is its 14-way softmax weight.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs-root", type=Path, default=Path("outputs"))
    parser.add_argument(
        "--seeds",
        nargs="+",
        default=["20250901", "20250902", "20250903"],
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/search_s003_3seed_final_architecture_weights.csv"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows: list[dict[str, object]] = []
    for seed in args.seeds:
        metrics_path = args.outputs_root / f"search_s003_seed{seed}" / "metrics.jsonl"
        if not metrics_path.is_file():
            raise FileNotFoundError(f"missing completed search log: {metrics_path}")
        final = json.loads(metrics_path.read_text(encoding="utf-8").splitlines()[-1])
        if final["epoch"] != 200:
            raise ValueError(f"{metrics_path} ends at epoch {final['epoch']}, not Search200")
        for node, values in sorted(final["paths"].items()):
            band, path = node.split("_path")
            for index, (alpha, probability) in enumerate(
                zip(values["alpha"], values["probabilities"])
            ):
                candidate = (
                    values["top1"]["candidate"]
                    if index == values["top1"]["index"]
                    else values["top2"]["candidate"]
                    if index == values["top2"]["index"]
                    else None
                )
                # Obtain the canonical name from the metrics' top entries only
                # when possible; the fully populated mapping is reconstructed
                # from tdarts below after importing the project package.
                rows.append(
                    {
                        "seed": seed,
                        "epoch": final["epoch"],
                        "band": band,
                        "path": int(path),
                        "candidate_index": index,
                        "candidate": candidate,
                        "alpha": alpha,
                        "probability": probability,
                        "is_top1": index == values["top1"]["index"],
                        "is_top2": index == values["top2"]["index"],
                        "node_margin": values["margin"],
                        "node_entropy": values["entropy"],
                    }
                )

    from tdarts.mixed_op import candidate_names

    for row in rows:
        row["candidate"] = candidate_names(str(row["band"]))[int(row["candidate_index"])]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
