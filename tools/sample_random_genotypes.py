#!/usr/bin/env python
"""Generate unique, reproducible random six-path temporal genotypes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tdarts.genotype import sample_random_genotype, save_genotype  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20250901)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _architecture_key(genotype) -> tuple[int, ...]:
    return tuple(gene.candidate_index for gene in genotype.genes)


def main() -> int:
    args = parse_args()
    if args.count < 1:
        raise ValueError("--count must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=False)

    records = []
    seen: set[tuple[int, ...]] = set()
    sample_seed = args.seed
    while len(records) < args.count:
        genotype = sample_random_genotype(sample_seed)
        key = _architecture_key(genotype)
        if key not in seen:
            index = len(records)
            path = args.output_dir / f"g{index:03d}.json"
            save_genotype(genotype, path)
            records.append(
                {
                    "index": index,
                    "file": path.name,
                    "sampling_seed": sample_seed,
                    "candidate_indices": list(key),
                }
            )
            seen.add(key)
        sample_seed += 1

    manifest = {
        "count": args.count,
        "base_seed": args.seed,
        "sampling": "independent uniform candidate per path; duplicates rejected",
        "records": records,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {args.count} unique genotypes to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
