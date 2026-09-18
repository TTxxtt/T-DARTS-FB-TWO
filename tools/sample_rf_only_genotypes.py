#!/usr/bin/env python
"""Pre-register the random RF-only baseline sample.

The RF-only search space has six unordered RF pairs per band, so 6^3 = 216
genotypes.  This tool fixes an RNG seed, samples 12 of them without
replacement, excludes the searched genotypes and the majority-vote genotype,
and writes the majority-vote genotype beside them.

    random g000..g011 + majority.json, all in --output-dir

The manifest is written in the same run, before any score exists: that ordering
is the pre-registration.  Do not re-run against a different seed after looking
at results.
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tdarts import config as C  # noqa: E402
from tdarts.genotype import Genotype, PathGene, save_genotype  # noqa: E402
from tdarts.mixed_op import candidate_names  # noqa: E402

BANDS = ("Low", "Mid", "High")

#: The six unordered RF pairs available under --no-duplicate-paths.
RF_PAIRS = ((15, 29), (15, 57), (15, 113), (29, 57), (29, 113), (57, 113))

#: The majority vote across the three RF-search seeds.
MAJORITY = {"Low": (15, 29), "Mid": (57, 113), "High": (15, 29)}


def build_genotype(pairs: dict[str, tuple[int, int]], *, seed: int = 0) -> Genotype:
    genes = []
    for band in C.BANDS:
        names = candidate_names(band)
        for path, rf in enumerate(sorted(pairs[band])):
            candidate = f"dilated_rf{rf}"
            genes.append(
                PathGene(band, path, names.index(candidate), candidate, 0.0, 1.0)
            )
    return Genotype(seed=seed, epoch=0, genes=tuple(genes))


def searched_pairs(search_root: Path) -> set[tuple[tuple[int, int], ...]]:
    found = set()
    for path in sorted(search_root.glob("search_s*_seed*/genotype.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        pairs = tuple(
            tuple(
                sorted(
                    int(g["candidate"].rsplit("_rf", 1)[1])
                    for g in payload["genes"]
                    if g["band"] == band
                )
            )
            for band in BANDS
        )
        found.add(pairs)
    return found


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20250917, help="sampling RNG seed (pre-registered)")
    parser.add_argument("--count", type=int, default=12)
    parser.add_argument(
        "--search-root",
        type=Path,
        default=Path("run/outputs/rfsearch/bci42a"),
        help="directory holding the searched genotypes to exclude",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.count < 1 or args.count > len(RF_PAIRS) ** 3:
        raise ValueError(f"count must be in 1..{len(RF_PAIRS) ** 3}")
    args.output_dir.mkdir(parents=True, exist_ok=False)

    universe = list(itertools.product(RF_PAIRS, repeat=len(BANDS)))
    excluded = searched_pairs(args.search_root)
    majority_key = tuple(MAJORITY[band] for band in BANDS)
    excluded.add(majority_key)

    eligible = [combo for combo in universe if combo not in excluded]
    rng = random.Random(args.seed)
    sample = rng.sample(eligible, args.count)

    records = []
    for index, combo in enumerate(sample):
        pairs = {band: tuple(combo[i]) for i, band in enumerate(BANDS)}
        path = args.output_dir / f"g{index:03d}.json"
        save_genotype(build_genotype(pairs), path)
        records.append(
            {
                "index": index,
                "file": path.name,
                "rf_pairs": [list(pair) for pair in combo],
            }
        )

    majority_path = args.output_dir / "majority.json"
    save_genotype(build_genotype(MAJORITY), majority_path)

    manifest = {
        "scheme": "rf_only_no_duplicate",
        "sampling_seed": args.seed,
        "count": args.count,
        "universe_size": len(universe),
        "eligible_size": len(eligible),
        "excluded_searched": sorted(list(pair) for pair in excluded - {majority_key}),
        "excluded_majority": list(majority_key),
        "majority_file": majority_path.name,
        "random": records,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"wrote {args.count} random genotypes + majority to {args.output_dir} "
        f"(seed={args.seed}, excluded {len(excluded)} combos)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
