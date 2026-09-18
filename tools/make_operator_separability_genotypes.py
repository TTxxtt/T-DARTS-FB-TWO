#!/usr/bin/env python
"""Write the four fixed operator-separability genotypes.

Path 0 is fixed at ``dilated_rf57`` in every band.  Path 1 is one of
``dilated_rf57`` / ``normal_rf57`` / ``dwsep_rf57`` / ``lkdw_rf57``, the same
choice in all three bands.  The four files therefore differ in exactly one
variable -- the convolution family of path 1 at a fixed receptive field -- which
is what makes them an operator comparison rather than a search result.

The files are ordinary genotype JSON, so they enter ``train_retrain.py``
through ``--genotype-json``, the same door a searched genotype uses.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tdarts.genotype import Genotype, PathGene, genotype_nodes, save_genotype  # noqa: E402
from tdarts.mixed_op import candidate_names  # noqa: E402

#: Fixed anchor: the same operator in path 0 of every band.
PATH0_CANDIDATE = "dilated_rf57"

#: Path-1 variants, in submission order.
PATH1_VARIANTS = ("dilated_rf57", "normal_rf57", "dwsep_rf57", "lkdw_rf57")


def _gene(band: str, path: int, candidate: str) -> PathGene:
    names = candidate_names(band)
    if candidate not in names:
        raise ValueError(
            f"{candidate!r} is not a canonical candidate for {band}: {names}"
        )
    index = names.index(candidate)
    # alpha/probability are placeholders: these genotypes were not extracted
    # from a search epoch, exactly like sample_genotype's convention.
    return PathGene(band, path, index, candidate, 0.0, 1.0)


def build_genotype(path1_candidate: str) -> Genotype:
    genes = tuple(
        _gene(band, path, PATH0_CANDIDATE if path == 0 else path1_candidate)
        for band, path in genotype_nodes()
    )
    return Genotype(seed=0, epoch=0, genes=genes)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)

    files = {}
    for candidate in PATH1_VARIANTS:
        path = args.output_dir / f"path1_{candidate}.json"
        save_genotype(build_genotype(candidate), path)
        files[candidate] = path.name

    manifest = {
        "path0": PATH0_CANDIDATE,
        "path1_variants": list(PATH1_VARIANTS),
        "receptive_field": 57,
        "bands": "Low/Mid/High share the same (path0, path1) pair",
        "search_produced": False,
        "files": files,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {len(files)} genotypes to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
