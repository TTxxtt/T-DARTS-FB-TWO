#!/usr/bin/env python
"""Write the fixed-RF57 single-band operator specificity probe.

Protocol: path 0 is ``dilated_rf57`` in every band; path 1 is ``dilated_rf57``
too, except in exactly one band, where it takes one of the other three
operator families -- all at the same effective RF 57, so only the convolution
mechanism varies:

    low_normal   Low  Path1=normal_rf57   Mid/High Path1=dilated_rf57
    low_dwsep    ...
    low_lkdw
    mid_normal / mid_dwsep / mid_lkdw
    high_normal / high_dwsep / high_lkdw
    baseline_dd  every band Path1=dilated_rf57

The three ``<band>_dilated`` configurations collapse to ``baseline_dd`` (the
same six genes), so only the ten distinct files are written.  The manifest
records the equivalence so nobody re-runs the same architecture three times.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tdarts import config as C  # noqa: E402
from tdarts.genotype import Genotype, PathGene, save_genotype  # noqa: E402
from tdarts.mixed_op import candidate_names  # noqa: E402

BANDS = ("Low", "Mid", "High")
OPERATORS = ("dilated", "normal", "dwsep", "lkdw")
RF = 57
#: Path 0 and every non-tested path 1 stay on this candidate.
ANCHOR = "dilated_rf57"


def build_genotype(path1_by_band: dict[str, str], *, seed: int = 0) -> Genotype:
    genes = []
    for band in C.BANDS:
        names = candidate_names(band)
        pair = (ANCHOR, path1_by_band[band])
        for path, candidate in enumerate(pair):
            genes.append(
                PathGene(band, path, names.index(candidate), candidate, 0.0, 1.0)
            )
    return Genotype(seed=seed, epoch=0, genes=tuple(genes))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)

    records = []
    for band in BANDS:
        for operator in OPERATORS:
            if operator == "dilated":
                continue  # collapses to baseline_dd; recorded below
            arm = f"{band.lower()}_{operator}"
            pairs = {name: ANCHOR for name in BANDS}
            pairs[band] = f"{operator}_rf{RF}"
            path = args.output_dir / f"{arm}.json"
            save_genotype(build_genotype(pairs), path)
            records.append({"arm": arm, "band": band, "operator": operator, "file": path.name})

    baseline_pairs = {name: ANCHOR for name in BANDS}
    baseline_path = args.output_dir / "baseline_dd.json"
    save_genotype(build_genotype(baseline_pairs), baseline_path)

    manifest = {
        "scheme": "band_operator_probe_rf57",
        "rf": RF,
        "anchor": ANCHOR,
        "bands": list(BANDS),
        "operators": list(OPERATORS),
        "baseline_file": baseline_path.name,
        "collapsed": {
            f"{band.lower()}_dilated": "identical to baseline_dd"
            for band in BANDS
        },
        "varied": records,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"wrote {len(records)} varied genotypes + baseline to {args.output_dir} "
        f"(RF{RF}; {len(BANDS)} collapsed dilated configs)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
