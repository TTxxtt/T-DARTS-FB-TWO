"""Extract fixed six-path temporal genotypes from Stage-3 search logs."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from tdarts.mixed_op import candidate_names

__all__ = ["PathGene", "Genotype", "extract_genotype"]


@dataclass(frozen=True)
class PathGene:
    band: str
    path: int  # zero-indexed internally
    candidate_index: int
    candidate: str
    alpha: float
    probability: float


@dataclass(frozen=True)
class Genotype:
    """The six fixed temporal choices selected from exactly one epoch."""

    seed: int
    epoch: int
    genes: tuple[PathGene, ...]

    def gene(self, band: str, path: int) -> PathGene:
        return next(gene for gene in self.genes if gene.band == band and gene.path == path)

    def to_dict(self) -> dict:
        return {"seed": self.seed, "epoch": self.epoch, "genes": [asdict(gene) for gene in self.genes]}


def extract_genotype(metrics_path: str | Path, *, epoch: int = 200) -> Genotype:
    """Extract six Top-1 candidates from an exact logged search epoch."""

    path = Path(metrics_path)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    row = next((item for item in rows if item["epoch"] == epoch), None)
    if row is None:
        raise ValueError(f"{path} does not contain epoch {epoch}")
    # Recover the seed from the search directory name.  This used to be
    # `name.removeprefix("search_s003_seed")` + int(), which hard-codes one
    # subject and only accepts the flat search_s<subject>_seed<seed> spelling.
    # Searching for the `seed<digits>` field instead accepts any directory name
    # that carries it, including the structured
    # <stamp>_s<subject>_seed<seed>_<arm> leaf.
    match = re.search(r"seed(\d+)", path.parent.name)
    if match is None:
        raise ValueError(f"cannot recover the search seed from directory {path.parent.name!r}")
    seed = int(match.group(1))
    genes = []
    for node, values in sorted(row["paths"].items()):
        band, path_text = node.split("_path")
        path_index = int(path_text) - 1
        candidate_index = int(values["top1"]["index"])
        names = candidate_names(band)
        if values["top1"]["candidate"] != names[candidate_index]:
            raise ValueError(f"candidate ordering mismatch in {path} at {node}")
        genes.append(
            PathGene(
                band=band,
                path=path_index,
                candidate_index=candidate_index,
                candidate=names[candidate_index],
                alpha=float(values["alpha"][candidate_index]),
                probability=float(values["probabilities"][candidate_index]),
            )
        )
    if len(genes) != 6:
        raise ValueError(f"expected six genes, got {len(genes)}")
    return Genotype(seed=seed, epoch=epoch, genes=tuple(genes))
