"""Fixed six-path temporal genotypes: extraction from search logs, and sampling.

``extract_genotype`` reads the Top-1 choices out of a logged search epoch.
``sample_random_genotype`` draws the same six choices at random, for the
architecture-landscape baseline that asks whether a searched genotype beats
chance at all.

``duplicate_structure_bands`` answers the structural question a genotype cannot:
two paths may carry different candidate *names* while realising the same
``(kernel, dilation, separable)`` function.  It compares the built operators'
``structure_key`` rather than their strings, so aliases such as
``dilated_rf15`` / ``normal_rf15`` are caught as duplicates too.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from tdarts import config as C
from tdarts.init_utils import stable_seed
from tdarts.mixed_op import candidate_names
from tdarts.temporal_ops import build_temporal_op

__all__ = [
    "PathGene",
    "Genotype",
    "extract_genotype",
    "sample_random_genotype",
    "save_genotype",
    "load_genotype",
    "genotype_nodes",
    "path_structure_keys",
    "duplicate_structure_bands",
]


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
    """The six fixed temporal choices selected from exactly one epoch.

    ``genes`` is normalised to :func:`genotype_nodes` order on construction, so
    a genotype built in ``C.BANDS`` order (as ``tdarts.anchored`` does) and one
    built in sorted metric-key order (as :func:`extract_genotype` does) are
    equal, serialise identically and load identically.  Producers therefore do
    not have to agree on iteration order.
    """

    seed: int
    epoch: int
    genes: tuple[PathGene, ...]

    def __post_init__(self) -> None:
        order = {node: index for index, node in enumerate(genotype_nodes())}
        genes = tuple(
            sorted(self.genes, key=lambda gene: order.get((gene.band, gene.path), len(order)))
        )
        if genes != self.genes:
            object.__setattr__(self, "genes", genes)

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


def genotype_nodes() -> tuple[tuple[str, int], ...]:
    """The six ``(band, path)`` slots, in :func:`extract_genotype`'s order.

    ``extract_genotype`` walks ``sorted(row["paths"].items())``, whose keys are
    the logged ``"<band>_path<n>"`` strings.  Sorting those puts High before Low
    before Mid -- not the declaration order of ``C.BANDS``.  A sampled genotype
    is only interchangeable with an extracted one if it is built in that same
    order, so both paths go through this helper and cannot drift apart.
    """
    nodes = [(band, path) for band in C.BANDS for path in range(C.NUM_PATHS)]
    nodes.sort(key=lambda node: f"{node[0]}_path{node[1] + 1}")
    return tuple(nodes)


def sample_random_genotype(seed: int, *, epoch: int = 0) -> Genotype:
    """One uniformly drawn candidate per slot, reproducible from ``seed``.

    Each slot draws from its own ``stable_seed(seed, node)`` rather than from a
    single running RNG, so the value a slot receives depends only on that slot
    and the seed -- not on how many slots were drawn before it.

    ``alpha`` and ``probability`` are placeholders (``0.0`` / ``1.0``): a sampled
    genotype has no architecture logits behind it.  That is the same convention
    the discrete-network tests use for hand-built genes, and ``epoch=0`` marks
    the genotype as sampled rather than extracted from a logged epoch.
    """
    genes = []
    for band, path in genotype_nodes():
        names = candidate_names(band)
        index = stable_seed(int(seed), f"{band}_path{path + 1}") % len(names)
        genes.append(
            PathGene(
                band=band,
                path=path,
                candidate_index=index,
                candidate=names[index],
                alpha=0.0,
                probability=1.0,
            )
        )
    return Genotype(seed=int(seed), epoch=int(epoch), genes=tuple(genes))


def save_genotype(genotype: Genotype, path: str | Path) -> Path:
    """Write ``genotype.to_dict()`` -- the format :func:`load_genotype` reads.

    ``Genotype.__post_init__`` stores genes in the canonical slot order, so the
    file is order-stable no matter which producer built the object.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(genotype.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def load_genotype(source: str | Path | dict) -> Genotype:
    """Read a genotype from a JSON file path, or reuse an already-loaded dict.

    ``--stage2-only`` rebuilds a genotype from ``final_summary.json``, which
    stores ``Genotype.to_dict()`` inline rather than a path, so the loader must
    accept that dict directly.  The candidate string is checked against
    ``candidate_names(band)[index]`` for the same reason
    :func:`extract_genotype` checks it: a file written against a different
    candidate ordering would otherwise decode to a *different* network than the
    one it claims to describe, and nothing downstream would notice.
    """
    if isinstance(source, dict):
        payload = source
        label = "<inline dict>"
    else:
        path = Path(source)
        label = str(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
    genes = []
    for entry in payload["genes"]:
        band = entry["band"]
        index = int(entry["candidate_index"])
        names = candidate_names(band)
        if entry["candidate"] != names[index]:
            raise ValueError(
                f"candidate ordering mismatch in {label}: {band}[{index}] is "
                f"{names[index]!r} here but the source says {entry['candidate']!r}"
            )
        genes.append(
            PathGene(
                band=band,
                path=int(entry["path"]),
                candidate_index=index,
                candidate=entry["candidate"],
                alpha=float(entry["alpha"]),
                probability=float(entry["probability"]),
            )
        )
    # Coverage, not serialization order: Genes are addressed by (band, path)
    # everywhere downstream, and different producers write different orders --
    # extract_genotype walks the sorted log keys (High, Low, Mid) while
    # tdarts.anchored builds in C.BANDS order (Low, Mid, High).  Requiring one
    # specific order here would reject a structurally valid file.  Duplicate or
    # missing slots are still errors.
    expected = set(genotype_nodes())
    found = [(gene.band, gene.path) for gene in genes]
    if len(found) != len(expected) or set(found) != expected:
        raise ValueError(f"{label} does not cover {sorted(expected)} exactly once; got {found}")
    return Genotype(
        seed=int(payload["seed"]), epoch=int(payload["epoch"]), genes=tuple(genes)
    )


def path_structure_keys(
    genotype: Genotype,
) -> dict[str, tuple[tuple[int, int, bool], tuple[int, int, bool]]]:
    """Per band, the built ``structure_key`` of both selected paths.

    ``(kernel, dilation, separable)`` is read off the actual
    :class:`~tdarts.temporal_ops.TemporalOp` that the candidate name builds,
    not off the name's text, so this is the structural identity
    :func:`duplicate_structure_bands` compares.
    """

    keys: dict[str, tuple[tuple[int, int, bool], tuple[int, int, bool]]] = {}
    for band in C.BANDS:
        pair = []
        for path in range(C.NUM_PATHS):
            gene = genotype.gene(band, path)
            op_name, rf_text = gene.candidate.rsplit("_rf", 1)
            op = build_temporal_op(op_name, band, int(rf_text), use_norm=False)
            pair.append(op.structure_key)
        keys[band] = (pair[0], pair[1])
    return keys


def duplicate_structure_bands(genotype: Genotype) -> tuple[str, ...]:
    """Bands whose two paths realise the same structure.

    The comparison is on ``structure_key``, so a genotype that mixes names for
    one function -- ``dilated_rf15`` in path 0 and ``normal_rf15`` in path 1 --
    is reported as a duplicate even though the two strings differ.  Returns the
    band names in ``C.BANDS`` order; empty means every band has two distinct
    structures.
    """

    return tuple(
        band
        for band, (first, second) in path_structure_keys(genotype).items()
        if first == second
    )
