"""Arm B's discrete network and genotype dialect.

Arm B's searched object is not a six-gene path list in the 14-candidate
registry: a band fixes **one family** (Phase A) and then chooses **one or two
receptive fields** within it (Phase B).  A band therefore holds ``k in {1, 2}``
paths, and the FBNAS width arithmetic gives each path ``F1 // k`` channels so
the band still emits ``F1 = 12`` and the backbone is untouched.

That grammar cannot be expressed as the plain six-gene file -- ``PathGene``
carries a single ``"<op>_rf<int>"`` string and one gene per ``(band, path)`` --
so this module defines its own dialect, tagged with :data:`BAND_SCHEME`, which
``train_retrain.py`` dispatches on exactly as it already dispatches the anchored
dialect.  Everything downstream of the genotype -- the two-stage protocol, the
Stage-2 stopping rule, params/MACs accounting, the Session-1 observation flag --
is shared verbatim, which is the point: this arm must differ from Arm A in its
*search space*, not in how its architecture is trained.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import torch
import torch.nn as nn

from tdarts import config as C
from tdarts.backbone import TemporalBackbone
from tdarts.band_supernet import (
    BAND_FAMILIES,
    BAND_RFS,
    EXCLUDED_FAMILY,
    build_arm_b_operator,
)

__all__ = [
    "BAND_SCHEME",
    "MAX_PATHS_PER_BAND",
    "BandGene",
    "BandGenotype",
    "load_band_genotype",
    "band_structure_keys",
    "band_duplicate_bands",
    "BandDiscreteCell",
    "BandDiscreteNet",
    "describe_genotype",
]

#: Scheme tag written into ``genotype.json`` and required by the loader.
BAND_SCHEME = "band_family_rf"

#: A band selects one or two receptive fields -- upstream's ``C(4,1) + C(4,2)``.
MAX_PATHS_PER_BAND = 2


@dataclass(frozen=True)
class BandGene:
    """One path: a family realised at one receptive field inside one band."""

    band: str
    path: int
    family: str
    target_rf: int

    @property
    def label(self) -> str:
        return f"{self.family}_rf{self.target_rf}"

    def to_dict(self) -> dict:
        return {
            "band": self.band,
            "path": self.path,
            "family": self.family,
            "target_rf": self.target_rf,
            "label": self.label,
        }


@dataclass(frozen=True)
class BandGenotype:
    """A searched Arm B architecture: per band, one family and 1-2 RFs."""

    seed: int
    epoch: int
    genes: tuple[BandGene, ...]
    phase_a: Mapping[str, str] | None = None
    phase_b: Mapping[str, tuple[int, ...]] | None = None

    def __post_init__(self) -> None:
        by_band: dict[str, list[BandGene]] = {band: [] for band in C.BANDS}
        for gene in self.genes:
            if gene.band not in by_band:
                raise ValueError(f"unknown band {gene.band!r} in genotype")
            by_band[gene.band].append(gene)
        for band, genes in by_band.items():
            if not genes:
                raise ValueError(f"band {band!r} has no genes")
            paths = sorted(gene.path for gene in genes)
            if paths != list(range(len(genes))):
                raise ValueError(
                    f"band {band!r}: path indices must be 0..k-1, got {paths}"
                )
            families = {gene.family for gene in genes}
            if len(families) != 1:
                raise ValueError(
                    f"band {band!r}: paths must share one family, got {sorted(families)}"
                )

    def paths_of(self, band: str) -> tuple[BandGene, ...]:
        return tuple(sorted((g for g in self.genes if g.band == band), key=lambda g: g.path))

    def family_of(self, band: str) -> str:
        return self.paths_of(band)[0].family

    def rfs_of(self, band: str) -> tuple[int, ...]:
        return tuple(gene.target_rf for gene in self.paths_of(band))

    def to_dict(self) -> dict:
        return {
            "scheme": BAND_SCHEME,
            "seed": self.seed,
            "epoch": self.epoch,
            "bands": {
                band: {
                    "family": self.family_of(band),
                    "rfs": list(self.rfs_of(band)),
                    "paths": [gene.to_dict() for gene in self.paths_of(band)],
                }
                for band in C.BANDS
            },
            "phase_a": dict(self.phase_a) if self.phase_a else None,
            "phase_b": {
                band: list(rfs) for band, rfs in (self.phase_b or {}).items()
            }
            or None,
        }


def describe_genotype(genotype: BandGenotype) -> str:
    """The human-readable form the phase report asks for.

    ::

        Subject003

        Phase A:
        Low  = dynamic_e
        Mid  = dilated_e
        High = gated_e

        Phase B:
        Low  = RF29 + RF57
        Mid  = RF57
        High = RF15 + RF29
    """

    lines = ["Phase A:"]
    for band in C.BANDS:
        lines.append(f"{band:<5}= {genotype.family_of(band)}")
    lines.append("")
    lines.append("Phase B:")
    for band in C.BANDS:
        rfs = genotype.rfs_of(band)
        joined = " + ".join(f"RF{rf}" for rf in rfs)
        lines.append(f"{band:<5}= {joined}")
    return "\n".join(lines)


def load_band_genotype(source: str | Path | dict) -> BandGenotype:
    """Read and validate an Arm B ``genotype.json``.

    Rejects rather than repairs, for the same reason the anchored loader does:
    a silently misread architecture would be indistinguishable from a searched
    one, and this dialect's whole purpose is that its architectures are
    reproducible from the file.
    """

    if isinstance(source, dict):
        payload = source
        label = "<inline dict>"
    else:
        path = Path(source)
        label = str(path)
        payload = json.loads(path.read_text(encoding="utf-8"))

    scheme = payload.get("scheme")
    if scheme != BAND_SCHEME:
        raise ValueError(f"{label}: expected scheme {BAND_SCHEME!r}, got {scheme!r}")

    raw_bands = payload.get("bands")
    if not isinstance(raw_bands, Mapping):
        raise ValueError(f"{label}: 'bands' must be a mapping")
    if set(raw_bands) != set(C.BANDS):
        raise ValueError(
            f"{label}: 'bands' must name exactly {list(C.BANDS)}, got {sorted(raw_bands)}"
        )

    genes: list[BandGene] = []
    for band in C.BANDS:
        entry = raw_bands[band]
        if not isinstance(entry, Mapping):
            raise ValueError(f"{label}: band {band!r} must be a mapping")
        family = str(entry.get("family", ""))
        if family == EXCLUDED_FAMILY:
            raise ValueError(
                f"{label}: {band}: {EXCLUDED_FAMILY} is not part of this arm's search space"
            )
        if family not in BAND_FAMILIES:
            raise ValueError(
                f"{label}: {band}: unknown family {family!r}; expected one of {list(BAND_FAMILIES)}"
            )
        rfs = entry.get("rfs")
        if not isinstance(rfs, Sequence) or isinstance(rfs, (str, bytes)):
            raise ValueError(f"{label}: {band}: 'rfs' must be a list")
        rfs = [int(rf) for rf in rfs]
        if not 1 <= len(rfs) <= MAX_PATHS_PER_BAND:
            raise ValueError(
                f"{label}: {band}: expected 1 or {MAX_PATHS_PER_BAND} rfs, got {rfs}"
            )
        if len(set(rfs)) != len(rfs):
            raise ValueError(f"{label}: {band}: repeated rf in {rfs}")
        for rf in rfs:
            if rf not in BAND_RFS:
                raise ValueError(
                    f"{label}: {band}: rf {rf} is not on the ladder {list(BAND_RFS)}"
                )
        for path_index, rf in enumerate(rfs):
            genes.append(BandGene(band=band, path=path_index, family=family, target_rf=rf))

    phase_a = payload.get("phase_a")
    phase_b = payload.get("phase_b")
    return BandGenotype(
        seed=int(payload.get("seed", 0)),
        epoch=int(payload.get("epochs", payload.get("epoch", 0))),
        genes=tuple(genes),
        phase_a=dict(phase_a) if isinstance(phase_a, Mapping) else None,
        phase_b=(
            {band: tuple(int(rf) for rf in phase_b[band]) for band in phase_b}
            if isinstance(phase_b, Mapping)
            else None
        ),
    )


def band_structure_keys(genotype: BandGenotype) -> dict[str, tuple[tuple, ...]]:
    """Per band, the built structure of each path.

    The analogue of :func:`tdarts.genotype.path_structure_keys`, which cannot be
    used here: it parses a ``"<op>_rf<int>"`` candidate string from the
    14-candidate registry, and it reads ``structure_key``, an attribute the
    V2/E families do not carry.  Built from real operators rather than from the
    file's strings, so the recorded key is evidence rather than a restatement.
    """

    keys: dict[str, tuple[tuple, ...]] = {}
    for band in C.BANDS:
        per_path = []
        for gene in genotype.paths_of(band):
            op = build_arm_b_operator(
                gene.family, band=band, target_rf=gene.target_rf, out_channels=6
            )
            kernel, dilation, _ = op.support
            per_path.append((gene.family, int(kernel), int(dilation)))
        keys[band] = tuple(per_path)
    return keys


def band_duplicate_bands(genotype: BandGenotype) -> tuple[str, ...]:
    """Bands whose two paths realise the same structure."""

    return tuple(
        band
        for band, keys in band_structure_keys(genotype).items()
        if len(keys) > 1 and len(set(keys)) != len(keys)
    )


class BandDiscreteCell(nn.Module):
    """One band: ``k`` paths at width ``F1 // k``, concatenated and normalised."""

    def __init__(
        self,
        band: str,
        genes: Iterable[BandGene],
        *,
        out_channels: int = C.PATH_CHANNELS * C.NUM_PATHS,
        in_channels: int = C.IN_CHANNELS,
    ) -> None:
        super().__init__()
        genes = tuple(genes)
        if not genes:
            raise ValueError(f"band {band!r}: a cell needs at least one path")
        if len(genes) > MAX_PATHS_PER_BAND:
            raise ValueError(
                f"band {band!r}: {len(genes)} paths exceeds the {MAX_PATHS_PER_BAND} this arm searches"
            )
        if out_channels % len(genes) != 0:
            raise ValueError(
                f"band {band!r}: out_channels={out_channels} is not divisible by {len(genes)} paths"
            )
        self.band = band
        self.genes = genes
        self.out_channels = int(out_channels)
        width = out_channels // len(genes)
        self.paths = nn.ModuleList(
            [
                build_arm_b_operator(
                    gene.family,
                    band=band,
                    target_rf=gene.target_rf,
                    in_channels=in_channels,
                    out_channels=width,
                )
                for gene in genes
            ]
        )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(torch.cat([path(x) for path in self.paths], dim=1))


class BandDiscreteNet(nn.Module):
    """Arm B's searched architecture -> unchanged SCB / LogVar / classifier."""

    def __init__(
        self,
        genotype: BandGenotype,
        *,
        n_electrodes: int = C.NUM_ELECTRODES,
        n_classes: int = C.NUM_CLASSES,
    ) -> None:
        super().__init__()
        self.genotype = genotype
        self.bands = tuple(C.BANDS)
        self.cells = nn.ModuleDict(
            {band: BandDiscreteCell(band, genotype.paths_of(band)) for band in self.bands}
        )
        self.backbone = TemporalBackbone(
            in_channels=C.NUM_FEAT, n_electrodes=n_electrodes, n_classes=n_classes
        )

    def split_bands(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if x.dim() == 5 and x.shape[1] == 1 and x.shape[-1] == C.NUM_BANDS:
            x = torch.squeeze(x.permute((0, 4, 2, 3, 1)), dim=4)
        if x.dim() != 4 or x.shape[1] != C.NUM_BANDS:
            raise ValueError(f"expected [B,9,E,T] or [B,1,E,T,9], got {tuple(x.shape)}")
        return dict(zip(self.bands, torch.split(x, C.NUM_BANDS_PER_GROUP, dim=1)))

    def forward(self, x: torch.Tensor):
        bands = self.split_bands(x)
        temporal = torch.cat([self.cells[band](bands[band]) for band in self.bands], dim=1)
        return self.backbone(temporal)

    def describe(self) -> dict:
        return {
            "genotype": self.genotype.to_dict(),
            "parameters": sum(p.numel() for p in self.parameters()),
        }
