"""Fixed temporal networks decoded from Stage-3 DARTS genotypes.

This module contains no alpha parameters and no 14-way candidate pools.  Each
of the six paths owns exactly one selected :class:`TemporalOp`; the downstream
FBNAS-compatible backbone is unchanged.
"""

from __future__ import annotations

from dataclasses import asdict

import torch
import torch.nn as nn

from tdarts import config as C
from tdarts.backbone import TemporalBackbone
from tdarts.genotype import Genotype, PathGene
from tdarts.temporal_ops import build_temporal_op

__all__ = ["DiscreteTemporalCell", "TemporalDiscreteNet", "transfer_supernet_weights"]


class DiscreteTemporalCell(nn.Module):
    def __init__(self, band: str, genes: tuple[PathGene, PathGene]):
        super().__init__()
        if any(gene.band != band for gene in genes):
            raise ValueError(f"all genes in {band} cell must belong to that band")
        if tuple(gene.path for gene in genes) != (0, 1):
            raise ValueError("a discrete cell needs exactly paths zero and one")
        self.band = band
        self.genes = genes
        self.paths = nn.ModuleList()
        for gene in genes:
            op_name, rf_text = gene.candidate.rsplit("_rf", 1)
            self.paths.append(build_temporal_op(op_name, band, int(rf_text)))
        self.bn = nn.BatchNorm2d(2 * C.PATH_CHANNELS)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(torch.cat([path(x) for path in self.paths], dim=1))


class TemporalDiscreteNet(nn.Module):
    """Six selected temporal paths -> unchanged SCB / LogVar / classifier."""

    def __init__(self, genotype: Genotype, *, n_electrodes: int = C.NUM_ELECTRODES, n_classes: int = C.NUM_CLASSES):
        super().__init__()
        self.genotype = genotype
        self.bands = tuple(C.BANDS)
        self.cells = nn.ModuleDict(
            {
                band: DiscreteTemporalCell(
                    band,
                    (genotype.gene(band, 0), genotype.gene(band, 1)),
                )
                for band in self.bands
            }
        )
        self.backbone = TemporalBackbone(in_channels=C.NUM_FEAT, n_electrodes=n_electrodes, n_classes=n_classes)

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
        return {"genotype": self.genotype.to_dict(), "parameters": sum(p.numel() for p in self.parameters())}


def transfer_supernet_weights(model: TemporalDiscreteNet, checkpoint: str | dict) -> list[str]:
    """Transfer selected candidate, Cell-BN, SCB and classifier state exactly.

    ``checkpoint`` is a Stage-3 ``checkpoint_epoch_200.pt`` path or its loaded
    mapping.  The return value is every destination state key copied, making
    coverage auditable in tests and training manifests.
    """

    if isinstance(checkpoint, str):
        checkpoint = torch.load(checkpoint, map_location="cpu")
    source = checkpoint["model_state_dict"]
    destination = model.state_dict()
    copied: dict[str, torch.Tensor] = {}
    for key in destination:
        source_key = key
        parts = key.split(".")
        # cells.Low.paths.0.<op-key> -> cells.Low.paths.0.ops.<selected-index>.<op-key>
        if len(parts) >= 5 and parts[0] == "cells" and parts[2] == "paths":
            band, path_index = parts[1], int(parts[3])
            gene = model.genotype.gene(band, path_index)
            source_key = ".".join(parts[:4] + ["ops", str(gene.candidate_index)] + parts[4:])
        if source_key not in source:
            raise KeyError(f"missing source tensor {source_key} for {key}")
        if source[source_key].shape != destination[key].shape:
            raise ValueError(f"shape mismatch {source_key}: {source[source_key].shape} vs {key}: {destination[key].shape}")
        copied[key] = source[source_key]
    model.load_state_dict(copied, strict=True)
    return sorted(copied)
