"""Standalone network for the Operator Separability V2 pilot.

One operator family per band, everything downstream unchanged
------------------------------------------------------------
The searched model (``tdarts.discrete_network.TemporalDiscreteNet``) puts two
6-channel temporal paths in each band and concatenates them to 12.  This pilot
replaces that pair with a single 12-channel path, so the tensor reaching the
backbone is the same ``[B, 36, E, T]`` and SCB / LogVar / classifier are the
same modules with the same weights.  The only difference between a V2 run and a
searched run is then the operator family, which is the whole point.

The cell mirrors ``DiscreteTemporalCell`` layer for layer: candidate
normalisation (``BatchNorm2d(12, affine=False)``, no trainable parameters, from
``V2TemporalOp``) followed by the cell's own ``BatchNorm2d(12, affine=True)``.
Keeping both means the parameter count and the normalisation depth match the
searched model rather than approximating it.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from tdarts import config as C
from tdarts.backbone import TemporalBackbone
from tdarts.operator_v2 import V2TemporalOp, build_v2_operator

__all__ = ["OperatorV2Cell", "OperatorV2Net", "V2_STANDALONE_PATH_CHANNELS"]


#: A standalone cell emits one 12-channel path per band, matching the two
#: 6-channel paths the searched model concatenates.  The dual-path variant this
#: pilot does not yet build would use ``PATH_CHANNELS`` (6) per path instead.
V2_STANDALONE_PATH_CHANNELS = C.PATH_CHANNELS * C.NUM_PATHS


class OperatorV2Cell(nn.Module):
    """One band: a single operator family, normalised like a searched cell."""

    def __init__(
        self,
        band: str,
        op_name: str,
        *,
        target_rf: int,
        in_channels: int = C.IN_CHANNELS,
        out_channels: int = V2_STANDALONE_PATH_CHANNELS,
    ) -> None:
        super().__init__()
        self.band = band
        self.op_name = op_name
        self.target_rf = int(target_rf)
        self.path: V2TemporalOp = build_v2_operator(
            op_name,
            band=band,
            target_rf=target_rf,
            in_channels=in_channels,
            out_channels=out_channels,
        )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(self.path(x))

    def describe(self) -> dict:
        positions, spacing, span = self.path.support
        return {
            "band": self.band,
            "operator": self.op_name,
            "target_rf": self.target_rf,
            "support_positions": positions,
            "support_spacing": spacing,
            "support_span": span,
            "path_params": self.path.num_params,
            "operator_params": self.path.num_op_params,
        }


class OperatorV2Net(nn.Module):
    """Three band cells, then the unchanged FBNAS-compatible backbone."""

    def __init__(
        self,
        op_name: str,
        *,
        target_rf: int = 57,
        n_electrodes: int = C.NUM_ELECTRODES,
        n_classes: int = C.NUM_CLASSES,
    ) -> None:
        super().__init__()
        self.op_name = op_name
        self.target_rf = int(target_rf)
        self.bands = tuple(C.BANDS)
        self.cells = nn.ModuleDict(
            {
                band: OperatorV2Cell(band, op_name, target_rf=self.target_rf)
                for band in self.bands
            }
        )
        self.backbone = TemporalBackbone(
            in_channels=C.NUM_FEAT, n_electrodes=n_electrodes, n_classes=n_classes
        )

    def split_bands(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Split the 9 filter-bank channels into Low/Mid/High groups of 3.

        Same layout handling as ``TemporalDiscreteNet.split_bands``: the loader
        hands trials over as ``[B, 1, E, T, 9]`` and the model wants
        ``[B, 9, E, T]``.
        """
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
            "operator": self.op_name,
            "target_rf": self.target_rf,
            "cells": [self.cells[band].describe() for band in self.bands],
            "parameters": sum(p.numel() for p in self.parameters()),
        }
