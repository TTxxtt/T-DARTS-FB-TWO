"""Network for the band-specific mechanism probe: one family per band.

The Expressive network picks a single family and uses it in all three bands.  A
probe has to name a family per band, so the cells cannot come from one
``op_name``.  Everything the two networks share -- the three cells in band
order, the 12-channel path, the ``[B, 3, C, T] -> [B, 12, C, T]`` contract, the
concat to ``[B, 36, E, T]`` and the untouched FBNAS backbone -- is *inherited*,
so ``split_bands`` and ``forward`` are the same code object on both.

The layout is nonetheless spelled out once more in :meth:`__init__`, because
the parent's constructor takes one name and this one takes a mapping; there is
no way to reach the parent's cell loop with three names.  That copy is the one
thing here that could drift, so it is guarded mechanically rather than by
comment: ``tests/test_operator_v2e_band.py`` builds the all-anchor band network
and the Expressive anchor network under one seed and asserts their
``state_dict()`` tensors are bit-identical.  A drifting cell order, builder,
channel count or backbone argument changes the RNG draw and turns that test
red.

Why the parent's ``__init__`` is skipped rather than called with a placeholder
--------------------------------------------------------------------------
Calling ``OperatorV2Net.__init__`` with one family and then rebuilding the two
other cells would draw the initial weights twice: every ``nn.Conv2d.__init__``
consumes the global torch RNG, so the resulting network would not be the one
its own ``--seed`` describes.  That is the same class of bug the entry points'
seeding clause exists to prevent, so it is avoided structurally here.
"""

from __future__ import annotations

from typing import Mapping

import torch
import torch.nn as nn

from tdarts import config as C
from tdarts.backbone import TemporalBackbone
from tdarts.operator_v2_network import (
    OperatorV2Cell,
    OperatorV2Net,
    V2_STANDALONE_PATH_CHANNELS,
)
from tdarts.operator_v2e import build_e_operator
from tdarts.operator_v2e_band import slug_for, validate_band_families

__all__ = ["OperatorV2EBandNet"]


class OperatorV2EBandNet(OperatorV2Net):
    """Three band cells, each built from its own Expressive family."""

    def __init__(
        self,
        band_ops: Mapping[str, str],
        *,
        target_rf: int = 57,
        n_electrodes: int = C.NUM_ELECTRODES,
        n_classes: int = C.NUM_CLASSES,
    ) -> None:
        # The band family is a per-band choice, so it is the vocabulary module
        # that decides what is legal -- not this class.  In particular the
        # single-varying-band rule is enforced before a module is built, so a
        # malformed configuration cannot produce a network whose summary claims
        # something the code did not do.
        ordered = {band: str(band_ops[band]) for band in C.BANDS}
        validate_band_families(ordered["Low"], ordered["Mid"], ordered["High"])
        # See the module docstring: the parent's parameterised constructor is
        # deliberately bypassed, so nn.Module's own initialisation is called
        # directly rather than through it.
        nn.Module.__init__(self)
        self.band_ops = ordered
        # The slug spelling lives in the vocabulary module so the network, the
        # entry point and the analyzer cannot disagree about what a run is.
        self.op_name = slug_for(ordered["Low"], ordered["Mid"], ordered["High"])
        self.target_rf = int(target_rf)
        self.bands = tuple(C.BANDS)
        # Same order, same constructor, same builder argument as
        # ``OperatorV2Net.__init__``: for an all-anchor configuration the two
        # networks are the same function under the same seed.  Asserted, not
        # assumed.
        self.cells = nn.ModuleDict(
            {
                band: OperatorV2Cell(
                    band, ordered[band], target_rf=self.target_rf, builder=build_e_operator
                )
                for band in self.bands
            }
        )
        self.backbone = TemporalBackbone(
            in_channels=C.NUM_FEAT, n_electrodes=n_electrodes, n_classes=n_classes
        )

    def describe(self) -> dict:
        return {
            "operator": self.op_name,
            "band_families": dict(self.band_ops),
            "target_rf": self.target_rf,
            "cells": [self.cells[band].describe() for band in self.bands],
            "path_channels": V2_STANDALONE_PATH_CHANNELS,
            "parameters": sum(p.numel() for p in self.parameters()),
        }
