"""One supernet for both hierarchical-search phases, in the FBNAS cell idiom.

The two searches this stage runs have the *same shape* -- four candidates per
band, and a band selects either one of them or two -- so they share this module
and differ only in what a candidate means:

===============  ===========================================  ===
phase            per-band candidate                           m
===============  ===========================================  ===
Phase A          one of four temporal families at RF 57       1
Phase B          the frozen family at one of four RFs         2
===============  ===========================================  ===

Structure follows the frozen baseline's ``Cell`` (``FBNAS/codes/centralRepo/
networks.py:445``): a band holding ``k`` selected candidates runs each at width
``F1 // k`` and concatenates, so the band's output is **always** ``F1 = 12``
channels and its convolutional parameter count is **always** the same --
``3*12*15 == 2*(3*6*15)``.  That invariance is the whole reason upstream uses
``F1//(i+1)`` addressing, and it is what makes a path-count comparison a
comparison of *structure* rather than of capacity.

The one deviation from the vendored family set is :class:`WidthGeneralGatedE`;
see its docstring.  It is injected through ``V2TemporalOp``'s ``registry``
keyword -- the extension point ``build_e_operator`` itself already uses -- so
nothing under ``tdarts/operator_v2e.py`` is modified.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import torch
import torch.nn as nn

from tdarts import config as C
from tdarts.backbone import TemporalBackbone
from tdarts.operator_v2 import V2TemporalOp, build_v2_operator
from tdarts.operator_v2e import (
    E_OPERATOR_REGISTRY,
    BandGatedConvE,
    GatedTemporalConvE,
)

__all__ = [
    "BAND_FAMILIES",
    "BAND_RFS",
    "EXCLUDED_FAMILY",
    "WidthGeneralGatedE",
    "WidthGeneralBandGatedE",
    "ARM_B_REGISTRY",
    "build_arm_b_operator",
    "BandCandidate",
    "family_candidates",
    "rf_candidates",
    "FBNASBandCell",
    "FBNASBandNet",
    "calibrated_scores",
]

#: The families Arm B searches over.  ``local_attention_e`` is excluded by the
#: band probe's verdict: worst validation NLL and the highest early-overfit
#: rate of the five, so a fourth slot spent on it buys nothing.
BAND_FAMILIES: tuple[str, ...] = ("dilated_e", "dynamic_e", "gated_e", "band_gated_e")

#: The receptive-field ladder, identical to upstream's four ``ConvBn`` dilations.
BAND_RFS: tuple[int, ...] = (15, 29, 57, 113)

#: Named so the exclusion is visible in code rather than only in a docstring.
EXCLUDED_FAMILY = "local_attention_e"

#: Phase A compares every family at one common temporal scale.
PHASE_A_RF = 57


class WidthGeneralGatedE(GatedTemporalConvE):
    """``gated_e`` with the gate width tied to ``out_channels``, not a constant.

    The vendored :class:`GatedTemporalConvE` raises unless ``out_channels`` is
    exactly 12, because its ``HIDDEN`` is the module constant
    ``PATH_CHANNELS * NUM_PATHS``.  A band holding two selected paths runs each
    at width ``F1 // 2 = 6``, so this arm needs a width-6 gate.

    At ``out_channels == 12`` this is **not a new operator**: ``HIDDEN``
    evaluates to 12 and every layer is constructed with the same arguments as
    the parent, so the parameter set is identical.  ``tests/test_arm_b_search``
    pins that by comparing state-dict shapes against
    ``build_e_operator("gated_e", ...)``.

    The parent's guard is a *budget* guard, not a correctness one -- its own
    message says a different output width "needs the budget redone".  Tying the
    hidden width to the output width is that redoing: the gate stays
    proportional to the representation it gates.
    """

    op_name = "gated_e"

    def _build_layers(self) -> None:
        hidden = self.out_channels
        kwargs = self.temporal_conv_kwargs()
        self.f = nn.Conv2d(self.in_channels, hidden, **kwargs)
        self.g = nn.Conv2d(self.in_channels, hidden, **kwargs)
        self.proj = nn.Conv2d(hidden, self.out_channels, kernel_size=(1, 1), bias=False)
        self.skip = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=(1, 1), bias=False)


class WidthGeneralBandGatedE(BandGatedConvE):
    """``band_gated_e`` with the gate bottleneck tied to ``out_channels``.

    Same argument as :class:`WidthGeneralGatedE`, and the same guarantee: the
    vendored class pins ``HIDDEN`` to 12, which happens to equal the width it
    was probed at, so binding it to ``out_channels`` is bit-identical there and
    well defined at width 6.

    Unlike ``gated_e`` this family carries no guard -- it builds at any width --
    so the reason to override is not to unlock a width but to keep the family's
    cost independent of how many paths are active.  Left at a fixed 12, a
    two-path band would pay for two 12-wide gate bottlenecks and cost 13.9%
    more than the one-path band; tied, the two are exactly equal.
    """

    op_name = "band_gated_e"

    def _build_layers(self) -> None:
        hidden = self.out_channels
        self.up = nn.Conv2d(self.in_channels, hidden, kernel_size=(1, 1), bias=True)
        self.act = nn.GELU()
        self.down = nn.Conv2d(hidden, self.in_channels, kernel_size=(1, 1), bias=True)
        self.conv = nn.Conv2d(self.in_channels, self.out_channels, **self.temporal_conv_kwargs())


#: The Expressive registry with the two families whose internal widths are
#: otherwise pinned to 12 swapped for their width-general forms.  Every other
#: family is used exactly as probed.
ARM_B_REGISTRY: dict[str, type] = {
    **E_OPERATOR_REGISTRY,
    "gated_e": WidthGeneralGatedE,
    "band_gated_e": WidthGeneralBandGatedE,
}


def build_arm_b_operator(
    family: str,
    band: str,
    target_rf: int,
    in_channels: int = C.IN_CHANNELS,
    out_channels: int = C.PATH_CHANNELS * C.NUM_PATHS,
) -> V2TemporalOp:
    """Build one candidate as a real :class:`V2TemporalOp`.

    Going through the shared wrapper (rather than a look-alike) is what keeps
    ``temporal_support`` and ``count_macs`` reading these operators with the
    same ruler as every earlier generation.
    """

    if family == EXCLUDED_FAMILY:
        raise ValueError(
            f"{EXCLUDED_FAMILY} is not part of this arm's search space; see BAND_FAMILIES"
        )
    return build_v2_operator(
        family,
        band=band,
        target_rf=target_rf,
        in_channels=in_channels,
        out_channels=out_channels,
        registry=ARM_B_REGISTRY,
    )


@dataclass(frozen=True)
class BandCandidate:
    """One selectable structure inside one band.

    Phase A candidates differ by ``family`` at a fixed RF; Phase B candidates
    differ by ``target_rf`` within a frozen family.  Keeping both in one type
    means the cell below never learns which phase it is serving.
    """

    label: str
    family: str
    target_rf: int

    def to_dict(self) -> dict:
        return {"label": self.label, "family": self.family, "target_rf": self.target_rf}


def family_candidates(families: Sequence[str] = BAND_FAMILIES, target_rf: int = PHASE_A_RF) -> tuple[BandCandidate, ...]:
    """Phase A's four candidates: one per family, all at the same RF."""

    return tuple(BandCandidate(label=f, family=f, target_rf=target_rf) for f in families)


def rf_candidates(family: str, rfs: Sequence[int] = BAND_RFS) -> tuple[BandCandidate, ...]:
    """Phase B's four candidates: the frozen family at each RF of the ladder."""

    return tuple(BandCandidate(label=f"{family}_rf{rf}", family=family, target_rf=rf) for rf in rfs)


class FBNASBandCell(nn.Module):
    """One band: a node bank indexed by ``(path count, candidate)``.

    Indexing is upstream's ``(len(path_ids) - 1) * 4 + id`` made explicit: the
    node for a candidate depends on *how many* candidates are selected, because
    that is what sets the per-node width.
    """

    def __init__(
        self,
        band: str,
        candidates: Sequence[BandCandidate],
        *,
        m: int,
        in_channels: int = C.IN_CHANNELS,
        out_channels: int = C.PATH_CHANNELS * C.NUM_PATHS,
        builder: Callable[..., V2TemporalOp] = build_arm_b_operator,
    ) -> None:
        super().__init__()
        if m < 1:
            raise ValueError(f"m must be >= 1, got {m}")
        if m > len(candidates):
            raise ValueError(
                f"m={m} exceeds the {len(candidates)} candidates available for band {band!r}"
            )
        for k in range(1, m + 1):
            if out_channels % k != 0:
                raise ValueError(
                    f"band {band!r}: out_channels={out_channels} is not divisible by "
                    f"k={k}; the FBNAS width arithmetic needs F1 // k to be an integer"
                )
        self.band = band
        self.m = int(m)
        self.out_channels = int(out_channels)
        self.candidates = tuple(candidates)
        self._labels = tuple(candidate.label for candidate in self.candidates)

        # nodes[k - 1][c]: candidate c at the width used when k candidates are
        # active.  Same bank layout as upstream's flat `nodes` list.
        self.nodes = nn.ModuleList(
            nn.ModuleList(
                [
                    builder(
                        candidate.family,
                        band=band,
                        target_rf=candidate.target_rf,
                        in_channels=in_channels,
                        out_channels=self.node_width(k),
                    )
                    for candidate in self.candidates
                ]
            )
            for k in range(1, m + 1)
        )
        # One normaliser per path count, as upstream's `bn_list`.  Sharing one
        # would mix the statistics of a single 12-wide output with those of a
        # concatenation of two 6-wide ones, which are not the same distribution.
        self.bn_list = nn.ModuleList([nn.BatchNorm2d(out_channels) for _ in range(m)])

    def node_width(self, k: int) -> int:
        """Per-node width when ``k`` candidates are active."""

        return self.out_channels // k

    def describe(self) -> dict:
        return {
            "band": self.band,
            "m": self.m,
            "out_channels": self.out_channels,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "node_widths": {str(k): self.node_width(k) for k in range(1, self.m + 1)},
        }

    def forward(self, x: torch.Tensor, selected: Sequence[int]) -> torch.Tensor:
        indices = tuple(int(i) for i in selected)
        k = len(indices)
        if not 1 <= k <= self.m:
            raise ValueError(f"band {self.band!r}: selected {k} candidates, expected 1..{self.m}")
        if len(set(indices)) != k:
            raise ValueError(f"band {self.band!r}: repeated candidate index in {indices}")
        for index in indices:
            if not 0 <= index < len(self.candidates):
                raise ValueError(
                    f"band {self.band!r}: candidate index {index} out of range "
                    f"[0, {len(self.candidates)})"
                )
        nodes = self.nodes[k - 1]
        out = torch.cat([nodes[index](x) for index in indices], dim=1)
        return self.bn_list[k - 1](out)


class FBNASBandNet(nn.Module):
    """Three band cells -> concat -> the unchanged SCB / LogVar / classifier.

    ``forward`` takes the subnet alongside the batch, mirroring upstream's
    ``SuperNet.forward(x, choice)``, so one set of weights serves every
    candidate during both training and the final traversal.
    """

    def __init__(
        self,
        band_candidates: Mapping[str, Sequence[BandCandidate]],
        *,
        m: int,
        in_channels: int = C.IN_CHANNELS,
        out_channels: int = C.PATH_CHANNELS * C.NUM_PATHS,
        n_electrodes: int = C.NUM_ELECTRODES,
        n_classes: int = C.NUM_CLASSES,
        builder: Callable[..., V2TemporalOp] = build_arm_b_operator,
    ) -> None:
        super().__init__()
        missing = [band for band in C.BANDS if band not in band_candidates]
        if missing:
            raise ValueError(f"band_candidates is missing {missing}")
        self.bands = tuple(C.BANDS)
        self.m = int(m)
        self.out_channels = int(out_channels)
        self.band_candidates = {
            band: tuple(band_candidates[band]) for band in self.bands
        }
        self.cells = nn.ModuleDict(
            {
                band: FBNASBandCell(
                    band,
                    self.band_candidates[band],
                    m=m,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    builder=builder,
                )
                for band in self.bands
            }
        )
        self.backbone = TemporalBackbone(
            in_channels=C.NUM_FEAT, n_electrodes=n_electrodes, n_classes=n_classes
        )

    def split_bands(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Split the 9 filter-bank channels into Low/Mid/High groups of 3."""

        if x.dim() == 5 and x.shape[1] == 1 and x.shape[-1] == C.NUM_BANDS:
            x = torch.squeeze(x.permute((0, 4, 2, 3, 1)), dim=4)
        if x.dim() != 4 or x.shape[1] != C.NUM_BANDS:
            raise ValueError(f"expected [B,9,E,T] or [B,1,E,T,9], got {tuple(x.shape)}")
        return dict(zip(self.bands, torch.split(x, C.NUM_BANDS_PER_GROUP, dim=1)))

    def forward(self, x: torch.Tensor, choice: Mapping[str, Sequence[int]]):
        bands = self.split_bands(x)
        temporal = torch.cat(
            [self.cells[band](bands[band], choice[band]) for band in self.bands], dim=1
        )
        return self.backbone(temporal)

    def describe(self) -> dict:
        return {
            "m": self.m,
            "out_channels": self.out_channels,
            "cells": [self.cells[band].describe() for band in self.bands],
            "parameters": sum(p.numel() for p in self.parameters()),
        }


@torch.no_grad()
def calibrated_scores(
    model: FBNASBandNet,
    choices: Sequence[Mapping[str, Sequence[int]]],
    val_inputs: torch.Tensor,
    val_targets: torch.Tensor,
    *,
    restore_from: Mapping[str, torch.Tensor] | None = None,
    criterion: nn.Module | None = None,
    device: torch.device | str | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> list[dict]:
    """Score every candidate the way upstream's traversal does.

    For each candidate: restore the supernet weights, run one **train-mode**
    forward over the whole validation split so the BatchNorms re-estimate their
    statistics under *that* subnet, then measure in eval mode.  Restoring
    matters -- the calibration pass mutates running statistics, so a candidate
    scored without it would inherit the previous candidate's calibration.

    Returns one row per candidate, in the input order.  Both accuracy and NLL
    are reported: this arm selects on accuracy (to match upstream's criterion
    exactly) and carries NLL as a sensitivity read.
    """

    if restore_from is None:
        restore_from = copy.deepcopy(model.state_dict())
    if criterion is None:
        criterion = nn.NLLLoss()
    if device is not None:
        model.to(device)
    target = next(model.parameters()).device

    inputs = val_inputs.to(target)
    targets = val_targets.to(target)
    rows: list[dict] = []
    total = len(choices)
    for position, choice in enumerate(choices):
        model.load_state_dict(restore_from)
        model.train()
        model(inputs, choice)
        model.eval()
        logits, _ = model(inputs, choice)
        loss = criterion(logits, targets)
        predictions = logits.argmax(dim=1)
        rows.append(
            {
                "index": position,
                "choice": {band: list(choice[band]) for band in C.BANDS},
                "accuracy": float((predictions == targets).float().mean().item()),
                "nll": float(loss.item()),
            }
        )
        if progress is not None:
            progress(position + 1, total)
    return rows
