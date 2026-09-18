"""Hierarchical Phase A: hard (Gumbel) operator search at a fixed RF.

The full method searches the convolution mechanism first, then the receptive
field.  This module implements the first stage:

* the receptive field is fixed at RF 57, so every candidate competes at one
  identical temporal scale -- no RF schedule can drift the operator ranking;
* path 0 stays the ``dilated`` anchor, path 1 samples one of the four operator
  families at each training step with a straight-through Gumbel-softmax draw,
  so the forward pass is a single operator rather than a four-way mixture and
  the architecture softmax cannot hedge at high entropy;
* ``beta`` (4 logits per band) is the only architecture parameter, updated by
  the same first-order :class:`tdarts.architect.SearchArchitect` used elsewhere.

Phase B (RF search over the frozen operators) lives in :mod:`tdarts.anchored`
and is driven by ``train_hierarchical_search.py`` after this stage finishes.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn

from tdarts import config as C
from tdarts.anchored import OperatorPool
from tdarts.backbone import TemporalBackbone
from tdarts.temporal_ops import build_temporal_op

__all__ = [
    "PHASE_A_RF",
    "gumbel_one_hot",
    "HardOperatorCell",
    "HardOperatorNet",
]

#: Phase A fixes the effective receptive field for every candidate.
PHASE_A_RF = 57


def gumbel_one_hot(logits: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
    """Straight-through Gumbel-softmax one-hot weights.

    Forward is a hard one-hot draw; backward flows through the relaxed softmax,
    so ``logits`` receives gradients.  Standard Gumbel-DARTS estimator.
    """

    if tau <= 0:
        raise ValueError(f"tau must be positive, got {tau}")
    gumbels = -torch.empty_like(logits).exponential_().log()
    relaxed = ((logits + gumbels) / tau).softmax(dim=-1)
    index = relaxed.argmax(dim=-1, keepdim=True)
    hard = torch.zeros_like(relaxed).scatter_(-1, index, 1.0)
    return (hard - relaxed).detach() + relaxed


class HardOperatorCell(nn.Module):
    """One band: fixed dilated anchor + one sampled operator from four families.

    All candidates sit at :data:`PHASE_A_RF`, so a selection is a mechanism
    choice at a fixed scale.  ``sampling`` controls whether training forwards
    draw with Gumbel (True) or use the soft mixture (False, used by tests and
    for a mixture reference); evaluation can additionally be forced to the
    argmax operator via :meth:`set_hard_eval`.
    """

    def __init__(
        self,
        band: str,
        in_channels: int = C.IN_CHANNELS,
        path_channels: int = C.PATH_CHANNELS,
        rf: int = PHASE_A_RF,
        use_candidate_norm: bool = C.USE_CANDIDATE_NORM,
        alpha_init_scale: float = C.ALPHA_INIT_SCALE,
        sampling: bool = True,
    ):
        super().__init__()
        if band not in C.BANDS:
            raise ValueError(f"unknown band {band!r}")
        if rf != PHASE_A_RF:
            raise ValueError(
                f"phase A fixes RF {PHASE_A_RF} for every candidate; got {rf}"
            )
        self.band = band
        self.rf = int(rf)
        self.sampling = bool(sampling)
        self.tau = 1.0
        self.hard_eval = False

        self.anchor = build_temporal_op(
            "dilated",
            band,
            self.rf,
            in_channels=in_channels,
            out_channels=path_channels,
            use_norm=use_candidate_norm,
        )
        self.pool = OperatorPool(
            band,
            self.rf,
            in_channels=in_channels,
            out_channels=path_channels,
            use_candidate_norm=use_candidate_norm,
        )
        self.beta = nn.Parameter(
            float(alpha_init_scale) * torch.randn(len(C.OPERATORS))
        )
        self.register_buffer(
            "selection_counts", torch.zeros(len(C.OPERATORS), dtype=torch.long)
        )
        self.bn = nn.BatchNorm2d(2 * path_channels)

    # -- controls ---------------------------------------------------------
    def set_tau(self, tau: float) -> None:
        self.tau = float(tau)

    def set_hard_eval(self, flag: bool) -> None:
        self.hard_eval = bool(flag)

    def reset_counts(self) -> None:
        self.selection_counts.zero_()

    # -- architecture -----------------------------------------------------
    def architecture_parameters(self):
        return [self.beta]

    def operator_weights(self) -> torch.Tensor:
        return torch.softmax(self.beta, dim=0)

    def selected_operator(self) -> str:
        return C.OPERATORS[int(self.operator_weights().detach().argmax())]

    # -- forward ----------------------------------------------------------
    def _weights(self) -> torch.Tensor:
        if self.hard_eval:
            with torch.no_grad():
                weights = torch.zeros_like(self.beta)
                weights[int(self.beta.detach().argmax())] = 1.0
            return weights
        if self.training and self.sampling:
            weights = gumbel_one_hot(self.beta, tau=self.tau)
            with torch.no_grad():
                self.selection_counts[int(weights.detach().argmax())] += 1
            return weights
        return self.operator_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        anchor = self.anchor(x)
        searched = self.pool(x, self._weights())
        return self.bn(torch.cat([anchor, searched], dim=1))


class HardOperatorNet(nn.Module):
    """Three hard operator cells over the fixed RF, feeding the fixed backbone.

    Exposes the same ``arch_parameters()`` / ``network_parameters()`` contract
    as the other search supernets, so :class:`SearchArchitect` drives it with
    no changes.  Architecture parameters: 3 bands x 4 betas = 12 numbers.
    """

    def __init__(
        self,
        n_electrodes: int = C.NUM_ELECTRODES,
        n_classes: int = C.NUM_CLASSES,
        in_channels: int = C.IN_CHANNELS,
        path_channels: int = C.PATH_CHANNELS,
        use_candidate_norm: bool = C.USE_CANDIDATE_NORM,
        alpha_init_scale: float = C.ALPHA_INIT_SCALE,
        sampling: bool = True,
        backbone: nn.Module | None = None,
    ):
        super().__init__()
        self.bands = tuple(C.BANDS)
        self.in_channels = in_channels
        self.path_channels = path_channels
        self.n_electrodes = n_electrodes
        self.cells = nn.ModuleDict(
            {
                band: HardOperatorCell(
                    band,
                    in_channels=in_channels,
                    path_channels=path_channels,
                    use_candidate_norm=use_candidate_norm,
                    alpha_init_scale=alpha_init_scale,
                    sampling=sampling,
                )
                for band in self.bands
            }
        )
        self.temporal_out_channels = len(self.bands) * 2 * path_channels
        if backbone is None:
            backbone = TemporalBackbone(
                in_channels=self.temporal_out_channels,
                n_electrodes=n_electrodes,
                n_classes=n_classes,
            )
        self.backbone = backbone

    # -- parameters -------------------------------------------------------
    def arch_parameters(self):
        return [cell.beta for cell in self.cells.values()]

    def network_parameters(self):
        arch_ids = {id(parameter) for parameter in self.arch_parameters()}
        for parameter in self.parameters():
            if id(parameter) not in arch_ids:
                yield parameter

    def num_arch_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.arch_parameters())

    def set_tau(self, tau: float) -> None:
        for cell in self.cells.values():
            cell.set_tau(tau)

    def set_hard_eval(self, flag: bool) -> None:
        for cell in self.cells.values():
            cell.set_hard_eval(flag)

    def selected_operators(self) -> dict[str, str]:
        return {band: cell.selected_operator() for band, cell in self.cells.items()}

    def selection_counts(self) -> dict[str, list[int]]:
        return {
            band: cell.selection_counts.tolist() for band, cell in self.cells.items()
        }

    # -- forward ----------------------------------------------------------
    def split_bands(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if x.dim() == 5 and x.shape[1] == 1 and x.shape[-1] == len(self.bands) * self.in_channels:
            x = torch.squeeze(x.permute((0, 4, 2, 3, 1)), dim=4)
        if x.dim() != 4 or x.shape[1] != len(self.bands) * self.in_channels:
            raise ValueError(
                f"expected [B, {len(self.bands) * self.in_channels}, C, T] or the "
                f"5-D official layout, got {tuple(x.shape)}"
            )
        return dict(zip(self.bands, torch.split(x, self.in_channels, dim=1)))

    def forward(self, x: torch.Tensor):
        bands = self.split_bands(x)
        temporal = torch.cat([self.cells[band](bands[band]) for band in self.bands], dim=1)
        return self.backbone(temporal)

    def describe(self) -> str:
        lines = [
            "HardOperatorNet  phase A (fixed RF, Gumbel one-hot)",
            f"  rf           : {PHASE_A_RF}",
            f"  bands        : {', '.join(self.bands)}",
            f"  candidates   : {len(C.OPERATORS)} operators x {PHASE_A_RF}",
            f"  arch params  : {self.num_arch_parameters()} "
            f"({len(self.arch_parameters())} containers x {len(C.OPERATORS)})",
            f"  net params   : {sum(p.numel() for p in self.network_parameters())}",
        ]
        return "\n".join(lines)
