"""Mixed temporal operations and the two-path temporal cell (stage 2).

Stage 2 builds the *architecture* that a DARTS search will later optimise.  It
deliberately does **not** implement the search: there is no architecture step, no
second optimiser, no genotype, no retrain.  The mixture weights exist as
parameters and start (near-)uniform, so everything downstream can be built and
verified against a well-defined forward pass first.

What this module composes
-------------------------
::

    band input [B, 3, C, T]
        |
        +-- Path A: MixedTemporalOp  -> [B, 6, C, T]
        +-- Path B: MixedTemporalOp  -> [B, 6, C, T]
        |
        concat + BatchNorm2d(12)          -> [B, 12, C, T]

Three such cells (Low / Mid / High) concatenate to ``[B, 36, C, T]``, which is
exactly what the unchanged backbone's SCB expects.

The mixture
-----------
Each path holds the 14 *distinct* candidates for its band, so a future softmax
over ``alpha`` cannot hand one function family two shares of the probability
mass.  ``y = sum_i softmax(alpha)_i * O_i(x)``.

Every candidate already carries its own ``BatchNorm2d(6, affine=False)`` from
stage 1, so the summands are scale-comparable before they are mixed.

Six independent architecture containers
--------------------------------------
``alpha`` is split per band *and* per path -- 6 tensors of 14 logits, 84 numbers
in total.  Paths must not share weights: they are meant to be able to select
different operators, and a shared container would make ``A == B`` structurally
rather than incidentally.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from tdarts import config as C
from tdarts.temporal_ops import (
    TemporalOp,
    build_all_candidates,
    canonical_candidates,
)

__all__ = [
    "MixedTemporalOp",
    "TwoPathTemporalCell",
    "TemporalDARTSNet",
    "candidate_names",
    "NUM_CANDIDATES_PER_BAND",
]

#: Distinct structures per band: 4 operators x 4 RFs minus the 2 RF15 aliases.
NUM_CANDIDATES_PER_BAND = 14


def candidate_names(band: str) -> list[str]:
    """Human-readable candidate labels for one band, in mixture order."""
    return [f"{op}_rf{rf}" for _, op, rf in canonical_candidates(band)]


class MixedTemporalOp(nn.Module):
    """Differentiable mixture over one band's canonical temporal candidates.

    Parameters
    ----------
    band:
        ``"Low"``, ``"Mid"`` or ``"High"``.
    in_channels, out_channels:
        Channel counts for each candidate.  3 -> 6 by default.
    use_candidate_norm:
        Passed to every candidate.  Keep ``True`` for real use; the RF audit
        needs ``False``.
    alpha_init_scale:
        Standard deviation of the initial architecture logits.  Zero would be
        exactly uniform.

    Attributes
    ----------
    op_names:
        ``[(op_name, target_rf), ...]`` -- the mixture order.  ``alpha[i]``
        corresponds to ``ops[i]``, and this list is the single place that
        mapping is defined.
    alpha:
        ``nn.Parameter`` of shape ``(14,)``.  It is the only architecture
        parameter in the model; everything else is a network weight.
    """

    def __init__(
        self,
        band: str,
        in_channels: int = C.IN_CHANNELS,
        out_channels: int = C.PATH_CHANNELS,
        use_candidate_norm: bool = True,
        alpha_init_scale: float = C.ALPHA_INIT_SCALE,
    ):
        super().__init__()
        if band not in C.BANDS:
            raise ValueError(
                f"unknown band {band!r}; expected one of {list(C.BANDS)}"
            )
        self.band = band
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.op_names: list[tuple[str, int]] = [
            (op_name, rf) for _, op_name, rf in canonical_candidates(band)
        ]
        if len(self.op_names) != NUM_CANDIDATES_PER_BAND:
            raise RuntimeError(
                f"band {band!r} yielded {len(self.op_names)} candidates; "
                f"expected {NUM_CANDIDATES_PER_BAND}"
            )

        pool = build_all_candidates(use_norm=use_candidate_norm)
        self.ops = nn.ModuleList(
            [pool[(band, op_name, rf)] for op_name, rf in self.op_names]
        )

        # The single architecture parameter.  Named `alpha` so a future
        # optimiser can find it without reflection.
        self.alpha = nn.Parameter(
            float(alpha_init_scale) * torch.randn(len(self.op_names))
        )

    # -- introspection ---------------------------------------------------
    @property
    def num_candidates(self) -> int:
        return len(self.op_names)

    def architecture_parameters(self):
        """The architecture logits, and nothing else."""
        return [self.alpha]

    def network_parameters(self):
        """Everything except the architecture logits."""
        for name, param in self.named_parameters():
            if name != "alpha":
                yield param

    def mixture_weights(self) -> torch.Tensor:
        """``softmax(alpha)`` -- the mixing coefficients."""
        return torch.softmax(self.alpha, dim=0)

    def candidate_outputs(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Per-candidate outputs, for tests and for one-hot equivalence checks.

        Not used by :meth:`forward`, which fuses the loop; allocating 14
        activations at once for a 1000-sample input is wasteful in training.
        """
        return [op(x) for op in self.ops]

    def describe(self) -> str:
        weights = self.mixture_weights().detach()
        lines = [
            f"MixedTemporalOp band={self.band} candidates={self.num_candidates}"
        ]
        for (op_name, rf), w in zip(self.op_names, weights.tolist()):
            lines.append(f"  {op_name:<8} RF{rf:<4} p={w:.6f}")
        return "\n".join(lines)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = self.mixture_weights()
        out = None
        for weight, op in zip(weights, self.ops):
            term = weight * op(x)
            out = term if out is None else out + term
        if out is None:  # pragma: no cover - construction guarantees non-empty
            raise RuntimeError("MixedTemporalOp has no candidates")
        return out


class TwoPathTemporalCell(nn.Module):
    """Two independent mixtures over one band's input, concatenated.

    ``[B, 3, C, T]`` -> two ``[B, 6, C, T]`` -> concat -> ``BatchNorm2d(12)``
    -> ``[B, 12, C, T]``.

    The two paths are separate :class:`MixedTemporalOp` instances with separate
    ``alpha`` containers, so they can select different operators.  The trailing
    BatchNorm matches the official FBNAS cell, which likewise normalises the
    concatenated output (``shadow_bn`` in ``FBNASCell``).

    There is no ``path_ids``, no ``len(path_ids)`` arithmetic and no node
    indexing: the width is a constructor argument, not something inferred from
    how many paths happen to be selected.
    """

    def __init__(
        self,
        band: str,
        num_paths: int = C.NUM_PATHS,
        in_channels: int = C.IN_CHANNELS,
        path_channels: int = C.PATH_CHANNELS,
        use_candidate_norm: bool = True,
        alpha_init_scale: float = C.ALPHA_INIT_SCALE,
    ):
        super().__init__()
        if num_paths < 1:
            raise ValueError(f"num_paths must be >= 1, got {num_paths}")
        self.band = band
        self.num_paths = int(num_paths)
        self.path_channels = int(path_channels)

        self.paths = nn.ModuleList(
            [
                MixedTemporalOp(
                    band=band,
                    in_channels=in_channels,
                    out_channels=path_channels,
                    use_candidate_norm=use_candidate_norm,
                    alpha_init_scale=alpha_init_scale,
                )
                for _ in range(self.num_paths)
            ]
        )
        self.out_channels = self.num_paths * self.path_channels
        self.bn = nn.BatchNorm2d(self.out_channels)

    def architecture_parameters(self):
        return [p for path in self.paths for p in path.architecture_parameters()]

    def network_parameters(self):
        for name, param in self.named_parameters():
            if not name.endswith("alpha"):
                yield param

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(torch.cat([path(x) for path in self.paths], dim=1))


class TemporalDARTSNet(nn.Module):
    """Filter-bank split -> per-band two-path cells -> unchanged backbone.

    ::

        [B, 9, C, T]
          split into 3 bands of 3 channels
          Low / Mid / High cells (independent weights)
          concat -> [B, 36, C, T]
          backbone (SCB -> LogVar -> classifier) -> [B, 4]

    The 9 filter-bank channels are grouped by index: Low = 0-2, Mid = 3-5,
    High = 6-8.  That matches the official FBNAS ``torch.split(x, 3, 1)``.

    ``backbone`` is the stage-1 :class:`tdarts.backbone.TemporalBackbone`, which
    is numerically equivalent to the official FBNAS SCB / LogVar / classifier.
    Nothing in it is modified to accommodate the new temporal stage.
    """

    def __init__(
        self,
        n_electrodes: int = C.NUM_ELECTRODES,
        n_classes: int = C.NUM_CLASSES,
        num_paths: int = C.NUM_PATHS,
        in_channels: int = C.IN_CHANNELS,
        path_channels: int = C.PATH_CHANNELS,
        use_candidate_norm: bool = True,
        alpha_init_scale: float = C.ALPHA_INIT_SCALE,
        backbone: nn.Module | None = None,
    ):
        super().__init__()
        self.bands = tuple(C.BANDS)
        self.in_channels = in_channels
        self.path_channels = path_channels
        self.num_paths = num_paths
        self.n_electrodes = n_electrodes

        self.cells = nn.ModuleDict(
            {
                band: TwoPathTemporalCell(
                    band=band,
                    num_paths=num_paths,
                    in_channels=in_channels,
                    path_channels=path_channels,
                    use_candidate_norm=use_candidate_norm,
                    alpha_init_scale=alpha_init_scale,
                )
                for band in self.bands
            }
        )
        self.temporal_out_channels = len(self.bands) * num_paths * path_channels

        if backbone is None:
            # Imported lazily so that tdarts.backbone does not have to import
            # this module (avoids a cycle and keeps stage 1 standalone).
            from tdarts.backbone import TemporalBackbone

            backbone = TemporalBackbone(
                in_channels=self.temporal_out_channels,
                n_electrodes=n_electrodes,
                n_classes=n_classes,
            )
        self.backbone = backbone

    # -- architecture / weight split -------------------------------------
    def arch_parameters(self):
        """The 6 alpha containers (2 paths x 3 bands), as a flat list."""
        return [p for cell in self.cells.values() for p in cell.architecture_parameters()]

    def network_parameters(self):
        """All network weights, alpha excluded."""
        arch_ids = {id(p) for p in self.arch_parameters()}
        for param in self.parameters():
            if id(param) not in arch_ids:
                yield param

    def alphas(self) -> dict[tuple[str, int], torch.Tensor]:
        """``{(band, path_index): alpha}`` for inspection and tests."""
        return {
            (band, i): cell.paths[i].alpha
            for band, cell in self.cells.items()
            for i in range(cell.num_paths)
        }

    def num_arch_parameters(self) -> int:
        return sum(p.numel() for p in self.arch_parameters())

    # -- forward ---------------------------------------------------------
    def split_bands(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Split ``[B, 9, C, T]`` into three ``[B, 3, C, T]`` band tensors.

        Also accepts the official FBNAS 5-D multiview layout
        ``[B, 1, C, T, 9]``, which it permutes the same way the upstream code
        does (``x.permute((0, 4, 2, 3, 1))`` then squeeze).
        """
        if x.dim() == 5 and x.shape[1] == 1 and x.shape[4] == len(self.bands) * self.in_channels:
            x = torch.squeeze(x.permute((0, 4, 2, 3, 1)), dim=4)

        if x.dim() != 4:
            raise ValueError(
                f"TemporalDARTSNet expects [B, {len(self.bands) * self.in_channels}, C, T] "
                f"or [B, 1, C, T, {len(self.bands) * self.in_channels}], "
                f"got shape {tuple(x.shape)}"
            )

        expected = len(self.bands) * self.in_channels
        if x.shape[1] != expected:
            raise ValueError(
                f"expected {expected} filter-bank channels "
                f"({len(self.bands)} bands x {self.in_channels} per band), "
                f"got {x.shape[1]}"
            )
        chunks = torch.split(x, self.in_channels, dim=1)
        return dict(zip(self.bands, chunks))

    def forward(self, x: torch.Tensor):
        bands = self.split_bands(x)
        temporal = torch.cat(
            [self.cells[band](bands[band]) for band in self.bands], dim=1
        )
        if temporal.shape[1] != self.temporal_out_channels:
            raise RuntimeError(
                f"temporal stage produced {temporal.shape[1]} channels, "
                f"expected {self.temporal_out_channels}"
            )
        logits, features = self.backbone(temporal)
        return logits, features

    def describe(self) -> str:
        lines = [
            f"TemporalDARTSNet  input=[B,{len(self.bands) * self.in_channels},"
            f"{self.n_electrodes},T]  temporal_out={self.temporal_out_channels}",
            f"  cells        : {', '.join(self.bands)}",
            f"  paths/band   : {self.num_paths}",
            f"  candidates   : {NUM_CANDIDATES_PER_BAND} per path",
            f"  arch params  : {self.num_arch_parameters()} "
            f"({len(self.arch_parameters())} containers x {NUM_CANDIDATES_PER_BAND})",
            f"  net params   : {sum(p.numel() for p in self.network_parameters())}",
        ]
        return "\n".join(lines)
