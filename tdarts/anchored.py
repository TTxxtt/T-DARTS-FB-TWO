"""Anchored two-phase temporal search: a dilated anchor plus a searched path.

This is the "anchored" arm agreed for the next experiment.  It deliberately
does not replace :mod:`tdarts.mixed_op` / ``train_search.py``: the 14-candidate
joint search stays as it is, and this module is a separate search space.

Phase A -- operator search
--------------------------
Path 0 is fixed to the ``dilated`` mechanism, path 1 is searched over the four
operator families.  Both paths share one scheduled receptive field per step:

    rf = OP_SEARCH_RFS[global_step % 3]        # 29, 57, 113

RF 15 is excluded because at dilation 1 ``dilated == normal`` and
``dwsep == lkdw``: the operator family would not be identifiable.  Giving all
four candidates the *same* RF in a step is what keeps operator and scale from
competing against each other.  The anchor's own RF is not fixed: it is only
marginalised by the schedule here, and gets its own logits in phase B.

Phase B -- receptive-field search
---------------------------------
Path 1's operator is frozen to the phase A choice; each path then has four RF
logits over the full ladder 15/29/57/113.  ``dilated``/``dwsep`` realise an RF
through dilation at kernel 15, ``normal``/``lkdw`` through the kernel itself --
that mapping already lives in :mod:`tdarts.temporal_ops`.

With ``no_duplicate_paths=True`` the two paths of a band are selected *jointly*:
the exported pair maximises the summed log-probability subject to the two
``structure_key = (kernel, dilation, separable)`` values being different.  The
comparison therefore uses the actual built geometry, not the candidate name --
at RF 15, ``normal`` is the same dense k15 convolution as ``dilated`` and must
count as a duplicate.  Hard-mode evaluation uses the same constrained pair, so
the reported hard subnet is exactly the exported genotype.

Architecture parameter counts
-----------------------------
* phase A: 3 bands x 1 beta container x 4 = 12 numbers;
* phase B: 3 bands x 2 gamma containers x 4 = 24 numbers.

The model exposes ``arch_parameters()`` / ``network_parameters()`` with the same
contract as :class:`tdarts.mixed_op.TemporalDARTSNet`, so the existing
:class:`tdarts.architect.SearchArchitect` drives either phase unchanged.
"""

from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import torch
import torch.nn as nn

from tdarts import config as C
from tdarts.search import should_update_alphas
from tdarts.temporal_ops import (
    build_all_candidates,
    build_temporal_op,
    canonical_candidates,
)

__all__ = [
    "ANCHORED_SCHEME",
    "ANCHOR_OPERATOR",
    "OP_SEARCH_RFS",
    "RF_SEARCH_SPACE",
    "rf_schedule",
    "canonical_candidate_name",
    "candidate_structure_key",
    "OperatorPool",
    "RFPool",
    "AnchoredOperatorCell",
    "AnchoredRFCell",
    "AnchoredOperatorNet",
    "AnchoredRFNet",
    "inherit_operator_weights",
    "build_anchored_genotype",
    "load_anchored_genotype",
    "operator_probability_summary",
    "rf_probability_summary",
    "run_anchored_epoch",
]

#: Scheme tag written into ``genotype.json`` and required by the loader.
ANCHORED_SCHEME = "anchored_operator_then_rf"

#: Path 0 is always this operator family in the anchored arm.
ANCHOR_OPERATOR = "dilated"

#: Phase A RF ladder.  RF 15 is excluded on purpose (see module docstring).
OP_SEARCH_RFS: tuple[int, ...] = (29, 57, 113)

#: Phase B RF ladder, the full FBNAS ladder.
RF_SEARCH_SPACE: tuple[int, ...] = tuple(C.RF_SPACE["Low"])

NUM_PATH_CHANNELS = C.NUM_PATHS * C.PATH_CHANNELS  # 12


def rf_schedule(step: int) -> int:
    """The deterministic, balanced phase A RF for a global train step.

    Repeating the tuple in order guarantees that over any window that is a
    multiple of three the three RFs are used equally often.  Drawing an RF per
    operator would let operator and scale covary again, which is exactly what
    the schedule exists to prevent.
    """

    if step < 0:
        raise ValueError(f"step must be non-negative, got {step}")
    return OP_SEARCH_RFS[int(step) % len(OP_SEARCH_RFS)]


@lru_cache(maxsize=None)
def _canonical_name_map(band: str) -> dict[tuple[str, int], str]:
    pool = build_all_candidates(use_norm=False)
    canonical = {
        pool[(band, op_name, rf)].structure_key: f"{op_name}_rf{rf}"
        for _, op_name, rf in canonical_candidates(band)
    }
    return {
        (op_name, rf): canonical[pool[(band, op_name, rf)].structure_key]
        for op_name in C.OPERATORS
        for rf in C.RF_SPACE[band]
    }


def canonical_candidate_name(band: str, op_name: str, rf: int) -> str:
    """Canonical 14-candidate name for one ``(operator, RF)`` choice.

    Phase B can select e.g. ``normal`` at RF 15, which is the same convolution
    structure as ``dilated`` at RF 15.  The canonical set keeps only one name
    per structure (:func:`tdarts.temporal_ops.canonical_candidates`), so the
    exported genotype must be normalised here or the same network would be
    recorded under two names across runs.
    """

    try:
        return _canonical_name_map(band)[(op_name, int(rf))]
    except KeyError as exc:
        raise ValueError(
            f"unknown (band, operator, RF) combination: "
            f"{(band, op_name, rf)!r}"
        ) from exc


def candidate_structure_key(band: str, candidate: str) -> tuple[int, int, bool]:
    """Actual built structure of a canonical ``"<operator>_rf<int>"`` name.

    This is the same ``(kernel, dilation, separable)`` key that
    :func:`tdarts.temporal_ops.canonical_candidates` uses to collapse aliases,
    so ``normal_rf15`` and ``dilated_rf15`` compare equal here.  It is what the
    ``no_duplicate_paths`` constraint must use -- comparing names would let the
    RF 15 aliases slip through as if they were different structures.
    """

    op_name, rf = _split_candidate(candidate)
    try:
        return _structure_key_map(band)[(op_name, rf)]
    except KeyError as exc:
        raise ValueError(
            f"unknown candidate {candidate!r} for band {band!r}"
        ) from exc


def _split_candidate(candidate: str) -> tuple[str, int]:
    op_name, separator, rf_text = str(candidate).rpartition("_rf")
    if not separator or not op_name or not rf_text.isdigit():
        raise ValueError(f"malformed candidate name {candidate!r}")
    return op_name, int(rf_text)


@lru_cache(maxsize=None)
def _structure_key_map(band: str) -> dict[tuple[str, int], tuple[int, int, bool]]:
    pool = build_all_candidates(use_norm=False)
    return {
        (op_name, rf): pool[(band, op_name, rf)].structure_key
        for op_name in C.OPERATORS
        for rf in C.RF_SPACE[band]
    }


def _mixture_output(
    weights: torch.Tensor, ops: nn.ModuleList, x: torch.Tensor
) -> torch.Tensor:
    out = None
    for weight, op in zip(weights, ops):
        term = weight * op(x)
        out = term if out is None else out + term
    if out is None:  # pragma: no cover - construction forbids empty pools
        raise RuntimeError("empty operator pool")
    return out


def _one_hot(logits: torch.Tensor, index: int) -> torch.Tensor:
    """One-hot mixing weights for hard evaluation, detached from autograd."""

    with torch.no_grad():
        weights = torch.zeros_like(logits)
        weights[int(index)] = 1.0
    return weights


def _band_input(x: torch.Tensor, expected_channels: int, bands: Sequence[str]) -> dict[str, torch.Tensor]:
    """Split ``[B, 9, C, T]`` (or the official 5-D layout) into band chunks."""

    if x.dim() == 5 and x.shape[1] == 1 and x.shape[4] == len(bands) * C.IN_CHANNELS:
        x = torch.squeeze(x.permute((0, 4, 2, 3, 1)), dim=4)
    if x.dim() != 4 or x.shape[1] != expected_channels:
        raise ValueError(
            f"expected [B, {expected_channels}, C, T] or "
            f"[B, 1, C, T, {expected_channels}], got shape {tuple(x.shape)}"
        )
    return dict(zip(bands, torch.split(x, C.IN_CHANNELS, dim=1)))


class OperatorPool(nn.Module):
    """The four operator families evaluated at one shared receptive field."""

    def __init__(
        self,
        band: str,
        rf: int,
        in_channels: int = C.IN_CHANNELS,
        out_channels: int = C.PATH_CHANNELS,
        use_candidate_norm: bool = C.USE_CANDIDATE_NORM,
    ):
        super().__init__()
        if band not in C.BANDS:
            raise ValueError(f"unknown band {band!r}")
        self.band = band
        self.rf = int(rf)
        self.op_names = tuple(C.OPERATORS)
        self.ops = nn.ModuleList(
            [
                build_temporal_op(
                    op_name,
                    band,
                    self.rf,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    use_norm=use_candidate_norm,
                )
                for op_name in self.op_names
            ]
        )

    def forward(self, x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return _mixture_output(weights, self.ops, x)


class RFPool(nn.Module):
    """One fixed operator evaluated at every receptive field in the ladder."""

    def __init__(
        self,
        band: str,
        op_name: str,
        in_channels: int = C.IN_CHANNELS,
        out_channels: int = C.PATH_CHANNELS,
        rfs: Sequence[int] = RF_SEARCH_SPACE,
        use_candidate_norm: bool = C.USE_CANDIDATE_NORM,
    ):
        super().__init__()
        if band not in C.BANDS:
            raise ValueError(f"unknown band {band!r}")
        if op_name not in C.OPERATORS:
            raise ValueError(f"unknown operator {op_name!r}")
        self.band = band
        self.op_name = op_name
        self.rfs = tuple(int(rf) for rf in rfs)
        if self.rfs != tuple(C.RF_SPACE[band]):
            raise ValueError(
                f"phase B must range over the full ladder {tuple(C.RF_SPACE[band])}, "
                f"got {self.rfs}"
            )
        self.ops = nn.ModuleList(
            [
                build_temporal_op(
                    op_name,
                    band,
                    rf,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    use_norm=use_candidate_norm,
                )
                for rf in self.rfs
            ]
        )

    def forward(self, x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return _mixture_output(weights, self.ops, x)


class AnchoredOperatorCell(nn.Module):
    """Phase A cell: fixed dilated anchor + 4-way operator mixture.

    The cell has one anchor module and one 4-operator pool per scheduled RF, but
    a single ``beta`` container shared across RFs: the operator preference is
    supposed to be a property of the mechanism, not of the sampled RF.  Which RF
    is active is set by :meth:`set_rf` before each forward.
    """

    def __init__(
        self,
        band: str,
        in_channels: int = C.IN_CHANNELS,
        path_channels: int = C.PATH_CHANNELS,
        rfs: Sequence[int] = OP_SEARCH_RFS,
        use_candidate_norm: bool = C.USE_CANDIDATE_NORM,
        alpha_init_scale: float = C.ALPHA_INIT_SCALE,
    ):
        super().__init__()
        if band not in C.BANDS:
            raise ValueError(f"unknown band {band!r}")
        self.band = band
        self.rfs = tuple(int(rf) for rf in rfs)
        if not self.rfs or any(rf not in OP_SEARCH_RFS for rf in self.rfs):
            raise ValueError(
                f"phase A RFs must be drawn from {OP_SEARCH_RFS}, got {self.rfs}"
            )
        if len(set(self.rfs)) != len(self.rfs):
            raise ValueError(f"duplicate phase A RFs: {self.rfs}")

        self.anchor_pools = nn.ModuleList(
            [
                build_temporal_op(
                    ANCHOR_OPERATOR,
                    band,
                    rf,
                    in_channels=in_channels,
                    out_channels=path_channels,
                    use_norm=use_candidate_norm,
                )
                for rf in self.rfs
            ]
        )
        self.searched_pools = nn.ModuleList(
            [
                OperatorPool(
                    band,
                    rf,
                    in_channels=in_channels,
                    out_channels=path_channels,
                    use_candidate_norm=use_candidate_norm,
                )
                for rf in self.rfs
            ]
        )
        self.beta = nn.Parameter(
            float(alpha_init_scale) * torch.randn(len(C.OPERATORS))
        )
        self.out_channels = 2 * path_channels
        self.bn = nn.BatchNorm2d(self.out_channels)
        self._current_rf: int | None = None

    # -- RF scheduling ---------------------------------------------------
    def set_rf(self, rf: int) -> None:
        if int(rf) not in self.rfs:
            raise ValueError(
                f"band {self.band!r}: RF {rf} is not scheduled in phase A "
                f"(expected one of {self.rfs})"
            )
        self._current_rf = int(rf)

    @property
    def current_rf(self) -> int | None:
        return self._current_rf

    def _active_index(self) -> int:
        if self._current_rf is None:
            raise RuntimeError(
                "AnchoredOperatorCell.set_rf() must be called before forward"
            )
        return self.rfs.index(self._current_rf)

    # -- architecture ----------------------------------------------------
    def architecture_parameters(self):
        return [self.beta]

    def operator_weights(self) -> torch.Tensor:
        return torch.softmax(self.beta, dim=0)

    def selected_operator(self) -> str:
        return C.OPERATORS[int(self.operator_weights().detach().argmax())]

    # -- forward ---------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        index = self._active_index()
        anchor = self.anchor_pools[index](x)
        searched = self.searched_pools[index](x, self.operator_weights())
        return self.bn(torch.cat([anchor, searched], dim=1))


class AnchoredRFCell(nn.Module):
    """Phase B cell: two fixed operators, each with four RF logits.

    With ``no_duplicate_paths`` the two RF choices are decoded jointly: the pair
    maximises ``log p_anchor(i) + log p_searched(j)`` over every combination
    whose actual structures differ.  Selecting each edge independently and only
    then checking for a collision would silently fall back to a worse pair
    whenever the independent argmax happens to collide.
    """

    def __init__(
        self,
        band: str,
        operator: str,
        in_channels: int = C.IN_CHANNELS,
        path_channels: int = C.PATH_CHANNELS,
        rfs: Sequence[int] = RF_SEARCH_SPACE,
        use_candidate_norm: bool = C.USE_CANDIDATE_NORM,
        alpha_init_scale: float = C.ALPHA_INIT_SCALE,
        no_duplicate_paths: bool = False,
    ):
        super().__init__()
        if band not in C.BANDS:
            raise ValueError(f"unknown band {band!r}")
        if operator not in C.OPERATORS:
            raise ValueError(f"unknown operator {operator!r}")
        self.band = band
        self.operator = operator
        self.anchor_pool = RFPool(
            band,
            ANCHOR_OPERATOR,
            in_channels=in_channels,
            out_channels=path_channels,
            rfs=rfs,
            use_candidate_norm=use_candidate_norm,
        )
        self.searched_pool = RFPool(
            band,
            operator,
            in_channels=in_channels,
            out_channels=path_channels,
            rfs=rfs,
            use_candidate_norm=use_candidate_norm,
        )
        self.rfs = self.anchor_pool.rfs
        self.gamma_anchor = nn.Parameter(
            float(alpha_init_scale) * torch.randn(len(self.rfs))
        )
        self.gamma_searched = nn.Parameter(
            float(alpha_init_scale) * torch.randn(len(self.rfs))
        )
        self.out_channels = 2 * path_channels
        self.bn = nn.BatchNorm2d(self.out_channels)
        self.hard = False
        self.no_duplicate_paths = bool(no_duplicate_paths)

    # -- architecture ----------------------------------------------------
    def architecture_parameters(self):
        return [self.gamma_anchor, self.gamma_searched]

    def set_hard(self, hard: bool) -> None:
        self.hard = bool(hard)

    def rf_weights(self) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.softmax(self.gamma_anchor, dim=0),
            torch.softmax(self.gamma_searched, dim=0),
        )

    def select_rf_indices(self) -> tuple[int, int]:
        """Decode ``(anchor_index, searched_index)`` for this band.

        Without the constraint this is the per-edge argmax.  With it, the pair
        maximises the joint log-probability subject to the two structures
        differing under :attr:`tdarts.temporal_ops.TemporalOp.structure_key`.
        Ties keep the lexicographically first ``(i, j)`` pair, so decoding is
        deterministic.
        """

        anchor_logits = self.gamma_anchor.detach()
        searched_logits = self.gamma_searched.detach()
        if not self.no_duplicate_paths:
            return int(anchor_logits.argmax()), int(searched_logits.argmax())

        anchor_keys = [op.structure_key for op in self.anchor_pool.ops]
        searched_keys = [op.structure_key for op in self.searched_pool.ops]
        best: tuple[float, int, int] | None = None
        for anchor_index in range(len(self.rfs)):
            for searched_index in range(len(self.rfs)):
                if anchor_keys[anchor_index] == searched_keys[searched_index]:
                    continue
                score = float(anchor_logits[anchor_index] + searched_logits[searched_index])
                if best is None or score > best[0]:
                    best = (score, anchor_index, searched_index)
        if best is None:  # pragma: no cover - the 4x4 grid always has a legal pair
            raise RuntimeError(
                f"band {self.band!r}: every RF pair has identical structure; "
                f"no_duplicate_paths cannot be satisfied"
            )
        return best[1], best[2]

    # -- forward ---------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.hard:
            anchor_index, searched_index = self.select_rf_indices()
            weight_anchor = _one_hot(self.gamma_anchor, anchor_index)
            weight_searched = _one_hot(self.gamma_searched, searched_index)
        else:
            weight_anchor = torch.softmax(self.gamma_anchor, dim=0)
            weight_searched = torch.softmax(self.gamma_searched, dim=0)
        anchor = self.anchor_pool(x, weight_anchor)
        searched = self.searched_pool(x, weight_searched)
        return self.bn(torch.cat([anchor, searched], dim=1))


class _AnchoredNetBase(nn.Module):
    """Band split -> anchored cells -> unchanged backbone.

    The shared plumbing mirrors :class:`tdarts.mixed_op.TemporalDARTSNet`: the
    temporal stage produces ``[B, 36, C, T]`` and the backbone is not adapted in
    any way.
    """

    def __init__(
        self,
        *,
        n_electrodes: int = C.NUM_ELECTRODES,
        n_classes: int = C.NUM_CLASSES,
        in_channels: int = C.IN_CHANNELS,
        path_channels: int = C.PATH_CHANNELS,
        backbone: nn.Module | None = None,
    ):
        super().__init__()
        self.bands = tuple(C.BANDS)
        self.in_channels = int(in_channels)
        self.path_channels = int(path_channels)
        self.n_electrodes = int(n_electrodes)
        self.temporal_out_channels = len(self.bands) * 2 * self.path_channels

        if backbone is None:
            from tdarts.backbone import TemporalBackbone

            backbone = TemporalBackbone(
                in_channels=self.temporal_out_channels,
                n_electrodes=n_electrodes,
                n_classes=n_classes,
            )
        self.backbone = backbone
        #: Assigned by the concrete subclass.
        self.cells: nn.ModuleDict

    # -- architecture / weight split -------------------------------------
    def arch_parameters(self):
        return [
            p for cell in self.cells.values() for p in cell.architecture_parameters()
        ]

    def network_parameters(self):
        arch_ids = {id(p) for p in self.arch_parameters()}
        for param in self.parameters():
            if id(param) not in arch_ids:
                yield param

    def num_arch_parameters(self) -> int:
        return sum(p.numel() for p in self.arch_parameters())

    # -- forward ---------------------------------------------------------
    def split_bands(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return _band_input(x, len(self.bands) * self.in_channels, self.bands)

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
        return self.backbone(temporal)


class AnchoredOperatorNet(_AnchoredNetBase):
    """Phase A supernet: three cells, one ``beta`` per cell (12 numbers)."""

    def __init__(
        self,
        *,
        n_electrodes: int = C.NUM_ELECTRODES,
        n_classes: int = C.NUM_CLASSES,
        in_channels: int = C.IN_CHANNELS,
        path_channels: int = C.PATH_CHANNELS,
        rfs: Sequence[int] = OP_SEARCH_RFS,
        use_candidate_norm: bool = C.USE_CANDIDATE_NORM,
        alpha_init_scale: float = C.ALPHA_INIT_SCALE,
        backbone: nn.Module | None = None,
    ):
        super().__init__(
            n_electrodes=n_electrodes,
            n_classes=n_classes,
            in_channels=in_channels,
            path_channels=path_channels,
            backbone=backbone,
        )
        self.rfs = tuple(int(rf) for rf in rfs)
        self.cells = nn.ModuleDict(
            {
                band: AnchoredOperatorCell(
                    band,
                    in_channels=in_channels,
                    path_channels=path_channels,
                    rfs=self.rfs,
                    use_candidate_norm=use_candidate_norm,
                    alpha_init_scale=alpha_init_scale,
                )
                for band in self.bands
            }
        )

    def set_rf(self, rf: int) -> None:
        for cell in self.cells.values():
            cell.set_rf(rf)

    @property
    def current_rf(self) -> int | None:
        values = {cell.current_rf for cell in self.cells.values()}
        if len(values) != 1:
            raise RuntimeError(f"cells disagree on the active RF: {values}")
        return next(iter(values))

    def selected_operators(self) -> dict[str, str]:
        return {band: cell.selected_operator() for band, cell in self.cells.items()}

    def describe(self) -> str:
        lines = [
            self.__class__.__name__,
            f"  phase      : operator (anchor={ANCHOR_OPERATOR})",
            f"  RFs        : {self.rfs}",
            f"  arch params: {self.num_arch_parameters()} "
            f"({len(self.arch_parameters())} containers x {len(C.OPERATORS)})",
            f"  net params : {sum(p.numel() for p in self.network_parameters())}",
        ]
        return "\n".join(lines)


class AnchoredRFNet(_AnchoredNetBase):
    """Phase B supernet: operators frozen, two gamma containers per band."""

    def __init__(
        self,
        operators: Mapping[str, str],
        *,
        n_electrodes: int = C.NUM_ELECTRODES,
        n_classes: int = C.NUM_CLASSES,
        in_channels: int = C.IN_CHANNELS,
        path_channels: int = C.PATH_CHANNELS,
        rfs: Sequence[int] = RF_SEARCH_SPACE,
        use_candidate_norm: bool = C.USE_CANDIDATE_NORM,
        alpha_init_scale: float = C.ALPHA_INIT_SCALE,
        no_duplicate_paths: bool = False,
        backbone: nn.Module | None = None,
    ):
        super().__init__(
            n_electrodes=n_electrodes,
            n_classes=n_classes,
            in_channels=in_channels,
            path_channels=path_channels,
            backbone=backbone,
        )
        missing = set(self.bands) - set(operators)
        extra = set(operators) - set(self.bands)
        if missing or extra:
            raise ValueError(
                f"operators must cover exactly {tuple(self.bands)}; "
                f"missing={sorted(missing)} extra={sorted(extra)}"
            )
        self.operators = {band: str(operators[band]) for band in self.bands}
        self.no_duplicate_paths = bool(no_duplicate_paths)
        self.cells = nn.ModuleDict(
            {
                band: AnchoredRFCell(
                    band,
                    self.operators[band],
                    in_channels=in_channels,
                    path_channels=path_channels,
                    rfs=rfs,
                    use_candidate_norm=use_candidate_norm,
                    alpha_init_scale=alpha_init_scale,
                    no_duplicate_paths=self.no_duplicate_paths,
                )
                for band in self.bands
            }
        )
        self.hard = False

    def set_hard(self, hard: bool) -> None:
        self.hard = bool(hard)
        for cell in self.cells.values():
            cell.set_hard(hard)

    def selected_genes(self) -> dict[str, dict[str, tuple[str, int]]]:
        genes: dict[str, dict[str, tuple[str, int]]] = {}
        for band, cell in self.cells.items():
            anchor_index, searched_index = cell.select_rf_indices()
            genes[band] = {
                "anchor": (ANCHOR_OPERATOR, cell.rfs[anchor_index]),
                "searched": (cell.operator, cell.rfs[searched_index]),
            }
        return genes

    def describe(self) -> str:
        lines = [
            self.__class__.__name__,
            f"  phase      : RF (operators frozen)",
            f"  operators  : "
            + ", ".join(f"{band}={op}" for band, op in self.operators.items()),
            f"  RFs        : {self.cells[self.bands[0]].rfs}",
            f"  arch params: {self.num_arch_parameters()} "
            f"({len(self.arch_parameters())} containers x {len(RF_SEARCH_SPACE)})",
            f"  net params : {sum(p.numel() for p in self.network_parameters())}",
        ]
        return "\n".join(lines)


def inherit_operator_weights(
    operator_net: AnchoredOperatorNet, rf_net: AnchoredRFNet
) -> list[str]:
    """Copy phase A weights into phase B wherever the geometry already exists.

    Phase B has no RF-15 modules in phase A (RF 15 is not searched there), so
    those four modules per band stay freshly initialised even when inheritance
    is requested.  The returned list names every copied destination prefix so a
    run can record exactly what was inherited.
    """

    copied: list[str] = []
    for band in rf_net.bands:
        source = operator_net.cells[band]
        target = rf_net.cells[band]
        for target_index, rf in enumerate(target.rfs):
            if rf not in source.rfs:
                continue
            source_index = source.rfs.index(rf)
            target.anchor_pool.ops[target_index].load_state_dict(
                source.anchor_pools[source_index].state_dict()
            )
            copied.append(f"cells.{band}.anchor_pool.ops.{target_index}")
            operator_index = source.searched_pools[source_index].op_names.index(
                rf_net.operators[band]
            )
            target.searched_pool.ops[target_index].load_state_dict(
                source.searched_pools[source_index].ops[operator_index].state_dict()
            )
            copied.append(f"cells.{band}.searched_pool.ops.{target_index}")
    return sorted(copied)


def build_anchored_genotype(model: AnchoredRFNet, *, seed: int, epoch: int):
    """Export the decoded phase B choice as a canonical genotype.

    Path 0 is the dilated anchor, path 1 the searched operator.  Decoding uses
    :meth:`AnchoredRFCell.select_rf_indices`, so with ``no_duplicate_paths`` the
    exported pair is the constrained joint choice, not two independent argmaxes.
    RF-15 aliases are normalised to their canonical names so the same structure
    always maps to the same gene, whichever operator name produced it.
    """

    from tdarts.genotype import Genotype, PathGene
    from tdarts.mixed_op import candidate_names

    genes: list[PathGene] = []
    for band in model.bands:
        cell = model.cells[band]
        anchor_index, searched_index = cell.select_rf_indices()
        weight_anchor, weight_searched = cell.rf_weights()
        names = candidate_names(band)
        edges = (
            (
                0,
                ANCHOR_OPERATOR,
                cell.gamma_anchor.detach(),
                weight_anchor.detach(),
                anchor_index,
            ),
            (
                1,
                cell.operator,
                cell.gamma_searched.detach(),
                weight_searched.detach(),
                searched_index,
            ),
        )
        for path_index, op_name, logits, probabilities, rf_index in edges:
            rf = cell.rfs[rf_index]
            candidate = canonical_candidate_name(band, op_name, rf)
            candidate_index = names.index(candidate)
            genes.append(
                PathGene(
                    band=band,
                    path=path_index,
                    candidate_index=candidate_index,
                    candidate=candidate,
                    alpha=float(logits[rf_index]),
                    probability=float(probabilities[rf_index]),
                )
            )
    if len(genes) != 6:
        raise RuntimeError(f"expected six genes, got {len(genes)}")
    return Genotype(seed=int(seed), epoch=int(epoch), genes=tuple(genes))


def load_anchored_genotype(path) -> "Genotype":
    """Read a ``genotype.json`` produced by ``train_anchor_search.py``.

    The payload is validated rather than trusted: the scheme tag must match,
    every gene must be one of the band's canonical candidates, path 0 must be
    the dilated anchor, and -- when the file says ``no_duplicate_paths`` -- the
    two paths of every band must really differ under ``structure_key``.
    """

    from tdarts.genotype import Genotype, PathGene
    from tdarts.mixed_op import candidate_names

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    scheme = payload.get("scheme")
    if scheme != ANCHORED_SCHEME:
        raise ValueError(
            f"{path}: expected scheme {ANCHORED_SCHEME!r}, got {scheme!r}"
        )

    raw_genes = payload.get("genes")
    if not isinstance(raw_genes, list) or len(raw_genes) != 6:
        raise ValueError(f"{path}: 'genes' must be a list of six entries")
    no_duplicate_paths = bool(payload.get("no_duplicate_paths", False))

    genes: list[PathGene] = []
    seen: set[tuple[str, int]] = set()
    for item in raw_genes:
        if not isinstance(item, dict):
            raise ValueError(f"{path}: every gene must be a mapping")
        band = item.get("band")
        if band not in C.BANDS:
            raise ValueError(f"{path}: unknown band {band!r}")
        path_index = int(item.get("path", -1))
        if path_index not in (0, 1):
            raise ValueError(f"{path}: gene path must be 0 or 1, got {path_index}")
        if (band, path_index) in seen:
            raise ValueError(f"{path}: duplicate gene for {band}/path{path_index}")
        seen.add((band, path_index))

        candidate = str(item.get("candidate", ""))
        names = candidate_names(band)
        if candidate not in names:
            raise ValueError(
                f"{path}: {candidate!r} is not a canonical candidate for "
                f"band {band!r}; expected one of {names}"
            )
        candidate_index = int(item.get("candidate_index", -1))
        if candidate_index != names.index(candidate):
            raise ValueError(
                f"{path}: candidate_index {candidate_index} does not match "
                f"{candidate!r} for band {band!r}"
            )
        op_name, _ = _split_candidate(candidate)
        if path_index == 0 and op_name != ANCHOR_OPERATOR:
            raise ValueError(
                f"{path}: the anchored scheme fixes path 0 to "
                f"{ANCHOR_OPERATOR!r}, got {candidate!r}"
            )
        genes.append(
            PathGene(
                band=band,
                path=path_index,
                candidate_index=candidate_index,
                candidate=candidate,
                alpha=float(item.get("alpha", 0.0)),
                probability=float(item.get("probability", 0.0)),
            )
        )

    missing = {
        (band, path_index) for band in C.BANDS for path_index in (0, 1)
    } - seen
    if missing:
        raise ValueError(f"{path}: missing genes for {sorted(missing)}")
    genes.sort(key=lambda gene: (C.BANDS.index(gene.band), gene.path))

    if no_duplicate_paths:
        for band in C.BANDS:
            pair = [gene for gene in genes if gene.band == band]
            keys = [candidate_structure_key(band, gene.candidate) for gene in pair]
            if keys[0] == keys[1]:
                raise ValueError(
                    f"{path}: band {band!r} violates no_duplicate_paths: "
                    f"{pair[0].candidate} and {pair[1].candidate} are the same "
                    f"structure {keys[0]}"
                )

    seed = int(payload.get("seed", 0))
    epoch = int(payload.get("epochs", payload.get("epoch", 0)))
    return Genotype(seed=seed, epoch=epoch, genes=tuple(genes))


def _probability_view(probabilities: torch.Tensor, labels: Sequence[str]) -> dict[str, Any]:
    top_values, top_indices = torch.topk(probabilities, k=2)
    entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum()
    return {
        "probabilities": [float(value) for value in probabilities],
        "labels": list(labels),
        "top1": {
            "index": int(top_indices[0]),
            "label": labels[int(top_indices[0])],
            "probability": float(top_values[0]),
        },
        "top2": {
            "index": int(top_indices[1]),
            "label": labels[int(top_indices[1])],
            "probability": float(top_values[1]),
        },
        "margin": float(top_values[0] - top_values[1]),
        "normalized_entropy": float(entropy / math.log(len(labels))),
    }


def operator_probability_summary(model: AnchoredOperatorNet) -> dict[str, Any]:
    """Per-band operator probabilities with entropy and margin."""

    summary: dict[str, Any] = {}
    for band, cell in model.cells.items():
        probabilities = cell.operator_weights().detach().cpu()
        summary[band] = _probability_view(probabilities, list(C.OPERATORS))
        summary[band]["selected"] = cell.selected_operator()
    return summary


def rf_probability_summary(model: AnchoredRFNet) -> dict[str, Any]:
    """Per band/path RF probabilities, keyed ``<band>_path<0|1>``."""

    summary: dict[str, Any] = {}
    for band, cell in model.cells.items():
        weight_anchor, weight_searched = cell.rf_weights()
        edges = (
            (0, ANCHOR_OPERATOR, weight_anchor.detach().cpu()),
            (1, cell.operator, weight_searched.detach().cpu()),
        )
        for path_index, op_name, probabilities in edges:
            labels = [f"{op_name}_rf{rf}" for rf in cell.rfs]
            summary[f"{band}_path{path_index}"] = _probability_view(
                probabilities, labels
            )
    return summary


def run_anchored_epoch(
    architect,
    train_loader: Iterable,
    val_loader: Iterable,
    criterion: nn.Module,
    *,
    epoch: int,
    device: torch.device,
    warmup_epochs: int = 20,
    start_step: int = 0,
    on_train_step: Callable[[int], None] | None = None,
) -> tuple[dict[str, Any], int]:
    """One alternating first-order epoch with an optional per-step hook.

    ``on_train_step`` is called with the global train-step index before each
    weight update.  Phase A uses it to install ``rf_schedule(step)`` on the
    model; the paired architecture step deliberately sees the same RF, so every
    four-way comparison within a step is made at one scale.  Returns the epoch
    metrics and the next global step index.
    """

    update_alpha = should_update_alphas(epoch, warmup_epochs)
    val_iterator = iter(val_loader)
    total_train_nll = total_train_acc = total_weight_grad = 0.0
    total_alpha_nll = total_alpha_acc = total_alpha_grad = 0.0
    train_steps = alpha_steps = 0
    step = int(start_step)

    for train_batch in train_loader:
        if on_train_step is not None:
            on_train_step(step)
        train_inputs, train_targets = (
            train_batch[0].to(device, non_blocking=True),
            train_batch[1].to(device, non_blocking=True),
        )
        weight_result = architect.weight_step(train_inputs, train_targets, criterion)
        train_steps += 1
        total_train_nll += weight_result.nll
        total_train_acc += weight_result.accuracy
        total_weight_grad += weight_result.grad_norm

        if update_alpha:
            try:
                val_batch = next(val_iterator)
            except StopIteration:
                val_iterator = iter(val_loader)
                val_batch = next(val_iterator)
            val_inputs, val_targets = (
                val_batch[0].to(device, non_blocking=True),
                val_batch[1].to(device, non_blocking=True),
            )
            alpha_result = architect.alpha_step(val_inputs, val_targets, criterion)
            alpha_steps += 1
            total_alpha_nll += alpha_result.nll
            total_alpha_acc += alpha_result.accuracy
            total_alpha_grad += alpha_result.grad_norm
        step += 1

    def mean(total: float, count: int) -> float:
        return total / count if count else 0.0

    metrics = {
        "train_nll": mean(total_train_nll, train_steps),
        "train_acc": mean(total_train_acc, train_steps),
        "network_grad_norm": mean(total_weight_grad, train_steps),
        "alpha_step_nll": mean(total_alpha_nll, alpha_steps),
        "alpha_step_acc": mean(total_alpha_acc, alpha_steps),
        "alpha_grad_norm": mean(total_alpha_grad, alpha_steps),
        "train_steps": train_steps,
        "alpha_steps": alpha_steps,
        "alpha_updated": update_alpha,
        "next_step": step,
    }
    return metrics, step
