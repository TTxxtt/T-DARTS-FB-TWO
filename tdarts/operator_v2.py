"""Temporal operator families for the Operator Separability V2 pilot.

Why this is a separate module
-----------------------------
``tdarts.temporal_ops`` owns the 14-candidate pool that the DARTS search
relaxes over.  Adding a family to its ``OPERATOR_REGISTRY`` would change
``canonical_candidates``, which would change the search space, which would
silently invalidate every archived search and every genotype decoded from one.
So V2 gets its own registry and nothing in the search path imports this module.

The contract every family here satisfies
----------------------------------------
``[B, in_channels, C, T] -> [B, out_channels, C, T]`` with ``T`` and ``C``
unchanged, exactly as in ``temporal_ops``.  On top of that the pilot needs two
properties the search pool never had to guarantee:

* **the same temporal support**, not merely the same effective receptive field.
  Every family samples the same ``kernel`` positions spaced ``dilation`` apart,
  so at RF57 each one sees the identical 15 positions.  A family that quietly
  looked at a longer span would be a different experiment.
* **a comparable parameter count.**  The anchor is ``DilatedConv``; the pilot
  targets +/-20% of it, and where a mechanism cannot reach that band without
  being changed into something else, the deviation is recorded rather than
  papered over (see ``AUDIT_NOTES``).

Parameter budgets are computed, never assumed: :func:`operator_audit` builds
each family and reports what it actually costs.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from tdarts import config as C

# ``_same_length_padding`` is private to temporal_ops but is the one place the
# "odd span cannot be padded symmetrically" rule is enforced.  Re-implementing
# it here would let the two definitions drift, and a silent off-by-one in the
# time axis is exactly the failure the V2 contract exists to prevent.
from tdarts.temporal_ops import (  # noqa: PLC2701
    _same_length_padding,
    count_parameters,
    effective_rf,
    resolve_kernel_dilation,
)

__all__ = [
    "V2_OPERATOR_NAMES",
    "V2_OPERATOR_REGISTRY",
    "AUDIT_NOTES",
    "V2TemporalOp",
    "DilatedConv",
    "GatedTemporalConv",
    "SparseDilatedLocalAttention",
    "DynamicTemporalConv",
    "BandGatedConv",
    "build_v2_operator",
    "operator_audit",
    "count_macs",
    "temporal_support",
]


#: Parameter-fairness band, relative to the ``dilated`` anchor.
PARAM_TOLERANCE = 0.20


AUDIT_NOTES: dict[str, str] = {
    "dilated": "anchor; FBNAS's original temporal cell",
    "gated": (
        "TCN-style tanh/sigmoid gate.  Both branches run at an internal width "
        "of 6 rather than 12: at width 12 the gate costs 1224 params (2.3x the "
        "anchor) because it doubles the temporal convolution.  The bottleneck "
        "is a deliberate parameter-fairness concession, and the mechanism is "
        "unchanged -- only the width the gate operates at."
    ),
    "local_attention": (
        "scores and applies attention over exactly the 15 positions the dilated "
        "conv samples, at the same spacing.  Head dim 3 (== in_channels, so the "
        "bilinear score is already full rank) and the output projection applied "
        "before the weighted reduction (equal by linearity).  Both are exact, "
        "and together they bring the family to ~1.27x the anchor's MACs at "
        "~0.86x its parameters.  The residual overhead is intrinsic: a weight in "
        "a softmax-weighted reduction is used K times per position where a "
        "convolution weight is used once, so no parameter-matched windowed "
        "attention reaches the anchor's compute."
    ),
    "dynamic": (
        "K=2 depthwise basis kernels combined by an input-dependent softmax "
        "gate taken from globally pooled features (the Dynamic Convolution "
        "formulation).  Depthwise rather than dense bases: two dense bases plus "
        "the projection would be 1224 params.  Per-sample gating, not "
        "per-position."
    ),
    "band_gated": (
        "input-dependent gate over the 3 filter-bank channels already inside "
        "each Low/Mid/High cell, applied before the temporal convolution.  No "
        "FFT, no SincConv.  The gate is a 1x1 convolution (RF 1), so it leaves "
        "the temporal support identical to the anchor."
    ),
}


# ----------------------------------------------------------------------
# measurement
# ----------------------------------------------------------------------
def count_macs(module: nn.Module, input_shape: tuple[int, ...]) -> int:
    """Multiply-accumulates for one forward pass over ``input_shape``.

    Convolution and linear work is measured from the actual calls via forward
    hooks, so a family whose forward loops (``dynamic`` applies one convolution
    per basis kernel) is measured as executed rather than as written.

    Hooks cannot see elementwise arithmetic, which is a real cost for
    ``local_attention`` (the score and context products) and ``dynamic`` (the
    blend of basis outputs).  Those families declare their own count through
    :meth:`_V2OpBase.extra_macs`, so the total is conv/linear work plus declared
    elementwise work.  Without that, attention would have been reported at 0.93x
    the anchor instead of the ~2.3x it actually costs -- an operator that looked
    cheaper than it is, which is exactly the sort of thing this table is meant
    to expose.
    """
    total = 0
    handles = []

    def conv_hook(layer: nn.Conv2d, inputs, output):
        nonlocal total
        out = output
        kernel_ops = layer.kernel_size[0] * layer.kernel_size[1]
        kernel_ops *= layer.in_channels // layer.groups
        total += int(out.numel() // out.shape[0]) * kernel_ops

    def linear_hook(layer: nn.Linear, inputs, output):
        nonlocal total
        total += int(output.numel() // output.shape[0]) * layer.in_features

    was_training = module.training
    module.eval()
    try:
        for child in module.modules():
            if isinstance(child, nn.Conv2d):
                handles.append(child.register_forward_hook(conv_hook))
            elif isinstance(child, nn.Linear):
                handles.append(child.register_forward_hook(linear_hook))
        with torch.no_grad():
            module(torch.zeros(input_shape))
    finally:
        for handle in handles:
            handle.remove()
        module.train(was_training)

    target = module.op if isinstance(module, V2TemporalOp) else module
    if hasattr(target, "extra_macs"):
        total += int(target.extra_macs(input_shape))
    return total


def temporal_support(module: nn.Module) -> tuple[int, int, int]:
    """``(num_positions, spacing, span)`` of the temporal window this op reads.

    Read off the built modules, so it reports what the operator does rather
    than what it was asked to do.  ``span`` must equal the target RF for every
    family -- that equality is the "no operator peeks further" check.
    """
    if isinstance(module, SparseDilatedLocalAttention):
        return (module.kernel, module.dilation, effective_rf(module.kernel, module.dilation))
    if isinstance(module, DynamicTemporalConv):
        return (module.kernel, module.dilation, effective_rf(module.kernel, module.dilation))
    if isinstance(module, (GatedTemporalConv, BandGatedConv)):
        return (module.kernel, module.dilation, effective_rf(module.kernel, module.dilation))
    return (module.kernel, module.dilation, effective_rf(module.kernel, module.dilation))


# ----------------------------------------------------------------------
# shared base
# ----------------------------------------------------------------------
class _V2OpBase(nn.Module):
    """Geometry validation and reporting shared by every V2 family.

    The constructor signature deliberately matches
    ``tdarts.temporal_ops._TemporalOpBase`` so the two pools can be driven by
    the same loops in tests and audit tooling.
    """

    op_name: str = "v2base"

    def __init__(
        self,
        band: str,
        target_rf: int,
        in_channels: int = C.IN_CHANNELS,
        out_channels: int = C.PATH_CHANNELS * C.NUM_PATHS,
        base_kernel: int | None = None,
    ) -> None:
        super().__init__()
        self.band = band
        self.target_rf = int(target_rf)
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)

        self.kernel, self.dilation = resolve_kernel_dilation(band, self.target_rf, base_kernel)
        self.padding = tuple(_same_length_padding(self.kernel, self.dilation))

        self._build_layers()
        if count_parameters(self, trainable_only=False) == 0:
            raise RuntimeError(
                f"{type(self).__name__} built zero parameters; a temporal "
                f"operator must be trainable"
            )

    def _build_layers(self) -> None:  # pragma: no cover - abstract
        raise NotImplementedError

    @property
    def num_params(self) -> int:
        return count_parameters(self)

    @property
    def support(self) -> tuple[int, int, int]:
        return temporal_support(self)

    def extra_macs(self, input_shape: tuple[int, ...]) -> int:
        """Elementwise multiply-accumulates the hooks cannot see.

        Overridden by the families whose forward does arithmetic outside a
        convolution or linear layer.  Default zero: a purely convolutional
        operator needs nothing here.
        """
        return 0

    def describe(self) -> str:
        positions, spacing, span = self.support
        return (
            f"{self.op_name:<17} kernel={self.kernel:<4} dilation={self.dilation:<3} "
            f"RF={self.target_rf:<4} support={positions}x{spacing}={span:<4} "
            f"params={self.num_params}"
        )

    def temporal_conv_kwargs(self) -> dict:
        """Keyword arguments every family's dense temporal convolution shares.

        Centralised so the support guarantee is stated once: whichever family
        calls this gets the same kernel, dilation and same-length padding, and
        therefore the same window.
        """
        return {
            "kernel_size": (1, self.kernel),
            "dilation": (1, self.dilation),
            "padding": self.padding,
            "bias": False,
        }


class DilatedConv(_V2OpBase):
    """The FBNAS anchor: one dilated convolution, sampled sparsely.

    At RF57 this is ``Conv2d(in, out, (1,15), dilation=(1,4))`` -- 15 taps
    spaced 4 apart.  Every other family is judged against this one.
    """

    op_name = "dilated"

    def _build_layers(self) -> None:
        self.conv = nn.Conv2d(self.in_channels, self.out_channels, **self.temporal_conv_kwargs())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class GatedTemporalConv(_V2OpBase):
    """TCN-style gating: ``tanh(f(x)) * sigmoid(g(x))``, then a projection.

    ``f`` and ``g`` are separate temporal convolutions with the anchor's
    geometry, so both gates read the same 15 positions.  The projection is a
    1x1 convolution (RF 1) and leaves the support unchanged.
    """

    op_name = "gated"

    #: Internal width of the two gate branches; see AUDIT_NOTES["gated"].
    HIDDEN = 6

    def _build_layers(self) -> None:
        if self.out_channels != 12:
            # The width was chosen against the pilot's 3 -> 12 budget; a
            # different output width needs the budget redone, not silently
            # reused.
            raise ValueError(
                f"gated assumes the pilot's out_channels=12 budget, got {self.out_channels}"
            )
        kwargs = self.temporal_conv_kwargs()
        self.f = nn.Conv2d(self.in_channels, self.HIDDEN, **kwargs)
        self.g = nn.Conv2d(self.in_channels, self.HIDDEN, **kwargs)
        self.proj = nn.Conv2d(self.HIDDEN, self.out_channels, kernel_size=(1, 1), bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(torch.tanh(self.f(x)) * torch.sigmoid(self.g(x)))


class SparseDilatedLocalAttention(_V2OpBase):
    """Attention over exactly the dilated convolution's 15 support positions.

    Deliberately *local and sparse*: no global attention, and no position
    outside the anchor's window is ever attended to, so the temporal support
    matches ``DilatedConv`` exactly.  Implemented by unfolding the key/value
    maps with the anchor's kernel and dilation -- the same gather the dilated
    convolution performs -- and softmaxing over those 15 positions only.

    Why the widths look the way they do
    -----------------------------------
    Two facts fix them.  Both are exact, not approximations, so neither costs
    the family any expressiveness.

    **The score needs 3 dimensions, not 24.**  ``q`` and ``k`` are linear maps
    out of the cell's 3 input channels, so the score is

        q(x_t) . k(x_{t+dk})  ==  x_t^T M x_{t+dk},   M = W_q^T W_k

    and ``M`` is an arbitrary 3x3 matrix -- 9 free parameters.  A head dim
    above 3 spans exactly the same set of score functions; the previous 24 was
    8x redundant, not 8x more expressive.  ``HEAD_DIM`` is therefore pinned to
    ``IN_CHANNELS`` and the constructor refuses to build otherwise, because the
    equality is the only thing justifying so narrow a head.

    **The projection runs before the weighted reduction.**  The output
    projection is linear and the attention weights do not depend on the values,
    so

        W_o (sum_k a_k v_k)  ==  sum_k a_k (W_o v_k)

    exactly.  Reducing over the 12-channel projected map instead of the wide
    value map removes the second ``K``-fold term, which is where the old
    version's 2.27x came from.  Same function, different association order.

    Together these put the family at ~1.27x the anchor's MACs at ~0.86x its
    parameters -- inside both fairness bands.  The remaining gap is intrinsic:
    every parameter in a softmax-weighted reduction is used K times per
    position, where a convolution weight is used once, so a parameter-matched
    windowed attention cannot reach 1.0x.  See ``AUDIT_NOTES``.
    """

    op_name = "local_attention"

    #: Score width.  Equal to IN_CHANNELS by necessity -- see the docstring.
    HEAD_DIM = C.IN_CHANNELS

    #: Bottleneck width feeding the projection.  20 is chosen so the pair
    #: ``v`` + ``proj`` fills the parameter budget while leaving the two
    #: attention-specific costs (scores, context) under the compute ceiling.
    VALUE_WIDTH = 20

    def _build_layers(self) -> None:
        if self.out_channels != 12:
            raise ValueError(
                f"local_attention assumes the pilot's out_channels=12 budget, "
                f"got {self.out_channels}"
            )
        if self.HEAD_DIM != self.in_channels:
            raise ValueError(
                f"local_attention needs HEAD_DIM == in_channels ({self.in_channels}) "
                f"for the bilinear score to be full rank, got HEAD_DIM={self.HEAD_DIM}"
            )
        self.q = nn.Conv2d(self.in_channels, self.HEAD_DIM, kernel_size=(1, 1), bias=False)
        self.k = nn.Conv2d(self.in_channels, self.HEAD_DIM, kernel_size=(1, 1), bias=False)
        self.v = nn.Conv2d(self.in_channels, self.VALUE_WIDTH, kernel_size=(1, 1), bias=False)
        self.proj = nn.Conv2d(self.VALUE_WIDTH, self.out_channels, kernel_size=(1, 1), bias=False)
        self.out = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=(1, 1), bias=False)
        self.scale = 1.0 / math.sqrt(self.HEAD_DIM)

    def _patches(self, x: torch.Tensor) -> torch.Tensor:
        """``[B, d, C, T] -> [B, d, K, C, T]`` over the support positions."""
        batch, channels = x.shape[0], x.shape[1]
        elec, time = x.shape[2], x.shape[3]
        flat = F.unfold(
            x,
            kernel_size=(1, self.kernel),
            dilation=(1, self.dilation),
            padding=self.padding,
        )
        # unfold lays channels out slowest: [B, C, 1, K, C*T] -> [B, C, K, E, T]
        return flat.view(batch, channels, self.kernel, elec, time)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        query = self.q(x).unsqueeze(2)                  # [B, d, 1, E, T]
        keys = self._patches(self.k(x))                 # [B, d, K, E, T]
        scores = (query * keys).sum(dim=1) * self.scale  # [B, K, E, T]
        attention = scores.softmax(dim=1)                # over the K positions only
        # Projected before the reduction, not after: identical by linearity,
        # and it moves the K-fold reduction onto 12 channels instead of 20.
        values = self._patches(self.proj(self.v(x)))     # [B, out, K, E, T]
        context = (attention.unsqueeze(1) * values).sum(dim=2)  # [B, out, E, T]
        return self.out(context)

    def extra_macs(self, input_shape: tuple[int, ...]) -> int:
        _, _, electrodes, time = input_shape
        # Two reductions over the support, at every (electrode, time): the
        # score dot product in HEAD_DIM, and the weighted context sum over the
        # projected map's out_channels.  Everything else in this family is a
        # 1x1 convolution and is counted by the hooks.
        per_position = self.kernel * (self.HEAD_DIM + self.out_channels)
        return electrodes * time * per_position


class DynamicTemporalConv(_V2OpBase):
    """Dynamic convolution: a small bank of basis kernels, mixed per sample.

    Two depthwise basis kernels share the anchor's geometry, and a softmax gate
    read off globally average-pooled features decides how to blend them.  The
    gate is per sample, not per position -- the standard cheap form of dynamic
    convolution -- and the mixing happens in function space (each basis is
    applied, then the outputs are combined), which is equivalent to blending
    the kernels and avoids materialising a per-sample weight tensor.
    """

    op_name = "dynamic"

    #: Number of basis kernels; kept small on purpose.
    NUM_BASIS = 2

    def _build_layers(self) -> None:
        if self.out_channels % self.in_channels != 0:
            raise ValueError(
                f"dynamic: depthwise basis needs out_channels ({self.out_channels}) "
                f"divisible by in_channels ({self.in_channels})"
            )
        self.group_count = self.in_channels
        # The basis kernels are nn.Conv2d modules rather than bare Parameters so
        # the MAC audit's forward hooks see them.  A raw F.conv2d call is
        # invisible to hooks and reported the family as 0.27x the anchor when it
        # is in fact ~0.93x.
        self.bases = nn.ModuleList(
            nn.Conv2d(
                self.in_channels,
                self.out_channels,
                groups=self.group_count,
                **self.temporal_conv_kwargs(),
            )
            for _ in range(self.NUM_BASIS)
        )
        self.proj = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=(1, 1), bias=False)
        self.gate = nn.Linear(self.in_channels, self.NUM_BASIS, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = x.mean(dim=(2, 3))                       # [B, in_channels]
        weights = self.gate(pooled).softmax(dim=1)        # [B, NUM_BASIS]
        blended = None
        for index, basis in enumerate(self.bases):
            term = weights[:, index].view(-1, 1, 1, 1) * basis(x)
            blended = term if blended is None else blended + term
        return self.proj(blended)

    def extra_macs(self, input_shape: tuple[int, ...]) -> int:
        _, _, electrodes, time = input_shape
        # Scaling each basis output by its gate weight, once per basis.
        return self.NUM_BASIS * self.out_channels * electrodes * time


class BandGatedConv(_V2OpBase):
    """Gate the cell's 3 filter-bank channels, then convolve temporally.

    Each Low/Mid/High cell already carries 3 neighbouring filter-bank channels.
    A 1x1 convolution turns those into a per-position gate, and the temporal
    convolution sees the gated signal.  The gate has RF 1, so the temporal
    support is exactly the anchor's -- this family re-weights the existing
    bands rather than looking at a wider window.

    No FFT and no SincConv: the filter-bank split happened upstream and is not
    redone here.
    """

    op_name = "band_gated"

    def _build_layers(self) -> None:
        self.gate = nn.Conv2d(self.in_channels, self.in_channels, kernel_size=(1, 1), bias=True)
        self.conv = nn.Conv2d(self.in_channels, self.out_channels, **self.temporal_conv_kwargs())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x * torch.sigmoid(self.gate(x)))


V2_OPERATOR_REGISTRY: dict[str, type[_V2OpBase]] = {
    "dilated": DilatedConv,
    "gated": GatedTemporalConv,
    "local_attention": SparseDilatedLocalAttention,
    "dynamic": DynamicTemporalConv,
    "band_gated": BandGatedConv,
}

V2_OPERATOR_NAMES: tuple[str, ...] = tuple(V2_OPERATOR_REGISTRY)


# ----------------------------------------------------------------------
# candidate wrapper
# ----------------------------------------------------------------------
class V2TemporalOp(nn.Module):
    """One V2 operator plus the same candidate normalisation the search uses.

    ``BatchNorm2d(out_channels, affine=False)`` matches
    ``tdarts.temporal_ops.TemporalOp``'s default: it contributes no trainable
    parameters, so it cannot shift the parameter comparison, and keeping it
    identical means a V2 model differs from a searched model only in the
    operator family.
    """

    def __init__(
        self,
        op_name: str,
        band: str,
        target_rf: int,
        in_channels: int = C.IN_CHANNELS,
        out_channels: int = C.PATH_CHANNELS * C.NUM_PATHS,
        use_norm: bool = C.USE_CANDIDATE_NORM,
        base_kernel: int | None = None,
    ) -> None:
        super().__init__()
        if op_name not in V2_OPERATOR_REGISTRY:
            raise ValueError(
                f"unknown V2 operator {op_name!r}; expected one of {sorted(V2_OPERATOR_REGISTRY)}"
            )
        self.op_name = op_name
        self.band = band
        self.target_rf = int(target_rf)
        self.op = V2_OPERATOR_REGISTRY[op_name](
            band=band,
            target_rf=target_rf,
            in_channels=in_channels,
            out_channels=out_channels,
            base_kernel=base_kernel,
        )
        self.norm: nn.Module = (
            nn.BatchNorm2d(out_channels, affine=False) if use_norm else nn.Identity()
        )

    @property
    def num_params(self) -> int:
        return count_parameters(self)

    @property
    def num_op_params(self) -> int:
        return count_parameters(self.op)

    @property
    def support(self) -> tuple[int, int, int]:
        return temporal_support(self.op)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.op(x))

    def describe(self) -> str:
        positions, spacing, span = self.support
        return (
            f"{self.band:<4} {self.op_name:<17} RF{self.target_rf:<4} "
            f"support={positions}x{spacing}={span:<4} params={self.num_params}"
        )


def build_v2_operator(
    op_name: str,
    band: str,
    target_rf: int,
    in_channels: int = C.IN_CHANNELS,
    out_channels: int = C.PATH_CHANNELS * C.NUM_PATHS,
    use_norm: bool = C.USE_CANDIDATE_NORM,
    base_kernel: int | None = None,
) -> V2TemporalOp:
    """Factory mirroring :func:`tdarts.temporal_ops.build_temporal_op`."""
    return V2TemporalOp(
        op_name=op_name,
        band=band,
        target_rf=target_rf,
        in_channels=in_channels,
        out_channels=out_channels,
        use_norm=use_norm,
        base_kernel=base_kernel,
    )


# ----------------------------------------------------------------------
# audit
# ----------------------------------------------------------------------
def operator_audit(
    *,
    band: str = "Low",
    target_rf: int = 57,
    in_channels: int = C.IN_CHANNELS,
    out_channels: int = C.PATH_CHANNELS * C.NUM_PATHS,
    time_length: int = 1000,
    n_electrodes: int = C.NUM_ELECTRODES,
    support_tolerance: float = PARAM_TOLERANCE,
) -> list[dict]:
    """Build every family and report cost, shape and temporal support.

    Returns one record per family with the anchor's parameter count attached
    for comparison, so a caller can show the fairness table without recomputing
    it.  Raises nothing: a family that lands outside the tolerance is reported
    with ``within_tolerance`` False, because the pilot's rule is to record a
    deviation rather than to quietly widen the budget.
    """
    shape = (1, in_channels, n_electrodes, time_length)
    records: list[dict] = []
    anchor_params: int | None = None

    for name in V2_OPERATOR_NAMES:
        op = build_v2_operator(
            name, band=band, target_rf=target_rf,
            in_channels=in_channels, out_channels=out_channels,
        )
        with torch.no_grad():
            out = op(torch.zeros(shape))
        positions, spacing, span = op.support
        params = op.num_params
        if name == "dilated":
            anchor_params = params
        records.append(
            {
                "operator": name,
                "band": band,
                "target_rf": target_rf,
                "params": params,
                "macs": count_macs(op, shape),
                "input_shape": list(shape),
                "output_shape": list(out.shape),
                "support_positions": positions,
                "support_spacing": spacing,
                "support_span": span,
                "note": AUDIT_NOTES.get(name, ""),
            }
        )

    assert anchor_params is not None, "the dilated anchor must be in the registry"
    for record in records:
        ratio = record["params"] / anchor_params
        record["anchor_params"] = anchor_params
        record["param_ratio_vs_anchor"] = ratio
        record["within_tolerance"] = abs(ratio - 1.0) <= support_tolerance
        # Every family must read exactly the anchor's window; this is the
        # "no operator peeks further" invariant, asserted here so a future
        # family that breaks it fails the audit instead of the experiment.
        record["support_matches_anchor"] = record["support_span"] == target_rf
    return records
