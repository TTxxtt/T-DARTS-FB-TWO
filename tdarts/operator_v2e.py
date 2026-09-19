"""Expressive-V2 temporal operator families: the same invariants, no capacity diet.

Why a second generation exists
------------------------------
The first generation (``tdarts.operator_v2``) held every family to +/-20% of the
``dilated`` anchor's 540 parameters and <=1.3x its MACs.  That produced a clean
mechanism-vs-mechanism comparison, and it also produced the compromises that
made the comparison less informative than it looked: ``gated`` ran its gate at
an internal width of 6, ``local_attention`` was pinned to a 3-dimensional head,
``dynamic`` used two depthwise bases.  Each of those is a place where the
mechanism was reshaped to fit a parameter budget rather than measured as
designed.

The 45-run Matched pilot then showed a strong *global* operator effect
(``V_operator / V_seed = 4.75``) with no *subject x operator* interaction, and a
per-seed ordering that never changed: ``{dilated, dynamic}`` always first,
``{local_attention, band_gated}`` always last, in all nine subject x seed runs.
A global ordering that never reorders is at least as consistent with "the
mechanism was flattened by the budget" as with "no subject prefers a different
mechanism".

So this generation keeps what actually made the comparison fair and drops what
did not.  **Fair now means**:

* the same tensor contract ``[B, in_channels, C, T] -> [B, out_channels, C, T]``
  with ``C`` and ``T`` unchanged;
* the same temporal support -- RF57, kernel 15, dilation 4, the identical 15
  positions.  A family that quietly looked at a longer span would be a
  different experiment, and no capacity increase may buy one;
* the same backbone and the same training protocol.

**Parameter counts are free, and reported.**  A ~5x-anchor ceiling is advisory
only: :func:`operator_capacity_audit` flags a family that crosses it and never
turns that into a failure, because the whole point is to stop treating a budget
as a design constraint.  What replaces the budget is a *control*: the two
wide-dilated families in ``WIDE_CONTROL_REGISTRY`` keep the anchor's mechanism
and RF while roughly 2.5x and 5x its parameters, so a later result can be asked
"is this the mechanism, or just more capacity?".

Registry separation
-------------------
``tdarts.temporal_ops.OPERATOR_REGISTRY`` is not touched here, for the same
reason ``operator_v2`` does not touch it: adding to it changes
``canonical_candidates``, which changes the DARTS search space, which silently
invalidates every archived search and every genotype decoded from one.  The wide
controls are separated from the candidates by living in their own registry, so
``E_OPERATOR_NAMES`` -- and therefore the pilot grid -- cannot contain one.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from tdarts import config as C

# ``_V2OpBase`` is private to operator_v2 but is exactly the contract under
# test: it resolves the geometry through ``resolve_kernel_dilation`` and hands
# every family the same kernel/dilation/padding through
# ``temporal_conv_kwargs()``.  Re-implementing it here would create a second
# definition of "the same 15 positions at RF57", which is the one thing that
# must not drift.  The same reasoning already makes operator_v2 import
# ``_same_length_padding`` from temporal_ops.
from tdarts.operator_v2 import (  # noqa: PLC2701
    _V2OpBase,
    V2TemporalOp,
    build_v2_operator,
    count_macs,
    temporal_support,
)

__all__ = [
    "CAPACITY_ADVISORY_RATIO",
    "CAPACITY_NOTES",
    "E_OPERATOR_NAMES",
    "E_OPERATOR_REGISTRY",
    "WIDE_CONTROL_NAMES",
    "WIDE_CONTROL_REGISTRY",
    "BandGatedConvE",
    "DilatedConvE",
    "DynamicTemporalConvE",
    "GatedTemporalConvE",
    "NonlinearLocalAttention",
    "WideDilated",
    "build_e_capacity_control",
    "build_e_operator",
    "operator_capacity_audit",
    "print_capacity_table",
]


#: Advisory, report-only ceiling relative to the anchor.  Exceeding it is
#: flagged and never asserted -- see the module docstring.
CAPACITY_ADVISORY_RATIO = 5.0


CAPACITY_NOTES: dict[str, str] = {
    "dilated_e": (
        "anchor; identical to the Matched generation's anchor.  Every ratio in "
        "the capacity table is measured against this family, so a change here "
        "moves the whole table."
    ),
    "gated_e": (
        "the Matched gate at full width.  The gate ran at an internal width of 6 "
        "there because width 12 costs 1224 params (2.3x the anchor); here both "
        "branches run at 12 and a 1x1 skip carries the input past the gate.  The "
        "skip is a real capability difference, not a formality: at initialisation "
        "the family is close to a 1x1 projection of its input, so it can decline "
        "to use the temporal convolution at all.  Recorded here rather than left "
        "for a reader to notice."
    ),
    "local_attention_e": (
        "attention over a learned *nonlinear* representation.  The Matched family "
        "pinned HEAD_DIM to in_channels because a bilinear score taken directly "
        "from 3 channels spans only 3x3 = 9 free parameters, so a wider head was "
        "8x redundant.  That argument dies once a 3->12 embedding with a "
        "nonlinearity sits in front: q and k then read 12 channels, a 6-wide head "
        "is no longer rank-redundant, and two heads give two independent bilinear "
        "forms.  **The nonlinearity is load-bearing**: without it q(embed(x)) "
        "collapses to a rank-3 linear map of the 3 input channels and the family "
        "falls back into exactly the Matched family's score class.  It is "
        "asserted by a test for that reason."
    ),
    "dynamic_e": (
        "K=4 full (dense) basis kernels, up from K=2 depthwise.  No output "
        "projection: with dense bases proj(sum_i w_i B_i(x)) == sum_i w_i "
        "(proj o B_i)(x), so a projection would absorb into the bases and add "
        "144 parameters and 3.17M MACs for no function class.  The Matched "
        "family is depthwise and its projection genuinely mixes channels, which "
        "is why it keeps one.  Gating is still per-sample from globally pooled "
        "features -- the upgrade is K and density, not gating granularity."
    ),
    "band_gated_e": (
        "the Matched band gate made nonlinear: 3->12 with a GELU, then 12->3 with "
        "a sigmoid, instead of a single 3->3 linear gate.  Both layers are 1x1 "
        "(RF 1), so the temporal support is still exactly the anchor's -- the "
        "upgrade widens the gate's capacity, not the window."
    ),
    "wide_dilated_2p5_e": (
        "capacity control, ablation only, never a NAS candidate.  The anchor's "
        "mechanism and RF with 2.5x its parameters, to test whether a family that "
        "beats the anchor is doing so through its mechanism or simply through "
        "having more of them."
    ),
    "wide_dilated_5_e": (
        "capacity control at 5x the anchor.  Deliberately exceeds the advisory "
        "ceiling -- that is what it is for, and the audit says so instead of "
        "calling it a violation."
    ),
}


# ----------------------------------------------------------------------
# candidate families
# ----------------------------------------------------------------------
class DilatedConvE(_V2OpBase):
    """The anchor: one dilated convolution.  Unchanged from the Matched family.

    Keeping the anchor bit-identical across generations is what makes the two
    comparable at all: the Matched and Expressive grids share a control, so a
    shift in the other families can be read against something that did not move.
    """

    op_name = "dilated_e"

    def _build_layers(self) -> None:
        self.conv = nn.Conv2d(self.in_channels, self.out_channels, **self.temporal_conv_kwargs())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class GatedTemporalConvE(_V2OpBase):
    """Full-width TCN gate: ``proj(tanh(f(x)) * sigmoid(g(x))) + skip(x)``.

    The Matched family ran ``f`` and ``g`` at width 6 to stay inside a parameter
    budget; both are width 12 here.  ``skip`` is a 1x1 path that lets the family
    bypass the temporal convolution entirely, which is a genuine capability the
    Matched version did not have -- see ``CAPACITY_NOTES``.
    """

    op_name = "gated_e"

    #: The two gate branches and the skip all run at the cell's 12 channels.
    HIDDEN = C.PATH_CHANNELS * C.NUM_PATHS

    def _build_layers(self) -> None:
        if self.out_channels != self.HIDDEN:
            # The width is budgeted against the pilot's 3 -> 12 cell; a
            # different output width needs the budget redone rather than a
            # hidden width silently reused against a different output.
            raise ValueError(
                f"gated_e assumes the pilot's out_channels={self.HIDDEN} budget, "
                f"got {self.out_channels}"
            )
        kwargs = self.temporal_conv_kwargs()
        self.f = nn.Conv2d(self.in_channels, self.HIDDEN, **kwargs)
        self.g = nn.Conv2d(self.in_channels, self.HIDDEN, **kwargs)
        self.proj = nn.Conv2d(self.HIDDEN, self.out_channels, kernel_size=(1, 1), bias=False)
        self.skip = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=(1, 1), bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gated = self.proj(torch.tanh(self.f(x)) * torch.sigmoid(self.g(x)))
        return gated + self.skip(x)

    def extra_macs(self, input_shape: tuple[int, ...]) -> int:
        _, _, electrodes, time = input_shape
        # The gate product and the residual addition, one multiply each per
        # output element.  Forward hooks cannot see either.
        return 2 * self.out_channels * electrodes * time


class NonlinearLocalAttention(_V2OpBase):
    """Local attention over a learned nonlinear embedding of the 3 input bands.

    Reads exactly the dilated convolution's 15 support positions and nothing
    else -- the same ``F.unfold`` gather, the same spacing -- so the temporal
    support matches the anchor and the Matched family.

    The reason the head can be wide here when the Matched family pinned it to 3
    is set out in ``CAPACITY_NOTES``; the short version is that the score is now
    taken over a nonlinear 12-channel embedding rather than the raw 3 channels,
    so a 6-wide head is no longer rank-redundant and the GELU is what makes that
    true.
    """

    op_name = "local_attention_e"

    NUM_HEADS = 2
    HEAD_DIM = 6
    #: Width of the nonlinear embedding, and therefore of q/k/v.
    EMBED_WIDTH = C.PATH_CHANNELS * C.NUM_PATHS

    def _build_layers(self) -> None:
        if self.out_channels != self.EMBED_WIDTH:
            raise ValueError(
                f"local_attention_e assumes the pilot's out_channels={self.EMBED_WIDTH} "
                f"budget, got {self.out_channels}"
            )
        if self.NUM_HEADS * self.HEAD_DIM != self.EMBED_WIDTH:
            raise ValueError(
                f"local_attention_e needs NUM_HEADS * HEAD_DIM == out_channels "
                f"({self.EMBED_WIDTH}), got {self.NUM_HEADS} * {self.HEAD_DIM}"
            )
        self.embed = nn.Conv2d(self.in_channels, self.EMBED_WIDTH, kernel_size=(1, 1), bias=True)
        # Not an Identity, and not a free choice: see the class docstring.
        self.act = nn.GELU()
        self.q = nn.Conv2d(self.EMBED_WIDTH, self.EMBED_WIDTH, kernel_size=(1, 1), bias=False)
        self.k = nn.Conv2d(self.EMBED_WIDTH, self.EMBED_WIDTH, kernel_size=(1, 1), bias=False)
        self.v = nn.Conv2d(self.EMBED_WIDTH, self.EMBED_WIDTH, kernel_size=(1, 1), bias=False)
        self.out = nn.Conv2d(self.EMBED_WIDTH, self.out_channels, kernel_size=(1, 1), bias=False)
        self.scale = 1.0 / math.sqrt(self.HEAD_DIM)

    def _patches(self, x: torch.Tensor) -> torch.Tensor:
        """``[B, d, C, T] -> [B, d, K, C, T]`` over the support positions.

        Byte-for-byte the gather the dilated convolution performs, and the same
        one the Matched family uses; a test compares the two on identical input.
        """
        batch, channels = x.shape[0], x.shape[1]
        elec, time = x.shape[2], x.shape[3]
        flat = F.unfold(
            x,
            kernel_size=(1, self.kernel),
            dilation=(1, self.dilation),
            padding=self.padding,
        )
        return flat.view(batch, channels, self.kernel, elec, time)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.act(self.embed(x))
        batch, _, elec, time = hidden.shape
        heads, dim = self.NUM_HEADS, self.HEAD_DIM

        query = self.q(hidden).view(batch, heads, dim, elec, time)
        keys = self._patches(self.k(hidden)).view(batch, heads, dim, self.kernel, elec, time)
        # Score over the head dimension, then softmax over the 15 positions only.
        scores = (query.unsqueeze(3) * keys).sum(dim=2) * self.scale
        attention = scores.softmax(dim=2)                      # [B, heads, K, E, T]

        values = self._patches(self.v(hidden)).view(batch, heads, dim, self.kernel, elec, time)
        context = (attention.unsqueeze(2) * values).sum(dim=3)  # [B, heads, dim, E, T]
        return self.out(context.reshape(batch, heads * dim, elec, time))

    def extra_macs(self, input_shape: tuple[int, ...]) -> int:
        _, _, electrodes, time = input_shape
        # Two reductions over the support at every (electrode, time): the score
        # dot product across the head, and the weighted context sum over the
        # positions.  Each costs out_channels multiplies per position, because
        # NUM_HEADS * HEAD_DIM == out_channels.  Hooks cannot see either, and a
        # counter that misses them reports attention as *cheaper* than a
        # convolution -- the exact failure this declaration exists to prevent.
        return 2 * self.out_channels * self.kernel * electrodes * time


class DynamicTemporalConvE(_V2OpBase):
    """Dynamic convolution with a bank of four full basis kernels.

    Up from the Matched family's two depthwise bases.  The bases are dense and
    there is no output projection, for the reason in ``CAPACITY_NOTES``: with
    dense bases a projection is absorbable into them and buys no function class.

    Gating is per sample, from globally pooled features, as in the Matched
    family.  The mixing happens in function space -- each basis is applied, then
    the outputs are blended -- which is equivalent to blending the kernels
    without materialising a per-sample weight tensor.
    """

    op_name = "dynamic_e"

    #: Four rather than two: the Matched bank was small enough that the family
    #: was nearly a reparameterised single convolution.
    NUM_BASIS = 4

    def _build_layers(self) -> None:
        self.bases = nn.ModuleList(
            nn.Conv2d(self.in_channels, self.out_channels, **self.temporal_conv_kwargs())
            for _ in range(self.NUM_BASIS)
        )
        self.gate = nn.Linear(self.in_channels, self.NUM_BASIS, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = x.mean(dim=(2, 3))
        weights = self.gate(pooled).softmax(dim=1)
        blended = None
        for index, basis in enumerate(self.bases):
            term = weights[:, index].view(-1, 1, 1, 1) * basis(x)
            blended = term if blended is None else blended + term
        return blended

    def extra_macs(self, input_shape: tuple[int, ...]) -> int:
        _, _, electrodes, time = input_shape
        # Scaling each basis output by its gate weight, once per basis.
        return self.NUM_BASIS * self.out_channels * electrodes * time


class BandGatedConvE(_V2OpBase):
    """Nonlinear band gate, then the temporal convolution.

    Each Low/Mid/High cell carries 3 filter-bank channels.  The Matched family
    gated them with a single 3->3 linear layer; here the gate is 3->12 with a
    GELU followed by 12->3 with a sigmoid.  Both are 1x1, so the temporal
    support stays exactly the anchor's -- this family re-weights the existing
    bands rather than looking at a wider window.

    No FFT and no SincConv: the filter-bank split happened upstream and is not
    redone here.
    """

    op_name = "band_gated_e"

    HIDDEN = C.PATH_CHANNELS * C.NUM_PATHS

    def _build_layers(self) -> None:
        self.up = nn.Conv2d(self.in_channels, self.HIDDEN, kernel_size=(1, 1), bias=True)
        self.act = nn.GELU()
        self.down = nn.Conv2d(self.HIDDEN, self.in_channels, kernel_size=(1, 1), bias=True)
        self.conv = nn.Conv2d(self.in_channels, self.out_channels, **self.temporal_conv_kwargs())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.down(self.act(self.up(x))))
        return self.conv(x * gate)

    def extra_macs(self, input_shape: tuple[int, ...]) -> int:
        _, _, electrodes, time = input_shape
        # The gated signal: one multiply per input channel per position.
        return self.in_channels * electrodes * time


# ----------------------------------------------------------------------
# capacity controls (ablation only -- never a NAS candidate)
# ----------------------------------------------------------------------
class WideDilated(_V2OpBase):
    """The anchor's mechanism with more channels, and deliberately no more RF.

    Capacity is bought by widening the hidden width, never by widening the
    window: the dilated convolution keeps kernel 15 / dilation 4, so the support
    is the anchor's 15 positions.  This is the family that answers "attention
    won -- or did it just have more parameters?" without conflating the two
    questions.  It is parameterised by ``WIDTH`` so the two rungs share one
    implementation and cannot drift apart.
    """

    op_name = "wide_dilated"
    WIDTH = 24

    def _build_layers(self) -> None:
        if self.WIDTH <= self.out_channels:
            raise ValueError(
                f"wide_dilated is a capacity control; WIDTH ({self.WIDTH}) must "
                f"exceed out_channels ({self.out_channels})"
            )
        self.conv = nn.Conv2d(self.in_channels, self.WIDTH, **self.temporal_conv_kwargs())
        self.proj = nn.Conv2d(self.WIDTH, self.out_channels, kernel_size=(1, 1), bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.conv(x))


class WideDilated2p5(WideDilated):
    op_name = "wide_dilated_2p5_e"
    WIDTH = 24


class WideDilated5(WideDilated):
    op_name = "wide_dilated_5_e"
    WIDTH = 48


# ----------------------------------------------------------------------
# registries
# ----------------------------------------------------------------------
E_OPERATOR_REGISTRY: dict[str, type[_V2OpBase]] = {
    "dilated_e": DilatedConvE,
    "gated_e": GatedTemporalConvE,
    "local_attention_e": NonlinearLocalAttention,
    "dynamic_e": DynamicTemporalConvE,
    "band_gated_e": BandGatedConvE,
}

E_OPERATOR_NAMES: tuple[str, ...] = tuple(E_OPERATOR_REGISTRY)

#: Separate on purpose.  Nothing here may appear in ``E_OPERATOR_NAMES``, so the
#: pilot grid cannot contain a control and a typo cannot launch one.
WIDE_CONTROL_REGISTRY: dict[str, type[_V2OpBase]] = {
    "wide_dilated_2p5_e": WideDilated2p5,
    "wide_dilated_5_e": WideDilated5,
}

WIDE_CONTROL_NAMES: tuple[str, ...] = tuple(WIDE_CONTROL_REGISTRY)


def build_e_operator(
    op_name: str,
    band: str,
    target_rf: int,
    in_channels: int = C.IN_CHANNELS,
    out_channels: int = C.PATH_CHANNELS * C.NUM_PATHS,
    use_norm: bool = C.USE_CANDIDATE_NORM,
    base_kernel: int | None = None,
) -> V2TemporalOp:
    """Build one Expressive candidate, wearing the frozen generation's wrapper.

    Returning a real :class:`~tdarts.operator_v2.V2TemporalOp` rather than a
    look-alike is what keeps the measurement honest: ``count_macs`` unwraps the
    inner operator through ``isinstance(module, V2TemporalOp)`` to reach
    ``extra_macs``, and ``temporal_support`` reads ``.kernel``/``.dilation`` off
    the built modules.  A parallel wrapper class would silently lose both and
    report the attention family as cheaper than a convolution.
    """
    return build_v2_operator(
        op_name,
        band=band,
        target_rf=target_rf,
        in_channels=in_channels,
        out_channels=out_channels,
        use_norm=use_norm,
        base_kernel=base_kernel,
        registry=E_OPERATOR_REGISTRY,
    )


def build_e_capacity_control(
    op_name: str,
    band: str,
    target_rf: int,
    in_channels: int = C.IN_CHANNELS,
    out_channels: int = C.PATH_CHANNELS * C.NUM_PATHS,
    use_norm: bool = C.USE_CANDIDATE_NORM,
    base_kernel: int | None = None,
) -> V2TemporalOp:
    """Build one wide-dilated control.  Not reachable from the pilot grid."""

    return build_v2_operator(
        op_name,
        band=band,
        target_rf=target_rf,
        in_channels=in_channels,
        out_channels=out_channels,
        use_norm=use_norm,
        base_kernel=base_kernel,
        registry=WIDE_CONTROL_REGISTRY,
    )


# ----------------------------------------------------------------------
# capacity audit
# ----------------------------------------------------------------------
def operator_capacity_audit(
    *,
    band: str = "Low",
    target_rf: int = 57,
    in_channels: int = C.IN_CHANNELS,
    out_channels: int = C.PATH_CHANNELS * C.NUM_PATHS,
    time_length: int = 1000,
    n_electrodes: int = C.NUM_ELECTRODES,
    advisory_ratio: float = CAPACITY_ADVISORY_RATIO,
    include_controls: bool = True,
) -> list[dict]:
    """Build every Expressive family and report what it costs.

    Deliberately *not* the Matched generation's audit.  That one attaches a
    ``within_tolerance`` boolean, because there the +/-20% band was a
    requirement.  Here the band is an advisory ceiling, so a ``within_tolerance``
    key would misrepresent the rule -- a test asserts it is absent rather than
    merely always True, so nobody can quietly reintroduce the requirement.

    Also reports ``macs_hook_only`` next to ``macs``.  The Expressive
    ``gated_e`` and ``band_gated_e`` declare elementwise work through
    ``extra_macs`` that the Matched ``gated`` and ``band_gated`` do not, so the
    two generations' MACs are not measured identically.  Surfacing the split
    keeps that asymmetry visible in the table instead of buried in a docstring;
    the Matched numbers are left alone, since changing them would move ratios
    that are already recorded.
    """

    shape = (1, in_channels, n_electrodes, time_length)
    records: list[dict] = []
    anchor_params: int | None = None

    families: list[tuple[str, str]] = [(name, "candidate") for name in E_OPERATOR_NAMES]
    if include_controls:
        families += [(name, "control") for name in WIDE_CONTROL_NAMES]

    for name, role in families:
        builder = build_e_operator if role == "candidate" else build_e_capacity_control
        op = builder(name, band=band, target_rf=target_rf, in_channels=in_channels, out_channels=out_channels)
        with torch.no_grad():
            out = op(torch.zeros(shape))
        positions, spacing, span = op.support
        params = op.num_params
        declared = int(op.op.extra_macs(shape))
        macs = count_macs(op, shape)
        if name == "dilated_e":
            anchor_params = params
        records.append(
            {
                "operator": name,
                "role": role,
                "band": band,
                "target_rf": target_rf,
                "params": params,
                "macs": macs,
                "macs_elementwise_declared": declared,
                "macs_hook_only": macs - declared,
                "input_shape": list(shape),
                "output_shape": list(out.shape),
                "support_positions": positions,
                "support_spacing": spacing,
                "support_span": span,
                "note": CAPACITY_NOTES.get(name, ""),
            }
        )

    assert anchor_params is not None, "the dilated_e anchor must be in the registry"
    for record in records:
        param_ratio = record["params"] / anchor_params
        mac_ratio = record["macs"] / records[0]["macs"]
        capacity_ratio = max(param_ratio, mac_ratio)
        record["anchor_params"] = anchor_params
        record["param_ratio_vs_anchor"] = param_ratio
        record["mac_ratio_vs_anchor"] = mac_ratio
        record["capacity_ratio"] = capacity_ratio
        record["exceeds_advisory"] = capacity_ratio > advisory_ratio
        # Every family must read exactly the anchor's window.  Capacity is the
        # variable this generation is allowed to move; the support is not.
        record["support_matches_anchor"] = record["support_span"] == target_rf
    return records


def print_capacity_table(records: list[dict], advisory_ratio: float = CAPACITY_ADVISORY_RATIO) -> None:
    """Render the audit as the capacity table, with its two caveats attached."""

    header = (
        f"{'operator':<20}{'role':>10}{'params':>9}{'xAnchor':>9}"
        f"{'MACs':>13}{'xAnchor':>9}{'hookOnly':>13}{'declared':>11}{'flag':>12}"
    )
    print(header)
    print("-" * len(header))
    for record in records:
        flag = ""
        if record["exceeds_advisory"]:
            flag = "control" if record["role"] == "control" else f">{advisory_ratio:.0f}x"
        print(
            f"{record['operator']:<20}{record['role']:>10}{record['params']:>9}"
            f"{record['param_ratio_vs_anchor']:>9.3f}{record['macs']:>13,}"
            f"{record['mac_ratio_vs_anchor']:>9.3f}{record['macs_hook_only']:>13,}"
            f"{record['macs_elementwise_declared']:>11,}{flag:>12}"
        )
    print(f"the {advisory_ratio:.0f}x ceiling is advisory and report-only: no test asserts it")
    print("wide_dilated rows are ablation controls, not pilot candidates")
    print("declared = elementwise work hooks cannot see; Matched gated/band_gated declare none")
