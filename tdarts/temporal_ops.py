"""Temporal operator pool for T-DARTS-FB, stage 1.

Four operator families are crossed with four receptive fields to give the
16 candidates per band that a later stage will relax into a DARTS MixedOp.

Design rules enforced by this module:

* every operator maps ``[B, in_channels, C, T] -> [B, out_channels, C, T]``;
* the time length ``T`` is preserved exactly (same-length convolution);
* the electrode dimension ``C`` is never touched -- temporal convolution uses
  ``kernel_size=(1, k)``, so it only ever sees the time axis;
* every operator of a given ``(band, target_rf)`` has the *same* effective
  receptive field, differing only in how it samples that span.

The four families, illustrated for ``band="Low"``, kernel 15 and target RF 57
(dilation 4):

============  ==============================================  ==========
name          layers                                          eff. RF
============  ==============================================  ==========
``dilated``   ``Conv2d(3, 6, (1,15), dilation=(1,4))``         57
``normal``    ``Conv2d(3, 6, (1,57), dilation=(1,1))``         57
``dwsep``     depthwise ``(1,15)`` d=4, then pointwise 1x1     57
``lkdw``      depthwise ``(1,57)`` d=1, then pointwise 1x1     57
============  ==============================================  ==========

``dilated`` and ``dwsep``/``lkdw`` sample their span sparsely; ``normal``
samples it densely.  The pointwise 1x1 convolution has RF 1 and so does not
change the effective receptive field of the separable variants.

Nothing here searches anything.  There is no alpha, no mixed op, no relaxation.
This stage only builds and audits the candidate pool.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from tdarts import config as C

__all__ = [
    "TemporalOp",
    "DilatedTemporalConv",
    "NormalTemporalConv",
    "DWSeparableTemporalConv",
    "LargeKernelDWTemporalConv",
    "OPERATOR_REGISTRY",
    "build_temporal_op",
    "build_all_candidates",
    "temporal_candidates",
    "canonical_candidates",
    "canonical_candidates_all",
    "num_canonical_candidates",
    "duplicate_groups",
    "resolve_kernel_dilation",
    "effective_rf",
    "count_parameters",
]


# ----------------------------------------------------------------------
# small helpers
# ----------------------------------------------------------------------
def effective_rf(kernel: int, dilation: int) -> int:
    """Effective receptive field of a dilated 1-D convolution.

    ``RF = 1 + (kernel - 1) * dilation``
    """
    if kernel < 1:
        raise ValueError(f"kernel must be >= 1, got {kernel}")
    if dilation < 1:
        raise ValueError(f"dilation must be >= 1, got {dilation}")
    return 1 + (kernel - 1) * dilation


def resolve_kernel_dilation(
    band: str, target_rf: int, base_kernel: int | None = None
) -> tuple[int, int]:
    """Return ``(kernel, dilation)`` realising ``target_rf`` for a band.

    The base kernel is looked up in :data:`tdarts.config.BASE_KERNEL` unless
    ``base_kernel`` is given explicitly.

    Raises
    ------
    ValueError
        If the band is unknown, the RF is not in the band's ``RF_SPACE``, or
        the RF cannot be realised exactly.  This is deliberately strict: an
        approximate RF would silently invalidate the
        "same effective RF, different mechanism" property the operator
        comparison rests on.
    """
    if band not in C.BASE_KERNEL:
        raise ValueError(
            f"unknown band {band!r}; expected one of "
            f"{sorted(C.BASE_KERNEL)}"
        )
    explicit_base = base_kernel is not None
    kernel = int(base_kernel) if explicit_base else C.BASE_KERNEL[band]

    # RF_SPACE describes the *stage-1* geometry (base kernel 15 for every band).
    # Supplying base_kernel explicitly means the caller is deliberately working
    # in a different geometry, so membership is not required -- but the value
    # must still be exactly realisable, which is checked below.
    if not explicit_base:
        expected = C.RF_SPACE.get(band)
        if expected is not None and target_rf not in expected:
            raise ValueError(
                f"band {band!r} does not offer target_rf={target_rf}; "
                f"RF_SPACE[{band!r}] = {expected}"
            )

    if kernel < 1:
        raise ValueError(f"base kernel for band {band!r} must be >= 1, got {kernel}")

    span = kernel - 1
    if span == 0:
        # kernel 1 gives RF 1 for every dilation; only RF 1 is representable.
        if target_rf != 1:
            raise ValueError(
                f"band {band!r}: kernel 1 cannot realise RF {target_rf}"
            )
        return 1, 1

    if (target_rf - 1) % span != 0:
        raise ValueError(
            f"band {band!r}: RF {target_rf} is not exactly realisable as "
            f"1 + ({kernel} - 1) * d, since {target_rf - 1} is not divisible "
            f"by {span}"
        )

    dilation = (target_rf - 1) // span
    # Cross-check rather than trust the arithmetic.
    if effective_rf(kernel, dilation) != target_rf:
        raise ValueError(
            f"band {band!r}: internal error resolving RF {target_rf} with "
            f"kernel {kernel}"
        )
    return kernel, dilation


def count_parameters(module: nn.Module, trainable_only: bool = True) -> int:
    """Number of scalar parameters in ``module``."""
    return sum(
        p.numel() for p in module.parameters() if p.requires_grad or not trainable_only
    )


def _same_length_padding(kernel: int, dilation: int) -> tuple[int, int]:
    """Padding for ``kernel_size=(1, kernel)``, ``dilation=(1, dilation)``.

    Returns ``(pad_channels, pad_time)``.  A ``(1, k)`` kernel does not span the
    channel axis, so that entry is always 0.

    With stride 1, ``output = T + 2*pad - dilation*(kernel-1)``.  So "same
    length" requires ``2*pad == dilation*(kernel-1)``, which is only possible
    with a *symmetric integer* padding when the span ``dilation*(kernel-1)`` is
    **even**.

    Every geometry in this stage satisfies that: the base kernel is 15, so the
    span is ``14 * d``, even for every d.  But the requirement is a property of
    the geometry, not a general truth -- an odd span (e.g. kernel 4, dilation 1,
    span 3) cannot be padded symmetrically and would silently produce
    ``T - 1`` outputs under ``// 2``.  Rather than quietly truncating the time
    axis, an odd span is rejected here.

    Supporting odd spans would need asymmetric padding (``F.pad`` with different
    left/right amounts) plus a decision about where the extra tap belongs; that
    is out of scope for this stage.
    """
    if kernel < 1:
        raise ValueError(f"kernel must be >= 1, got {kernel}")
    if dilation < 1:
        raise ValueError(f"dilation must be >= 1, got {dilation}")

    span = dilation * (kernel - 1)
    if span % 2 != 0:
        raise ValueError(
            f"cannot preserve time length for kernel={kernel}, "
            f"dilation={dilation}: the span dilation*(kernel-1) = {span} is odd, "
            f"so symmetric integer padding cannot equalise it. Same-length "
            f"padding needs an even span. Use an odd kernel with an odd or even "
            f"dilation, or add asymmetric-padding support."
        )
    return (0, span // 2)


# ----------------------------------------------------------------------
# operator families
# ----------------------------------------------------------------------
class _TemporalOpBase(nn.Module):
    """Shared plumbing: validation, resolved geometry, parameter init.

    Subclasses build their layers in :meth:`_build_layers` and must not change
    the ``(band, target_rf, in_channels, out_channels)`` contract.
    """

    #: Name used by the factory and in audit output.
    op_name: str = "base"

    def __init__(
        self,
        band: str,
        target_rf: int,
        in_channels: int = C.IN_CHANNELS,
        out_channels: int = C.PATH_CHANNELS,
        base_kernel: int | None = None,
    ) -> None:
        super().__init__()
        self.band = band
        self.target_rf = int(target_rf)
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)

        self.kernel, self.dilation = resolve_kernel_dilation(
            band, self.target_rf, base_kernel
        )
        padding = _same_length_padding(self.kernel, self.dilation)
        self.padding = tuple(padding)
        self.group_count = 1

        self._layers = nn.ModuleList()
        self._build_layers()

        if count_parameters(self, trainable_only=False) == 0:
            raise RuntimeError(
                f"{type(self).__name__} built zero parameters; a temporal "
                f"operator must be trainable"
            )

    def _build_layers(self) -> None:  # pragma: no cover - abstract
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError

    # -- reporting ------------------------------------------------------
    @property
    def num_params(self) -> int:
        return count_parameters(self)

    def describe(self) -> str:
        return (
            f"{self.op_name:<7} kernel={self.kernel:<4} dilation={self.dilation:<3} "
            f"RF={self.target_rf:<4} groups={self.group_count:<3} "
            f"params={self.num_params}"
        )


class DilatedTemporalConv(_TemporalOpBase):
    """Full convolution with a fixed small kernel and a large dilation.

    ``Conv2d(in, out, (1, K), dilation=(1, D))`` where ``K`` is the band base
    kernel and ``D`` is chosen so that ``1 + (K-1)*D`` equals the target RF.
    Samples the receptive field sparsely.
    """

    op_name = "dilated"

    def _build_layers(self) -> None:
        self.conv = nn.Conv2d(
            self.in_channels,
            self.out_channels,
            kernel_size=(1, self.kernel),
            dilation=(1, self.dilation),
            padding=self.padding,
            bias=False,
        )
        self._layers.append(self.conv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class NormalTemporalConv(_TemporalOpBase):
    """Plain convolution whose kernel length *is* the target RF.

    ``dilation = 1`` and ``kernel = target_rf``, so sampling is dense.  Its
    effective receptive field therefore matches the dilated operator of the
    same ``(band, target_rf)`` exactly -- the only difference is how densely
    the span is sampled.
    """

    op_name = "normal"

    def _build_layers(self) -> None:
        # Geometry is imposed, not resolved: kernel is the RF and dilation is 1.
        self.kernel = self.target_rf
        self.dilation = 1
        self.padding = _same_length_padding(self.kernel, self.dilation)
        self._sync_geometry()
        self.conv = nn.Conv2d(
            self.in_channels,
            self.out_channels,
            kernel_size=(1, self.kernel),
            dilation=(1, 1),
            padding=self.padding,
            bias=False,
        )
        self._layers.append(self.conv)

    def _sync_geometry(self) -> None:
        """No-op hook; ``normal`` resolves its geometry directly."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class _SeparableTemporalConv(_TemporalOpBase):
    """Depthwise temporal convolution followed by a pointwise projection.

    The depthwise stage sets the receptive field; the 1x1 pointwise stage has
    RF 1 and so leaves it unchanged.

    ``groups=in_channels`` (true depthwise): each input channel gets its own
    temporal filter, and the pointwise convolution mixes them into
    ``out_channels``.  This requires only ``out_channels % in_channels == 0``,
    which holds for the default 3 -> 6.

    The official FBNAS code instead uses ``ConvBn(3, 6, 15, ...)`` with
    ``groups=1`` (a dense 3->6 convolution, not depthwise); see
    ``docs/fbnas_audit.md``.  The depthwise form here is what this stage's
    specification calls for, and the two differ in parameter count only, not
    in shape or receptive field.
    """

    def _build_layers(self) -> None:
        if self.out_channels % self.in_channels != 0:
            raise ValueError(
                f"{self.op_name}: depthwise grouping needs out_channels "
                f"({self.out_channels}) divisible by in_channels "
                f"({self.in_channels})"
            )
        self.group_count = self.in_channels

        self.dw = nn.Conv2d(
            self.in_channels,
            self.out_channels,
            kernel_size=(1, self._dw_kernel()),
            dilation=(1, self._dw_dilation()),
            padding=self._dw_padding(),
            groups=self.group_count,
            bias=False,
        )
        self.pw = nn.Conv2d(
            self.out_channels,
            self.out_channels,
            kernel_size=(1, 1),
            bias=False,
        )
        self._layers.extend([self.dw, self.pw])
        # Report the depthwise geometry, which for lkdw differs from the
        # resolved dilated geometry handed down by the base class.
        self._sync_geometry()

    def _sync_geometry(self) -> None:
        """Point the reported geometry at the depthwise convolution.

        The base class resolves the *dilated* geometry (band base kernel), which
        is what :class:`DWSeparableTemporalConv` uses, so this is a no-op there.
        :class:`LargeKernelDWTemporalConv` overrides it because its depthwise
        convolution is the full target RF at dilation 1.
        """

    def _dw_kernel(self) -> int:
        return self.kernel

    def _dw_dilation(self) -> int:
        return self.dilation

    def _dw_padding(self) -> tuple[int, int]:
        return self.padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pw(self.dw(x))


class DWSeparableTemporalConv(_SeparableTemporalConv):
    """Depthwise dilated conv (band base kernel) + pointwise.

    For ``Low``/RF 57: depthwise ``(1,15)`` with dilation 4, then 1x1.
    """

    op_name = "dwsep"


class LargeKernelDWTemporalConv(_SeparableTemporalConv):
    """Depthwise *large-kernel dense* conv + pointwise.

    For ``Low``/RF 57: depthwise ``(1,57)`` with dilation 1, then 1x1.  Same
    effective RF as :class:`DWSeparableTemporalConv`, but densely sampled.
    """

    op_name = "lkdw"

    def _dw_kernel(self) -> int:
        return self.target_rf

    def _dw_dilation(self) -> int:
        return 1

    def _dw_padding(self) -> tuple[int, int]:
        return _same_length_padding(self.target_rf, 1)

    def _sync_geometry(self) -> None:
        # Unlike dwsep, this variant does not use the resolved dilated geometry:
        # its depthwise kernel is the full target RF at dilation 1.  Without
        # this the reporter would claim kernel=15/dilation=4 for a conv that is
        # actually kernel=57/dilation=1.
        self.kernel = self._dw_kernel()
        self.dilation = self._dw_dilation()
        self.padding = self._dw_padding()


OPERATOR_REGISTRY: dict[str, type[_TemporalOpBase]] = {
    "dilated": DilatedTemporalConv,
    "normal": NormalTemporalConv,
    "dwsep": DWSeparableTemporalConv,
    "lkdw": LargeKernelDWTemporalConv,
}


# ----------------------------------------------------------------------
# candidate wrapper
# ----------------------------------------------------------------------
class TemporalOp(nn.Module):
    """One temporal candidate: ``operator -> optional candidate normalisation``.

    ``use_norm=True`` appends ``norm_cls(out_channels, **norm_kwargs)``, by
    default ``BatchNorm2d(out_channels, affine=False)``.  Differently-scaled
    operators would otherwise bias a future softmax architecture gradient.

    ``use_norm=False`` substitutes :class:`torch.nn.Identity`, so the module can
    be evaluated in exactly the same way while the normalisation contributes
    nothing.  The receptive-field audit uses that mode: BatchNorm is affine and
    would distort a gradient-based receptive-field measurement.

    Note that the normalisation is *outside* the operator: ``.op`` is always the
    raw operator, so geometry and parameter counts can be inspected without
    the normalisation in the way.
    """

    def __init__(
        self,
        op_name: str,
        band: str,
        target_rf: int,
        in_channels: int = C.IN_CHANNELS,
        out_channels: int = C.PATH_CHANNELS,
        use_norm: bool = C.USE_CANDIDATE_NORM,
        base_kernel: int | None = None,
        norm_cls: type[nn.Module] | None = None,
        norm_kwargs: dict | None = None,
    ) -> None:
        super().__init__()
        if op_name not in OPERATOR_REGISTRY:
            raise ValueError(
                f"unknown operator {op_name!r}; expected one of "
                f"{sorted(OPERATOR_REGISTRY)}"
            )

        self.op_name = op_name
        self.band = band
        self.target_rf = int(target_rf)
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.use_norm = bool(use_norm)

        self.op = OPERATOR_REGISTRY[op_name](
            band=band,
            target_rf=target_rf,
            in_channels=in_channels,
            out_channels=out_channels,
            base_kernel=base_kernel,
        )

        if self.use_norm:
            kwargs = dict(norm_kwargs) if norm_kwargs else {}
            if norm_cls is None:
                # Default: BatchNorm2d(out_channels, affine=False).  affine=False
                # is the point -- the normalisation must not add trainable
                # parameters, or it would become part of what DARTS searches.
                self.norm: nn.Module = nn.BatchNorm2d(out_channels, affine=False, **kwargs)
            else:
                # An explicit class may want to receive the channel count under
                # a different keyword (GroupNorm uses num_channels).  Let
                # norm_kwargs win if it supplies one.
                if "num_channels" in kwargs and "in_channels" not in kwargs:
                    self.norm = norm_cls(**kwargs)
                else:
                    self.norm = norm_cls(out_channels, **kwargs)
        else:
            self.norm = nn.Identity()

    # -- geometry -------------------------------------------------------
    @property
    def kernel(self) -> int:
        """Kernel length of the receptive-field-defining convolution."""
        return self.op.kernel

    @property
    def dilation(self) -> int:
        return self.op.dilation

    @property
    def padding(self) -> tuple[int, int]:
        return self.op.padding

    @property
    def effective_rf(self) -> int:
        """Effective receptive field implied by the actual convolution layers.

        Computed from the built modules, so it reflects what the model really
        does rather than what was intended.
        """
        if isinstance(self.op, _SeparableTemporalConv):
            return effective_rf(self.op.dw.kernel_size[1], self.op.dw.dilation[1])
        return effective_rf(self.op.kernel, self.op.dilation)

    @property
    def num_params(self) -> int:
        """Trainable parameters in the whole candidate, normalisation included."""
        return count_parameters(self)

    @property
    def num_op_params(self) -> int:
        """Trainable parameters in the operator alone."""
        return count_parameters(self.op)

    @property
    def is_separable(self) -> bool:
        """True for the depthwise+pointwise families (``dwsep``, ``lkdw``)."""
        return isinstance(self.op, _SeparableTemporalConv)

    @property
    def structure_key(self) -> tuple[int, int, bool]:
        """Identity of the *function family* this candidate computes.

        Two candidates with the same key implement the same convolution
        structure and differ only in their random initialisation.  The key is
        ``(kernel, dilation, separable)``:

        * ``kernel``/``dilation`` are the depthwise geometry -- for the
          separable families the depthwise stage, since the 1x1 pointwise stage
          has RF 1;
        * ``separable`` distinguishes a dense ``3->6`` convolution from a
          depthwise ``3->6`` plus pointwise projection.  These can share a
          geometry while being different operators (and different parameter
          counts), so geometry alone is not enough to identify a family.

        Used by :func:`canonical_candidates` to collapse duplicates.
        """
        return (self.kernel, self.dilation, self.is_separable)

    def describe(self) -> str:
        return (
            f"{self.band:<4} {self.op_name:<7} RF{self.target_rf:<4} "
            f"kernel={self.kernel:<4} dilation={self.dilation:<3} "
            f"pad={self.padding[1]:<4} params={self.num_params}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.op(x))


# ----------------------------------------------------------------------
# factories
# ----------------------------------------------------------------------
def build_temporal_op(
    op_name: str,
    band: str,
    target_rf: int,
    in_channels: int = C.IN_CHANNELS,
    out_channels: int = C.PATH_CHANNELS,
    use_norm: bool = C.USE_CANDIDATE_NORM,
    base_kernel: int | None = None,
    norm_cls: type[nn.Module] | None = None,
    norm_kwargs: dict | None = None,
) -> TemporalOp:
    """Build one temporal candidate.

    Parameters
    ----------
    op_name:
        One of :data:`tdarts.config.OPERATORS`.
    band:
        ``"Low"``, ``"Mid"`` or ``"High"``.
    target_rf:
        Target effective receptive field; must appear in the band's
        ``RF_SPACE`` entry.
    in_channels, out_channels:
        Channel counts.  3 -> 6 by default, matching one filter-bank group in
        and one temporal path out.
    use_norm:
        Apply the candidate-level normalisation.  Set ``False`` for
        receptive-field auditing.
    base_kernel:
        Override the band base kernel.  Tests use this to exercise a different
        geometry; production call sites leave it as ``None``.
    norm_cls, norm_kwargs:
        Override the normalisation layer.  Defaults to
        ``BatchNorm2d(out_channels, affine=False)``.
    """
    return TemporalOp(
        op_name=op_name,
        band=band,
        target_rf=target_rf,
        in_channels=in_channels,
        out_channels=out_channels,
        use_norm=use_norm,
        base_kernel=base_kernel,
        norm_cls=norm_cls,
        norm_kwargs=norm_kwargs,
    )


def temporal_candidates() -> list[tuple[str, str, int]]:
    """Every ``(band, op_name, target_rf)`` triple in the *declared* grid.

    ``3 bands x 4 operators x 4 receptive fields = 48`` triples.  Ordering is
    deterministic (band, then operator, then ascending RF) so audit output is
    stable and diffable.

    This is the full cross product and deliberately **contains duplicates**: at
    the smallest receptive field the dilation is 1, so ``dilated`` coincides
    with ``normal`` and ``dwsep`` with ``lkdw``.  Use
    :func:`canonical_candidates` for the distinct structure set that a search
    should actually range over.
    """
    return [
        (band, op_name, rf)
        for band in ("Low", "Mid", "High")
        for op_name in C.OPERATORS
        for rf in C.RF_SPACE[band]
    ]


def duplicate_groups(band: str = "Low") -> dict[tuple[int, int, bool], list[tuple[str, int]]]:
    """Group a band's declared grid by :attr:`TemporalOp.structure_key`.

    Returns ``{key: [(op_name, target_rf), ...]}``.  Any key with more than one
    member is a collapsed duplicate: several names for one function family.
    """
    pool = build_all_candidates(use_norm=False)
    groups: dict[tuple[int, int, bool], list[tuple[str, int]]] = {}
    for op_name in C.OPERATORS:
        for rf in C.RF_SPACE[band]:
            groups.setdefault(pool[(band, op_name, rf)].structure_key, []).append(
                (op_name, rf)
            )
    return groups


def canonical_candidates(band: str) -> list[tuple[str, str, int]]:
    """The distinct structure set for one band: 14, not 16.

    Walks the declared grid in :func:`temporal_candidates` order and keeps the
    first triple for each :attr:`TemporalOp.structure_key`.  Collapsing is
    derived from the built geometry, not hard-coded, so it stays correct if the
    RF ladder or base kernel changes.

    Why this matters for a gradient-based search
    -------------------------------------------
    Keeping both names for one family would give that family two independent
    sets of logits, and therefore two shares of the softmax mass, while
    ``argmax`` sees only one of them.  A family that is genuinely preferred
    could split its probability across two aliases and lose the argmax to a
    weaker but uniquely-named candidate -- the architecture decision would then
    reflect naming, not structure.

    At ``kernel = 15`` the duplicates are exactly the RF15 pair, giving
    ``2 + 4 + 4 + 4 = 14`` distinct structures per band.
    """
    seen: set[tuple[int, int, bool]] = set()
    pool = build_all_candidates(use_norm=False)
    out: list[tuple[str, str, int]] = []
    for op_name in C.OPERATORS:
        for rf in C.RF_SPACE[band]:
            key = pool[(band, op_name, rf)].structure_key
            if key in seen:
                continue
            seen.add(key)
            out.append((band, op_name, rf))
    return out


def canonical_candidates_all() -> dict[str, list[tuple[str, str, int]]]:
    """Canonical structure set for each band, keyed by band name."""
    return {band: canonical_candidates(band) for band in ("Low", "Mid", "High")}


def num_canonical_candidates() -> int:
    """Total distinct candidates across bands (42 for the stage-1 ladder)."""
    return sum(len(v) for v in canonical_candidates_all().values())


def build_all_candidates(
    use_norm: bool = C.USE_CANDIDATE_NORM,
    **kwargs,
) -> dict[tuple[str, str, int], TemporalOp]:
    """Build the whole candidate pool, keyed by ``(band, op_name, target_rf)``.

    The tuple order matches :func:`temporal_candidates`.  Note it is *not* the
    argument order of :func:`build_temporal_op`, so the fields are passed by
    keyword rather than splatted.
    """
    return {
        (band, op_name, rf): build_temporal_op(
            op_name=op_name,
            band=band,
            target_rf=rf,
            use_norm=use_norm,
            **kwargs,
        )
        for band, op_name, rf in temporal_candidates()
    }
