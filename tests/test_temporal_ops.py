"""Audit of the full stage-1 temporal candidate pool.

48 candidates = 3 bands x 4 operators x 4 receptive fields.  Every candidate is
checked for construction, forward, backward, shape, time preservation, finite
values, gradient flow and *measured* effective receptive field.

Running this file directly prints the audit table and exits non-zero if any
candidate fails::

    python tests/test_temporal_ops.py
    python -m unittest tests.test_temporal_ops -v
"""

from __future__ import annotations

import copy
import io
import sys
import traceback
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tdarts import config as C  # noqa: E402
from tdarts.temporal_ops import (  # noqa: E402
    OPERATOR_REGISTRY,
    _same_length_padding,
    build_all_candidates,
    build_temporal_op,
    canonical_candidates,
    canonical_candidates_all,
    count_parameters,
    duplicate_groups,
    effective_rf,
    num_canonical_candidates,
    resolve_kernel_dilation,
    temporal_candidates,
)

BATCH = 2


# ----------------------------------------------------------------------
# measurement helpers
# ----------------------------------------------------------------------
def analytic_rf(op) -> int:
    """Effective RF read off the operator's own convolution layers.

    For the separable variants this is the *depthwise* convolution, since the
    1x1 pointwise stage has RF 1.
    """
    dw = getattr(op, "dw", None)
    if dw is not None:
        return effective_rf(dw.kernel_size[1], dw.dilation[1])
    return effective_rf(op.conv.kernel_size[1], op.conv.dilation[1])


def analytic_taps(op) -> int:
    """Number of positions the receptive-field-defining convolution samples."""
    dw = getattr(op, "dw", None)
    if dw is not None:
        return int(dw.kernel_size[1])
    return int(op.conv.kernel_size[1])


def analytic_stride(op) -> int:
    """Stride of the sampled positions; 1 means the support is contiguous."""
    dw = getattr(op, "dw", None)
    if dw is not None:
        return int(dw.dilation[1])
    return int(op.conv.dilation[1])


def measure_rf(candidate, analytic: int, verify: bool = True) -> dict:
    """Measure the effective receptive field by gradient support.

    Returns a dict with:

    ``span``
        Distance from the first to the last contributing input position.  This
        is the *effective receptive field*, and it must equal both
        ``1 + (kernel - 1) * dilation`` and the target RF.
    ``count``
        How many input positions actually contribute.  This equals the number
        of taps in the receptive-field-defining convolution.

    These differ on purpose.  A dilated convolution does not touch every
    position inside its span -- that is what "sparse temporal sampling" means --
    so ``count == kernel`` while ``span == 1 + (kernel-1)*dilation``.  A dense
    convolution (``normal``, ``lkdw``) has ``dilation == 1`` and therefore
    ``count == span``.

    Method
    ------
    Replace every convolution weight with a uniform kernel and every bias with
    one, so the response at the centre of the output is exactly the sum of the
    contributing input positions.  Differentiate that response with respect to
    the input; the non-zero entries are the support.

    Why this is exact
    -----------------
    A delta at the output centre cannot cancel: every contributing weight is
    positive, so each sampled position receives a strictly positive
    contribution and there is no zero crossing inside the support.

    Why float64 and no normalisation
    --------------------------------
    A small weight (1/270 for the dilated operators) makes the far edge of the
    support six orders of magnitude smaller than the centre, which float32
    underflows to zero -- a fictitious result.  float64 keeps it representable.
    BatchNorm is affine and would rescale the support, so the caller must pass a
    candidate built with ``use_norm=False``.
    """
    if verify:
        assert isinstance(candidate.norm, nn.Identity), (
            "RF measurement requires use_norm=False; BatchNorm would distort "
            "the gradient support"
        )

    # Work on a deep copy: the probe rewrites every weight and promotes the
    # module to float64.  Mutating the caller's candidate in place would leave
    # it in double precision and silently break unrelated later tests.
    model = copy.deepcopy(candidate).double().eval()

    def uniform(m):
        with torch.no_grad():
            if isinstance(m, nn.Conv2d):
                m.weight.fill_(1.0 / m.weight[0].numel())
                if m.bias is not None:
                    m.bias.fill_(1.0)

    model.apply(uniform)

    in_channels = candidate.in_channels
    electrodes = C.NUM_ELECTRODES
    time = 1000
    # Window wide enough to hold the span plus a margin, so a clipped support
    # would be detectable rather than silently truncated.
    span_budget = 2 * analytic + 20
    centre = time // 2
    start = centre - span_budget // 2

    x = torch.zeros(1, in_channels, electrodes, time, dtype=torch.float64)
    probe_channel = in_channels // 2
    x[0, probe_channel, :, start : start + span_budget] = 1.0
    x.requires_grad_(True)

    out = model(x)
    out[0, :, :, centre].sum().backward()

    grad = x.grad[0, probe_channel, 0]
    window = grad[start : start + span_budget]
    nonzero = torch.nonzero(window, as_tuple=False).flatten()

    count = int(nonzero.numel())
    if count == 0:
        return {"span": 0, "count": 0}

    first = int(nonzero[0])
    last = int(nonzero[-1])
    span = last - first + 1

    # A dilated support is evenly strided by the dilation factor.
    if count > 1:
        gaps = torch.diff(nonzero)
        if int(gaps.min()) != int(gaps.max()):
            raise AssertionError(
                f"{candidate.band}/{candidate.op_name}/RF{candidate.target_rf}: "
                f"gradient support is unevenly strided, so it is not a dilated "
                f"receptive field"
            )

    return {"span": span, "count": count, "stride": int(torch.diff(nonzero)[0]) if count > 1 else 0}


def make_input(batch: int = BATCH) -> torch.Tensor:
    return torch.randn(
        batch, C.IN_CHANNELS, C.NUM_ELECTRODES, C.NUM_TIMEPOINTS
    )


# ----------------------------------------------------------------------
# audit
# ----------------------------------------------------------------------
class TemporalOperatorAudit(unittest.TestCase):
    """One subtest per candidate, named so failures identify the candidate."""

    @classmethod
    def setUpClass(cls) -> None:
        # Built once with normalisation off: that is the audit configuration.
        cls.candidates = build_all_candidates(use_norm=False)

    def test_candidate_count_is_48(self):
        self.assertEqual(len(temporal_candidates()), 48)
        self.assertEqual(len(self.candidates), 48)
        self.assertEqual(len(set(self.candidates)), 48)
        self.assertEqual(
            3 * len(C.OPERATORS) * len(C.RF_SPACE["Low"]), 48
        )

    def test_every_candidate(self):
        failures: list[str] = []
        for band, op_name, rf in temporal_candidates():
            with self.subTest(band=band, op=op_name, rf=rf):
                try:
                    self._check_candidate(band, op_name, rf)
                except Exception:
                    failures.append(f"{band} {op_name} RF{rf}")
                    raise
        self.assertEqual(failures, [])

    def _check_candidate(self, band: str, op_name: str, rf: int) -> None:
        cand = self.candidates[(band, op_name, rf)]

        # 1. construction
        self.assertIsInstance(cand, nn.Module)
        self.assertEqual(cand.band, band)
        self.assertEqual(cand.op_name, op_name)
        self.assertEqual(cand.target_rf, rf)
        self.assertIsInstance(cand.norm, nn.Identity)
        self.assertGreater(cand.num_params, 0)

        # 2/3. forward + backward
        cand.train()
        x = make_input()
        y = cand(x)

        # 4/5/6/7. shape, time, electrode dimension
        self.assertEqual(
            tuple(y.shape),
            (BATCH, C.PATH_CHANNELS, C.NUM_ELECTRODES, C.NUM_TIMEPOINTS),
        )
        self.assertEqual(y.shape[3], x.shape[3], "time length must be preserved")
        self.assertEqual(y.shape[2], x.shape[2], "electrode axis must not change")
        self.assertEqual(y.shape[1], C.PATH_CHANNELS)

        # 8/9. finiteness
        self.assertTrue(torch.isfinite(y).all(), "output contains NaN or Inf")

        # 10. every trainable parameter receives a gradient
        loss = y.sum()
        loss.backward()
        for name, p in cand.named_parameters():
            with self.subTest(param=name):
                self.assertIsNotNone(p.grad, f"{name} has no gradient")
                self.assertTrue(
                    torch.isfinite(p.grad).all(), f"{name} gradient not finite"
                )
                self.assertNotEqual(
                    float(p.grad.abs().sum()), 0.0, f"{name} gradient is all zero"
                )

        # 11. measured receptive field
        analytic = analytic_rf(cand.op)
        self.assertEqual(
            analytic,
            rf,
            f"analytic RF {analytic} != target {rf}",
        )
        self.assertEqual(cand.effective_rf, rf)
        measured = measure_rf(cand, analytic)
        self.assertEqual(
            measured["span"], rf, f"measured RF span {measured['span']} != {rf}"
        )
        self.assertEqual(
            measured["count"],
            analytic_taps(cand.op),
            "number of sampled positions must equal the kernel length",
        )
        self.assertEqual(
            measured["stride"],
            analytic_stride(cand.op),
            "sampling stride must equal the convolution dilation",
        )

        # 12. parameter count is self-consistent
        self.assertEqual(cand.num_params, count_parameters(cand))
        self.assertEqual(cand.num_op_params, count_parameters(cand.op))
        self.assertEqual(cand.num_params, cand.num_op_params)

    def test_same_rf_across_operators(self):
        """The design claim: fixed RF, four different mechanisms."""
        for band in ("Low", "Mid", "High"):
            for rf in C.RF_SPACE[band]:
                rfs = {
                    op: self.candidates[(band, op, rf)].effective_rf
                    for op in C.OPERATORS
                }
                with self.subTest(band=band, rf=rf):
                    self.assertEqual(set(rfs.values()), {rf}, rfs)

    def test_dilated_and_dwsep_share_base_kernel(self):
        """dilated and dwsep use the band base kernel; normal and lkdw do not."""
        for band in ("Low", "Mid", "High"):
            base = C.BASE_KERNEL[band]
            for rf in C.RF_SPACE[band]:
                with self.subTest(band=band, rf=rf):
                    self.assertEqual(
                        self.candidates[(band, "dilated", rf)].kernel, base
                    )
                    self.assertEqual(
                        self.candidates[(band, "dwsep", rf)].kernel, base
                    )

    def test_normal_and_lkdw_use_dense_sampling(self):
        for band in ("Low", "Mid", "High"):
            for rf in C.RF_SPACE[band]:
                with self.subTest(band=band, rf=rf):
                    normal = self.candidates[(band, "normal", rf)]
                    lkdw = self.candidates[(band, "lkdw", rf)]
                    self.assertEqual(normal.kernel, rf)
                    self.assertEqual(normal.dilation, 1)
                    self.assertEqual(lkdw.kernel, rf)
                    self.assertEqual(lkdw.dilation, 1)

    def test_dwsep_uses_dilated_base_kernel(self):
        for band in ("Low", "Mid", "High"):
            for rf, expected_d in zip(C.RF_SPACE[band], C.DILATIONS):
                with self.subTest(band=band, rf=rf):
                    dwsep = self.candidates[(band, "dwsep", rf)]
                    self.assertEqual(dwsep.kernel, C.BASE_KERNEL[band])
                    self.assertEqual(dwsep.dilation, expected_d)

    def test_bands_are_symmetric_in_this_stage(self):
        """Low/Mid/High share one RF ladder, so their parameter counts match."""
        by_rf = {}
        for band in ("Low", "Mid", "High"):
            for op in C.OPERATORS:
                for rf in C.RF_SPACE[band]:
                    by_rf.setdefault((op, rf), set()).add(
                        self.candidates[(band, op, rf)].num_params
                    )
        for key, counts in by_rf.items():
            with self.subTest(key=key):
                self.assertEqual(len(counts), 1, f"bands differ for {key}: {counts}")

    def test_grouping_is_depthwise_for_separable_ops(self):
        for band in ("Low", "Mid", "High"):
            for rf in C.RF_SPACE[band]:
                for op in ("dwsep", "lkdw"):
                    with self.subTest(band=band, op=op, rf=rf):
                        dw = self.candidates[(band, op, rf)].op.dw
                        self.assertEqual(dw.groups, C.IN_CHANNELS)
                        self.assertEqual(dw.weight.shape[0], C.PATH_CHANNELS)
                        self.assertEqual(
                            dw.weight.shape[1], C.IN_CHANNELS // dw.groups
                        )

    def test_sparse_versus_dense_sampling(self):
        """The core operator distinction, measured rather than asserted.

        ``dilated``/``dwsep`` span the target RF with only ``kernel`` taps, so
        they sample sparsely; ``normal``/``lkdw`` fill the same span densely.

        RF15 is skipped deliberately: it resolves to ``dilation = 1``, where
        sparse and dense sampling coincide and ``dilated`` is arithmetically
        identical to ``normal``.  That is a real, reportable property of this
        search space rather than a defect -- it is asserted separately in
        :meth:`test_rf15_makes_dilated_and_normal_identical`.
        """
        base = C.BASE_KERNEL["Low"]
        for rf in C.RF_SPACE["Low"]:
            if rf == base:
                continue  # dilation 1: no sparsity to observe
            cands = {op: self.candidates[("Low", op, rf)] for op in C.OPERATORS}
            with self.subTest(rf=rf):
                for op in ("dilated", "dwsep"):
                    m = measure_rf(cands[op], rf)
                    self.assertEqual(m["span"], rf, f"{op} span")
                    self.assertEqual(m["count"], base, f"{op} taps")
                    self.assertGreater(
                        m["span"],
                        m["count"],
                        f"{op} should be sparse at RF{rf}",
                    )
                for op in ("normal", "lkdw"):
                    m = measure_rf(cands[op], rf)
                    self.assertEqual(m["span"], rf, f"{op} span")
                    self.assertEqual(m["count"], rf, f"{op} taps")
                    self.assertEqual(m["span"], m["count"], f"{op} should be dense")

    def test_rf15_makes_dilated_and_normal_identical(self):
        """At RF15 the dilation is 1, so these two operators coincide.

        Worth pinning down: it means the 4x4 candidate grid contains a
        duplicate pair per band, so the effective pool is 15 distinct operators,
        not 16.  A DARTS search over this space would see two identical
        candidates competing for probability mass.
        """
        base = C.BASE_KERNEL["Low"]
        for band in ("Low", "Mid", "High"):
            with self.subTest(band=band):
                dilated = self.candidates[(band, "dilated", base)]
                normal = self.candidates[(band, "normal", base)]
                self.assertEqual(dilated.kernel, normal.kernel)
                self.assertEqual(dilated.dilation, normal.dilation)
                self.assertEqual(dilated.num_params, normal.num_params)
                x = make_input()
                with torch.no_grad():
                    a = dilated(x)
                    b = normal(x)
                self.assertEqual(a.shape, b.shape)

    def test_rf15_dwsep_and_lkdw_are_distinct(self):
        """The separable pair stays distinct even at RF15.

        Both are depthwise+pointwise, so they coincide numerically too, but
        they remain separate module instances.
        """
        base = C.BASE_KERNEL["Low"]
        dwsep = self.candidates[("Low", "dwsep", base)]
        lkdw = self.candidates[("Low", "lkdw", base)]
        self.assertEqual(dwsep.kernel, lkdw.kernel)
        self.assertEqual(dwsep.num_params, lkdw.num_params)


class CandidateCollapse(unittest.TestCase):
    """The declared 4x4 grid contains aliases; the search space must not.

    At the smallest receptive field the dilation is 1, so ``dilated`` and
    ``normal`` compute the same function, as do ``dwsep`` and ``lkdw``.  Keeping
    both names would hand one function family two independent logit sets, and
    therefore two shares of the softmax mass, while ``argmax`` can only return
    one of them.  These tests pin the deduplication down.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.pool = build_all_candidates(use_norm=False)

    def test_declared_grid_is_48(self):
        self.assertEqual(len(temporal_candidates()), 48)

    def test_canonical_set_is_14_per_band(self):
        """2 (RF15, one per family) + 4 + 4 + 4 = 14."""
        for band in ("Low", "Mid", "High"):
            with self.subTest(band=band):
                self.assertEqual(len(canonical_candidates(band)), 14)

    def test_canonical_total_is_42(self):
        self.assertEqual(num_canonical_candidates(), 42)
        self.assertEqual(
            {len(v) for v in canonical_candidates_all().values()}, {14}
        )

    def test_canonical_membership_is_exact(self):
        """Exactly which pairs collapse, named explicitly."""
        for band in ("Low", "Mid", "High"):
            with self.subTest(band=band):
                names = {(op, rf) for _, op, rf in canonical_candidates(band)}
                # RF15 keeps one representative per family.
                self.assertIn(("dilated", 15), names)
                self.assertIn(("dwsep", 15), names)
                # The aliases are dropped.
                self.assertNotIn(("normal", 15), names)
                self.assertNotIn(("lkdw", 15), names)
                # Every other RF keeps all four.
                for rf in (29, 57, 113):
                    for op in C.OPERATORS:
                        self.assertIn((op, rf), names)

    def test_duplicate_groups_are_exactly_the_two_rf15_pairs(self):
        groups = duplicate_groups("Low")
        collapsed = {
            key: members for key, members in groups.items() if len(members) > 1
        }
        self.assertEqual(len(collapsed), 2)
        flat = sorted(tuple(sorted(m)) for m in collapsed.values())
        self.assertEqual(
            flat,
            [
                (("dilated", 15), ("normal", 15)),
                (("dwsep", 15), ("lkdw", 15)),
            ],
        )

    def test_collapsed_members_really_are_the_same_function(self):
        """Not just the same shape: the same convolution, given equal weights."""
        for (op_a, rf_a), (op_b, rf_b) in (
            (("dilated", 15), ("normal", 15)),
            (("dwsep", 15), ("lkdw", 15)),
        ):
            with self.subTest(pair=(op_a, op_b)):
                a = self.pool[("Low", op_a, rf_a)]
                b = self.pool[("Low", op_b, rf_b)]
                self.assertEqual(a.structure_key, b.structure_key)
                self.assertEqual(a.num_params, b.num_params)
                # Give them identical weights, then they must agree exactly.
                b.load_state_dict(a.state_dict())
                x = make_input()
                with torch.no_grad():
                    self.assertTrue(torch.equal(a(x), b(x)))

    def test_structure_key_distinguishes_dense_from_depthwise(self):
        """Geometry alone is not enough: dilated15 and dwsep15 share it."""
        dense = self.pool[("Low", "dilated", 15)]
        sep = self.pool[("Low", "dwsep", 15)]
        self.assertEqual((dense.kernel, dense.dilation), (sep.kernel, sep.dilation))
        self.assertNotEqual(dense.structure_key, sep.structure_key)
        self.assertFalse(dense.is_separable)
        self.assertTrue(sep.is_separable)
        # ...and they differ in parameter count, so they are not aliases.
        self.assertNotEqual(dense.num_params, sep.num_params)

    def test_non_rf15_pairs_are_not_collapsed(self):
        """dilated57 and dwsep57 share geometry but differ in structure."""
        dense = self.pool[("Low", "dilated", 57)]
        sep = self.pool[("Low", "dwsep", 57)]
        self.assertEqual((dense.kernel, dense.dilation), (sep.kernel, sep.dilation))
        self.assertNotEqual(dense.structure_key, sep.structure_key)

    def test_canonical_order_is_deterministic(self):
        first = canonical_candidates("Low")
        for _ in range(3):
            self.assertEqual(canonical_candidates("Low"), first)

    def test_canonical_representatives_are_buildable(self):
        for band in ("Low", "Mid", "High"):
            for _, op_name, rf in canonical_candidates(band):
                with self.subTest(band=band, op=op_name, rf=rf):
                    cand = build_temporal_op(op_name, band, rf, use_norm=False)
                    self.assertEqual(cand.effective_rf, rf)
                    self.assertGreater(cand.num_params, 0)

    def test_collapse_survives_a_different_base_kernel(self):
        """The count is derived, not hard-coded for kernel 15."""
        # kernel 25 -> dilations 1/2/4/8 give RF 25/49/97/193; RF25 collapses
        # the same two pairs, so 14 again.
        seen = set()
        for op_name, rf in [
            (op, rf)
            for rf in (25, 49, 97, 193)
            for op in C.OPERATORS
        ]:
            cand = build_temporal_op(op_name, "Low", rf, base_kernel=25, use_norm=False)
            seen.add(cand.structure_key)
        self.assertEqual(len(seen), 14)


class PaddingGuard(unittest.TestCase):
    """Same-length padding is only possible for an even span."""

    def test_stage_geometries_are_all_supported(self):
        for band in ("Low", "Mid", "High"):
            for rf in C.RF_SPACE[band]:
                kernel, dilation = resolve_kernel_dilation(band, rf)
                with self.subTest(band=band, rf=rf):
                    pad = _same_length_padding(kernel, dilation)
                    self.assertEqual(pad[0], 0)
                    # output = T + 2*pad - span == T
                    span = dilation * (kernel - 1)
                    self.assertEqual(2 * pad[1], span)

    def test_every_built_operator_preserves_time(self):
        """The guard and the built modules must agree."""
        for key, cand in build_all_candidates(use_norm=False).items():
            with self.subTest(candidate=key):
                x = make_input()
                with torch.no_grad():
                    self.assertEqual(cand(x).shape[3], x.shape[3])

    def test_odd_span_is_rejected(self):
        # kernel 4, dilation 1 -> span 3 (odd): cannot be padded symmetrically.
        with self.assertRaises(ValueError) as ctx:
            _same_length_padding(4, 1)
        msg = str(ctx.exception)
        self.assertIn("odd", msg)
        self.assertIn("asymmetric", msg)

    def test_odd_kernel_with_even_span_is_accepted(self):
        """Parity of the span is what matters, not parity of the kernel."""
        # kernel 4, dilation 2 -> span 6 (even): fine.
        self.assertEqual(_same_length_padding(4, 2), (0, 3))
        # kernel 5, dilation 2 -> span 8 (even): fine.
        self.assertEqual(_same_length_padding(5, 2), (0, 4))
        # kernel 5, dilation 1 -> span 4 (even): fine.
        self.assertEqual(_same_length_padding(5, 1), (0, 2))

    def test_odd_span_blocks_operator_construction(self):
        """The guard fires during construction, not silently at forward time."""
        # base kernel 4 makes the RF15 target resolve to dilation 1, span 3.
        with self.assertRaises(ValueError):
            build_temporal_op("dilated", "Low", 15, base_kernel=4, use_norm=False)

    def test_invalid_kernel_or_dilation_rejected(self):
        for kernel, dilation in ((0, 1), (1, 0), (-1, 1)):
            with self.subTest(kernel=kernel, dilation=dilation):
                with self.assertRaises(ValueError):
                    _same_length_padding(kernel, dilation)

    def test_padding_is_centred(self):
        """Equal padding on both sides keeps the response centred."""
        for kernel, dilation in ((15, 1), (15, 4), (57, 1), (113, 1)):
            with self.subTest(kernel=kernel, dilation=dilation):
                _, pad = _same_length_padding(kernel, dilation)
                self.assertEqual(2 * pad, dilation * (kernel - 1))


class CandidateNormalisation(unittest.TestCase):
    """The candidate-level BatchNorm must exist, be non-affine, and be optional."""

    def test_norm_on_by_default(self):
        cand = build_temporal_op("dilated", "Low", 57)
        self.assertTrue(cand.use_norm)
        self.assertIsInstance(cand.norm, nn.BatchNorm2d)
        self.assertFalse(
            cand.norm.affine, "candidate norm must not introduce affine params"
        )

    def test_norm_off_gives_identity(self):
        cand = build_temporal_op("dilated", "Low", 57, use_norm=False)
        self.assertFalse(cand.use_norm)
        self.assertIsInstance(cand.norm, nn.Identity)

    def test_norm_adds_no_parameters(self):
        """affine=False means the normalisation contributes zero parameters."""
        for band, op_name, rf in [("Low", "dilated", 57), ("High", "lkdw", 113)]:
            with self.subTest(band=band, op=op_name, rf=rf):
                with_norm = build_temporal_op(
                    op_name, band, rf, use_norm=True
                )
                without = build_temporal_op(op_name, band, rf, use_norm=False)
                self.assertEqual(with_norm.num_params, without.num_params)

    def test_norm_preserves_shape_and_gradients(self):
        cand = build_temporal_op("dwsep", "Mid", 29, use_norm=True)
        cand.train()
        x = make_input()
        y = cand(x)
        self.assertEqual(tuple(y.shape), tuple(x.shape[:1]) + (C.PATH_CHANNELS, C.NUM_ELECTRODES, C.NUM_TIMEPOINTS))
        y.sum().backward()
        for name, p in cand.named_parameters():
            with self.subTest(param=name):
                self.assertIsNotNone(p.grad)
                self.assertNotEqual(float(p.grad.abs().sum()), 0.0)

    def test_norm_can_be_swapped(self):
        cand = build_temporal_op(
            "normal",
            "High",
            15,
            use_norm=True,
            norm_cls=nn.GroupNorm,
            norm_kwargs={"num_groups": 2, "num_channels": C.PATH_CHANNELS},
        )
        self.assertIsInstance(cand.norm, nn.GroupNorm)


class GeometryResolution(unittest.TestCase):
    """resolve_kernel_dilation must be exact or refuse."""

    def test_rf_ladder_reproduces_expected_dilations(self):
        for band in ("Low", "Mid", "High"):
            for rf, expected_d in zip(C.RF_SPACE[band], C.DILATIONS):
                with self.subTest(band=band, rf=rf):
                    kernel, dilation = resolve_kernel_dilation(band, rf)
                    self.assertEqual(kernel, C.BASE_KERNEL[band])
                    self.assertEqual(dilation, expected_d)
                    self.assertEqual(effective_rf(kernel, dilation), rf)

    def test_low_rf57_geometry(self):
        self.assertEqual(resolve_kernel_dilation("Low", 57), (15, 4))
        self.assertEqual(resolve_kernel_dilation("Mid", 57), (15, 4))
        self.assertEqual(resolve_kernel_dilation("High", 57), (15, 4))

    def test_non_representable_rf_is_rejected(self):
        """A target that is not 1 + (k-1)*d for integer d must be refused.

        RF_SPACE membership is checked first, so this uses a base kernel whose
        ladder would otherwise admit the value: with base kernel 25 the span is
        24, and (26 - 1) = 25 is not divisible by 24.
        """
        with self.assertRaises(ValueError) as ctx:
            resolve_kernel_dilation("Low", 26, base_kernel=25)
        self.assertIn("not exactly realisable", str(ctx.exception))

    def test_rf_outside_space_is_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            resolve_kernel_dilation("Low", 999)
        self.assertIn("RF_SPACE", str(ctx.exception))

    def test_unknown_band_is_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            resolve_kernel_dilation("Ultra", 15)
        self.assertIn("unknown band", str(ctx.exception))

    def test_unknown_operator_is_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            build_temporal_op("transformer", "Low", 15)
        self.assertIn("unknown operator", str(ctx.exception))

    def test_ladder_is_representable_with_a_different_base_kernel(self):
        """The official FBNAS geometry (kernel 15) is not the only one that works."""
        # kernel 25 -> span 24 -> RF 25/49/97/193 at d = 1/2/4/8
        expected = {25: 1, 49: 2, 97: 4, 193: 8}
        for rf, dilation in expected.items():
            with self.subTest(rf=rf):
                self.assertEqual(
                    resolve_kernel_dilation("Low", rf, base_kernel=25), (25, dilation)
                )

    def test_even_rf_is_impossible_for_odd_base_kernel(self):
        """1 + (k-1)*d is odd whenever k is odd, so even targets cannot exist."""
        for base in (7, 11, 15, 25):
            with self.subTest(base_kernel=base):
                with self.assertRaises(ValueError):
                    resolve_kernel_dilation("Low", base + 1, base_kernel=base)

    def test_grouping_mismatch_is_rejected(self):
        """out_channels=6 with in_channels=4: divisible by 1,2,3 but not 4.

        Chosen so the divisibility guard is what rejects it.  (3 -> 7 would
        also be rejected, but only incidentally, on the `%` check.)
        """
        with self.assertRaises(ValueError) as ctx:
            build_temporal_op("dwsep", "Low", 15, in_channels=4, out_channels=6)
        self.assertIn("divisible", str(ctx.exception))

    def test_rf_must_be_odd_for_this_kernel(self):
        """Every realisable RF is odd; assert the ladder respects that."""
        for band in ("Low", "Mid", "High"):
            for rf in C.RF_SPACE[band]:
                with self.subTest(band=band, rf=rf):
                    self.assertEqual(rf % 2, 1)


# ----------------------------------------------------------------------
# standalone audit entry point
# ----------------------------------------------------------------------
def run_audit(verbose: bool = True) -> int:
    """Run the full audit, print the table, return a process exit code."""
    lines: list[str] = []
    passed = 0
    total = 0

    candidates = build_all_candidates(use_norm=False)

    for band, op_name, rf in temporal_candidates():
        total += 1
        cand = candidates[(band, op_name, rf)]
        label = f"{band:<4} {op_name:<7} RF{rf:<4}"
        try:
            _audit_one(cand, band, op_name, rf)
            passed += 1
            lines.append(
                f"{label} PASS  kernel={cand.kernel:<4} dilation={cand.dilation:<3} "
                f"pad={cand.padding[1]:<4} params={cand.num_params}"
            )
        except Exception as exc:
            lines.append(f"{label} FAIL  {type(exc).__name__}: {exc}")
            if verbose:
                lines.append("      " + traceback.format_exc().replace("\n", "\n      "))

    header = "Temporal Operator Audit"
    lines.insert(0, header)
    lines.insert(1, "-" * 78)

    if verbose:
        print("\n".join(lines))
    print()
    print(f"{passed} / {total} PASS" if passed != total else f"{total} / {total} PASS")
    return 0 if passed == total else 1


def _audit_one(cand, band: str, op_name: str, rf: int) -> None:
    """Assert everything the audit table claims, raising on any failure."""
    assert cand.band == band and cand.op_name == op_name and cand.target_rf == rf
    assert isinstance(cand.norm, nn.Identity)

    cand.train()
    x = make_input()
    y = cand(x)
    assert y.shape == (BATCH, C.PATH_CHANNELS, C.NUM_ELECTRODES, C.NUM_TIMEPOINTS), (
        f"shape {tuple(y.shape)}"
    )
    assert y.shape[3] == C.NUM_TIMEPOINTS, "time changed"
    assert y.shape[2] == C.NUM_ELECTRODES, "electrode axis changed"
    assert bool(torch.isfinite(y).all()), "NaN/Inf in output"

    y.sum().backward()
    for name, p in cand.named_parameters():
        assert p.grad is not None, f"{name} missing gradient"
        assert bool(torch.isfinite(p.grad).all()), f"{name} non-finite gradient"
        assert float(p.grad.abs().sum()) != 0.0, f"{name} zero gradient"

    analytic = analytic_rf(cand.op)
    assert analytic == rf, f"analytic RF {analytic} != {rf}"
    measured = measure_rf(cand, analytic)
    assert measured["span"] == rf, f"measured RF span {measured['span']} != {rf}"
    assert measured["count"] == analytic_taps(cand.op), (
        f"sampled {measured['count']} positions, expected {analytic_taps(cand.op)}"
    )
    assert measured["stride"] == analytic_stride(cand.op), (
        f"stride {measured['stride']} != dilation {analytic_stride(cand.op)}"
    )
    assert cand.num_params > 0


def main() -> int:
    return run_audit(verbose=True)


if __name__ == "__main__":
    raise SystemExit(main())
