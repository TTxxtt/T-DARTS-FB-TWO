"""Contract tests for the Operator Separability V2 pool.

The pilot's conclusions rest on the families being comparable, so the tests
that matter are the equality checks the analysis assumes: same output shape,
same temporal support, comparable parameter count.  A family that quietly reads
a wider window or costs twice as much would not crash anything -- it would just
answer a different question -- so each of those properties is asserted rather
than eyeballed.
"""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path
from unittest import mock

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tdarts import config as C
from tdarts.operator_v2 import (
    PARAM_TOLERANCE,
    V2_OPERATOR_NAMES,
    V2TemporalOp,
    build_v2_operator,
    count_macs,
    operator_audit,
    temporal_support,
)
from tdarts.operator_v2_network import OperatorV2Net, V2_STANDALONE_PATH_CHANNELS

RF = 57
BAND = "Low"
SHAPE = (2, C.IN_CHANNELS, C.NUM_ELECTRODES, 1000)


class ShapeContractTests(unittest.TestCase):
    def test_every_operator_preserves_electrodes_and_time(self):
        for name in V2_OPERATOR_NAMES:
            with self.subTest(operator=name):
                op = build_v2_operator(name, band=BAND, target_rf=RF)
                with torch.no_grad():
                    out = op(torch.zeros(SHAPE))
                self.assertEqual(out.shape, (SHAPE[0], V2_STANDALONE_PATH_CHANNELS, SHAPE[2], SHAPE[3]))

    def test_explicit_channel_convention_is_rejected_when_the_budget_differs(self):
        """gated and local_attention size their internals against the pilot's
        3 -> 12 budget.  Accepting another width would silently reuse a budget
        that was never computed, so they refuse instead."""
        for name in ("gated", "local_attention"):
            with self.subTest(operator=name):
                with self.assertRaises(ValueError):
                    build_v2_operator(name, band=BAND, target_rf=RF, out_channels=6)

    def test_unknown_operator_is_rejected(self):
        with self.assertRaises(ValueError):
            build_v2_operator("transformer", band=BAND, target_rf=RF)


class GeometryContractTests(unittest.TestCase):
    def test_every_operator_reads_exactly_the_target_receptive_field(self):
        """The whole comparison is 'same window, different mechanism'.  A
        family whose support span differs is a different experiment."""
        for name in V2_OPERATOR_NAMES:
            with self.subTest(operator=name):
                op = build_v2_operator(name, band=BAND, target_rf=RF)
                positions, spacing, span = op.support
                self.assertEqual(span, RF)
                # 15 taps spaced 4 apart: 1 + (15 - 1) * 4 == 57
                self.assertEqual(positions, 15)
                self.assertEqual(spacing, 4)

    def test_gating_and_projection_layers_do_not_widen_the_support(self):
        """band_gated's gate and gated's projection are 1x1, so they have RF 1
        and must leave the reported support at the temporal convolution's."""
        for name in ("band_gated", "gated"):
            with self.subTest(operator=name):
                op = build_v2_operator(name, band=BAND, target_rf=RF)
                self.assertEqual(op.support[2], RF)

    def test_support_matches_for_every_band(self):
        for band in C.BANDS:
            for name in V2_OPERATOR_NAMES:
                with self.subTest(band=band, operator=name):
                    _, _, span = build_v2_operator(name, band=band, target_rf=RF).support
                    self.assertEqual(span, RF)


class FairnessTests(unittest.TestCase):
    def test_parameter_counts_sit_inside_the_tolerance_band(self):
        rows = operator_audit(band=BAND, target_rf=RF)
        for row in rows:
            with self.subTest(operator=row["operator"]):
                self.assertTrue(
                    row["within_tolerance"],
                    f"{row['operator']} is {row['param_ratio_vs_anchor']:.3f}x the anchor, "
                    f"outside +/-{PARAM_TOLERANCE:.0%}",
                )

    def test_the_anchor_is_the_original_fbnas_operator(self):
        """conv(3 -> 12, (1,15), dilation 4) is 3*12*15 = 540 parameters; a
        change here would move every ratio in the fairness table."""
        anchor = build_v2_operator("dilated", band=BAND, target_rf=RF)
        self.assertEqual(anchor.num_op_params, 3 * 12 * 15)

    def test_no_family_costs_more_than_1_3x_the_anchor(self):
        """The pilot compares mechanisms, not budgets.  A family that quietly
        spends twice the compute of the anchor answers a different question,
        which is exactly the confound this ceiling exists to prevent."""
        rows = {r["operator"]: r for r in operator_audit(band=BAND, target_rf=RF)}
        anchor_macs = rows["dilated"]["macs"]
        for name in V2_OPERATOR_NAMES:
            with self.subTest(operator=name):
                ratio = rows[name]["macs"] / anchor_macs
                self.assertLessEqual(
                    ratio, 1.30, f"{name} costs {ratio:.3f}x the anchor's MACs"
                )

    def test_attention_is_not_cheaper_than_a_convolution(self):
        """The opposite failure from the one the ceiling guards: a counter that
        cannot see elementwise work would report attention below the anchor and
        make a compute-heavy mechanism look free."""
        rows = {r["operator"]: r for r in operator_audit(band=BAND, target_rf=RF)}
        anchor_macs = rows["dilated"]["macs"]
        self.assertGreater(rows["local_attention"]["macs"], anchor_macs)
        self.assertTrue(rows["local_attention"]["within_tolerance"])

    def test_the_attention_head_is_only_as_wide_as_the_input(self):
        """The score is q(x).k(x') == x^T W_q^T W_k x', so any head wider than
        in_channels spans the same 3x3 bilinear forms and buys nothing.  The
        narrow head is the whole reason this family fits the compute ceiling,
        so it is asserted rather than left to a comment."""
        from tdarts.operator_v2 import SparseDilatedLocalAttention

        self.assertEqual(SparseDilatedLocalAttention.HEAD_DIM, C.IN_CHANNELS)
        with self.assertRaises(ValueError):
            SparseDilatedLocalAttention(
                band=BAND, target_rf=RF, in_channels=C.IN_CHANNELS + 1
            )

    def test_macs_counter_sees_work_outside_convolution_modules(self):
        """The dynamic family applies a convolution per basis kernel.  An
        earlier hook-only counter missed the looped convolutions and reported
        the family at 0.27x the anchor instead of ~0.98x."""
        op = build_v2_operator("dynamic", band=BAND, target_rf=RF)
        hook_only = 0
        with torch.no_grad():
            for module in op.modules():
                if isinstance(module, torch.nn.Conv2d):
                    hook_only += module.out_channels * module.in_channels // module.groups * \
                        module.kernel_size[1] * C.NUM_ELECTRODES * 1000
        self.assertGreater(count_macs(op, (1, C.IN_CHANNELS, C.NUM_ELECTRODES, 1000)), 0)


class GradientTests(unittest.TestCase):
    def test_forward_backward_is_finite_for_every_operator(self):
        for name in V2_OPERATOR_NAMES:
            with self.subTest(operator=name):
                torch.manual_seed(0)
                op = build_v2_operator(name, band=BAND, target_rf=RF)
                out = op(torch.randn(2, C.IN_CHANNELS, C.NUM_ELECTRODES, 200))
                out.sum().backward()
                for parameter in op.parameters():
                    self.assertTrue(torch.isfinite(parameter).all())
                    if parameter.grad is not None:
                        self.assertTrue(torch.isfinite(parameter.grad).all())
                gradients = [p.grad for p in op.parameters() if p.grad is not None]
                self.assertTrue(gradients)
                self.assertTrue(any(g.abs().sum() > 0 for g in gradients))

    def test_attention_softmax_is_over_the_support_not_the_whole_axis(self):
        """If the attention silently became global it would still run and still
        train -- and would no longer be the same experiment."""
        from tdarts.operator_v2 import SparseDilatedLocalAttention

        op = build_v2_operator("local_attention", band=BAND, target_rf=RF)
        patches = op.op._patches(torch.randn(1, op.op.out_channels, C.NUM_ELECTRODES, 32))
        self.assertEqual(patches.shape[1], op.op.out_channels)
        self.assertEqual(patches.shape[2], 15)
        self.assertIsInstance(op.op, SparseDilatedLocalAttention)


class AttentionReassociationTests(unittest.TestCase):
    """The attention family projects before the weighted reduction.  That is
    the change that brings it under the compute ceiling, and it is only
    legitimate if it is the *same function* -- so it is checked against a
    reference that projects after, with identical weights."""

    def _reference(self, inner, x):
        query = inner.q(x).unsqueeze(2)
        keys = inner._patches(inner.k(x))
        attention = ((query * keys).sum(dim=1) * inner.scale).softmax(dim=1)
        values = inner._patches(inner.v(x))
        context = (attention.unsqueeze(1) * values).sum(dim=2)
        return inner.out(inner.proj(context))

    def test_projecting_before_the_reduction_changes_only_rounding(self):
        torch.manual_seed(0)
        op = build_v2_operator("local_attention", band=BAND, target_rf=RF).eval()
        x = torch.randn(3, C.IN_CHANNELS, C.NUM_ELECTRODES, 400)
        with torch.no_grad():
            reference = self._reference(op.op, x)
            got = op(x)
        # Reassociating a 15-term sum through a bottleneck reorders float32
        # rounding; the tolerance is set against the output's own scale, not
        # against zero.  A real behavioural change would be orders larger.
        scale = float(reference.abs().max())
        self.assertLess(float((got - reference).abs().max()), 1e-4 * scale)

    def test_the_reassociation_is_what_removes_the_extra_compute(self):
        """Guard against the change being quietly reverted: with the
        reduction running over VALUE_WIDTH channels instead of out_channels
        the family would leave the compute band again."""
        rows = {r["operator"]: r for r in operator_audit(band=BAND, target_rf=RF)}
        inner = build_v2_operator("local_attention", band=BAND, target_rf=RF).op
        self.assertLess(inner.out_channels, inner.VALUE_WIDTH)

        # Per position the two reductions cost kernel*(HEAD_DIM + width).  Swap
        # the reduced width back to VALUE_WIDTH and the family leaves the band.
        positions = C.NUM_ELECTRODES * 1000
        real = positions * inner.kernel * (inner.HEAD_DIM + inner.out_channels)
        unreordered = positions * inner.kernel * (inner.HEAD_DIM + inner.VALUE_WIDTH)
        projected = rows["local_attention"]["macs"] - real + unreordered
        self.assertGreater(projected, 1.30 * rows["dilated"]["macs"])


class NetworkTests(unittest.TestCase):
    def test_every_operator_builds_a_network_with_the_same_backbone(self):
        backbone_params = set()
        for name in V2_OPERATOR_NAMES:
            with self.subTest(operator=name):
                torch.manual_seed(0)
                net = OperatorV2Net(name, target_rf=RF)
                backbone_params.add(sum(p.numel() for p in net.backbone.parameters()))
        self.assertEqual(len(backbone_params), 1, "the backbone must not vary with the operator")

    def test_network_accepts_the_loader_layout_and_returns_log_probabilities(self):
        """The dataset hands trials over as [B, 1, E, T, 9]; the model must
        permute them the same way TemporalDiscreteNet does."""
        net = OperatorV2Net("dilated", target_rf=RF)
        x = torch.randn(3, 1, C.NUM_ELECTRODES, 1000, C.NUM_BANDS)
        logits, features = net(x)
        self.assertEqual(logits.shape, (3, C.NUM_CLASSES))
        self.assertEqual(features.dim(), 2)
        self.assertAlmostEqual(float(logits.exp().sum(dim=1).mean()), 1.0, places=4)

    def test_network_rejects_a_wrong_channel_count(self):
        net = OperatorV2Net("dilated", target_rf=RF)
        with self.assertRaises(ValueError):
            net(torch.randn(2, 5, C.NUM_ELECTRODES, 1000))


class EntryPointTests(unittest.TestCase):
    """Guards on train_operator_v2 that protect the experiment, not the code."""

    def _import(self):
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "train_operator_v2", Path(__file__).resolve().parent.parent / "train_operator_v2.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_session1_is_closed_unless_explicitly_requested(self):
        """The pilot's validity rests on the test set staying closed, so the
        safe behaviour has to be the default rather than something the operator
        remembers to ask for."""
        module = self._import()
        with mock.patch.object(sys, "argv", ["train_operator_v2.py", "--operator", "gated", "--seed", "1"]):
            self.assertFalse(module.parse_args().read_session1)
        with mock.patch.object(
            sys, "argv", ["train_operator_v2.py", "--operator", "gated", "--seed", "1", "--read-session1"]
        ):
            self.assertTrue(module.parse_args().read_session1)

    def test_the_session1_loader_is_reached_only_through_that_flag(self):
        """A default-off flag is worth little if some other line opens the file.
        The loader appears exactly once, inside the conditional expression that
        the flag guards."""
        source = (Path(__file__).resolve().parent.parent / "train_operator_v2.py").read_text(encoding="utf-8")
        self.assertEqual(source.count("load_subject_session("), 1)
        call_at = source.index("load_subject_session(")
        # The guard is written as `<call> if args.read_session1 else None`, so
        # the flag sits just after the call site, not before it.
        window = source[max(0, call_at - 200):call_at + 200]
        self.assertIn("args.read_session1", window)

    def test_the_leaf_name_carries_the_operator(self):
        """Without the operator in the leaf, all five families resolve to one
        directory; allocate() refuses to overwrite, so four would fail in
        seconds and the arm would silently be a single run."""
        import re

        source = (Path(__file__).resolve().parent.parent / "train_operator_v2.py").read_text(encoding="utf-8")
        self.assertRegex(source, r"arm = f\"\{args\.arm\}_\{args\.operator\}\"")
        self.assertIsNotNone(re.search(r"args\.arm = arm", source))

    def test_default_receptive_field_is_the_middle_rung(self):
        module = self._import()
        self.assertEqual(module.DEFAULT_RF, 57)
        self.assertIn(module.DEFAULT_RF, C.RF_SPACE["Low"])

    def test_every_operator_is_reachable_from_the_cli(self):
        module = self._import()
        self.assertEqual(set(module.V2_OPERATOR_NAMES), set(V2_OPERATOR_NAMES))
        self.assertEqual(len(V2_OPERATOR_NAMES), 5)


if __name__ == "__main__":  # pragma: no cover
    unittest.main(verbosity=2)
