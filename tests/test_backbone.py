"""Shape audit for the fixed backbone (SCB -> LogVar -> classifier).

The backbone is the part of FBNAS stage 1 must *not* change, so these tests
pin its documented data flow::

    [B, 36, 22, 1000]
      -> SCB        [B, 288, 1, 1000]
      -> reshape    [B, 288, 8, 125]
      -> LogVar     [B, 288, 8, 1]
      -> flatten    [B, 2304]
      -> classifier [B, 4]

Run directly for a printed report::

    python tests/test_backbone.py
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tdarts import config as C  # noqa: E402
from tdarts.backbone import (  # noqa: E402
    LogVarLayer,
    SpatialConvBlock,
    TemporalBackbone,
    TemporalClassifier,
    backbone_feature_dim,
)

BATCH = 2


class ScbShapes(unittest.TestCase):
    def test_scb_collapses_electrode_axis_and_preserves_time(self):
        scb = SpatialConvBlock(C.NUM_FEAT, C.NUM_FEAT * C.SCB_DILATABILITY, C.NUM_ELECTRODES)
        x = torch.randn(BATCH, C.NUM_FEAT, C.NUM_ELECTRODES, C.NUM_TIMEPOINTS)
        with torch.no_grad():
            y = scb(x)
        self.assertEqual(tuple(y.shape), (BATCH, 288, 1, 1000))

    def test_scb_is_grouped_over_bands(self):
        scb = SpatialConvBlock(C.NUM_FEAT, 288, C.NUM_ELECTRODES)
        conv = scb.block[0]
        self.assertEqual(conv.groups, C.NUM_FEAT)
        self.assertEqual(conv.kernel_size, (C.NUM_ELECTRODES, 1))
        self.assertEqual(conv.max_norm, 2)
        # 8 output channels per input group: 36 groups x 8 = 288
        self.assertEqual(conv.weight.shape[0], 288)
        self.assertEqual(conv.weight.shape[1], 1)

    def test_scb_rejects_indivisible_channel_counts(self):
        with self.assertRaises(ValueError) as ctx:
            SpatialConvBlock(36, 287, C.NUM_ELECTRODES)
        self.assertIn("divisible", str(ctx.exception))


class LogVarShapes(unittest.TestCase):
    def test_logvar_reduces_only_the_requested_axis(self):
        layer = LogVarLayer(dim=3)
        x = torch.randn(BATCH, 288, 8, 125) * 3 + 5
        with torch.no_grad():
            y = layer(x)
        self.assertEqual(tuple(y.shape), (BATCH, 288, 8, 1))

    def test_logvar_is_variance_not_std(self):
        layer = LogVarLayer(dim=3)
        x = torch.rand(BATCH, 4, 2, 64) * 2.0
        with torch.no_grad():
            y = layer(x)
        expected = torch.log(torch.clamp(x.var(dim=3, keepdim=True), 1e-6, 1e6))
        self.assertTrue(torch.allclose(y, expected))

    def test_logvar_clamps_lower_bound(self):
        layer = LogVarLayer(dim=3)
        x = torch.ones(1, 1, 1, 16)  # zero variance
        with torch.no_grad():
            y = layer(x)
        self.assertTrue(torch.isfinite(y).all())
        self.assertAlmostEqual(float(y.item()), float(torch.log(torch.tensor(1e-6))), places=5)

    def test_logvar_clamps_upper_bound(self):
        layer = LogVarLayer(dim=3)
        x = torch.zeros(1, 1, 1, 16)
        x[0, 0, 0, 0] = 1e6
        with torch.no_grad():
            y = layer(x)
        self.assertTrue(torch.isfinite(y).all())


class ClassifierShapes(unittest.TestCase):
    def test_classifier_maps_features_to_log_probabilities(self):
        clf = TemporalClassifier(2304, C.NUM_CLASSES)
        x = torch.randn(BATCH, 2304)
        with torch.no_grad():
            y = clf(x)
        self.assertEqual(tuple(y.shape), (BATCH, C.NUM_CLASSES))
        self.assertTrue(torch.allclose(y.exp().sum(dim=1), torch.ones(BATCH), atol=1e-5))
        self.assertEqual(clf.block[0].max_norm, 0.5)


class BackboneChain(unittest.TestCase):
    def test_documented_shape_chain(self):
        backbone = TemporalBackbone()
        backbone.eval()
        x = torch.randn(BATCH, C.NUM_FEAT, C.NUM_ELECTRODES, C.NUM_TIMEPOINTS)

        with torch.no_grad():
            scb_out = backbone.scb(x)
        self.assertEqual(tuple(scb_out.shape), (BATCH, 288, 1, 1000))

        reshaped = scb_out.reshape(
            BATCH, 288, C.STRIDEFACTOR, C.NUM_TIMEPOINTS // C.STRIDEFACTOR
        )
        self.assertEqual(tuple(reshaped.shape), (BATCH, 288, 8, 125))

        with torch.no_grad():
            agg = backbone.temporal_layer(reshaped)
        self.assertEqual(tuple(agg.shape), (BATCH, 288, 8, 1))

        features = torch.flatten(agg, start_dim=1)
        self.assertEqual(tuple(features.shape), (BATCH, 2304))

        with torch.no_grad():
            logits = backbone.classifier(features)
        self.assertEqual(tuple(logits.shape), (BATCH, C.NUM_CLASSES))

    def test_forward_matches_the_manual_chain(self):
        backbone = TemporalBackbone()
        backbone.eval()
        x = torch.randn(BATCH, C.NUM_FEAT, C.NUM_ELECTRODES, C.NUM_TIMEPOINTS)
        with torch.no_grad():
            logits, features = backbone(x)
            manual = backbone.scb(x)
            manual = manual.reshape(BATCH, 288, 8, 125)
            manual = backbone.temporal_layer(manual)
            manual = torch.flatten(manual, start_dim=1)
        self.assertTrue(torch.equal(features, manual))
        self.assertEqual(tuple(logits.shape), (BATCH, C.NUM_CLASSES))

    def test_feature_dim_derivation(self):
        self.assertEqual(backbone_feature_dim(), 2304)
        self.assertEqual(backbone_feature_dim(36, 8, 8), 2304)
        backbone = TemporalBackbone()
        self.assertEqual(backbone.feature_dim, 2304)
        self.assertEqual(backbone.classifier.block[0].in_features, 2304)

    def test_backward_through_full_backbone(self):
        backbone = TemporalBackbone()
        backbone.train()
        x = torch.randn(BATCH, C.NUM_FEAT, C.NUM_ELECTRODES, C.NUM_TIMEPOINTS)
        logits, _ = backbone(x)
        loss = nn.NLLLoss()(logits, torch.tensor([0, 1]))
        loss.backward()
        for name, p in backbone.named_parameters():
            with self.subTest(param=name):
                self.assertIsNotNone(p.grad, f"{name} has no gradient")
                self.assertTrue(torch.isfinite(p.grad).all())

    def test_rejects_odd_time_length(self):
        backbone = TemporalBackbone()
        x = torch.randn(BATCH, C.NUM_FEAT, C.NUM_ELECTRODES, 999)
        with self.assertRaises(ValueError) as ctx:
            backbone(x)
        self.assertIn("divisible", str(ctx.exception))

    def test_rejects_wrong_rank(self):
        backbone = TemporalBackbone()
        x = torch.randn(BATCH, C.NUM_FEAT, C.NUM_TIMEPOINTS)
        with self.assertRaises(ValueError) as ctx:
            backbone(x)
        self.assertIn("expects", str(ctx.exception))

    def test_rejects_unknown_temporal_layer(self):
        with self.assertRaises(ValueError) as ctx:
            TemporalBackbone(temporal_layer="mean")
        self.assertIn("logvar", str(ctx.exception))

    def test_single_path_configuration_is_accepted(self):
        """1 path per band -> 3 * 1 * 6 = 18 channels; SCB still works."""
        backbone = TemporalBackbone(in_channels=18, dilatability=8)
        x = torch.randn(BATCH, 18, C.NUM_ELECTRODES, C.NUM_TIMEPOINTS)
        with torch.no_grad():
            logits, features = backbone(x)
        self.assertEqual(tuple(logits.shape), (BATCH, C.NUM_CLASSES))
        self.assertEqual(features.shape[1], 18 * 8 * 8)


def main() -> int:
    print("Backbone Shape Audit")
    print("-" * 78)
    backbone = TemporalBackbone()
    backbone.eval()
    x = torch.randn(BATCH, C.NUM_FEAT, C.NUM_ELECTRODES, C.NUM_TIMEPOINTS)
    with torch.no_grad():
        scb = backbone.scb(x)
        resh = scb.reshape(BATCH, 288, C.STRIDEFACTOR, C.NUM_TIMEPOINTS // C.STRIDEFACTOR)
        agg = backbone.temporal_layer(resh)
        logits, feats = backbone(x)
    rows = [
        ("input        ", tuple(x.shape), "3 bands x 2 paths x 6 ch"),
        ("SCB          ", tuple(scb.shape), "electrode axis -> 1"),
        ("reshape      ", tuple(resh.shape), f"strideFactor={C.STRIDEFACTOR}"),
        ("LogVar(dim=3)", tuple(agg.shape), "variance over the 125 axis"),
        ("flatten      ", tuple(feats.shape), "feature vector"),
        ("classifier   ", tuple(logits.shape), "LogSoftmax"),
    ]
    for name, shape, note in rows:
        print(f"  {name} {str(shape):<22} {note}")

    import unittest as _unittest

    loader = _unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])
    import os

    runner = _unittest.TextTestRunner(stream=open(os.devnull, "w"), verbosity=0)
    result = runner.run(suite)
    total = result.testsRun
    bad = len(result.failures) + len(result.errors)
    print()
    print(f"{total - bad} / {total} PASS")
    if bad:
        for case, tb in list(result.failures) + list(result.errors):
            print(f"  FAIL {case}")
            print("      " + tb.strip().replace("\n", "\n      "))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
