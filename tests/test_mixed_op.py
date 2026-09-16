"""Stage-2 tests: 14-candidate MixedOp, two-path cell, full TemporalDARTSNet.

Stage 2 builds the architecture only.  There is no architecture optimiser here,
so these tests verify that the *structure* is exactly as specified and that the
mixture is well defined before any search is attempted:

* the mixture ranges over 14 distinct candidates, not 16 aliases;
* there are six independent alpha containers (2 paths x 3 bands);
* the output is ``[B, 36, 22, 1000]`` and enters the unchanged backbone;
* forcing alpha to a one-hot vector makes the mixed output identical to the
  corresponding single operator -- which is what makes a later discretisation
  meaningful.

Run directly for a report::

    python tests/test_mixed_op.py
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tdarts import config as C  # noqa: E402
from tdarts.backbone import TemporalBackbone  # noqa: E402
from tdarts.mixed_op import (  # noqa: E402
    NUM_CANDIDATES_PER_BAND,
    MixedTemporalOp,
    TemporalDARTSNet,
    TwoPathTemporalCell,
    candidate_names,
)
from tdarts.temporal_ops import canonical_candidates  # noqa: E402

BATCH = 2


def band_input(batch: int = BATCH) -> torch.Tensor:
    return torch.randn(batch, C.IN_CHANNELS, C.NUM_ELECTRODES, C.NUM_TIMEPOINTS)


def full_input(batch: int = BATCH) -> torch.Tensor:
    return torch.randn(batch, C.NUM_BANDS, C.NUM_ELECTRODES, C.NUM_TIMEPOINTS)


def one_hot(model: MixedTemporalOp, index: int) -> None:
    """Force candidate ``index`` to probability 1.

    Uses a large finite logit rather than ``inf``: ``softmax`` with an infinite
    logit yields NaN gradients, which would make the equivalence check pass for
    the wrong reason.
    """
    with torch.no_grad():
        model.alpha.fill_(0.0)
        model.alpha[index] = 60.0


class MixedTemporalOpBasics(unittest.TestCase):
    def test_fourteen_distinct_candidates(self):
        for band in ("Low", "Mid", "High"):
            with self.subTest(band=band):
                mix = MixedTemporalOp(band)
                self.assertEqual(mix.num_candidates, NUM_CANDIDATES_PER_BAND)
                self.assertEqual(mix.num_candidates, 14)
                # The canonical set, not the declared 16.
                self.assertEqual(
                    mix.op_names,
                    [(op, rf) for _, op, rf in canonical_candidates(band)],
                )

    def test_candidate_names_are_unique(self):
        names = candidate_names("Low")
        self.assertEqual(len(names), len(set(names)))
        self.assertNotIn("normal_rf15", names)
        self.assertNotIn("lkdw_rf15", names)

    def test_forward_shape(self):
        mix = MixedTemporalOp("Low")
        x = band_input()
        with torch.no_grad():
            y = mix(x)
        self.assertEqual(
            tuple(y.shape), (BATCH, C.PATH_CHANNELS, C.NUM_ELECTRODES, C.NUM_TIMEPOINTS)
        )
        self.assertTrue(torch.isfinite(y).all())

    def test_backward_reaches_every_candidate_and_alpha(self):
        mix = MixedTemporalOp("Low")
        mix.train()
        y = mix(band_input())
        y.sum().backward()

        self.assertIsNotNone(mix.alpha.grad)
        self.assertTrue(torch.isfinite(mix.alpha.grad).all())
        self.assertNotEqual(float(mix.alpha.grad.abs().sum()), 0.0)

        # Every candidate must receive gradient, otherwise it is inert in the
        # mixture and could never be selected.
        for (op_name, rf), op in zip(mix.op_names, mix.ops):
            with self.subTest(candidate=f"{op_name}_rf{rf}"):
                grads = [p.grad for p in op.parameters()]
                self.assertTrue(grads, "candidate has no parameters")
                for g in grads:
                    self.assertIsNotNone(g, "candidate parameter missing gradient")
                    self.assertNotEqual(float(g.abs().sum()), 0.0)

    def test_alpha_is_the_only_architecture_parameter(self):
        mix = MixedTemporalOp("Low")
        arch = list(mix.architecture_parameters())
        self.assertEqual(len(arch), 1)
        self.assertIs(arch[0], mix.alpha)
        self.assertEqual(tuple(arch[0].shape), (14,))

        # Compare by identity: `in` on a list of tensors would try to broadcast
        # and compare elementwise rather than test membership.
        net = list(mix.network_parameters())
        self.assertFalse(
            any(p is mix.alpha for p in net), "alpha leaked into network weights"
        )
        self.assertEqual(
            sum(p.numel() for p in net) + mix.alpha.numel(),
            sum(p.numel() for p in mix.parameters()),
        )

    def test_initial_weights_are_near_uniform(self):
        mix = MixedTemporalOp("Low")
        weights = mix.mixture_weights().detach()
        self.assertAlmostEqual(float(weights.sum()), 1.0, places=6)
        self.assertTrue(
            torch.allclose(weights, torch.full_like(weights, 1.0 / 14), atol=1e-3),
            f"initial mixture is not near-uniform: {weights.tolist()}",
        )
        self.assertTrue((weights > 0).all())

    def test_alpha_init_scale_controls_uniformity(self):
        exact = MixedTemporalOp("Low", alpha_init_scale=0.0)
        self.assertTrue(
            torch.allclose(
                exact.mixture_weights(),
                torch.full((14,), 1.0 / 14),
                atol=1e-6,
            )
        )

    def test_mixture_weights_sum_to_one_after_perturbation(self):
        mix = MixedTemporalOp("Low")
        with torch.no_grad():
            mix.alpha.copy_(torch.randn(14) * 3.0)
        weights = mix.mixture_weights().detach()
        self.assertAlmostEqual(float(weights.sum()), 1.0, places=5)
        self.assertTrue((weights > 0).all())

    def test_rejects_unknown_band(self):
        with self.assertRaises(ValueError):
            MixedTemporalOp("Ultra")


class OneHotEquivalence(unittest.TestCase):
    """Forcing alpha to a one-hot vector must reproduce the single operator.

    This is the property that makes a later discretisation meaningful: the
    supernet contains the discrete model as an exact special case.
    """

    @classmethod
    def setUpClass(cls) -> None:
        torch.manual_seed(0)
        cls.x = band_input()

    def test_every_candidate_is_recoverable(self):
        for band in ("Low", "Mid", "High"):
            for index, (op_name, rf) in enumerate(
                [(op, r) for _, op, r in canonical_candidates(band)]
            ):
                with self.subTest(band=band, candidate=f"{op_name}_rf{rf}"):
                    torch.manual_seed(1)
                    mix = MixedTemporalOp(band)
                    mix.eval()
                    one_hot(mix, index)
                    with torch.no_grad():
                        single = mix.ops[index](self.x)
                        mixed = mix(self.x)
                    self.assertTrue(
                        torch.allclose(mixed, single, atol=1e-6),
                        f"one-hot mixture != single operator "
                        f"(max diff {(mixed - single).abs().max().item():.3e})",
                    )

    def test_one_hot_weight_is_exactly_one(self):
        mix = MixedTemporalOp("Low")
        for index in range(mix.num_candidates):
            with self.subTest(index=index):
                one_hot(mix, index)
                weights = mix.mixture_weights().detach()
                self.assertGreater(float(weights[index]), 1.0 - 1e-9)
                self.assertEqual(int(weights.argmax()), index)

    def test_mixture_equals_weighted_sum_of_candidates(self):
        """forward must be exactly sum_i p_i * O_i(x), not an approximation."""
        torch.manual_seed(2)
        mix = MixedTemporalOp("Low")
        mix.eval()
        with torch.no_grad():
            mix.alpha.copy_(torch.randn(14))
            weights = mix.mixture_weights()
            fused = mix(self.x)
            explicit = sum(
                w * op(self.x) for w, op in zip(weights, mix.ops)
            )
        self.assertTrue(torch.allclose(fused, explicit, atol=1e-6))

    def test_mixture_is_not_a_single_candidate_after_perturbation(self):
        """Guard against a vacuous equivalence test."""
        torch.manual_seed(3)
        mix = MixedTemporalOp("Low")
        mix.eval()
        with torch.no_grad():
            mix.alpha.copy_(torch.randn(14))
            mixed = mix(self.x)
            single = mix.ops[int(mix.mixture_weights().argmax())](self.x)
        self.assertFalse(torch.allclose(mixed, single, atol=1e-6))


class TwoPathTemporalCellTests(unittest.TestCase):
    def test_output_width_and_shape(self):
        cell = TwoPathTemporalCell("Low")
        with torch.no_grad():
            y = cell(band_input())
        self.assertEqual(
            tuple(y.shape), (BATCH, 12, C.NUM_ELECTRODES, C.NUM_TIMEPOINTS)
        )
        self.assertEqual(cell.out_channels, 12)

    def test_paths_have_independent_alpha(self):
        cell = TwoPathTemporalCell("Low")
        a, b = cell.paths[0].alpha, cell.paths[1].alpha
        self.assertIsNot(a, b)
        self.assertFalse(torch.equal(a, b), "paths must not start identical")

    def test_paths_have_independent_weights(self):
        cell = TwoPathTemporalCell("Low")
        pa = list(cell.paths[0].network_parameters())
        pb = list(cell.paths[1].network_parameters())
        self.assertEqual(len(pa), len(pb))
        for x, y in zip(pa, pb):
            self.assertIsNot(x, y)

    def test_perturbing_one_path_changes_its_half_only(self):
        """Path B's parameters must not influence path A's output."""
        torch.manual_seed(4)
        cell = TwoPathTemporalCell("Low")
        cell.eval()
        x = band_input()

        with torch.no_grad():
            a_before = cell.paths[0](x)
            b_before = cell.paths[1](x)
            # Perturb every network weight of path B.
            for p in cell.paths[1].network_parameters():
                p.add_(0.5)
            a_after = cell.paths[0](x)
            b_after = cell.paths[1](x)

        self.assertTrue(torch.equal(a_before, a_after), "path A changed")
        self.assertFalse(torch.equal(b_before, b_after), "path B did not change")

    def test_perturbing_alpha_of_one_path_does_not_move_the_other(self):
        torch.manual_seed(5)
        cell = TwoPathTemporalCell("Low")
        cell.eval()
        x = band_input()
        with torch.no_grad():
            a_before = cell.paths[0](x)
            cell.paths[1].alpha.add_(torch.randn(14) * 3.0)
            a_after = cell.paths[0](x)
        self.assertTrue(torch.equal(a_before, a_after))

    def test_cell_output_is_normalised_concat(self):
        torch.manual_seed(6)
        cell = TwoPathTemporalCell("Low")
        cell.eval()
        x = band_input()
        with torch.no_grad():
            concat = torch.cat([p(x) for p in cell.paths], dim=1)
            out = cell(x)
        self.assertEqual(concat.shape[1], 12)
        self.assertTrue(torch.allclose(out, cell.bn(concat), atol=1e-6))

    def test_arch_parameters_are_two_containers(self):
        cell = TwoPathTemporalCell("Low")
        self.assertEqual(len(cell.architecture_parameters()), 2)
        self.assertEqual(
            sum(p.numel() for p in cell.architecture_parameters()), 28
        )

    def test_num_paths_is_configurable(self):
        cell = TwoPathTemporalCell("Low", num_paths=1)
        self.assertEqual(cell.out_channels, 6)
        with torch.no_grad():
            self.assertEqual(cell(band_input()).shape[1], 6)
        cell3 = TwoPathTemporalCell("Low", num_paths=3)
        self.assertEqual(cell3.out_channels, 18)
        with torch.no_grad():
            self.assertEqual(cell3(band_input()).shape[1], 18)

    def test_rejects_zero_paths(self):
        with self.assertRaises(ValueError):
            TwoPathTemporalCell("Low", num_paths=0)


class TemporalDARTSNetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.manual_seed(7)
        cls.net = TemporalDARTSNet()
        cls.net.eval()

    def test_temporal_stage_output_shape(self):
        bands = self.net.split_bands(full_input())
        self.assertEqual(sorted(bands), ["High", "Low", "Mid"])
        with torch.no_grad():
            temporal = torch.cat(
                [self.net.cells[b](bands[b]) for b in self.net.bands], dim=1
            )
        self.assertEqual(
            tuple(temporal.shape), (BATCH, 36, C.NUM_ELECTRODES, C.NUM_TIMEPOINTS)
        )
        self.assertEqual(self.net.temporal_out_channels, 36)

    def test_full_forward(self):
        with torch.no_grad():
            logits, features = self.net(full_input())
        self.assertEqual(tuple(logits.shape), (BATCH, C.NUM_CLASSES))
        self.assertEqual(tuple(features.shape), (BATCH, 2304))
        self.assertTrue(torch.isfinite(logits).all())
        self.assertTrue(
            torch.allclose(logits.exp().sum(dim=1), torch.ones(BATCH), atol=1e-5)
        )

    def test_backward_through_the_whole_network(self):
        net = TemporalDARTSNet()
        net.train()
        logits, _ = net(full_input())
        nn.NLLLoss()(logits, torch.tensor([0, 1])).backward()

        for name, p in net.named_parameters():
            with self.subTest(param=name):
                self.assertIsNotNone(p.grad, f"{name} has no gradient")
                self.assertTrue(torch.isfinite(p.grad).all())

    def test_enters_the_unchanged_backbone(self):
        """The temporal output must feed TemporalBackbone with no adaptation."""
        net = TemporalDARTSNet(backbone=TemporalBackbone())
        net.eval()
        bands = net.split_bands(full_input())
        with torch.no_grad():
            temporal = torch.cat([net.cells[b](bands[b]) for b in net.bands], dim=1)
            scb = net.backbone.scb(temporal)
        self.assertEqual(tuple(scb.shape), (BATCH, 288, 1, 1000))

    def test_six_independent_alpha_containers(self):
        alphas = self.net.alphas()
        self.assertEqual(len(alphas), 6)
        for band in ("Low", "Mid", "High"):
            for path in (0, 1):
                with self.subTest(band=band, path=path):
                    a = alphas[(band, path)]
                    self.assertEqual(tuple(a.shape), (14,))

        ids = {id(p) for p in self.net.arch_parameters()}
        self.assertEqual(len(ids), 6, "alpha containers must be distinct tensors")
        self.assertEqual(self.net.num_arch_parameters(), 84)

    def test_alpha_containers_are_not_aliases(self):
        alphas = self.net.alphas()
        values = [a.detach().clone() for a in alphas.values()]
        for i in range(len(values)):
            for j in range(i + 1, len(values)):
                with self.subTest(i=i, j=j):
                    self.assertFalse(torch.equal(values[i], values[j]))

    def test_bands_have_independent_weights(self):
        low = list(self.net.cells["Low"].network_parameters())
        mid = list(self.net.cells["Mid"].network_parameters())
        for a, b in zip(low, mid):
            self.assertIsNot(a, b)

    def test_perturbing_one_band_leaves_the_others_alone(self):
        torch.manual_seed(8)
        net = TemporalDARTSNet()
        net.eval()
        bands = net.split_bands(full_input())
        with torch.no_grad():
            before = {b: net.cells[b](bands[b]) for b in net.bands}
            for p in net.cells["Mid"].network_parameters():
                p.add_(0.25)
            after = {b: net.cells[b](bands[b]) for b in net.bands}
        self.assertTrue(torch.equal(before["Low"], after["Low"]))
        self.assertTrue(torch.equal(before["High"], after["High"]))
        self.assertFalse(torch.equal(before["Mid"], after["Mid"]))

    def test_perturbing_one_band_alpha_leaves_the_others_alone(self):
        torch.manual_seed(9)
        net = TemporalDARTSNet()
        net.eval()
        bands = net.split_bands(full_input())
        with torch.no_grad():
            before = {b: net.cells[b](bands[b]) for b in net.bands}
            net.cells["High"].paths[0].alpha.add_(torch.randn(14) * 3)
            after = {b: net.cells[b](bands[b]) for b in net.bands}
        self.assertTrue(torch.equal(before["Low"], after["Low"]))
        self.assertTrue(torch.equal(before["Mid"], after["Mid"]))
        self.assertFalse(torch.equal(before["High"], after["High"]))

    def test_arch_and_network_parameters_partition_the_model(self):
        arch_ids = {id(p) for p in self.net.arch_parameters()}
        net_ids = {id(p) for p in self.net.network_parameters()}
        self.assertEqual(arch_ids & net_ids, set(), "alpha leaked into net weights")
        self.assertEqual(
            len(arch_ids | net_ids),
            len({id(p) for p in self.net.parameters()}),
        )
        self.assertEqual(len(arch_ids), 6)

    def test_split_bands_partitions_the_input(self):
        x = full_input()
        bands = self.net.split_bands(x)
        self.assertEqual(
            torch.cat([bands[b] for b in self.net.bands], dim=1).shape, x.shape
        )
        self.assertTrue(
            torch.equal(
                torch.cat([bands[b] for b in self.net.bands], dim=1), x
            )
        )
        for band, chunk in bands.items():
            with self.subTest(band=band):
                self.assertEqual(chunk.shape[1], 3)

    def test_accepts_official_5d_layout(self):
        """Official FBNAS also feeds [B, 1, C, T, 9]; it must still work."""
        x5 = torch.randn(BATCH, 1, C.NUM_ELECTRODES, C.NUM_TIMEPOINTS, C.NUM_BANDS)
        with torch.no_grad():
            logits, features = self.net(x5)
        self.assertEqual(tuple(logits.shape), (BATCH, C.NUM_CLASSES))
        self.assertEqual(tuple(features.shape), (BATCH, 2304))

    def test_rejects_wrong_channel_count(self):
        with self.assertRaises(ValueError) as ctx:
            self.net(torch.randn(BATCH, 6, C.NUM_ELECTRODES, C.NUM_TIMEPOINTS))
        message = str(ctx.exception)
        self.assertIn("filter-bank channels", message)
        self.assertIn("got 6", message)

    def test_rejects_wrong_rank(self):
        with self.assertRaises(ValueError):
            self.net(torch.randn(BATCH, 9, C.NUM_TIMEPOINTS))

    def test_single_path_configuration(self):
        net = TemporalDARTSNet(num_paths=1)
        self.assertEqual(net.temporal_out_channels, 18)
        net.eval()
        with torch.no_grad():
            logits, features = net(full_input())
        self.assertEqual(tuple(logits.shape), (BATCH, C.NUM_CLASSES))
        self.assertEqual(features.shape[1], 18 * 8 * 8)
        self.assertEqual(len(net.arch_parameters()), 3)

    def test_mixtures_start_near_uniform(self):
        net = TemporalDARTSNet()
        for key, alpha in net.alphas().items():
            with self.subTest(key=key):
                weights = torch.softmax(alpha, dim=0).detach()
                self.assertTrue(
                    torch.allclose(
                        weights, torch.full_like(weights, 1 / 14), atol=1e-3
                    )
                )


class DiscretisationPrecondition(unittest.TestCase):
    """A one-hot supernet must equal the discrete model, per band and path."""

    def test_band_cell_one_hot_matches_single_operators(self):
        torch.manual_seed(10)
        net = TemporalDARTSNet()
        net.eval()
        x = full_input()
        bands = net.split_bands(x)

        cell = net.cells["Low"]
        # Look the candidate up by name rather than trusting a hard-coded
        # index, so this test cannot silently check the wrong operator.
        chosen = cell.paths[0].op_names.index(("dwsep", 57))
        one_hot(cell.paths[0], chosen)
        with torch.no_grad():
            single = cell.paths[0].ops[chosen](bands["Low"])
            mixed = cell.paths[0](bands["Low"])
        self.assertTrue(torch.allclose(mixed, single, atol=1e-6))
        self.assertEqual(cell.paths[0].op_names[chosen], ("dwsep", 57))


def main() -> int:
    print("Stage-2 Architecture Audit")
    print("-" * 78)
    torch.manual_seed(0)
    net = TemporalDARTSNet()
    net.eval()
    x = full_input()
    bands = net.split_bands(x)
    with torch.no_grad():
        lows = [net.cells[b](bands[b]) for b in net.bands]
        temporal = torch.cat(lows, dim=1)
        logits, features = net(x)
    print(f"  input            {tuple(x.shape)}")
    for band, t in zip(net.bands, lows):
        print(f"  {band:<4} cell       {tuple(t.shape)}")
    print(f"  temporal concat  {tuple(temporal.shape)}")
    print(f"  backbone logits  {tuple(logits.shape)}")
    print(f"  features         {tuple(features.shape)}")
    print(f"  candidates/path  {NUM_CANDIDATES_PER_BAND}")
    print(f"  alpha containers {len(net.arch_parameters())} "
          f"({net.num_arch_parameters()} numbers)")
    print(f"  net parameters   {sum(p.numel() for p in net.network_parameters())}")

    import os
    import unittest as _unittest

    suite = _unittest.TestLoader().loadTestsFromModule(sys.modules[__name__])
    runner = _unittest.TextTestRunner(stream=open(os.devnull, "w"), verbosity=0)
    result = runner.run(suite)
    bad = len(result.failures) + len(result.errors)
    print()
    print(f"{result.testsRun - bad} / {result.testsRun} PASS")
    if bad:
        for case, tb in list(result.failures) + list(result.errors):
            print(f"  FAIL {case}")
            print("      " + tb.strip().replace("\n", "\n      "))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
