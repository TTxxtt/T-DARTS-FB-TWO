"""Phase A hard-operator checks: fixed RF, Gumbel one-hot, beta isolation."""

from __future__ import annotations

import unittest

import torch

from tdarts import config as C
from tdarts.anchored import AnchoredRFNet, build_anchored_genotype
from tdarts.genotype import Genotype, duplicate_structure_bands, load_genotype, save_genotype
from tdarts.hierarchical import PHASE_A_RF, HardOperatorNet, gumbel_one_hot
from tdarts.architect import SearchArchitect

ELECTRODES = 2
TIMEPOINTS = 32


class GumbelEstimatorTests(unittest.TestCase):
    def test_forward_is_one_hot_and_backward_reaches_logits(self):
        torch.manual_seed(0)
        logits = torch.zeros(4, requires_grad=True)
        weights = gumbel_one_hot(logits, tau=1.0)
        self.assertAlmostEqual(float(weights.sum()), 1.0, places=6)
        self.assertEqual(int((weights > 0.5).sum()), 1)
        weights.sum().backward()
        self.assertIsNotNone(logits.grad)
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertGreater(float(logits.grad.abs().sum()), 0.0)

    def test_rejects_non_positive_temperature(self):
        with self.assertRaises(ValueError):
            gumbel_one_hot(torch.zeros(4), tau=0.0)


class HardOperatorNetTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1)
        self.model = HardOperatorNet(n_electrodes=ELECTRODES, sampling=True)

    def test_architecture_parameters_are_twelve_betas(self):
        self.assertEqual(self.model.num_arch_parameters(), 12)
        self.assertEqual(len(self.model.arch_parameters()), 3)
        for cell in self.model.cells.values():
            self.assertEqual(cell.rf, PHASE_A_RF)

    def test_arch_and_network_parameters_partition_the_model(self):
        network_ids = {id(p) for p in self.model.network_parameters()}
        arch_ids = {id(p) for p in self.model.arch_parameters()}
        self.assertFalse(network_ids & arch_ids)
        self.assertEqual(len(network_ids | arch_ids), len(list(self.model.parameters())))

    def test_training_forward_samples_exactly_one_operator_per_band(self):
        self.model.train()
        x = torch.randn(2, 9, ELECTRODES, TIMEPOINTS)
        self.model(x)
        self.model(x)
        for counts in self.model.selection_counts().values():
            # One sampled operator per forward: exactly two increments overall,
            # spread over at most two candidates (draws may differ).
            self.assertEqual(sum(counts), 2)
            self.assertLessEqual(max(counts), 2)
            self.assertLessEqual(len([c for c in counts if c > 0]), 2)

    def test_beta_receives_gradient_through_the_sample(self):
        self.model.train()
        logits, _ = self.model(torch.randn(2, 9, ELECTRODES, TIMEPOINTS))
        logits.sum().backward()
        for cell in self.model.cells.values():
            self.assertIsNotNone(cell.beta.grad)
            self.assertTrue(torch.isfinite(cell.beta.grad).all())

    def test_hard_eval_uses_the_argmax_operator(self):
        with torch.no_grad():
            for cell in self.model.cells.values():
                cell.beta.fill_(0.0)
                cell.beta[1] = 5.0
        self.model.eval()
        self.model.set_hard_eval(True)
        self.assertEqual(self.model.selected_operators(), {band: C.OPERATORS[1] for band in C.BANDS})
        logits, _ = self.model(torch.randn(2, 9, ELECTRODES, TIMEPOINTS))
        self.assertEqual(tuple(logits.shape), (2, 4))

    def test_set_tau_propagates(self):
        self.model.set_tau(0.25)
        for cell in self.model.cells.values():
            self.assertEqual(cell.tau, 0.25)


class HierarchicalIntegrationTests(unittest.TestCase):
    def test_phase_b_accepts_decoded_operators_and_exports_a_loadable_genotype(self):
        torch.manual_seed(2)
        phase_a = HardOperatorNet(n_electrodes=ELECTRODES)
        with torch.no_grad():
            phase_a.cells["Low"].beta[0] = 5.0  # dilated: same structure as the anchor
            phase_a.cells["Mid"].beta[2] = 5.0  # dwsep
            phase_a.cells["High"].beta[1] = 5.0  # normal
        operators = phase_a.selected_operators()
        self.assertEqual(operators, {"Low": "dilated", "Mid": "dwsep", "High": "normal"})

        model = AnchoredRFNet(operators, n_electrodes=ELECTRODES, no_duplicate_paths=True)
        genotype = build_anchored_genotype(model, seed=3, epoch=1)
        self.assertIsInstance(genotype, Genotype)
        self.assertEqual(set(duplicate_structure_bands(genotype)), set())

        from pathlib import Path
        import tempfile

        with tempfile.TemporaryDirectory() as temporary:
            path = save_genotype(genotype, Path(temporary) / "genotype.json")
            loaded = load_genotype(path)
            self.assertEqual(loaded, genotype)

    def test_phase_b_model_runs_with_search_architect(self):
        torch.manual_seed(3)
        model = AnchoredRFNet(
            {"Low": "dilated", "Mid": "dwsep", "High": "normal"},
            n_electrodes=ELECTRODES,
            no_duplicate_paths=True,
        )
        optimizer = torch.optim.Adam(model.network_parameters(), lr=1e-3)
        architect = SearchArchitect(model, optimizer)
        x = torch.randn(2, 9, ELECTRODES, TIMEPOINTS)
        targets = torch.tensor([0, 1])
        result = architect.weight_step(x, targets, torch.nn.NLLLoss())
        self.assertGreaterEqual(result.accuracy, 0.0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main(verbosity=2)
