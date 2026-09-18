"""Stage-3 isolation tests for first-order DARTS search."""

from __future__ import annotations

import math
import unittest

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from tdarts.architect import SearchArchitect
from tdarts.mixed_op import TemporalDARTSNet
from tdarts.search import run_search_epoch, should_update_alphas
from tdarts.search_data import session0_split_index


class FirstOrderIsolation(unittest.TestCase):
    """Use the real supernet so all 88 BatchNorm layers are exercised."""

    @classmethod
    def setUpClass(cls):
        torch.manual_seed(20250901)
        # Small C/T dimensions keep the test fast; all 84 candidate and four
        # downstream BatchNorm modules remain present and active.
        cls.model = TemporalDARTSNet(n_electrodes=2)
        cls.criterion = nn.NLLLoss()

    def setUp(self):
        torch.manual_seed(17)
        self.model = TemporalDARTSNet(n_electrodes=2)
        self.optimizer = torch.optim.Adam(self.model.network_parameters(), lr=1e-3)
        self.architect = SearchArchitect(self.model, self.optimizer)
        self.inputs = torch.randn(2, 9, 2, 32)
        self.targets = torch.tensor([0, 1])

    def _network_snapshot(self):
        return [p.detach().clone() for p in self.model.network_parameters()]

    def _alpha_snapshot(self):
        return [p.detach().clone() for p in self.model.arch_parameters()]

    def _bn_snapshot(self):
        return {
            name: value.detach().clone()
            for name, value in self.model.named_buffers()
            if "running_" in name or "num_batches_tracked" in name
        }

    def test_weight_step_changes_w_but_not_alpha(self):
        before_w = self._network_snapshot()
        before_alpha = self._alpha_snapshot()
        result = self.architect.weight_step(self.inputs, self.targets, self.criterion)
        after_w = self._network_snapshot()
        after_alpha = self._alpha_snapshot()
        self.assertGreater(result.grad_norm, 0.0)
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(before_w, after_w)))
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(before_alpha, after_alpha)))

    def test_alpha_step_changes_alpha_but_not_w_or_bn_stats(self):
        before_w = self._network_snapshot()
        before_alpha = self._alpha_snapshot()
        before_bn = self._bn_snapshot()
        result = self.architect.alpha_step(self.inputs, self.targets, self.criterion)
        after_w = self._network_snapshot()
        after_alpha = self._alpha_snapshot()
        after_bn = self._bn_snapshot()
        self.assertGreater(result.grad_norm, 0.0)
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(before_alpha, after_alpha)))
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(before_w, after_w)))
        self.assertEqual(len(before_bn), 264)  # mean, variance, and counter for 88 BNs
        self.assertEqual(before_bn.keys(), after_bn.keys())
        self.assertTrue(all(torch.equal(before_bn[name], after_bn[name]) for name in before_bn))

    def test_alpha_step_loader_changes_alpha_but_not_w_or_bn_stats(self):
        # 5 samples at batch size 2 gives a 2/2/1 split, so the short trailing
        # batch exercises the sample-count weighting rather than just running
        # one full batch through.
        loader = DataLoader(
            TensorDataset(torch.randn(5, 9, 2, 32), torch.tensor([0, 1, 2, 3, 0])),
            batch_size=2,
        )
        before_w = self._network_snapshot()
        before_alpha = self._alpha_snapshot()
        before_bn = self._bn_snapshot()
        result = self.architect.alpha_step_loader(
            loader, self.criterion, device=torch.device("cpu")
        )
        after_w = self._network_snapshot()
        after_alpha = self._alpha_snapshot()
        after_bn = self._bn_snapshot()
        self.assertTrue(math.isfinite(result.nll))
        self.assertGreater(result.grad_norm, 0.0)
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(before_alpha, after_alpha)))
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(before_w, after_w)))
        self.assertEqual(len(before_bn), 264)  # mean, variance, and counter for 88 BNs
        self.assertEqual(before_bn.keys(), after_bn.keys())
        self.assertTrue(all(torch.equal(before_bn[name], after_bn[name]) for name in before_bn))

    def test_fullval_mode_uses_one_alpha_step(self):
        loader = DataLoader(
            TensorDataset(torch.randn(6, 9, 2, 32), torch.tensor([0, 1, 2, 3, 0, 1])),
            batch_size=2,
        )
        metrics = run_search_epoch(
            self.architect,
            loader,
            loader,
            self.criterion,
            epoch=21,
            warmup_epochs=20,
            device=torch.device("cpu"),
            alpha_update_mode="fullval",
        )
        self.assertTrue(metrics["alpha_updated"])
        self.assertEqual(metrics["alpha_steps"], 1)
        self.assertEqual(metrics["train_steps"], 3)

    def test_minibatch_remains_the_default_schedule(self):
        # Guards the other half of the contract: an unparameterised call must
        # still make one alpha step per weight step, which is the behaviour the
        # already-published darts_200ep / darts_lr1e3 searches were produced by.
        loader = DataLoader(
            TensorDataset(torch.randn(6, 9, 2, 32), torch.tensor([0, 1, 2, 3, 0, 1])),
            batch_size=2,
        )
        metrics = run_search_epoch(
            self.architect,
            loader,
            loader,
            self.criterion,
            epoch=21,
            warmup_epochs=20,
            device=torch.device("cpu"),
        )
        self.assertTrue(metrics["alpha_updated"])
        self.assertEqual(metrics["alpha_steps"], metrics["train_steps"])
        self.assertEqual(metrics["alpha_steps"], 3)

    def test_unknown_alpha_update_mode_is_rejected(self):
        loader = DataLoader(
            TensorDataset(torch.randn(2, 9, 2, 32), torch.tensor([0, 1])),
            batch_size=2,
        )
        with self.assertRaises(ValueError):
            run_search_epoch(
                self.architect,
                loader,
                loader,
                self.criterion,
                epoch=21,
                warmup_epochs=20,
                device=torch.device("cpu"),
                alpha_update_mode="full-val",
            )

    def test_warmup_epochs_do_not_update_alpha(self):
        before_alpha = self._alpha_snapshot()
        loader = DataLoader(TensorDataset(self.inputs, self.targets), batch_size=2)
        metrics = run_search_epoch(
            self.architect,
            loader,
            loader,
            self.criterion,
            epoch=20,
            warmup_epochs=20,
            device=torch.device("cpu"),
        )
        after_alpha = self._alpha_snapshot()
        self.assertFalse(metrics["alpha_updated"])
        self.assertEqual(metrics["alpha_steps"], 0)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(before_alpha, after_alpha)))
        self.assertFalse(should_update_alphas(20, 20))
        self.assertTrue(should_update_alphas(21, 20))


class SearchSplitProtocol(unittest.TestCase):
    def test_fbnas_80_20_boundary_for_subject003_session0(self):
        self.assertEqual(session0_split_index(288), 231)

    def test_warmup_schedule_rejects_zero_indexed_epoch(self):
        with self.assertRaises(ValueError):
            should_update_alphas(0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main(verbosity=2)
