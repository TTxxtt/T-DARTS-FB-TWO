"""Anchored search tests: schedule, phase parameter counts, isolation, export.

The anchored arm fixes path 0 to ``dilated`` and searches differently per phase,
so these tests pin the properties the design rests on:

* phase A never sees RF 15, and all four operators in a step share one RF;
* the RF schedule is deterministic and balanced;
* phase A exposes exactly three 4-way ``beta`` containers, phase B exactly six
  4-way ``gamma`` containers, and neither leaks into network weights;
* the existing :class:`SearchArchitect` drives both phases with full isolation
  (no network weight and no BatchNorm buffer moves during an alpha step);
* the exported genotype builds the unchanged discrete network, with RF-15
  aliases normalised to their canonical candidate names.

Run directly for a report::

    python tests/test_anchored.py
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tdarts import config as C  # noqa: E402
from tdarts.anchored import (  # noqa: E402
    ANCHORED_SCHEME,
    OP_SEARCH_RFS,
    RF_SEARCH_SPACE,
    AnchoredOperatorNet,
    AnchoredRFNet,
    build_anchored_genotype,
    candidate_structure_key,
    canonical_candidate_name,
    inherit_operator_weights,
    load_anchored_genotype,
    rf_schedule,
    run_anchored_epoch,
)
from tdarts.architect import SearchArchitect  # noqa: E402
from tdarts.discrete_network import TemporalDiscreteNet  # noqa: E402
from tdarts.mixed_op import candidate_names  # noqa: E402
from tdarts.temporal_ops import TemporalOp  # noqa: E402

BATCH = 2
ELECTRODES = 2
TIMEPOINTS = 32


def full_input(batch: int = BATCH) -> torch.Tensor:
    return torch.randn(batch, C.NUM_BANDS, ELECTRODES, TIMEPOINTS)


def band_input(batch: int = BATCH) -> torch.Tensor:
    return torch.randn(batch, C.IN_CHANNELS, ELECTRODES, TIMEPOINTS)


def states_equal(left: nn.Module, right: nn.Module) -> bool:
    a, b = left.state_dict(), right.state_dict()
    return a.keys() == b.keys() and all(torch.equal(a[key], b[key]) for key in a)


def force_peak(logits: torch.Tensor, index: int, peak: float = 60.0) -> None:
    """Make ``index`` the certain argmax without using infinities."""

    with torch.no_grad():
        logits.zero_()
        logits[index] = peak


class PhaseASchedule(unittest.TestCase):
    def test_rf15_is_excluded_everywhere(self):
        self.assertNotIn(15, OP_SEARCH_RFS)
        net = AnchoredOperatorNet(n_electrodes=ELECTRODES)
        for band, cell in net.cells.items():
            with self.subTest(band=band):
                self.assertNotIn(15, cell.rfs)
                for op in cell.anchor_pools:
                    self.assertNotEqual(op.target_rf, 15)
                for pool in cell.searched_pools:
                    self.assertNotEqual(pool.rf, 15)
                    for op in pool.ops:
                        self.assertNotEqual(op.target_rf, 15)
        with self.assertRaises(ValueError):
            net.set_rf(15)

    def test_schedule_is_balanced_and_deterministic(self):
        counts = Counter(rf_schedule(step) for step in range(300))
        self.assertEqual(counts, Counter({29: 100, 57: 100, 113: 100}))
        for step in range(6):
            self.assertEqual(rf_schedule(step), OP_SEARCH_RFS[step % 3])
        with self.assertRaises(ValueError):
            rf_schedule(-1)

    def test_all_four_operators_share_the_step_rf(self):
        net = AnchoredOperatorNet(n_electrodes=ELECTRODES)
        net.eval()
        recorded: list[int] = []
        handles = [
            module.register_forward_hook(
                lambda mod, inputs, output: recorded.append(mod.target_rf)
            )
            for module in net.modules()
            if isinstance(module, TemporalOp)
        ]
        try:
            with torch.no_grad():
                net.set_rf(57)
                net(full_input())
        finally:
            for handle in handles:
                handle.remove()

        # One anchor plus the four operators, per band.
        self.assertEqual(len(recorded), len(C.BANDS) * (1 + len(C.OPERATORS)))
        self.assertEqual(set(recorded), {57})
        for band, cell in net.cells.items():
            with self.subTest(band=band):
                index = cell.rfs.index(57)
                for op in cell.searched_pools[index].ops:
                    self.assertEqual(op.target_rf, 57)

    def test_set_rf_is_required_before_forward(self):
        cell = AnchoredOperatorNet(n_electrodes=ELECTRODES).cells["Low"]
        with self.assertRaises(RuntimeError):
            cell(band_input())


class PhaseAParameters(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.net = AnchoredOperatorNet(n_electrodes=ELECTRODES)

    def assert_partition(self, model):
        arch = {id(p) for p in model.arch_parameters()}
        net = {id(p) for p in model.network_parameters()}
        self.assertEqual(arch & net, set(), "architecture params leaked into weights")
        self.assertEqual(
            len(arch | net), len({id(p) for p in model.parameters()})
        )

    def test_three_betas_of_four(self):
        arch = list(self.net.arch_parameters())
        self.assertEqual(len(arch), len(C.BANDS))
        for beta in arch:
            self.assertEqual(tuple(beta.shape), (len(C.OPERATORS),))
        self.assertEqual(self.net.num_arch_parameters(), 12)
        self.assert_partition(self.net)

    def test_forward_is_the_weighted_mixture(self):
        self.net.eval()
        self.net.set_rf(29)
        cell = self.net.cells["Low"]
        index = cell.rfs.index(29)
        x = band_input()
        with torch.no_grad():
            anchor = cell.anchor_pools[index](x)
            weights = cell.operator_weights()
            searched = sum(
                weight * op(x)
                for weight, op in zip(weights, cell.searched_pools[index].ops)
            )
            expected = cell.bn(torch.cat([anchor, searched], dim=1))
            actual = cell(x)
        self.assertTrue(torch.allclose(actual, expected, atol=1e-6))

    def test_only_the_active_pool_receives_gradient(self):
        self.net.train()
        self.net.set_rf(57)
        cell = self.net.cells["Low"]
        cell(band_input()).sum().backward()
        active = cell.rfs.index(57)

        self.assertIsNotNone(cell.beta.grad)
        self.assertNotEqual(float(cell.beta.grad.abs().sum()), 0.0)
        for op in cell.searched_pools[active].ops:
            grads = [p.grad for p in op.parameters()]
            self.assertTrue(grads)
            for grad in grads:
                self.assertIsNotNone(grad)
                self.assertNotEqual(float(grad.abs().sum()), 0.0)
        for index, pool in enumerate(cell.searched_pools):
            if index == active:
                continue
            for op in pool.ops:
                for grad in [p.grad for p in op.parameters()]:
                    self.assertIsNone(grad, "inactive RF pool received gradient")

    def test_selected_operator_is_argmax(self):
        force_peak(self.net.cells["High"].beta, 2)
        self.assertEqual(self.net.selected_operators()["High"], C.OPERATORS[2])


class PhaseBParameters(unittest.TestCase):
    OPERATORS = {"Low": "normal", "Mid": "dilated", "High": "lkdw"}

    def setUp(self):
        torch.manual_seed(1)
        self.net = AnchoredRFNet(self.OPERATORS, n_electrodes=ELECTRODES)

    def test_six_gammas_of_four(self):
        arch = list(self.net.arch_parameters())
        self.assertEqual(len(arch), 2 * len(C.BANDS))
        for gamma in arch:
            self.assertEqual(tuple(gamma.shape), (len(RF_SEARCH_SPACE),))
        self.assertEqual(self.net.num_arch_parameters(), 24)
        self.assertEqual(RF_SEARCH_SPACE, (15, 29, 57, 113))

    def test_operators_are_frozen(self):
        for band, cell in self.net.cells.items():
            with self.subTest(band=band):
                self.assertEqual(cell.operator, self.OPERATORS[band])
                self.assertEqual(cell.rfs, tuple(C.RF_SPACE[band]))
        keys = set(self.net.state_dict())
        self.assertNotIn("beta", keys)
        self.assertEqual(
            len(list(self.net.arch_parameters())), 6, "RF phase must only expose gammas"
        )

    def test_forward_is_the_weighted_mixture(self):
        self.net.eval()
        cell = self.net.cells["Low"]
        x = band_input()
        weight_anchor, weight_searched = cell.rf_weights()
        with torch.no_grad():
            anchor = sum(
                weight * op(x)
                for weight, op in zip(weight_anchor, cell.anchor_pool.ops)
            )
            searched = sum(
                weight * op(x)
                for weight, op in zip(weight_searched, cell.searched_pool.ops)
            )
            expected = cell.bn(torch.cat([anchor, searched], dim=1))
            actual = cell(x)
        self.assertTrue(torch.allclose(actual, expected, atol=1e-6))

    def test_hard_mode_is_the_one_hot_argmax(self):
        self.net.eval()
        for band, cell in self.net.cells.items():
            force_peak(cell.gamma_anchor, 3)
            force_peak(cell.gamma_searched, 0)
        self.net.set_hard(True)
        x = full_input()
        with torch.no_grad():
            hard = self.net(x)
            self.net.set_hard(False)
            soft = self.net(x)
        # With a peak logit the softmax is numerically one-hot.
        self.assertTrue(torch.allclose(hard[0], soft[0], atol=1e-5))
        for band, cell in self.net.cells.items():
            with self.subTest(band=band):
                self.assertEqual(
                    self.net.selected_genes()[band]["anchor"],
                    ("dilated", cell.rfs[3]),
                )
                self.assertEqual(
                    self.net.selected_genes()[band]["searched"],
                    (self.OPERATORS[band], cell.rfs[0]),
                )

    def test_gradient_reaches_every_gamma_and_operator(self):
        self.net.train()
        self.net(full_input())[0].sum().backward()
        for name, param in self.net.named_parameters():
            with self.subTest(param=name):
                self.assertIsNotNone(param.grad, f"{name} has no gradient")
                self.assertTrue(torch.isfinite(param.grad).all())


class GenotypeExport(unittest.TestCase):
    def test_rf15_aliases_are_canonicalised(self):
        self.assertEqual(canonical_candidate_name("Low", "normal", 15), "dilated_rf15")
        self.assertEqual(canonical_candidate_name("Low", "lkdw", 15), "dwsep_rf15")
        self.assertEqual(canonical_candidate_name("Low", "normal", 29), "normal_rf29")

    def test_export_matches_forced_choices(self):
        torch.manual_seed(2)
        operators = {"Low": "normal", "Mid": "dilated", "High": "lkdw"}
        net = AnchoredRFNet(operators, n_electrodes=ELECTRODES)
        force_peak(net.cells["Low"].gamma_searched, net.cells["Low"].rfs.index(15))
        force_peak(net.cells["Mid"].gamma_anchor, net.cells["Mid"].rfs.index(113))
        force_peak(net.cells["High"].gamma_searched, net.cells["High"].rfs.index(15))

        genotype = build_anchored_genotype(net, seed=7, epoch=400)
        genes = {(gene.band, gene.path): gene for gene in genotype.genes}
        self.assertEqual(len(genes), 6)

        low_searched = genes[("Low", 1)]
        self.assertEqual(low_searched.candidate, "dilated_rf15")
        self.assertEqual(
            low_searched.candidate_index,
            candidate_names("Low").index("dilated_rf15"),
        )
        self.assertEqual(genes[("Mid", 0)].candidate, "dilated_rf113")
        self.assertEqual(genes[("High", 1)].candidate, "dwsep_rf15")

    def test_discrete_network_builds_and_runs(self):
        torch.manual_seed(3)
        operators = {"Low": "normal", "Mid": "dwsep", "High": "lkdw"}
        net = AnchoredRFNet(operators, n_electrodes=ELECTRODES)
        force_peak(net.cells["Low"].gamma_searched, net.cells["Low"].rfs.index(15))
        genotype = build_anchored_genotype(net, seed=8, epoch=400)

        model = TemporalDiscreteNet(genotype, n_electrodes=ELECTRODES)
        model.eval()
        with torch.no_grad():
            logits, features = model(full_input())
        self.assertEqual(tuple(logits.shape), (BATCH, C.NUM_CLASSES))
        keys = set(model.state_dict())
        self.assertFalse(any("alpha" in key for key in keys), "discrete net has alpha")
        self.assertFalse(any("ops." in key for key in keys), "discrete net has a candidate pool")


class SearchIntegration(unittest.TestCase):
    @staticmethod
    def _tiny_loader():
        inputs = torch.randn(4, C.NUM_BANDS, ELECTRODES, TIMEPOINTS)
        targets = torch.tensor([0, 1, 1, 0])
        return DataLoader(TensorDataset(inputs, targets), batch_size=2)

    def _make(self, model):
        optimizer = torch.optim.Adam(model.network_parameters(), lr=1e-3)
        return SearchArchitect(model, optimizer)

    def test_alpha_step_is_isolated_phase_a(self):
        torch.manual_seed(4)
        model = AnchoredOperatorNet(n_electrodes=ELECTRODES)
        model.set_rf(29)
        architect = self._make(model)
        inputs = full_input()
        targets = torch.tensor([0, 1])
        criterion = nn.NLLLoss()

        weights = [p.detach().clone() for p in model.network_parameters()]
        arch = [p.detach().clone() for p in model.arch_parameters()]
        buffers = {name: value.detach().clone() for name, value in model.named_buffers()}

        result = architect.alpha_step(inputs, targets, criterion)

        self.assertGreater(result.grad_norm, 0.0)
        self.assertTrue(
            any(not torch.equal(a, b) for a, b in zip(arch, model.arch_parameters()))
        )
        self.assertTrue(
            all(
                torch.equal(a, b)
                for a, b in zip(weights, model.network_parameters())
            ),
            "alpha step modified network weights",
        )
        for name, value in model.named_buffers():
            self.assertTrue(torch.equal(buffers[name], value), f"BN buffer {name} moved")

    def test_alpha_step_is_isolated_phase_b(self):
        torch.manual_seed(5)
        model = AnchoredRFNet({"Low": "dilated", "Mid": "dwsep", "High": "lkdw"}, n_electrodes=ELECTRODES)
        architect = self._make(model)
        inputs = full_input()
        targets = torch.tensor([1, 0])
        criterion = nn.NLLLoss()

        weights = [p.detach().clone() for p in model.network_parameters()]
        arch = [p.detach().clone() for p in model.arch_parameters()]
        buffers = {name: value.detach().clone() for name, value in model.named_buffers()}

        result = architect.alpha_step(inputs, targets, criterion)

        self.assertGreater(result.grad_norm, 0.0)
        self.assertTrue(
            any(not torch.equal(a, b) for a, b in zip(arch, model.arch_parameters()))
        )
        self.assertTrue(
            all(
                torch.equal(a, b)
                for a, b in zip(weights, model.network_parameters())
            ),
            "alpha step modified network weights",
        )
        for name, value in model.named_buffers():
            self.assertTrue(torch.equal(buffers[name], value), f"BN buffer {name} moved")

    def test_warmup_then_alpha_updates(self):
        torch.manual_seed(6)
        model = AnchoredOperatorNet(n_electrodes=ELECTRODES)
        architect = self._make(model)
        loader = self._tiny_loader()
        criterion = nn.NLLLoss()
        before = [p.detach().clone() for p in model.arch_parameters()]
        steps_seen: list[int] = []

        def install(step: int) -> None:
            steps_seen.append(step)
            model.set_rf(rf_schedule(step))

        metrics, step = run_anchored_epoch(
            architect, loader, loader, criterion,
            epoch=1, device=torch.device("cpu"), warmup_epochs=1,
            start_step=0, on_train_step=install,
        )
        self.assertFalse(metrics["alpha_updated"])
        self.assertEqual(metrics["alpha_steps"], 0)
        self.assertTrue(
            all(torch.equal(a, b) for a, b in zip(before, model.arch_parameters()))
        )
        self.assertEqual(step, 2)
        self.assertEqual(steps_seen, [0, 1])
        self.assertEqual(model.current_rf, rf_schedule(1))

        metrics, step = run_anchored_epoch(
            architect, loader, loader, criterion,
            epoch=2, device=torch.device("cpu"), warmup_epochs=1,
            start_step=step, on_train_step=install,
        )
        self.assertTrue(metrics["alpha_updated"])
        self.assertGreater(metrics["alpha_steps"], 0)
        self.assertTrue(
            any(not torch.equal(a, b) for a, b in zip(before, model.arch_parameters()))
        )
        self.assertEqual(step, 4)
        self.assertEqual(steps_seen, [0, 1, 2, 3])

    def test_inheritance_copies_only_shared_geometry(self):
        torch.manual_seed(7)
        source = AnchoredOperatorNet(n_electrodes=ELECTRODES)
        source.set_rf(29)
        operators = source.selected_operators()
        target = AnchoredRFNet(operators, n_electrodes=ELECTRODES)
        copied = inherit_operator_weights(source, target)

        self.assertEqual(len(copied), len(C.BANDS) * 2 * len(OP_SEARCH_RFS))
        for band in C.BANDS:
            with self.subTest(band=band):
                cell_a, cell_b = source.cells[band], target.cells[band]
                for target_index, rf in enumerate(cell_b.rfs):
                    if rf not in cell_a.rfs:
                        self.assertNotIn(
                            f"cells.{band}.anchor_pool.ops.{target_index}", copied
                        )
                        continue
                    source_index = cell_a.rfs.index(rf)
                    self.assertTrue(
                        states_equal(
                            cell_b.anchor_pool.ops[target_index],
                            cell_a.anchor_pools[source_index],
                        )
                    )
                    operator_index = cell_a.searched_pools[source_index].op_names.index(
                        operators[band]
                    )
                    self.assertTrue(
                        states_equal(
                            cell_b.searched_pool.ops[target_index],
                            cell_a.searched_pools[source_index].ops[operator_index],
                        )
                    )


class NoDuplicatePaths(unittest.TestCase):
    """The two paths of a band are decoded jointly, by actual structure."""

    @staticmethod
    def _collision_net(no_duplicate: bool):
        net = AnchoredRFNet(
            {"Low": "normal", "Mid": "dilated", "High": "dilated"},
            n_electrodes=ELECTRODES,
            no_duplicate_paths=no_duplicate,
        )
        cell = net.cells["Low"]
        # Both independent argmaxes point at RF 15: dilated_rf15 == normal_rf15.
        force_peak(cell.gamma_anchor, 0)
        force_peak(cell.gamma_searched, 0)
        return net, cell

    def test_names_differ_but_structures_are_equal(self):
        self.assertEqual(
            candidate_structure_key("Low", "normal_rf15"),
            candidate_structure_key("Low", "dilated_rf15"),
        )
        self.assertNotEqual(
            candidate_structure_key("Low", "normal_rf29"),
            candidate_structure_key("Low", "dilated_rf29"),
        )
        self.assertNotEqual(
            candidate_structure_key("Low", "dilated_rf15"),
            candidate_structure_key("Low", "dwsep_rf15"),
        )

    def test_without_the_flag_the_collision_is_allowed(self):
        net, cell = self._collision_net(no_duplicate=False)
        self.assertEqual(cell.select_rf_indices(), (0, 0))
        self.assertEqual(
            net.selected_genes()["Low"],
            {"anchor": ("dilated", 15), "searched": ("normal", 15)},
        )

    def test_with_the_flag_the_pair_must_differ(self):
        net, cell = self._collision_net(no_duplicate=True)
        anchor_index, searched_index = cell.select_rf_indices()
        genes = net.selected_genes()["Low"]
        self.assertNotEqual(
            candidate_structure_key(
                "Low", canonical_candidate_name("Low", *genes["anchor"])
            ),
            candidate_structure_key(
                "Low", canonical_candidate_name("Low", *genes["searched"])
            ),
        )
        # The collision at RF 15 cannot survive: one side must move.
        self.assertFalse(anchor_index == 0 and searched_index == 0)

    def test_joint_choice_is_the_best_allowed_pair(self):
        net = AnchoredRFNet(
            {"Low": "normal", "Mid": "dilated", "High": "dilated"},
            n_electrodes=ELECTRODES,
            no_duplicate_paths=True,
        )
        cell = net.cells["Low"]
        with torch.no_grad():
            cell.gamma_anchor.copy_(torch.tensor([10.0, 0.0, 0.0, 0.0]))
            cell.gamma_searched.copy_(torch.tensor([9.0, 1.0, 8.0, 0.0]))
        # Independent argmax would be (0, 0), which collides at RF 15.  Allowed
        # sums: (0,1)=11, (0,2)=18, (0,3)=10, (1,0)=9, ... so (0, 2) wins.
        self.assertEqual(cell.select_rf_indices(), (0, 2))

    def test_cross_family_rf15_is_not_a_duplicate(self):
        net = AnchoredRFNet(
            {"Low": "dwsep", "Mid": "dilated", "High": "dilated"},
            n_electrodes=ELECTRODES,
            no_duplicate_paths=True,
        )
        cell = net.cells["Low"]
        force_peak(cell.gamma_anchor, 0)  # dilated_rf15: dense k15
        force_peak(cell.gamma_searched, 0)  # dwsep_rf15: separable k15+dilation1
        self.assertEqual(cell.select_rf_indices(), (0, 0))

    def test_export_uses_the_constrained_pair(self):
        net, cell = self._collision_net(no_duplicate=True)
        genotype = build_anchored_genotype(net, seed=3, epoch=10)
        low = [gene for gene in genotype.genes if gene.band == "Low"]
        self.assertEqual(len(low), 2)
        self.assertNotEqual(
            candidate_structure_key("Low", low[0].candidate),
            candidate_structure_key("Low", low[1].candidate),
        )
        self.assertEqual(low[0].candidate, "dilated_rf15")

    def test_hard_mode_uses_the_constrained_pair(self):
        net, cell = self._collision_net(no_duplicate=True)
        net.eval()
        x = band_input()
        net.set_hard(True)
        with torch.no_grad():
            actual = cell(x)
            anchor_index, searched_index = cell.select_rf_indices()
            weight_anchor = torch.zeros_like(cell.gamma_anchor)
            weight_anchor[anchor_index] = 1.0
            weight_searched = torch.zeros_like(cell.gamma_searched)
            weight_searched[searched_index] = 1.0
            expected = cell.bn(
                torch.cat(
                    [
                        cell.anchor_pool(x, weight_anchor),
                        cell.searched_pool(x, weight_searched),
                    ],
                    dim=1,
                )
            )
            # What the unconstrained one-hot would have produced.
            colliding = cell.bn(
                torch.cat(
                    [
                        cell.anchor_pool(x, _unit_vector(cell.gamma_anchor, 0)),
                        cell.searched_pool(x, _unit_vector(cell.gamma_searched, 0)),
                    ],
                    dim=1,
                )
            )
        self.assertTrue(torch.allclose(actual, expected, atol=1e-6))
        self.assertFalse(torch.allclose(actual, colliding, atol=1e-6))


def _unit_vector(logits: torch.Tensor, index: int) -> torch.Tensor:
    weights = torch.zeros_like(logits)
    weights[index] = 1.0
    return weights


class AnchoredGenotypeLoader(unittest.TestCase):
    @staticmethod
    def _payload(net, *, seed: int = 13, epochs: int = 400, no_duplicate_paths: bool = False) -> dict:
        genotype = build_anchored_genotype(net, seed=seed, epoch=epochs)
        return {
            "scheme": ANCHORED_SCHEME,
            "seed": seed,
            "epochs": epochs,
            "no_duplicate_paths": no_duplicate_paths,
            "genes": [asdict(gene) for gene in genotype.genes],
        }

    @staticmethod
    def _write(directory: str, payload: dict) -> Path:
        path = Path(directory) / "genotype.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_round_trip(self):
        torch.manual_seed(11)
        net = AnchoredRFNet(
            {"Low": "normal", "Mid": "dwsep", "High": "lkdw"},
            n_electrodes=ELECTRODES,
            no_duplicate_paths=True,
        )
        payload = self._payload(net, no_duplicate_paths=True)
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(tmp, payload)
            genotype = load_anchored_genotype(path)
        self.assertEqual(genotype.seed, 13)
        self.assertEqual(genotype.epoch, 400)
        self.assertEqual(len(genotype.genes), 6)
        for gene in genotype.genes:
            with self.subTest(band=gene.band, path=gene.path):
                self.assertEqual(
                    gene.candidate, candidate_names(gene.band)[gene.candidate_index]
                )
                if gene.path == 0:
                    self.assertTrue(gene.candidate.startswith("dilated_rf"))

    def test_loaded_genotype_trains_a_discrete_step(self):
        torch.manual_seed(12)
        net = AnchoredRFNet(
            {"Low": "normal", "Mid": "dilated", "High": "dwsep"},
            n_electrodes=ELECTRODES,
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(tmp, self._payload(net))
            genotype = load_anchored_genotype(path)

        model = TemporalDiscreteNet(genotype, n_electrodes=ELECTRODES)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.NLLLoss()
        inputs = full_input()
        targets = torch.tensor([0, 1])
        before = [p.detach().clone() for p in model.parameters()]

        logits, _ = model(inputs)
        loss = criterion(logits, targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(
            any(not torch.equal(a, b) for a, b in zip(before, model.parameters())),
            "a discrete retrain step did not change any parameter",
        )

    def test_rejects_wrong_scheme(self):
        torch.manual_seed(13)
        net = AnchoredRFNet({"Low": "normal", "Mid": "dilated", "High": "dilated"}, n_electrodes=ELECTRODES)
        payload = self._payload(net)
        payload["scheme"] = "something_else"
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                load_anchored_genotype(self._write(tmp, payload))

    def test_rejects_non_canonical_candidate(self):
        torch.manual_seed(14)
        net = AnchoredRFNet({"Low": "normal", "Mid": "dilated", "High": "dilated"}, n_electrodes=ELECTRODES)
        payload = self._payload(net)
        for gene in payload["genes"]:
            if gene["band"] == "Low" and gene["path"] == 1:
                gene["candidate"] = "normal_rf15"
                gene["candidate_index"] = candidate_names("Low").index("dilated_rf15")
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                load_anchored_genotype(self._write(tmp, payload))

    def test_rejects_non_dilated_path0(self):
        torch.manual_seed(15)
        net = AnchoredRFNet({"Low": "normal", "Mid": "dilated", "High": "dilated"}, n_electrodes=ELECTRODES)
        payload = self._payload(net)
        for gene in payload["genes"]:
            if gene["band"] == "Low" and gene["path"] == 0:
                gene["candidate"] = "dwsep_rf29"
                gene["candidate_index"] = candidate_names("Low").index("dwsep_rf29")
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                load_anchored_genotype(self._write(tmp, payload))

    def test_rejects_duplicate_structure_when_flagged(self):
        torch.manual_seed(16)
        net = AnchoredRFNet(
            {"Low": "normal", "Mid": "dilated", "High": "dilated"},
            n_electrodes=ELECTRODES,
        )
        cell = net.cells["Low"]
        force_peak(cell.gamma_anchor, 0)
        force_peak(cell.gamma_searched, 0)
        payload = self._payload(net, no_duplicate_paths=True)
        # Both Low genes canonicalise to dilated_rf15; the file claims the
        # constraint held, so loading must refuse it.
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                load_anchored_genotype(self._write(tmp, payload))


if __name__ == "__main__":  # pragma: no cover
    unittest.main(verbosity=2)
