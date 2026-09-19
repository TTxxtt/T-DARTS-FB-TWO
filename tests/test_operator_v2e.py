"""Expressive-V2: contracts, capacity arithmetic, and the frozen-stage guard.

The Matched generation's test file asserted a fairness *rule* -- every family
inside +/-20% of the anchor.  This one cannot assert that, because the whole
point of the generation is to drop it.  What it asserts instead is:

* the invariants that still bind (shape, RF, support, gradients);
* the exact parameter and MACs arithmetic, so an architecture change fails a
  test rather than quietly moving a number that a results table quotes;
* that the budget rule really is gone -- the audit carries no pass/fail key;
* and the cross-generation guard, which is the part that protects the frozen
  45-run result from this work.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tdarts import config as C
from tdarts import temporal_ops

from tdarts.operator_v2 import (
    PARAM_TOLERANCE,
    V2_OPERATOR_NAMES,
    V2_OPERATOR_REGISTRY,
    V2TemporalOp,
    build_v2_operator,
    count_macs,
    operator_audit,
)
from tdarts.operator_v2e import (
    CAPACITY_ADVISORY_RATIO,
    E_OPERATOR_NAMES,
    E_OPERATOR_REGISTRY,
    WIDE_CONTROL_NAMES,
    build_e_capacity_control,
    build_e_operator,
    operator_capacity_audit,
)
from tdarts.operator_v2_network import V2_STANDALONE_PATH_CHANNELS
from tdarts.operator_v2e_network import OperatorV2ENet
from tools.analyze_operator_v2e import INTERACTION_P_THRESHOLD, MIN_KENDALL_TAU, assess, kendall_tau, rank_stats
from tools import analyze_operator_v2e as v2e_analyzer
from tools.operator_anova import (
    EXPRESSIVE,
    MATCHED,
    generation_mix,
    generation_of,
    refuse_mixed_generations,
)

RF = 57
BAND = "Low"
SHAPE = (2, C.IN_CHANNELS, C.NUM_ELECTRODES, 1000)

REPO_ROOT = Path(__file__).resolve().parent.parent
FROZEN_ROOT = REPO_ROOT / "run" / "outputs" / "operator_v2"

#: The frozen generation, as archived in run/outputs/operator_v2/.  These are
#: the numbers the Matched pilot's tables quote, so this work must not move them.
FROZEN_COST = {
    "dilated": (540, 11_880_000),
    "gated": (612, 13_464_000),
    "local_attention": (462, 15_114_000),
    "dynamic": (512, 11_616_006),
    "band_gated": (552, 12_078_000),
}

#: (params, MACs, declared elementwise) per Expressive family, at RF57 on
#: [1, 3, 22, 1000].  Pinned deliberately: these integers are what the capacity
#: table reports, so a change to any layer must turn a test red.
EXPRESSIVE_COST = {
    "dilated_e": (540, 11_880_000, 0),
    "gated_e": (1260, 28_248_000, 528_000),
    "local_attention_e": (624, 21_384_000, 7_920_000),
    # The extra 12 over 4 x 11,880,000 + 1,056,000 is the gate Linear counted at
    # batch size 1 -- in_features 3 x NUM_BASIS 4.  Kept exact on purpose.
    "dynamic_e": (2176, 48_576_012, 1_056_000),
    "band_gated_e": (627, 13_530_000, 66_000),
}

CONTROL_COST = {
    "wide_dilated_2p5_e": (1368, 30_096_000),
    "wide_dilated_5_e": (2736, 60_192_000),
}


class ShapeContractTests(unittest.TestCase):
    def test_every_family_preserves_electrodes_and_time(self):
        for name in E_OPERATOR_NAMES:
            with self.subTest(operator=name):
                op = build_e_operator(name, band=BAND, target_rf=RF)
                with torch.no_grad():
                    out = op(torch.zeros(SHAPE))
                self.assertEqual(
                    tuple(out.shape),
                    (SHAPE[0], V2_STANDALONE_PATH_CHANNELS, SHAPE[2], SHAPE[3]),
                )

    def test_the_contract_holds_for_every_band(self):
        for band in C.BANDS:
            for name in E_OPERATOR_NAMES:
                with self.subTest(band=band, operator=name):
                    with torch.no_grad():
                        out = build_e_operator(name, band=band, target_rf=RF)(torch.zeros(SHAPE))
                    self.assertEqual(out.shape[1], V2_STANDALONE_PATH_CHANNELS)

    def test_controls_preserve_the_contract_too(self):
        """A control that changed the shape would not be a control."""
        for name in WIDE_CONTROL_NAMES:
            with self.subTest(operator=name):
                op = build_e_capacity_control(name, band=BAND, target_rf=RF)
                with torch.no_grad():
                    out = op(torch.zeros(SHAPE))
                self.assertEqual(
                    tuple(out.shape),
                    (SHAPE[0], V2_STANDALONE_PATH_CHANNELS, SHAPE[2], SHAPE[3]),
                )

    def test_explicit_channel_convention_is_rejected_when_the_budget_differs(self):
        for name in ("gated_e", "local_attention_e"):
            with self.subTest(operator=name):
                with self.assertRaises(ValueError):
                    build_e_operator(name, band=BAND, target_rf=RF, out_channels=6)

    def test_unknown_operator_is_rejected(self):
        with self.assertRaises(ValueError):
            build_e_operator("transformer", band=BAND, target_rf=RF)

    def test_the_two_registries_cannot_be_confused(self):
        """A control reachable through build_e_operator could be submitted as a
        pilot candidate by a one-word typo; a candidate reachable through the
        control builder would corrupt the ablation."""
        self.assertTrue(set(E_OPERATOR_NAMES).isdisjoint(WIDE_CONTROL_NAMES))
        with self.assertRaises(ValueError):
            build_e_operator("wide_dilated_5_e", band=BAND, target_rf=RF)
        with self.assertRaises(ValueError):
            build_e_capacity_control("dilated_e", band=BAND, target_rf=RF)

    def test_a_control_must_actually_widen(self):
        with self.assertRaises(ValueError):
            build_e_capacity_control(
                "wide_dilated_2p5_e", band=BAND, target_rf=RF, out_channels=64
            )


class GeometryContractTests(unittest.TestCase):
    def test_every_family_reads_exactly_the_target_receptive_field(self):
        for name in E_OPERATOR_NAMES:
            with self.subTest(operator=name):
                self.assertEqual(build_e_operator(name, band=BAND, target_rf=RF).support, (15, 4, RF))

    def test_capacity_is_never_bought_with_support(self):
        """The one thing the wide controls must not do.  A control with a wider
        window would be a different experiment wearing a capacity label."""
        for name in WIDE_CONTROL_NAMES:
            with self.subTest(operator=name):
                self.assertEqual(
                    build_e_capacity_control(name, band=BAND, target_rf=RF).support, (15, 4, RF)
                )

    def test_support_matches_for_every_band(self):
        for band in C.BANDS:
            for name in E_OPERATOR_NAMES + WIDE_CONTROL_NAMES:
                with self.subTest(band=band, operator=name):
                    builder = build_e_operator if name in E_OPERATOR_NAMES else build_e_capacity_control
                    self.assertEqual(builder(name, band=band, target_rf=RF).support[2], RF)

    def test_one_by_one_layers_do_not_widen_the_window(self):
        """The gate and projection layers are RF 1; if one of them grew a
        temporal kernel the family's support would move without the dilated
        convolution changing at all."""
        for name in ("gated_e", "local_attention_e", "band_gated_e"):
            with self.subTest(operator=name):
                self.assertEqual(build_e_operator(name, band=BAND, target_rf=RF).support, (15, 4, RF))

    def test_attention_gathers_the_same_positions_as_the_matched_family(self):
        """The only forward-code this generation shares with the frozen one is
        the unfold gather.  Comparing the two implementations directly is
        stronger than asserting each one's shape."""
        frozen = build_v2_operator("local_attention", band=BAND, target_rf=RF).op
        expressive = build_e_operator("local_attention_e", band=BAND, target_rf=RF).op
        x = torch.randn(2, C.IN_CHANNELS, C.NUM_ELECTRODES, 40)
        self.assertTrue(torch.equal(frozen._patches(x), expressive._patches(x)))
        self.assertEqual(expressive._patches(x).shape[2], 15)


class CapacityInvariantTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = {row["operator"]: row for row in operator_capacity_audit(band=BAND, target_rf=RF)}

    def test_the_audit_carries_no_pass_fail_band(self):
        """The Matched audit attaches `within_tolerance` because +/-20% was a
        requirement there.  Here it is advisory, so the key must be absent --
        asserting its absence rather than its truth is what stops the budget
        rule being quietly reinstated."""
        for name, row in self.rows.items():
            with self.subTest(operator=name):
                self.assertNotIn("within_tolerance", row)

    def test_every_family_records_that_it_matches_the_anchor_support(self):
        for name, row in self.rows.items():
            with self.subTest(operator=name):
                self.assertTrue(row["support_matches_anchor"])

    def test_only_the_five_x_control_crosses_the_advisory(self):
        for name in E_OPERATOR_NAMES:
            with self.subTest(operator=name):
                self.assertFalse(self.rows[name]["exceeds_advisory"])
        self.assertFalse(self.rows["wide_dilated_2p5_e"]["exceeds_advisory"])
        self.assertTrue(
            self.rows["wide_dilated_5_e"]["exceeds_advisory"],
            "wide_dilated_5 exists to exceed the advisory ceiling",
        )
        self.assertGreater(CAPACITY_ADVISORY_RATIO, 1.0)

    def test_capacity_ratio_is_the_worse_of_params_and_macs(self):
        for name, row in self.rows.items():
            with self.subTest(operator=name):
                self.assertAlmostEqual(
                    row["capacity_ratio"],
                    max(row["param_ratio_vs_anchor"], row["mac_ratio_vs_anchor"]),
                )

    def test_the_anchor_is_the_only_family_at_one_x(self):
        self.assertEqual(self.rows["dilated_e"]["param_ratio_vs_anchor"], 1.0)
        self.assertEqual(self.rows["dilated_e"]["mac_ratio_vs_anchor"], 1.0)

    def test_attention_declares_the_work_hooks_cannot_see(self):
        """The opposite failure from the capacity flag: a counter that misses
        elementwise work reports attention as *cheaper* than a convolution and
        makes a compute-heavy mechanism look free."""
        row = self.rows["local_attention_e"]
        self.assertGreater(row["macs_elementwise_declared"], 0)
        self.assertGreater(row["macs"], row["macs_hook_only"])
        self.assertGreater(row["mac_ratio_vs_anchor"], 1.0)

    def test_the_gate_families_declare_their_elementwise_work(self):
        """Their Matched counterparts declare nothing, so the two generations'
        MACs are not measured identically.  Pinning the declaration here makes
        that asymmetry deliberate rather than incidental."""
        for name in ("gated_e", "band_gated_e"):
            with self.subTest(operator=name):
                self.assertGreater(self.rows[name]["macs_elementwise_declared"], 0)

    def test_the_audit_can_omit_the_controls(self):
        rows = operator_capacity_audit(band=BAND, target_rf=RF, include_controls=False)
        self.assertEqual({row["operator"] for row in rows}, set(E_OPERATOR_NAMES))
        self.assertTrue(all(row["role"] == "candidate" for row in rows))

    def test_attention_really_is_two_heads_of_six(self):
        """A silent collapse to one head of 12 would leave the parameter count
        and the output shape identical, and would only show up as a weaker
        mechanism -- so it is asserted on the tensor, not on the constants."""
        op = build_e_operator("local_attention_e", band=BAND, target_rf=RF).op
        self.assertEqual(op.NUM_HEADS, 2)
        self.assertEqual(op.HEAD_DIM, 6)
        hidden = op.act(op.embed(torch.zeros(SHAPE)))
        batch, _, elec, time = hidden.shape
        query = op.q(hidden).view(batch, op.NUM_HEADS, op.HEAD_DIM, elec, time)
        self.assertEqual(tuple(query.shape), (SHAPE[0], 2, 6, elec, time))

    def test_the_attention_embedding_is_nonlinear(self):
        """The nonlinearity is what makes a 6-wide head non-redundant.  Without
        it the score collapses to a rank-3 linear map of the input channels and
        the family falls back into the Matched family's score class."""
        op = build_e_operator("local_attention_e", band=BAND, target_rf=RF).op
        self.assertNotIsInstance(op.act, torch.nn.Identity)
        x = torch.randn(1, C.IN_CHANNELS, 4, 8) * 3.0
        self.assertFalse(torch.allclose(op.act(x), x))

    def test_dynamic_uses_four_full_basis_kernels(self):
        op = build_e_operator("dynamic_e", band=BAND, target_rf=RF).op
        self.assertEqual(op.NUM_BASIS, 4)
        self.assertEqual(len(op.bases), 4)
        # Dense, not depthwise: groups == 1 is what makes the K=4 bank a real
        # basis expansion rather than four depthwise filters.
        self.assertTrue(all(basis.groups == 1 for basis in op.bases))


class CapacityArithmeticTests(unittest.TestCase):
    def test_expressive_families_cost_exactly_what_is_recorded(self):
        for name, (params, macs, declared) in EXPRESSIVE_COST.items():
            with self.subTest(operator=name):
                op = build_e_operator(name, band=BAND, target_rf=RF)
                self.assertEqual(op.num_op_params, params)
                self.assertEqual(count_macs(op, (1, C.IN_CHANNELS, C.NUM_ELECTRODES, 1000)), macs)
                self.assertEqual(int(op.op.extra_macs((1, C.IN_CHANNELS, C.NUM_ELECTRODES, 1000))), declared)

    def test_controls_cost_exactly_what_is_recorded(self):
        for name, (params, macs) in CONTROL_COST.items():
            with self.subTest(operator=name):
                op = build_e_capacity_control(name, band=BAND, target_rf=RF)
                self.assertEqual(op.num_op_params, params)
                self.assertEqual(count_macs(op, (1, C.IN_CHANNELS, C.NUM_ELECTRODES, 1000)), macs)

    def test_the_expressive_anchor_is_the_frozen_anchor(self):
        expressive = build_e_operator("dilated_e", band=BAND, target_rf=RF)
        frozen = build_v2_operator("dilated", band=BAND, target_rf=RF)
        self.assertEqual(expressive.num_op_params, frozen.num_op_params)

    def test_macs_counter_probes_on_the_modules_own_device(self):
        """Mirrors the Matched guard: a CPU probe against a module elsewhere
        fails, and a GPU training run measures the cell after `.to(device)`."""
        shape = (1, C.IN_CHANNELS, C.NUM_ELECTRODES, 1000)
        for name in E_OPERATOR_NAMES + WIDE_CONTROL_NAMES:
            with self.subTest(operator=name):
                builder = build_e_operator if name in E_OPERATOR_NAMES else build_e_capacity_control
                op = builder(name, band=BAND, target_rf=RF)
                self.assertEqual(count_macs(op, shape), count_macs(op.to("meta"), shape))


class GradientTests(unittest.TestCase):
    def test_forward_backward_is_finite_for_every_family(self):
        for name in E_OPERATOR_NAMES:
            with self.subTest(operator=name):
                torch.manual_seed(0)
                op = build_e_operator(name, band=BAND, target_rf=RF)
                x = torch.randn(2, C.IN_CHANNELS, C.NUM_ELECTRODES, 200, requires_grad=True)
                out = op(x)
                self.assertTrue(torch.isfinite(out).all(), f"{name} produced non-finite output")
                out.sum().backward()
                self.assertTrue(torch.isfinite(x.grad).all())
                grads = [p.grad for p in op.parameters() if p.grad is not None]
                self.assertTrue(grads, f"{name} produced no gradients")
                self.assertTrue(all(torch.isfinite(g).all() for g in grads))
                self.assertTrue(any(g.abs().sum() > 0 for g in grads))


class NetworkTests(unittest.TestCase):
    def test_every_family_builds_a_network_with_the_same_backbone(self):
        counts = {
            sum(p.numel() for p in OperatorV2ENet(name, target_rf=RF).backbone.parameters())
            for name in E_OPERATOR_NAMES
        }
        self.assertEqual(len(counts), 1, f"backbone differs across families: {counts}")

    def test_the_expressive_backbone_is_the_frozen_backbone(self):
        from tdarts.operator_v2_network import OperatorV2Net

        expressive = sum(p.numel() for p in OperatorV2ENet("dilated_e", target_rf=RF).backbone.parameters())
        frozen = sum(p.numel() for p in OperatorV2Net("dilated", target_rf=RF).backbone.parameters())
        self.assertEqual(expressive, frozen)

    def test_network_accepts_the_loader_layout_and_returns_log_probabilities(self):
        net = OperatorV2ENet("gated_e", target_rf=RF)
        x = torch.zeros(3, 1, C.NUM_ELECTRODES, 1000, C.NUM_BANDS)
        logits, features = net(x)
        self.assertEqual(tuple(logits.shape), (3, C.NUM_CLASSES))
        self.assertEqual(features.dim(), 2)
        self.assertTrue(torch.allclose(logits.exp().sum(dim=1), torch.ones(3), atol=1e-5))

    def test_network_rejects_a_wrong_channel_count(self):
        net = OperatorV2ENet("dynamic_e", target_rf=RF)
        with self.assertRaises(ValueError):
            net(torch.zeros(1, 1, C.NUM_ELECTRODES, 1000, 5))

    def test_every_family_is_reachable_from_the_network(self):
        for name in E_OPERATOR_NAMES:
            with self.subTest(operator=name):
                net = OperatorV2ENet(name, target_rf=RF)
                logits, _ = net(torch.zeros(1, C.NUM_BANDS, C.NUM_ELECTRODES, 1000))
                self.assertEqual(tuple(logits.shape), (1, C.NUM_CLASSES))

    def test_the_network_can_build_a_control_but_the_pilot_builder_cannot(self):
        """The ablation runs through the same network and protocol as the pilot
        -- that is what makes the capacity comparison valid.  What keeps a
        control out of the grid is the CLI flag and the strict candidate
        builder, not the network."""
        net = OperatorV2ENet("wide_dilated_5_e", target_rf=RF)
        logits, _ = net(torch.zeros(1, C.NUM_BANDS, C.NUM_ELECTRODES, 1000))
        self.assertEqual(tuple(logits.shape), (1, C.NUM_CLASSES))
        with self.assertRaises(ValueError):
            build_e_operator("wide_dilated_5_e", band=BAND, target_rf=RF)


class CrossGenerationGuardTests(unittest.TestCase):
    """These tests exist to protect the frozen stage from this work."""

    def test_the_generations_are_disjoint_and_both_five_wide(self):
        self.assertEqual(len(V2_OPERATOR_NAMES), 5)
        self.assertEqual(len(E_OPERATOR_NAMES), 5)
        self.assertTrue(set(E_OPERATOR_NAMES).isdisjoint(V2_OPERATOR_NAMES))

    def test_every_expressive_name_carries_the_generation_suffix(self):
        """Generation is derived from the name, not from the output root, so a
        name without the suffix is filed as Matched and slips past the pooling
        guard.  The wide controls are the easy ones to get wrong -- they are
        named for their width, and an earlier revision of this module left them
        unsuffixed, which is how the flaw was found.
        """
        for name in V2_OPERATOR_NAMES:
            with self.subTest(operator=name):
                self.assertEqual(generation_of(name), MATCHED)
        for name in E_OPERATOR_NAMES + WIDE_CONTROL_NAMES:
            with self.subTest(operator=name):
                self.assertTrue(name.endswith("_e"))
                self.assertEqual(generation_of(name), EXPRESSIVE)

    def test_frozen_families_still_cost_what_was_archived(self):
        shape = (1, C.IN_CHANNELS, C.NUM_ELECTRODES, 1000)
        for name, (params, macs) in FROZEN_COST.items():
            with self.subTest(operator=name):
                op = build_v2_operator(name, band=BAND, target_rf=RF)
                self.assertEqual(op.num_op_params, params)
                self.assertEqual(count_macs(op, shape), macs)
                self.assertEqual(op.support, (15, 4, RF))

    def test_the_registry_fallback_tests_identity_against_none(self):
        """``registry or V2_OPERATOR_REGISTRY`` would treat an explicitly empty
        registry as "not supplied" and quietly build a frozen family instead --
        a caller asking for a namespace that cannot resolve anything would get
        a working operator back.  The fallback must test identity, nothing else.
        """
        for name in ("dilated", "dilated_e"):
            with self.subTest(operator=name):
                with self.assertRaises(ValueError):
                    build_v2_operator(name, band=BAND, target_rf=RF, registry={})

    def test_a_supplied_registry_is_never_mutated(self):
        registry = dict(E_OPERATOR_REGISTRY)
        before = dict(registry)
        build_v2_operator("gated_e", band=BAND, target_rf=RF, registry=registry)
        self.assertEqual(registry, before)

    def test_omitting_the_registry_is_the_frozen_default(self):
        default = build_v2_operator("gated", band=BAND, target_rf=RF)
        explicit = build_v2_operator("gated", band=BAND, target_rf=RF, registry=V2_OPERATOR_REGISTRY)
        self.assertIs(type(default.op), type(explicit.op))
        self.assertEqual(default.num_op_params, explicit.num_op_params)
        # And it really is the frozen registry, not a copy that could drift.
        self.assertIs(V2TemporalOp.__init__.__defaults__[-1], None)

    def test_the_frozen_audit_keeps_its_pass_fail_band(self):
        """The Expressive audit has no such key.  The Matched one must keep
        both, or the frozen generation's rule silently changed."""
        self.assertAlmostEqual(PARAM_TOLERANCE, 0.20)
        for row in operator_audit(band=BAND, target_rf=RF):
            self.assertIn("within_tolerance", row)

    def test_the_darts_search_pool_was_not_touched(self):
        """The invariant behind the whole separate-registry design: adding a
        family to temporal_ops.OPERATOR_REGISTRY would change
        canonical_candidates, which changes the search space, which invalidates
        every archived search and every genotype decoded from one."""
        counts = {band: len(candidates) for band, candidates in temporal_ops.canonical_candidates_all().items()}
        self.assertEqual(counts, {band: 14 for band in C.BANDS})
        self.assertEqual(temporal_ops.num_canonical_candidates(), 42)
        for name in E_OPERATOR_NAMES + WIDE_CONTROL_NAMES:
            self.assertNotIn(name, temporal_ops.OPERATOR_REGISTRY)

    def test_the_frozen_entry_point_guards_still_hold(self):
        module = _load_entry_point("train_operator_v2.py", "frozen_entry_under_test")
        self.assertEqual(module.DEFAULT_RF, 57)
        source = (REPO_ROOT / "train_operator_v2.py").read_text(encoding="utf-8")
        self.assertEqual(source.count("load_subject_session("), 1)
        self.assertEqual(set(module.V2_OPERATOR_NAMES), set(V2_OPERATOR_NAMES))

    @unittest.skipUnless(FROZEN_ROOT.is_dir(), "the frozen Matched-V2 runs are not on disk")
    def test_the_archived_runs_agree_with_the_pinned_numbers(self):
        """The strongest form of the guard: the numbers above are checked
        against what the 45 finished runs actually recorded, not just against
        each other."""
        seen = set()
        for summary in sorted(FROZEN_ROOT.glob(f"**/final_summary.json")):
            payload = json.loads(summary.read_text(encoding="utf-8"))
            operator = payload["operator"]
            seen.add(operator)
            with self.subTest(operator=operator, run=summary.parent.name):
                params, macs = FROZEN_COST[operator]
                self.assertEqual(payload["operator_params"], params)
                self.assertEqual(payload["operator_macs"], macs)
                self.assertEqual(payload["support"]["span"], RF)
                self.assertFalse(payload["session1_opened"])
        self.assertEqual(seen, set(FROZEN_COST))


class AnalyzerControlExclusionTests(unittest.TestCase):
    """The wide controls must not reach the pilot ANOVA by default.

    One control run in a root would add a sixth operator level with five empty
    cells, which does not merely shrink n -- it changes the operator main effect
    and the df of the interaction, i.e. the numbers the verdict is read from.
    """

    def _root(self, tmp: Path, *, with_control: bool) -> Path:
        base = tmp / "bci42a"
        subjects = ("003", "005")
        operators = ("dilated_e", "gated_e")
        runs = [
            (subject, operator, seed)
            for subject in subjects
            for operator in operators
            for seed in (20250901, 20250902)
        ]
        if with_control:
            runs.append(("003", "wide_dilated_5_e", 20250901))
        for subject, operator, seed in runs:
            leaf = base / f"train_s{subject}_seed{seed}_operator_v2e_{operator}"
            leaf.mkdir(parents=True)
            nll = 0.4 + 0.1 * operators.index(operator) if operator in operators else 5.0
            (leaf / "final_summary.json").write_text(
                json.dumps(
                    {
                        "subject": subject,
                        "seed": seed,
                        "operator": operator,
                        "operator_role": "candidate" if operator in operators else "capacity_control",
                        "params": 18112,
                        "macs": 11880000,
                        "operator_params": 540,
                        "operator_macs": 11880000,
                        "operator_param_ratio_vs_anchor": 1.0,
                        "val_best_acc": 0.5,
                        "val_best_nll": nll,
                        "support": {"positions": 15, "spacing": 4, "span": 57},
                        "target_rf": 57,
                        "session1_opened": False,
                        "screening_only": True,
                        "test": None,
                    }
                ),
                encoding="utf-8",
            )
        return tmp

    def _run(self, root: Path, out: Path, *extra: str) -> dict:
        argv = ["analyze_operator_v2e.py", "--root", str(root), "--json", str(out), *extra]
        with contextlib.redirect_stdout(io.StringIO()):
            with mock.patch.object(sys, "argv", argv):
                self.assertEqual(v2e_analyzer.main(), 0)
        return json.loads(out.read_text(encoding="utf-8"))

    def test_a_control_run_is_excluded_from_the_anova_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            payload = self._run(self._root(Path(tmp), with_control=True), Path(tmp) / "a.json")
        self.assertNotIn("wide_dilated_5_e", payload["anova"]["operators"])
        self.assertEqual(sorted(payload["anova"]["operators"]), ["dilated_e", "gated_e"])
        # ...but it is still reported, not silently dropped.
        self.assertIn("wide_dilated_5_e", payload["capacity"])

    def test_the_control_is_never_a_documented_candidate(self):
        self.assertNotIn("wide_dilated_5_e", v2e_analyzer.OPERATORS)
        self.assertNotIn("wide_dilated_2p5_e", v2e_analyzer.OPERATORS)

    def test_the_grid_without_controls_is_unchanged_by_their_presence(self):
        """Adding a control run must not move the candidates' decomposition."""
        with tempfile.TemporaryDirectory() as tmp:
            plain = self._run(self._root(Path(tmp) / "plain", with_control=False), Path(tmp) / "p.json")
        with tempfile.TemporaryDirectory() as tmp:
            mixed = self._run(self._root(Path(tmp) / "mixed", with_control=True), Path(tmp) / "m.json")
        self.assertEqual(plain["anova"]["components"], mixed["anova"]["components"])


class GenerationPoolingTests(unittest.TestCase):
    """Cross-generation pooling is refused by default, not merely discouraged."""

    @staticmethod
    def _runs(*operators):
        return [
            {"operator": operator, "path": f"/x/train_s003_seed1_operator_v2_{operator}"}
            for operator in operators
        ]

    def test_a_single_generation_is_always_allowed(self):
        for names in (("gated", "dynamic"), ("gated_e", "dynamic_e")):
            with self.subTest(operators=names):
                self.assertTrue(refuse_mixed_generations(self._runs(*names), combine=False))

    def test_pooling_two_generations_is_refused_by_default(self):
        runs = self._runs("gated", "dynamic_e")
        with contextlib.redirect_stdout(io.StringIO()) as captured:
            self.assertFalse(refuse_mixed_generations(runs, combine=False))
        # The refusal must name the offending runs, or the reader cannot act.
        output = captured.getvalue()
        self.assertIn("gated", output)
        self.assertIn("dynamic_e", output)

    def test_pooling_is_allowed_only_when_asked_for_explicitly(self):
        runs = self._runs("gated", "dynamic_e")
        self.assertTrue(refuse_mixed_generations(runs, combine=True))

    def test_the_mix_counts_are_reported(self):
        self.assertEqual(generation_mix(self._runs("gated", "dynamic", "gated_e")), {MATCHED: 2, EXPRESSIVE: 1})


class AssessTests(unittest.TestCase):
    """The verdict must not be a function of the variance ratio alone.

    The Matched stage decided on a single `V_interaction / V_seed > 1` switch,
    and that switch flipped between 0.833 and 1.187 depending only on whether
    the metric was raw or logged.  These tests pin the replacement's defining
    property: a large ratio is *necessary but not sufficient*.
    """

    @staticmethod
    def _assess(**overrides):
        inputs = dict(
            interaction_over_seed=1.5,
            interaction_p=0.01,
            distinct_winners=3,
            separated_subjects=3,
            unstable_subjects=0,
            min_tau=0.8,
            n_subjects=3,
        )
        inputs.update(overrides)
        return assess(**inputs)

    def test_all_three_conditions_met_proceeds(self):
        result = self._assess()
        self.assertTrue(result["c1_interaction_large_enough"])
        self.assertTrue(result["c2_subjects_differ"])
        self.assertTrue(result["c3_winner_stable"])
        self.assertEqual(result["verdict"], "PROCEED")

    def test_a_large_ratio_with_unstable_winners_is_not_proceed(self):
        """The case the whole redesign exists for: a ratio well above 1 that
        still must not authorise a per-subject search, because the winner moves
        between seeds."""
        result = self._assess(interaction_over_seed=3.0, unstable_subjects=2, min_tau=0.1)
        self.assertTrue(result["c1_interaction_large_enough"])
        self.assertFalse(result["c3_winner_stable"])
        self.assertEqual(result["verdict"], "AMBIGUOUS")

    def test_a_large_ratio_with_one_winner_is_ambiguous(self):
        result = self._assess(interaction_over_seed=3.0, distinct_winners=1, separated_subjects=0)
        self.assertTrue(result["c1_interaction_large_enough"])
        self.assertFalse(result["c2_subjects_differ"])
        self.assertEqual(result["verdict"], "AMBIGUOUS")

    def test_no_interaction_is_weak_global_however_stable_the_ranking(self):
        result = self._assess(interaction_over_seed=0.4, interaction_p=0.6)
        self.assertFalse(result["c1_interaction_large_enough"])
        self.assertEqual(result["verdict"], "WEAK_GLOBAL")

    def test_the_p_value_alone_can_satisfy_c1(self):
        """A ratio below 1 with a significant interaction is exactly the Matched
        data's situation on the raw scale; it must not be read as no effect."""
        result = self._assess(interaction_over_seed=0.83, interaction_p=0.0057)
        self.assertTrue(result["c1_via_p"])
        self.assertFalse(result["c1_via_ratio"])
        self.assertTrue(result["c1_interaction_large_enough"])
        self.assertEqual(result["verdict"], "PROCEED")

    def test_c2_needs_more_than_one_subject_to_separate(self):
        """Otherwise a single outlying subject carries the interaction."""
        result = self._assess(distinct_winners=2, separated_subjects=1)
        self.assertFalse(result["c2_subjects_differ"])
        result = self._assess(distinct_winners=2, separated_subjects=2)
        self.assertTrue(result["c2_subjects_differ"])

    def test_c3_needs_a_rank_floor_as_well_as_argmax_agreement(self):
        result = self._assess(min_tau=MIN_KENDALL_TAU - 0.1)
        self.assertFalse(result["c3_winner_stable"])
        result = self._assess(min_tau=float("nan"))
        self.assertFalse(result["c3_winner_stable"], "an unmeasurable tau is not a pass")
        result = self._assess(interaction_p=None)
        self.assertFalse(result["c1_via_p"], "no scipy means no p-value, not a free pass")

    def test_the_p_threshold_is_the_documented_one(self):
        result = self._assess(interaction_over_seed=0.5, interaction_p=INTERACTION_P_THRESHOLD)
        self.assertTrue(result["c1_via_p"])
        result = self._assess(interaction_over_seed=0.5, interaction_p=INTERACTION_P_THRESHOLD + 0.01)
        self.assertFalse(result["c1_via_p"])


class RankStabilityTests(unittest.TestCase):
    """C2 and C3 are scale-free, which is what lets them carry the decision."""

    @staticmethod
    def _rows(transform=lambda value: value):
        """A grid where each subject has a distinct, seed-stable winner."""
        preference = {"003": 0.1, "005": 0.3, "006": 0.5}
        rows = []
        for subject, offset in preference.items():
            for seed in (20250901, 20250902, 20250903):
                for index, operator in enumerate(E_OPERATOR_NAMES):
                    # A small seed-dependent jitter that never reorders.
                    jitter = 0.001 * ((seed % 10) + index)
                    rows.append(
                        {
                            "subject": subject,
                            "seed": seed,
                            "operator": operator,
                            "val_best_nll": transform(offset + 0.1 * index + jitter),
                        }
                    )
        return rows

    def _stats(self, rows):
        self._ = None
        return rank_stats(rows, ["003", "005", "006"], list(E_OPERATOR_NAMES), "val_best_nll")

    def test_a_stable_grid_reports_no_instability(self):
        stats = self._stats(self._rows())
        self.assertEqual(stats["hard_instability"], [])
        self.assertAlmostEqual(stats["min_tau"], 1.0)

    def test_kendall_tau_is_one_for_identical_orderings(self):
        names = ["a", "b", "c", "d"]
        self.assertAlmostEqual(kendall_tau(names, names), 1.0)
        self.assertAlmostEqual(kendall_tau(names, list(reversed(names))), -1.0)
        self.assertAlmostEqual(kendall_tau(names, ["a", "b", "d", "c"]), 2 / 3)

    def test_a_flipping_argmax_is_detected(self):
        rows = self._rows()
        for row in rows:
            if row["subject"] == "003" and row["seed"] == 20250903:
                # Push one operator to the front for a single seed only.
                row["val_best_nll"] = -1.0 - row["val_best_nll"]
        stats = self._stats(rows)
        self.assertIn("003", stats["hard_instability"])

    def test_rank_stability_is_identical_under_a_monotone_transform(self):
        """The property that makes the scale choice unable to rig the verdict:
        log moves the variance decomposition, not the orderings."""
        raw = self._rows()
        logged = self._rows(transform=lambda value: __import__("math").log(value))
        self.assertEqual(self._stats(raw), self._stats(logged))


def _load_entry_point(filename: str, module_name: str):
    """Load an entry point by path; the repo has no package to import it from."""
    path = REPO_ROOT / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class EntryPointTests(unittest.TestCase):
    @staticmethod
    def _module():
        return _load_entry_point("train_operator_v2e.py", "v2e_entry_under_test")

    @staticmethod
    def _source() -> str:
        return (REPO_ROOT / "train_operator_v2e.py").read_text(encoding="utf-8")

    def test_session1_is_closed_unless_explicitly_requested(self):
        module = self._module()
        with mock.patch.object(sys, "argv", ["train_operator_v2e.py", "--operator", "gated_e", "--seed", "1"]):
            self.assertFalse(module.parse_args().read_session1)
        with mock.patch.object(
            sys, "argv", ["train_operator_v2e.py", "--operator", "gated_e", "--seed", "1", "--read-session1"]
        ):
            self.assertTrue(module.parse_args().read_session1)

    def test_the_session1_loader_is_reached_only_through_that_flag(self):
        source = self._source()
        self.assertEqual(source.count("load_subject_session("), 1)
        index = source.index("load_subject_session(")
        window = source[max(0, index - 200): index + 200]
        self.assertIn("args.read_session1", window)

    def test_the_leaf_name_carries_the_operator(self):
        source = self._source()
        self.assertRegex(source, r'arm = f"\{args\.arm\}_\{target\}"')
        self.assertRegex(source, r"args\.arm = arm")

    def test_default_receptive_field_is_the_middle_rung(self):
        module = self._module()
        self.assertEqual(module.DEFAULT_RF, 57)
        self.assertIn(module.DEFAULT_RF, C.RF_SPACE["Low"])

    def test_every_family_is_reachable_from_the_cli(self):
        module = self._module()
        self.assertEqual(set(module.E_OPERATOR_NAMES), set(E_OPERATOR_NAMES))
        self.assertEqual(len(module.E_OPERATOR_NAMES), 5)

    def test_the_pilot_grid_cannot_contain_a_control(self):
        """`--operator` must not accept a control: the grid is a grid of
        candidates, and a control in it would add an ANOVA level with mostly
        empty cells."""
        module = self._module()
        for name in WIDE_CONTROL_NAMES:
            self.assertNotIn(name, module.E_OPERATOR_NAMES)
        self.assertEqual(set(module.WIDE_CONTROL_NAMES), set(WIDE_CONTROL_NAMES))

    def test_operator_and_capacity_control_are_mutually_exclusive(self):
        module = self._module()
        # argparse writes its usage to stderr before raising; swallowing it
        # keeps a passing run's output readable.
        with contextlib.redirect_stderr(io.StringIO()):
            with mock.patch.object(
                sys,
                "argv",
                ["train_operator_v2e.py", "--operator", "gated_e", "--capacity-control", "wide_dilated_5_e", "--seed", "1"],
            ):
                with self.assertRaises(SystemExit):
                    module.parse_args()
            with mock.patch.object(sys, "argv", ["train_operator_v2e.py", "--seed", "1"]):
                with self.assertRaises(SystemExit):
                    module.parse_args()

    def test_an_arm_must_stay_in_the_expressive_tree(self):
        """The most damaging thing this script could do is write a run into the
        frozen stage's output root, where the Matched analyzer's directory walk
        would fold it into the archived ANOVA."""
        module = self._module()
        with mock.patch.object(
            sys,
            "argv",
            ["train_operator_v2e.py", "--operator", "gated_e", "--seed", "1", "--arm", "operator_v2"],
        ):
            with self.assertRaises(SystemExit):
                module.main()
        self.assertIn('args.arm.startswith(ARM_PREFIX)', self._source())
        self.assertEqual(module.ARM_PREFIX, "operator_v2e")

    def test_the_default_arm_matches_the_prefix_guard(self):
        module = self._module()
        with mock.patch.object(sys, "argv", ["train_operator_v2e.py", "--operator", "gated_e", "--seed", "1"]):
            self.assertTrue(module.parse_args().arm.startswith(module.ARM_PREFIX))

    def test_the_summary_keeps_the_keys_the_analyzer_requires(self):
        """discover_runs skips a run whose summary is missing any of these, so a
        dropped key would silently shrink the grid rather than fail loudly."""
        source = self._source()
        for key in ("val_best_acc", "val_best_nll", "operator_params", "operator_macs",
                    "session1_opened", "screening_only"):
            with self.subTest(key=key):
                self.assertIn(f'"{key}"', source)


if __name__ == "__main__":
    unittest.main(verbosity=2)
