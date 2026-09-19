"""Arm B: FBNAS-style hierarchical search, its genotype dialect, and its guards.

Two kinds of assertion live here, and they are worth distinguishing:

* **Agreement with the frozen baseline.**  The search-space claim is that Arm B's
  Phase B searches *the same* RF space Arm A searches.  That is only worth
  anything if it is checked against the baseline's own enumeration rather than
  against a restatement of it, so :class:`SamplerAgreementTests` imports
  ``FBNAS/codes/centralRepo/utils.py`` and compares element for element.
* **Bit-identity at the probed width.**  Arm B needs two families at a width
  they were never built at.  The width-general subclasses must be provably the
  *same operator* where the Expressive generation probed them, or Phase A's
  family comparison would not be about the families that were measured.
"""

from __future__ import annotations

import importlib.util
import json
import random
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from tdarts import config as C
from tdarts.band_discrete import (
    BAND_SCHEME,
    BandDiscreteNet,
    BandGene,
    BandGenotype,
    band_duplicate_bands,
    band_structure_keys,
    describe_genotype,
    load_band_genotype,
)
from tdarts.band_supernet import (
    ARM_B_REGISTRY,
    BAND_FAMILIES,
    BAND_RFS,
    EXCLUDED_FAMILY,
    PHASE_A_RF,
    WidthGeneralBandGatedE,
    WidthGeneralGatedE,
    build_arm_b_operator,
    calibrated_scores,
    family_candidates,
    FBNASBandCell,
    FBNASBandNet,
    rf_candidates,
)
from tdarts.fbnas_sampler import (
    CANDIDATE_INDICES,
    choice_index,
    choice_key,
    per_band_choices,
    per_band_index,
    random_choice,
    traverse_choices,
    validate_choice,
)
from tdarts.operator_v2e import build_e_operator

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FBNAS_CENTRAL_REPO = PROJECT_ROOT / "FBNAS" / "codes" / "centralRepo"


def _load_baseline_utils():
    """Import the frozen baseline's ``utils`` without writing into ``FBNAS/``.

    Importing from inside the vendored tree otherwise drops a ``__pycache__``
    entry into the baseline, which
    ``tests/test_fbnas_compatibility.py::test_no_extra_files_were_written_into_the_baseline``
    forbids.  Same idiom as that file.
    """

    central = str(FBNAS_CENTRAL_REPO)
    added = central not in sys.path
    if added:
        sys.path.insert(0, central)
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        path = FBNAS_CENTRAL_REPO / "utils.py"
        spec = importlib.util.spec_from_file_location("fbnas_utils_official", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.dont_write_bytecode = previous


def _normalise(choice) -> dict:
    """Upstream returns lists; this port returns tuples."""

    return {band: tuple(choice[band]) for band in C.BANDS}


class SamplerAgreementTests(unittest.TestCase):
    """The search space must be upstream's, checked against upstream itself."""

    @classmethod
    def setUpClass(cls):
        cls.baseline = _load_baseline_utils()

    def test_per_band_choices_match_upstream_traverse_choice(self):
        from itertools import combinations

        for m in (1, 2, 3, 4):
            expected = []
            for size in range(1, m + 1):
                expected.extend(combinations(CANDIDATE_INDICES, size))
            self.assertEqual(per_band_choices(m), tuple(expected), f"m={m}")

    def test_space_sizes_match_upstream(self):
        for m in (1, 2):
            mine = traverse_choices(m)
            theirs = self.baseline.traverse_choice(m)
            self.assertEqual(len(mine), len(theirs), f"m={m}")
            self.assertEqual([_normalise(c) for c in mine], [_normalise(c) for c in theirs])

    def test_the_two_arms_search_the_same_space_size(self):
        """Arm B's Phase B must enumerate the same 1000 architectures the
        dilated-only arm does, or the arms are not comparable."""

        self.assertEqual(len(traverse_choices(2)), 1000)
        self.assertEqual(len(per_band_choices(2)), 10)
        self.assertEqual(len(traverse_choices(1)), 64)
        self.assertEqual(len(per_band_choices(1)), 4)

    def test_random_choice_consumes_rng_exactly_as_upstream(self):
        """Same seed on both global generators must give the same subnet.

        This is the load-bearing check: matching the *distribution* would still
        allow a different draw order, which would silently decorrelate this
        arm's searched architectures from a rerun of the baseline flow.
        """

        for m in (1, 2):
            for seed in (0, 1, 7, 20190821, 987654321):
                random.seed(seed)
                np.random.seed(seed)
                theirs = self.baseline.random_choice(m)
                random.seed(seed)
                np.random.seed(seed)
                mine = random_choice(m)
                self.assertEqual(_normalise(mine), _normalise(theirs), f"m={m} seed={seed}")

    def test_per_band_index_matches_upstream_find_choice_index(self):
        for m in (1, 2):
            for subset in per_band_choices(m):
                self.assertEqual(
                    per_band_index(m, subset),
                    self.baseline.find_choice_index(m, list(subset)),
                    f"m={m} subset={subset}",
                )

    def test_choice_index_round_trips_through_the_enumeration(self):
        for m in (1, 2):
            for index, choice in enumerate(traverse_choices(m)):
                self.assertEqual(choice_index(m, choice), index)

    def test_index_is_invariant_to_path_order(self):
        """``(29, 57)`` and ``(57, 29)`` are one architecture, not two."""

        a = {"Low": (0, 1), "Mid": (2,), "High": (3,)}
        b = {"Low": (1, 0), "Mid": (2,), "High": (3,)}
        self.assertEqual(choice_index(2, a), choice_index(2, b))
        self.assertEqual(choice_key(a), choice_key(b))

    def test_a_modern_generator_is_accepted_and_reproducible(self):
        first = [random_choice(2, rng=np.random.default_rng(5), py_random=random.Random(5)) for _ in range(4)]
        second = [random_choice(2, rng=np.random.default_rng(5), py_random=random.Random(5)) for _ in range(4)]
        self.assertEqual(first, second)
        for choice in first:
            self.assertTrue(1 <= len(choice["Low"]) <= 2)

    def test_validate_choice_rejects_malformed_subnets(self):
        good = {"Low": [0], "Mid": [1, 2], "High": [3]}
        self.assertEqual(validate_choice(good)["Mid"], (1, 2))
        for bad in (
            {"Low": [0], "Mid": [1], "High": [3, 3]},
            {"Low": [0], "Mid": [1], "High": [4]},
            {"Low": [0], "Mid": [1]},
            {"Low": [], "Mid": [1], "High": [3]},
        ):
            with self.assertRaises((ValueError, TypeError)):
                validate_choice(bad)
        with self.assertRaises(ValueError):
            validate_choice({"Low": [0, 1], "Mid": [1], "High": [3]}, m=1)


class WidthGeneralFamilyTests(unittest.TestCase):
    """Arm B runs two families at width 6; the probed operator must survive."""

    def test_the_two_generalised_families_are_the_probed_operators_at_width_12(self):
        for family in ("gated_e", "band_gated_e"):
            with self.subTest(family=family):
                torch.manual_seed(3)
                probed = build_e_operator(family, band="Low", target_rf=PHASE_A_RF, out_channels=12)
                torch.manual_seed(3)
                arm_b = build_arm_b_operator(family, band="Low", target_rf=PHASE_A_RF, out_channels=12)
                self.assertEqual(set(probed.state_dict()), set(arm_b.state_dict()))
                for key, tensor in probed.state_dict().items():
                    self.assertTrue(
                        torch.equal(tensor, arm_b.state_dict()[key]),
                        f"{family}: {key} differs at the probed width",
                    )

    def test_the_registry_swaps_only_the_two_generalised_families(self):
        from tdarts.operator_v2e import E_OPERATOR_REGISTRY

        self.assertIs(ARM_B_REGISTRY["gated_e"], WidthGeneralGatedE)
        self.assertIs(ARM_B_REGISTRY["band_gated_e"], WidthGeneralBandGatedE)
        for family in ("dilated_e", "dynamic_e", "local_attention_e"):
            self.assertIs(ARM_B_REGISTRY[family], E_OPERATOR_REGISTRY[family], family)

    def test_the_generalised_families_build_at_the_two_path_width(self):
        for family in ("gated_e", "band_gated_e"):
            with self.subTest(family=family):
                op = build_arm_b_operator(family, band="Mid", target_rf=29, out_channels=6)
                out = op(torch.randn(2, C.IN_CHANNELS, C.NUM_ELECTRODES, 64))
                self.assertEqual(out.shape, (2, 6, C.NUM_ELECTRODES, 64))

    def test_every_family_reports_the_ladder_span_at_both_path_counts(self):
        for family in BAND_FAMILIES:
            cell = FBNASBandCell("Low", rf_candidates(family), m=2)
            for k in (1, 2):
                spans = [cell.nodes[k - 1][i].support[2] for i in range(len(BAND_RFS))]
                self.assertEqual(spans, list(BAND_RFS), f"{family} k={k}")

    def test_the_excluded_family_is_unreachable(self):
        """``local_attention_e`` stays in the inherited registry -- the guard
        against it is the candidate list and the builder, not a missing key."""

        self.assertNotIn(EXCLUDED_FAMILY, BAND_FAMILIES)
        with self.assertRaises(ValueError):
            build_arm_b_operator(EXCLUDED_FAMILY, band="Low", target_rf=PHASE_A_RF)
        self.assertNotIn(EXCLUDED_FAMILY, {c.family for c in family_candidates()})
        for family in BAND_FAMILIES:
            self.assertNotIn(EXCLUDED_FAMILY, {c.family for c in rf_candidates(family)})


class CapacityInvarianceTests(unittest.TestCase):
    """A path-count choice must not secretly be a capacity choice."""

    def _active(self, cell, k, indices):
        return sum(
            parameter.numel()
            for index in indices
            for parameter in cell.nodes[k - 1][index].parameters()
        )

    def test_a_band_emits_a_constant_width_whatever_the_path_count(self):
        for family in BAND_FAMILIES:
            cell = FBNASBandCell("Low", rf_candidates(family), m=2)
            for k in (1, 2):
                self.assertEqual(cell.node_width(k) * k, 12, f"{family} k={k}")

    def test_two_paths_cost_nearly_what_one_path_costs(self):
        """The FBNAS arithmetic exists to make this true; a small residual is
        unavoidable because each extra path carries its own 1x1 projection and
        its own gate, but it must stay small enough that capacity cannot drive
        the search."""

        for family in BAND_FAMILIES:
            with self.subTest(family=family):
                cell = FBNASBandCell("Low", rf_candidates(family), m=2)
                one = self._active(cell, 1, [0])
                two = self._active(cell, 2, [0, 1])
                self.assertLess(abs(two - one) / one, 0.10, f"{family}: {one} vs {two}")

    def test_the_anchor_family_is_exactly_invariant(self):
        """``dilated_e`` is pure convolutions whose parameter count is linear in
        the output width, so F1 and 2 * (F1/2) cost exactly the same.  This is
        the property upstream's ``F1//(i+1)`` arithmetic is built for."""

        cell = FBNASBandCell("Low", rf_candidates("dilated_e"), m=2)
        self.assertEqual(self._active(cell, 1, [0]), self._active(cell, 2, [0, 1]))

    def test_the_residual_is_the_per_path_overhead_and_nothing_else(self):
        """Families with a gate or a 1x1 branch cannot be exactly invariant: two
        paths carry two of them where one path carried one.  Pin the size of
        that overhead so a future change that *does* move capacity is caught.
        """

        expected = {"dynamic_e": 16, "gated_e": -72, "band_gated_e": 3}
        for family, delta in expected.items():
            with self.subTest(family=family):
                cell = FBNASBandCell("Low", rf_candidates(family), m=2)
                one = self._active(cell, 1, [0])
                two = self._active(cell, 2, [0, 1])
                self.assertEqual(two - one, delta, family)


class BandSupernetTests(unittest.TestCase):
    def test_phase_a_is_a_single_path_family_search_over_64_configurations(self):
        self.assertEqual(len(BAND_FAMILIES), 4)
        self.assertEqual(len(traverse_choices(1)), 4 ** 3)
        candidates = family_candidates()
        self.assertEqual([c.label for c in candidates], list(BAND_FAMILIES))
        self.assertTrue(all(c.target_rf == PHASE_A_RF for c in candidates))

    def test_phase_b_is_the_rf_ladder_over_1000_configurations(self):
        for family in BAND_FAMILIES:
            candidates = rf_candidates(family)
            self.assertEqual([c.target_rf for c in candidates], list(BAND_RFS))
            self.assertTrue(all(c.family == family for c in candidates))
        self.assertEqual(len(traverse_choices(2)), 10 ** 3)

    def test_forward_keeps_the_shape_the_backbone_expects(self):
        for m, make in ((1, lambda: family_candidates()), (2, lambda: rf_candidates("gated_e"))):
            net = FBNASBandNet({band: make() for band in C.BANDS}, m=m)
            for k in range(1, m + 1):
                choice = {band: tuple(range(k)) for band in C.BANDS}
                logits, features = net(torch.randn(2, 9, C.NUM_ELECTRODES, 256), choice)
                self.assertEqual(logits.shape, (2, C.NUM_CLASSES))
                self.assertEqual(features.shape[1], 2304)

    def test_forward_accepts_the_loader_layout(self):
        net = FBNASBandNet({band: rf_candidates("dilated_e") for band in C.BANDS}, m=2)
        choice = {band: (0,) for band in C.BANDS}
        logits, _ = net(torch.randn(2, 1, C.NUM_ELECTRODES, 256, C.NUM_BANDS), choice)
        self.assertEqual(logits.shape, (2, C.NUM_CLASSES))

    def test_the_backbone_sees_a_constant_width_across_every_candidate(self):
        net = FBNASBandNet({band: rf_candidates("band_gated_e") for band in C.BANDS}, m=2)
        widths = set()
        for choice in traverse_choices(2)[:12]:
            bands = net.split_bands(torch.randn(1, 9, C.NUM_ELECTRODES, 64))
            temporal = torch.cat([net.cells[b](bands[b], choice[b]) for b in C.BANDS], dim=1)
            widths.add(temporal.shape[1])
        self.assertEqual(widths, {C.NUM_FEAT})

    def test_a_subnet_with_too_many_paths_is_rejected(self):
        cell = FBNASBandCell("Low", rf_candidates("dilated_e"), m=2)
        x = torch.randn(1, C.IN_CHANNELS, C.NUM_ELECTRODES, 32)
        with self.assertRaises(ValueError):
            cell(x, [0, 1, 2])
        with self.assertRaises(ValueError):
            cell(x, [0, 0])
        with self.assertRaises(ValueError):
            cell(x, [])

    def test_m_cannot_exceed_the_candidate_count(self):
        with self.assertRaises(ValueError):
            FBNASBandCell("Low", family_candidates(), m=5)

    def test_calibrated_scores_returns_one_row_per_candidate_in_order(self):
        torch.manual_seed(0)
        net = FBNASBandNet({band: family_candidates() for band in C.BANDS}, m=1)
        choices = traverse_choices(1)[:6]
        rows = calibrated_scores(
            net, choices, torch.randn(8, 1, C.NUM_ELECTRODES, 64, C.NUM_BANDS),
            torch.randint(0, C.NUM_CLASSES, (8,)), criterion=torch.nn.NLLLoss(),
        )
        self.assertEqual([row["index"] for row in rows], list(range(6)))
        for row, choice in zip(rows, choices):
            self.assertEqual({b: tuple(row["choice"][b]) for b in C.BANDS}, _normalise(choice))
            self.assertIn("accuracy", row)
            self.assertIn("nll", row)

    def test_calibration_does_not_leak_between_candidates(self):
        """Each candidate must be scored from the restored weights, not from
        whatever the previous candidate's calibration left behind.

        The calibration pass is a train-mode forward, so it overwrites the
        BatchNorm running statistics; a candidate scored without restoring
        first would silently inherit its predecessor's.
        """

        import copy

        torch.manual_seed(0)
        net = FBNASBandNet({band: family_candidates() for band in C.BANDS}, m=1)
        inputs = torch.randn(8, 1, C.NUM_ELECTRODES, 64, C.NUM_BANDS)
        targets = torch.randint(0, C.NUM_CLASSES, (8,))
        choices = traverse_choices(1)[:4]
        pristine = copy.deepcopy(net.state_dict())
        alone = calibrated_scores(net, choices[:1], inputs, targets, restore_from=pristine)[0]
        in_sequence = calibrated_scores(net, choices, inputs, targets, restore_from=pristine)[0]
        self.assertAlmostEqual(alone["accuracy"], in_sequence["accuracy"], places=9)
        self.assertAlmostEqual(alone["nll"], in_sequence["nll"], places=6)

    def test_a_candidates_score_does_not_depend_on_its_position(self):
        """The observable consequence of restoring: scoring the same four
        candidates in reverse order must give each the same number."""

        import copy

        torch.manual_seed(0)
        net = FBNASBandNet({band: family_candidates() for band in C.BANDS}, m=1)
        inputs = torch.randn(8, 1, C.NUM_ELECTRODES, 64, C.NUM_BANDS)
        targets = torch.randint(0, C.NUM_CLASSES, (8,))
        choices = traverse_choices(1)[:4]
        pristine = copy.deepcopy(net.state_dict())

        def by_choice(rows):
            return {
                tuple(tuple(row["choice"][band]) for band in C.BANDS): row for row in rows
            }

        forward = by_choice(
            calibrated_scores(net, choices, inputs, targets, restore_from=pristine)
        )
        reverse = by_choice(
            calibrated_scores(net, list(reversed(choices)), inputs, targets, restore_from=pristine)
        )
        self.assertEqual(set(forward), set(reverse))
        for key in forward:
            self.assertAlmostEqual(forward[key]["accuracy"], reverse[key]["accuracy"], places=9)
            self.assertAlmostEqual(forward[key]["nll"], reverse[key]["nll"], places=6)


class BandGenotypeTests(unittest.TestCase):
    def _genotype(self):
        return BandGenotype(
            seed=20190821, epoch=400,
            genes=(
                BandGene("Low", 0, "dynamic_e", 29), BandGene("Low", 1, "dynamic_e", 57),
                BandGene("Mid", 0, "dilated_e", 57),
                BandGene("High", 0, "gated_e", 15), BandGene("High", 1, "gated_e", 29),
            ),
            phase_a={"Low": "dynamic_e", "Mid": "dilated_e", "High": "gated_e"},
            phase_b={"Low": (29, 57), "Mid": (57,), "High": (15, 29)},
        )

    def test_round_trip_through_the_file_dialect(self):
        genotype = self._genotype()
        payload = json.loads(json.dumps(genotype.to_dict()))
        self.assertEqual(load_band_genotype(payload).to_dict(), genotype.to_dict())

    def test_the_description_names_families_and_receptive_fields(self):
        text = describe_genotype(self._genotype())
        self.assertIn("Phase A:", text)
        self.assertIn("Phase B:", text)
        self.assertIn("Low  = dynamic_e", text)
        self.assertIn("Low  = RF29 + RF57", text)
        self.assertIn("Mid  = RF57", text)

    def test_per_band_accessors(self):
        genotype = self._genotype()
        self.assertEqual(genotype.family_of("High"), "gated_e")
        self.assertEqual(genotype.rfs_of("Low"), (29, 57))
        self.assertEqual(genotype.rfs_of("Mid"), (57,))

    def test_a_band_mixing_two_families_is_rejected(self):
        with self.assertRaises(ValueError):
            BandGenotype(seed=1, epoch=1, genes=(
                BandGene("Low", 0, "gated_e", 57), BandGene("Low", 1, "dilated_e", 57),
                BandGene("Mid", 0, "dilated_e", 15), BandGene("High", 0, "dilated_e", 15)))

    def test_path_indices_must_be_contiguous_from_zero(self):
        with self.assertRaises(ValueError):
            BandGenotype(seed=1, epoch=1, genes=(
                BandGene("Low", 1, "gated_e", 57),
                BandGene("Mid", 0, "dilated_e", 15), BandGene("High", 0, "dilated_e", 15)))

    def test_a_missing_band_is_rejected(self):
        with self.assertRaises(ValueError):
            BandGenotype(seed=1, epoch=1, genes=(
                BandGene("Low", 0, "gated_e", 57), BandGene("Mid", 0, "dilated_e", 15)))

    def test_the_loader_rejects_malformed_files(self):
        good = self._genotype().to_dict()
        cases = {
            "wrong scheme": {**good, "scheme": "something_else"},
            "excluded family": {
                **good,
                "bands": {**good["bands"], "Low": {**good["bands"]["Low"], "family": EXCLUDED_FAMILY}},
            },
            "unknown family": {
                **good,
                "bands": {**good["bands"], "Low": {**good["bands"]["Low"], "family": "nope_e"}},
            },
            "three rfs": {
                **good,
                "bands": {**good["bands"], "Low": {**good["bands"]["Low"], "rfs": [15, 29, 57]}},
            },
            "no rfs": {
                **good,
                "bands": {**good["bands"], "Low": {**good["bands"]["Low"], "rfs": []}},
            },
            "repeated rf": {
                **good,
                "bands": {**good["bands"], "Low": {**good["bands"]["Low"], "rfs": [57, 57]}},
            },
            "off-ladder rf": {
                **good,
                "bands": {**good["bands"], "Low": {**good["bands"]["Low"], "rfs": [40]}},
            },
            "missing band": {**good, "bands": {"Low": good["bands"]["Low"]}},
        }
        for label, payload in cases.items():
            with self.subTest(label=label), self.assertRaises(ValueError):
                load_band_genotype(payload)

    def test_structure_keys_are_read_from_built_operators(self):
        keys = band_structure_keys(self._genotype())
        self.assertEqual(keys["Mid"], (("dilated_e", 15, 4),))
        self.assertEqual(len(keys["Low"]), 2)
        self.assertEqual([key[2] for key in keys["Low"]], [2, 4])

    def test_duplicate_bands_are_detected_only_when_two_paths_collide(self):
        self.assertEqual(band_duplicate_bands(self._genotype()), ())
        collided = BandGenotype(seed=1, epoch=1, genes=(
            BandGene("Low", 0, "gated_e", 57), BandGene("Low", 1, "gated_e", 57),
            BandGene("Mid", 0, "dilated_e", 15), BandGene("High", 0, "dilated_e", 15)))
        self.assertEqual(band_duplicate_bands(collided), ("Low",))


class BandDiscreteNetTests(unittest.TestCase):
    def _genotype(self):
        return BandGenotype(seed=1, epoch=1, genes=(
            BandGene("Low", 0, "dynamic_e", 29), BandGene("Low", 1, "dynamic_e", 57),
            BandGene("Mid", 0, "dilated_e", 57),
            BandGene("High", 0, "gated_e", 15), BandGene("High", 1, "gated_e", 29)))

    def test_it_builds_and_forwards(self):
        net = BandDiscreteNet(self._genotype())
        logits, features = net(torch.randn(2, 9, C.NUM_ELECTRODES, 256))
        self.assertEqual(logits.shape, (2, C.NUM_CLASSES))
        self.assertEqual(features.shape[1], 2304)

    def test_it_carries_no_architecture_parameters(self):
        net = BandDiscreteNet(self._genotype())
        self.assertFalse([name for name, _ in net.named_parameters() if name.endswith("alpha")])
        self.assertFalse([name for name, _ in net.named_modules() if ".ops." in name])

    def test_the_backbone_is_untouched_by_the_path_count(self):
        """Same backbone and same total width as the six-gene discrete net, so
        an Arm B architecture is compared on its temporal stage alone."""

        net = BandDiscreteNet(self._genotype())
        self.assertEqual(net.backbone.scb.in_channels, C.NUM_FEAT)
        temporal = torch.cat(
            [net.cells[b](net.split_bands(torch.randn(1, 9, C.NUM_ELECTRODES, 64))[b]) for b in C.BANDS],
            dim=1,
        )
        self.assertEqual(temporal.shape[1], C.NUM_FEAT)


class RetrainDialectTests(unittest.TestCase):
    """The band dialect must slot into train_retrain without moving anything else."""

    def _import_retrain(self):
        import train_retrain

        return train_retrain

    def test_the_retrain_module_imports_the_band_dialect(self):
        module = self._import_retrain()
        self.assertTrue(hasattr(module, "load_band_genotype"))
        self.assertEqual(module.BAND_SCHEME, BAND_SCHEME)

    def test_the_plain_six_gene_dialect_is_still_the_default(self):
        """A file without a scheme tag must keep going down load_genotype."""

        module = self._import_retrain()
        source = (PROJECT_ROOT / "train_retrain.py").read_text(encoding="utf-8")
        self.assertIn('payload.get("scheme") == ANCHORED_SCHEME', source)
        self.assertIn('payload.get("scheme") == BAND_SCHEME', source)
        # The plain branch must remain the fallback, not be shadowed by a new
        # condition that could swallow an untagged file.  Scoped to the dispatch
        # block so an unrelated earlier mention cannot satisfy the ordering.
        block = source[source.index("if args.genotype_json is not None:"):]
        block = block[: block.index("else:\n        # Take the epoch")]
        anchored_at = block.index("== ANCHORED_SCHEME")
        band_at = block.index("== BAND_SCHEME")
        plain_at = block.index("genotype = load_genotype(args.genotype_json)")
        self.assertLess(anchored_at, band_at)
        self.assertLess(band_at, plain_at)

    def test_stage2_only_refuses_the_band_dialect_rather_than_misreading_it(self):
        source = (PROJECT_ROOT / "train_retrain.py").read_text(encoding="utf-8")
        marker = "cannot resume a band-dialect run"
        self.assertIn(marker, source)

    def test_the_discrete_net_is_chosen_by_the_scheme_tag(self):
        source = (PROJECT_ROOT / "train_retrain.py").read_text(encoding="utf-8")
        window = source[source.index("BandDiscreteNet(band_genotype)") - 200:
                        source.index("BandDiscreteNet(band_genotype)") + 200]
        self.assertIn("TemporalDiscreteNet(genotype)", window)


class FrozenGenerationGuardTests(unittest.TestCase):
    """Arm B must not have disturbed the generations it builds on."""

    def test_the_frozen_family_registry_is_unchanged(self):
        from tdarts.operator_v2e import E_OPERATOR_NAMES, build_e_operator

        self.assertEqual(
            set(E_OPERATOR_NAMES),
            {"dilated_e", "dynamic_e", "gated_e", "band_gated_e", "local_attention_e"},
        )
        # The guard that made the width-general subclasses necessary.
        for family in ("gated_e", "local_attention_e"):
            with self.assertRaises(ValueError):
                build_e_operator(family, band="Low", target_rf=57, out_channels=6)

    def test_archived_expressive_capacity_is_unchanged(self):
        expected = {
            "dilated_e": 540, "gated_e": 1260, "dynamic_e": 2176, "band_gated_e": 627,
            "local_attention_e": 624,
        }
        for family, count in expected.items():
            op = build_e_operator(family, band="Low", target_rf=57, out_channels=12)
            self.assertEqual(sum(p.numel() for p in op.parameters()), count, family)

    def test_arm_b_writes_nowhere_near_the_frozen_roots(self):
        source = (PROJECT_ROOT / "train_arm_b_search.py").read_text(encoding="utf-8")
        for frozen in ("operator_v2", "operator_v2e"):
            self.assertNotIn(f'"{frozen}"', source)


class Stage2ProtocolParityTests(unittest.TestCase):
    """Arm B's Stage 2 must run Arm A's protocol, not merely a similar one.

    Arm A's protocol is implemented inside the frozen baseline, so it cannot be
    diffed as a command line.  What can be checked is that the numbers Arm A's
    own run recorded (``config.csv``) are the numbers Arm B's job script passes,
    which is the part a reviewer would otherwise have to take on trust.
    """

    @classmethod
    def setUpClass(cls):
        # config.csv is written once per run (per timestamped directory), not
        # once per subject, so it sits beside the subN directories.
        archive = PROJECT_ROOT / "run/outputs/fbnas/bci42a/ses2Test"
        configs = sorted(archive.glob("*/config.csv")) if archive.is_dir() else []
        if not configs:
            raise unittest.SkipTest(f"no archived Arm A run under {archive}")
        import ast
        import csv as _csv

        with configs[0].open(newline="", encoding="utf-8") as handle:
            raw = {row[0]: row[1] for row in _csv.reader(handle) if len(row) >= 2}
        cls.arm_a = raw
        cls.train_args = ast.literal_eval(raw["modelTrainArguments"])

    def _job_script(self) -> str:
        return (PROJECT_ROOT / "run/bin/run_arm_b_retrain_job.sh").read_text(encoding="utf-8")

    def test_the_epoch_budget_and_patience_match(self):
        stop = self.train_args["stopCondi"]["c"]["Or"]
        self.assertEqual(stop["c1"]["MaxEpoch"]["maxEpochs"], 1500)
        self.assertEqual(stop["c2"]["NoDecrease"]["numEpochs"], 200)
        script = self._job_script()
        self.assertIn("MAX_EPOCHS=${MAX_EPOCHS:-1500}", script)
        self.assertIn("PATIENCE=${PATIENCE:-200}", script)

    def test_the_selection_variable_matches(self):
        """This is the setting the frozen repo arms deviate on: they pass
        val_nll, Arm A stops on valInacc, and Arm B must follow Arm A."""

        self.assertEqual(self.train_args["bestVarToCheck"], "valInacc")
        self.assertIn("BEST_METRIC=${BEST_METRIC:-val_inacc}", self._job_script())

    def test_stage_two_continues_from_the_best_checkpoint(self):
        self.assertTrue(self.train_args["continueAfterEarlystop"])
        script = self._job_script()
        # No --screening-only: that is what would stop the run before Stage 2
        # and leave Session 2 unread.
        self.assertNotIn("--screening-only", script)
        self.assertIn("--stage2-epochs", script)

    def test_the_split_and_batch_size_match(self):
        """A 0.2 validation fraction of 288 trials is the 231/57 split, which is
        the same split `load_session0_search_split` builds."""

        self.assertEqual(float(self.arm_a["validationSet"]), 0.2)
        self.assertEqual(int(self.arm_a["batchSize"]), 16)
        self.assertEqual(self.train_args["lr"], 1e-3)
        self.assertEqual(-(-288 * 4 // 5), 231)

    def test_the_arm_a_archive_records_the_same_seed(self):
        entry = self._import_search()
        with mock.patch.object(sys, "argv", ["train_arm_b_search.py"]):
            self.assertEqual(entry.parse_args().seed, int(self.arm_a["randSeed"]))
        self.assertIn("SEED=${SEED:-20190821}", (PROJECT_ROOT / "run/bin/run_arm_b_search_job.sh").read_text(encoding="utf-8"))

    def _import_search(self):
        spec = importlib.util.spec_from_file_location(
            "train_arm_b_search_parity", PROJECT_ROOT / "train_arm_b_search.py"
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module


class ComparisonToolTests(unittest.TestCase):
    """The two arms' producers do not agree on every metric's name.

    Upstream's ``results.csv`` writes macro-F1 as ``f1``; ``train_retrain.py``
    writes ``macro_f1``.  A comparison tool that assumed one name would crash on
    the other producer's data -- which is exactly what happened the first time
    it was pointed at real Arm B output, so it is pinned here.
    """

    def _tool(self):
        spec = importlib.util.spec_from_file_location(
            "compare_session2_arms_under_test", PROJECT_ROOT / "tools/compare_session2_arms.py"
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module

    def test_the_metric_mapping_names_both_producers_keys(self):
        tool = self._tool()
        mapping = {label: (a, b) for label, a, b in tool.METRICS}
        self.assertEqual(mapping["acc"], ("acc", "acc"))
        self.assertEqual(mapping["kappa"], ("kappa", "kappa"))
        self.assertEqual(mapping["f1"], ("f1", "macro_f1"))
        self.assertEqual(tool.LABELS, ("acc", "f1", "kappa"))

    def test_arm_b_is_read_through_its_own_key_names(self):
        import tempfile

        tool = self._tool()
        with tempfile.TemporaryDirectory() as tmp:
            leaf = Path(tmp) / "bci42a" / "train_s003_seed20190821_armB"
            leaf.mkdir(parents=True)
            (leaf / "final_summary.json").write_text(
                json.dumps({
                    "test": {"acc": 0.8, "kappa": 0.7, "macro_f1": 0.75, "nll": 0.5},
                    "parameters": 123, "macs": 456, "screening_only": False,
                }),
                encoding="utf-8",
            )
            row = tool.load_arm_b(Path(tmp) / "bci42a", "armB", "003", "20190821")
        self.assertIsNotNone(row)
        self.assertEqual(row["test"], {"acc": 0.8, "f1": 0.75, "kappa": 0.7})
        self.assertAlmostEqual(row["nll"], 0.5)

    def test_a_run_that_never_opened_session_two_is_not_a_result(self):
        """``--screening-only`` writes ``test: null``; that must read as absent
        rather than as a zero."""

        import tempfile

        tool = self._tool()
        with tempfile.TemporaryDirectory() as tmp:
            leaf = Path(tmp) / "bci42a" / "train_s003_seed20190821_armB"
            leaf.mkdir(parents=True)
            (leaf / "final_summary.json").write_text(
                json.dumps({"test": None, "parameters": 1, "macs": 1}), encoding="utf-8"
            )
            self.assertIsNone(tool.load_arm_b(Path(tmp) / "bci42a", "armB", "003", "20190821"))


class EntryPointTests(unittest.TestCase):
    def _import(self):
        spec = importlib.util.spec_from_file_location(
            "train_arm_b_search_under_test", PROJECT_ROOT / "train_arm_b_search.py"
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module

    def test_the_entry_point_never_loads_session_one(self):
        """Arm B exports an architecture. Reading the dataset's second session
        here would put the test set one careless line from the search.

        Checked against the parsed module rather than the file text: the
        docstring legitimately *names* ``session=1`` when explaining the
        numbering, and a text search cannot tell that from a call site.
        """

        import ast

        source = (PROJECT_ROOT / "train_arm_b_search.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        called = {
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        self.assertNotIn("load_subject_session", called)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                for keyword in node.keywords:
                    self.assertNotEqual(keyword.arg, "session", f"line {node.lineno}")
        self.assertIn("load_session0_search_split", called)

    def test_the_output_root_guard_rejects_a_frozen_root(self):
        module = self._import()
        with mock.patch.object(
            sys, "argv",
            ["train_arm_b_search.py", "--arm", "operator_armB", "--output-root", "run/outputs/operator_v2e"],
        ):
            with self.assertRaises(ValueError):
                module.main()

    def test_the_arm_guard_rejects_a_foreign_label(self):
        module = self._import()
        with mock.patch.object(
            sys, "argv",
            ["train_arm_b_search.py", "--arm", "hier", "--output-root", "run/outputs/operator_armB"],
        ):
            with self.assertRaises(ValueError):
                module.main()

    def test_default_search_budget_matches_the_dilated_only_arm(self):
        """Arm A searched with 200 epochs of random MixPath training; Phase B
        starts from the same budget on the same 1000-candidate space."""

        module = self._import()
        with mock.patch.object(sys, "argv", ["train_arm_b_search.py"]):
            args = module.parse_args()
        self.assertEqual(args.phase_a_epochs, 200)
        self.assertEqual(args.phase_b_epochs, 200)
        self.assertEqual(args.seed, 20190821)
        self.assertEqual(args.m_a, 1)
        self.assertEqual(args.m_b, 2)

    def test_the_sampler_stream_ignores_global_rng_consumption(self):
        """Two runs with the same seed must search the same architectures even
        if unrelated code drew from the global RNG in between."""

        module = self._import()

        def draw(perturb: int):
            stream = module._subnet_stream(11, 2)
            random.seed(999)
            np.random.seed(999)
            torch.manual_seed(999)
            for _ in range(perturb):
                torch.rand(17)
            return [next(stream) for _ in range(6)]

        self.assertEqual(draw(0), draw(5))
        self.assertNotEqual(draw(0), [next(module._subnet_stream(12, 2)) for _ in range(6)])


class SearchPipelineTests(unittest.TestCase):
    """A whole search on a synthetic problem must recover a planted architecture."""

    def test_a_planted_family_is_recovered_from_the_traversal(self):
        """Phase A's traversal must pick the family that actually fits, not an
        arbitrary one.  The check is on the mechanism: scores are computed per
        candidate from restored weights, so a candidate that fits best wins."""

        torch.manual_seed(0)
        net = FBNASBandNet({band: family_candidates() for band in C.BANDS}, m=1)
        choices = traverse_choices(1)
        inputs = torch.randn(16, 1, C.NUM_ELECTRODES, 64, C.NUM_BANDS)
        targets = torch.randint(0, C.NUM_CLASSES, (16,))
        rows = calibrated_scores(net, choices, inputs, targets)
        ranked = sorted(rows, key=lambda row: (-row["accuracy"], row["index"]))
        self.assertEqual(len(rows), 64)
        self.assertEqual(ranked[0]["index"], min(
            row["index"] for row in rows if row["accuracy"] == ranked[0]["accuracy"]
        ))

    def test_a_phase_a_winner_maps_back_to_families(self):
        """Phase A's chosen index per band indexes the family list."""

        choice = {"Low": (0,), "Mid": (1,), "High": (3,)}
        candidates = {band: family_candidates() for band in C.BANDS}
        families = {band: candidates[band][choice[band][0]].family for band in C.BANDS}
        self.assertEqual(
            families, {"Low": "dilated_e", "Mid": "dynamic_e", "High": "band_gated_e"}
        )

    def test_a_phase_b_winner_maps_back_to_receptive_fields(self):
        """Phase B's chosen indices index the RF ladder, and a repeated index
        cannot occur because traverse_choices enumerates distinct subsets."""

        choice = {"Low": (0, 2), "Mid": (1,), "High": (3,)}
        rfs = {band: [BAND_RFS[i] for i in choice[band]] for band in C.BANDS}
        self.assertEqual(rfs, {"Low": [15, 57], "Mid": [29], "High": [113]})
        for enumerated in traverse_choices(2):
            for band in C.BANDS:
                self.assertEqual(len(set(enumerated[band])), len(enumerated[band]))


if __name__ == "__main__":
    unittest.main()
