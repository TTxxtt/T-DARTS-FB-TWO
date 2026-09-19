"""Band-specific mechanism probe: vocabulary, equivalence, guards, and the rule.

Three things this file is here to hold down, in order of how expensive the
failure would be:

1. **The anchor equivalence.**  The probe does not retrain the all-dilated
   baseline; it reuses the Expressive ``dilated_e`` arm and pairs against it.
   That reuse is only valid if the band entry point, given an all-anchor
   configuration, *is* the Expressive entry point.  The test builds both
   networks under one seed and asserts their ``state_dict()`` tensors are
   bit-identical, and that the same input produces bit-identical logits.  It
   also runs both entry points end to end for a few epochs and compares the
   metrics, when the data is reachable -- the network check pins the initial
   weights, the end-to-end check pins everything after them.

2. **The configuration vocabulary.**  A slug is the only thing tying a
   directory on disk back to a configuration, so the round trip and the
   refusals are pinned: exactly one band may vary, and ``local_attention_e``
   stays out for a stated reason rather than by omission.

3. **The frozen trees.**  ``operator_v2`` and the Expressive ``operator_v2e``
   are both archived.  Their pinned numbers are re-asserted here, and the entry
   point's two tree guards are exercised.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tdarts import config as C
from tdarts.operator_v2e import E_OPERATOR_NAMES, build_e_operator
from tdarts.operator_v2e_band import (
    ANCHOR_FAMILY,
    ANCHOR_SLUG,
    BAND_FAMILIES,
    EXCLUDED_FAMILIES,
    STAGE_PREFIX,
    VARYING_FAMILIES,
    config_of,
    parse_slug,
    probe_configs,
    slug_for,
    validate_band_families,
)
from tdarts.operator_v2e_band_network import OperatorV2EBandNet
from tdarts.operator_v2e_network import OperatorV2ENet

REPO_ROOT = Path(__file__).resolve().parent.parent
#: The frozen Expressive tree the anchor is reused from.
EXPREESSIVE_ROOT = REPO_ROOT / "run" / "outputs" / "operator_v2e"
#: The frozen Matched tree, which neither this stage nor the Expressive one may touch.
FROZEN_MATCHED_ROOT = REPO_ROOT / "run" / "outputs" / "operator_v2"

sys.path.insert(0, str(REPO_ROOT))
from tools.analyze_operator_v2e_band import (  # noqa: E402
    ANCHOR_FAMILY as ANALYZER_ANCHOR,
    CANDIDATES,
    assess_band_probe,
    band_effect_table,
    build_pairs,
    cell_table,
    winner_table,
)

DATA_ROOT = REPO_ROOT.parent / "FBNAS-master-main" / "data" / "bci42a" / "multiviewPython"


def _load_entry_point(filename: str, module_name: str):
    """Load an entry point by path; the repo has no package to import it from."""

    spec = importlib.util.spec_from_file_location(module_name, REPO_ROOT / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _all_configs() -> list[dict[str, str]]:
    configs = [config_of(ANCHOR_FAMILY, ANCHOR_FAMILY, ANCHOR_FAMILY)]
    for band, family in probe_configs():
        families = config_of(ANCHOR_FAMILY, ANCHOR_FAMILY, ANCHOR_FAMILY)
        families[band] = family
        configs.append(families)
    return configs


class BandVocabularyTests(unittest.TestCase):
    def test_the_grid_is_nine_single_band_replacements(self):
        configs = probe_configs()
        self.assertEqual(len(configs), 9)
        self.assertEqual({band for band, _ in configs}, set(C.BANDS))
        self.assertEqual({family for _, family in configs}, set(VARYING_FAMILIES))
        # No duplicates: a repeated (band, family) would silently halve a cell.
        self.assertEqual(len(set(configs)), 9)

    def test_every_grid_configuration_is_a_single_band_change(self):
        for band, family in probe_configs():
            with self.subTest(band=band, family=family):
                families = config_of(ANCHOR_FAMILY, ANCHOR_FAMILY, ANCHOR_FAMILY)
                families[band] = family
                varying_band, varying_family = validate_band_families(
                    families["Low"], families["Mid"], families["High"]
                )
                self.assertEqual(varying_band, band)
                self.assertEqual(varying_family, family)

    def test_slugs_round_trip(self):
        for families in _all_configs():
            slug = slug_for(families["Low"], families["Mid"], families["High"])
            with self.subTest(slug=slug):
                self.assertEqual(parse_slug(slug), families)

    def test_the_baseline_slug_is_named_and_has_no_varying_band(self):
        families = config_of(ANCHOR_FAMILY, ANCHOR_FAMILY, ANCHOR_FAMILY)
        self.assertEqual(slug_for(**{"low": ANCHOR_FAMILY, "mid": ANCHOR_FAMILY, "high": ANCHOR_FAMILY}),
                         ANCHOR_SLUG)
        self.assertEqual(parse_slug(ANCHOR_SLUG), families)
        self.assertEqual(ANALYZER_ANCHOR, ANCHOR_FAMILY)

    def test_two_varying_bands_are_refused(self):
        """The design is a *controlled* single-band replacement; a joint change
        cannot be attributed to either band."""
        with self.assertRaises(ValueError) as caught:
            validate_band_families("dynamic_e", "gated_e", ANCHOR_FAMILY)
        self.assertIn("exactly one band", str(caught.exception))
        with self.assertRaises(ValueError):
            validate_band_families("dynamic_e", "gated_e", "band_gated_e")

    def test_local_attention_is_out_of_scope_and_says_why(self):
        self.assertIn("local_attention_e", EXCLUDED_FAMILIES)
        self.assertNotIn("local_attention_e", BAND_FAMILIES)
        self.assertNotIn("local_attention_e", E_OPERATOR_NAMES[0:0] + BAND_FAMILIES)
        self.assertTrue(EXCLUDED_FAMILIES["local_attention_e"].strip())
        with self.assertRaises(ValueError) as caught:
            validate_band_families("local_attention_e", ANCHOR_FAMILY, ANCHOR_FAMILY)
        # The message has to state the decision, not just refuse the input.
        self.assertIn("out of scope", str(caught.exception))

    def test_unknown_and_malformed_slugs_are_refused(self):
        for slug in ("", "low", "low_unknown_e", "sideways_dynamic_e", "low_dynamic"):
            with self.subTest(slug=slug):
                with self.assertRaises(ValueError):
                    parse_slug(slug)

    def test_the_anchor_family_is_not_a_varying_family(self):
        """Replacing the anchor with the anchor is the baseline, not a probe."""
        self.assertNotIn(ANCHOR_FAMILY, VARYING_FAMILIES)
        self.assertEqual(CANDIDATES[0], ANCHOR_FAMILY)
        self.assertEqual(set(CANDIDATES), {ANCHOR_FAMILY} | set(VARYING_FAMILIES))


class BandNetworkShapeTests(unittest.TestCase):
    def test_every_configuration_keeps_the_tensor_contract(self):
        shape = (2, 9, C.NUM_ELECTRODES, 200)
        x = torch.zeros(shape)
        for families in _all_configs():
            with self.subTest(families=families):
                net = OperatorV2EBandNet(families, target_rf=57)
                net.eval()
                with torch.no_grad():
                    logits, _ = net(x)
                self.assertEqual(tuple(logits.shape), (2, C.NUM_CLASSES))
                net.train()
                with torch.no_grad():
                    temporal = torch.cat(
                        [net.cells[band](net.split_bands(x)[band]) for band in C.BANDS], dim=1
                    )
                self.assertEqual(tuple(temporal.shape), (2, C.NUM_FEAT, C.NUM_ELECTRODES, 200))

    def test_each_cell_maps_three_channels_to_twelve(self):
        for families in _all_configs():
            net = OperatorV2EBandNet(families, target_rf=57)
            for band in C.BANDS:
                with self.subTest(families=families, band=band):
                    cell = net.cells[band]
                    self.assertEqual(cell.op_name, families[band])
                    self.assertEqual(cell.path.band, band)

    def test_the_backbone_is_the_frozen_backbone(self):
        """The band probe may change the cells and nothing else; the backbone
        sees the same [B, 36, E, T] either way."""
        from tdarts.backbone import TemporalBackbone

        net = OperatorV2EBandNet(config_of("dynamic_e", ANCHOR_FAMILY, ANCHOR_FAMILY))
        reference = TemporalBackbone(in_channels=C.NUM_FEAT)
        self.assertEqual(
            [name for name, _ in net.backbone.named_parameters()],
            [name for name, _ in reference.named_parameters()],
        )

    def test_a_configuration_missing_a_band_is_refused(self):
        with self.assertRaises(KeyError):
            OperatorV2EBandNet({"Low": "dynamic_e", "Mid": ANCHOR_FAMILY})


class BandNetworkGeometryTests(unittest.TestCase):
    def test_every_band_reads_exactly_rf57_in_every_configuration(self):
        for families in _all_configs():
            net = OperatorV2EBandNet(families, target_rf=57)
            for band in C.BANDS:
                with self.subTest(families=families, band=band):
                    self.assertEqual(net.cells[band].path.support, (15, 4, 57))
                    self.assertEqual(net.cells[band].describe()["support_span"], 57)

    def test_the_varying_family_does_not_widen_the_window(self):
        """A family that bought its result with a longer span would be a
        different experiment, so this is checked per family, not per grid."""
        for family in VARYING_FAMILIES:
            with self.subTest(family=family):
                op = build_e_operator(family, band="Low", target_rf=57)
                self.assertEqual(op.support, (15, 4, 57))


class AnchorEquivalenceTests(unittest.TestCase):
    """The reuse of the Expressive anchor rests entirely on these.

    If the band entry point were not bit-identical to the Expressive one for the
    all-anchor configuration, the paired differences would carry a second,
    silent difference between two code paths on top of the one being measured.
    """

    ANCHOR_CONFIG = {"Low": ANCHOR_FAMILY, "Mid": ANCHOR_FAMILY, "High": ANCHOR_FAMILY}

    def test_the_two_networks_are_the_same_function_under_one_seed(self):
        for seed in (20250901, 7, 123456):
            with self.subTest(seed=seed):
                torch.manual_seed(seed)
                expressive = OperatorV2ENet(ANCHOR_FAMILY, target_rf=57)
                torch.manual_seed(seed)
                band = OperatorV2EBandNet(self.ANCHOR_CONFIG, target_rf=57)
                left, right = expressive.state_dict(), band.state_dict()
                self.assertEqual(list(left), list(right))
                mismatched = [key for key in left if not torch.equal(left[key], right[key])]
                self.assertEqual(mismatched, [])

    def test_the_two_networks_produce_bit_identical_logits(self):
        torch.manual_seed(20250901)
        expressive = OperatorV2ENet(ANCHOR_FAMILY, target_rf=57).eval()
        torch.manual_seed(20250901)
        band = OperatorV2EBandNet(self.ANCHOR_CONFIG, target_rf=57).eval()
        torch.manual_seed(5)
        x = torch.randn(2, C.NUM_BANDS, C.NUM_ELECTRODES, 200)
        with torch.no_grad():
            left, _ = expressive(x)
            right, _ = band(x)
        self.assertTrue(torch.equal(left, right))

    def test_the_parameter_count_is_the_archived_one(self):
        net = OperatorV2EBandNet(self.ANCHOR_CONFIG, target_rf=57)
        self.assertEqual(net.describe()["parameters"], sum(p.numel() for p in net.parameters()))
        # Same object as the Expressive anchor, so the same total.
        torch.manual_seed(1)
        self.assertEqual(
            net.describe()["parameters"],
            OperatorV2ENet(ANCHOR_FAMILY, target_rf=57).describe()["parameters"],
        )

    def test_the_anchor_network_reports_the_anchor_slug(self):
        net = OperatorV2EBandNet(self.ANCHOR_CONFIG, target_rf=57)
        self.assertEqual(net.op_name, ANCHOR_SLUG)
        self.assertEqual(net.describe()["band_families"], self.ANCHOR_CONFIG)
        for band in C.BANDS:
            self.assertEqual(net.describe()["cells"][C.BANDS.index(band)]["band"], band)

    def test_the_archived_anchor_runs_are_intact_and_closed(self):
        """The nine runs the pairing reads.  They are the reference for every
        number this stage produces, so their presence and their closed Session 1
        are asserted rather than assumed."""
        if not EXPREESSIVE_ROOT.is_dir():
            self.skipTest("the Expressive tree is not on disk")
        expected = {(subject, seed) for subject in ("003", "005", "006")
                    for seed in (20250901, 20250902, 20250903)}
        seen = set()
        for path in EXPREESSIVE_ROOT.glob("**/final_summary.json"):
            summary = json.loads(path.read_text(encoding="utf-8"))
            if summary.get("operator") != ANCHOR_FAMILY:
                continue
            seen.add((str(summary["subject"]), int(summary["seed"])))
            self.assertFalse(summary["session1_opened"])
            self.assertIsNone(summary["test"])
            self.assertTrue(summary["screening_only"])
            self.assertEqual(summary["target_rf"], 57)
        self.assertTrue(expected.issubset(seen), f"missing anchors: {expected - seen}")


class FrozenTreeGuardTests(unittest.TestCase):
    def test_the_expressive_registry_is_unchanged(self):
        self.assertEqual(len(E_OPERATOR_NAMES), 5)
        self.assertEqual(set(E_OPERATOR_NAMES) & set(BAND_FAMILIES),
                         {"dilated_e", "gated_e", "dynamic_e", "band_gated_e"})

    def test_the_band_stage_adds_nothing_to_the_search_pool(self):
        from tdarts import temporal_ops

        candidates = temporal_ops.canonical_candidates_all()
        for band, entries in candidates.items():
            with self.subTest(band=band):
                for name in (entry if isinstance(entry, str) else entry[0] for entry in entries):
                    self.assertNotIn("_e", str(name))

    def test_neither_frozen_tree_is_written_by_this_stage(self):
        """Source-level: the new modules must not name the frozen output roots
        as write targets.  The entry point's guards are the runtime half."""
        for name in ("tdarts/operator_v2e_band.py", "tdarts/operator_v2e_band_network.py",
                     "train_operator_v2e_band.py"):
            with self.subTest(module=name):
                source = (REPO_ROOT / name).read_text(encoding="utf-8")
                self.assertNotIn('"run/outputs/operator_v2"', source)
                self.assertNotIn("'run/outputs/operator_v2'", source)

    def test_the_frozen_matched_numbers_are_still_the_archived_ones(self):
        pinned = {
            "dilated": (540, 11880000),
            "gated": (612, 13464000),
            "local_attention": (462, 15114000),
            "dynamic": (512, 11616006),
            "band_gated": (552, 12078000),
        }
        from tdarts.operator_v2 import build_v2_operator, count_macs

        for name, (params, macs) in pinned.items():
            with self.subTest(operator=name):
                op = build_v2_operator(name, band="Low", target_rf=57)
                self.assertEqual(op.num_params, params)
                self.assertEqual(count_macs(op, (1, 3, 22, 1000)), macs)

    def test_the_expressive_pinned_numbers_are_still_the_archived_ones(self):
        """The band stage builds the same families, so an accidental edit here
        would move both generations' ratios at once."""
        pinned = {
            "dilated_e": 540,
            "gated_e": 1260,
            "local_attention_e": 624,
            "dynamic_e": 2176,
            "band_gated_e": 627,
        }
        for name, params in pinned.items():
            with self.subTest(operator=name):
                self.assertEqual(build_e_operator(name, band="Low", target_rf=57).num_params, params)


class EntryPointTests(unittest.TestCase):
    @staticmethod
    def _module():
        return _load_entry_point("train_operator_v2e_band.py", "band_entry_under_test")

    @staticmethod
    def _source() -> str:
        return (REPO_ROOT / "train_operator_v2e_band.py").read_text(encoding="utf-8")

    def test_session1_is_closed_unless_explicitly_requested(self):
        module = self._module()
        with mock.patch.object(sys, "argv", ["train_operator_v2e_band.py", "--seed", "1"]):
            self.assertFalse(module.parse_args().read_session1)
        with mock.patch.object(
            sys, "argv", ["train_operator_v2e_band.py", "--seed", "1", "--read-session1"]
        ):
            self.assertTrue(module.parse_args().read_session1)

    def test_the_session1_loader_is_reached_only_through_that_flag(self):
        source = self._source()
        self.assertEqual(source.count("load_subject_session("), 1)
        index = source.index("load_subject_session(")
        window = source[max(0, index - 200): index + 200]
        self.assertIn("args.read_session1", window)

    def test_the_leaf_name_carries_the_configuration_slug(self):
        source = self._source()
        self.assertRegex(source, r'target = slug_for\(')
        self.assertRegex(source, r'arm = f"\{args\.arm\}_\{target\}"')

    def test_default_receptive_field_is_the_middle_rung(self):
        module = self._module()
        self.assertEqual(module.DEFAULT_RF, 57)
        self.assertIn(module.DEFAULT_RF, C.RF_SPACE["Low"])

    def test_the_cli_default_is_the_anchor_in_all_three_bands(self):
        module = self._module()
        with mock.patch.object(sys, "argv", ["train_operator_v2e_band.py", "--seed", "1"]):
            args = module.parse_args()
        self.assertEqual((args.low, args.mid, args.high), (ANCHOR_FAMILY,) * 3)

    def test_local_attention_is_refused_with_the_reason(self):
        module = self._module()
        with contextlib.redirect_stderr(io.StringIO()):
            with mock.patch.object(
                sys,
                "argv",
                ["train_operator_v2e_band.py", "--seed", "1", "--low", "local_attention_e"],
            ):
                with self.assertRaises(SystemExit):
                    module.parse_args()

    def test_two_varying_bands_are_refused_at_runtime(self):
        module = self._module()
        with mock.patch.object(
            sys,
            "argv",
            ["train_operator_v2e_band.py", "--seed", "1", "--low", "dynamic_e", "--mid", "gated_e"],
        ):
            with self.assertRaises(ValueError):
                module.main()

    def test_an_arm_must_stay_in_the_band_tree(self):
        module = self._module()
        with mock.patch.object(
            sys, "argv", ["train_operator_v2e_band.py", "--seed", "1", "--arm", "operator_v2e"]
        ):
            with self.assertRaises(SystemExit):
                module.main()
        self.assertEqual(module.STAGE_PREFIX, STAGE_PREFIX)

    def test_the_output_root_must_be_a_band_tree(self):
        """A band run in the Expressive tree would be read as a global-family
        arm whose operator field is a configuration slug."""
        module = self._module()
        with mock.patch.object(
            sys,
            "argv",
            ["train_operator_v2e_band.py", "--seed", "1", "--output-root", "run/outputs/operator_v2e"],
        ):
            with self.assertRaises(SystemExit):
                module.main()

    def test_the_default_arm_matches_the_prefix_guard(self):
        module = self._module()
        with mock.patch.object(sys, "argv", ["train_operator_v2e_band.py", "--seed", "1"]):
            self.assertTrue(module.parse_args().arm.startswith(module.STAGE_PREFIX))

    def test_the_summary_records_the_red_line_keys(self):
        """The stage's own checklist: the three band families, the RF, and an
        explicitly closed Session 1, in the file the analysis reads back."""
        source = self._source()
        for key in ("band_families", "varying_band", "varying_family", "target_rf",
                    "session1_opened", "session1_test_size", "val_best_nll", "screening_only"):
            with self.subTest(key=key):
                self.assertIn(f'"{key}"', source)
        self.assertIn('"session1_test_size": None if test_data is None else len(test_data)', source)
        self.assertIn('"test": test', source)
        # config.json is the provenance record for one run, so subject and seed
        # are written at its top level and not only inside the nested args block.
        self.assertIn('"subject": str(args.subject)', source)
        self.assertIn('"seed": int(args.seed)', source)

    def test_the_config_records_all_three_bands_by_their_canonical_names(self):
        """The band names come from the shared vocabulary, not from three
        literals in the entry point: one spelling, one definition."""
        source = self._source()
        self.assertIn('"band_families": families', source)
        self.assertIn("families = config_of(args.low, args.mid, args.high)", source)
        self.assertEqual(config_of("gated_e", "dynamic_e", "band_gated_e"),
                         {"Low": "gated_e", "Mid": "dynamic_e", "High": "band_gated_e"})

    def test_the_protocol_constants_match_the_expressive_entry_point(self):
        """Same batch size, LR, budget, patience and stopping metric: no family
        is allowed its own training hyper-parameters."""
        band = self._module()
        from train_operator_v2e import parse_args as expressive_args  # noqa: E402

        with mock.patch.object(sys, "argv", ["x", "--operator", "gated_e", "--seed", "1"]):
            reference = expressive_args()
        with mock.patch.object(sys, "argv", ["x", "--seed", "1"]):
            mine = band.parse_args()
        for field in ("batch_size", "lr", "epochs", "patience", "rf", "num_workers"):
            with self.subTest(field=field):
                self.assertEqual(getattr(mine, field), getattr(reference, field))


class AnalyzerTests(unittest.TestCase):
    """``assess_band_probe`` is deliberately a pure function of its tables."""

    SUBJECTS = ["003", "005", "006"]
    BANDS = C.BANDS
    SEEDS = [20250901, 20250902, 20250903]

    @staticmethod
    def _tables(deltas):
        cells = cell_table(deltas, AnalyzerTests.SUBJECTS, AnalyzerTests.BANDS)
        winners = winner_table(deltas, AnalyzerTests.SUBJECTS, AnalyzerTests.BANDS, AnalyzerTests.SEEDS)
        effects = band_effect_table(deltas, AnalyzerTests.SUBJECTS, AnalyzerTests.BANDS, AnalyzerTests.SEEDS)
        return cells, winners, effects

    @staticmethod
    def _fill(fn):
        """A complete grid: a Delta for every subject x band x family x seed."""

        deltas = {}
        for subject in AnalyzerTests.SUBJECTS:
            for band in AnalyzerTests.BANDS:
                for family in VARYING_FAMILIES:
                    deltas[(subject, band, family)] = {
                        seed: fn(subject, band, family, index)
                        for index, seed in enumerate(AnalyzerTests.SEEDS)
                    }
        return deltas

    def _assess(self, deltas):
        cells, winners, effects = self._tables(deltas)
        return assess_band_probe(
            winners=winners, effects=effects, cell_table_=cells,
            subjects=AnalyzerTests.SUBJECTS, bands=AnalyzerTests.BANDS, seeds=AnalyzerTests.SEEDS,
        )

    def test_an_incomplete_grid_is_refused(self):
        """A shortened grid and a finished grid look the same in a summary
        table; the tool must not average over the hole."""
        deltas = self._fill(lambda *_: -0.5)
        del deltas[("006", "High", "band_gated_e")]
        result = self._assess(deltas)
        self.assertEqual(result["verdict"], "NOT_COMPUTABLE")
        self.assertIn("incomplete", result["reason"])

    def test_a_flat_grid_reads_as_weak_personalization(self):
        """Every replacement is a wash, so the anchor wins every cell."""
        result = self._assess(self._fill(lambda *_: 0.02))
        self.assertEqual(result["verdict"], "WEAK_PERSONALIZATION")
        self.assertIsNotNone(result["verdict_notice"])
        self.assertIn("evidence weak", result["verdict_notice"])

    def test_a_perfectly_structured_grid_reads_as_personalized(self):
        """Each subject wants a different family in a different band, the band
        effect beats the seed noise, and every seed agrees."""
        wanted = {
            ("003", "Low", "dynamic_e"),
            ("005", "Mid", "band_gated_e"),
            ("006", "High", "gated_e"),
        }

        def value(subject, band, family, index):
            if (subject, band, family) in wanted:
                return -0.30 - index * 0.01
            # A band effect that is real but points the other way in other bands.
            return 0.20 + index * 0.01

        result = self._assess(self._fill(value))
        self.assertEqual(result["verdict"], "PERSONALIZED_BAND_MECHANISM")
        for condition in result["conditions"].values():
            self.assertTrue(condition["passed"])

    def test_subjects_disagreeing_entirely_off_the_anchor_is_personalized(self):
        """The strongest form of the result, and the one a rule demanding an
        anchor on one side of the disagreement would have thrown away:

            003  Low -> dynamic_e   Mid -> anchor      High -> gated_e
            005  Low -> gated_e     Mid -> anchor      High -> gated_e
            006  Low -> dynamic_e   Mid -> band_gated  High -> anchor

        No subject wants the anchor where another wants a replacement, and the
        families still differ from subject to subject.  That is subject-specific
        mechanism preference with the anchor nowhere in the argument.
        """
        wanted = {
            ("003", "Low"): "dynamic_e",
            ("003", "High"): "gated_e",
            ("005", "Low"): "gated_e",
            ("005", "High"): "gated_e",
            ("006", "Low"): "dynamic_e",
            ("006", "Mid"): "band_gated_e",
        }

        def value(subject, band, family, index):
            if wanted.get((subject, band)) == family:
                return -0.30 - index * 0.01
            return 0.15 + index * 0.01

        result = self._assess(self._fill(value))
        conditions = result["conditions"]
        self.assertEqual(result["verdict"], "PERSONALIZED_BAND_MECHANISM")
        self.assertTrue(conditions["C2_subject_specific"]["passed"])
        # The disagreement is real, and it touches every band.
        self.assertEqual(conditions["C2_subject_specific"]["bands_where_subjects_disagree"],
                         ["High", "Low", "Mid"])
        # The Low band is the case in point: three subjects, two families, and
        # the anchor is not one of them.
        signatures = conditions["C2_subject_specific"]["signatures"]
        low_winners = {signatures[subject][0] for subject in signatures}
        self.assertNotIn(ANCHOR_FAMILY, low_winners)
        self.assertEqual(low_winners, {"dynamic_e", "gated_e"})

    def test_a_replacement_shared_by_every_subject_is_not_personalized(self):
        """The same replacement helps the same band for everybody: that is a
        band effect, and it is not evidence of personalisation.  Every subject's
        signature is identical, so nothing disagrees."""
        def value(subject, band, family, index):
            if band == "Low" and family == "dynamic_e":
                return -0.30 - index * 0.01
            return 0.20 + index * 0.01

        result = self._assess(self._fill(value))
        conditions = result["conditions"]
        self.assertEqual(result["verdict"], "BAND_EFFECT_WITHOUT_STABLE_PERSONALIZATION")
        self.assertTrue(conditions["C1_band_effect"]["passed"])
        self.assertFalse(conditions["C2_subject_specific"]["passed"])
        # One signature across all three subjects, and no pair disagrees.
        self.assertEqual(conditions["C2_subject_specific"]["distinct_signatures"], 1)
        self.assertEqual(conditions["C2_subject_specific"]["disagreeing_subject_pairs"], [])
        self.assertEqual(conditions["C2_subject_specific"]["bands_where_subjects_disagree"], [])

    def test_a_signal_carried_by_one_subject_gets_its_own_label(self):
        """003 prefers a replacement, 005 and 006 want the anchor everywhere.

        ``003 != 005 = 006`` is a real subject-specific signal with thin
        evidence -- not the same finding as "no personalization", and not the
        same as a cohort-wide effect.  It gets its own label so the next step is
        "run more subjects" rather than "the idea failed".
        """
        def value(subject, band, family, index):
            if subject == "003" and band == "Low" and family == "dynamic_e":
                return -0.30 - index * 0.01
            return 0.15 + index * 0.01

        result = self._assess(self._fill(value))
        conditions = result["conditions"]
        self.assertEqual(result["verdict"], "SINGLE_SUBJECT_PERSONALIZATION_SIGNAL")
        for name in ("C1_band_effect", "C2_subject_specific", "C3_stable_across_seeds"):
            with self.subTest(condition=name):
                self.assertTrue(conditions[name]["passed"])
        self.assertEqual(result["subjects_with_a_stable_replacement"], ["003"])
        self.assertEqual(conditions["C2_subject_specific"]["n_subjects_with_a_stable_replacement"], 1)
        # One subject is enough for the band-effect condition; the count is a
        # label question, not a condition.
        self.assertEqual(conditions["C1_band_effect"]["needs_at_least"], 1)
        self.assertEqual(conditions["C1_band_effect"]["subjects_with_a_stable_band_effect"], ["003"])
        self.assertIn("扩展 subject", result["verdict_notice"])

    def test_one_more_subject_promotes_the_label(self):
        """The label boundary, and only that: add a second subject carrying a
        stable replacement and the verdict becomes the cohort-wide one.  The
        three conditions are satisfied in both grids."""
        def single(subject, band, family, index):
            if subject == "003" and band == "Low" and family == "dynamic_e":
                return -0.30 - index * 0.01
            return 0.15 + index * 0.01

        def pair(subject, band, family, index):
            if subject == "003" and band == "Low" and family == "dynamic_e":
                return -0.30 - index * 0.01
            if subject == "005" and band == "High" and family == "gated_e":
                return -0.30 - index * 0.01
            return 0.15 + index * 0.01

        first = self._assess(self._fill(single))
        second = self._assess(self._fill(pair))
        self.assertEqual(first["verdict"], "SINGLE_SUBJECT_PERSONALIZATION_SIGNAL")
        self.assertEqual(second["verdict"], "PERSONALIZED_BAND_MECHANISM")
        for result in (first, second):
            for name in ("C1_band_effect", "C2_subject_specific", "C3_stable_across_seeds"):
                with self.subTest(verdict=result["verdict"], condition=name):
                    self.assertTrue(result["conditions"][name]["passed"])
        self.assertEqual(second["subjects_with_a_stable_replacement"], ["003", "005"])

    def test_a_single_subject_grid_is_not_read_as_no_personalization(self):
        """The branch must not be reachable through WEAK_PERSONALIZATION's
        notice: the two verdicts mean different things and say so."""
        def value(subject, band, family, index):
            if subject == "003" and band == "Low" and family == "dynamic_e":
                return -0.30 - index * 0.01
            return 0.15 + index * 0.01

        result = self._assess(self._fill(value))
        self.assertNotEqual(result["verdict"], "WEAK_PERSONALIZATION")
        self.assertNotIn("evidence weak", result["verdict_notice"])

    def test_both_stability_figures_are_reported_and_only_the_contested_one_decides(self):
        """A grid where nothing was ever replaced reads 100% overall stability
        while deciding nothing.  The overall figure stays in the report -- "this
        band stably needs no replacement" is a fact worth keeping -- but the
        condition uses the contested cells only."""
        result = self._assess(self._fill(lambda *_: 0.02))
        condition = result["conditions"]["C3_stable_across_seeds"]
        self.assertFalse(condition["passed"])
        self.assertEqual(condition["contested_cells"], [])
        self.assertEqual(condition["contested_cell_stability"], "0/0")
        # The descriptive companion is still reported, and it is the misleading
        # 100% that motivated the split.
        self.assertEqual(condition["overall_winner_stability"], "9/9")
        self.assertEqual(condition["overall_stable_fraction"], 1.0)

    def test_the_two_stability_figures_differ_when_a_contested_cell_flips(self):
        """A stable-but-uncontested grid and a contested one are not the same
        claim, and the report has to separate them."""
        def value(subject, band, family, index):
            if (subject, band) in {("003", "Low"), ("005", "Mid")}:
                # Contested, and stable: all three seeds agree.
                return -0.30 - index * 0.01
            return 0.15 + index * 0.01

        condition = self._assess(self._fill(value))["conditions"]["C3_stable_across_seeds"]
        self.assertTrue(condition["passed"])
        self.assertEqual(condition["contested_cell_stability"], "2/2")
        self.assertEqual(condition["overall_winner_stability"], "9/9")

        def flipping(subject, band, family, index):
            if (subject, band) == ("003", "Low") and family == "dynamic_e":
                return (-0.6, 0.7, -0.5)[index]
            return 0.15 + index * 0.01

        condition = self._assess(self._fill(flipping))["conditions"]["C3_stable_across_seeds"]
        self.assertFalse(condition["passed"])
        self.assertEqual(condition["contested_cell_stability"], "0/1")
        # The grid is otherwise untouched, so the overall figure still looks
        # healthy: the two numbers say different things on purpose.
        self.assertEqual(condition["overall_winner_stability"], "8/9")

    def test_a_winner_that_flips_across_seeds_is_not_stable(self):
        """Directions that disagree with each other are a coin flip with a
        label on it, however large the mean looks."""
        def value(subject, band, family, index):
            if (subject, band, family) in {("003", "Low", "dynamic_e"), ("005", "Mid", "gated_e"),
                                           ("006", "High", "band_gated_e")}:
                # Seed 0 helps, seed 1 hurts, seed 2 helps: mean still negative.
                return (-0.6, 0.7, -0.5)[index]
            return 0.1

        result = self._assess(self._fill(value))
        self.assertFalse(result["conditions"]["C3_stable_across_seeds"]["passed"])
        self.assertNotEqual(result["verdict"], "PERSONALIZED_BAND_MECHANISM")

    def test_the_anchor_is_a_candidate_at_zero(self):
        """A replacement has to be strictly better than doing nothing."""
        deltas = self._fill(lambda *_: 0.0)
        _, winners, _ = self._tables(deltas)
        for row in winners["cells"].values():
            self.assertEqual(row["winner"], ANCHOR_FAMILY)
        self.assertEqual(winners["distinct_winners"], [ANCHOR_FAMILY])

    def test_a_tie_goes_to_the_anchor(self):
        deltas = self._fill(lambda *_: 0.0)
        deltas[("003", "Low", "dynamic_e")] = {seed: 0.0 for seed in self.SEEDS}
        _, winners, _ = self._tables(deltas)
        self.assertEqual(winners["cells"]["003/Low"]["winner"], ANCHOR_FAMILY)

    def test_a_band_effect_needs_to_beat_the_seed_noise(self):
        """Same band spread, different noise: only one of the two is an effect."""
        clean = self._fill(lambda subject, band, family, index: {"Low": -0.2, "Mid": 0.0, "High": 0.2}[band])
        noisy = self._fill(lambda subject, band, family, index: {"Low": -0.2, "Mid": 0.0, "High": 0.2}[band]
                           + (-0.6, 0.0, 0.6)[index])
        _, _, clean_effects = self._tables(clean)
        _, _, noisy_effects = self._tables(noisy)
        self.assertTrue(clean_effects["003/dynamic_e"]["exceeds_seed_noise"])
        self.assertFalse(noisy_effects["003/dynamic_e"]["exceeds_seed_noise"])


class PairingTests(unittest.TestCase):
    """The pairing itself: the anchor is subtracted per subject and seed."""

    def _probe(self, subject, seed, band, family, nll):
        slug = ANCHOR_SLUG if band is None else f"{band.lower()}_{family}"
        return {"path": Path(f"/tmp/{subject}_{seed}_{band}"), "subject": subject, "seed": seed,
                "band": band, "family": family, "slug": slug,
                "val_best_nll": nll, "val_best_acc": 0.5, "summary": {}}

    def _anchor(self, subject, seed, nll):
        return {"path": Path("/tmp/anchor"), "summary": {"val_best_nll": nll, "val_best_acc": 0.6}}

    def test_the_delta_is_a_log_ratio_against_the_matched_seed(self):
        import math

        anchors = {("003", 20250901): self._anchor("003", 20250901, 0.5),
                   ("003", 20250902): self._anchor("003", 20250902, 0.8)}
        probes = [self._probe("003", 20250901, "Low", "dynamic_e", 0.25),
                  self._probe("003", 20250902, "Mid", "gated_e", 0.8)]
        rows, missing = build_pairs(anchors, probes)
        self.assertEqual(missing, [])
        self.assertAlmostEqual(rows[0]["delta_log_nll"], math.log(0.25 / 0.5))
        # Same NLL as the anchor, different seed: Delta is exactly zero.
        self.assertAlmostEqual(rows[1]["delta_log_nll"], 0.0)
        self.assertAlmostEqual(rows[0]["delta_raw_nll"], -0.25)

    def test_an_unmatched_seed_is_reported_rather_than_dropped_silently(self):
        probes = [self._probe("003", 20250909, "Low", "dynamic_e", 0.25)]
        rows, missing = build_pairs({}, probes)
        self.assertEqual(rows, [])
        self.assertEqual(len(missing), 1)
        self.assertIn("no anchor run", missing[0])

    def test_the_all_anchor_configuration_is_not_a_probe_to_pair(self):
        probes = [self._probe("003", 20250901, None, ANCHOR_FAMILY, 0.5)]
        rows, missing = build_pairs({("003", 20250901): self._anchor("003", 20250901, 0.5)}, probes)
        self.assertEqual(rows, [])
        self.assertIn("already the anchor", missing[0])


class SubmitChainTests(unittest.TestCase):
    """The driver is what makes the grid 81 runs and keeps Session 1 shut."""

    @staticmethod
    def _submit_source() -> str:
        return (REPO_ROOT / "run" / "bin" / "submit_operator_v2e_band.sh").read_text(encoding="utf-8")

    @staticmethod
    def _job_source() -> str:
        return (REPO_ROOT / "run" / "bin" / "run_operator_v2e_band_job.sh").read_text(encoding="utf-8")

    def test_the_grid_is_eighty_one_runs(self):
        source = self._submit_source()
        self.assertIn("SUBJECTS=\"${SUBJECTS:-2 4 5}\"", source)
        self.assertIn("20250901 20250902 20250903", source)
        self.assertIn("9 configurations", source)

    def test_the_driver_never_opens_session1(self):
        job = self._job_source()
        self.assertNotIn("--read-session1", job)
        # The entry point's flag is the only route to the loader; the job must
        # not pass a stray variable that could expand to it.
        self.assertIn("session1=closed", job)

    def test_the_base_configurations_never_vary_two_bands(self):
        """Every generated configuration is one band plus two anchors."""
        job = self._job_source()
        self.assertIn("--low \"$LOW\"", job)
        self.assertIn("--mid \"$MID\"", job)
        self.assertIn("--high \"$HIGH\"", job)
        self.assertIn("dilated_e", job)

    def test_the_driver_owns_its_own_ledger_and_tree(self):
        source = self._submit_source()
        self.assertIn(f"operator_v2e_band_submitted.txt", source)
        self.assertIn(f"OUTARM=\"${{OUTARM:-{STAGE_PREFIX}}}\"", source)
        # The frozen Expressive ledger must not be shareable with this grid.
        self.assertNotIn("opv2e_submitted.txt", source)


class FullGridPipelineTests(unittest.TestCase):
    """Discovery, pairing, decomposition and verdict, on a synthetic full grid.

    ``AnalyzerTests`` feeds the rule tables directly, which pins the decision but
    not the plumbing that builds those tables.  This one writes the shape of tree
    the 81 runs will leave on disk -- including a planted ground truth -- and
    requires the tool to recover it.  No training happens; the summaries are
    synthetic, and the point is that a known answer comes back out.
    """

    SUBJECTS = ("003", "005", "006")
    SEEDS = (20250901, 20250902, 20250903)
    BASE = {"003": 0.5, "005": 0.6, "006": 0.8}
    #: One subject wants one family in one band, and the three answers differ.
    WANTED = {
        ("003", "Low", "dynamic_e"): -0.30,
        ("005", "Mid", "band_gated_e"): -0.30,
        ("006", "High", "gated_e"): -0.30,
    }

    def _write(self, root: Path) -> None:
        for subject in self.SUBJECTS:
            for seed in self.SEEDS:
                anchor = self.BASE[subject] * (1 + (seed - 20250901) * 1e-4)
                leaf = root / "anchor" / "bci42a" / f"train_s{subject}_seed{seed}_operator_v2e_dilated_e"
                leaf.mkdir(parents=True, exist_ok=True)
                (leaf / "final_summary.json").write_text(json.dumps({
                    "subject": subject, "seed": seed, "operator": ANCHOR_FAMILY,
                    "val_best_nll": anchor, "val_best_acc": 0.8,
                    "session1_opened": False, "test": None, "screening_only": True,
                    "target_rf": 57, "best_epoch": 300,
                }))
                (leaf / "metrics.jsonl").write_text(
                    json.dumps({"epoch": 300, "val_nll": anchor, "val_acc": 0.8,
                                "train_nll_eval": 0.01, "train_acc": 1.0}) + "\n"
                )
                for band in C.BANDS:
                    for family in VARYING_FAMILIES:
                        slug = f"{band.lower()}_{family}"
                        families = config_of(ANCHOR_FAMILY, ANCHOR_FAMILY, ANCHOR_FAMILY)
                        families[band] = family
                        delta = self.WANTED.get((subject, band, family), 0.15)
                        delta += (seed - 20250902) * 0.01
                        leaf = root / "probe" / "bci42a" / f"train_s{subject}_seed{seed}_{STAGE_PREFIX}_{slug}"
                        leaf.mkdir(parents=True, exist_ok=True)
                        nll = anchor * math.exp(delta)
                        (leaf / "final_summary.json").write_text(json.dumps({
                            "subject": subject, "seed": seed, "operator": slug,
                            "band_families": families, "varying_band": band,
                            "varying_family": family, "val_best_nll": nll,
                            "val_best_acc": 0.8, "session1_opened": False, "test": None,
                            "screening_only": True, "target_rf": 57, "best_epoch": 300,
                        }))
                        (leaf / "metrics.jsonl").write_text(
                            json.dumps({"epoch": 300, "val_nll": nll, "val_acc": 0.8,
                                        "train_nll_eval": 0.01, "train_acc": 1.0}) + "\n"
                        )

    def test_a_planted_ground_truth_is_recovered(self):
        with tempfile.TemporaryDirectory() as container:
            root = Path(container)
            self._write(root)
            argc = ["prog",
                    "--probe-root", str(root / "probe"),
                    "--anchor-root", str(root / "anchor"),
                    "--json", str(root / "out.json")]
            stream = io.StringIO()
            with mock.patch.object(sys, "argv", argc), contextlib.redirect_stdout(stream):
                from tools import analyze_operator_v2e_band as tool

                self.assertEqual(tool.main(), 0)
            payload = json.loads((root / "out.json").read_text(encoding="utf-8"))

        self.assertEqual(payload["paired_runs"], 81)
        self.assertEqual(payload["unpaired"], [])
        for (subject, band, family), target in self.WANTED.items():
            with self.subTest(subject=subject, band=band, family=family):
                row = payload["cells"][f"{subject}/{band}/{family}"]
                self.assertAlmostEqual(row["mean_delta_log_nll"], target, places=6)
                self.assertEqual(row["improved_seeds"], 3)
                self.assertEqual(payload["winners"]["cells"][f"{subject}/{band}"]["winner"], family)
                self.assertTrue(payload["winners"]["cells"][f"{subject}/{band}"]["stable_across_seeds"])

        # The sensitivity figure has to average over runs, not over cells: a
        # per-cell average here would silently collapse the three seeds into one.
        self.assertEqual(payload["raw_delta_nll_mean_runs"], 81)

        assessment = payload["assessment"]
        self.assertEqual(assessment["verdict"], "PERSONALIZED_BAND_MECHANISM")
        # The decomposition runs on the paired difference, so every family has
        # the full 3x3 grid and estimable residual degrees of freedom.
        for family in VARYING_FAMILIES:
            with self.subTest(family=family):
                self.assertIsNotNone(payload["decomposition"][family])

    def test_the_report_states_that_session1_was_never_open(self):
        with tempfile.TemporaryDirectory() as container:
            root = Path(container)
            self._write(root)
            argc = ["prog", "--probe-root", str(root / "probe"), "--anchor-root", str(root / "anchor"),
                    "--json", str(root / "out.json")]
            stream = io.StringIO()
            with mock.patch.object(sys, "argv", argc), contextlib.redirect_stdout(stream):
                from tools import analyze_operator_v2e_band as tool

                tool.main()
            text = stream.getvalue()
        self.assertIn("Session 1 始终关闭", text)
        # The early-overfit diagnostic is printed with its own disclaimer, so it
        # cannot be lifted out of the report as a selection rule.
        self.assertIn("不得用于删 run", text)


class DatasetBackedEquivalenceTests(unittest.TestCase):
    """End-to-end: both entry points, same seed, same data, same epochs.

    The network tests above pin the initial weights.  This one pins everything
    downstream of them -- loader order, optimiser steps, evaluation -- which is
    what the anchor reuse actually depends on.  Skipped when the dataset or the
    compute is not available, because its absence must not look like a pass.
    """

    def test_the_band_entry_reproduces_the_expressive_entry_bit_for_bit(self):
        if not (DATA_ROOT / "dataLabels.csv").is_file():
            self.skipTest(f"dataset not reachable at {DATA_ROOT}")
        band_module = _load_entry_point("train_operator_v2e_band.py", "band_entry_e2e")
        reference_module = _load_entry_point("train_operator_v2e.py", "expressive_entry_e2e")
        with tempfile.TemporaryDirectory() as container:
            # The band entry point requires an output-root whose last component
            # carries the stage prefix, so the two trees live under one scratch
            # directory with names the guard accepts.
            left = Path(container) / f"{STAGE_PREFIX}_expressive"
            right = Path(container) / f"{STAGE_PREFIX}_band"
            common = ["--subject", "003", "--seed", "20250901", "--rf", "57",
                      "--data-root", str(DATA_ROOT), "--dataset", "bci42a",
                      "--epochs", "2", "--patience", "2", "--device", "cpu"]
            argv = ["x", "--operator", "dilated_e", "--output-root", str(left),
                    "--log-root", str(left) + "_log", "--arm", "operator_v2e"] + common
            with mock.patch.object(sys, "argv", argv):
                self.assertEqual(reference_module.main(), 0)
            argv = ["x", "--output-root", str(right), "--log-root", str(right) + "_log",
                    f"--arm={STAGE_PREFIX}"] + common
            with mock.patch.object(sys, "argv", argv):
                self.assertEqual(band_module.main(), 0)

            def metrics(root: Path):
                leaf = next((root / "bci42a").iterdir())
                return [json.loads(line) for line in
                        (leaf / "metrics.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]

            left_rows, right_rows = metrics(left), metrics(right)
            self.assertEqual(len(left_rows), len(right_rows))
            fields = ["epoch", "train_nll", "train_acc", "train_nll_eval", "val_nll", "val_acc", "is_best"]
            for left_row, right_row in zip(left_rows, right_rows):
                for field in fields:
                    with self.subTest(epoch=left_row["epoch"], field=field):
                        self.assertEqual(left_row[field], right_row[field])


if __name__ == "__main__":
    unittest.main(verbosity=2)
