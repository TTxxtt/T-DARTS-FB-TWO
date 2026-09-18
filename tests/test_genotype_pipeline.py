"""Search-log -> genotype.json -> discrete-network pipeline checks.

These close the loop that Time-Conv V1 depends on: a logged search epoch is
decoded, saved, loaded back and built into a discrete network without any alpha
or candidate pool.  They also pin the structural duplicate check that
``train_retrain.py --no-duplicate-paths`` relies on.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

import train_retrain
from tdarts.discrete_network import TemporalDiscreteNet
from tdarts.genotype import (
    Genotype,
    PathGene,
    duplicate_structure_bands,
    extract_genotype,
    load_genotype,
    path_structure_keys,
    save_genotype,
)
from tdarts.mixed_op import candidate_names

# A genotype with six distinct structures, one per path.
DISTINCT_INDICES = {"Low": (2, 5), "Mid": (9, 12), "High": (7, 13)}


def _paths_block(indices: dict[str, tuple[int, int]]) -> dict[str, dict]:
    """One logged ``paths`` mapping in exactly train_search.py's shape."""

    paths: dict[str, dict] = {}
    for band, pair in indices.items():
        names = candidate_names(band)
        for path, index in enumerate(pair):
            top2 = (index + 1) % len(names)
            probabilities = [1.0 / len(names)] * len(names)
            paths[f"{band}_path{path + 1}"] = {
                "alpha": [0.01 * (i + 1) for i in range(len(names))],
                "probabilities": probabilities,
                "top1": {
                    "index": index,
                    "candidate": names[index],
                    "probability": probabilities[index],
                },
                "top2": {
                    "index": top2,
                    "candidate": names[top2],
                    "probability": probabilities[top2],
                },
                "margin": probabilities[index] - probabilities[top2],
                "entropy": 0.0,
            }
    return paths


def _genotype_from_names(pairs: dict[str, tuple[str, str]]) -> Genotype:
    """Build a genotype for tests that name candidates directly."""

    genes = []
    for band, pair in pairs.items():
        names = candidate_names(band)
        for path, candidate in enumerate(pair):
            index = names.index(candidate) if candidate in names else 0
            genes.append(PathGene(band, path, index, candidate, 0.0, 1.0))
    return Genotype(seed=1, epoch=0, genes=tuple(genes))


class SearchToGenotypeTests(unittest.TestCase):
    def setUp(self):
        self._temporary = tempfile.TemporaryDirectory()
        self.root = Path(self._temporary.name)
        # extract_genotype recovers the seed from the directory name, so the
        # synthetic search leaf must spell it exactly like a real one.
        self.search_dir = self.root / "search_s003_seed20250901"
        self.search_dir.mkdir()
        self.metrics = self.search_dir / "metrics.jsonl"
        self.metrics.write_text(
            json.dumps({"epoch": 200, "paths": _paths_block(DISTINCT_INDICES)}) + "\n",
            encoding="utf-8",
        )

    def tearDown(self):
        self._temporary.cleanup()

    def test_extract_save_load_round_trip(self):
        extracted = extract_genotype(self.metrics, epoch=200)
        self.assertEqual(extracted.seed, 20250901)
        expected = {
            (band, path): candidate_names(band)[index]
            for band, pair in DISTINCT_INDICES.items()
            for path, index in enumerate(pair)
        }
        for (band, path), candidate in expected.items():
            self.assertEqual(extracted.gene(band, path).candidate, candidate)

        path = save_genotype(extracted, self.root / "genotype.json")
        loaded = load_genotype(path)
        self.assertEqual(loaded, extracted)

    def test_loaded_genotype_builds_an_alpha_free_discrete_network(self):
        loaded = load_genotype(
            save_genotype(extract_genotype(self.metrics, epoch=200), self.root / "g.json")
        )
        model = TemporalDiscreteNet(loaded, n_electrodes=2)
        names = [name for name, _ in model.named_parameters()]
        self.assertFalse(any(name.endswith("alpha") for name in names))
        self.assertFalse(any(".ops." in name for name, _ in model.named_modules()))

        logits, features = model(torch.randn(2, 9, 2, 32))
        self.assertEqual(tuple(logits.shape), (2, 4))
        self.assertEqual(features.shape[0], 2)
        logits.sum().backward()

    def test_load_rejects_a_candidate_ordering_mismatch(self):
        payload = json.loads(
            save_genotype(
                extract_genotype(self.metrics, epoch=200), self.root / "g.json"
            ).read_text(encoding="utf-8")
        )
        entry = payload["genes"][0]
        names = candidate_names(entry["band"])
        entry["candidate"] = names[(entry["candidate_index"] + 1) % len(names)]
        broken = self.root / "broken.json"
        broken.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaises(ValueError):
            load_genotype(broken)

    def test_load_accepts_an_inline_dict(self):
        """--stage2-only rebuilds the genotype from final_summary.json's inline
        Genotype.to_dict() rather than from a path; the loader must accept the
        dict directly (passing it as a path would TypeError in Path(dict))."""
        genotype = extract_genotype(self.metrics, epoch=200)
        inline = genotype.to_dict()
        loaded = load_genotype(inline)
        self.assertEqual(loaded, genotype)
        # Same validation as the path form: a corrupted inline dict is rejected.
        inline["genes"][0]["candidate"] = "not-a-real-op"
        with self.assertRaises(ValueError):
            load_genotype(inline)


class DuplicateStructureTests(unittest.TestCase):
    def test_distinct_structures_are_not_reported(self):
        genotype = _genotype_from_names(
            {
                "Low": ("dilated_rf57", "normal_rf57"),
                "Mid": ("dwsep_rf113", "lkdw_rf29"),
                "High": ("normal_rf113", "dwsep_rf15"),
            }
        )
        self.assertEqual(duplicate_structure_bands(genotype), ())

    def test_same_name_in_both_paths_is_reported(self):
        genotype = _genotype_from_names(
            {
                "Low": ("dilated_rf57", "dilated_rf57"),
                "Mid": ("dwsep_rf113", "lkdw_rf29"),
                "High": ("normal_rf113", "dwsep_rf15"),
            }
        )
        self.assertEqual(duplicate_structure_bands(genotype), ("Low",))

    def test_aliases_of_one_function_are_reported(self):
        # dilated_rf15 and normal_rf15 are different names for kernel 15,
        # dilation 1, dense: the check must run on structure_key, not strings.
        genotype = _genotype_from_names(
            {
                "Low": ("dilated_rf15", "normal_rf15"),
                "Mid": ("dwsep_rf113", "lkdw_rf29"),
                "High": ("normal_rf113", "dwsep_rf15"),
            }
        )
        self.assertEqual(duplicate_structure_bands(genotype), ("Low",))

    def test_structure_keys_come_from_the_built_operators(self):
        genotype = _genotype_from_names(
            {
                "Low": ("dilated_rf57", "normal_rf57"),
                "Mid": ("dwsep_rf57", "lkdw_rf57"),
                "High": ("dilated_rf15", "normal_rf15"),
            }
        )
        keys = path_structure_keys(genotype)
        self.assertEqual(keys["Low"], ((15, 4, False), (57, 1, False)))
        self.assertEqual(keys["Mid"], ((15, 4, True), (57, 1, True)))
        self.assertEqual(keys["High"], ((15, 1, False), (15, 1, False)))


class RetrainCliTests(unittest.TestCase):
    def _parse(self, argv: list[str]):
        with mock.patch.object(sys, "argv", ["train_retrain.py", *argv]):
            return train_retrain.parse_args()

    def test_no_duplicate_paths_defaults_to_off(self):
        args = self._parse(["--genotype-json", "g.json", "--seed", "1"])
        self.assertFalse(args.no_duplicate_paths)

    def test_no_duplicate_paths_can_be_requested(self):
        args = self._parse(
            ["--genotype-json", "g.json", "--seed", "1", "--no-duplicate-paths"]
        )
        self.assertTrue(args.no_duplicate_paths)

    def test_stage2_fixed_epochs_defaults_to_off(self):
        """Default None keeps the historical threshold rule, which is what every
        archived run -- and therefore every published number -- used."""
        args = self._parse(["--genotype-json", "g.json", "--seed", "1"])
        self.assertIsNone(args.stage2_fixed_epochs)

    def test_stage2_fixed_epochs_can_be_requested(self):
        args = self._parse(
            ["--genotype-json", "g.json", "--seed", "1", "--stage2-fixed-epochs", "200"]
        )
        self.assertEqual(args.stage2_fixed_epochs, 200)

    def test_stage2_min_epochs_defaults_to_off(self):
        """0 is the historical rule: the threshold break is allowed at any
        epoch.  Every archived run was produced under it."""
        args = self._parse(["--genotype-json", "g.json", "--seed", "1"])
        self.assertEqual(args.stage2_min_epochs, 0)

    def test_stage2_min_epochs_can_be_requested(self):
        args = self._parse(
            ["--genotype-json", "g.json", "--seed", "1", "--stage2-min-epochs", "100"]
        )
        self.assertEqual(args.stage2_min_epochs, 100)

    def test_stage2_min_epochs_rejects_negative(self):
        with self.assertRaises(ValueError):
            with mock.patch.object(
                sys, "argv",
                ["train_retrain.py", "--genotype-json", "g.json", "--seed", "1",
                 "--stage2-min-epochs", "-5"],
            ):
                train_retrain.main()

    def test_stage2_min_and_fixed_are_mutually_exclusive(self):
        """Both knobs pin the length in different ways; silently letting one win
        would make final_summary.json's stop_reason the only record of which."""
        with self.assertRaises(ValueError):
            with mock.patch.object(
                sys, "argv",
                ["train_retrain.py", "--genotype-json", "g.json", "--seed", "1",
                 "--stage2-min-epochs", "100", "--stage2-fixed-epochs", "200"],
            ):
                train_retrain.main()

    def test_stage2_fixed_epochs_must_be_positive(self):
        """A zero or negative fixed length would make Stage 2 run no epochs at
        all and still report a Session-1 test reading from the Stage-1 model,
        which reads as a completed Stage 2 in final_summary.json."""
        for value in ("0", "-1"):
            with self.assertRaises(ValueError):
                with mock.patch.object(
                    sys,
                    "argv",
                    ["train_retrain.py", "--genotype-json", "g.json", "--seed", "1",
                     "--stage2-fixed-epochs", value],
                ):
                    train_retrain.main()


if __name__ == "__main__":  # pragma: no cover
    unittest.main(verbosity=2)
