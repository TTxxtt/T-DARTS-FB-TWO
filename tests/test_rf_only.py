"""Phase B RF-only checks: one operator, joint RF decode, unordered pairs.

The RF-only arm fixes both paths to ``dilated`` and searches only the receptive
field.  These tests pin the properties it relies on:

* the supernet exposes exactly 24 gammas (3 bands x 2 paths x 4 RFs);
* with ``no_duplicate_paths`` the two RFs of a band are decoded jointly, so a
  softmax tie on one RF still yields two *different* structures;
* without the flag the same tie would export a duplicate, which is why the
  main experiment enables it;
* the exported genotype is a plain six-gene file that round-trips through
  ``save_genotype`` / ``load_genotype`` and builds the unchanged discrete net;
* the two paths are treated as an unordered pair ``(15, 57) == (57, 15)``.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

import train_rf_search
from tdarts import config as C
from tdarts.anchored import AnchoredRFNet, build_anchored_genotype
from tdarts.discrete_network import TemporalDiscreteNet
from tdarts.genotype import (
    duplicate_structure_bands,
    load_genotype,
    save_genotype,
)

ELECTRODES = 2
TIMEPOINTS = 32


def build_model(no_duplicate_paths: bool) -> AnchoredRFNet:
    return AnchoredRFNet(
        {band: train_rf_search.FIXED_OPERATOR for band in C.BANDS},
        n_electrodes=ELECTRODES,
        no_duplicate_paths=no_duplicate_paths,
    )


def force_rf_collision(model: AnchoredRFNet, rf: int = 57) -> None:
    """Point both gamma containers of every band at the same RF."""

    with torch.no_grad():
        for cell in model.cells.values():
            index = cell.rfs.index(rf)
            cell.gamma_anchor.fill_(-5.0)
            cell.gamma_searched.fill_(-5.0)
            cell.gamma_anchor[index] = 5.0
            cell.gamma_searched[index] = 5.0


class RfOnlyArchitectureTests(unittest.TestCase):
    def test_arch_parameters_are_24_gammas_in_six_containers(self):
        model = build_model(no_duplicate_paths=True)
        self.assertEqual(model.num_arch_parameters(), 24)
        self.assertEqual(len(model.arch_parameters()), 6)
        for cell in model.cells.values():
            self.assertEqual(len(cell.rfs), 4)
            self.assertEqual(len(cell.architecture_parameters()), 2)

    def test_forward_and_backward_reach_every_gamma(self):
        model = build_model(no_duplicate_paths=True)
        logits, _ = model(torch.randn(2, 9, ELECTRODES, TIMEPOINTS))
        self.assertEqual(tuple(logits.shape), (2, 4))
        logits.sum().backward()
        for gamma in model.arch_parameters():
            self.assertIsNotNone(gamma.grad)
            self.assertTrue(torch.isfinite(gamma.grad).all())

    def test_forced_collision_is_resolved_jointly(self):
        model = build_model(no_duplicate_paths=True)
        force_rf_collision(model, rf=57)
        for cell in model.cells.values():
            anchor_index, searched_index = cell.select_rf_indices()
            self.assertNotEqual(
                cell.rfs[anchor_index],
                cell.rfs[searched_index],
                msg=f"band {cell.band}: joint decode returned a duplicate RF",
            )

    def test_without_the_flag_the_collision_survives(self):
        model = build_model(no_duplicate_paths=False)
        force_rf_collision(model, rf=57)
        genotype = build_anchored_genotype(model, seed=1, epoch=1)
        self.assertEqual(
            duplicate_structure_bands(genotype), tuple(C.BANDS)
        )


class RfOnlyGenotypeTests(unittest.TestCase):
    def setUp(self):
        model = build_model(no_duplicate_paths=True)
        force_rf_collision(model, rf=57)
        self.genotype = build_anchored_genotype(model, seed=7, epoch=200)

    def test_export_is_dilated_only_and_duplicate_free(self):
        self.assertEqual(duplicate_structure_bands(self.genotype), ())
        for gene in self.genotype.genes:
            self.assertTrue(gene.candidate.startswith("dilated_rf"), gene.candidate)

    def test_round_trip_and_discrete_build(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = save_genotype(self.genotype, Path(temporary) / "genotype.json")
            loaded = load_genotype(path)
            self.assertEqual(loaded, self.genotype)
            discrete = TemporalDiscreteNet(loaded, n_electrodes=ELECTRODES)
            logits, _ = discrete(torch.randn(2, 9, ELECTRODES, TIMEPOINTS))
            self.assertEqual(tuple(logits.shape), (2, 4))

    def test_unordered_pairs_ignore_path_labels(self):
        pairs = train_rf_search._unordered_rf_pairs(self.genotype)
        swapped = self.genotype
        # Rebuild with the two paths of every band exchanged.
        from tdarts.genotype import Genotype, PathGene

        genes = []
        for band in C.BANDS:
            for path in (0, 1):
                source = swapped.gene(band, 1 - path)
                genes.append(
                    PathGene(band, path, source.candidate_index, source.candidate, 0.0, 1.0)
                )
        exchanged = Genotype(seed=1, epoch=0, genes=tuple(genes))
        self.assertEqual(pairs, train_rf_search._unordered_rf_pairs(exchanged))
        for pair in pairs.values():
            self.assertEqual(pair, sorted(pair))
            self.assertEqual(len(set(pair)), 2)


if __name__ == "__main__":  # pragma: no cover
    unittest.main(verbosity=2)
