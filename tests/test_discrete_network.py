"""Stage-4 checks: decode fixed paths without changing the selected function."""

from __future__ import annotations

import unittest

import torch

from tdarts.discrete_network import TemporalDiscreteNet, transfer_supernet_weights
from tdarts.genotype import Genotype, PathGene
from tdarts.mixed_op import TemporalDARTSNet, candidate_names


def test_genotype() -> Genotype:
    genes = []
    indices = {"Low": (4, 9), "Mid": (3, 13), "High": (7, 8)}
    for band, pair in indices.items():
        names = candidate_names(band)
        for path, index in enumerate(pair):
            genes.append(PathGene(band, path, index, names[index], 0.0, 1.0))
    return Genotype(seed=1, epoch=200, genes=tuple(genes))


class DiscreteNetworkTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.genotype = test_genotype()
        self.supernet = TemporalDARTSNet(n_electrodes=2)
        self.discrete = TemporalDiscreteNet(self.genotype, n_electrodes=2)
        self.copied = transfer_supernet_weights(
            self.discrete, {"model_state_dict": self.supernet.state_dict()}
        )

    def test_transfer_covers_every_discrete_state_tensor(self):
        self.assertEqual(self.copied, sorted(self.discrete.state_dict()))

    def test_one_hot_supernet_and_discrete_network_are_identical(self):
        with torch.no_grad():
            for band, cell in self.supernet.cells.items():
                for path_index, path in enumerate(cell.paths):
                    gene = self.genotype.gene(band, path_index)
                    path.alpha.fill_(-60.0)
                    path.alpha[gene.candidate_index] = 60.0
        self.supernet.eval()
        self.discrete.eval()
        x = torch.randn(2, 9, 2, 32)
        with torch.no_grad():
            supernet_logits, supernet_features = self.supernet(x)
            discrete_logits, discrete_features = self.discrete(x)
        self.assertEqual(tuple(discrete_logits.shape), (2, 4))
        self.assertEqual(tuple(discrete_features.shape), tuple(supernet_features.shape))
        self.assertTrue(torch.allclose(supernet_logits, discrete_logits, atol=1e-6, rtol=1e-6))
        self.assertTrue(torch.allclose(supernet_features, discrete_features, atol=1e-6, rtol=1e-6))

    def test_discrete_model_has_no_architecture_logits_or_candidate_pool(self):
        names = [name for name, _ in self.discrete.named_parameters()]
        modules = [name for name, _ in self.discrete.named_modules()]
        self.assertFalse(any(name.endswith("alpha") for name in names))
        self.assertFalse(any(".ops." in name for name in modules))


if __name__ == "__main__":  # pragma: no cover
    unittest.main(verbosity=2)
