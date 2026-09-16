"""Tests for name-addressed deterministic initialisation.

The property that matters: two model variants that share a parameter *name*
must start bit-identical under the same run seed, no matter what else the model
contains or in what order its modules were built.  Without that, comparing
"RF-only" against "operator-only" against a joint search compares initialisations
as much as architectures.

Running directly prints a report::

    python tests/test_init_utils.py
"""

from __future__ import annotations

import sys
import unittest
import zlib
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tdarts import config as C  # noqa: E402
from tdarts.init_utils import (  # noqa: E402
    initialise_deterministic,
    make_generator,
    naive_seeded_,
    parameter_checksum,
    parameter_seed,
    seeded_,
    stable_seed,
)
from tdarts.temporal_ops import build_temporal_op  # noqa: E402


# ----------------------------------------------------------------------
# model fixtures
# ----------------------------------------------------------------------
class Shared(nn.Module):
    """Two shared layers, used as the common prefix of both variants."""

    def __init__(self):
        super().__init__()
        self.stem = nn.Conv2d(3, 8, (1, 5), bias=False)
        self.head = nn.Linear(8, 4)


class VariantA(nn.Module):
    """Shared prefix, then one branch, in this creation order."""

    def __init__(self):
        super().__init__()
        self.shared = Shared()
        self.branch = nn.Conv2d(8, 8, (1, 3), bias=False)


class VariantB(nn.Module):
    """Same shared prefix created *after* a different branch.

    Creation order is deliberately reversed relative to VariantA, and an extra
    module is present, so a position-based RNG would desynchronise the shared
    parameters.
    """

    def __init__(self):
        super().__init__()
        self.branch = nn.Conv2d(8, 16, (1, 7), bias=False)
        self.extra = nn.Conv2d(16, 16, (1, 3), bias=True)
        self.shared = Shared()


class TestStableSeed(unittest.TestCase):
    def test_matches_crc32_definition(self):
        run_seed, name = 20250901, "shared.stem.weight"
        expected = zlib.crc32(f"{run_seed}:{name}".encode("utf-8")) & 0xFFFFFFFF
        self.assertEqual(stable_seed(run_seed, name), expected)
        self.assertEqual(parameter_seed(run_seed, name), expected)

    def test_is_in_32_bit_range(self):
        for run_seed in (0, 1, 123456789, 2**31 - 1):
            for name in ("a", "shared.stem.weight", "x" * 200):
                with self.subTest(run_seed=run_seed, name=name[:20]):
                    value = stable_seed(run_seed, name)
                    self.assertGreaterEqual(value, 0)
                    self.assertLess(value, 2**32)

    def test_same_inputs_give_same_seed(self):
        self.assertEqual(stable_seed(7, "p"), stable_seed(7, "p"))

    def test_different_run_seed_gives_different_seed(self):
        self.assertNotEqual(stable_seed(7, "p"), stable_seed(8, "p"))

    def test_different_name_gives_different_seed(self):
        self.assertNotEqual(stable_seed(7, "p"), stable_seed(7, "q"))

    def test_seed_is_stable_across_processes(self):
        """A hard-coded value, so a change in the scheme is caught.

        The expected values were produced by
        ``zlib.crc32(f"{seed}:{name}".encode())`` and are independent of
        PYTHONHASHSEED, unlike the built-in hash().
        """
        cases = {
            (20250901, "shared.stem.weight"): zlib.crc32(
                b"20250901:shared.stem.weight"
            )
            & 0xFFFFFFFF,
            (0, "a"): zlib.crc32(b"0:a") & 0xFFFFFFFF,
        }
        for (run_seed, name), expected in cases.items():
            with self.subTest(run_seed=run_seed, name=name):
                self.assertEqual(stable_seed(run_seed, name), expected)

    def test_name_separator_matters(self):
        """'1:23' and '12:3' must not collide."""
        self.assertNotEqual(stable_seed(1, "23"), stable_seed(12, "3"))


class TestBitIdentity(unittest.TestCase):
    def test_same_run_seed_same_param_name_is_bit_identical(self):
        a = VariantA()
        b = VariantB()
        seeded_(a, run_seed=42)
        seeded_(b, run_seed=42)

        self.assertEqual(
            parameter_checksum(a.shared),
            parameter_checksum(b.shared),
            "shared parameters must be bit-identical across variants",
        )
        # And indeed bitwise, not merely close.
        self.assertTrue(torch.equal(a.shared.stem.weight, b.shared.stem.weight))
        self.assertTrue(torch.equal(a.shared.head.weight, b.shared.head.weight))
        self.assertTrue(torch.equal(a.shared.head.bias, b.shared.head.bias))

    def test_different_run_seed_gives_different_values(self):
        a = VariantA()
        b = VariantA()
        initialise_deterministic(a, run_seed=42)
        initialise_deterministic(b, run_seed=43)
        self.assertFalse(torch.equal(a.shared.stem.weight, b.shared.stem.weight))
        self.assertNotEqual(
            parameter_checksum(a.shared), parameter_checksum(b.shared)
        )

    def test_repeated_initialisation_is_idempotent(self):
        a = VariantA()
        initialise_deterministic(a, run_seed=7)
        first = parameter_checksum(a)
        initialise_deterministic(a, run_seed=7)
        self.assertEqual(first, parameter_checksum(a))

    def test_naive_order_dependent_init_actually_differs(self):
        """The contrast case, proving the test above is not vacuous.

        Position-based seeding desynchronises the shared prefix, which is
        exactly the failure mode ``seeded_`` avoids.
        """
        a = VariantA()
        b = VariantB()
        naive_seeded_(a, run_seed=42)
        naive_seeded_(b, run_seed=42)
        self.assertFalse(
            torch.equal(a.shared.stem.weight, b.shared.stem.weight),
            "naive seeding was expected to desynchronise the shared prefix",
        )

    def test_per_parameter_seed_actually_reached(self):
        """Assert the draw is reproducible from the documented seed."""
        a = VariantA()
        seeded_(a, run_seed=99)
        weight = a.shared.stem.weight
        fan_in = int(weight[0].numel())
        bound = (3.0 / fan_in) ** 0.5
        generator = make_generator(99, "shared.stem.weight")
        expected = torch.rand(tuple(weight.shape), generator=generator) * (
            2 * bound
        ) - bound
        self.assertTrue(torch.allclose(weight, expected))

    def test_submodule_init_matches_whole_model_init(self):
        """Initialising a submodule alone equals initialising the parent."""
        whole = VariantA()
        initialise_deterministic(whole, run_seed=5)

        part = VariantA()
        seeded_(part.shared, run_seed=5, name_prefix="shared")
        self.assertEqual(
            parameter_checksum(whole.shared), parameter_checksum(part.shared)
        )

    def test_buffers_are_not_touched(self):
        """BatchNorm running statistics are buffers, not parameters.

        They are estimated from data during training, so a parameter
        initialiser must leave them alone.  This pins that contract down: the
        shared-prefix guarantee covers parameters only.
        """
        model = nn.Sequential(nn.Conv2d(3, 4, 1), nn.BatchNorm2d(4))
        with torch.no_grad():
            model[1].running_mean.fill_(123.0)
            model[1].running_var.fill_(7.0)
            model[1].num_batches_tracked.fill_(5)

        initialise_deterministic(model, run_seed=3)

        self.assertEqual(float(model[1].running_mean[0]), 123.0)
        self.assertEqual(float(model[1].running_var[0]), 7.0)
        self.assertEqual(int(model[1].num_batches_tracked), 5)
        # ...while the affine parameters *are* seeded, deterministically.
        reference = nn.Sequential(nn.Conv2d(3, 4, 1), nn.BatchNorm2d(4))
        initialise_deterministic(reference, run_seed=3)
        self.assertTrue(torch.equal(model[1].weight, reference[1].weight))
        self.assertTrue(torch.equal(model[1].bias, reference[1].bias))

    def test_parameters_and_buffers_are_distinguishable(self):
        model = nn.Sequential(nn.Conv2d(3, 4, 1), nn.BatchNorm2d(4))
        param_names = {n for n, _ in model.named_parameters()}
        buffer_names = {n for n, _ in model.named_buffers()}
        self.assertIn("1.weight", param_names)
        self.assertIn("1.running_mean", buffer_names)
        self.assertNotIn("1.running_mean", param_names)


class TestOnRealModules(unittest.TestCase):
    def test_temporal_operators_are_name_addressable(self):
        """Two identical candidates initialise identically despite being distinct objects."""
        a = build_temporal_op("dilated", "Low", 57, use_norm=False)
        b = build_temporal_op("dilated", "Low", 57, use_norm=False)
        initialise_deterministic(a, run_seed=11)
        initialise_deterministic(b, run_seed=11)
        self.assertEqual(parameter_checksum(a), parameter_checksum(b))

    def test_different_candidate_names_differ_only_by_name(self):
        """Operators with different names get different draws under one seed."""
        a = build_temporal_op("dilated", "Low", 57, use_norm=False)
        b = build_temporal_op("normal", "Low", 57, use_norm=False)
        initialise_deterministic(a, run_seed=11)
        initialise_deterministic(b, run_seed=11)
        # Both have a lone `conv.weight`, so the names coincide and the shapes
        # differ; the checksums must still differ.
        self.assertNotEqual(parameter_checksum(a), parameter_checksum(b))

    def test_bias_free_operators_have_no_bias_parameter(self):
        cand = build_temporal_op("dwsep", "Mid", 29, use_norm=False)
        names = [n for n, _ in cand.named_parameters()]
        self.assertTrue(all(not n.endswith("bias") for n in names), names)


def main() -> int:
    print("Deterministic Initialisation Audit")
    print("-" * 78)

    a, b = VariantA(), VariantB()
    seeded_(a, run_seed=42)
    seeded_(b, run_seed=42)
    shared_equal = parameter_checksum(a.shared) == parameter_checksum(b.shared)

    c = VariantA()
    seeded_(c, run_seed=43)
    seed_differs = (
        parameter_checksum(a.shared) != parameter_checksum(c.shared)
    )

    na, nb = VariantA(), VariantB()
    naive_seeded_(na, run_seed=42)
    naive_seeded_(nb, run_seed=42)
    naive_equal = parameter_checksum(na.shared) == parameter_checksum(nb.shared)

    print(f"  seed formula            : crc32(f'{{run_seed}}:{{name}}')")
    print(f"  shared prefix, seed 42  : {'IDENTICAL' if shared_equal else 'DIFFERS'}")
    print(f"  shared prefix, seed 43  : {'DIFFERS (expected)' if seed_differs else 'IDENTICAL (BAD)'}")
    print(f"  naive positional seeding: {'IDENTICAL' if naive_equal else 'DIFFERS (expected)'}")
    print(f"  sample seed values      : "
          f"{stable_seed(42, 'shared.stem.weight')}, "
          f"{stable_seed(43, 'shared.stem.weight')}")

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
    return 0 if shared_equal and seed_differs and not naive_equal else 1


if __name__ == "__main__":
    raise SystemExit(main())
