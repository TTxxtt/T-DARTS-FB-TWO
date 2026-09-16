"""Compatibility audit: can the new temporal operators replace the official
FBNAS temporal path without touching SCB / LogVar / classifier?

The official baseline is treated as a read-only oracle: its modules are imported
from ``FBNAS/codes/centralRepo`` (with that directory on ``sys.path``, which is
what the upstream code itself expects) and compared against the stage-1
equivalents in :mod:`tdarts.backbone`.

Three things are established:

1. the official temporal cell's output shape and channel width, measured;
2. that width versus the new operators' per-band width;
3. that the new operators' concatenated output drives the *unmodified*
   spatial convolution block, and that the stage-1 ``SpatialConvBlock`` and
   ``LogVarLayer`` are numerically identical to the official ones.

Running directly prints a report and exits non-zero on any incompatibility::

    python tests/test_fbnas_compatibility.py
"""

from __future__ import annotations

import importlib.util
import sys
import traceback
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tdarts import config as C  # noqa: E402
from tdarts.backbone import (  # noqa: E402
    Conv2dWithConstraint as TdartsConv2dWithConstraint,
)
from tdarts.backbone import (  # noqa: E402
    LinearWithConstraint as TdartsLinearWithConstraint,
)
from tdarts.backbone import LogVarLayer as TdartsLogVarLayer  # noqa: E402
from tdarts.backbone import SpatialConvBlock, TemporalBackbone  # noqa: E402
from tdarts.temporal_ops import build_all_candidates  # noqa: E402

FBNAS_CENTRAL_REPO = ROOT / "FBNAS" / "codes" / "centralRepo"
FBNAS_AVAILABLE = False
FBNAS_IMPORT_ERROR: str | None = None


def _load_official_networks():
    """Import the official FBNAS ``networks.py`` as a module.

    The upstream file uses flat imports (``from utils import ...``) and expects
    its own directory on ``sys.path`` -- exactly how ``codes/classify/ho.py``
    sets itself up.  Nothing in the vendored files is modified.

    Bytecode writing is suppressed for the duration: importing from inside
    ``FBNAS/`` otherwise drops a ``networks.cpython-3xx.pyc`` into the frozen
    baseline, which would make this project's own test mutate the very tree it
    is supposed to be verifying.
    """
    central = str(FBNAS_CENTRAL_REPO)
    added = central not in sys.path
    if added:
        sys.path.insert(0, central)

    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        path = FBNAS_CENTRAL_REPO / "networks.py"
        spec = importlib.util.spec_from_file_location("fbnas_networks_official", path)
        module = importlib.util.module_from_spec(spec)
        sys.modules["fbnas_networks_official"] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.dont_write_bytecode = previous


# Import-time suppression only covers the import performed here; the `utils`
# module that `networks.py` pulls in is imported during exec_module above and is
# therefore covered too.
try:
    official = _load_official_networks()
    FBNAS_AVAILABLE = True
except Exception as exc:  # pragma: no cover - environment dependent
    official = None
    FBNAS_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"


def official_scb(in_channels: int, out_channels: int, n_electrodes: int):
    """Instantiate the official SCB exactly as ``FBNASNet.SCB`` builds it.

    The official ``SCB`` is a method on the network classes rather than a
    standalone module, so it is reconstructed here verbatim from
    ``FBNASNet.SCB`` (identical in ``FBCNet``, ``FBMSNet`` and ``SuperNet``).
    """
    import torch.nn as nn

    return nn.Sequential(
        official.Conv2dWithConstraint(
            in_channels,
            out_channels,
            (n_electrodes, 1),
            groups=in_channels,
            max_norm=2,
            doWeightNorm=True,
            padding=0,
        ),
        nn.BatchNorm2d(out_channels),
        official.swish(),
    )


def make_band_input(band_index: int = 0, batch: int = 2) -> torch.Tensor:
    """One Low/Mid/High group: ``[B, 3, 22, 1000]``."""
    del band_index  # all groups have the same shape in this stage
    return torch.randn(
        batch, C.IN_CHANNELS, C.NUM_ELECTRODES, C.NUM_TIMEPOINTS
    )


@unittest.skipUnless(FBNAS_AVAILABLE, f"official FBNAS unavailable: {FBNAS_IMPORT_ERROR}")
class OfficialTemporalPath(unittest.TestCase):
    """Measure what the official temporal cell actually produces."""

    def test_fbnas_cell_output_shape(self):
        f1 = C.NUM_FEAT // 3  # 12, matching official num_Feat=36
        cell = official.FBNASCell(F1=f1, shadow_bn=True, choice=[1, 3])
        cell.eval()
        x = make_band_input()
        with torch.no_grad():
            y = cell(x)
        self.assertEqual(y.dim(), 4)
        self.assertEqual(y.shape[0], 2)
        self.assertEqual(y.shape[2:], (C.NUM_ELECTRODES, C.NUM_TIMEPOINTS))
        self.assertEqual(
            y.shape[1],
            f1,
            "official 2-path cell produces F1 channels per band",
        )

    def test_path_length_one_also_produces_f1_channels(self):
        """Both path lengths land on F1 channels, despite different widths."""
        f1 = C.NUM_FEAT // 3
        for choice in ([0], [2]):
            with self.subTest(choice=choice):
                cell = official.FBNASCell(F1=f1, shadow_bn=True, choice=choice)
                cell.eval()
                with torch.no_grad():
                    y = cell(make_band_input())
                self.assertEqual(y.shape[1], f1)

    def test_official_node_addressing_matches_the_audit(self):
        """Confirm the ``(len(path_ids)-1)*4 + id`` layout recorded earlier.

        For a single-path choice the node is drawn from indices 0-3 with width
        ``F1``; for a two-path choice from indices 4-7 with width ``F1 // 2``.
        Concatenating two width-``F1//2`` paths gives ``F1``.
        """
        f1 = C.NUM_FEAT // 3
        cell = official.FBNASCell(F1=f1, shadow_bn=True, choice=[0, 1])
        self.assertEqual(cell.nodes[0].op[0].out_channels, f1)
        self.assertEqual(cell.nodes[4].op[0].out_channels, f1 // 2)
        self.assertEqual(cell.nodes[5].op[0].out_channels, f1 // 2)
        self.assertEqual(cell.nodes[4].op[0].weight.shape[2:], (1, 15))
        self.assertEqual(cell.nodes[4].op[0].dilation, (1, 1))
        self.assertEqual(cell.nodes[5].op[0].dilation, (1, 2))
        self.assertEqual(cell.nodes[5].op[0].padding, (0, 14))

    def test_official_conv_geometry_table(self):
        """The four dilations and their realised padding, read from the code."""
        expected = {0: 1, 1: 2, 2: 4, 3: 8}
        f1 = C.NUM_FEAT // 3
        cell = official.FBNASCell(F1=f1, shadow_bn=True, choice=[0])
        for idx, dilation in expected.items():
            with self.subTest(node=idx):
                conv = cell.nodes[4 + idx].op[0]  # 2-path group
                self.assertEqual(conv.kernel_size, (1, 15))
                self.assertEqual(conv.dilation, (1, dilation))
                self.assertEqual(conv.padding, (0, 7 * dilation))
                self.assertEqual(conv.groups, 1, "official conv is NOT depthwise")


@unittest.skipUnless(FBNAS_AVAILABLE, f"official FBNAS unavailable: {FBNAS_IMPORT_ERROR}")
class WidthCompatibility(unittest.TestCase):
    """The new per-band width must equal the official per-band width."""

    def test_new_two_path_width_equals_official_band_width(self):
        candidates = build_all_candidates(use_norm=False)
        x = make_band_input()
        with torch.no_grad():
            a = candidates[("Low", "dilated", 57)](x)
            b = candidates[("Low", "lkdw", 113)](x)
        joined = torch.cat([a, b], dim=1)
        official_width = C.NUM_FEAT // 3
        self.assertEqual(
            joined.shape[1],
            official_width,
            "2 paths x PATH_CHANNELS must equal the official per-band width",
        )
        self.assertEqual(
            joined.shape, (2, official_width, C.NUM_ELECTRODES, C.NUM_TIMEPOINTS)
        )

    def test_full_three_band_stack_width_equals_num_feat(self):
        candidates = build_all_candidates(use_norm=False)
        x = make_band_input()
        bands = []
        for band in ("Low", "Mid", "High"):
            with torch.no_grad():
                outs = [
                    candidates[(band, "dilated", 57)](x),
                    candidates[(band, "normal", 29)](x),
                ]
            bands.append(torch.cat(outs, dim=1))
        stacked = torch.cat(bands, dim=1)
        self.assertEqual(
            stacked.shape, (2, C.NUM_FEAT, C.NUM_ELECTRODES, C.NUM_TIMEPOINTS)
        )

    def test_num_feat_derivation_is_consistent(self):
        self.assertEqual(C.NUM_FEAT, 3 * C.NUM_PATHS * C.PATH_CHANNELS)
        self.assertEqual(C.NUM_FEAT, 36)


@unittest.skipUnless(FBNAS_AVAILABLE, f"official FBNAS unavailable: {FBNAS_IMPORT_ERROR}")
class DownstreamUnchanged(unittest.TestCase):
    """The new operators must drive the unmodified SCB, LogVar and classifier."""

    def test_new_operators_feed_spatial_conv_block(self):
        """A new temporal stack must be accepted by SCB with no modification."""
        candidates = build_all_candidates(use_norm=False)
        x = make_band_input()
        bands = []
        for band in ("Low", "Mid", "High"):
            with torch.no_grad():
                bands.append(
                    torch.cat(
                        [
                            candidates[(band, "dilated", 57)](x),
                            candidates[(band, "dwsep", 113)](x),
                        ],
                        dim=1,
                    )
                )
        stacked = torch.cat(bands, dim=1)
        scb = SpatialConvBlock(
            in_channels=C.NUM_FEAT,
            out_channels=C.NUM_FEAT * C.SCB_DILATABILITY,
            n_electrodes=C.NUM_ELECTRODES,
        )
        scb.eval()
        with torch.no_grad():
            out = scb(stacked)
        self.assertEqual(out.shape[1], C.NUM_FEAT * C.SCB_DILATABILITY)  # 288
        self.assertEqual(out.shape[2], 1, "electrode axis must collapse to 1")
        self.assertEqual(out.shape[3], C.NUM_TIMEPOINTS, "time must be preserved")

    def test_scb_is_numerically_equivalent_to_official(self):
        mine = SpatialConvBlock(
            in_channels=C.NUM_FEAT,
            out_channels=C.NUM_FEAT * C.SCB_DILATABILITY,
            n_electrodes=C.NUM_ELECTRODES,
        )
        theirs = official_scb(
            C.NUM_FEAT, C.NUM_FEAT * C.SCB_DILATABILITY, C.NUM_ELECTRODES
        )

        # Copy the FULL state across, bias included.  The official constraint
        # convolutions are built without bias=False, so they do carry a bias;
        # copying only the weight would compare two different functions and
        # report a large spurious difference.
        with torch.no_grad():
            theirs[0].load_state_dict(mine.block[0].state_dict())
            theirs[1].load_state_dict(mine.block[1].state_dict())
        self.assertTrue(
            torch.equal(mine.block[0].weight, theirs[0].weight),
            "weight transfer failed",
        )
        self.assertTrue(
            torch.equal(mine.block[0].bias, theirs[0].bias),
            "bias transfer failed",
        )

        x = torch.randn(2, C.NUM_FEAT, C.NUM_ELECTRODES, C.NUM_TIMEPOINTS)
        mine.eval()
        theirs.eval()
        with torch.no_grad():
            a, b = mine(x), theirs(x)
        self.assertEqual(a.shape, b.shape)
        self.assertTrue(
            torch.allclose(a, b, atol=1e-5, rtol=1e-4),
            f"SCB differs from official; max abs diff "
            f"{(a - b).abs().max().item():.3e}",
        )

    def test_official_constraint_convs_carry_a_bias(self):
        """Record a real difference from the stage-1 operators.

        The official temporal ``ConvBn`` passes ``bias=False``, but the
        official constraint convolutions (SCB) and the ``LastBlock`` linear do
        not, so they have biases.  The stage-1 temporal operators all use
        ``bias=False``.  Neither is changed here; this test just pins the fact
        down so a future refactor cannot silently alter parameter counts.
        """
        scb = SpatialConvBlock(C.NUM_FEAT, 288, C.NUM_ELECTRODES)
        self.assertIsNotNone(scb.block[0].bias)

        # Official temporal conv: bias=False.
        cell = official.FBNASCell(F1=C.NUM_FEAT // 3, shadow_bn=True, choice=[0])
        self.assertIsNone(
            cell.nodes[4].op[0].bias,
            "official temporal ConvBn is expected to have bias=False",
        )

        # Stage-1 temporal operators: bias=False for every family.
        from tdarts.temporal_ops import build_all_candidates

        for key, cand in build_all_candidates(use_norm=False).items():
            for name, module in cand.named_modules():
                if isinstance(module, torch.nn.Conv2d):
                    with self.subTest(candidate=key, module=name):
                        self.assertIsNone(module.bias)

    def test_official_scb_weight_is_renormed_but_ours_matches(self):
        """The constraint renorm is a no-op at init, and behaves identically."""
        mine = SpatialConvBlock(C.NUM_FEAT, 288, C.NUM_ELECTRODES)
        theirs = official_scb(C.NUM_FEAT, 288, C.NUM_ELECTRODES)
        with torch.no_grad():
            theirs[0].load_state_dict(mine.block[0].state_dict())

        x = torch.randn(1, C.NUM_FEAT, C.NUM_ELECTRODES, C.NUM_TIMEPOINTS)
        mine.eval()
        theirs.eval()
        with torch.no_grad():
            mine(x)
            theirs(x)
        self.assertTrue(
            torch.equal(mine.block[0].weight, theirs[0].weight),
            "post-forward weights must be identical",
        )

    def test_logvar_matches_official(self):
        mine = TdartsLogVarLayer(dim=3)
        theirs = official.LogVarLayer(dim=3)
        x = torch.rand(2, 8, 8, 125) * 10.0 + 1.0
        self.assertTrue(torch.equal(mine(x), theirs(x)))

    def test_logvar_clamp_bounds_are_preserved(self):
        """A near-constant segment must still clamp rather than reach -inf."""
        layer = TdartsLogVarLayer(dim=3)
        x = torch.ones(1, 2, 4, 8)
        out = layer(x)
        self.assertTrue(torch.isfinite(out).all())
        self.assertAlmostEqual(float(out.max()), float(torch.log(torch.tensor(1e-6))), places=5)

    def test_constraint_layers_match_official_geometry(self):
        mine = TdartsConv2dWithConstraint(
            C.NUM_FEAT, 288, (C.NUM_ELECTRODES, 1), groups=C.NUM_FEAT,
            max_norm=2, padding=0,
        )
        theirs = official.Conv2dWithConstraint(
            C.NUM_FEAT, 288, (C.NUM_ELECTRODES, 1), groups=C.NUM_FEAT,
            max_norm=2, padding=0,
        )
        self.assertEqual(mine.weight.shape, theirs.weight.shape)
        self.assertEqual(mine.groups, theirs.groups)
        self.assertEqual(mine.max_norm, theirs.max_norm)
        self.assertEqual(mine.kernel_size, theirs.kernel_size)

        lin_mine = TdartsLinearWithConstraint(2304, 4, max_norm=0.5)
        lin_theirs = official.LinearWithConstraint(2304, 4, max_norm=0.5)
        self.assertEqual(lin_mine.weight.shape, lin_theirs.weight.shape)
        self.assertEqual(lin_mine.max_norm, lin_theirs.max_norm)

    def test_full_pipeline_end_to_end(self):
        """New temporal ops -> unchanged backbone -> [B, 4] log-probabilities."""
        candidates = build_all_candidates(use_norm=False)
        backbone = TemporalBackbone()
        backbone.eval()
        x = make_band_input(batch=2)
        bands = []
        for band in ("Low", "Mid", "High"):
            with torch.no_grad():
                bands.append(
                    torch.cat(
                        [
                            candidates[(band, "dilated", 29)](x),
                            candidates[(band, "lkdw", 57)](x),
                        ],
                        dim=1,
                    )
                )
        stacked = torch.cat(bands, dim=1)
        with torch.no_grad():
            logits, features = backbone(stacked)
        self.assertEqual(tuple(logits.shape), (2, C.NUM_CLASSES))
        self.assertEqual(features.shape[1], backbone.feature_dim)
        self.assertEqual(backbone.feature_dim, 2304)
        self.assertTrue(torch.isfinite(logits).all())
        # LogSoftmax output: exponentiated probabilities must sum to 1.
        self.assertTrue(
            torch.allclose(logits.exp().sum(dim=1), torch.ones(2), atol=1e-5)
        )

    def test_backbone_shape_chain_matches_the_documented_flow(self):
        backbone = TemporalBackbone()
        backbone.eval()
        x = torch.randn(2, C.NUM_FEAT, C.NUM_ELECTRODES, C.NUM_TIMEPOINTS)
        with torch.no_grad():
            scb_out = backbone.scb(x)
        self.assertEqual(tuple(scb_out.shape), (2, 288, 1, 1000))
        reshaped = scb_out.reshape(2, 288, C.STRIDEFACTOR, 1000 // C.STRIDEFACTOR)
        self.assertEqual(tuple(reshaped.shape), (2, 288, 8, 125))
        with torch.no_grad():
            agg = backbone.temporal_layer(reshaped)
        self.assertEqual(tuple(agg.shape), (2, 288, 8, 1))
        self.assertEqual(agg.numel() // 2, 2304)

    def test_time_length_must_be_divisible_by_stridefactor(self):
        backbone = TemporalBackbone()
        x = torch.randn(1, C.NUM_FEAT, C.NUM_ELECTRODES, 999)
        with self.assertRaises(ValueError) as ctx:
            backbone(x)
        self.assertIn("divisible", str(ctx.exception))


class BaselineIntegrity(unittest.TestCase):
    """The vendored baseline must be present and unmodified."""

    def test_manifest_says_complete(self):
        import json

        manifest = json.loads(
            (ROOT / "FBNAS" / "_MANIFEST.json").read_text(encoding="utf-8")
        )
        self.assertTrue(manifest["complete"], "mirror is not marked complete")
        self.assertEqual(manifest["failed"], [])
        self.assertEqual(manifest["skipped"], [])
        self.assertEqual(manifest["saved"], manifest["total_blobs"])

    def test_expected_official_files_exist(self):
        for rel in (
            "FBNAS/requirements.txt",
            "FBNAS/codes/centralRepo/networks.py",
            "FBNAS/codes/centralRepo/NAS.py",
            "FBNAS/codes/centralRepo/eegDataset.py",
            "FBNAS/codes/centralRepo/stopCriteria.py",
            "FBNAS/codes/centralRepo/utils.py",
            "FBNAS/codes/classify/ho.py",
        ):
            with self.subTest(path=rel):
                self.assertTrue((ROOT / rel).is_file(), f"missing {rel}")

    def test_official_files_are_byte_identical_to_the_manifest(self):
        """Guard against anyone editing the frozen baseline in place."""
        import hashlib
        import json

        manifest = json.loads(
            (ROOT / "FBNAS" / "_MANIFEST.json").read_text(encoding="utf-8")
        )
        mismatches = []
        for entry in manifest["blobs"]:
            # Manifest paths are POSIX/git paths even when tests run on
            # Windows.  Path() converts them correctly on every platform;
            # replacing with a literal backslash makes every file appear
            # missing on POSIX hosts.
            path = ROOT / "FBNAS" / Path(entry["path"])
            if not path.is_file():
                mismatches.append(f"missing: {entry['path']}")
                continue
            # Git blob SHA-1 = sha1(b"blob <len>\0" + content)
            raw = path.read_bytes()
            header = f"blob {len(raw)}\0".encode()
            sha = hashlib.sha1(header + raw).hexdigest()
            if sha != entry["sha"]:
                mismatches.append(f"modified: {entry['path']}")
        self.assertEqual(mismatches, [], f"baseline altered: {mismatches}")

    def test_no_extra_files_were_written_into_the_baseline(self):
        """The baseline must contain exactly the upstream blobs.

        Importing the official modules puts ``FBNAS/codes/centralRepo`` on
        ``sys.path``, and Python would normally drop ``.pyc`` files next to the
        sources -- the project's own tests mutating the frozen tree.  Bytecode
        writing is suppressed while importing; this asserts it stays clean.
        """
        import json

        manifest = json.loads(
            (ROOT / "FBNAS" / "_MANIFEST.json").read_text(encoding="utf-8")
        )
        expected = {str(Path(b["path"])) for b in manifest["blobs"]}
        expected.add("_MANIFEST.json")

        actual = {
            str(p.relative_to(ROOT / "FBNAS"))
            for p in (ROOT / "FBNAS").rglob("*")
            if p.is_file()
        }
        extra = sorted(actual - expected)
        self.assertEqual(
            extra,
            [],
            "unexpected files inside the frozen baseline (bytecode pollution?)",
        )

    def test_import_did_not_write_bytecode_into_the_baseline(self):
        """Explicitly assert no .pyc appeared under centralRepo."""
        cache = FBNAS_CENTRAL_REPO / "__pycache__"
        if not cache.is_dir():
            return
        offenders = [
            p.name
            for p in cache.iterdir()
            if p.suffix == ".pyc" and "cpython-313" in p.name
        ]
        self.assertEqual(
            offenders,
            [],
            f"this interpreter's bytecode leaked into the baseline: {offenders}",
        )


def run_report() -> int:
    """Print the compatibility report and return an exit code."""
    print("FBNAS Compatibility Audit")
    print("-" * 78)

    ok = True

    if not FBNAS_AVAILABLE:
        print(f"official FBNAS import FAILED: {FBNAS_IMPORT_ERROR}")
        return 1

    candidates = build_all_candidates(use_norm=False)
    x = make_band_input()

    f1 = C.NUM_FEAT // 3
    cell = official.FBNASCell(F1=f1, shadow_bn=True, choice=[0, 1])
    cell.eval()
    with torch.no_grad():
        off_out = cell(x)
    print(f"official FBNASCell(2 paths) output : {tuple(off_out.shape)}")

    with torch.no_grad():
        new_a = candidates[("Low", "dilated", 57)](x)
        new_b = candidates[("Low", "lkdw", 113)](x)
        new_join = torch.cat([new_a, new_b], dim=1)
    print(f"new operator output (1 path)       : {tuple(new_a.shape)}")
    print(f"new 2-path concat                  : {tuple(new_join.shape)}")

    same = tuple(off_out.shape) == tuple(new_join.shape)
    print(f"per-band shape match               : {'YES' if same else 'NO'}")
    if not same:
        ok = False
        for name, a, b in zip(("batch", "channels", "electrodes", "time"),
                              off_out.shape, new_join.shape):
            print(f"  {name:<10} official={a} new={b} {'OK' if a == b else 'DIFFERENT'}")

    scb = SpatialConvBlock(C.NUM_FEAT, C.NUM_FEAT * C.SCB_DILATABILITY, C.NUM_ELECTRODES)
    scb.eval()
    with torch.no_grad():
        scb_out = scb(torch.cat([new_join, new_join, new_join], dim=1))
    expected_scb = (2, C.NUM_FEAT * C.SCB_DILATABILITY, 1, C.NUM_TIMEPOINTS)
    scb_ok = tuple(scb_out.shape) == expected_scb
    print(f"new ops -> unmodified SCB          : {tuple(scb_out.shape)} "
          f"{'OK' if scb_ok else 'MISMATCH'}")
    ok = ok and scb_ok

    backbone = TemporalBackbone()
    backbone.eval()
    with torch.no_grad():
        logits, feats = backbone(torch.cat([new_join] * 3, dim=1))
    print(f"new ops -> full backbone           : logits {tuple(logits.shape)} "
          f"features {tuple(feats.shape)}")
    ok = ok and tuple(logits.shape) == (2, C.NUM_CLASSES) and feats.shape[1] == 2304

    print()
    print("COMPATIBLE" if ok else "INCOMPATIBLE")
    return 0 if ok else 1


def main() -> int:
    # Silence the unittest run's own output; we want the report only.
    buffer = []
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(stream=open(__import__("os").devnull, "w"), verbosity=0)
    result = runner.run(suite)

    code = run_report()
    if not result.wasSuccessful():
        print()
        print("test failures:")
        for case, tb in list(result.failures) + list(result.errors):
            print(f"  FAIL {case}")
            print("      " + tb.strip().replace("\n", "\n      "))
        return 1
    return code


if __name__ == "__main__":
    raise SystemExit(main())
