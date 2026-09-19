"""Pre-filter-bank frontends and the raw-trial pipeline.

Two things are worth checking hard here, and both are the kind that fail
quietly:

* **The sinc kernels must be at the frequencies they claim.**  A kernel scaled
  by a constant, or normalised at the wrong frequency, still produces a
  plausible-looking ``[B, 9, E, T]`` tensor and still trains -- it just trains on
  the wrong bands.  The frequency response is measured rather than assumed.
* **The raw pipeline must split the data exactly as the existing one does.**
  Two pipelines that are each "80/20 on Session 0" can still disagree, and a
  frequency result is not comparable to earlier stages if they do.
"""

from __future__ import annotations

import unittest
from pathlib import Path

import torch

from tdarts.frontend import (
    BANDS_HZ,
    FRONTENDS,
    FS_HZ,
    PrecomputedFrontend,
    SincFrontend,
    build_frontend,
)

DATA = Path("/gpfs/home/W125221190/FBNAS-master-main/data/bci42a")
RAW_ROOT = DATA / "rawPython"
MULTIVIEW_ROOT = DATA / "multiviewPython"

#: How far below the passband the stopband must sit. The Hamming-windowed
#: design measures about 300x; 100x is a floor that a wrong normalisation,
#: a wrong frequency or a too-short kernel all fall through.
MIN_SUPPRESSION = 100.0


def _response(frontend: SincFrontend, nfft: int = 4096) -> tuple[torch.Tensor, torch.Tensor]:
    spectra = torch.fft.rfft(frontend.kernels().squeeze(1), n=nfft).abs()
    return spectra, torch.fft.rfftfreq(nfft, d=1 / frontend.fs)


class SincResponseTests(unittest.TestCase):
    def test_every_band_passes_unit_gain_at_its_own_centre(self):
        frontend = build_frontend("frozen_sinc")
        spectra, freqs = _response(frontend)
        for index, (low, high) in enumerate(BANDS_HZ):
            centre = (low + high) / 2
            nearest = int((freqs - centre).abs().argmin())
            with self.subTest(band=index):
                self.assertAlmostEqual(float(spectra[index][nearest]), 1.0, places=2)

    def test_each_band_peaks_inside_itself(self):
        frontend = build_frontend("frozen_sinc")
        spectra, freqs = _response(frontend)
        for index, (low, high) in enumerate(BANDS_HZ):
            peak = float(freqs[spectra[index].argmax()])
            with self.subTest(band=index):
                self.assertGreaterEqual(peak, low)
                self.assertLessEqual(peak, high)

    def test_each_band_rejects_everything_outside_itself(self):
        frontend = build_frontend("frozen_sinc")
        spectra, freqs = _response(frontend)
        for index, (low, high) in enumerate(BANDS_HZ):
            inside = (freqs >= low) & (freqs <= high)
            ratio = float(spectra[index][inside].mean() / spectra[index][~inside].mean())
            with self.subTest(band=index):
                self.assertGreater(ratio, MIN_SUPPRESSION)

    def test_the_nine_bands_are_distinct_filters(self):
        kernels = build_frontend("frozen_sinc").kernels()
        self.assertEqual(kernels.shape[0], len(BANDS_HZ))
        for i in range(len(BANDS_HZ)):
            for j in range(i + 1, len(BANDS_HZ)):
                self.assertFalse(torch.allclose(kernels[i], kernels[j]), f"{i} vs {j}")

    def test_a_kernel_too_short_to_resolve_the_bands_is_refused(self):
        """129 taps at 250 Hz has a ~6 Hz transition -- wider than the bands."""

        with self.assertRaises(ValueError):
            SincFrontend(taps=129)
        with self.assertRaises(ValueError):
            SincFrontend(taps=128)  # even
        SincFrontend(taps=401)  # the default must construct


class FrontendInterfaceTests(unittest.TestCase):
    def test_all_three_emit_the_same_shape(self):
        raw = torch.randn(4, 22, 1000)
        precomputed = torch.randn(4, 22, 1000, 9)
        for name in FRONTENDS:
            frontend = build_frontend(name)
            inputs = precomputed if name == "fixed" else raw
            with self.subTest(frontend=name):
                self.assertEqual(tuple(frontend(inputs).shape), (4, 9, 22, 1000))

    def test_only_the_learnable_frontend_has_parameters(self):
        counts = {
            name: sum(p.numel() for p in build_frontend(name).parameters() if p.requires_grad)
            for name in FRONTENDS
        }
        self.assertEqual(counts["fixed"], 0)
        self.assertEqual(counts["frozen_sinc"], 0)
        self.assertEqual(counts["learnable_sinc"], 2 * len(BANDS_HZ))

    def test_learning_reaches_the_band_edges(self):
        frontend = build_frontend("learnable_sinc")
        frontend(torch.randn(2, 22, 500)).sum().backward()
        for name in ("low_hz", "width_hz"):
            gradient = getattr(frontend, name).grad
            self.assertIsNotNone(gradient, name)
            self.assertGreater(float(gradient.abs().sum()), 0.0, name)

    def test_the_learnable_frontend_starts_at_the_upstream_edges(self):
        frontend = build_frontend("learnable_sinc")
        low, high = frontend.edges()
        for index, (expected_low, expected_high) in enumerate(BANDS_HZ):
            self.assertAlmostEqual(float(low[index]), expected_low, places=5)
            self.assertAlmostEqual(float(high[index]), expected_high, places=5)

    def test_bands_cannot_invert_however_the_parameters_move(self):
        """softplus keeps both (low, width) positive, so high > low always."""

        frontend = build_frontend("learnable_sinc")
        with torch.no_grad():
            frontend.low_hz.fill_(-50.0)
            frontend.width_hz.fill_(-50.0)
            low, high = frontend.edges()
            self.assertTrue(bool((high > low).all()))
            self.assertTrue(bool((low >= 0).all()))

    def test_the_wrong_input_shape_is_rejected(self):
        with self.assertRaises(ValueError):
            PrecomputedFrontend()(torch.randn(4, 22, 1000))
        with self.assertRaises(ValueError):
            build_frontend("frozen_sinc")(torch.randn(4, 22, 1000, 9))

    def test_an_unknown_name_is_rejected(self):
        with self.assertRaises(ValueError):
            build_frontend("cheby2")
        with self.assertRaises(ValueError):
            build_frontend("fixed", taps=401)

    def test_the_fixed_frontend_is_a_pure_permutation(self):
        precomputed = torch.randn(3, 22, 1000, 9)
        out = PrecomputedFrontend()(precomputed)
        for band in range(9):
            self.assertTrue(torch.equal(out[:, band], precomputed[..., band]))


@unittest.skipUnless(RAW_ROOT.is_dir(), "raw BCI-IV-2a trials not available")
class RawPipelineTests(unittest.TestCase):
    def test_the_raw_split_matches_the_multiview_split_exactly(self):
        from tdarts.frontend_data import load_raw_session0_split
        from tdarts.search_data import load_session0_search_split

        _, _, raw_split = load_raw_session0_split(RAW_ROOT, "003")
        _, _, multiview_split = load_session0_search_split(MULTIVIEW_ROOT, "003")
        self.assertEqual(raw_split.train_indices, multiview_split.train_indices)
        self.assertEqual(raw_split.val_indices, multiview_split.val_indices)
        self.assertEqual((raw_split.train_size, raw_split.val_size), (231, 57))

    def test_raw_trials_are_two_dimensional_then_nine_banded(self):
        from tdarts.frontend_data import load_raw_session0_split

        train, _, _ = load_raw_session0_split(RAW_ROOT, "003")
        raw, _ = train[0]
        self.assertEqual(tuple(raw.shape), (22, 1000))

        banded, _ = load_raw_session0_split(
            RAW_ROOT, "003", frontend=build_frontend("frozen_sinc")
        )[0][0]
        self.assertEqual(tuple(banded.shape), (9, 22, 1000))
        self.assertFalse(bool(torch.isnan(banded).any()))
        self.assertGreater(float(banded.std()), 0.0)

    def test_pointing_the_raw_loader_at_multiview_data_fails_loudly(self):
        """The two roots are one flag apart and produce different shapes; a
        silent mismatch here would train the frontend on already-filtered bands."""

        from tdarts.frontend_data import load_raw_session0_split

        with self.assertRaises(ValueError):
            load_raw_session0_split(MULTIVIEW_ROOT, "003")[0][0]

    def test_preload_does_not_change_values(self):
        from tdarts.frontend_data import load_raw_session0_split

        frontend = build_frontend("frozen_sinc")
        lazy = load_raw_session0_split(RAW_ROOT, "003", frontend=frontend)[0]
        eager = load_raw_session0_split(RAW_ROOT, "003", frontend=frontend, preload=True)[0]
        for index in (0, 5, len(lazy) - 1):
            self.assertTrue(torch.equal(lazy[index][0], eager[index][0]), index)


if __name__ == "__main__":
    unittest.main()
