"""Pre-filter-bank frontends: raw EEG -> the nine filter-bank bands.

Everything upstream of this project takes the nine bands as given -- the
dataset ships them precomputed by the official ``filterBank`` transform, a
Chebyshev-type-II IIR bank of nine 4 Hz bands from 4 to 40 Hz
(``FBNAS/codes/centralRepo/saveData.py:477``).  This module makes that stage
swappable and asks whether *which bands you look at* is worth searching, the
same question the temporal work asked about *how you scan them*.

Three frontends, all emitting ``[B, 9, E, T]`` so the downstream stage is
unchanged::

    raw [B, E, T]
      ├─ fixed          the upstream Cheby2 bank, taken from the dataset as-is
      ├─ frozen_sinc    nine sinc band-passes at the upstream edges, frozen
      └─ learnable_sinc the same, with the band edges learned

``frozen_sinc`` exists to separate two things a ``fixed`` vs ``learnable`` pair
would confound: it shares ``learnable_sinc``'s parameterisation while sharing
``fixed``'s immobility.  So ``fixed`` vs ``frozen_sinc`` isolates the
parameterisation, and ``frozen_sinc`` vs ``learnable_sinc`` isolates the
learning.

The band edges are the upstream ones on purpose.  A learnable frontend that
starts somewhere else would be tested against a different reference, and the
question here is whether *learning* the bands helps -- not whether some other
fixed choice would have.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "BANDS_HZ",
    "FS_HZ",
    "FRONTENDS",
    "SincFrontend",
    "PrecomputedFrontend",
    "build_frontend",
]

#: The upstream bank: nine contiguous 4 Hz bands, 4-40 Hz.
BANDS_HZ: tuple[tuple[float, float], ...] = (
    (4, 8), (8, 12), (12, 16), (16, 20), (20, 24),
    (24, 28), (28, 32), (32, 36), (36, 40),
)

#: BCI-IV-2a is sampled at 250 Hz; the trial is 1000 samples = 4 s.
FS_HZ = 250

FRONTENDS = ("fixed", "frozen_sinc", "learnable_sinc")


def _inverse_softplus(value: torch.Tensor) -> torch.Tensor:
    """``x`` such that ``softplus(x) == value``, for positive ``value``.

    ``log(expm1(v))`` rather than ``log(exp(v) - 1)`` so the small-value end
    keeps its precision.
    """

    return torch.log(torch.expm1(value))


class SincFrontend(nn.Module):
    """Nine windowed-sinc band-pass filters applied along time.

    An ideal band-pass has impulse response ``2*f2*sinc(2*f2*t) - 2*f1*sinc(2*f1*t)``;
    windowing it with a Hamming window trades the ideal brick wall for a finite
    kernel.  The same kernel is applied to every electrode, so this is a filter
    bank and not a spatial operation.

    With ``learnable`` the *edges* move, but each kernel is renormalised to unit
    gain at its own centre frequency.  Without that, the cheapest way to reduce
    the loss would be to scale a band's amplitude, which is not what "learning
    which frequencies matter" should mean.
    """

    def __init__(
        self,
        *,
        fs: float = FS_HZ,
        bands=BANDS_HZ,
        taps: int = 401,
        learnable: bool = False,
    ) -> None:
        super().__init__()
        if taps < 3 or taps % 2 == 0:
            raise ValueError(f"taps must be an odd number >= 3, got {taps}")
        self.fs = float(fs)
        self.taps = int(taps)
        self.learnable = bool(learnable)
        self.band_count = len(bands)
        # A Hamming window's transition width is about 3.3 * fs / taps.  These
        # bands are contiguous and 4 Hz wide, so a transition as wide as the
        # band would leave neighbouring filters crossing -6 dB at the shared
        # edge; anything wider and the "nine bands" stop being nine bands.
        narrowest = min(hi - lo for lo, hi in bands)
        if 3.3 * self.fs / taps >= narrowest:
            raise ValueError(
                f"taps={taps} is too few: the transition width "
                f"({3.3 * self.fs / taps:.1f} Hz) is as wide as the narrowest band "
                f"({narrowest:.1f} Hz), so the bands would overlap instead of resolving"
            )

        low = torch.tensor([lo for lo, _ in bands], dtype=torch.float32)
        high = torch.tensor([hi for _, hi in bands], dtype=torch.float32)
        if learnable:
            # Parameterise (low, width) rather than (low, high): softplus keeps
            # both positive, so the bands can never invert or fold through zero.
            # Invert it at construction so the first forward pass sees exactly
            # the upstream edges -- softplus(4.0) is 4.018, and a learnable
            # frontend that starts off-band is compared against a frozen one
            # that does not, which is not the comparison being made.
            self.low_hz = nn.Parameter(_inverse_softplus(low))
            self.width_hz = nn.Parameter(_inverse_softplus(high - low))
        else:
            self.register_buffer("low_hz", low)
            self.register_buffer("width_hz", high - low)

    def edges(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.learnable:
            low = F.softplus(self.low_hz)
            width = F.softplus(self.width_hz)
        else:
            low, width = self.low_hz, self.width_hz
        return low, low + width

    def kernels(self) -> torch.Tensor:
        """``[9, 1, taps]`` band-pass kernels, each of unit centre gain."""

        low, high = self.edges()
        n = torch.arange(self.taps, dtype=torch.float32, device=low.device) - (self.taps - 1) / 2
        t = n / self.fs
        # 2*f*sinc(2*f*t) is the ideal low-pass with cutoff f.
        kernel = 2 * high[:, None] * torch.sinc(2 * high[:, None] * t) - \
                 2 * low[:, None] * torch.sinc(2 * low[:, None] * t)
        kernel = kernel * torch.hamming_window(self.taps, device=low.device)
        # DTFT at the band centre f = (low + high) / 2:
        #   sum_n h[n] exp(-j 2 pi f n / fs) = sum_n h[n] cos(pi (low+high) n / fs)
        # The kernel is symmetric, so the imaginary part cancels and cos is
        # enough.  Dropping the /fs here silently normalises at the wrong
        # frequency, which leaves every kernel miscaled by a large factor.
        centre = math.pi * (low + high)[:, None] * n / self.fs
        gain = (kernel * torch.cos(centre)).sum(dim=1, keepdim=True)
        return (kernel / gain.clamp_min(1e-8)).unsqueeze(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``[B, E, T]`` -> ``[B, 9, E, T]``."""

        if x.dim() != 3:
            raise ValueError(f"SincFrontend expects [B, E, T], got {tuple(x.shape)}")
        batch, electrodes, time = x.shape
        flat = x.reshape(batch * electrodes, 1, time)
        out = F.conv1d(flat, self.kernels(), padding=self.taps // 2)
        return out.reshape(batch, electrodes, self.band_count, time).permute(0, 2, 1, 3)

    def describe(self) -> dict:
        low, high = self.edges()
        return {
            "frontend": "learnable_sinc" if self.learnable else "frozen_sinc",
            "taps": self.taps,
            "fs": self.fs,
            "learnable": self.learnable,
            "bands_hz": [[float(a), float(b)] for a, b in zip(low.detach(), high.detach())],
            "parameters": sum(p.numel() for p in self.parameters() if p.requires_grad),
        }


class PrecomputedFrontend(nn.Module):
    """The upstream Cheby2 bank, taken from the dataset rather than applied here.

    The ``multiviewPython`` trials are already ``[E, T, 9]`` because the official
    transform filtered them.  Re-deriving that with :mod:`scipy` would be a
    second implementation of the reference -- and any disagreement between the
    two would be a difference between filter *implementations*, not between
    frontends.  So the reference is whatever the dataset already contains.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``[B, E, T, 9]`` (already filtered) -> ``[B, 9, E, T]``."""

        if x.dim() != 4 or x.shape[-1] != len(BANDS_HZ):
            raise ValueError(
                f"PrecomputedFrontend expects [B, E, T, {len(BANDS_HZ)}], got {tuple(x.shape)}"
            )
        return x.permute(0, 3, 1, 2)

    def describe(self) -> dict:
        return {
            "frontend": "fixed",
            "filter": "cheby2 IIR, nine 4 Hz bands 4-40 Hz (upstream filterBank transform)",
            "bands_hz": [list(band) for band in BANDS_HZ],
            "parameters": 0,
        }


def build_frontend(name: str, **kwargs) -> nn.Module:
    """``fixed`` | ``frozen_sinc`` | ``learnable_sinc``."""

    if name == "fixed":
        if kwargs:
            raise ValueError(f"the fixed frontend takes no arguments, got {sorted(kwargs)}")
        return PrecomputedFrontend()
    if name == "frozen_sinc":
        return SincFrontend(learnable=False, **kwargs)
    if name == "learnable_sinc":
        return SincFrontend(learnable=True, **kwargs)
    raise ValueError(f"unknown frontend {name!r}; expected one of {FRONTENDS}")
