"""Fixed backbone downstream of the temporal cells.

This is the part of FBNAS that stage 1 deliberately does **not** change: the
spatial convolution block, the log-variance temporal aggregator, and the
classifier.  A later stage replaces only the temporal operator; everything in
this file must keep working untouched.

Data flow for BCI-IV-2a (``B`` = batch):

    temporal cells   [B, 36, 22, 1000]     3 bands x 12 channels
      -> SCB         [B, 288, 1, 1000]     36 -> 288, electrode axis collapsed
      -> reshape     [B, 288, 8, 125]
      -> LogVar      [B, 288, 8, 1]
      -> flatten     [B, 2304]
      -> classifier  [B, 4]                log-softmax

Source attribution
------------------
``Conv2dWithConstraint``, ``LinearWithConstraint``, ``swish``, ``LogVarLayer``,
the ``SCB`` layout and the ``LastBlock`` layout are adapted from the official
FBNAS implementation (``codes/centralRepo/networks.py`` in
wang1239435478/FBNAS-master, ``FBCNet``/``FBMSNet``/``FBNASNet``), which itself
derives the constraint layers from FBCNet.  The mathematical behaviour is
intentionally unchanged.

What is *not* copied: ``FBNASCell`` and its
``nodes[(len(path_ids) - 1) * n_ops + offset]`` addressing, the ``SuperNet``
wrapper, ``MixedConv2d`` and the FBMSNet variants.  Those are baseline-only; new
code must not acquire a dependency on the path/node mechanism.

Upstream note: the original ``Conv2dWithConstraint`` / ``LinearWithConstraint``
renormalise ``self.weight.data`` inside ``forward`` with no ``torch.no_grad()``
guard.  That is preserved here for behavioural fidelity, but the resulting
autograd warning is suppressed explicitly so it is a known, local decision
rather than ambient noise.
"""

from __future__ import annotations

import warnings

import torch
import torch.nn as nn

from tdarts import config as C

__all__ = [
    "Conv2dWithConstraint",
    "LinearWithConstraint",
    "Swish",
    "LogVarLayer",
    "SpatialConvBlock",
    "TemporalClassifier",
    "TemporalBackbone",
    "backbone_feature_dim",
]


# ----------------------------------------------------------------------
# adapted from official FBNAS networks.py
# ----------------------------------------------------------------------
class Conv2dWithConstraint(nn.Conv2d):
    """``nn.Conv2d`` whose weight rows are renormalised to ``max_norm``.

    Adapted from official FBNAS.  ``doWeightNorm`` is applied in ``forward``
    exactly as upstream does it.
    """

    def __init__(self, *args, doWeightNorm: bool = True, max_norm: float = 1, **kwargs):
        self.max_norm = max_norm
        self.doWeightNorm = doWeightNorm
        super().__init__(*args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.doWeightNorm:
            with warnings.catch_warnings():
                # Upstream mutates .data inside forward. Keep the behaviour,
                # silence the repeated autograd notice.
                warnings.filterwarnings(
                    "ignore",
                    message=".*non-full backward hook.*|.*CopySlices.*",
                )
                self.weight.data = torch.renorm(
                    self.weight.data, p=2, dim=0, maxnorm=self.max_norm
                )
        return super().forward(x)


class LinearWithConstraint(nn.Linear):
    """``nn.Linear`` whose weight rows are renormalised to ``max_norm``.

    Adapted from official FBNAS.
    """

    def __init__(self, *args, doWeightNorm: bool = True, max_norm: float = 1, **kwargs):
        self.max_norm = max_norm
        self.doWeightNorm = doWeightNorm
        super().__init__(*args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.doWeightNorm:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=".*non-full backward hook.*|.*CopySlices.*",
                )
                self.weight.data = torch.renorm(
                    self.weight.data, p=2, dim=0, maxnorm=self.max_norm
                )
        return super().forward(x)


class Swish(nn.Module):
    """``x * sigmoid(x)``.  Adapted from official FBNAS ``swish``."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


class LogVarLayer(nn.Module):
    """Log-variance along ``dim`` with the upstream clamp.

    Adapted from official FBNAS::

        torch.log(torch.clamp(x.var(dim=self.dim, keepdim=True), 1e-6, 1e6))

    The clamp bounds matter -- without them a near-constant segment would drive
    the log to -inf.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.log(torch.clamp(x.var(dim=self.dim, keepdim=True), 1e-6, 1e6))


# ----------------------------------------------------------------------
# fixed backbone modules
# ----------------------------------------------------------------------
class SpatialConvBlock(nn.Module):
    """SCB: grouped spatial convolution, BatchNorm, swish.

    Adapted from official FBNAS ``FBCNet.SCB`` / ``FBMSNet.SCB`` /
    ``FBNASNet.SCB`` (all three are the same block).

    The ``(n_electrodes, 1)`` kernel collapses the electrode axis to length 1
    and acts as a spatial filter; ``groups=in_channels`` keeps bands separate
    so that each filter-bank channel learns its own spatial pattern.

    With official defaults (``in_channels=36``, ``out_channels=288``,
    ``n_electrodes=22``) the output is ``[B, 288, 1, T]``.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_electrodes: int,
        doWeightNorm: bool = True,
    ):
        super().__init__()
        if out_channels % in_channels != 0:
            raise ValueError(
                f"SCB needs out_channels ({out_channels}) divisible by "
                f"in_channels ({in_channels}) because it is grouped over "
                f"input channels"
            )
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_electrodes = n_electrodes
        self.block = nn.Sequential(
            Conv2dWithConstraint(
                in_channels,
                out_channels,
                (n_electrodes, 1),
                groups=in_channels,
                max_norm=2,
                doWeightNorm=doWeightNorm,
                padding=0,
            ),
            nn.BatchNorm2d(out_channels),
            Swish(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class TemporalClassifier(nn.Module):
    """``LinearWithConstraint`` followed by ``LogSoftmax``.

    Adapted from official FBNAS ``LastBlock``.  Note the constraint is
    ``max_norm=0.5`` here, whereas ``SCB`` uses 2.
    """

    def __init__(
        self,
        in_features: int,
        n_classes: int = C.NUM_CLASSES,
        doWeightNorm: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.n_classes = n_classes
        self.block = nn.Sequential(
            LinearWithConstraint(
                in_features, n_classes, max_norm=0.5, doWeightNorm=doWeightNorm
            ),
            nn.LogSoftmax(dim=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


def backbone_feature_dim(
    num_feat: int = C.NUM_FEAT,
    dilatability: int = C.SCB_DILATABILITY,
    stride_factor: int = C.STRIDEFACTOR,
) -> int:
    """Flat feature size after SCB + reshape + LogVar.

    ``num_feat * dilatability * stride_factor`` -- 36 * 8 * 8 = 2304 by default.
    """
    return num_feat * dilatability * stride_factor


class TemporalBackbone(nn.Module):
    """SCB -> reshape -> LogVar -> flatten -> classifier.

    Drop-in for the fixed tail of official FBNAS, minus any dependency on the
    temporal cell implementation.

    Parameters
    ----------
    in_channels:
        Channels arriving from the temporal stage.  Defaults to
        ``NUM_FEAT`` (36) = 3 bands x NUM_PATHS x PATH_CHANNELS.  The SCB's
        constraint that this be divisible by 3 (bands) is checked.
    num_feat, n_electrodes, n_classes:
        See :mod:`tdarts.config`.
    """

    def __init__(
        self,
        in_channels: int = C.NUM_FEAT,
        n_electrodes: int = C.NUM_ELECTRODES,
        n_classes: int = C.NUM_CLASSES,
        dilatability: int = C.SCB_DILATABILITY,
        stride_factor: int = C.STRIDEFACTOR,
        temporal_layer: str = "logvar",
        doWeightNorm: bool = True,
    ):
        super().__init__()
        if temporal_layer != "logvar":
            raise ValueError(
                f"stage 1 only supports temporal_layer='logvar', got "
                f"{temporal_layer!r}"
            )
        self.in_channels = in_channels
        self.num_feat = in_channels
        self.n_electrodes = n_electrodes
        self.n_classes = n_classes
        self.dilatability = dilatability
        self.stride_factor = stride_factor

        self.scb = SpatialConvBlock(
            in_channels=in_channels,
            out_channels=in_channels * dilatability,
            n_electrodes=n_electrodes,
            doWeightNorm=doWeightNorm,
        )
        self.temporal_layer = LogVarLayer(dim=3)
        self.feature_dim = backbone_feature_dim(
            in_channels, dilatability, stride_factor
        )
        self.classifier = TemporalClassifier(
            self.feature_dim, n_classes, doWeightNorm=doWeightNorm
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(log_probabilities [B, n_classes], features [B, feature_dim])``.

        Matches the official FBNAS ``(c, f)`` return convention.
        """
        if x.dim() != 4:
            raise ValueError(
                f"TemporalBackbone expects [B, C, E, T], got shape "
                f"{tuple(x.shape)}"
            )
        out = self.scb(x)
        time = out.shape[3]
        if time % self.stride_factor != 0:
            raise ValueError(
                f"time length {time} is not divisible by stride_factor "
                f"{self.stride_factor}; LogVar segmentation would silently "
                f"drop samples"
            )
        out = out.reshape(
            [*out.shape[0:2], self.stride_factor, time // self.stride_factor]
        )
        out = self.temporal_layer(out)
        features = torch.flatten(out, start_dim=1)
        logits = self.classifier(features)
        return logits, features
