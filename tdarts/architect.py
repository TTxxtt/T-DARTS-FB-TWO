"""First-order DARTS updates with strict parameter and BatchNorm isolation.

``SearchArchitect`` deliberately implements only the first-order alternating
updates used in stage 3:

* a train batch updates network weights ``w``;
* a validation batch updates architecture logits ``alpha``.

There is no unrolling, genotype decoding, discrete model, or retraining here.
The alpha update is particularly defensive: network parameters are disabled and
restored, while every BatchNorm module is switched to evaluation mode so its
running statistics cannot be contaminated by validation data.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator, Sequence

import torch
import torch.nn as nn

__all__ = [
    "StepResult",
    "SearchArchitect",
    "freeze_batch_norm_stats",
]


@dataclass(frozen=True)
class StepResult:
    """Metrics from one optimiser step, detached from the computation graph."""

    nll: float
    accuracy: float
    grad_norm: float


def _grad_norm(parameters: Sequence[nn.Parameter]) -> float:
    squares = [p.grad.detach().pow(2).sum() for p in parameters if p.grad is not None]
    if not squares:
        return 0.0
    return float(torch.sqrt(torch.stack(squares).sum()).item())


@contextmanager
def _requires_grad(parameters: Sequence[nn.Parameter], enabled: bool) -> Iterator[None]:
    """Temporarily set ``requires_grad`` and restore each original state."""

    prior = [p.requires_grad for p in parameters]
    for parameter in parameters:
        parameter.requires_grad_(enabled)
    try:
        yield
    finally:
        for parameter, was_enabled in zip(parameters, prior):
            parameter.requires_grad_(was_enabled)


@contextmanager
def freeze_batch_norm_stats(module: nn.Module) -> Iterator[None]:
    """Keep every BatchNorm running buffer immutable for a validation forward.

    BatchNorm's ``eval`` path uses existing running statistics and, unlike its
    train path, does not increment ``num_batches_tracked`` or update running
    mean/variance.  The original train/eval state of *each* BN is restored on
    exit, including models that intentionally contain mixed module states.
    """

    bns = [m for m in module.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    prior = [bn.training for bn in bns]
    for bn in bns:
        bn.eval()
    try:
        yield
    finally:
        for bn, was_training in zip(bns, prior):
            bn.train(was_training)


class SearchArchitect:
    """Own the isolated first-order DARTS weight and architecture steps."""

    def __init__(
        self,
        model: nn.Module,
        weight_optimizer: torch.optim.Optimizer,
        *,
        alpha_lr: float = 3e-4,
        alpha_betas: tuple[float, float] = (0.5, 0.999),
        alpha_weight_decay: float = 1e-3,
    ):
        if not hasattr(model, "network_parameters") or not hasattr(model, "arch_parameters"):
            raise TypeError("model must expose network_parameters() and arch_parameters()")
        self.model = model
        self.weight_optimizer = weight_optimizer
        self.network_parameters = list(model.network_parameters())
        self.arch_parameters = list(model.arch_parameters())
        if not self.network_parameters or not self.arch_parameters:
            raise ValueError("both network and architecture parameter sets must be non-empty")

        network_ids = {id(p) for p in self.network_parameters}
        arch_ids = {id(p) for p in self.arch_parameters}
        if network_ids & arch_ids:
            raise ValueError("network and architecture parameter sets overlap")

        optimiser_ids = {
            id(p) for group in weight_optimizer.param_groups for p in group["params"]
        }
        if optimiser_ids != network_ids:
            raise ValueError("weight optimizer must contain exactly network_parameters()")

        self.alpha_optimizer = torch.optim.Adam(
            self.arch_parameters,
            lr=alpha_lr,
            betas=alpha_betas,
            weight_decay=alpha_weight_decay,
        )

    @staticmethod
    def _result(logits: torch.Tensor, targets: torch.Tensor, loss: torch.Tensor, grad_norm: float) -> StepResult:
        accuracy = (logits.detach().argmax(dim=1) == targets).float().mean()
        return StepResult(float(loss.detach().item()), float(accuracy.item()), grad_norm)

    def weight_step(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        criterion: nn.Module,
    ) -> StepResult:
        """Update only ``w`` from one Session-0 train batch."""

        self.model.train()
        self.weight_optimizer.zero_grad(set_to_none=True)
        self.alpha_optimizer.zero_grad(set_to_none=True)
        with _requires_grad(self.arch_parameters, False):
            logits, _ = self.model(inputs)
            loss = criterion(logits, targets)
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite train NLL")
            loss.backward()
            result = self._result(logits, targets, loss, _grad_norm(self.network_parameters))
            self.weight_optimizer.step()
        return result

    def alpha_step(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        criterion: nn.Module,
    ) -> StepResult:
        """Update only ``alpha`` from one validation batch.

        The fixed backbone contains FBNAS-compatible constrained layers whose
        forward path may renormalise ``.data``.  A byte-for-byte snapshot of all
        network parameters is restored after this validation forward, making
        the contract stronger than merely excluding ``w`` from Adam.
        """

        self.model.train()
        self.alpha_optimizer.zero_grad(set_to_none=True)
        for parameter in self.network_parameters:
            parameter.grad = None
        network_snapshot = [p.detach().clone() for p in self.network_parameters]
        try:
            with _requires_grad(self.network_parameters, False), freeze_batch_norm_stats(self.model):
                logits, _ = self.model(inputs)
                loss = criterion(logits, targets)
                if not torch.isfinite(loss):
                    raise FloatingPointError("non-finite validation NLL")
                loss.backward()
                result = self._result(logits, targets, loss, _grad_norm(self.arch_parameters))
                self.alpha_optimizer.step()
        finally:
            with torch.no_grad():
                for parameter, original in zip(self.network_parameters, network_snapshot):
                    parameter.copy_(original)
        return result
