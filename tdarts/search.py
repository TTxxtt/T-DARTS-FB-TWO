"""Small, testable epoch primitives for the Stage-3 search runner."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
import torch.nn as nn

from tdarts.architect import SearchArchitect

__all__ = ["should_update_alphas", "run_search_epoch", "evaluate"]


def should_update_alphas(epoch: int, warmup_epochs: int = 20) -> bool:
    """Epochs 1..warmup are weight-only; epoch warmup+1 starts DARTS."""

    if epoch < 1:
        raise ValueError("epochs are one-indexed")
    if warmup_epochs < 0:
        raise ValueError("warmup_epochs cannot be negative")
    return epoch > warmup_epochs


def _move_batch(batch: tuple[torch.Tensor, torch.Tensor], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    inputs, targets = batch
    return inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)


def _mean(total: float, count: int) -> float:
    return total / count if count else 0.0


def run_search_epoch(
    architect: SearchArchitect,
    train_loader: Iterable[tuple[torch.Tensor, torch.Tensor]],
    val_loader: Iterable[tuple[torch.Tensor, torch.Tensor]],
    criterion: nn.Module,
    *,
    epoch: int,
    device: torch.device,
    warmup_epochs: int = 20,
    alpha_update_mode: str = "minibatch",
) -> dict[str, Any]:
    """Run alternating first-order DARTS updates for one epoch.

    ``alpha_update_mode`` selects where the architecture gradient comes from:

    ``"minibatch"``
        Validation batches are cycled so that every weight update has one alpha
        update after warm-up -- roughly one Adam step per train batch.  This is
        the original DARTS schedule and the default.

    ``"fullval"``
        The train loop runs to completion first, then the architecture takes a
        single step accumulated over the whole validation loader.  One step per
        epoch, with a lower-variance gradient; the step-frequency change is
        itself part of the method, so it is deliberately not offset elsewhere.

    Either way, epoch-level validation metrics are computed separately by
    :func:`evaluate`; alpha-step samples are optimizer inputs rather than a full
    validation report.
    """

    if alpha_update_mode not in {"minibatch", "fullval"}:
        raise ValueError(f"unknown alpha_update_mode={alpha_update_mode!r}")

    update_alpha = should_update_alphas(epoch, warmup_epochs)
    val_iterator = iter(val_loader) if alpha_update_mode == "minibatch" else None
    total_train_nll = total_train_acc = total_weight_grad = 0.0
    total_alpha_nll = total_alpha_acc = total_alpha_grad = 0.0
    train_steps = alpha_steps = 0

    for train_batch in train_loader:
        train_inputs, train_targets = _move_batch(train_batch, device)
        weight_result = architect.weight_step(train_inputs, train_targets, criterion)
        train_steps += 1
        total_train_nll += weight_result.nll
        total_train_acc += weight_result.accuracy
        total_weight_grad += weight_result.grad_norm

        if update_alpha and alpha_update_mode == "minibatch":
            try:
                val_batch = next(val_iterator)
            except StopIteration:
                val_iterator = iter(val_loader)
                val_batch = next(val_iterator)
            val_inputs, val_targets = _move_batch(val_batch, device)
            alpha_result = architect.alpha_step(val_inputs, val_targets, criterion)
            alpha_steps += 1
            total_alpha_nll += alpha_result.nll
            total_alpha_acc += alpha_result.accuracy
            total_alpha_grad += alpha_result.grad_norm

    # Everything that updates w has now run, so the architecture sees the
    # epoch-end network rather than a mixture of states within the epoch.
    if update_alpha and alpha_update_mode == "fullval":
        alpha_result = architect.alpha_step_loader(val_loader, criterion, device=device)
        alpha_steps = 1
        total_alpha_nll = alpha_result.nll
        total_alpha_acc = alpha_result.accuracy
        total_alpha_grad = alpha_result.grad_norm

    return {
        "train_nll": _mean(total_train_nll, train_steps),
        "train_acc": _mean(total_train_acc, train_steps),
        "network_grad_norm": _mean(total_weight_grad, train_steps),
        "alpha_step_nll": _mean(total_alpha_nll, alpha_steps),
        "alpha_step_acc": _mean(total_alpha_acc, alpha_steps),
        "alpha_grad_norm": _mean(total_alpha_grad, alpha_steps),
        "train_steps": train_steps,
        "alpha_steps": alpha_steps,
        "alpha_updated": update_alpha,
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: Iterable[tuple[torch.Tensor, torch.Tensor]],
    criterion: nn.Module,
    *,
    device: torch.device,
) -> dict[str, float]:
    """Compute full-loader NLL and accuracy without updating BN statistics."""

    was_training = model.training
    model.eval()
    nll_total = correct = samples = 0
    try:
        for batch in loader:
            inputs, targets = _move_batch(batch, device)
            logits, _ = model(inputs)
            loss = criterion(logits, targets)
            nll_total += float(loss.item()) * targets.numel()
            correct += int((logits.argmax(dim=1) == targets).sum().item())
            samples += targets.numel()
    finally:
        model.train(was_training)
    if samples == 0:
        raise ValueError("cannot evaluate an empty loader")
    return {"nll": nll_total / samples, "acc": correct / samples}
