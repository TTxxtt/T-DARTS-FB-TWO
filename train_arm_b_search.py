#!/usr/bin/env python
"""Arm B: four-mechanism hierarchical search, FBNAS sampling, no architecture gradient.

This is the second of the two final arms.  Arm A searches only the receptive
field of a dilated convolution; this arm searches the *mechanism* first and the
scale second:

Phase A -- family search at a fixed RF
    Every band runs a single path at RF 57 and selects one of four temporal
    families (``dilated_e``/``dynamic_e``/``gated_e``/``band_gated_e``).  Four
    candidates per band, one path, so the space is ``4**3 = 64``.

Phase B -- RF search within the frozen families
    Each band's family is frozen and its receptive field is searched over the
    FBNAS ladder 15/29/57/113, taking one or two RFs.  ``C(4,1) + C(4,2) = 10``
    per band, so the space is ``10**3 = 1000`` -- the same space Arm A searches,
    which is what makes "searching mechanism too" the only difference.

Both phases use the frozen baseline's search protocol rather than DARTS: one
subnet is drawn per step and applied to the whole batch, weights are trained
with plain Adam, there is no architecture parameter and no architecture
gradient.  The final architecture is chosen by enumerating every candidate,
reloading the trained weights, running one train-mode pass over the validation
split so the BatchNorms recalibrate under *that* subnet, and taking the argmax.

**Selection metric is calibrated validation accuracy**, matching upstream's
``nas_phase`` (``FBNAS/codes/centralRepo/NAS.py:308-311``) exactly rather than
the val-NLL rule used elsewhere in this repo.  Arm A's architecture was chosen
by that rule and cannot be re-chosen without editing frozen code, so matching it
is what keeps the two arms comparable in *how* they choose, leaving the search
space as the variable under test.  NLL is recorded alongside as a sensitivity
read; it is never used to select.

Session 1 (the dataset's **second** recording session, code ``session=1``) is
never read: this stage exports an architecture, not a performance claim.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from run_layout import allocate
from tdarts import config as C
from tdarts.band_discrete import BAND_SCHEME, describe_genotype, BandGene, BandGenotype
from tdarts.band_supernet import (
    BAND_FAMILIES,
    BAND_RFS,
    family_candidates,
    rf_candidates,
    calibrated_scores,
    FBNASBandNet,
)
from tdarts.fbnas_sampler import choice_index, random_choice, traverse_choices
from tdarts.search_data import load_session0_search_split

#: Every artifact this stage writes lives under a root carrying this prefix, so
#: an Arm B run can never be mistaken for -- or land inside -- a frozen arm.
STAGE_PREFIX = "operator_armB"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("/root/autodl-tmp/bci42a/multiviewPython"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    parser.add_argument("--log-root", type=Path, default=Path("logs"))
    parser.add_argument("--dataset", default="bci42a")
    parser.add_argument("--arm", default="")
    parser.add_argument("--subject", default="003")
    parser.add_argument("--seed", type=int, default=20190821)
    parser.add_argument(
        "--phase-a-epochs", type=int, default=200,
        help="family-search supernet epochs; 200 matches Arm A's search budget",
    )
    parser.add_argument(
        "--phase-b-epochs", type=int, default=200,
        help="RF-search supernet epochs; 200 matches Arm A's search budget",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3, help="weight learning rate, as upstream NAS.py")
    parser.add_argument("--m-a", type=int, default=1, help="candidates per band in phase A (1 = single path)")
    parser.add_argument("--m-b", type=int, default=2, help="candidates per band in phase B (2 = one or two RFs)")
    parser.add_argument("--preload-data", action="store_true")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--checkpoint-interval", type=int, default=50)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _worker_init(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def _write_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def _serialisable_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        name: str(value) if isinstance(value, Path) else value
        for name, value in vars(args).items()
    }


def train_supernet_epoch(
    model: FBNASBandNet,
    loader: Iterable[tuple[torch.Tensor, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    *,
    device: torch.device,
    sampler: Iterator[dict],
) -> dict[str, float]:
    """One epoch of random-subnet training, upstream's ``NAS.train``.

    A fresh subnet is drawn **per batch** and applied to the whole batch; there
    is no architecture parameter to update, so this is ordinary supervised
    training on a randomly sampled sub-network.
    """

    model.train()
    nll_total = correct = samples = 0
    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        choice = next(sampler)
        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(inputs, choice)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        nll_total += float(loss.item()) * targets.numel()
        correct += int((logits.argmax(dim=1) == targets).sum().item())
        samples += targets.numel()
    return {"train_nll": nll_total / samples, "train_acc": correct / samples}


@torch.no_grad()
def evaluate_choice(
    model: FBNASBandNet,
    loader: Iterable[tuple[torch.Tensor, torch.Tensor]],
    choice: Mapping[str, Sequence[int]],
    criterion: nn.Module,
    *,
    device: torch.device,
) -> dict[str, float]:
    """Score the supernet under one fixed subnet, without touching BN statistics."""

    was_training = model.training
    model.eval()
    nll_total = correct = samples = 0
    try:
        for inputs, targets in loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            logits, _ = model(inputs, choice)
            nll_total += float(criterion(logits, targets).item()) * targets.numel()
            correct += int((logits.argmax(dim=1) == targets).sum().item())
            samples += targets.numel()
    finally:
        model.train(was_training)
    if samples == 0:
        raise ValueError("cannot evaluate an empty loader")
    return {"nll": nll_total / samples, "acc": correct / samples}


def _subnet_stream(seed: int, m: int) -> Iterator[dict]:
    """An endless, reproducible stream of subnets.

    The sampler draws from its own generators rather than the global RNG, so the
    architecture this run searches does not depend on how many draws the data
    loader happened to make first.  Modelling weight initialisation keeps using
    the global RNG, which is what the seed-discipline elsewhere in the repo
    depends on.
    """

    rng = np.random.default_rng(seed)
    py_random = random.Random(seed)
    while True:
        yield random_choice(m, rng=rng, py_random=py_random)


def _summarise_traversal(rows: Sequence[dict]) -> dict[str, Any]:
    accuracies = [row["accuracy"] for row in rows]
    ranked = sorted(rows, key=lambda row: (-row["accuracy"], row["index"]))
    return {
        "candidates": len(rows),
        "accuracy_min": min(accuracies),
        "accuracy_max": max(accuracies),
        "accuracy_mean": float(np.mean(accuracies)),
        "accuracy_std": float(np.std(accuracies)),
        "nll_mean": float(np.mean([row["nll"] for row in rows])),
        "argmax_index": ranked[0]["index"],
        "argmax_accuracy": ranked[0]["accuracy"],
        "runner_up_index": ranked[1]["index"] if len(ranked) > 1 else None,
        "runner_up_accuracy": ranked[1]["accuracy"] if len(ranked) > 1 else None,
        "margin": (ranked[0]["accuracy"] - ranked[1]["accuracy"]) if len(ranked) > 1 else None,
    }


def _phase(
    *,
    label: str,
    model: FBNASBandNet,
    choices: Sequence[Mapping[str, Sequence[int]]],
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epochs: int,
    lr: float,
    seed: int,
    m: int,
    checkpoint_path: Path | None = None,
    checkpoint_interval: int = 0,
    metrics_handle=None,
    progress=None,
) -> tuple[dict, list[dict]]:
    """Train one supernet, then run the full-candidate traversal over it."""

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    sampler = _subnet_stream(seed, m)
    for epoch in range(1, epochs + 1):
        started = time.perf_counter()
        metrics = train_supernet_epoch(
            model, train_loader, optimizer, criterion, device=device, sampler=sampler
        )
        metrics.update(
            {
                "phase": label,
                "epoch": epoch,
                "epoch_seconds": time.perf_counter() - started,
                "sampler": "FBNAS random subnet, one per batch",
                "space_size": len(choices),
            }
        )
        if metrics_handle is not None:
            metrics_handle.write(json.dumps(metrics, ensure_ascii=False, sort_keys=True) + "\n")
            metrics_handle.flush()
        line = (
            f"[armB{label}] epoch {epoch:03d} | trainLoss {metrics['train_nll']:.4f} | "
            f"trainAcc {metrics['train_acc']:.4f} | {metrics['epoch_seconds']:.1f}s"
        )
        print(line, flush=True)
        if progress is not None:
            progress.write(line + "\n")
            progress.flush()
        if checkpoint_path is not None and checkpoint_interval and epoch % checkpoint_interval == 0:
            temporary = checkpoint_path.with_suffix(".pt.tmp")
            torch.save(
                {"phase": label, "epoch": epoch, "model_state_dict": model.state_dict()},
                temporary,
            )
            temporary.replace(checkpoint_path)

    if checkpoint_path is not None:
        temporary = checkpoint_path.with_suffix(".pt.tmp")
        torch.save(
            {"phase": label, "epoch": epochs, "model_state_dict": model.state_dict()}, temporary
        )
        temporary.replace(checkpoint_path)

    inputs = torch.cat([batch[0] for batch in val_loader], dim=0)
    targets = torch.cat([batch[1] for batch in val_loader], dim=0)
    print(f"# traversal [{label}]: scoring {len(choices)} candidates", flush=True)
    traversal_started = time.perf_counter()

    def report(done: int, total: int) -> None:
        # The Phase B sweep is 1000 candidates with a BN recalibration each, so
        # on CPU it can run for ten minutes.  Report often enough that a stalled
        # job is distinguishable from a slow one.
        if done % 100 == 0 or done == total:
            rate = (time.perf_counter() - traversal_started) / max(done, 1)
            line = (
                f"[armB{label}] traversal {done}/{total} "
                f"| {rate * 1000:.0f} ms/candidate | eta {rate * (total - done):.0f}s"
            )
            print(line, flush=True)
            if progress is not None:
                progress.write(line + "\n")
                progress.flush()

    rows = calibrated_scores(
        model, choices, inputs, targets, criterion=criterion, device=device, progress=report
    )
    return _summarise_traversal(rows), rows


def main() -> int:
    args = parse_args()
    if args.phase_a_epochs < 1 or args.phase_b_epochs < 1:
        raise ValueError("phase epochs must be positive")
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    if args.m_a < 1 or args.m_b < 1:
        raise ValueError("m-a and m-b must be at least 1")
    if not args.arm.startswith(STAGE_PREFIX):
        raise ValueError(
            f"--arm must start with {STAGE_PREFIX!r}, got {args.arm!r}; this guard keeps an "
            f"Arm B search from landing inside a frozen arm's run tree"
        )
    if not Path(args.output_root).name.startswith(STAGE_PREFIX):
        raise ValueError(
            f"--output-root must be a directory whose name starts with {STAGE_PREFIX!r}, "
            f"got {args.output_root!r}"
        )
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    args.output_dir, progress_path = allocate(
        root=args.output_root, log_root=args.log_root, dataset=args.dataset,
        phase=f"{STAGE_PREFIX}_search", subject=args.subject, seed=args.seed, arm=None,
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"run dir: {args.output_dir}\nlog:     {progress_path}", flush=True)

    train_dataset, val_dataset, split = load_session0_search_split(
        args.data_root, args.subject, preload=args.preload_data
    )
    loader_generator = torch.Generator().manual_seed(args.seed)
    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "worker_init_fn": _worker_init if args.num_workers else None,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, generator=loader_generator, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    criterion = nn.NLLLoss()

    choices_a = traverse_choices(args.m_a)
    config_payload = {
        **_serialisable_args(args),
        "device": str(device),
        "screening_only": True,
        "session1_opened": False,
        "session2_test": None,
        "scheme": BAND_SCHEME,
        "band_families": list(BAND_FAMILIES),
        "rf_ladder": list(BAND_RFS),
        "phase_a": {"candidates_per_band": len(BAND_FAMILIES), "m": args.m_a, "space": len(choices_a)},
        # Phase B's space size is only known once the families are frozen, so it
        # is filled in below rather than guessed here.
        "phase_b": {"candidates_per_band": len(BAND_RFS), "m": args.m_b, "space": None},
        "selection_metric": "calibrated validation accuracy (matches upstream NAS.nas_phase)",
        "selection_secondary": "validation NLL, recorded only",
        "session0_train_size": split.train_size,
        "session0_val_size": split.val_size,
        "session1_test_size": None,
    }
    _write_json(args.output_dir / "config.json", config_payload)

    manifest = {
        "stage": "armB_hierarchical_family_then_rf",
        "subject": split.subject,
        "session": "0 only (dataset Session 1)",
        "session1_used": False,
        "sampler": "FBNAS random subnet, one per batch, no architecture gradient",
        "selection_metric": "calibrated validation accuracy",
        "phase_a": {
            "rf": 57, "families": list(BAND_FAMILIES), "epochs": args.phase_a_epochs,
            "m": args.m_a, "space_size": len(choices_a),
        },
        "phase_b": {
            "rf_ladder": list(BAND_RFS), "epochs": args.phase_b_epochs, "m": args.m_b,
        },
    }
    _write_json(args.output_dir / "manifest.json", manifest)
    metrics_path = args.output_dir / "metrics.jsonl"
    started = time.perf_counter()

    with metrics_path.open("x", encoding="utf-8") as handle, progress_path.open("w", encoding="utf-8") as progress:
        progress.write(f"# run        {args.output_dir.name} (Arm B hierarchical search)\n")
        progress.write(
            f"# phase A: {len(BAND_FAMILIES)} families at RF57, m={args.m_a}, "
            f"{len(choices_a)} candidates, {args.phase_a_epochs} epochs\n"
        )
        progress.flush()

        # ---------------- Phase A: family search --------------------------
        set_seed(args.seed)
        model_a = FBNASBandNet(
            {band: family_candidates() for band in C.BANDS}, m=args.m_a
        ).to(device)
        manifest["phase_a"]["model"] = model_a.describe()
        _write_json(args.output_dir / "manifest.json", manifest)

        summary_a, rows_a = _phase(
            label="family", model=model_a, choices=choices_a,
            train_loader=train_loader, val_loader=val_loader, criterion=criterion,
            device=device, epochs=args.phase_a_epochs, lr=args.lr, seed=args.seed,
            m=args.m_a, checkpoint_path=args.output_dir / "phase_a_last.pt",
            checkpoint_interval=args.checkpoint_interval,
            metrics_handle=handle, progress=progress,
        )
        winner_a = rows_a[summary_a["argmax_index"]]["choice"]
        frozen = {band: BAND_FAMILIES[winner_a[band][0]] for band in C.BANDS}
        phase_a_result = {
            "phase": "family",
            "space_size": len(choices_a),
            "selection_metric": "calibrated validation accuracy",
            "summary": summary_a,
            "top5": sorted(rows_a, key=lambda row: (-row["accuracy"], row["index"]))[:5],
            "frozen_families": frozen,
            "winner_choice": winner_a,
        }
        _write_json(args.output_dir / "phase_a_result.json", phase_a_result)
        _write_json(args.output_dir / "phase_a_traversal.json", rows_a)
        print(f"# Phase A frozen families: {frozen}", flush=True)
        progress.write(f"# Phase A frozen families: {frozen}\n")
        progress.flush()

        # ---------------- Phase B: RF search within the frozen families ---
        band_candidates_b = {band: rf_candidates(frozen[band]) for band in C.BANDS}
        choices_b = traverse_choices(args.m_b)
        config_payload["phase_b"]["space"] = len(choices_b)
        config_payload["frozen_families"] = frozen
        _write_json(args.output_dir / "config.json", config_payload)
        manifest["phase_b"]["families"] = frozen
        manifest["phase_b"]["space_size"] = len(choices_b)
        _write_json(args.output_dir / "manifest.json", manifest)

        progress.write(
            f"# phase B: families frozen {frozen}, RF ladder {list(BAND_RFS)}, "
            f"m={args.m_b}, {len(choices_b)} candidates, {args.phase_b_epochs} epochs\n"
        )
        progress.flush()
        set_seed(args.seed)
        model_b = FBNASBandNet(band_candidates_b, m=args.m_b).to(device)
        manifest["phase_b"]["model"] = model_b.describe()
        _write_json(args.output_dir / "manifest.json", manifest)

        summary_b, rows_b = _phase(
            label="rf", model=model_b, choices=choices_b,
            train_loader=train_loader, val_loader=val_loader, criterion=criterion,
            device=device, epochs=args.phase_b_epochs, lr=args.lr, seed=args.seed,
            m=args.m_b, checkpoint_path=args.output_dir / "phase_b_last.pt",
            checkpoint_interval=args.checkpoint_interval,
            metrics_handle=handle, progress=progress,
        )
        winner_b = rows_b[summary_b["argmax_index"]]["choice"]

    # ---------------- export the genotype --------------------------------
    genes: list[BandGene] = []
    for band in C.BANDS:
        for path_index, rf_index in enumerate(winner_b[band]):
            genes.append(
                BandGene(
                    band=band, path=path_index, family=frozen[band],
                    target_rf=BAND_RFS[rf_index],
                )
            )
    genotype = BandGenotype(
        seed=args.seed,
        epoch=args.phase_a_epochs + args.phase_b_epochs,
        genes=tuple(genes),
        phase_a=frozen,
        phase_b={band: tuple(BAND_RFS[i] for i in winner_b[band]) for band in C.BANDS},
    )
    _write_json(args.output_dir / "genotype.json", genotype.to_dict())

    elapsed = time.perf_counter() - started
    final_summary = {
        "completed": True,
        "scheme": BAND_SCHEME,
        "total_search_seconds": elapsed,
        "sampler": "FBNAS random subnet",
        "selection_metric": "calibrated validation accuracy",
        "phase_a": summary_a,
        "phase_b": summary_b,
        "frozen_families": frozen,
        "rf_choices": {band: [BAND_RFS[i] for i in winner_b[band]] for band in C.BANDS},
        "genotype": genotype.to_dict(),
        "genotype_description": describe_genotype(genotype),
        "genotype_json": str(args.output_dir / "genotype.json"),
        "parameters": sum(p.numel() for p in model_b.parameters()),
        "screening_only": True,
        "session1_opened": False,
        "session2_test": None,
        "sample_size": split.val_size,
    }
    _write_json(args.output_dir / "final_summary.json", final_summary)
    print(f"\n{describe_genotype(genotype)}", flush=True)
    print(f"\n# genotype -> {args.output_dir / 'genotype.json'}")
    print(f"# total search time {elapsed/60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
