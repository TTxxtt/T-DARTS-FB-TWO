#!/usr/bin/env python
"""Run Stage-3 first-order DARTS search on BCI-IV-2a Session 0.

This entry point intentionally has no Session-1 loader and no discrete-network
logic.  Its only job is to establish whether the continuous supernet learns
stable architecture preferences without BN leakage.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from run_layout import allocate
from tdarts.architect import SearchArchitect
from tdarts.genotype import Genotype, PathGene, extract_genotype, save_genotype
from tdarts.mixed_op import TemporalDARTSNet, candidate_names
from tdarts.search import evaluate, run_search_epoch, should_update_alphas
from tdarts.search_data import load_session0_search_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("/root/autodl-tmp/bci42a/multiviewPython"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs"), help="root of the <dataset>/<phase>/<leaf> run tree")
    parser.add_argument("--log-root", type=Path, default=Path("logs"), help="root of the mirrored per-epoch progress log tree")
    parser.add_argument("--dataset", default="bci42a")
    parser.add_argument("--arm", default="tdarts", help="label appended to the run leaf")
    parser.add_argument("--subject", default="003")
    parser.add_argument("--seed", type=int, default=20250901)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--warmup-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--weight-lr", type=float, default=1e-3)
    parser.add_argument("--alpha-lr", type=float, default=3e-4)
    parser.add_argument("--alpha-weight-decay", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--checkpoint-interval", type=int, default=10)
    parser.add_argument(
        "--alpha-update-mode",
        choices=("minibatch", "fullval"),
        default="minibatch",
        help="minibatch (default) cycles one validation batch per weight step, "
        "the original DARTS schedule; fullval runs one accumulated step over "
        "the whole validation loader after each epoch's weight training.",
    )
    parser.add_argument(
        "--ema-decay",
        type=float,
        default=0.9,
        help="decay for the EMA of softmax(alpha) used by --decode-mode ema; "
        "the EMA starts at the first epoch warmup ends, never at epoch 1.",
    )
    parser.add_argument(
        "--decode-mode",
        choices=("last", "ema"),
        default="last",
        help="which genotype genotype.json points at: the final epoch's argmax "
        "(default, historical behaviour) or the argmax of the EMA "
        "probabilities.  Both are written either way.",
    )
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


def _architecture_summary(model: TemporalDARTSNet) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for (band, path), alpha in model.alphas().items():
        probabilities = torch.softmax(alpha.detach(), dim=0).cpu()
        top_values, top_indices = torch.topk(probabilities, k=2)
        entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum()
        labels = candidate_names(band)
        key = f"{band}_path{path + 1}"
        summary[key] = {
            "alpha": [float(value) for value in alpha.detach().cpu()],
            "probabilities": [float(value) for value in probabilities],
            "top1": {"index": int(top_indices[0]), "candidate": labels[int(top_indices[0])], "probability": float(top_values[0])},
            "top2": {"index": int(top_indices[1]), "candidate": labels[int(top_indices[1])], "probability": float(top_values[1])},
            "margin": float(top_values[0] - top_values[1]),
            "entropy": float(entropy),
        }
    return summary


def _update_ema_probabilities(
    ema_probabilities: dict[tuple[str, int], torch.Tensor],
    model: TemporalDARTSNet,
    decay: float,
) -> None:
    """Fold the current ``softmax(alpha)`` into the running average, in place.

    Only ever called once the warmup has ended, so the average never contains
    the near-uniform logits every search starts from.
    """

    with torch.no_grad():
        for key, alpha in model.alphas().items():
            probabilities = torch.softmax(alpha.detach(), dim=0).cpu()
            if key in ema_probabilities:
                ema_probabilities[key].mul_(decay).add_(probabilities, alpha=1.0 - decay)
            else:
                ema_probabilities[key] = probabilities.clone()


def _genotype_from_ema(
    model: TemporalDARTSNet,
    ema_probabilities: dict[tuple[str, int], torch.Tensor],
    *,
    seed: int,
    epoch: int,
) -> Genotype:
    """Build the genotype from the EMA probabilities instead of one epoch.

    ``extract_genotype`` reads a single logged epoch -- at this budget, the
    last -- and the last epoch's argmax is precisely the quantity that keeps
    moving.  Averaging the probabilities first asks which candidate wins on
    average rather than which one happened to lead when training stopped.
    """

    if not ema_probabilities:
        raise ValueError("no EMA probabilities were accumulated")
    genes = []
    for (band, path), probabilities in sorted(
        ema_probabilities.items(), key=lambda item: f"{item[0][0]}_path{item[0][1] + 1}"
    ):
        index = int(probabilities.argmax())
        names = candidate_names(band)
        alpha = model.alphas()[(band, path)]
        genes.append(
            PathGene(
                band=band,
                path=path,
                candidate_index=index,
                candidate=names[index],
                alpha=float(alpha.detach().cpu()[index]),
                probability=float(probabilities[index]),
            )
        )
    return Genotype(seed=seed, epoch=epoch, genes=tuple(genes))


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


def _checkpoint(path: Path, *, epoch: int, model: TemporalDARTSNet, architect: SearchArchitect, args: argparse.Namespace) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "stage": "stage3_first_order_search",
            "epoch": epoch,
            "args": _serialisable_args(args),
            "model_state_dict": model.state_dict(),
            "weight_optimizer_state_dict": architect.weight_optimizer.state_dict(),
            "alpha_optimizer_state_dict": architect.alpha_optimizer.state_dict(),
        },
        temporary,
    )
    temporary.replace(path)


def main() -> int:
    args = parse_args()
    if args.epochs < 1 or args.warmup_epochs < 0 or args.batch_size < 1:
        raise ValueError("epochs/batch-size must be positive and warmup-epochs non-negative")
    if args.epochs <= args.warmup_epochs:
        raise ValueError("epochs must exceed warmup-epochs so alpha can be updated")
    if not 0.0 <= args.ema_decay < 1.0:
        raise ValueError("ema-decay must lie in [0, 1)")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    set_seed(args.seed)
    args.output_dir, progress_path = allocate(
        root=args.output_root, log_root=args.log_root, dataset=args.dataset, phase="search",
        subject=args.subject, seed=args.seed, arm=args.arm,
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"run dir: {args.output_dir}\nlog:     {progress_path}", flush=True)
    _write_json(args.output_dir / "config.json", {**_serialisable_args(args), "device": str(device)})
    train_dataset, val_dataset, split = load_session0_search_split(args.data_root, args.subject)
    loader_generator = torch.Generator().manual_seed(args.seed)
    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "worker_init_fn": _worker_init if args.num_workers else None,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, generator=loader_generator, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)

    model = TemporalDARTSNet().to(device)
    weight_optimizer = torch.optim.Adam(model.network_parameters(), lr=args.weight_lr)
    architect = SearchArchitect(
        model,
        weight_optimizer,
        alpha_lr=args.alpha_lr,
        alpha_weight_decay=args.alpha_weight_decay,
    )
    criterion = nn.NLLLoss()
    manifest = {
        "stage": "stage3_first_order_search",
        "subject": split.subject,
        "session": "0 only",
        "train_size": split.train_size,
        "val_size": split.val_size,
        "session1_used": False,
        "bn_layers_frozen_during_alpha_step": sum(
            isinstance(module, nn.modules.batchnorm._BatchNorm) for module in model.modules()
        ),
        "model": model.describe(),
    }
    _write_json(args.output_dir / "manifest.json", manifest)

    metrics_path = args.output_dir / "metrics.jsonl"
    ema_probabilities: dict[tuple[str, int], torch.Tensor] = {}
    started = time.perf_counter()
    with metrics_path.open("x", encoding="utf-8") as metrics_handle, \
            progress_path.open("w", encoding="utf-8") as progress:
        progress.write(f"# run            {args.output_dir.name} (DARTS search)\n")
        progress.write(f"# subject={args.subject} seed={args.seed} epochs={args.epochs} warmup_epochs={args.warmup_epochs}\n")
        progress.write("# trainLoss is the search-phase loss, trained jointly with the architecture parameters\n")
        progress.flush()
        for epoch in range(1, args.epochs + 1):
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
                torch.cuda.synchronize(device)
            epoch_started = time.perf_counter()
            metrics = run_search_epoch(
                architect,
                train_loader,
                val_loader,
                criterion,
                epoch=epoch,
                device=device,
                warmup_epochs=args.warmup_epochs,
                alpha_update_mode=args.alpha_update_mode,
            )
            # Same gate the alpha update uses, so the average starts on the
            # first epoch whose logits were actually trained.
            if should_update_alphas(epoch, args.warmup_epochs):
                _update_ema_probabilities(ema_probabilities, model, args.ema_decay)
            validation = evaluate(model, val_loader, criterion, device=device)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                metrics["gpu_peak_allocated_mib"] = torch.cuda.max_memory_allocated(device) / 2**20
                metrics["gpu_peak_reserved_mib"] = torch.cuda.max_memory_reserved(device) / 2**20
            else:
                metrics["gpu_peak_allocated_mib"] = 0.0
                metrics["gpu_peak_reserved_mib"] = 0.0
            metrics.update(
                {
                    "epoch": epoch,
                    "epoch_seconds": time.perf_counter() - epoch_started,
                    "val_nll": validation["nll"],
                    "val_acc": validation["acc"],
                    "nan_detected": False,
                    "paths": _architecture_summary(model),
                }
            )
            metrics_handle.write(json.dumps(metrics, ensure_ascii=False, sort_keys=True) + "\n")
            metrics_handle.flush()
            line = (
                f"[search] epoch {epoch:03d} | trainLoss {metrics['train_nll']:.4f} | "
                f"valLoss {metrics['val_nll']:.4f} | valAcc {metrics['val_acc']:.4f} | "
                f"alphaGrad {metrics['alpha_grad_norm']:.5f} | time {metrics['epoch_seconds']:.1f}s"
            )
            print(line, flush=True)
            progress.write(line + "\n")
            progress.flush()
            _checkpoint(args.output_dir / "last.pt", epoch=epoch, model=model, architect=architect, args=args)
            if epoch % args.checkpoint_interval == 0 or epoch == args.epochs:
                _checkpoint(args.output_dir / f"checkpoint_epoch_{epoch:03d}.pt", epoch=epoch, model=model, architect=architect, args=args)

    # Export the genotypes the retrain stage consumes.  train_retrain.py
    # --genotype-json reads these without re-deriving the architecture from the
    # alpha logits, so a search and a hand-built genotype enter retraining
    # through the identical door.  The last-epoch one is built from the logged
    # metrics (rather than from model.alphas()) so it reuses the ordering check
    # extract_genotype already applies to every other consumer.
    #
    # Both are written regardless of --decode-mode: they come from the same
    # search, so comparing the two retrains is the only way to separate "did the
    # full-validation step change the search" from "did averaging the
    # probabilities change the decoded architecture".
    last_genotype = extract_genotype(metrics_path, epoch=args.epochs)
    last_path = save_genotype(last_genotype, args.output_dir / "genotype_last.json")
    ema_genotype = _genotype_from_ema(
        model, ema_probabilities, seed=args.seed, epoch=args.epochs
    )
    ema_path = save_genotype(ema_genotype, args.output_dir / "genotype_ema.json")
    selected = ema_genotype if args.decode_mode == "ema" else last_genotype
    genotype_path = save_genotype(selected, args.output_dir / "genotype.json")

    agreeing = [
        gene.candidate == ema_genotype.genes[index].candidate
        for index, gene in enumerate(last_genotype.genes)
    ]
    final = {
        "completed": True,
        "total_search_seconds": time.perf_counter() - started,
        "final_paths": _architecture_summary(model),
        "genotype_json": str(genotype_path),
        "decode_mode": args.decode_mode,
        "genotype_last_json": str(last_path),
        "genotype_ema_json": str(ema_path),
        "genotype_last_vs_ema_agreeing_paths": sum(agreeing),
    }
    _write_json(args.output_dir / "final_summary.json", final)
    print(json.dumps(final, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
