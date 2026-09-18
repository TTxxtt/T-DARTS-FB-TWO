#!/usr/bin/env python
"""Anchored two-phase search on Session 0: operator first, then RF.

Run shape for the first anchored experiment:

* one FBNAS-ordered Session-0 80/20 split (``load_session0_search_split``) and
  one seed -- the fast closed loop agreed for this arm, not a K-fold protocol;
* phase A fixes path 0 to ``dilated`` and searches path 1 over the four
  operator families under a balanced RF schedule (29/57/113);
* phase B freezes the selected operators and searches the RF of both paths
  over 15/29/57/113, by default from a fresh initialisation; with
  ``--no-duplicate-paths`` the two paths of a band are decoded jointly and must
  differ under ``structure_key``;
* Session 1 is never loaded: the export is a genotype, not a performance claim.

Artifacts: ``config.json``, ``manifest.json``, ``metrics.jsonl`` (phase A and
phase B rows under one continuous epoch axis), ``phase_a_result.json``,
``genotype.json``, ``final_summary.json`` and checkpoints.
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
from tdarts import config as C
from tdarts.anchored import (
    ANCHORED_SCHEME,
    OP_SEARCH_RFS,
    AnchoredOperatorNet,
    AnchoredRFNet,
    build_anchored_genotype,
    inherit_operator_weights,
    operator_probability_summary,
    rf_probability_summary,
    rf_schedule,
    run_anchored_epoch,
)
from tdarts.architect import SearchArchitect
from tdarts.search import evaluate
from tdarts.search_data import load_session0_search_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("/root/autodl-tmp/bci42a/multiviewPython"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs"), help="root of the <dataset>/<phase>_s<subject>_seed<seed>_<arm> run tree")
    parser.add_argument("--log-root", type=Path, default=Path("logs"), help="root of the mirrored per-epoch progress log tree")
    parser.add_argument("--dataset", default="bci42a")
    parser.add_argument("--arm", default="anchored", help="label appended to the run leaf")
    parser.add_argument("--subject", default="003")
    parser.add_argument("--seed", type=int, default=20250901)
    parser.add_argument("--operator-epochs", type=int, default=200)
    parser.add_argument("--rf-epochs", type=int, default=200)
    parser.add_argument("--warmup-epochs", type=int, default=20, help="applied per phase: alpha/beta/gamma stay frozen for the first N epochs of each phase")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--weight-lr", type=float, default=1e-3)
    parser.add_argument("--alpha-lr", type=float, default=3e-4)
    parser.add_argument("--alpha-weight-decay", type=float, default=1e-3)
    parser.add_argument("--inherit-weights", action="store_true", help="experimental: initialise phase B from phase A where the geometry already exists (RF 15 stays fresh); default is fresh initialisation")
    parser.add_argument("--no-duplicate-paths", action="store_true", help="phase B decodes the two paths of a band jointly and forbids the same actual structure (structure_key) on both paths; RF 15 aliases count as duplicates")
    parser.add_argument("--preload-data", action="store_true", help="read every trial into memory once (see tdarts.search_data.BCI42aSearchDataset)")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--checkpoint-interval", type=int, default=10)
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


def _checkpoint(
    path: Path,
    *,
    stage: str,
    epoch: int,
    model: nn.Module,
    architect: SearchArchitect,
    args: argparse.Namespace,
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "stage": stage,
            "epoch": epoch,
            "args": _serialisable_args(args),
            "model_state_dict": model.state_dict(),
            "weight_optimizer_state_dict": architect.weight_optimizer.state_dict(),
            "alpha_optimizer_state_dict": architect.alpha_optimizer.state_dict(),
        },
        temporary,
    )
    temporary.replace(path)


def _gpu_peak(metrics: dict[str, Any], device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        metrics["gpu_peak_allocated_mib"] = torch.cuda.max_memory_allocated(device) / 2**20
        metrics["gpu_peak_reserved_mib"] = torch.cuda.max_memory_reserved(device) / 2**20
    else:
        metrics["gpu_peak_allocated_mib"] = 0.0
        metrics["gpu_peak_reserved_mib"] = 0.0


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def main() -> int:
    args = parse_args()
    if args.operator_epochs < 1 or args.rf_epochs < 1 or args.batch_size < 1:
        raise ValueError("operator-epochs, rf-epochs and batch-size must be positive")
    if args.warmup_epochs < 0 or args.warmup_epochs >= args.operator_epochs or args.warmup_epochs >= args.rf_epochs:
        raise ValueError("warmup-epochs must be non-negative and smaller than both phase lengths")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    set_seed(args.seed)
    args.output_dir, progress_path = allocate(
        root=args.output_root, log_root=args.log_root, dataset=args.dataset,
        phase="anchor_search", subject=args.subject, seed=args.seed, arm=args.arm,
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"run dir: {args.output_dir}\nlog:     {progress_path}", flush=True)

    total_epochs = args.operator_epochs + args.rf_epochs
    config = {
        **_serialisable_args(args),
        "device": str(device),
        "epochs": total_epochs,
        "phases": {"operator": args.operator_epochs, "rf": args.rf_epochs},
        "op_search_rfs": list(OP_SEARCH_RFS),
        "rf_search_space": list(C.RF_SPACE["Low"]),
        "no_duplicate_paths": args.no_duplicate_paths,
        "search_folds": 1,
        "session1_used": False,
    }
    _write_json(args.output_dir / "config.json", config)

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

    model_operator = AnchoredOperatorNet().to(device)
    optimizer = torch.optim.Adam(model_operator.network_parameters(), lr=args.weight_lr)
    architect = SearchArchitect(
        model_operator,
        optimizer,
        alpha_lr=args.alpha_lr,
        alpha_weight_decay=args.alpha_weight_decay,
    )
    manifest = {
        "stage": "anchored_operator_then_rf",
        "subject": split.subject,
        "session": "0 only",
        "train_size": split.train_size,
        "val_size": split.val_size,
        "session1_used": False,
        "search_folds": 1,
        "phase_a": model_operator.describe(),
        "bn_layers_frozen_during_alpha_step": sum(
            isinstance(module, nn.modules.batchnorm._BatchNorm)
            for module in model_operator.modules()
        ),
    }
    _write_json(args.output_dir / "manifest.json", manifest)

    rf_counts = {rf: 0 for rf in OP_SEARCH_RFS}
    step = 0

    def install_scheduled_rf(global_step: int) -> None:
        rf = rf_schedule(global_step)
        model_operator.set_rf(rf)
        rf_counts[rf] += 1

    metrics_path = args.output_dir / "metrics.jsonl"
    started = time.perf_counter()
    phase_a_seconds = 0.0
    with metrics_path.open("x", encoding="utf-8") as metrics_handle, \
            progress_path.open("w", encoding="utf-8") as progress:
        progress.write(f"# run            {args.output_dir.name} (anchored: operator then RF)\n")
        progress.write(f"# subject={args.subject} seed={args.seed} operator_epochs={args.operator_epochs} rf_epochs={args.rf_epochs} warmup_epochs={args.warmup_epochs}\n")
        progress.write("# phase A fixes path0=dilated; path1 searches the four operators under a balanced RF schedule\n")
        progress.flush()

        for epoch in range(1, args.operator_epochs + 1):
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            epoch_started = time.perf_counter()
            metrics, step = run_anchored_epoch(
                architect,
                train_loader,
                val_loader,
                criterion,
                epoch=epoch,
                device=device,
                warmup_epochs=args.warmup_epochs,
                start_step=step,
                on_train_step=install_scheduled_rf,
            )
            per_rf = {}
            for rf in OP_SEARCH_RFS:
                model_operator.set_rf(rf)
                per_rf[str(rf)] = evaluate(model_operator, val_loader, criterion, device=device)
            metrics.update(
                {
                    "phase": "operator",
                    "epoch": epoch,
                    "local_epoch": epoch,
                    "val_nll": _mean([value["nll"] for value in per_rf.values()]),
                    "val_acc": _mean([value["acc"] for value in per_rf.values()]),
                    "val_per_rf": per_rf,
                    "rf_counts": {str(rf): count for rf, count in rf_counts.items()},
                    "operators": operator_probability_summary(model_operator),
                    "nan_detected": False,
                }
            )
            _gpu_peak(metrics, device)
            metrics["epoch_seconds"] = time.perf_counter() - epoch_started
            metrics_handle.write(json.dumps(metrics, ensure_ascii=False, sort_keys=True) + "\n")
            metrics_handle.flush()
            line = (
                f"[anchorA] epoch {epoch:03d} | trainLoss {metrics['train_nll']:.4f} | "
                f"valAcc {metrics['val_acc']:.4f} | betaGrad {metrics['alpha_grad_norm']:.5f} | "
                f"rfCounts {list(metrics['rf_counts'].values())} | time {metrics['epoch_seconds']:.1f}s"
            )
            print(line, flush=True)
            progress.write(line + "\n")
            progress.flush()
            _checkpoint(args.output_dir / "last.pt", stage="anchored_operator", epoch=epoch, model=model_operator, architect=architect, args=args)
            if epoch % args.checkpoint_interval == 0 or epoch == args.operator_epochs:
                _checkpoint(args.output_dir / f"operator_epoch_{epoch:03d}.pt", stage="anchored_operator", epoch=epoch, model=model_operator, architect=architect, args=args)

        phase_a_seconds = time.perf_counter() - started
        operators = model_operator.selected_operators()
        phase_a_result = {
            "selected_operators": operators,
            "operators": operator_probability_summary(model_operator),
            "rf_counts": {str(rf): count for rf, count in rf_counts.items()},
            "steps": step,
        }
        _write_json(args.output_dir / "phase_a_result.json", phase_a_result)
        progress.write(f"# phase A frozen operators: {operators}\n")
        progress.flush()

        model_rf = AnchoredRFNet(
            operators, no_duplicate_paths=args.no_duplicate_paths
        ).to(device)
        inherited: list[str] = []
        if args.inherit_weights:
            inherited = inherit_operator_weights(model_operator, model_rf)
            print(f"inherited {len(inherited)} module prefixes from phase A", flush=True)
        optimizer_rf = torch.optim.Adam(model_rf.network_parameters(), lr=args.weight_lr)
        architect_rf = SearchArchitect(
            model_rf,
            optimizer_rf,
            alpha_lr=args.alpha_lr,
            alpha_weight_decay=args.alpha_weight_decay,
        )
        manifest["phase_b"] = model_rf.describe()
        manifest["inherited_state_prefixes"] = inherited
        _write_json(args.output_dir / "manifest.json", manifest)
        progress.write(f"# phase B fresh init{'' if args.inherit_weights else ' (no inheritance)'}; operators={operators}\n")
        progress.flush()

        for local_epoch in range(1, args.rf_epochs + 1):
            global_epoch = args.operator_epochs + local_epoch
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            epoch_started = time.perf_counter()
            metrics, _ = run_anchored_epoch(
                architect_rf,
                train_loader,
                val_loader,
                criterion,
                epoch=local_epoch,
                device=device,
                warmup_epochs=args.warmup_epochs,
                start_step=0,
            )
            soft = evaluate(model_rf, val_loader, criterion, device=device)
            model_rf.set_hard(True)
            try:
                hard = evaluate(model_rf, val_loader, criterion, device=device)
            finally:
                model_rf.set_hard(False)
            metrics.update(
                {
                    "phase": "rf",
                    "epoch": global_epoch,
                    "local_epoch": local_epoch,
                    "operators": operators,
                    "val_nll": soft["nll"],
                    "val_acc": soft["acc"],
                    "val_soft": soft,
                    "val_hard": hard,
                    "soft_hard_acc_gap": soft["acc"] - hard["acc"],
                    "rf_mixtures": rf_probability_summary(model_rf),
                    "nan_detected": False,
                }
            )
            _gpu_peak(metrics, device)
            metrics["epoch_seconds"] = time.perf_counter() - epoch_started
            metrics_handle.write(json.dumps(metrics, ensure_ascii=False, sort_keys=True) + "\n")
            metrics_handle.flush()
            line = (
                f"[anchorB] epoch {global_epoch:03d} | trainLoss {metrics['train_nll']:.4f} | "
                f"valSoft {metrics['val_soft']['acc']:.4f} | valHard {metrics['val_hard']['acc']:.4f} | "
                f"gap {metrics['soft_hard_acc_gap']:+.4f} | gammaGrad {metrics['alpha_grad_norm']:.5f} | "
                f"time {metrics['epoch_seconds']:.1f}s"
            )
            print(line, flush=True)
            progress.write(line + "\n")
            progress.flush()
            _checkpoint(args.output_dir / "last.pt", stage="anchored_rf", epoch=global_epoch, model=model_rf, architect=architect_rf, args=args)
            if local_epoch % args.checkpoint_interval == 0 or local_epoch == args.rf_epochs:
                _checkpoint(args.output_dir / f"checkpoint_epoch_{global_epoch:03d}.pt", stage="anchored_rf", epoch=global_epoch, model=model_rf, architect=architect_rf, args=args)

    genotype = build_anchored_genotype(model_rf, seed=args.seed, epoch=total_epochs)
    genotype_payload = {
        "scheme": ANCHORED_SCHEME,
        "subject": split.subject,
        "seed": args.seed,
        "epochs": total_epochs,
        "fold": 0,
        "no_duplicate_paths": args.no_duplicate_paths,
        "session1_used": False,
        "stability": "single_run_no_cross_run_vote",
        "selected_operators": operators,
        "phase_a": phase_a_result,
        "rf_mixtures": rf_probability_summary(model_rf),
        "genes": [gene.__dict__ for gene in genotype.genes],
        "inherited_state_prefixes": inherited,
    }
    _write_json(args.output_dir / "genotype.json", genotype_payload)

    final = {
        "completed": True,
        "total_seconds": time.perf_counter() - started,
        "phase_a_seconds": phase_a_seconds,
        "no_duplicate_paths": args.no_duplicate_paths,
        "genotype": genotype.to_dict(),
        "selected_genes": model_rf.selected_genes(),
        "final_val_soft": soft,
        "final_val_hard": hard,
    }
    _write_json(args.output_dir / "final_summary.json", final)
    print(json.dumps(final, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
