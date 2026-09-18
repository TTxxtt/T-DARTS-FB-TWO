#!/usr/bin/env python
"""Frozen hierarchical search: hard operator phase (RF57) then RF phase.

One seed runs its own complete chain and exports its own genotype; no
cross-seed majority vote is ever formed.

Phase A -- operator search, fixed RF 57
    Path 0 is the ``dilated`` anchor.  Path 1 draws one of the four operator
    families at every training step with a straight-through Gumbel-softmax
    sample, so the forward is a single operator at a fixed scale: no RF
    schedule and no four-way soft mixture.  Beta is updated by the first-order
    architecture step; the temperature is annealed over the phase.

Phase B -- RF search, operators frozen
    The operators decoded from Phase A are frozen (one per band), and both
    paths search the full FBNAS ladder 15/29/57/113.  With
    ``--no-duplicate-paths`` the two RFs of a band are decoded jointly so the
    two structures differ; the comparison uses the built ``structure_key``.

Artifacts: config.json, manifest.json, metrics.jsonl, phase_a_result.json,
genotype.json (plain six-gene file for ``train_retrain.py --genotype-json``),
final_summary.json and checkpoints.  Session 1 is never read.
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
    AnchoredRFNet,
    build_anchored_genotype,
    rf_probability_summary,
    run_anchored_epoch,
)
from tdarts.architect import SearchArchitect
from tdarts.genotype import save_genotype
from tdarts.hierarchical import HardOperatorNet, PHASE_A_RF
from tdarts.search import evaluate, run_search_epoch
from tdarts.search_data import load_session0_search_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("/root/autodl-tmp/bci42a/multiviewPython"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    parser.add_argument("--log-root", type=Path, default=Path("logs"))
    parser.add_argument("--dataset", default="bci42a")
    parser.add_argument("--arm", default="")
    parser.add_argument("--subject", default="003")
    parser.add_argument("--seed", type=int, default=20250901)
    parser.add_argument("--phase-a-epochs", type=int, default=300, help="operator phase; longer than the 50-epoch screening so slow operators are not punished")
    parser.add_argument("--phase-b-epochs", type=int, default=200, help="RF phase")
    parser.add_argument("--warmup-a", type=int, default=30)
    parser.add_argument("--warmup-b", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--weight-lr", type=float, default=1e-3)
    parser.add_argument("--alpha-lr", type=float, default=3e-4)
    parser.add_argument("--alpha-weight-decay", type=float, default=1e-3)
    parser.add_argument("--tau-start", type=float, default=1.0, help="Gumbel temperature at Phase A start")
    parser.add_argument("--tau-end", type=float, default=0.3, help="Gumbel temperature at Phase A end")
    parser.add_argument("--no-duplicate-paths", action="store_true", help="Phase B: joint RF decode, the two path structures must differ")
    parser.add_argument("--preload-data", action="store_true")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--checkpoint-interval", type=int, default=25)
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


def _beta_summary(model: HardOperatorNet) -> dict[str, Any]:
    summary = {}
    for band, cell in model.cells.items():
        probabilities = cell.operator_weights().detach().cpu()
        top_values, top_indices = torch.topk(probabilities, k=2)
        entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum()
        summary[band] = {
            "probabilities": [float(value) for value in probabilities],
            "labels": list(C.OPERATORS),
            "top1": {"operator": C.OPERATORS[int(top_indices[0])], "probability": float(top_values[0])},
            "top2": {"operator": C.OPERATORS[int(top_indices[1])], "probability": float(top_values[1])},
            "margin": float(top_values[0] - top_values[1]),
            "normalized_entropy": float(entropy / torch.log(torch.tensor(float(len(C.OPERATORS))))),
            "selection_counts": cell.selection_counts.tolist(),
        }
    return summary


def _unordered_rf_pairs(genotype) -> dict[str, list[int]]:
    pairs = {}
    for band in C.BANDS:
        pairs[band] = sorted(
            int(genotype.gene(band, path).candidate.rsplit("_rf", 1)[1])
            for path in (0, 1)
        )
    return pairs


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


def main() -> int:
    args = parse_args()
    if args.phase_a_epochs < 1 or args.phase_b_epochs < 1 or args.batch_size < 1:
        raise ValueError("phase epochs and batch-size must be positive")
    if args.warmup_a < 0 or args.warmup_a >= args.phase_a_epochs:
        raise ValueError("warmup-a must be non-negative and smaller than phase-a-epochs")
    if args.warmup_b < 0 or args.warmup_b >= args.phase_b_epochs:
        raise ValueError("warmup-b must be non-negative and smaller than phase-b-epochs")
    if args.tau_start <= 0 or args.tau_end <= 0:
        raise ValueError("Gumbel temperatures must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    set_seed(args.seed)
    args.output_dir, progress_path = allocate(
        root=args.output_root, log_root=args.log_root, dataset=args.dataset, phase="hier_search",
        subject=args.subject, seed=args.seed, arm=args.arm,
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"run dir: {args.output_dir}\nlog:     {progress_path}", flush=True)
    _write_json(args.output_dir / "config.json", {**_serialisable_args(args), "device": str(device)})

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

    manifest = {
        "stage": "hierarchical_hard_operator_then_rf",
        "subject": split.subject,
        "session": "0 only",
        "train_size": split.train_size,
        "val_size": split.val_size,
        "session1_used": False,
        "phase_a": {"rf": PHASE_A_RF, "epochs": args.phase_a_epochs, "warmup": args.warmup_a,
                    "tau_start": args.tau_start, "tau_end": args.tau_end,
                    "sampling": "straight-through Gumbel one-hot"},
        "phase_b": {"rf_space": list(C.RF_SPACE["Low"]), "epochs": args.phase_b_epochs,
                    "warmup": args.warmup_b, "no_duplicate_paths": args.no_duplicate_paths},
    }
    _write_json(args.output_dir / "manifest.json", manifest)

    metrics_path = args.output_dir / "metrics.jsonl"
    started = time.perf_counter()

    # ---------------- Phase A: hard operator search ----------------------
    model_a = HardOperatorNet(sampling=True).to(device)
    manifest["phase_a"]["model"] = model_a.describe()
    _write_json(args.output_dir / "manifest.json", manifest)
    optimizer_a = torch.optim.Adam(model_a.network_parameters(), lr=args.weight_lr)
    architect_a = SearchArchitect(
        model_a, optimizer_a, alpha_lr=args.alpha_lr, alpha_weight_decay=args.alpha_weight_decay
    )
    with metrics_path.open("x", encoding="utf-8") as handle, progress_path.open("w", encoding="utf-8") as progress:
        progress.write(f"# run        {args.output_dir.name} (hierarchical search)\n")
        progress.write(f"# phase A: fixed RF{PHASE_A_RF}, Gumbel one-hot, {args.phase_a_epochs} epochs (warmup {args.warmup_a})\n")
        progress.flush()
        tau = args.tau_start
        for epoch in range(1, args.phase_a_epochs + 1):
            epoch_started = time.perf_counter()
            # Linear temperature anneal across the whole phase.
            if args.phase_a_epochs > 1:
                fraction = (epoch - 1) / (args.phase_a_epochs - 1)
                tau = args.tau_start + fraction * (args.tau_end - args.tau_start)
            model_a.set_tau(tau)
            metrics = run_search_epoch(
                architect_a, train_loader, val_loader, criterion,
                epoch=epoch, device=device, warmup_epochs=args.warmup_a,
            )
            soft = evaluate(model_a, val_loader, criterion, device=device)
            model_a.set_hard_eval(True)
            try:
                hard = evaluate(model_a, val_loader, criterion, device=device)
            finally:
                model_a.set_hard_eval(False)
            metrics.update(
                {
                    "phase": "operator",
                    "epoch": epoch,
                    "tau": tau,
                    "epoch_seconds": time.perf_counter() - epoch_started,
                    "val_soft_acc": soft["acc"],
                    "val_hard_acc": hard["acc"],
                    "betas": _beta_summary(model_a),
                }
            )
            handle.write(json.dumps(metrics, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            top = {band: values["top1"]["operator"] for band, values in metrics["betas"].items()}
            line = (
                f"[hierA] epoch {epoch:03d} | trainLoss {metrics['train_nll']:.4f} | "
                f"valSoft {soft['acc']:.4f} | valHard {hard['acc']:.4f} | tau {tau:.2f} | "
                f"top {top} | alphaGrad {metrics['alpha_grad_norm']:.5f} | time {metrics['epoch_seconds']:.1f}s"
            )
            print(line, flush=True)
            progress.write(line + "\n")
            progress.flush()
            _checkpoint(args.output_dir / "phase_a_last.pt", stage="hier_operator", epoch=epoch, model=model_a, architect=architect_a, args=args)

    operators = model_a.selected_operators()
    phase_a_result = {
        "operators": operators,
        "rf": PHASE_A_RF,
        "epochs": args.phase_a_epochs,
        "warmup": args.warmup_a,
        "tau_start": args.tau_start,
        "tau_end": args.tau_end,
        "betas": _beta_summary(model_a),
        "selection_counts": model_a.selection_counts(),
        "final_val_soft": soft,
        "final_val_hard": hard,
    }
    _write_json(args.output_dir / "phase_a_result.json", phase_a_result)
    print(f"# Phase A frozen operators: {operators}", flush=True)

    # ---------------- Phase B: RF search on frozen operators -------------
    model_b = AnchoredRFNet(operators, no_duplicate_paths=args.no_duplicate_paths).to(device)
    manifest["phase_b"]["operators"] = operators
    manifest["phase_b"]["model"] = model_b.describe()
    _write_json(args.output_dir / "manifest.json", manifest)
    optimizer_b = torch.optim.Adam(model_b.network_parameters(), lr=args.weight_lr)
    architect_b = SearchArchitect(
        model_b, optimizer_b, alpha_lr=args.alpha_lr, alpha_weight_decay=args.alpha_weight_decay
    )
    with metrics_path.open("a", encoding="utf-8") as handle, progress_path.open("a", encoding="utf-8") as progress:
        progress.write(f"# phase B: operators frozen {operators}, RF search {list(C.RF_SPACE['Low'])}\n")
        progress.flush()
        for epoch in range(1, args.phase_b_epochs + 1):
            epoch_started = time.perf_counter()
            metrics, _ = run_anchored_epoch(
                architect_b, train_loader, val_loader, criterion,
                epoch=epoch, device=device, warmup_epochs=args.warmup_b,
            )
            soft = evaluate(model_b, val_loader, criterion, device=device)
            model_b.set_hard(True)
            try:
                hard = evaluate(model_b, val_loader, criterion, device=device)
            finally:
                model_b.set_hard(False)
            metrics.update(
                {
                    "phase": "rf",
                    "epoch": epoch,
                    "epoch_seconds": time.perf_counter() - epoch_started,
                    "val_soft_acc": soft["acc"],
                    "val_hard_acc": hard["acc"],
                    "soft_hard_acc_gap": soft["acc"] - hard["acc"],
                    "rf_mixtures": rf_probability_summary(model_b),
                }
            )
            handle.write(json.dumps(metrics, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            line = (
                f"[hierB] epoch {epoch:03d} | trainLoss {metrics['train_nll']:.4f} | "
                f"valSoft {soft['acc']:.4f} | valHard {hard['acc']:.4f} | "
                f"gap {metrics['soft_hard_acc_gap']:+.4f} | gammaGrad {metrics['alpha_grad_norm']:.5f} | "
                f"time {metrics['epoch_seconds']:.1f}s"
            )
            print(line, flush=True)
            progress.write(line + "\n")
            progress.flush()
            _checkpoint(args.output_dir / "phase_b_last.pt", stage="hier_rf", epoch=epoch, model=model_b, architect=architect_b, args=args)

    genotype = build_anchored_genotype(model_b, seed=args.seed, epoch=args.phase_a_epochs + args.phase_b_epochs)
    # Duplicate-path prohibition is judged on the built structure_key, not on RF
    # equality: dilated+normal at the same RF are different structures and are a
    # legal pair, while an identical structure at equal RF is not.  AnchoredRFNet
    # already enforces that in its joint decode, so the exported genotype is
    # checked here with the same rule the retrain stage re-applies.
    pairs = _unordered_rf_pairs(genotype)
    if args.no_duplicate_paths:
        base = model_b
        for band in C.BANDS:
            anchor_index, searched_index = base.cells[band].select_rf_indices()
            anchor_key = base.cells[band].anchor_pool.ops[anchor_index].structure_key
            searched_key = base.cells[band].searched_pool.ops[searched_index].structure_key
            if anchor_key == searched_key:
                raise RuntimeError(
                    f"no_duplicate_paths produced an identical structure in {band}: {anchor_key}"
                )
    save_genotype(genotype, args.output_dir / "genotype.json")
    final = {
        "completed": True,
        "total_seconds": time.perf_counter() - started,
        "operators": operators,
        "no_duplicate_paths": args.no_duplicate_paths,
        "genotype": genotype.to_dict(),
        "unordered_rf_pairs": pairs,
        "rf_mixtures": rf_probability_summary(model_b),
        "final_val_soft": soft,
        "final_val_hard": hard,
        "soft_hard_acc_gap": soft["acc"] - hard["acc"],
        "genotype_json": str(args.output_dir / "genotype.json"),
    }
    _write_json(args.output_dir / "final_summary.json", final)
    print(json.dumps(final, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
