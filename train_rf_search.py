#!/usr/bin/env python
"""Phase B RF-only search on Session 0: both paths fixed to ``dilated``.

No operator search runs here.  The operator pilot did not identify a stable
winner on Subject 003, so the conv mechanism is frozen to the FBNAS prior
(``dilated``) and only the receptive field of each path is searched over the
full FBNAS ladder 15/29/57/113 -- 3 bands x 2 paths x 4 RF = 24 gammas.

``--no-duplicate-paths`` decodes each band's two paths *jointly* (the pair
maximises the summed log-probability subject to the two ``structure_key``
values differing), so a band never exports two identical RFs.  With both paths
fixed to ``dilated``, equal structure means equal RF.

Session 1 is never read: this run exports a genotype, not a performance claim.
The retrain stage consumes ``genotype.json`` directly.
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
from tdarts.search import evaluate
from tdarts.search_data import load_session0_search_split

#: The only operator family in this search space.
FIXED_OPERATOR = "dilated"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("/root/autodl-tmp/bci42a/multiviewPython"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs"), help="root of the <dataset>/<phase>_s<subject>_seed<seed>_<arm> run tree")
    parser.add_argument("--log-root", type=Path, default=Path("logs"), help="root of the mirrored per-epoch progress log tree")
    parser.add_argument("--dataset", default="bci42a")
    parser.add_argument("--arm", default="rf", help="label appended to the run leaf")
    parser.add_argument("--subject", default="003")
    parser.add_argument("--seed", type=int, default=20250901)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--warmup-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--weight-lr", type=float, default=1e-3)
    parser.add_argument("--alpha-lr", type=float, default=3e-4)
    parser.add_argument("--alpha-weight-decay", type=float, default=1e-3)
    parser.add_argument("--no-duplicate-paths", action="store_true", help="decode each band's two paths jointly and forbid the same RF on both; the main RF experiment enables this")
    parser.add_argument("--preload-data", action="store_true", help="read every trial into memory once (numerically identical; removes GPFS re-reads)")
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


def _unordered_rf_pairs(genotype) -> dict[str, list[int]]:
    """Per band, the two selected RFs as a sorted (unordered) pair.

    The two paths are interchangeable at the architecture level, so
    ``(29, 57)`` and ``(57, 29)`` must not be counted as different
    architectures when seeds are aggregated.
    """

    pairs = {}
    for band in C.BANDS:
        rfs = sorted(
            int(genotype.gene(band, path).candidate.rsplit("_rf", 1)[1])
            for path in (0, 1)
        )
        pairs[band] = rfs
    return pairs


def _checkpoint(path: Path, *, epoch: int, model: AnchoredRFNet, architect: SearchArchitect, args: argparse.Namespace) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "stage": "phase_b_rf_only_search",
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
        raise ValueError("epochs must exceed warmup-epochs so gamma can be updated")
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

    model = AnchoredRFNet(
        {band: FIXED_OPERATOR for band in C.BANDS},
        no_duplicate_paths=args.no_duplicate_paths,
    ).to(device)
    weight_optimizer = torch.optim.Adam(model.network_parameters(), lr=args.weight_lr)
    architect = SearchArchitect(
        model,
        weight_optimizer,
        alpha_lr=args.alpha_lr,
        alpha_weight_decay=args.alpha_weight_decay,
    )
    criterion = nn.NLLLoss()
    manifest = {
        "stage": "phase_b_rf_only_search",
        "subject": split.subject,
        "session": "0 only",
        "train_size": split.train_size,
        "val_size": split.val_size,
        "session1_used": False,
        "operators": {band: FIXED_OPERATOR for band in C.BANDS},
        "rf_search_space": list(C.RF_SPACE["Low"]),
        "num_gammas": model.num_arch_parameters(),
        "no_duplicate_paths": args.no_duplicate_paths,
        "bn_layers_frozen_during_alpha_step": sum(
            isinstance(module, nn.modules.batchnorm._BatchNorm) for module in model.modules()
        ),
        "model": model.describe(),
    }
    _write_json(args.output_dir / "manifest.json", manifest)

    metrics_path = args.output_dir / "metrics.jsonl"
    started = time.perf_counter()
    with metrics_path.open("x", encoding="utf-8") as metrics_handle, \
            progress_path.open("w", encoding="utf-8") as progress:
        progress.write(f"# run            {args.output_dir.name} (RF-only search)\n")
        progress.write(f"# subject={args.subject} seed={args.seed} epochs={args.epochs} warmup_epochs={args.warmup_epochs} no_duplicate_paths={args.no_duplicate_paths}\n")
        progress.write("# operators are fixed to dilated; gamma mixes the four RFs per path\n")
        progress.flush()
        for epoch in range(1, args.epochs + 1):
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
                torch.cuda.synchronize(device)
            epoch_started = time.perf_counter()
            metrics, _ = run_anchored_epoch(
                architect,
                train_loader,
                val_loader,
                criterion,
                epoch=epoch,
                device=device,
                warmup_epochs=args.warmup_epochs,
            )
            soft = evaluate(model, val_loader, criterion, device=device)
            model.set_hard(True)
            try:
                hard = evaluate(model, val_loader, criterion, device=device)
            finally:
                model.set_hard(False)
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
                    "val_nll": soft["nll"],
                    "val_acc": soft["acc"],
                    "val_hard_nll": hard["nll"],
                    "val_hard_acc": hard["acc"],
                    "soft_hard_acc_gap": soft["acc"] - hard["acc"],
                    "nan_detected": False,
                    "rf_mixtures": rf_probability_summary(model),
                }
            )
            metrics_handle.write(json.dumps(metrics, ensure_ascii=False, sort_keys=True) + "\n")
            metrics_handle.flush()
            line = (
                f"[rfsearch] epoch {epoch:03d} | trainLoss {metrics['train_nll']:.4f} | "
                f"valSoft {soft['acc']:.4f} | valHard {hard['acc']:.4f} | gap {metrics['soft_hard_acc_gap']:+.4f} | "
                f"gammaGrad {metrics['alpha_grad_norm']:.5f} | time {metrics['epoch_seconds']:.1f}s"
            )
            print(line, flush=True)
            progress.write(line + "\n")
            progress.flush()
            _checkpoint(args.output_dir / "last.pt", epoch=epoch, model=model, architect=architect, args=args)
            if epoch % args.checkpoint_interval == 0 or epoch == args.epochs:
                _checkpoint(args.output_dir / f"checkpoint_epoch_{epoch:03d}.pt", epoch=epoch, model=model, architect=architect, args=args)

    genotype = build_anchored_genotype(model, seed=args.seed, epoch=args.epochs)
    pairs = _unordered_rf_pairs(genotype)
    if args.no_duplicate_paths and any(len(set(pair)) != 2 for pair in pairs.values()):
        raise RuntimeError(f"no_duplicate_paths produced a duplicate RF pair: {pairs}")
    save_genotype(genotype, args.output_dir / "genotype.json")
    final = {
        "completed": True,
        "total_search_seconds": time.perf_counter() - started,
        "operators": {band: FIXED_OPERATOR for band in C.BANDS},
        "no_duplicate_paths": args.no_duplicate_paths,
        "genotype": genotype.to_dict(),
        "unordered_rf_pairs": pairs,
        "rf_mixtures": rf_probability_summary(model),
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
