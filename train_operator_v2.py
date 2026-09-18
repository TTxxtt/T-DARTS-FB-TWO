#!/usr/bin/env python
"""Standalone operator-separability pilot: one temporal family, RF57, no NAS.

This is deliberately *not* a search.  Each run fixes one operator family for
all three bands and trains it from a fresh initialisation on the same Session-0
231/57 split the rest of the project uses.  The only thing that varies across a
run's arm is the family; everything downstream of the temporal cells is the
unchanged FBNAS backbone.

Session 1 is closed by default
------------------------------
A run reads Session 1 only when ``--read-session1`` is passed, and the first
round of the pilot never passes it.  The default matters more than the flag: an
architecture comparison that can see the test set is not a comparison, and a
default that opens the file makes "we did not look" a claim about discipline
rather than about the code.  With the default the file is never opened, so no
later edit can quietly turn it into a selection signal.

Seeding
-------
``set_seed`` is called immediately before the model is constructed.  Building a
network consumes the global torch RNG (every ``nn.Conv2d.__init__`` draws from
it), and the genotype helpers in the sibling retrain script do too.  An earlier
version seeded before resolving the genotype, which gave a ``--search-dir`` run
a different initialisation than a ``--genotype-json`` run at the same
``--seed`` -- invisible, and enough to move a subject by several points.  Here
nothing RNG-consuming sits between the seed and the model.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from run_layout import allocate
from tdarts.operator_v2 import V2_OPERATOR_NAMES, count_macs, operator_audit
from tdarts.operator_v2_network import OperatorV2Net
from tdarts.search_data import load_session0_search_split, load_subject_session

#: The pilot fixes one receptive field so the families are compared as
#: mechanisms rather than as RF ladders.  57 is the middle rung, and at the
#: base kernel of 15 it is exactly realisable (kernel 15, dilation 4).
DEFAULT_RF = 57


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--operator",
        choices=V2_OPERATOR_NAMES,
        required=True,
        help=f"temporal family for all three bands; one of {V2_OPERATOR_NAMES}",
    )
    parser.add_argument("--data-root", type=Path, default=Path("/root/autodl-tmp/bci42a/multiviewPython"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    parser.add_argument("--log-root", type=Path, default=Path("logs"))
    parser.add_argument("--dataset", default="bci42a")
    parser.add_argument(
        "--arm",
        default="operator_v2",
        help="label prefixed to the operator in the run leaf.  The operator "
        "name is always appended: without it all five families would resolve "
        "to one leaf, and allocate() refuses to overwrite, so only the first "
        "would ever run and the other four would fail in seconds.",
    )
    parser.add_argument("--subject", default="003")
    parser.add_argument("--seed", type=int, required=True, help="training/DataLoader seed")
    parser.add_argument("--rf", type=int, default=DEFAULT_RF, help="target receptive field, fixed across families")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=1500)
    parser.add_argument("--patience", type=int, default=200)
    parser.add_argument(
        "--read-session1",
        action="store_true",
        help="open Session 1 and report a test block at the end.  OFF by "
        "default: the pilot compares families on the validation split, and a "
        "test number produced while the comparison is still being designed is "
        "a selection signal waiting to be used.",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--preload-data", action="store_true")
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


def move(batch, device: torch.device):
    x, y = batch
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)


def train_epoch(model, loader, optimizer, criterion, device: torch.device) -> dict[str, float]:
    model.train()
    loss_sum = correct = count = 0
    for batch in loader:
        x, y = move(batch, device)
        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(x)
        loss = criterion(logits, y)
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite V2 training NLL")
        loss.backward()
        optimizer.step()
        loss_sum += float(loss.item()) * y.numel()
        correct += int((logits.argmax(1) == y).sum().item())
        count += y.numel()
    return {"nll": loss_sum / count, "acc": correct / count}


@torch.no_grad()
def evaluate(model, loader, criterion, device: torch.device, *, confusion: bool = False) -> dict[str, float]:
    was_training = model.training
    model.eval()
    loss_sum = correct = count = 0
    matrix = torch.zeros(4, 4, dtype=torch.long)
    try:
        for batch in loader:
            x, y = move(batch, device)
            logits, _ = model(x)
            loss_sum += float(criterion(logits, y).item()) * y.numel()
            prediction = logits.argmax(1)
            correct += int((prediction == y).sum().item())
            count += y.numel()
            if confusion:
                for target, predicted in zip(y.cpu(), prediction.cpu()):
                    matrix[int(target), int(predicted)] += 1
    finally:
        model.train(was_training)
    result = {"nll": loss_sum / count, "acc": correct / count}
    if confusion:
        true = matrix.sum(1).float()
        predicted = matrix.sum(0).float()
        true_positive = matrix.diag().float()
        f1 = 2 * true_positive / (true + predicted).clamp_min(1)
        observed = float(true_positive.sum().item() / count)
        expected = float((true * predicted).sum().item() / (count * count))
        result.update(
            {
                "macro_f1": float(f1.mean().item()),
                "kappa": (observed - expected) / (1 - expected) if expected < 1 else 0.0,
                "confusion_matrix": matrix.tolist(),
            }
        )
    return result


def format_progress_line(epoch: int, fields: dict[str, float], notes: tuple[str, ...] = ()) -> str:
    parts = " | ".join(f"{key} {value:.4f}" for key, value in fields.items())
    suffix = f" | {' '.join(notes)}" if notes else ""
    return f"[v2] epoch {epoch:04d} | {parts}{suffix}"


def main() -> int:
    args = parse_args()
    if args.epochs < 1 or args.patience < 1:
        raise ValueError("epochs and patience must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    arm = f"{args.arm}_{args.operator}" if args.arm else args.operator
    # Record the resolved arm, not the flag's raw value: config.json is read
    # back by analysis tooling to locate the run, and the two would disagree.
    args.arm = arm
    args.output_dir, progress_path = allocate(
        root=args.output_root, log_root=args.log_root, dataset=args.dataset,
        phase="train", subject=args.subject, seed=args.seed, arm=arm,
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"run dir: {args.output_dir}\nlog:     {progress_path}", flush=True)

    # Seed immediately before the model: nothing that draws from the global
    # torch RNG may sit in between.  See the module docstring.
    set_seed(args.seed)
    model = OperatorV2Net(args.operator, target_rf=args.rf).to(device)

    train_data, val_data, split = load_session0_search_split(
        args.data_root, args.subject, preload=args.preload_data
    )
    test_data = (
        load_subject_session(args.data_root, args.subject, session=1, preload=args.preload_data)
        if args.read_session1
        else None
    )
    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
    }
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_data, shuffle=True, generator=generator, **loader_kwargs)
    val_loader = DataLoader(val_data, shuffle=False, **loader_kwargs)
    train_eval_loader = DataLoader(train_data, shuffle=False, **loader_kwargs)
    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    shape = (1, 3, 22, 1000)
    audit = [r for r in operator_audit(band="Low", target_rf=args.rf, time_length=1000) if r["operator"] == args.operator][0]
    # The pilot's per-run cost figures: measured on the model actually built,
    # not read off a table, so a changed architecture cannot leave a stale
    # number in the summary.
    cell = model.cells["Low"].path
    params = sum(p.numel() for p in model.parameters())
    macs = count_macs(cell, shape)
    config = {
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "operator": args.operator,
        "target_rf": args.rf,
        "model": model.describe(),
        "operator_params": audit["params"],
        "operator_macs": audit["macs"],
        "operator_param_ratio_vs_anchor": audit["param_ratio_vs_anchor"],
        "operator_support": {
            "positions": audit["support_positions"],
            "spacing": audit["support_spacing"],
            "span": audit["support_span"],
        },
        "parameters": params,
        "session0_train_size": split.train_size,
        "session0_val_size": split.val_size,
        "session1_test_size": None if test_data is None else len(test_data),
        "session1_opened": test_data is not None,
    }
    (args.output_dir / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    best_score = float("inf")
    best_epoch = 0
    best_val: dict[str, float] = {}
    no_improvement = 0
    started = time.perf_counter()
    with (args.output_dir / "metrics.jsonl").open("x", encoding="utf-8") as log, \
            progress_path.open("w", encoding="utf-8") as progress:
        progress.write(f"# operator_v2 run   {args.output_dir.name}\n")
        progress.write(f"# operator={args.operator} subject={args.subject} seed={args.seed} rf={args.rf}\n")
        progress.write(f"# session1_opened={test_data is not None}\n")
        progress.flush()
        epoch = 0
        for epoch in range(1, args.epochs + 1):
            train = train_epoch(model, train_loader, optimizer, criterion, device)
            train_eval = evaluate(model, train_eval_loader, criterion, device)
            val = evaluate(model, val_loader, criterion, device)
            record = {
                "epoch": epoch,
                "operator": args.operator,
                "subject": args.subject,
                "seed": args.seed,
                "train_nll": train["nll"],
                "train_acc": train["acc"],
                "train_nll_eval": train_eval["nll"],
                "val_nll": val["nll"],
                "val_acc": val["acc"],
                "val_inacc": 1.0 - val["acc"],
                "seconds": time.perf_counter() - started,
            }
            # Validation NLL selects: it is continuous where accuracy is
            # quantised to 1/57, and it is the metric the rest of the project
            # already selects on, so a V2 arm is comparable to the others.
            score = val["nll"]
            record["best_metric"] = "val_nll"
            record["best_metric_value"] = score
            if score < best_score:
                best_score, best_epoch, no_improvement = score, epoch, 0
                best_val = {"nll": val["nll"], "acc": val["acc"]}
                record["is_best"] = True
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_nll": val["nll"],
                        "val_acc": val["acc"],
                        "operator": args.operator,
                    },
                    args.output_dir / "best.pt",
                )
            else:
                no_improvement += 1
                record["is_best"] = False
            log.write(json.dumps(record, sort_keys=True) + "\n")
            log.flush()
            fields = {"trainLoss": train["nll"], "trainAcc": train["acc"], "valLoss": val["nll"], "valAcc": val["acc"]}
            notes = ["best"] if record["is_best"] else []
            notes.append(f"patience {no_improvement}")
            line = format_progress_line(epoch, fields, tuple(notes))
            print(line, flush=True)
            progress.write(line + "\n")
            progress.flush()
            if no_improvement >= args.patience:
                break
        progress.write(
            f"[v2] done | stopEpoch {epoch} | bestEpoch {best_epoch} | bestValNLL {best_score:.4f} "
            f"| bestValAcc {best_val.get('acc', float('nan')):.4f}\n"
        )
        progress.flush()

    best_ckpt = torch.load(args.output_dir / "best.pt", map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"])
    test = None
    if test_data is not None:
        test_loader = DataLoader(test_data, shuffle=False, **loader_kwargs)
        test = evaluate(model, test_loader, criterion, device, confusion=True)

    summary = {
        "subject": str(args.subject),
        "seed": int(args.seed),
        "operator": args.operator,
        "target_rf": int(args.rf),
        "best_epoch": best_epoch,
        "val_best_nll": float(best_val["nll"]),
        "val_best_acc": float(best_val["acc"]),
        "params": params,
        "macs": macs,
        "operator_params": audit["params"],
        "operator_macs": audit["macs"],
        "operator_param_ratio_vs_anchor": audit["param_ratio_vs_anchor"],
        "support": {
            "positions": audit["support_positions"],
            "spacing": audit["support_spacing"],
            "span": audit["support_span"],
        },
        "stage1": {
            "best_epoch": best_epoch,
            "stop_epoch": epoch,
            "best_metric": "val_nll",
            "best_score": best_score,
        },
        "training_seconds": time.perf_counter() - started,
        "session1_opened": test_data is not None,
        "test": test,
        "screening_only": test_data is None,
    }
    (args.output_dir / "final_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
