#!/usr/bin/env python
"""FBNAS-style final training for one fixed Stage-3 temporal genotype.

Session 0 is split in its original order into the FBNAS-compatible 231/57
train/validation split.  Session 1 is loaded exactly once, after restoring the
best validation-accuracy checkpoint.
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
from torch.utils.data import ConcatDataset, DataLoader

from tdarts.discrete_network import TemporalDiscreteNet, transfer_supernet_weights
from run_layout import allocate
from tdarts.genotype import extract_genotype
from tdarts.search_data import load_session0_search_split, load_subject_session


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--search-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("/root/autodl-tmp/bci42a/multiviewPython"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs"), help="root of the <dataset>/<phase>/<leaf> run tree")
    parser.add_argument("--log-root", type=Path, default=Path("logs"), help="root of the mirrored per-epoch progress log tree")
    parser.add_argument("--dataset", default="bci42a")
    parser.add_argument("--arm", default="tdarts", help="label appended to the run leaf; distinguishes arms inside one train/ directory")
    parser.add_argument("--subject", default="003")
    parser.add_argument("--seed", type=int, required=True, help="training/DataLoader seed")
    parser.add_argument("--initialization", choices=("transfer", "random"), default="random")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-epochs", type=int, default=1500)
    parser.add_argument("--patience", type=int, default=200)
    parser.add_argument("--stage2-epochs", type=int, default=600)
    parser.add_argument("--observe-test", action="store_true", help="log Session1 metrics only; never select on them")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--preload-data",
        action="store_true",
        help="read every trial into memory once instead of re-unpickling it on "
        "each pass.  Numerically identical, but removes ~650 MB of GPFS reads "
        "per epoch; the upstream baseline preloads the same way (eegDataset "
        "with loadNonLoadedData=True, ho.py:269).",
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


def worker_init(_: int) -> None:
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


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
            raise FloatingPointError("non-finite retraining NLL")
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
        macro_f1 = float(f1.mean().item())
        observed = float(true_positive.sum().item() / count)
        expected = float((true * predicted).sum().item() / (count * count))
        result.update(
            {
                "macro_f1": macro_f1,
                "kappa": (observed - expected) / (1 - expected) if expected < 1 else 0.0,
                "confusion_matrix": matrix.tolist(),
            }
        )
    return result


def count_macs(model: nn.Module, device: torch.device) -> int:
    """Count multiply-accumulates for Conv2d/Linear on one BCI trial."""

    total = 0
    hooks = []
    def hook(module, _inputs, output):
        nonlocal total
        if isinstance(module, nn.Conv2d):
            total += output.numel() * (module.in_channels // module.groups) * module.kernel_size[0] * module.kernel_size[1]
        elif isinstance(module, nn.Linear):
            total += output.numel() * module.in_features
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            hooks.append(module.register_forward_hook(hook))
    was_training = model.training
    model.eval()
    with torch.no_grad():
        model(torch.zeros(1, 1, 22, 1000, 9, device=device))
    model.train(was_training)
    for handle in hooks:
        handle.remove()
    return int(total)


def serialise_args(args: argparse.Namespace) -> dict:
    return {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}


def format_progress_line(stage: str, epoch: int, fields: dict[str, float], notes: tuple[str, ...] = ()) -> str:
    """Build one human-readable progress line.

    The field set mirrors what upstream prints every epoch (baseModel.py:386-390
    -- "Train loss = ... Train Acc = ... Val Acc = ... Val loss = ..."),
    collapsed onto a single greppable line so the run can be followed with
    ``tail -f``.
    """

    cells = [f"[{stage}] epoch {epoch:04d}"]
    cells.extend(f"{name} {value:.4f}" for name, value in fields.items())
    cells.extend(notes)
    return " | ".join(cells)


def main() -> int:
    args = parse_args()
    if args.max_epochs < 1 or args.patience < 1 or args.stage2_epochs < 1:
        raise ValueError("max-epochs, patience, and stage2-epochs must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    set_seed(args.seed)
    args.output_dir, progress_path = allocate(
        root=args.output_root, log_root=args.log_root, dataset=args.dataset, phase="train",
        subject=args.subject, seed=args.seed, arm=args.arm,
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"run dir: {args.output_dir}\nlog:     {progress_path}", flush=True)
    # Take the epoch from the search's own config rather than assuming 200:
    # train_search.py writes checkpoint_epoch_<epochs>.pt at its final epoch, so
    # a search run with a different --epochs would otherwise leave both the
    # genotype lookup and (for --initialization transfer) the checkpoint path
    # pointing at a file that was never written.
    search_config_path = args.search_dir / "config.json"
    if not search_config_path.is_file():
        raise FileNotFoundError(f"{args.search_dir} does not look like a search run: no config.json")
    search_epochs = int(json.loads(search_config_path.read_text(encoding="utf-8"))["epochs"])
    metrics_path = args.search_dir / "metrics.jsonl"
    checkpoint_path = args.search_dir / f"checkpoint_epoch_{search_epochs:03d}.pt"
    genotype = extract_genotype(metrics_path, epoch=search_epochs)
    model = TemporalDiscreteNet(genotype).to(device)
    transferred = []
    if args.initialization == "transfer":
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"--initialization transfer needs {checkpoint_path.name}, which the search writes at its "
                f"final epoch; use --initialization random for a from-scratch run"
            )
        transferred = transfer_supernet_weights(model, str(checkpoint_path))
    train_data, val_data, split = load_session0_search_split(
        args.data_root, args.subject, preload=args.preload_data
    )
    test_data = load_subject_session(
        args.data_root, args.subject, session=1, preload=args.preload_data
    )
    generator = torch.Generator().manual_seed(args.seed)
    loader_kwargs = {"batch_size": args.batch_size, "num_workers": args.num_workers, "pin_memory": device.type == "cuda", "worker_init_fn": worker_init if args.num_workers else None}
    train_loader = DataLoader(train_data, shuffle=True, generator=generator, **loader_kwargs)
    val_loader = DataLoader(val_data, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_data, shuffle=False, **loader_kwargs)
    # Dedicated shuffle=False loader for the frozen-model train-loss pass.  A
    # separate loader (rather than reusing train_loader) keeps that pass from
    # drawing from the training generator's random stream.
    train_eval_loader = DataLoader(train_data, shuffle=False, **loader_kwargs)
    # Stage 2 trains on Session 0 train+val combined, so its frozen-model loss
    # pass must cover that same union (upstream merges the two sets before
    # resuming, baseModel.py:415).
    session0_eval_loader = DataLoader(ConcatDataset((train_data, val_data)), shuffle=False, **loader_kwargs)
    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    macs = count_macs(model, device)
    config = {"args": serialise_args(args), "genotype": genotype.to_dict(), "session0_train_size": split.train_size, "session0_val_size": split.val_size, "session1_test_size": len(test_data), "transferred_state_keys": transferred, "parameters": parameters, "macs": macs}
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    best_val_inacc = float("inf")
    best_epoch = 0
    no_improvement = 0
    started = time.perf_counter()
    with (args.output_dir / "metrics.jsonl").open("x", encoding="utf-8") as log, \
            progress_path.open("w", encoding="utf-8") as progress:
        progress.write(f"# run            {args.output_dir.name}\n")
        progress.write(f"# subject={args.subject} seed={args.seed} initialization={args.initialization}\n")
        progress.write(f"# max_epochs={args.max_epochs} patience={args.patience} stage2_epochs={args.stage2_epochs}\n")
        progress.write("# trainLoss      online per-batch mean collected inside train_epoch()\n")
        progress.write("# trainLossEval  frozen eval-mode pass over the whole training set == upstream's trainLoss\n")
        progress.write(f"# observe_test={args.observe_test} -- Session 1 metrics are recorded for observation only; "
                       "no stop, checkpoint or selection decision reads them.\n")
        progress.flush()
        for epoch in range(1, args.max_epochs + 1):
            train = train_epoch(model, train_loader, optimizer, criterion, device)
            # FBNAS measures trainLoss with a second pass over the whole training
            # set using the frozen epoch-end model, in eval mode:
            # baseModel.py:370 runs trainOneEpoch(), then :374 calls
            # self.predict(trainData, ...), which does net.eval() at :572.  The
            # online average collected inside train_epoch() is a different
            # quantity -- it spans parameters that kept changing and uses BN
            # batch statistics -- and Stage 2's stop rule weighs it against an
            # eval-mode valLoss, so both sides must be measured the same way.
            train_eval = evaluate(model, train_eval_loader, criterion, device)
            val = evaluate(model, val_loader, criterion, device)
            val_inacc = 1.0 - val["acc"]
            record = {"epoch": epoch, "train_nll": train["nll"], "train_acc": train["acc"], "train_nll_eval": train_eval["nll"], "val_nll": val["nll"], "val_acc": val["acc"], "val_inacc": val_inacc, "seconds": time.perf_counter() - started}
            record["stage"] = "stage1"
            if val_inacc < best_val_inacc:
                best_val_inacc = val_inacc
                best_epoch = epoch
                no_improvement = 0
                torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "val_inacc": val_inacc}, args.output_dir / "best.pt")
                record["is_best"] = True
            else:
                no_improvement += 1
                record["is_best"] = False
            if args.observe_test:
                record["test_observation"] = evaluate(model, test_loader, criterion, device)
            log.write(json.dumps(record, sort_keys=True) + "\n")
            log.flush()
            fields = {"trainLoss": train["nll"], "trainAcc": train["acc"], "trainLossEval": train_eval["nll"], "valLoss": val["nll"], "valAcc": val["acc"]}
            if args.observe_test:
                fields["testAcc"] = record["test_observation"]["acc"]
                fields["testLoss"] = record["test_observation"]["nll"]
            notes = []
            if record["is_best"]:
                notes.append("best")
            notes.append(f"patience {no_improvement}")
            line = format_progress_line("stage1", epoch, fields, tuple(notes))
            print(line, flush=True)
            progress.write(line + "\n")
            progress.flush()
            if no_improvement >= args.patience:
                break
        progress.write(f"[stage1] done | stopEpoch {epoch} | bestEpoch {best_epoch} | bestValInacc {best_val_inacc:.4f} | threshold {train_eval['nll']:.4f}\n")
        progress.flush()
    # FBNAS-style Stage 2: restore Stage 1's validation-best state, then train
    # on all Session 0 data.  Session 1 remains observation-only, if requested.
    best = torch.load(args.output_dir / "best.pt", map_location=device)
    model.load_state_dict(best["model_state_dict"])
    optimizer.load_state_dict(best["optimizer_state_dict"])
    stage1_stop_epoch = epoch
    # Upstream reads monitors['trainLoss'] at the moment of the switch, i.e. the
    # frozen-model train loss of Stage 1's terminal epoch (baseModel.py:377 sets
    # it, :420 consumes it).  Stage 1's best checkpoint was restored just above,
    # but the threshold deliberately still comes from the stop epoch.
    stage1_terminal_train_nll = train_eval["nll"]
    stage2_generator = torch.Generator().manual_seed(args.seed + 1)
    all_session0_loader = DataLoader(
        ConcatDataset((train_data, val_data)), shuffle=True, generator=stage2_generator, **loader_kwargs
    )
    stage2_stop_reason = "max_epochs"
    with (args.output_dir / "metrics.jsonl").open("a", encoding="utf-8") as log, \
            progress_path.open("a", encoding="utf-8") as progress:
        for stage2_epoch in range(1, args.stage2_epochs + 1):
            train = train_epoch(model, all_session0_loader, optimizer, criterion, device)
            train_eval = evaluate(model, session0_eval_loader, criterion, device)
            val = evaluate(model, val_loader, criterion, device)
            record = {
                "stage": "stage2", "stage2_epoch": stage2_epoch,
                "train_nll": train["nll"], "train_acc": train["acc"],
                "train_nll_eval": train_eval["nll"],
                "val_nll": val["nll"], "val_acc": val["acc"],
                "seconds": time.perf_counter() - started,
            }
            if args.observe_test:
                record["test_observation"] = evaluate(model, test_loader, criterion, device)
            log.write(json.dumps(record, sort_keys=True) + "\n")
            log.flush()
            fields = {"trainLoss": train["nll"], "trainAcc": train["acc"], "trainLossEval": train_eval["nll"], "valLoss": val["nll"], "valAcc": val["acc"]}
            if args.observe_test:
                fields["testAcc"] = record["test_observation"]["acc"]
                fields["testLoss"] = record["test_observation"]["nll"]
            notes = [f"threshold {stage1_terminal_train_nll:.4f}"]
            if val["nll"] < stage1_terminal_train_nll:
                notes.append("stop")
            line = format_progress_line("stage2", stage2_epoch, fields, tuple(notes))
            print(line, flush=True)
            progress.write(line + "\n")
            progress.flush()
            if val["nll"] < stage1_terminal_train_nll:
                stage2_stop_reason = "official_val_nll_lt_stage1_terminal_train_nll"
                break
        progress.write(f"[stage2] done | epochs {stage2_epoch} | reason {stage2_stop_reason} | threshold {stage1_terminal_train_nll:.4f}\n")
        progress.flush()
    torch.save({"stage2_epoch": stage2_epoch, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict()}, args.output_dir / "stage2_final.pt")
    test = evaluate(model, test_loader, criterion, device, confusion=True)
    summary = {
        "initialization": args.initialization, "genotype": genotype.to_dict(),
        "parameters": parameters, "macs": macs,
        "stage1": {"best_epoch": best_epoch, "stop_epoch": stage1_stop_epoch, "best_val_inacc": best_val_inacc},
        "stage2": {"epochs": stage2_epoch, "stop_reason": stage2_stop_reason, "session0_train_size": len(train_data) + len(val_data)},
        "test": test, "training_seconds": time.perf_counter() - started,
        "test_observation_only": args.observe_test,
    }
    (args.output_dir / "final_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
