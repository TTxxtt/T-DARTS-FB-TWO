#!/usr/bin/env python
"""FBNAS-style final training for one fixed temporal genotype.

Two genotype sources are supported:

* ``--search-dir``   the 14-candidate DARTS search, read from ``metrics.jsonl``;
* ``--genotype-json`` any exported genotype file.  A plain six-gene file (RF-only
  search, sampled genotypes, operator-separability genotypes) is read with
  :func:`tdarts.genotype.load_genotype`; an anchored export, which carries the
  ``"scheme": "anchored_operator_then_rf"`` tag, is read with
  :func:`tdarts.anchored.load_anchored_genotype` and gets its extra validation.

Either way the discrete network starts from a fresh initialisation.  Session 0
is split in its original order into the FBNAS-compatible 231/57
train/validation split.  Session 1 is loaded exactly once, after restoring the
best validation-accuracy checkpoint; ``--screening-only`` never opens it.
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

from tdarts.anchored import ANCHORED_SCHEME, load_anchored_genotype
from tdarts.band_discrete import (
    BAND_SCHEME,
    BandDiscreteNet,
    band_duplicate_bands,
    band_structure_keys,
    load_band_genotype,
)
from tdarts.discrete_network import TemporalDiscreteNet, transfer_supernet_weights
from run_layout import allocate
from tdarts.genotype import (
    duplicate_structure_bands,
    extract_genotype,
    load_genotype,
    path_structure_keys,
)
from tdarts.search_data import load_session0_search_split, load_subject_session


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--search-dir",
        type=Path,
        default=None,
        help="a completed search run to decode the genotype from; mutually "
        "exclusive with --genotype-json",
    )
    parser.add_argument(
        "--genotype-json",
        type=Path,
        default=None,
        help="a saved six-path genotype.  Plain files -- train_search.py's "
        "<search-dir>/genotype.json, the RF-only search export, samplers such "
        "as tools/sample_random_genotypes.py -- load through load_genotype; an "
        "anchored export (scheme tag) loads through load_anchored_genotype.  "
        "Mutually exclusive with --search-dir, and requires --initialization "
        "random because such a genotype has no supernet checkpoint behind it",
    )
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
    parser.add_argument(
        "--best-metric",
        choices=("val_inacc", "val_nll"),
        default="val_inacc",
        help="Stage-1 checkpoint/early-stop metric.  val_inacc (default) keeps "
        "the historical behaviour; val_nll selects the validation-loss-best "
        "epoch, which the frozen hierarchical protocol uses so every arm shares "
        "one selection rule.",
    )
    parser.add_argument("--stage2-epochs", type=int, default=600)
    parser.add_argument(
        "--stage2-min-epochs",
        type=int,
        default=0,
        help="floor on Stage 2's length: the val_nll < stage-1-terminal-train_nll "
        "break is not allowed before this epoch.  Default 0 is the historical "
        "rule, which let a subject whose validation NLL dips early stop after a "
        "handful of epochs (s009 stopped at 9, one random run at 7) -- far less "
        "training than every other subject got.  The break still applies once "
        "the floor is reached, so this lowers nothing and only extends the "
        "shortest runs.",
    )
    parser.add_argument(
        "--stage2-fixed-epochs",
        type=int,
        default=None,
        help="run Stage 2 for exactly this many epochs, ignoring the "
        "val_nll < stage-1-terminal-train_nll early stop.  Default None keeps "
        "the historical threshold rule, which is what every archived run used.  "
        "A run using this flag is NOT on that protocol, so its Session-1 "
        "reading is comparable only to another fixed-epoch run -- not to the "
        "threshold-stopped majority.  Equivalent to --stage2-min-epochs N "
        "--stage2-epochs N; give one or the other, not both.",
    )
    parser.add_argument("--observe-test", action="store_true", help="log Session1 metrics only; never select on them")
    parser.add_argument(
        "--no-duplicate-paths",
        action="store_true",
        help="refuse a genotype in which a band's two paths realise the same "
        "structure.  The check compares built operators' structure_key, so "
        "different names for one function (dilated_rf15 vs normal_rf15) count "
        "as duplicates.  Off by default: the operator-separability pilot "
        "deliberately includes a dilated+dilated model.",
    )
    parser.add_argument(
        "--screening-only",
        action="store_true",
        help="Stage 1 only, and Session 1 is never read.  For architecture "
        "screening, where the point is that the test set stays closed: the "
        "file is not opened at all, so no later edit can quietly turn it into "
        "a selection signal",
    )
    parser.add_argument(
        "--stage2-only",
        action="store_true",
        help="Skip Stage 1; load best.pt from an existing run and run "
        "Stage 2 (Session 0+1 train, Session 1 test) only.  The run "
        "directory must already exist with a valid best.pt and "
        "final_summary.json.",
    )
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


def _stage1_terminal_train_nll(run_dir: Path, stage1: dict) -> float | None:
    """The FBNAS Stage-2 threshold: Stage-1 terminal *training* loss.

    FBNAS's ``baseModel.train`` (baseModel.py:417-421) resumes after early stop
    on the full session-0 set and stops once the validation loss drops back to
    (below) the training loss measured at the moment early stopping fired.  Our
    Stage-1 screening records that moment as ``stage1["stop_epoch"]``; the value
    is the ``train_nll`` of the ``metrics.jsonl`` row at that epoch.  A run whose
    metrics file or stop epoch is missing returns ``None`` so the Stage-2 job
    falls back to threshold A alone rather than failing.
    """
    stop_epoch = stage1.get("stop_epoch")
    metrics_path = run_dir / "metrics.jsonl"
    if stop_epoch is None or not metrics_path.is_file():
        return None
    for line in metrics_path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row.get("stage") != "stage1" or row.get("epoch") != stop_epoch:
            continue
        return float(row["train_nll"])
    return None


def main() -> int:
    args = parse_args()
    if args.max_epochs < 1 or args.patience < 1 or args.stage2_epochs < 1:
        raise ValueError("max-epochs, patience, and stage2-epochs must be positive")
    if args.stage2_fixed_epochs is not None and args.stage2_fixed_epochs < 1:
        raise ValueError("--stage2-fixed-epochs must be positive when given")
    if args.stage2_min_epochs < 0:
        raise ValueError("--stage2-min-epochs cannot be negative")
    if args.stage2_fixed_epochs is not None and args.stage2_min_epochs:
        raise ValueError(
            "--stage2-fixed-epochs already pins the length; drop "
            "--stage2-min-epochs to keep which rule ran unambiguous"
        )
    if (args.search_dir is None) == (args.genotype_json is None):
        raise ValueError("give exactly one of --search-dir or --genotype-json")
    if args.genotype_json is not None and args.initialization == "transfer":
        raise ValueError(
            "--genotype-json has no supernet checkpoint to transfer from; "
            "pass --initialization random"
        )
    if args.screening_only and args.observe_test:
        raise ValueError(
            "--screening-only never loads Session 1, so --observe-test has "
            "nothing to observe; drop one of them"
        )
    if args.stage2_only and args.screening_only:
        raise ValueError("--stage2-only and --screening-only are mutually exclusive")
    if args.stage2_only and args.observe_test:
        raise ValueError("--stage2-only and --observe-test are mutually exclusive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    # --stage2-only: load an existing Stage 1 run and run Stage 2 only.
    if args.stage2_only:
        if args.genotype_json is None and args.search_dir is None:
            raise ValueError("--stage2-only requires --genotype-json or --search-dir to locate the run")
        # Locate the run directory: it must already exist.
        candidate = allocate(
            root=args.output_root, log_root=args.log_root, dataset=args.dataset, phase="train",
            subject=args.subject, seed=args.seed, arm=args.arm,
        )
        run_dir = candidate[0] if isinstance(candidate, tuple) else candidate
        if not run_dir.is_dir():
            raise FileNotFoundError(f"--stage2-only: run directory {run_dir} does not exist")
        summary_path = run_dir / "final_summary.json"
        best_path = run_dir / "best.pt"
        if not summary_path.is_file() or not best_path.is_file():
            raise FileNotFoundError(f"--stage2-only: {run_dir} missing final_summary.json or best.pt")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        saved = summary.get("genotype")
        if isinstance(saved, dict) and saved.get("scheme") == BAND_SCHEME:
            # This branch rebuilds the model through load_genotype, which reads
            # the plain six-gene grammar.  The band dialect would be silently
            # rejected there rather than resumed, and a half-resumed run is
            # worse than a refusal: say so instead of letting it fail obscurely.
            raise ValueError(
                "--stage2-only cannot resume a band-dialect run: it reconstructs the "
                "genotype with load_genotype, which reads the plain six-gene grammar. "
                "Re-run the arm's own Stage 1 command; it runs both stages in one process."
            )
        # Reconstruct genotype from the saved summary.  final_summary.json stores
        # Genotype.to_dict() inline, so hand the dict to load_genotype directly
        # (passing it as a path would TypeError).
        genotype = load_genotype(saved)
        genotype_source = summary.get("genotype_source", {"kind": "final_summary"})
        stage1 = summary.get("stage1", {})
        # Two Stage-2 stopping thresholds, both recorded and read out on Session
        # 1 so the protocols can be compared directly:
        #   * A = Stage-1 validation-best NLL -- the historic threshold here.
        #   * B = Stage-1 terminal *training* loss (baseModel.py:417-421), the
        #     faithful FBNAS rule.  For every subject sampled so far B << A, so
        #     it trains far longer before qualifying.
        threshold_a = float(stage1["best_score"])
        threshold_b = _stage1_terminal_train_nll(run_dir, stage1)
        # --stage2-fixed-epochs turns the budget into the exact length: the loop
        # still runs to this count, but the threshold break below is disabled.
        stage2_epochs = args.stage2_fixed_epochs or args.stage2_epochs
        progress_path = run_dir.parent.parent / "logs" / args.dataset / f"train_s{args.subject}_seed{args.seed}_{args.arm}.log"
        progress_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"run dir (stage2-only): {run_dir}", flush=True)
        # Load data.
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
        session0_eval_loader = DataLoader(ConcatDataset((train_data, val_data)), shuffle=False, **loader_kwargs)
        all_session0_loader = DataLoader(
            ConcatDataset((train_data, val_data)), shuffle=True,
            generator=torch.Generator().manual_seed(args.seed + 1), **loader_kwargs
        )
        criterion = nn.NLLLoss()
        # Same seeding rule as the main path: seed right before the model is
        # built.  Stage 2 overwrites the weights from best.pt immediately below,
        # so this only matters for the RNG state the dataloaders and any later
        # draw see -- but leaving the branch unseeded made a --stage2-only run
        # depend on whatever the process inherited.
        set_seed(args.seed)
        model = TemporalDiscreteNet(genotype).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        best_ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(best_ckpt["model_state_dict"])
        optimizer.load_state_dict(best_ckpt["optimizer_state_dict"])
        # Stage 2 training: keep training until BOTH thresholds are crossed (or
        # the epoch budget), checkpointing the first epoch each threshold is
        # met, so each protocol (A and B) gets its own Session-1 test reading.
        stage2_stop_reason = "max_epochs"
        started = time.perf_counter()
        crossed_a: dict | None = None  # first Stage-2 epoch with val NLL <= A
        crossed_b: dict | None = None  # ... <= B
        with (run_dir / "metrics.jsonl").open("a", encoding="utf-8") as log, \
                progress_path.open("a", encoding="utf-8") as progress:
            progress.write(
                f"# stage2-only: stage2_epochs={stage2_epochs} "
                f"thresholdA={threshold_a:.4f} "
                f"thresholdB={'%.4f' % threshold_b if threshold_b is not None else 'n/a'}\n"
            )
            progress.flush()
            for stage2_epoch in range(1, stage2_epochs + 1):
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
                log.write(json.dumps(record, sort_keys=True) + "\n")
                log.flush()
                fields = {"trainLoss": train["nll"], "trainAcc": train["acc"], "trainLossEval": train_eval["nll"], "valLoss": val["nll"], "valAcc": val["acc"]}
                notes = [f"thrA {threshold_a:.4f} thrB {'%.4f' % threshold_b if threshold_b is not None else 'n/a'}"]
                if args.stage2_min_epochs:
                    notes.append(f"floor {stage2_epoch}/{args.stage2_min_epochs}")
                b_crossed_now = threshold_b is not None and crossed_b is None and val["nll"] <= threshold_b
                a_crossed_now = crossed_a is None and val["nll"] <= threshold_a
                if b_crossed_now or a_crossed_now:
                    # Snapshot the crossing checkpoint so it can be re-read on
                    # Session 1 after the loop, once training has moved on.
                    snapshot = {k: v.detach().clone() for k, v in model.state_dict().items()}
                    if b_crossed_now:
                        crossed_b = {"epoch": stage2_epoch, "val_nll": val["nll"], "state_dict": snapshot}
                        notes.append(f"crossB@{stage2_epoch}")
                    if a_crossed_now:
                        crossed_a = {"epoch": stage2_epoch, "val_nll": val["nll"], "state_dict": snapshot}
                        notes.append(f"crossA@{stage2_epoch}")
                line = format_progress_line("stage2", stage2_epoch, fields, tuple(notes))
                print(line, flush=True)
                progress.write(line + "\n")
                progress.flush()
                if (args.stage2_fixed_epochs is None
                        and stage2_epoch >= args.stage2_min_epochs
                        and crossed_a is not None
                        and (threshold_b is None or crossed_b is not None)):
                    stage2_stop_reason = "all_thresholds_crossed"
                    break
            if args.stage2_fixed_epochs is not None:
                stage2_stop_reason = "fixed_epochs"
            progress.write(f"[stage2] done | epochs {stage2_epoch} | reason {stage2_stop_reason}\n")
            progress.flush()
        torch.save({"stage2_epoch": stage2_epoch, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict()}, run_dir / "stage2_final.pt")
        # The post-loop model is the terminal Stage-2 model: report it on the
        # validation and Session-1 test sets before loading departure snapshots.
        final_val = evaluate(model, val_loader, criterion, device, confusion=True)
        final_test = evaluate(model, test_loader, criterion, device, confusion=True)

        def session1_test(ckpt: dict | None) -> dict | None:
            """Session-1 test on one threshold-crossing checkpoint, else None."""
            if ckpt is None:
                return None
            model.load_state_dict(ckpt["state_dict"])
            return evaluate(model, test_loader, criterion, device, confusion=True)

        test_a = session1_test(crossed_a)
        test_b = session1_test(crossed_b)
        # Update final_summary.json with Stage 2 results.  validation_best is
        # deliberately left untouched: it is the Stage-1 selection, and keeping
        # it stable preserves the reproducibility of the screening comparison.
        summary["stage2"] = {
            "epochs": stage2_epoch,
            "stop_reason": stage2_stop_reason,
            "session0_train_size": len(train_data) + len(val_data),
            "thresholds": {
                "stage1_val_best_nll": {
                    "value": threshold_a,
                    "epoch": None if crossed_a is None else crossed_a["epoch"],
                    "val_nll": None if crossed_a is None else crossed_a["val_nll"],
                    "test": test_a,
                },
                "stage1_terminal_train_nll": {
                    "value": threshold_b,
                    "epoch": None if crossed_b is None else crossed_b["epoch"],
                    "val_nll": None if crossed_b is None else crossed_b["val_nll"],
                    "test": test_b,
                },
            },
            "val_final": final_val,
        }
        # Headline Session-1 test = the faithful FBNAS threshold (B) when it was
        # reached, else the terminal-model test.
        summary["test"] = test_b if test_b is not None else final_test
        summary["test_threshold"] = "terminal_train_nll" if test_b is not None else "final"
        summary["screening_only"] = False
        summary["test_observation_only"] = False
        summary["training_seconds"] = time.perf_counter() - started
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0

    # Resolve the genotype -- and apply the structural duplicate check -- before
    # the run directory is allocated.  A rejected genotype must not cost an
    # empty run directory: allocate() refuses to overwrite, so a leftover leaf
    # would make the corrected command fail on its second attempt.
    checkpoint_path: Path | None = None
    genotype_source: dict = {}
    band_genotype = None
    if args.genotype_json is not None:
        # One flag, three file dialects.  The anchored export carries a scheme
        # tag and stricter validation; the band export carries a different
        # grammar altogether (a family per band plus one or two RFs, rather than
        # one candidate per path); every other producer writes the plain
        # six-gene file.  Dispatch on the tag rather than on the caller, so a
        # plain file can never be misread as either tagged dialect.
        payload = json.loads(args.genotype_json.read_text(encoding="utf-8"))
        if payload.get("scheme") == ANCHORED_SCHEME:
            genotype = load_anchored_genotype(args.genotype_json)
            genotype_source = {"kind": "anchored_genotype_json", "path": str(args.genotype_json)}
        elif payload.get("scheme") == BAND_SCHEME:
            band_genotype = load_band_genotype(args.genotype_json)
            genotype = None
            genotype_source = {"kind": "band_genotype_json", "path": str(args.genotype_json)}
        else:
            genotype = load_genotype(args.genotype_json)
            genotype_source = {"kind": "genotype_json", "path": str(args.genotype_json)}
    else:
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
        genotype_source = {
            "kind": "darts_search_metrics",
            "path": str(metrics_path),
            "epoch": search_epochs,
        }
    # structure_keys is recorded either way: it is the evidence that the flag's
    # verdict came from built operators rather than from string equality.  The
    # band dialect owns its own reader because its candidates are (family, RF)
    # pairs, not 14-registry candidate strings, and because the V2/E families
    # carry support rather than structure_key.
    if band_genotype is not None:
        structure_keys = band_structure_keys(band_genotype)
        duplicate_bands = band_duplicate_bands(band_genotype)
        genotype_payload = band_genotype.to_dict()
    else:
        structure_keys = path_structure_keys(genotype)
        duplicate_bands = duplicate_structure_bands(genotype)
        genotype_payload = genotype.to_dict()
    if args.no_duplicate_paths and duplicate_bands:
        details = "; ".join(
            f"{band}: {structure_keys[band][0]} == {structure_keys[band][1]}"
            for band in duplicate_bands
        )
        raise ValueError(
            "--no-duplicate-paths: both paths realise the same structure in "
            f"{len(duplicate_bands)} band(s) -- {details}"
        )
    args.output_dir, progress_path = allocate(
        root=args.output_root, log_root=args.log_root, dataset=args.dataset, phase="train",
        subject=args.subject, seed=args.seed, arm=args.arm,
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"run dir: {args.output_dir}\nlog:     {progress_path}", flush=True)
    # Seeding happens HERE, immediately before the model is built, and not a
    # line earlier.  Resolving the genotype is not free with respect to the
    # global torch RNG: load_genotype and extract_genotype build the whole
    # candidate operator pool, and every nn.Conv2d.__init__ draws from that RNG.
    # So with set_seed placed before the resolution, a --search-dir run and a
    # --genotype-json run consume different numbers of draws and start from
    # DIFFERENT initial weights for the same --seed.  That silently breaks the
    # seed-matching every architecture comparison here rests on -- measured on
    # s009 at epoch 1: 2.78 vs 2.19 train NLL, from the seed alone.
    set_seed(args.seed)
    model = (
        BandDiscreteNet(band_genotype) if band_genotype is not None else TemporalDiscreteNet(genotype)
    ).to(device)
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
    # --screening-only must not merely skip the test *metric*; it must not read
    # the file.  Loading Session 1 and then declining to report it would leave
    # the closed set one careless line away from becoming a selection signal.
    test_data = (
        None
        if args.screening_only
        else load_subject_session(
            args.data_root, args.subject, session=1, preload=args.preload_data
        )
    )
    # --stage2-only also needs Session 1 for the final test evaluation.
    if test_data is None and args.stage2_only:
        test_data = load_subject_session(
            args.data_root, args.subject, session=1, preload=args.preload_data
        )
    generator = torch.Generator().manual_seed(args.seed)
    loader_kwargs = {"batch_size": args.batch_size, "num_workers": args.num_workers, "pin_memory": device.type == "cuda", "worker_init_fn": worker_init if args.num_workers else None}
    train_loader = DataLoader(train_data, shuffle=True, generator=generator, **loader_kwargs)
    val_loader = DataLoader(val_data, shuffle=False, **loader_kwargs)
    test_loader = None if test_data is None else DataLoader(test_data, shuffle=False, **loader_kwargs)
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
    config = {
        "args": serialise_args(args),
        "genotype": genotype_payload,
        "genotype_source": genotype_source,
        "path_structure_keys": {band: [list(key) for key in keys] for band, keys in structure_keys.items()},
        "duplicate_structure_bands": list(duplicate_bands),
        "session0_train_size": split.train_size,
        "session0_val_size": split.val_size,
        "session1_test_size": None if test_data is None else len(test_data),
        "transferred_state_keys": transferred,
        "parameters": parameters,
        "macs": macs,
    }
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    best_val_inacc = float("inf")
    best_score = float("inf")
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
            score = val_inacc if args.best_metric == "val_inacc" else val["nll"]
            record["best_metric"] = args.best_metric
            record["best_metric_value"] = score
            if score < best_score:
                best_score = score
                best_val_inacc = val_inacc
                best_epoch = epoch
                no_improvement = 0
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_inacc": val_inacc,
                        "val_nll": val["nll"],
                        "best_metric": args.best_metric,
                        "best_score": score,
                    },
                    args.output_dir / "best.pt",
                )
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
        progress.write(f"[stage1] done | stopEpoch {epoch} | bestEpoch {best_epoch} | bestValInacc {best_val_inacc:.4f} | bestScore({args.best_metric}) {best_score:.4f} | threshold {train_eval['nll']:.4f}\n")
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
    validation_best = None
    if args.screening_only:
        # No Stage 2 and no test pass.  The screening score is Stage 1's
        # validation-best epoch, already on disk in best.pt and summarised
        # below.  stage2_final.pt is deliberately left unwritten, so its absence
        # is itself the record that this run never entered the two-stage
        # protocol -- and Session 1 was never opened, not merely unreported.
        stage2_epoch = 0
        stage2_stop_reason = "skipped_screening_only"
        test = None
        # The screening read-out is the restored validation-best state, measured
        # with the same full-set pass the test evaluation uses (accuracy, NLL,
        # macro-F1, kappa).  A screening run has no test block, so without this
        # there would be no kappa for the operator-separability comparison at
        # all; it also keeps every arm's read-out on one code path.
        validation_best = evaluate(model, val_loader, criterion, device, confusion=True)
        # The Stage 1 `with` block closed `progress` on the way out, so this
        # needs its own handle rather than reusing the name.
        with progress_path.open("a", encoding="utf-8") as progress:
            progress.write("[screening-only] stage 2 skipped; Session 1 never loaded\n")
            progress.flush()
    else:
        stage2_generator = torch.Generator().manual_seed(args.seed + 1)
        all_session0_loader = DataLoader(
            ConcatDataset((train_data, val_data)), shuffle=True, generator=stage2_generator, **loader_kwargs
        )
        stage2_stop_reason = "max_epochs"
        # --stage2-fixed-epochs makes the budget the exact length and disables
        # the threshold break, so Stage 2 cannot end early on a subject whose
        # validation NLL dips below the Stage-1 threshold within a few epochs.
        stage2_limit = args.stage2_fixed_epochs or args.stage2_epochs
        with (args.output_dir / "metrics.jsonl").open("a", encoding="utf-8") as log, \
                progress_path.open("a", encoding="utf-8") as progress:
            for stage2_epoch in range(1, stage2_limit + 1):
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
                threshold_met = val["nll"] < stage1_terminal_train_nll
                past_floor = stage2_epoch >= args.stage2_min_epochs
                notes = [f"threshold {stage1_terminal_train_nll:.4f}"]
                if args.stage2_fixed_epochs is not None:
                    notes.append(f"fixed {stage2_epoch}/{stage2_limit}")
                elif args.stage2_min_epochs:
                    notes.append(f"floor {stage2_epoch}/{args.stage2_min_epochs}")
                if threshold_met:
                    if args.stage2_fixed_epochs is not None:
                        notes.append("thr-met")
                    elif past_floor:
                        notes.append("stop")
                    else:
                        # Named separately from "stop": the run continues, and
                        # a reader scanning the log must not think it ended.
                        notes.append("thr-met-held")
                line = format_progress_line("stage2", stage2_epoch, fields, tuple(notes))
                print(line, flush=True)
                progress.write(line + "\n")
                progress.flush()
                if args.stage2_fixed_epochs is None and threshold_met and past_floor:
                    stage2_stop_reason = "official_val_nll_lt_stage1_terminal_train_nll"
                    if args.stage2_min_epochs:
                        stage2_stop_reason += f"_after_floor{args.stage2_min_epochs}"
                    break
            if args.stage2_fixed_epochs is not None:
                stage2_stop_reason = "fixed_epochs"
            progress.write(f"[stage2] done | epochs {stage2_epoch} | reason {stage2_stop_reason} | threshold {stage1_terminal_train_nll:.4f}\n")
            progress.flush()
        torch.save({"stage2_epoch": stage2_epoch, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict()}, args.output_dir / "stage2_final.pt")
        test = evaluate(model, test_loader, criterion, device, confusion=True)
    summary = {
        "initialization": args.initialization, "genotype": genotype_payload,
        "genotype_source": genotype_source,
        "path_structure_keys": {band: [list(key) for key in keys] for band, keys in structure_keys.items()},
        "duplicate_structure_bands": list(duplicate_bands),
        "parameters": parameters, "macs": macs,
        "stage1": {"best_epoch": best_epoch, "stop_epoch": stage1_stop_epoch,
                   "best_val_inacc": best_val_inacc, "best_metric": args.best_metric,
                   "best_score": best_score},
        "stage2": {"epochs": stage2_epoch, "stop_reason": stage2_stop_reason, "session0_train_size": len(train_data) + len(val_data)},
        "test": test, "validation_best": validation_best,
        "training_seconds": time.perf_counter() - started,
        "test_observation_only": args.observe_test,
        "screening_only": args.screening_only,
    }
    (args.output_dir / "final_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
