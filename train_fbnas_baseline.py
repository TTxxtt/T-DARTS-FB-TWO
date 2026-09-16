#!/usr/bin/env python
"""Protocol cross-check for the FBNAS arm -- not the baseline the paper reports.

The headline FBNAS baseline comes from running the authors' own code:
``run/py/run_fbnas_subject.py`` imports ``ho.py`` and lets upstream drive both
the search and the two-phase final training.  That is the number to report.

This script is the secondary source.  It re-implements the same two-phase
protocol on top of the ``tdarts`` data pipeline, so running it on one subject
and finding it lands where the official run landed is evidence that
``train_retrain.py`` -- which implements that same protocol for the DARTS arm --
is faithful rather than merely plausible.  Use it as the agreement check, not
as the baseline.

The search phase calls the unmodified upstream ``NAS.nas_phase`` implementation:
200 epochs of random MixPath training followed by its 1,000-candidate traversal
with candidate-specific BN calibration.  The selected FBNASNet inherits the
upstream name-matched supernet weights exactly as ``ho.py`` does.

Its final supervised training reproduces upstream's two-phase protocol
(``continueAfterEarlystop=True``): ordered Session-0 80/20 training stopped on
validation accuracy, restoration of the best validation checkpoint, then a
second phase that merges the validation trials back into training and stops as
soon as the validation loss falls below Stage 1's terminal train loss, or after
600 epochs.  Stage 1's train loss is measured the way upstream measures it --
a second, frozen, eval-mode pass over the whole training set (baseModel.py:374
-> predict(), which calls net.eval() at :572) -- so the threshold and the
validation loss it is compared against are the same kind of quantity.

This is the same protocol the T-DARTS arm runs in ``train_retrain.py``, so the
two are directly comparable.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import random
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader

from run_layout import allocate
from tdarts.search_data import load_session0_search_split, load_subject_session
from train_retrain import count_macs, evaluate, format_progress_line, set_seed, train_epoch, worker_init


ROOT = Path(__file__).resolve().parent
OFFICIAL_REPO = ROOT / "FBNAS" / "codes"
OFFICIAL_CENTRAL = OFFICIAL_REPO / "centralRepo"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("/root/autodl-tmp/bci42a/multiviewPython"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs"), help="root of the <dataset>/<phase>/<leaf> run tree")
    parser.add_argument("--log-root", type=Path, default=Path("logs"), help="root of the mirrored per-epoch progress log tree")
    parser.add_argument("--dataset", default="bci42a")
    parser.add_argument("--arm", default="fbnas", help="label appended to the run leaf; distinguishes arms inside one train/ directory")
    parser.add_argument("--subject", default="003")
    parser.add_argument("--seed", type=int, default=20190821)
    parser.add_argument("--initialization", choices=("random", "transfer"), default="random")
    parser.add_argument("--search-epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-epochs", type=int, default=1500)
    parser.add_argument("--patience", type=int, default=200)
    parser.add_argument("--stage2-epochs", type=int, default=600)
    parser.add_argument("--observe-test", action="store_true", help="log Session1 metrics every epoch for observation only; never select on them.  Off by default so the baseline stays blind.")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def subject_id(subject: str | int) -> str:
    return f"{int(subject):03d}" if str(subject).isdigit() else str(subject)


@contextmanager
def official_nas_arguments(args: argparse.Namespace) -> Iterator[None]:
    """Supply argv consumed internally by the frozen ``NAS.nas_phase``."""

    original_argv = sys.argv
    sys.argv = [
        "NAS.py",
        "--seed", str(args.seed),
        "--epochs", str(args.search_epochs),
        "--batch_size", str(args.batch_size),
        "--learning_rate", str(args.lr),
        "--m", "2",
    ]
    try:
        yield
    finally:
        sys.argv = original_argv


def import_official_modules():
    """Import the upstream modules without changing files under ``FBNAS/``."""

    # Importing from inside FBNAS/ otherwise drops __pycache__/*.pyc into the
    # frozen baseline, which trips
    # tests/test_fbnas_compatibility.py::test_no_extra_files_were_written_into_the_baseline.
    # Same idiom as test_fbnas_compatibility.py:69-79.  Set rather than restored,
    # so it also covers every module the official code imports in turn.
    sys.dont_write_bytecode = True
    central = str(OFFICIAL_CENTRAL)
    if central not in sys.path:
        sys.path.insert(0, central)
    import NAS  # noqa: PLC0415
    import networks  # noqa: PLC0415
    from eegDataset import eegDataset  # noqa: PLC0415

    return NAS, networks, eegDataset


def official_subject_split(eeg_dataset_cls, data_root: Path, subject: str):
    """Reproduce the official ``ho.py`` Session-0 ordered 80/20 split."""

    all_data = eeg_dataset_cls(
        dataPath=str(data_root),
        dataLabelsPath=str(data_root / "dataLabels.csv"),
        preloadData=False,
    )
    indices = [index for index, label in enumerate(all_data.labels) if label[3] == subject]
    if len(indices) != 576:
        raise ValueError(f"expected 576 trials for Subject{subject}, found {len(indices)}")
    all_data.createPartialDataset(indices, loadNonLoadedData=True)
    session0 = copy.deepcopy(all_data)
    session0.createPartialDataset([index for index, label in enumerate(session0.labels) if label[4] == "0"])
    if len(session0) != 288:
        raise ValueError(f"expected 288 Session-0 trials, found {len(session0)}")
    boundary = int(np.ceil(len(session0) * 0.8))
    validation = copy.deepcopy(session0)
    validation.createPartialDataset(list(range(boundary, len(session0))))
    session0.createPartialDataset(list(range(boundary)))
    return session0, validation, boundary


def serialise_args(args: argparse.Namespace) -> dict[str, object]:
    return {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}


def main() -> int:
    args = parse_args()
    if args.search_epochs < 1 or args.max_epochs < 1 or args.patience < 1:
        raise ValueError("epoch counts and patience must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    data_root = args.data_root.expanduser().resolve()
    if not (data_root / "dataLabels.csv").is_file():
        raise FileNotFoundError(f"missing labels file under {data_root}")
    # The FBNAS arm runs its search and its final training in one invocation, so
    # the two phases sit side by side under the same dataset directory and pair
    # up by their shared s<subject>_seed<seed> suffix.
    search_dir, _ = allocate(
        root=args.output_root, log_root=args.log_root, dataset=args.dataset, phase="search",
        subject=args.subject, seed=args.seed, arm=args.arm,
    )
    args.output_dir, progress_path = allocate(
        root=args.output_root, log_root=args.log_root, dataset=args.dataset, phase="train",
        subject=args.subject, seed=args.seed, arm=args.arm,
    )
    search_dir.mkdir(parents=True, exist_ok=False)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"search dir: {search_dir}\nrun dir:    {args.output_dir}\nlog:        {progress_path}", flush=True)
    subject = subject_id(args.subject)
    NAS, networks, eeg_dataset_cls = import_official_modules()

    # Search: upstream code, upstream layout, upstream ordered Session-0 split.
    set_seed(args.seed)
    search_train, search_validation, boundary = official_subject_split(eeg_dataset_cls, data_root, subject)
    supernet = networks.SuperNet(nChan=22, nTime=1000, nClass=4, nBands=9).to(device)
    search_started = time.perf_counter()
    with official_nas_arguments(args):
        opt_choice, calibration_accuracy = NAS.nas_phase(
            supernet,
            search_train,
            search_validation,
            classes=4,
            supernet_path=str(search_dir / "supernet.pth"),
        )
    search_seconds = time.perf_counter() - search_started
    calibration = [float(value) for value in calibration_accuracy]
    (search_dir / "opt_choice.json").write_text(
        json.dumps(opt_choice, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    np.save(search_dir / "cali_bn_acc.npy", np.asarray(calibration, dtype=np.float64))

    # ``ho.py`` transfers every state_dict entry whose name matches.  Keep that
    # exact rule rather than trying to transfer only visibly selected nodes.
    #
    # Under ``--initialization random`` the supernet is deliberately NOT
    # inherited: the freshly built FBNASNet keeps its own init and the search
    # only supplies the architecture.  Re-seed first so that init does not
    # depend on how much randomness the search happened to consume.
    set_seed(args.seed)
    model = networks.FBNASNet(nChan=22, nTime=1000, nClass=4, nBands=9, opt_choice=opt_choice).to(device)
    transfer: dict[str, torch.Tensor] = {}
    if args.initialization == "transfer":
        target_state = model.state_dict()
        source_state = torch.load(search_dir / "supernet.pth", map_location=device)
        transfer = {key: value for key, value in source_state.items() if key in target_state and target_state[key].shape == value.shape}
        target_state.update(transfer)
        model.load_state_dict(target_state)
        if set(transfer) != set(target_state):
            missing = sorted(set(target_state) - set(transfer))
            raise RuntimeError(f"official name-matched transfer unexpectedly missed: {missing}")

    # Final stage: ordered Session-0 80/20 with valInacc early stopping, then
    # upstream's continueAfterEarlystop phase -- the same two-phase protocol the
    # T-DARTS arm runs in train_retrain.py.
    set_seed(args.seed)
    train_data, validation_data, split = load_session0_search_split(data_root, subject)
    test_data = load_subject_session(data_root, subject, session=1)
    generator = torch.Generator().manual_seed(args.seed)
    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "worker_init_fn": worker_init if args.num_workers else None,
    }
    train_loader = DataLoader(train_data, shuffle=True, generator=generator, **loader_kwargs)
    validation_loader = DataLoader(validation_data, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_data, shuffle=False, **loader_kwargs)
    # Frozen-model loss passes, kept on separate shuffle=False loaders so they
    # never draw from the training generator's random stream.
    train_eval_loader = DataLoader(train_data, shuffle=False, **loader_kwargs)
    session0_eval_loader = DataLoader(ConcatDataset((train_data, validation_data)), shuffle=False, **loader_kwargs)
    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    macs = count_macs(model, device)
    config = {
        "args": serialise_args(args),
        "official_source": "FBNAS/codes/centralRepo/NAS.py::nas_phase",
        "search_protocol": "official MixPath random subnet training + 1000 candidate BN-calibrated traversal",
        "final_training_protocol": (
            "upstream continueAfterEarlystop=True: ordered Session0 80/20 stopped on valInacc patience, "
            "restore best checkpoint, then merge Session0 train+val and stop on "
            "valLoss < Stage1 terminal frozen-model trainLoss, max 600 epochs"
        ),
        "subject": subject,
        "session0_train_size": split.train_size,
        "session0_validation_size": split.val_size,
        "session1_test_size": len(test_data),
        "search_split_boundary": boundary,
        "opt_choice": opt_choice,
        "transferred_state_keys": sorted(transfer),
        "parameters": parameters,
        "macs": macs,
    }
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    best_val_inacc = float("inf")
    best_epoch = 0
    no_improvement = 0
    train_started = time.perf_counter()
    with (args.output_dir / "metrics.jsonl").open("x", encoding="utf-8") as log, \
            progress_path.open("w", encoding="utf-8") as progress:
        progress.write(f"# run            {args.output_dir.name} (FBNAS arm)\n")
        progress.write(f"# subject={subject} seed={args.seed} initialization={args.initialization}\n")
        progress.write(f"# max_epochs={args.max_epochs} patience={args.patience} stage2_epochs={args.stage2_epochs}\n")
        progress.write("# trainLoss      online per-batch mean collected inside train_epoch()\n")
        progress.write("# trainLossEval  frozen eval-mode pass over the whole training set == upstream's trainLoss\n")
        progress.write(f"# observe_test={args.observe_test} -- Session 1 metrics are recorded for observation only; "
                       "no stop, checkpoint or selection decision reads them.\n")
        progress.flush()
        for epoch in range(1, args.max_epochs + 1):
            train = train_epoch(model, train_loader, optimizer, criterion, device)
            train_eval = evaluate(model, train_eval_loader, criterion, device)
            validation = evaluate(model, validation_loader, criterion, device)
            val_inacc = 1.0 - validation["acc"]
            is_best = val_inacc < best_val_inacc
            record = {
                "stage": "stage1",
                "epoch": epoch,
                "train_nll": train["nll"],
                "train_acc": train["acc"],
                "train_nll_eval": train_eval["nll"],
                "val_nll": validation["nll"],
                "val_acc": validation["acc"],
                "val_inacc": val_inacc,
                "seconds": time.perf_counter() - train_started,
                "is_best": is_best,
            }
            if is_best:
                best_val_inacc = val_inacc
                best_epoch = epoch
                no_improvement = 0
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_inacc": val_inacc,
                    },
                    args.output_dir / "best.pt",
                )
            else:
                no_improvement += 1
            if args.observe_test:
                record["test_observation"] = evaluate(model, test_loader, criterion, device)
            log.write(json.dumps(record, sort_keys=True) + "\n")
            log.flush()
            fields = {"trainLoss": train["nll"], "trainAcc": train["acc"], "trainLossEval": train_eval["nll"], "valLoss": validation["nll"], "valAcc": validation["acc"]}
            if args.observe_test:
                fields["testAcc"] = record["test_observation"]["acc"]
                fields["testLoss"] = record["test_observation"]["nll"]
            notes = []
            if is_best:
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

    # Upstream's continueAfterEarlystop phase (baseModel.py:402-422): restore the
    # best validation checkpoint *and* its optimizer state, merge the validation
    # trials back into training, then stop as soon as the validation loss drops
    # below Stage 1's terminal train loss.
    best = torch.load(args.output_dir / "best.pt", map_location=device)
    model.load_state_dict(best["model_state_dict"])
    optimizer.load_state_dict(best["optimizer_state_dict"])
    stage1_stop_epoch = epoch
    stage1_terminal_train_nll = train_eval["nll"]
    stage2_generator = torch.Generator().manual_seed(args.seed + 1)
    all_session0_loader = DataLoader(
        ConcatDataset((train_data, validation_data)), shuffle=True, generator=stage2_generator, **loader_kwargs
    )
    stage2_stop_reason = "max_epochs"
    with (args.output_dir / "metrics.jsonl").open("a", encoding="utf-8") as log, \
            progress_path.open("a", encoding="utf-8") as progress:
        for stage2_epoch in range(1, args.stage2_epochs + 1):
            train = train_epoch(model, all_session0_loader, optimizer, criterion, device)
            train_eval = evaluate(model, session0_eval_loader, criterion, device)
            validation = evaluate(model, validation_loader, criterion, device)
            record = {
                "stage": "stage2",
                "stage2_epoch": stage2_epoch,
                "train_nll": train["nll"],
                "train_acc": train["acc"],
                "train_nll_eval": train_eval["nll"],
                "val_nll": validation["nll"],
                "val_acc": validation["acc"],
                "seconds": time.perf_counter() - train_started,
            }
            if args.observe_test:
                record["test_observation"] = evaluate(model, test_loader, criterion, device)
            log.write(json.dumps(record, sort_keys=True) + "\n")
            log.flush()
            fields = {"trainLoss": train["nll"], "trainAcc": train["acc"], "trainLossEval": train_eval["nll"], "valLoss": validation["nll"], "valAcc": validation["acc"]}
            if args.observe_test:
                fields["testAcc"] = record["test_observation"]["acc"]
                fields["testLoss"] = record["test_observation"]["nll"]
            notes = [f"threshold {stage1_terminal_train_nll:.4f}"]
            if validation["nll"] < stage1_terminal_train_nll:
                notes.append("stop")
            line = format_progress_line("stage2", stage2_epoch, fields, tuple(notes))
            print(line, flush=True)
            progress.write(line + "\n")
            progress.flush()
            if validation["nll"] < stage1_terminal_train_nll:
                stage2_stop_reason = "official_val_nll_lt_stage1_terminal_train_nll"
                break
        progress.write(f"[stage2] done | epochs {stage2_epoch} | reason {stage2_stop_reason} | threshold {stage1_terminal_train_nll:.4f}\n")
        progress.flush()

    test = evaluate(model, test_loader, criterion, device, confusion=True)
    summary = {
        "subject": subject,
        "seed": args.seed,
        "opt_choice": opt_choice,
        "search_seconds": search_seconds,
        "calibration_candidates": len(calibration),
        "calibration_best_acc": max(calibration),
        "parameters": parameters,
        "macs": macs,
        "stage1": {
            "best_epoch": best_epoch,
            "stop_epoch": stage1_stop_epoch,
            "best_val_inacc": best_val_inacc,
            "terminal_train_nll": stage1_terminal_train_nll,
        },
        "stage2": {
            "epochs": stage2_epoch,
            "stop_reason": stage2_stop_reason,
            "session0_train_size": len(train_data) + len(validation_data),
        },
        "test": test,
        "final_training_seconds": time.perf_counter() - train_started,
    }
    (args.output_dir / "final_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
