#!/usr/bin/env python
"""Band-specific mechanism probe: one family per band, RF57, Session 1 closed.

The global grids asked whether *a family* is better than another one everywhere.
This one asks the narrower question they could not see: with the anchor in two
bands, does replacing the family in the *third* band help, and does the answer
depend on the subject and the band?

Design
------
The baseline is ``Low=dilated_e, Mid=dilated_e, High=dilated_e``; each
configuration swaps exactly one band and leaves the other two on the anchor.
Two- and three-band changes are refused, because a joint change cannot be
attributed to either band.  The nine configurations are
``{Low,Mid,High} x {dynamic_e, gated_e, band_gated_e}``, and the baseline is
*not* retrained: its nine runs (3 subjects x 3 seeds) already exist as the
Expressive ``dilated_e`` arm, and the analysis pairs against them.

``local_attention_e`` is out of scope.  It has the worst validation NLL in both
generations and the highest early-overfit rate, and it fits its training set --
so the reason to drop it is a generalisation verdict, not a tuning miss.  See
``tdarts.operator_v2e_band.EXCLUDED_FAMILIES``.

Everything except the per-band family is the Expressive protocol, byte for
byte: same Session-0 231/57 split, same RF57 / kernel 15 / dilation 4 support,
same backbone, same optimiser, LR, batch size, stopping rule and checkpoint
metric.  No family gets its own training hyper-parameters, and nothing here
searches RF.

Session 1 is closed
-------------------
A run reads Session 1 only when ``--read-session1`` is passed, and the probe
never passes it.  This stage is architecture-space diagnosis; a test number
produced while the space is still being screened is a selection signal waiting
to be used.  Cross-session evaluation happens once, later, after the search
method and the selection rule are frozen.

Seeding
-------
``set_seed`` is called immediately before the model is constructed, with nothing
RNG-consuming in between.  The capacity audit, which builds every family, runs
*after* the model for that reason -- see ``train_operator_v2e.py``'s docstring
for the measurement that made this a rule.

Staying out of the frozen trees
-------------------------------
Two guards, because the failure they prevent is silent.  ``--arm`` must start
with ``operator_v2e_band``, and the output root's last path component must too.
An Expressive run written under ``run/outputs/operator_v2/`` would be folded
into that stage's archived ANOVA; a band run written under
``run/outputs/operator_v2e/`` would be read as a global-family arm whose
``operator`` field is a configuration slug.  Both are one typo away, so both are
checked rather than documented.
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
from tdarts.operator_v2e import count_macs, operator_capacity_audit
from tdarts.operator_v2e_band import (
    ANCHOR_FAMILY,
    BAND_FAMILIES,
    EXCLUDED_FAMILIES,
    STAGE_PREFIX,
    config_of,
    slug_for,
    validate_band_families,
)
from tdarts.operator_v2e_band_network import OperatorV2EBandNet
from tdarts.search_data import load_session0_search_split, load_subject_session

#: Same fixed receptive field as the global grids, so a band effect is not an RF
#: effect.  57 is the middle rung and is exactly realisable at kernel 15
#: (dilation 4), which is the support every Expressive family already reads.
DEFAULT_RF = 57


def band_family(value: str) -> str:
    """``--low``/``--mid``/``--high`` argument type.

    ``local_attention_e`` is a real family in the Expressive registry, so a
    plain ``choices=BAND_FAMILIES`` would reject it with "invalid choice" and
    leave the reader to guess whether that was an oversight.  It was not, and
    the message says so.
    """

    if value in EXCLUDED_FAMILIES:
        raise argparse.ArgumentTypeError(
            f"{value!r} is out of scope for the band probe: {EXCLUDED_FAMILIES[value]}"
        )
    if value not in BAND_FAMILIES:
        raise argparse.ArgumentTypeError(f"{value!r} is not one of {BAND_FAMILIES}")
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    # One flag per band rather than a single "configuration" string: the shell
    # driver composes them from a (band, family) grid, and a per-band flag keeps
    # the two fixed bands visible in the command line that produced a run.
    for band in ("low", "mid", "high"):
        parser.add_argument(
            f"--{band}",
            type=band_family,
            default=ANCHOR_FAMILY,
            help=f"{band.capitalize()} band family (default: the {ANCHOR_FAMILY} anchor)",
        )
    parser.add_argument("--data-root", type=Path, default=Path("/root/autodl-tmp/bci42a/multiviewPython"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    parser.add_argument("--log-root", type=Path, default=Path("logs"))
    parser.add_argument("--dataset", default="bci42a")
    parser.add_argument(
        "--arm",
        default=STAGE_PREFIX,
        help="label prefixed to the configuration slug in the run leaf; must "
        f"start with {STAGE_PREFIX!r}",
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
        "default: this stage is a screening pass over the architecture space, "
        "and the test set is read once, later, under the frozen protocol.",
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
            raise FloatingPointError("non-finite band-probe training NLL")
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
    return f"[v2eband] epoch {epoch:04d} | {parts}{suffix}"


def main() -> int:
    args = parse_args()
    if args.epochs < 1 or args.patience < 1:
        raise ValueError("epochs and patience must be positive")
    # The configuration is checked first: it is a property of the request, and a
    # two-band change should be reported as one rather than as whatever
    # environment problem happens to be hit next.
    families = config_of(args.low, args.mid, args.high)
    varying_band, varying_family = validate_band_families(args.low, args.mid, args.high)
    if not args.arm.startswith(STAGE_PREFIX):
        # See the module docstring.  Cheap, and the failure it prevents is a
        # band run being read as a global-family arm.
        raise SystemExit(
            f"--arm must start with {STAGE_PREFIX!r}, got {args.arm!r}; a band "
            f"probe run must not be written into a global grid's tree"
        )
    if not Path(args.output_root).name.startswith(STAGE_PREFIX):
        raise SystemExit(
            f"--output-root must end in a directory starting with {STAGE_PREFIX!r}, "
            f"got {args.output_root}; the analysers locate their own stage by tree"
        )
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    target = slug_for(args.low, args.mid, args.high)
    arm = f"{args.arm}_{target}" if args.arm else target
    args.arm = arm
    args.target = target
    args.band_families = families
    args.varying_band = varying_band
    args.varying_family = varying_family
    args.output_dir, progress_path = allocate(
        root=args.output_root, log_root=args.log_root, dataset=args.dataset,
        phase="train", subject=args.subject, seed=args.seed, arm=arm,
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"run dir: {args.output_dir}\nlog:     {progress_path}", flush=True)

    # Seed immediately before the model: nothing that draws from the global
    # torch RNG may sit in between.  The capacity audit below builds every
    # family and therefore sits after this point on purpose.
    set_seed(args.seed)
    model = OperatorV2EBandNet(families, target_rf=args.rf).to(device)

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
    # One audit record per band, from the model actually built rather than read
    # off a table, so a changed architecture cannot leave a stale number in the
    # summary.  The wrapper is passed to count_macs, not the inner operator, so
    # it can reach `extra_macs` -- the elementwise work the hooks miss.
    band_audit: dict[str, dict] = {}
    band_macs: dict[str, int] = {}
    for band in model.bands:
        name = families[band]
        record = [
            row for row in operator_capacity_audit(band=band, target_rf=args.rf, time_length=1000)
            if row["operator"] == name
        ][0]
        band_audit[band] = record
        band_macs[band] = count_macs(model.cells[band].path, shape)
    varying_audit = None if varying_band is None else band_audit[varying_band]
    params = sum(p.numel() for p in model.parameters())
    config = {
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        # Subject, seed and the three band families are also written at the top
        # level, next to the tensor contract: this file is the provenance record
        # for one run, and reading the seed out of a nested `args` block is one
        # indirection more than a provenance record should need.
        "subject": str(args.subject),
        "seed": int(args.seed),
        "screening_only": test_data is None,
        "operator": target,
        "band_families": families,
        "varying_band": varying_band,
        "varying_family": varying_family,
        "target_rf": args.rf,
        "model": model.describe(),
        "parameters": params,
        "band_operator_params": {band: band_audit[band]["params"] for band in model.bands},
        "band_operator_macs": band_macs,
        "band_operator_support": {
            band: {
                "positions": band_audit[band]["support_positions"],
                "spacing": band_audit[band]["support_spacing"],
                "span": band_audit[band]["support_span"],
            }
            for band in model.bands
        },
        # The varying band's family is the only thing this grid moves, so its
        # capacity ratio is the one the paired reading has to be able to weigh.
        "varying_operator_capacity_ratio": (
            None if varying_audit is None else varying_audit["capacity_ratio"]
        ),
        "varying_operator_exceeds_advisory": (
            None if varying_audit is None else varying_audit["exceeds_advisory"]
        ),
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
        progress.write(f"# operator_v2e_band run   {args.output_dir.name}\n")
        progress.write(
            f"# config={target} band={varying_band} family={varying_family} "
            f"subject={args.subject} seed={args.seed} rf={args.rf}\n"
        )
        progress.write(f"# session1_opened={test_data is not None}\n")
        progress.flush()
        epoch = 0
        for epoch in range(1, args.epochs + 1):
            train = train_epoch(model, train_loader, optimizer, criterion, device)
            train_eval = evaluate(model, train_eval_loader, criterion, device)
            val = evaluate(model, val_loader, criterion, device)
            record = {
                "epoch": epoch,
                "operator": target,
                "varying_band": varying_band,
                "varying_family": varying_family,
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
            # Same selection metric as both global grids: validation NLL, which
            # is continuous where accuracy is quantised to 1/57.
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
                        "operator": target,
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
            f"[v2eband] done | stopEpoch {epoch} | bestEpoch {best_epoch} | bestValNLL {best_score:.4f} "
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
        "operator": target,
        "band_families": families,
        "varying_band": varying_band,
        "varying_family": varying_family,
        "target_rf": int(args.rf),
        "best_epoch": best_epoch,
        "val_best_nll": float(best_val["nll"]),
        "val_best_acc": float(best_val["acc"]),
        "params": params,
        "band_operator_params": {band: band_audit[band]["params"] for band in model.bands},
        "band_operator_macs": band_macs,
        "varying_operator_capacity_ratio": (
            None if varying_audit is None else varying_audit["capacity_ratio"]
        ),
        "support": {
            band: {
                "positions": band_audit[band]["support_positions"],
                "spacing": band_audit[band]["support_spacing"],
                "span": band_audit[band]["support_span"],
            }
            for band in model.bands
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
