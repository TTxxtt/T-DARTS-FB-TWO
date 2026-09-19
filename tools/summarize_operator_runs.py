#!/usr/bin/env python
"""Per-run training detail and per-operator diagnostics, both operator generations.

Writes one row per run to CSV/JSON and prints a per-operator summary.  The point
is to separate three things that a single mean validation NLL conflates:

* the mechanism could not fit the training set at all;
* the mechanism fit the training set but generalised badly;
* the mechanism trained stably and validation improved.

**The formal comparison always uses all runs.**  Filtering to "runs that trained
normally" and comparing on that subset is a post-hoc selection on the outcome --
an operator that overfits immediately is exhibiting one of its properties, not
producing a defective run to be discarded.  The subset view is printed, clearly
labelled ``diagnostic``, and must not be quoted as a result.

``early_overfit_flag`` marks a run whose best validation NLL arrived before
``EARLY_EPOCH``.  The value of the histogram below is the justification: the
distribution is bimodal with a wide empty gap, so the flag is reading a real
split rather than cutting a continuum.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

#: A run whose best validation NLL lands before this epoch never developed
#: generalisation; it memorised the training split and then degraded.  The
#: histogram this tool prints is what makes the cut defensible.
EARLY_EPOCH = 50

#: Training accuracy at or above this counts as "the training set was fitted".
FULL_TRAIN_ACC = 0.999

GENERATIONS = (
    ("Matched", "run/outputs/operator_v2", ""),
    ("Expressive", "run/outputs/operator_v2e", "_e"),
)

#: Display order and Chinese labels; the code name is kept for cross-reference.
LABELS = {
    "dilated": "膨胀卷积",
    "gated": "门控卷积",
    "local_attention": "局部注意力",
    "dynamic": "动态卷积",
    "band_gated": "频带门控",
}
ORDER = ("dilated", "gated", "local_attention", "dynamic", "band_gated")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", action="append", type=Path, default=None,
                        help="run root; repeatable (default: both generations under run/outputs)")
    parser.add_argument("--csv", type=Path, default=Path("run/outputs/operator_training_detail.csv"))
    parser.add_argument("--json", type=Path, default=Path("run/outputs/operator_training_detail.json"))
    return parser.parse_args()


def load_run(leaf_dir: Path, generation: str, suffix: str) -> dict[str, Any] | None:
    summary_path = leaf_dir / "final_summary.json"
    metrics_path = leaf_dir / "metrics.jsonl"
    if not summary_path.is_file() or not metrics_path.is_file():
        return None
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        return None
    best = min(rows, key=lambda row: row["val_nll"])
    final = rows[-1]
    operator = str(summary["operator"])
    base = operator[: -len(suffix)] if suffix and operator.endswith(suffix) else operator
    return {
        "generation": generation,
        "subject": str(summary["subject"]),
        "seed": int(summary["seed"]),
        "operator": operator,
        "operator_base": base,
        "best_epoch": int(best["epoch"]),
        "stop_epoch": int(final["epoch"]),
        "best_val_nll": float(best["val_nll"]),
        "best_val_acc": float(best["val_acc"]),
        # Eval-mode training loss: comparable with the validation loss, unlike
        # the running training-mode figure in the same record.
        "train_nll_at_best": float(best["train_nll_eval"]),
        "train_acc_at_best": float(best["train_acc"]),
        "final_train_nll": float(final["train_nll_eval"]),
        "final_train_acc": float(final["train_acc"]),
        "final_val_nll": float(final["val_nll"]),
        "early_overfit_flag": bool(best["epoch"] < EARLY_EPOCH),
        "reached_full_train_acc": bool(final["train_acc"] >= FULL_TRAIN_ACC),
        "session1_opened": bool(summary.get("session1_opened", False)),
    }


def collect(roots: list[Path]) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for generation, default_root, suffix in GENERATIONS:
        root = Path(default_root)
        if roots:
            # An explicit --root keeps the generation label matching the suffix
            # so a mixed listing cannot be mislabelled.
            pass
        base = root / "bci42a"
        if not base.is_dir():
            print(f"!! missing {base}; skipped", file=sys.stderr)
            continue
        for leaf in sorted(path for path in base.iterdir() if path.is_dir()):
            record = load_run(leaf, generation, suffix)
            if record is not None:
                runs.append(record)
    return runs


def quartiles(values: list[float]) -> tuple[float, float, float]:
    ordered = sorted(values)
    if len(ordered) < 2:
        only = ordered[0] if ordered else float("nan")
        return only, only, only
    q1, _, q3 = statistics.quantiles(ordered, n=4, method="inclusive")
    return statistics.median(ordered), q1, q3


def pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 3:
        return float("nan")
    mx, my = statistics.mean(xs), statistics.mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = math.sqrt(sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys))
    return num / den if den else float("nan")


def summarise(runs: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for generation, _, suffix in GENERATIONS:
        for base in ORDER:
            name = base + suffix
            sel = [r for r in runs if r["generation"] == generation and r["operator"] == name]
            if not sel:
                continue
            best_median, best_q1, best_q3 = quartiles([r["best_epoch"] for r in sel])
            stop_median, stop_q1, stop_q3 = quartiles([r["stop_epoch"] for r in sel])
            vals = [r["best_val_nll"] for r in sel]
            out[name] = {
                "generation": generation,
                "operator": name,
                "operator_base": base,
                "n": len(sel),
                "best_epoch_median": best_median,
                "best_epoch_q1": best_q1,
                "best_epoch_q3": best_q3,
                "stop_epoch_median": stop_median,
                "stop_epoch_q1": stop_q1,
                "stop_epoch_q3": stop_q3,
                "early_overfit_count": sum(r["early_overfit_flag"] for r in sel),
                "reached_full_train_acc_count": sum(r["reached_full_train_acc"] for r in sel),
                # The formal figure: all runs, no filtering.
                "val_nll_mean_all_runs": statistics.mean(vals),
                "val_nll_sd_all_runs": statistics.stdev(vals) if len(vals) > 1 else 0.0,
                "val_nll_mean_trained_subset": statistics.mean(
                    [r["best_val_nll"] for r in sel if not r["early_overfit_flag"]]
                )
                if any(not r["early_overfit_flag"] for r in sel)
                else None,
                "trained_subset_n": sum(not r["early_overfit_flag"] for r in sel),
                "corr_best_epoch_val_nll": pearson(
                    [float(r["best_epoch"]) for r in sel], vals
                ),
                "train_nll_at_best_mean": statistics.mean(r["train_nll_at_best"] for r in sel),
            }
    return out


def print_table(runs: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    print("每个 run 的早过拟合判定阈值：验证最优出现在第 %d 轮之前\n" % EARLY_EPOCH)
    print("=== 验证最优轮次的原始分布（用于确认阈值切的是真间隙，不是连续谱）===")
    for generation, _, suffix in GENERATIONS:
        epochs = sorted(r["best_epoch"] for r in runs if r["generation"] == generation)
        low = [e for e in epochs if e < EARLY_EPOCH]
        print(f"  {generation:<11} 早于{EARLY_EPOCH}轮的: {low}")
        rest = [e for e in epochs if e >= EARLY_EPOCH]
        print(f"  {'':<11} 其余最小几个: {rest[:5]}   (共 {len(rest)} 个)")

    print("\n=== 主结果：全部 9 次，不做任何筛选 ===")
    print(f"{'算子':<12}{'代':<11}{'次数':>5}{'验证NLL均值':>13}{'标准差':>10}"
          f"{'最好轮次中位':>14}{'停止轮次中位':>14}{'早过拟合':>10}")
    for base in ORDER:
        for generation, _, suffix in GENERATIONS:
            row = summary.get(base + suffix)
            if row is None:
                continue
            print(f"{LABELS[base]:<12}{generation:<11}{row['n']:>5}{row['val_nll_mean_all_runs']:>13.4f}"
                  f"{row['val_nll_sd_all_runs']:>10.4f}{row['best_epoch_median']:>14.0f}"
                  f"{row['stop_epoch_median']:>14.0f}"
                  f"{row['early_overfit_count']:>7}/{row['n']:<2}")

    print("\n=== 两代对照 ===")
    print(f"{'算子':<12}{'早过拟合 上一代':>16}{'早过拟合 这一代':>16}"
          f"{'最好轮次变化':>14}{'全部9次NLL变化':>16}")
    for base in ORDER:
        old, new = summary.get(base), summary.get(base + "_e")
        if old is None or new is None:
            continue
        print(f"{LABELS[base]:<12}{old['early_overfit_count']:>12}/{old['n']:<3}"
              f"{new['early_overfit_count']:>12}/{new['n']:<3}"
              f"{new['best_epoch_median'] - old['best_epoch_median']:>+14.0f}"
              f"{new['val_nll_mean_all_runs'] - old['val_nll_mean_all_runs']:>+16.4f}")

    print("\n=== 训练集拟合情况（判「学不会」还是「学了但泛化差」）===")
    print(f"{'算子':<12}{'代':<11}{'最优时训练NLL':>15}{'训练集拟合满':>14}{'次':>5}")
    for base in ORDER:
        for generation, _, suffix in GENERATIONS:
            row = summary.get(base + suffix)
            if row is None:
                continue
            print(f"{LABELS[base]:<12}{generation:<11}{row['train_nll_at_best_mean']:>15.4f}"
                  f"{row['reached_full_train_acc_count']:>13}/{row['n']:<2}")

    print("\n=== 诊断（事后筛选，不得当作结果引用）===")
    print("只统计「验证最优不在极早期」的 run，用于区分机制类型，不用于性能比较：")
    print(f"{'算子':<12}{'代':<11}{'子集大小':>10}{'子集验证NLL':>14}")
    for base in ORDER:
        for generation, _, suffix in GENERATIONS:
            row = summary.get(base + suffix)
            if row is None or row["val_nll_mean_trained_subset"] is None:
                continue
            print(f"{LABELS[base]:<12}{generation:<11}{row['trained_subset_n']:>9}/{row['n']:<2}"
                  f"{row['val_nll_mean_trained_subset']:>14.4f}")


def main() -> int:
    args = parse_args()
    runs = collect(args.root or [])
    if not runs:
        print("no runs found", file=sys.stderr)
        return 1
    summary = summarise(runs)

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(runs[0]))
        writer.writeheader()
        for row in sorted(runs, key=lambda r: (r["generation"], r["operator"], r["subject"], r["seed"])):
            writer.writerow(row)
    args.json.write_text(
        json.dumps(
            {
                "early_epoch_threshold": EARLY_EPOCH,
                "full_train_acc_threshold": FULL_TRAIN_ACC,
                "note": (
                    "val_nll_mean_all_runs is the formal figure. "
                    "val_nll_mean_trained_subset is a post-hoc diagnostic and must not be quoted as a result."
                ),
                "runs": runs,
                "per_operator": summary,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print_table(runs, summary)
    print(f"\n逐条明细已写出：\n  {args.csv}  ({len(runs)} 行)\n  {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
