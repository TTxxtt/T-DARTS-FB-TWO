#!/usr/bin/env python
"""Summarise the architecture-landscape screening sweep.

Answers, for the subject set that was pre-registered before any score existed:

  * how do the sampled architectures spread out, and how much of that spread is
    architecture rather than seed noise;
  * where does each DARTS Top-1 genotype land inside that spread;
  * how consistent is the ranking of architectures across seeds.

Every run it reads is checked to be a ``--screening-only`` run with Session 1
never loaded.  That check is the point of reading ``config.json`` rather than
trusting the directory name: the whole experiment is only valid if the test set
stayed closed, so a single run that does not prove it is a hard error.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RANDOM_PREFIX = "g"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT / "run" / "outputs" / "landscape")
    parser.add_argument("--dataset", default="bci42a")
    parser.add_argument(
        "--darts-arms",
        default="darts_200ep,darts_lr1e3",
        help="comma-separated arm names whose searched Top-1 genotype is compared",
    )
    parser.add_argument("--json-out", type=Path, default=None)
    return parser.parse_args()


def spearman(xs: list[float], ys: list[float]) -> float:
    """Spearman rho with average ranks for ties, so it needs no scipy."""

    def ranks(values: list[float]) -> list[float]:
        order = sorted(range(len(values)), key=lambda i: values[i])
        out = [0.0] * len(values)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
                j += 1
            average = (i + j) / 2 + 1
            for k in range(i, j + 1):
                out[order[k]] = average
            i = j + 1
        return out

    if len(xs) < 3:
        return float("nan")
    rx, ry = ranks(xs), ranks(ys)
    mx, my = statistics.mean(rx), statistics.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else float("nan")


def variance_components(groups: dict[str, list[float]]) -> dict[str, float]:
    """One-way random-effects split of total variance into architecture and seed.

    ``Var_seed`` is the pooled within-architecture variance.  ``Var_arch`` is the
    between-architecture component, ``(MS_between - MS_within) / n_seeds``
    clamped at zero -- negative estimates are what a genuinely flat landscape
    looks like, and reporting them as negative variances would be meaningless.
    """
    usable = {k: v for k, v in groups.items() if len(v) >= 2}
    if len(usable) < 2:
        return {"var_arch": float("nan"), "var_seed": float("nan"), "icc": float("nan")}
    sizes = {len(v) for v in usable.values()}
    grand = statistics.mean([x for v in usable.values() for x in v])
    ms_between = sum(
        len(v) * (statistics.mean(v) - grand) ** 2 for v in usable.values()
    ) / (len(usable) - 1)
    ss_within = sum(
        sum((x - statistics.mean(v)) ** 2 for x in v) for v in usable.values()
    )
    df_within = sum(len(v) - 1 for v in usable.values())
    ms_within = ss_within / df_within
    n_seeds = statistics.mean(sizes)
    var_seed = ms_within
    var_arch = max(0.0, (ms_between - ms_within) / n_seeds)
    total = var_arch + var_seed
    return {
        "var_arch": var_arch,
        "var_seed": var_seed,
        "icc": var_arch / total if total else float("nan"),
        "sd_arch": var_arch ** 0.5,
        "sd_seed": var_seed ** 0.5,
    }


def load_runs(root: Path, dataset: str, darts_arms: set[str]) -> list[dict]:
    runs = []
    for run_dir in sorted((root / dataset).glob("train_s*")):
        summary_path = run_dir / "final_summary.json"
        config_path = run_dir / "config.json"
        if not summary_path.is_file() or not config_path.is_file():
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        config = json.loads(config_path.read_text(encoding="utf-8"))
        # The validity condition for the whole experiment.
        if not summary.get("screening_only"):
            raise SystemExit(f"{run_dir} is not a --screening-only run; refusing to summarise")
        if config.get("session1_test_size") is not None:
            raise SystemExit(f"{run_dir} loaded Session 1 (session1_test_size is set); refusing to summarise")
        arm = config["args"]["arm"]
        runs.append(
            {
                "dir": run_dir.name,
                "subject": config["args"]["subject"],
                "seed": int(config["args"]["seed"]),
                "arm": arm,
                "kind": "random" if arm.startswith(RANDOM_PREFIX) and arm[1:].isdigit() else "search",
                "val_acc": 1.0 - float(summary["stage1"]["best_val_inacc"]),
                "best_epoch": summary["stage1"]["best_epoch"],
                "stop_epoch": summary["stage1"]["stop_epoch"],
                "genotype": tuple(g["candidate_index"] for g in summary["genotype"]["genes"]),
                "candidates": tuple(g["candidate"] for g in summary["genotype"]["genes"]),
            }
        )
    return runs


def main() -> int:
    args = parse_args()
    arms = {a for a in args.darts_arms.split(",") if a}
    runs = load_runs(args.root, args.dataset, arms)
    if not runs:
        raise SystemExit(f"no completed runs under {args.root / args.dataset}")

    subjects = sorted({r["subject"] for r in runs})
    random_runs = [r for r in runs if r["kind"] == "random"]
    search_runs = [r for r in runs if r["kind"] == "search"]
    seeds = sorted({r["seed"] for r in runs})
    genotypes = sorted({r["arm"] for r in random_runs})

    expected = len(subjects) * len(genotypes) * len(seeds)
    print(f"subjects={subjects}  seeds={seeds}")
    print(f"sampled genotypes: {len(genotypes)}   runs: {len(random_runs)}/{expected} random, {len(search_runs)} searched")

    by_genotype: dict[tuple[str, str], list[float]] = {}
    for run in random_runs:
        by_genotype.setdefault((run["subject"], run["arm"]), []).append(run["val_acc"])
    incomplete = {k: v for k, v in by_genotype.items() if len(v) != len(seeds)}
    if incomplete:
        print(f"\n!! incomplete (expected {len(seeds)} seeds each):")
        for (subject, arm), values in sorted(incomplete.items()):
            print(f"   s{subject} {arm}: {len(values)} seed(s)")

    print("\n" + "=" * 78)
    print("1+5. per-genotype 3-seed mean/std and labelled six-tuple")
    print("=" * 78)
    tuples: dict[str, tuple[int, ...]] = {}
    for run in random_runs:
        tuples.setdefault(run["arm"], run["genotype"])
    for subject in subjects:
        print(f"\n-- subject {subject}")
        print(f"   {'arm':<6} {'mean':>7} {'sd':>7} {'n':>2}   six-tuple (High0,High1,Low0,Low1,Mid0,Mid1)")
        for arm in genotypes:
            values = by_genotype.get((subject, arm), [])
            if not values:
                continue
            mean = statistics.mean(values)
            sd = statistics.stdev(values) if len(values) > 1 else float("nan")
            print(f"   {arm:<6} {mean:7.4f} {sd:7.4f} {len(values):>2}   {tuples[arm]}")

    print("\n" + "=" * 78)
    print("2. random-architecture distribution")
    print("=" * 78)
    for subject in subjects:
        values = [r["val_acc"] for r in random_runs if r["subject"] == subject]
        if not values:
            continue
        values_sorted = sorted(values)
        print(
            f"   subject {subject}: n={len(values)}  mean={statistics.mean(values):.4f}  "
            f"sd={statistics.stdev(values) if len(values) > 1 else float('nan'):.4f}  "
            f"min={min(values):.4f}  median={statistics.median(values):.4f}  max={max(values):.4f}"
        )

    print("\n" + "=" * 78)
    print("4. architecture variance vs seed variance")
    print("=" * 78)
    pooled: list[dict[str, float]] = []
    for subject in subjects:
        groups = {a: by_genotype[(subject, a)] for a in genotypes if (subject, a) in by_genotype}
        stats_ = variance_components(groups)
        pooled.append(stats_)
        print(
            f"   subject {subject}: sd_arch={stats_['sd_arch']:.4f}  sd_seed={stats_['sd_seed']:.4f}  "
            f"ICC={stats_['icc']:.3f}"
        )
    if pooled and all(p["icc"] == p["icc"] for p in pooled):
        print(
            f"   mean ICC over subjects: {statistics.mean(p['icc'] for p in pooled):.3f}"
            "   (>0.5 = architecture differences dominate seed noise)"
        )

    print("\n" + "=" * 78)
    print("3. DARTS Top-1 percentile inside the random distribution")
    print("=" * 78)
    if not search_runs:
        print("   (no searched-genotype runs found)")
    # The reference distribution is one score per sampled *architecture*, averaged
    # over seeds -- not the 3x larger pool of individual runs.  Pooling runs would
    # widen the reference with seed noise and quietly make the searched genotype
    # look better than it is.
    for subject in subjects:
        means = [
            statistics.mean(by_genotype[(subject, arm)])
            for arm in genotypes
            if (subject, arm) in by_genotype
        ]
        if not means:
            continue
        for arm in sorted({r["arm"] for r in search_runs}):
            searched = [r for r in search_runs if r["subject"] == subject and r["arm"] == arm]
            if not searched:
                continue
            score = statistics.mean([r["val_acc"] for r in searched])
            below = sum(1 for v in means if v < score)
            print(
                f"   subject {subject} {arm:<12} searched={score:.4f}  "
                f"sampled mean={statistics.mean(means):.4f}  "
                f"below {below}/{len(means)} sampled  ->  {100.0 * below / len(means):.0f}th percentile"
            )
            # Per-seed ranks, so a single lucky training seed cannot manufacture
            # the comparison on its own: each is out of the sampled genotypes at
            # that same seed.
            per_seed = []
            for seed in seeds:
                s = [r["val_acc"] for r in searched if r["seed"] == seed]
                same_seed = [r["val_acc"] for r in random_runs if r["subject"] == subject and r["seed"] == seed]
                if s and same_seed:
                    per_seed.append(sum(1 for v in same_seed if v < s[0]))
            if per_seed:
                print(f"      per-seed rank among the {len(means)} sampled: {per_seed}")

    print("\n" + "=" * 78)
    print("   architecture ranking consistency across seeds (Spearman)")
    print("=" * 78)
    for subject in subjects:
        per_seed = {
            seed: {
                r["arm"]: r["val_acc"]
                for r in random_runs
                if r["subject"] == subject and r["seed"] == seed
            }
            for seed in seeds
        }
        shared = [a for a in genotypes if all(a in per_seed[s] for s in seeds)]
        for i, a in enumerate(seeds):
            for b in seeds[i + 1:]:
                rho = spearman([per_seed[a][g] for g in shared], [per_seed[b][g] for g in shared])
                print(f"   subject {subject}  seed {a} vs {b}: rho={rho:+.3f}  (n={len(shared)})")

    if args.json_out:
        payload = {
            "subjects": subjects,
            "seeds": seeds,
            "genotypes": {a: list(tuples[a]) for a in genotypes},
            "runs": [{k: (list(v) if isinstance(v, tuple) else v) for k, v in r.items()} for r in runs],
        }
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
