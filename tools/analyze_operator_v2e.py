#!/usr/bin/env python
"""Variance decomposition for the Expressive-V2 pilot.

Reads one ``final_summary.json`` per run and asks the same question the Matched
tool asks -- is the *subject x operator* interaction larger than the *seed
noise*? -- but answers it differently in two ways that were pre-registered
before the grid was submitted.

Scale
-----
The primary scale is **``log(val_best_nll)``**.  The residual spread grows with
the mean (sd/mean was 0.062 / 0.091 / 0.102 across the three Matched subjects),
which is the signature of multiplicative noise, and log is its natural scale.
Raw NLL is reported alongside as a sensitivity analysis.

This matters more than it sounds.  On the Matched data the two scales disagree
about the headline: raw gives ``V_interaction / V_seed = 0.833`` and log gives
``1.187``.  A verdict read off one threshold on one scale would be a coin flip
dressed as a result, which is why the decision below no longer rests on that
ratio alone.

Note what the transform can and cannot move: ``log`` is strictly monotone, so
every per-seed operator ranking and every per-subject argmax is *identical* on
the two scales.  It can only move the variance decomposition -- criterion C1.
Criteria C2 and C3 are therefore scale-free, and they are what actually carry
the decision.

Decision
--------
Three conditions, all reported with their own numbers, rather than one ratio
compared against 1:

* **C1 -- is the interaction large enough?** ``V_SxO / V_seed >= 1.0`` *or*
  ``p(interaction) <= 0.10``.  Deliberately an OR of two soft signals: neither
  alone decides, and the OR is reported as such.
* **C2 -- do subjects genuinely differ?** At least two distinct per-subject
  winners, *and* at least two subjects whose best-minus-worst operator margin
  exceeds their own within-cell seed spread.  The second clause is what stops
  "two subjects swapped places inside the noise" counting as an interaction.
* **C3 -- is the winner stable?** No subject whose per-seed argmax changes
  across seeds, and a minimum per-subject Kendall tau of at least 0.5.

``PROCEED`` needs all three.  C1 without C2/C3 is ``AMBIGUOUS`` -- a real but
unspecialisable effect.  No C1 is ``WEAK_GLOBAL``.

Generation separation
---------------------
An operator's generation is derived from its name (an ``_e`` suffix means
Expressive), not from the directory it was found in, so a run that landed in the
wrong tree is still classified correctly.  Analysing runs from both generations
at once is refused unless ``--combine-generations`` is passed, because the two
are recorded on different scales and a pooled number is comparable to neither.

The Matched result (``run/outputs/operator_v2``, 45 runs) stands as recorded: a
``WEAK_GLOBAL`` verdict on the **raw** ``val_best_nll`` scale.  Its variance
components are not numerically comparable to this tool's; only the qualitative
direction is.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from run_layout import subject_id  # noqa: E402
from tdarts.operator_v2e import CAPACITY_ADVISORY_RATIO, E_OPERATOR_NAMES, WIDE_CONTROL_NAMES  # noqa: E402
from tools.operator_anova import (  # noqa: E402
    _fmt_ratio,
    generation_mix,
    print_anova,
    refuse_mixed_generations,
    two_way_anova,
    variance_ratio,
)

#: Canonical operator set of this generation; ``dilated_e`` is the anchor.
OPERATORS = E_OPERATOR_NAMES
ANCHOR = "dilated_e"

#: Metrics we know how to interpret, with the direction of "better".
#: ``log_val_best_nll`` is the pre-registered primary; the raw scale is kept
#: reachable so the sensitivity analysis is a real second computation rather
#: than a note.
METRIC_DIRECTION = {
    "log_val_best_nll": "lower",
    "val_best_nll": "lower",
    "val_best_acc": "higher",
}
PRIMARY_METRIC = "log_val_best_nll"

#: Metrics this tool computes rather than reads.  Kept apart from the ones that
#: must be present in ``final_summary.json``: requiring a derived key in the
#: file would make every run look incomplete, which is exactly what happened
#: the first time this was run against a populated grid.
DERIVED_METRICS = {"log_val_best_nll"}
FILE_METRICS = tuple(key for key in METRIC_DIRECTION if key not in DERIVED_METRICS)

#: Printed on every run of this tool.  The two generations are not on one scale
#: and the recorded Matched verdict is not up for revision by this tool.
GENERATION_STATEMENT = (
    "Frozen Matched-V2 numbers under run/outputs/operator_v2 are on the RAW val_best_nll scale "
    "and their recorded WEAK_GLOBAL verdict stands unchanged.  This tool's primary scale is "
    "log(val_best_nll); its variance components are not numerically comparable to the archived "
    "ones -- only the qualitative direction is."
)

#: C1 accepts either soft signal; C3's rank floor.
INTERACTION_P_THRESHOLD = 0.10
MIN_KENDALL_TAU = 0.5

LEAF_RE = re.compile(r"^train_s(?P<subject>[^_]+)_seed(?P<seed>[^_]+?)(?:_(?P<arm>.+))?$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--root",
        type=Path,
        action="append",
        default=None,
        help="output root holding <dataset>/train_s*_seed*_*/; repeatable "
        "(default: run/outputs/operator_v2e)",
    )
    parser.add_argument("--dataset", default="bci42a")
    parser.add_argument(
        "--metric",
        default=PRIMARY_METRIC,
        choices=sorted(METRIC_DIRECTION),
        help=f"metric to decompose (default: {PRIMARY_METRIC}, the pre-registered "
        "primary scale; val_best_nll is the sensitivity analysis)",
    )
    parser.add_argument("--json", type=Path, default=None, help="dump the full analysis here")
    parser.add_argument(
        "--combine-generations",
        action="store_true",
        help="allow runs from more than one operator generation in one "
        "decomposition.  Requires an explicit --metric, because the two "
        "generations are recorded on different scales and pooling them is only "
        "meaningful once somebody has chosen a common one on purpose.",
    )
    parser.add_argument(
        "--include-controls",
        action="store_true",
        help="include the wide-dilated capacity controls in the analysis.  OFF "
        "by default: they are ablation arms, not pilot candidates, and one "
        "control run would add an ANOVA level with mostly empty cells.",
    )
    parser.add_argument(
        "--min-runs-per-cell",
        type=int,
        default=2,
        help="a (subject, operator) cell with fewer seeds than this is flagged (default: 2)",
    )
    args = parser.parse_args()
    if args.root is None:
        args.root = [Path("run/outputs/operator_v2e")]
    return args


def _derive_log_metric(payload: dict[str, Any], path: Path) -> float | None:
    """``log(val_best_nll)``, or None if the summary cannot yield one.

    NLL is a positive loss, so a non-positive value means a corrupt or
    mislabelled summary.  Returning None makes the run a loud skip; letting a
    ``log`` of it through would put a NaN into the decomposition, where it
    poisons every component silently.
    """
    value = payload.get("val_best_nll")
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value) or value <= 0:
        print(f"!! {path}: val_best_nll={value!r} is not positive; cannot take log, skipped", file=sys.stderr)
        return None
    return math.log(value)


def discover_runs(roots: list[Path], dataset: str) -> tuple[list[dict[str, Any]], list[str]]:
    """Walk every root and load the runs; unreadable entries warn instead of raising."""

    runs: list[dict[str, Any]] = []
    warnings: list[str] = []
    seen: set[Path] = set()
    seen_cells: dict[tuple[str, int, str], str] = {}
    for root in roots:
        base = root / dataset if (root / dataset).is_dir() else root
        if not base.is_dir():
            warnings.append(f"root does not exist yet: {root}")
            print(f"!! root does not exist yet: {root}", file=sys.stderr)
            continue
        for path in sorted(base.glob("**/final_summary.json")):
            if path in seen:
                continue
            seen.add(path)
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                warnings.append(f"unreadable {path}: {exc}")
                print(f"!! unreadable {path}: {exc}", file=sys.stderr)
                continue
            match = LEAF_RE.match(path.parent.name)
            if match is None:
                warnings.append(f"unexpected leaf name: {path.parent.name}")
                print(f"!! unexpected leaf name, skipped: {path.parent}", file=sys.stderr)
                continue
            missing = [
                key
                for key in ("subject", "seed", "operator", "params", "macs", "operator_params", "operator_macs")
                if payload.get(key) is None
            ] + [key for key in FILE_METRICS if payload.get(key) is None]
            if missing:
                warnings.append(f"incomplete {path}: missing {missing}")
                print(f"!! incomplete, skipped: {path} (missing {missing})", file=sys.stderr)
                continue
            log_metric = _derive_log_metric(payload, path)
            if log_metric is None:
                warnings.append(f"{path}: no usable val_best_nll for the log scale")
                continue
            leaf_subject = subject_id(str(match.group("subject")))
            if leaf_subject != subject_id(str(payload["subject"])):
                warnings.append(f"{path}: leaf subject {leaf_subject} != json subject {payload['subject']}")
                print(
                    f"!! {path}: leaf subject {leaf_subject} != json subject {payload['subject']}; using json",
                    file=sys.stderr,
                )
            triple = (subject_id(str(payload["subject"])), int(payload["seed"]), str(payload["operator"]))
            if triple in seen_cells:
                # A duplicate would be counted as an extra replicate, shrinking
                # the residual MS that the whole decision rule is scaled by.
                message = f"{path}: duplicate of {seen_cells[triple]} for {triple}; skipped"
                warnings.append(message)
                print(f"!! {message}", file=sys.stderr)
                continue
            seen_cells[triple] = str(path)
            runs.append(
                {
                    "path": str(path),
                    "leaf": path.parent.name,
                    "arm": match.group("arm") or "",
                    "subject": subject_id(str(payload["subject"])),
                    "seed": int(payload["seed"]),
                    "operator": str(payload["operator"]),
                    "role": str(payload.get("operator_role", "candidate")),
                    "params": int(payload["params"]),
                    "macs": int(payload["macs"]),
                    "operator_params": int(payload["operator_params"]),
                    "operator_macs": int(payload["operator_macs"]),
                    "operator_param_ratio_vs_anchor": payload.get("operator_param_ratio_vs_anchor"),
                    "operator_capacity_ratio": payload.get("operator_capacity_ratio"),
                    "operator_exceeds_advisory": payload.get("operator_exceeds_advisory"),
                    "target_rf": payload.get("target_rf"),
                    "support": payload.get("support"),
                    "session1_opened": bool(payload.get("session1_opened", False)),
                    "screening_only": payload.get("screening_only"),
                    "test": payload.get("test"),
                    "training_seconds": payload.get("training_seconds"),
                    "log_val_best_nll": log_metric,
                    **{key: float(payload[key]) for key in FILE_METRICS},
                }
            )
    return runs, warnings


# ----------------------------------------------------------------------
# rank stability
# ----------------------------------------------------------------------
def kendall_tau(left: list[str], right: list[str]) -> float:
    """Kendall tau between two orderings of the same items.

    A 5-item ranking from 3 seeds is a noisy statistic, so this is the *soft*
    half of criterion C3 -- the argmax-agreement check is the hard one.  Ties
    cannot occur here: the values are means of floating-point losses.
    """
    if len(left) != len(right) or len(left) < 2:
        return float("nan")
    position = {name: index for index, name in enumerate(right)}
    concordant = discordant = 0
    for i in range(len(left)):
        for j in range(i + 1, len(left)):
            # left[i] is ranked ahead of left[j] by position in the list, so the
            # pair is concordant when `right` also places it earlier -- i.e.
            # when the difference in `right` positions is negative.
            delta = position[left[i]] - position[left[j]]
            if delta == 0:
                continue
            if delta < 0:
                concordant += 1
            else:
                discordant += 1
    total = concordant + discordant
    return (concordant - discordant) / total if total else float("nan")


def rank_stats(rows: list[dict[str, Any]], subjects: list[str], operators: list[str], metric: str) -> dict[str, Any]:
    """Per-subject ordering stability across seeds, and the per-seed winners."""

    result: dict[str, Any] = {"per_subject": {}, "min_tau": float("nan"), "hard_instability": []}
    taus: list[float] = []
    for subject in subjects:
        seed_rows = {}
        for seed in sorted({row["seed"] for row in rows if row["subject"] == subject}):
            seed_rows[seed] = sorted(
                [row for row in rows if row["subject"] == subject and row["seed"] == seed],
                key=lambda row: row[metric],
                reverse=METRIC_DIRECTION[metric] == "higher",
            )
        if len(seed_rows) < 2:
            continue
        winners = [ranked[0]["operator"] for ranked in seed_rows.values()]
        orderings = [[row["operator"] for row in ranked] for ranked in seed_rows.values()]
        pairwise = [
            kendall_tau(orderings[i], orderings[j])
            for i in range(len(orderings))
            for j in range(i + 1, len(orderings))
        ]
        pairwise = [tau for tau in pairwise if not math.isnan(tau)]
        mean_tau = statistics.mean(pairwise) if pairwise else float("nan")
        if not math.isnan(mean_tau):
            taus.append(mean_tau)
        unstable = len(set(winners)) > 1
        if unstable:
            result["hard_instability"].append(subject)
        result["per_subject"][subject] = {
            "seed_winners": winners,
            "unstable": unstable,
            "mean_tau": mean_tau,
        }
    if taus:
        result["min_tau"] = min(taus)
    return result


# ----------------------------------------------------------------------
# the three-part judgement
# ----------------------------------------------------------------------
def assess(
    *,
    interaction_over_seed: float | None,
    interaction_p: float | None,
    distinct_winners: int,
    separated_subjects: int,
    unstable_subjects: int,
    min_tau: float,
    n_subjects: int,
) -> dict[str, Any]:
    """The pre-registered three-part verdict, as a pure function.

    Pure so it can be tested on synthetic inputs: the whole point of replacing a
    single threshold is that the verdict is *not* a function of the variance
    ratio alone, and that property is only checkable if the decision is
    separable from the ANOVA that produced the inputs.
    """

    c1_ratio = interaction_over_seed is not None and interaction_over_seed >= 1.0
    c1_p = interaction_p is not None and interaction_p <= INTERACTION_P_THRESHOLD
    c1 = bool(c1_ratio or c1_p)

    c2_winners = distinct_winners >= 2
    # Requiring at least two subjects keeps a single outlying subject from
    # carrying the interaction on its own.
    c2_separation = separated_subjects >= min(2, n_subjects)
    c2 = bool(c2_winners and c2_separation)

    c3_stable = unstable_subjects == 0
    c3_rank = not math.isnan(min_tau) and min_tau >= MIN_KENDALL_TAU
    c3 = bool(c3_stable and c3_rank)

    if not c1:
        verdict = "WEAK_GLOBAL"
    elif c2 and c3:
        verdict = "PROCEED"
    else:
        verdict = "AMBIGUOUS"

    return {
        "c1_interaction_large_enough": c1,
        "c1_via_ratio": bool(c1_ratio),
        "c1_via_p": bool(c1_p),
        "c2_subjects_differ": c2,
        "c2_distinct_winners": distinct_winners,
        "c2_separated_subjects": separated_subjects,
        "c3_winner_stable": c3,
        "c3_unstable_subjects": unstable_subjects,
        "c3_min_kendall_tau": None if math.isnan(min_tau) else float(min_tau),
        "verdict": verdict,
    }


def print_cells(rows: list[dict[str, Any]], subjects: list[str], operators: list[str], metric: str) -> dict[str, Any]:
    print(f"\nper subject x operator: mean ± std of {metric} over seeds")
    header = f"{'subject':<9}" + "".join(f"{operator:>22}" for operator in operators)
    print(header)
    print("-" * len(header))
    cells: dict[str, Any] = {}
    for subject in subjects:
        line = f"{subject:<9}"
        for operator in operators:
            values = [row[metric] for row in rows if row["subject"] == subject and row["operator"] == operator]
            cell = {
                "n": len(values),
                "mean": statistics.mean(values) if values else None,
                "std": statistics.stdev(values) if len(values) > 1 else None,
            }
            cells[f"{subject}/{operator}"] = cell
            if not values:
                line += f"{'--':>22}"
            elif len(values) == 1:
                line += f"{values[0]:>22.4f}"
            else:
                line += f"{cell['mean']:>13.4f} ± {cell['std']:<6.4f}"
        print(line)
    return cells


def print_missing(rows: list[dict[str, Any]], min_runs: int) -> list[dict[str, Any]]:
    subjects = sorted({row["subject"] for row in rows})
    operators = sorted({row["operator"] for row in rows})
    counts = {
        (subject, operator): sum(1 for row in rows if row["subject"] == subject and row["operator"] == operator)
        for subject in subjects
        for operator in operators
    }
    missing = [{"subject": s, "operator": o, "n": 0} for (s, o), count in counts.items() if count == 0]
    thin = [{"subject": s, "operator": o, "n": count} for (s, o), count in counts.items() if 0 < count < min_runs]
    if missing:
        print(f"\nMISSING cells (no runs at all): {len(missing)}")
        for entry in missing:
            print(f"  !! subject {entry['subject']} operator {entry['operator']}: 0 seeds")
    if thin:
        print(f"\ncells with < {min_runs} seeds (unbalanced design, ANOVA weakened): {len(thin)}")
        for entry in thin:
            print(f"  !! subject {entry['subject']} operator {entry['operator']}: {entry['n']} seed(s)")
    if not missing and not thin:
        print(f"\ndesign is complete: {len(subjects)} subjects x {len(operators)} operators, >= {min_runs} seeds per cell")
    return missing + thin


def print_preference(
    rows: list[dict[str, Any]], subjects: list[str], operators: list[str], metric: str, ranks: dict[str, Any]
) -> dict[str, Any]:
    print(f"\nper-subject operator preference ({metric}; {'higher' if METRIC_DIRECTION[metric] == 'higher' else 'lower'} is better)")
    header = f"{'subject':<9}{'winner':<19}{'margin':>10}{'seedSD':>10}{'separated':>11}{'tau':>8}   ranking (best -> worst)"
    print(header)
    print("-" * len(header))
    per_subject: dict[str, Any] = {}
    for subject in subjects:
        means = {
            operator: statistics.mean(values)
            for operator in operators
            if (values := [row[metric] for row in rows if row["subject"] == subject and row["operator"] == operator])
        }
        if not means:
            continue
        ranked = sorted(means, key=means.get, reverse=METRIC_DIRECTION[metric] == "higher")
        # Always max - min, independent of metric direction, so a bigger margin
        # always means "the operators matter more for this subject".
        margin = max(means.values()) - min(means.values()) if len(ranked) > 1 else 0.0
        # The subject's own run-to-run scatter, pooled across operators: the
        # yardstick for whether the margin means anything.
        cell_stds = [
            statistics.stdev([row[metric] for row in rows if row["subject"] == subject and row["operator"] == operator])
            for operator in operators
            if len([row for row in rows if row["subject"] == subject and row["operator"] == operator]) > 1
        ]
        seed_sd = statistics.mean(cell_stds) if cell_stds else 0.0
        info = ranks["per_subject"].get(subject, {})
        per_subject[subject] = {
            "means": means,
            "ranking": ranked,
            "margin": margin,
            "seed_sd": seed_sd,
            "separated": margin > seed_sd,
            "winner": ranked[0],
            "seed_winners": info.get("seed_winners", []),
            "seed_unstable": info.get("unstable", False),
            "mean_tau": info.get("mean_tau", float("nan")),
        }
        ranking_text = " > ".join(f"{operator}({means[operator]:.4f})" for operator in ranked)
        tau = per_subject[subject]["mean_tau"]
        print(
            f"{subject:<9}{ranked[0]:<19}{margin:>10.4f}{seed_sd:>10.4f}"
            f"{('YES' if margin > seed_sd else 'no'):>11}{('--' if math.isnan(tau) else f'{tau:.2f}'):>8}   {ranking_text}"
        )
    return per_subject


def print_capacity(rows: list[dict[str, Any]], operators: list[str]) -> dict[str, Any]:
    """Capacity table.  No pass/fail band: the ceiling is advisory here."""

    print(f"\ncapacity vs the {ANCHOR} anchor (read from the runs, not from config)")
    header = (
        f"{'operator':<20}{'role':>11}{'opParams':>10}{'paramRatio':>12}"
        f"{'opMACs':>13}{'macRatio':>10}{'flag':>12}"
    )
    print(header)
    print("-" * len(header))

    def mean_of(operator: str, key: str) -> float | None:
        values = [row[key] for row in rows if row["operator"] == operator]
        return statistics.mean(values) if values else None

    anchor_params = mean_of(ANCHOR, "operator_params")
    anchor_macs = mean_of(ANCHOR, "operator_macs")
    capacity: dict[str, Any] = {}
    for operator in operators:
        params = mean_of(operator, "operator_params")
        macs = mean_of(operator, "operator_macs")
        param_ratio = params / anchor_params if params is not None and anchor_params else None
        mac_ratio = macs / anchor_macs if macs is not None and anchor_macs else None
        role = next((row["role"] for row in rows if row["operator"] == operator), "candidate")
        flag = ""
        if param_ratio is not None and param_ratio > CAPACITY_ADVISORY_RATIO:
            flag = "control" if role == "control" else f">{CAPACITY_ADVISORY_RATIO:.0f}x"
        capacity[operator] = {
            "role": role,
            "operator_params": params,
            "operator_macs": macs,
            "param_ratio_vs_anchor": param_ratio,
            "mac_ratio_vs_anchor": mac_ratio,
            "flag": flag,
        }
        print(
            f"{operator:<20}{role:>11}{'--' if params is None else f'{params:.0f}':>10}"
            f"{'--' if param_ratio is None else f'{param_ratio:.3f}':>12}"
            f"{'--' if macs is None else f'{macs:.0f}':>13}"
            f"{'--' if mac_ratio is None else f'{mac_ratio:.3f}':>10}{flag:>12}"
        )
    if anchor_params is None:
        print(f"!! anchor '{ANCHOR}' has no runs -- capacity ratios are not interpretable")
    else:
        print(
            f"the {CAPACITY_ADVISORY_RATIO:.0f}x ceiling is advisory and report-only; exceeding it is a "
            "finding to report, not a failure"
        )
        print("wide-dilated rows are ablation controls: they answer 'is it the mechanism or the capacity?'")
    return capacity


def main() -> int:
    args = parse_args()
    runs, warnings = discover_runs(args.root, args.dataset)
    print(f"operator_v2e variance decomposition | roots {[str(root) for root in args.root]} | dataset {args.dataset}")
    print(GENERATION_STATEMENT)
    print(f"metric: {args.metric} ({METRIC_DIRECTION[args.metric]} is better) | min runs per cell: {args.min_runs_per_cell}")

    if not runs:
        print(f"\nno runs found under {[str(root) for root in args.root]}/{args.dataset}")
        print("nothing to decompose -- this is the expected state before the pilot grid is submitted")
        if warnings:
            for warning in warnings:
                print(f"  note: {warning}")
        if args.json is not None:
            args.json.parent.mkdir(parents=True, exist_ok=True)
            args.json.write_text(
                json.dumps(
                    {"runs": [], "warnings": warnings, "metric": args.metric, "verdict": "NO_DATA"},
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            print(f"wrote {args.json}")
        return 1

    print(f"found {len(runs)} run(s)")

    mix = generation_mix(runs)
    if len(mix) > 1:
        print(f"\ngenerations present: {mix}")
        if args.combine_generations and args.metric == PRIMARY_METRIC:
            # Pooling is only coherent once someone has chosen a common scale.
            print(
                "!! --combine-generations was passed but --metric is still the "
                "default.  The generations are recorded on different scales; pass "
                "--metric explicitly to say which one you are pooling on.",
                file=sys.stderr,
            )
            return 2
    if not refuse_mixed_generations(runs, combine=args.combine_generations):
        return 2

    # The pilot is only valid while Session 1 stays closed.
    opened = [row for row in runs if row["session1_opened"]]
    if opened:
        print("\n" + "!" * 78)
        print(f"!! WARNING: Session 1 was opened in {len(opened)} run(s) -- the pilot is only valid")
        print("!! while the test set is closed. Exclude these runs before any decision.")
        for row in opened:
            print(f"!!   {row['leaf']} ({row['path']})")
        print("!" * 78)

    controls = [row for row in runs if row["operator"] in WIDE_CONTROL_NAMES]
    # The capacity table is a reporting artefact, so it keeps the controls even
    # when the ANOVA drops them: a control's cost is exactly what a reader
    # needs in order to interpret a candidate that beat the anchor.  Only the
    # statistics see the filtered set.
    capacity_runs = list(runs)
    if controls and not args.include_controls:
        names = sorted({row["operator"] for row in controls})
        print(f"\ncapacity control run(s) present ({len(controls)}) and excluded from the pilot ANOVA: {names}")
        print("pass --include-controls to analyse them alongside the candidates")
        runs = [row for row in runs if row["operator"] not in WIDE_CONTROL_NAMES]

    subjects = sorted({str(row["subject"]) for row in runs})
    observed = {str(row["operator"]) for row in runs}
    known = [operator for operator in OPERATORS if operator in observed]
    operators = known + sorted(observed - set(OPERATORS))
    unknown = sorted(observed - set(OPERATORS))
    if unknown:
        print(f"\n!! operator(s) outside the documented set {OPERATORS}: {unknown}")

    missing = print_missing(runs, args.min_runs_per_cell)
    cells = print_cells(runs, subjects, operators, args.metric)
    anova = two_way_anova(runs, args.metric)
    ranks = rank_stats(runs, subjects, operators, args.metric)
    per_subject = print_preference(runs, subjects, operators, args.metric, ranks)
    observed_capacity = {str(row["operator"]) for row in capacity_runs}
    capacity_operators = [name for name in OPERATORS if name in observed_capacity] + sorted(
        observed_capacity - set(OPERATORS)
    )
    capacity = print_capacity(capacity_runs, capacity_operators)

    # The sensitivity analysis is a second full decomposition, not a footnote.
    anova_raw = two_way_anova(runs, "val_best_nll") if args.metric != "val_best_nll" else None

    verdict_payload: dict[str, Any] = {"verdict": "NO_DATA"}
    ratios: dict[str, float | None] = {"interaction_over_seed": None, "operator_over_seed": None, "subject_over_seed": None}
    if anova is None:
        print("\nverdict: not computable -- the design has no residual degrees of freedom")
        verdict_payload = {"verdict": "NOT_COMPUTABLE"}
    else:
        if anova_raw is not None:
            print_anova(anova_raw)
            print("  ^ sensitivity: the same decomposition on the RAW val_best_nll scale")
        print_anova(anova)
        print(f"  ^ primary scale: {args.metric}")
        components = anova["components"]
        ratios = {
            "interaction_over_seed": variance_ratio(components["interaction"], components["seed"]),
            "operator_over_seed": variance_ratio(components["operator"], components["seed"]),
            "subject_over_seed": variance_ratio(components["subject"], components["seed"]),
        }
        print("\nvariance ratios (component / V_seed, the run-to-run noise scale)")
        print(f"  V_subject x operator / V_seed : {_fmt_ratio(ratios['interaction_over_seed'])}")
        print(f"  V_operator          / V_seed : {_fmt_ratio(ratios['operator_over_seed'])}")
        print(f"  V_subject           / V_seed : {_fmt_ratio(ratios['subject_over_seed'])}")

        distinct_winners = sorted({entry["winner"] for entry in per_subject.values()})
        separated = sum(1 for entry in per_subject.values() if entry["separated"])
        unstable = sum(1 for entry in per_subject.values() if entry["seed_unstable"])
        p_interaction = anova["p"].get("interaction") if anova["scipy"] else None

        verdict_payload = assess(
            interaction_over_seed=ratios["interaction_over_seed"],
            interaction_p=p_interaction,
            distinct_winners=len(distinct_winners),
            separated_subjects=separated,
            unstable_subjects=unstable,
            min_tau=ranks["min_tau"],
            n_subjects=len(per_subject),
        )

        print("\nverdict -- three conditions, each reported rather than thresholded once")
        print(
            f"  C1 interaction large enough : "
            f"{'MET' if verdict_payload['c1_interaction_large_enough'] else 'NOT MET'} "
            f"(ratio {_fmt_ratio(ratios['interaction_over_seed'])} >= 1.0: "
            f"{'yes' if verdict_payload['c1_via_ratio'] else 'no'}; "
            f"p {('--' if p_interaction is None else f'{p_interaction:.4g}')} <= {INTERACTION_P_THRESHOLD}: "
            f"{'yes' if verdict_payload['c1_via_p'] else 'no'})"
        )
        print(
            f"  C2 subjects differ          : "
            f"{'MET' if verdict_payload['c2_subjects_differ'] else 'NOT MET'} "
            f"({len(distinct_winners)} distinct winner(s) {distinct_winners}, "
            f"{separated} subject(s) with margin > their own seed sd)"
        )
        print(
            f"  C3 winner stable            : "
            f"{'MET' if verdict_payload['c3_winner_stable'] else 'NOT MET'} "
            f"({unstable} subject(s) flip their argmax across seeds; "
            f"min tau {verdict_payload['c3_min_kendall_tau']})"
        )
        print(f"\n  VERDICT: {verdict_payload['verdict']}")
        if verdict_payload["verdict"] == "WEAK_GLOBAL":
            print(
                "  No family separates globally on the primary scale.  As in the "
                "Matched stage this bounds GLOBAL family separability only: a "
                "band-specific preference averages out across the three bands and "
                "is invisible to this design.  The follow-up is the band-specific "
                "probe, not abandoning the per-band search."
            )
        elif verdict_payload["verdict"] == "AMBIGUOUS":
            print(
                "  A real interaction that does not currently support per-subject "
                "specialisation: either the winners do not differ enough between "
                "subjects, or they are not stable across seeds.  More seeds would "
                "separate the two -- instability shrinks, a genuine absence does not."
            )
        else:
            print(
                "  All three hold: the interaction survives seed noise, subjects "
                "prefer different families, and those preferences are stable across "
                "seeds.  A per-subject architecture search has something to find."
            )
        if not anova["balanced"]:
            print("  caveat: cell seed counts differ; the Type III SS and EMS components are approximations")
        if missing:
            print(f"  caveat: {len(missing)} cell(s) are missing or under-seeded (listed above)")

    if warnings:
        print(f"\n{len(warnings)} warning(s) during discovery (see stderr)")

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(
                {
                    "roots": [str(root) for root in args.root],
                    "dataset": args.dataset,
                    "metric": args.metric,
                    "primary_metric": PRIMARY_METRIC,
                    "generation_statement": GENERATION_STATEMENT,
                    "generations": mix,
                    "runs": runs,
                    "cells": cells,
                    "missing_cells": missing,
                    "anova": anova,
                    "anova_raw_sensitivity": anova_raw,
                    "ratios": {key: (None if value is None or math.isinf(value) else value) for key, value in ratios.items()},
                    "ratios_infinite": {key: bool(value is not None and math.isinf(value)) for key, value in ratios.items()},
                    "per_subject": per_subject,
                    "rank_stability": ranks,
                    "capacity": capacity,
                    "verdict_detail": verdict_payload,
                    "session1_opened_runs": [row["leaf"] for row in opened],
                    "warnings": warnings,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
