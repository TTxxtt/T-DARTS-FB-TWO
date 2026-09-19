#!/usr/bin/env python
"""Variance decomposition for the Operator Separability V2 pilot.

The pilot fixes one temporal operator family across all three bands (arm
``operator_v2``) and repeats the run for several subjects, seeds and operators.
Each run writes one ``final_summary.json``.  This tool reads those summaries and
answers a single question:

    is the *subject x operator interaction* larger than the *seed noise*?

That is deliberately **not** "which operator is best on average".  A main
effect of operator that is identical for every subject is useless for a
per-subject NAS: the search would pick the same family for everyone, and the
whole NAS stage has nothing subject-specific to find.  A family preference is
only worth searching over if it *changes* with the subject -- i.e. if the
interaction variance exceeds the run-to-run (seed) variance.

Model
-----
With ``y[i, j, k]`` the metric for subject ``i``, operator ``j``, seed ``k``::

    y[i, j, k] = mu + a_i + b_j + (ab)_ij + e_ijk

The three sums of squares are computed as **Type III** sums of squares from an
effect-coded least-squares fit (+1 for a level, -1 for the reference level, and
the product of the two codings for the interaction): the SS of a term is the
increase in residual sum of squares when that term is dropped from the full
model while every other term stays in.  For a balanced grid this is identical
to the textbook two-way ANOVA closed forms; for an unbalanced grid it is the
usual (approximate) generalisation, which is why unbalanced cells are printed
loudly -- they weaken the ANOVA, they do not merely shrink n.

Variance components use the expected-mean-square (EMS) method, so the
components are on a common "variance of a single run" scale and can be compared
against each other::

    V_seed         = MS_residual
    V_interaction  = (MS_interaction - MS_residual) / n_rep
    V_operator     = (MS_operator    - MS_interaction) / (n_subjects * n_rep)
    V_subject      = (MS_subject     - MS_interaction) / (n_operators * n_rep)

where ``n_rep`` is the number of seeds per cell (harmonic mean of the per-cell
counts when the grid is unbalanced, which is the standard unbalanced EMS
substitute).  Negative estimates are clamped at 0 -- the clamp is reported
whenever it fires, because a clamped component means "no detectable effect",
not "a small negative one".

Decision rule (Tier-1 global pilot)
-----------------------------------
    V_subject x operator >  V_seed  -> PROCEED, but only if the second
        condition also holds: the best family differs across subjects (>= 2
        distinct per-subject argmax winners).  A large interaction driven by
        two subjects swapping places is still an interaction, but a single
        winner means there is nothing to specialise.
    V_subject x operator <= V_seed  -> WEAK_GLOBAL: no family preference
        survives seed noise *when one family serves all three bands at once*.

The second outcome is **not** a verdict on the NAS idea, and the tool refuses
to word it as one.  Each run in this pilot varies a single family across Low,
Mid and High simultaneously, so a preference that is band-specific -- low
wants attention, high wants something else -- cancels out across the three
bands and cannot appear here.  WEAK_GLOBAL therefore bounds *global* family
separability only.  The next experiment is a band-specific probe (vary one
band, pin the other two to the anchor), not abandoning the per-band search;
see ``docs/operator_separability_v2.md``.

Metric
------
``test`` is null in every pilot run on purpose (Session 1 is never opened).
The **primary metric is ``val_best_nll``**.  The validation split holds 57
trials, so a single trial moves accuracy by ~1/57 = 1.75pp and a variance
decomposition over that granularity measures quantisation as much as signal.
NLL is continuous and is what the training loop already selects on, so it is
the metric the decomposition runs on; ``val_best_acc`` is accepted and
reported alongside as the secondary, human-readable number.
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

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from run_layout import subject_id  # noqa: E402
from tools.operator_anova import (  # noqa: E402
    _effect_coding,
    _fmt_ratio,
    _sse,
    generation_mix,
    print_anova,
    refuse_mixed_generations,
    two_way_anova,
    variance_ratio,
)

# Re-exported from the shared module so the rest of this file is unchanged.
from tools.operator_anova import _scipy_stats, np  # noqa: E402, F401

#: Canonical operator set of the pilot; ``dilated`` is the FBNAS anchor.
OPERATORS = ("dilated", "gated", "local_attention", "dynamic", "band_gated")
ANCHOR = "dilated"

#: Metrics we know how to interpret, with the direction of "better".
METRIC_DIRECTION = {
    "val_best_acc": "higher",
    "val_best_nll": "lower",
}

#: Parameter fairness band vs the anchor (mirrors tdarts.operator_v2.PARAM_TOLERANCE).
PARAM_TOLERANCE = 0.20
#: Ratio of the anchor's MACs outside which a family is flagged.  All five
#: families are matched by construction -- the widest is attention at 1.272x --
#: so this is a regression detector, not a budget check: it fires if a family
#: drifts far enough that a score difference could be read as "more arithmetic"
#: rather than "better mechanism".
MAC_FLAG_RATIO = 1.5

LEAF_RE = re.compile(r"^train_s(?P<subject>[^_]+)_seed(?P<seed>[^_]+?)(?:_(?P<arm>.+))?$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--root",
        type=Path,
        action="append",
        default=None,
        help="output root holding <dataset>/train_s*_seed*_*/; repeatable "
        "(default: run/outputs/operator_v2)",
    )
    parser.add_argument("--dataset", default="bci42a")
    parser.add_argument(
        "--metric",
        default="val_best_nll",
        choices=sorted(METRIC_DIRECTION),
        help="metric to decompose (default: val_best_nll -- accuracy is "
        "quantised to 1/57 on the validation split and is reported alongside "
        "as the secondary metric)",
    )
    parser.add_argument("--json", type=Path, default=None, help="dump the full analysis here")
    parser.add_argument(
        "--combine-generations",
        action="store_true",
        help="allow runs from more than one operator generation in one "
        "decomposition.  OFF by default and deliberately awkward: the two "
        "generations are recorded on different metric scales, so a pooled "
        "number is comparable to neither.  This tool's default root holds only "
        "the Matched generation, so the guard never fires unless a root is "
        "pointed somewhere it should not be.",
    )
    parser.add_argument(
        "--min-runs-per-cell",
        type=int,
        default=2,
        help="a (subject, operator) cell with fewer seeds than this is flagged (default: 2)",
    )
    args = parser.parse_args()
    if args.root is None:
        args.root = [Path("run/outputs/operator_v2")]
    return args


def _best(rows: list[dict[str, Any]], metric: str) -> dict[str, Any]:
    pick = max if METRIC_DIRECTION[metric] == "higher" else min
    return pick(rows, key=lambda row: row[metric])


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
            # The directory name is authoritative for the arm (the JSON does not
            # carry it); subject/seed come from the JSON and are cross-checked
            # against the leaf so a renamed directory cannot silently mis-group.
            match = LEAF_RE.match(path.parent.name)
            if match is None:
                warnings.append(f"unexpected leaf name: {path.parent.name}")
                print(f"!! unexpected leaf name, skipped: {path.parent}", file=sys.stderr)
                continue
            missing = [
                key
                for key in ("subject", "seed", "operator", "params", "macs", "operator_params", "operator_macs")
                if payload.get(key) is None
            ] + [key for key in METRIC_DIRECTION if payload.get(key) is None]
            if missing:
                warnings.append(f"incomplete {path}: missing {missing}")
                print(f"!! incomplete, skipped: {path} (missing {missing})", file=sys.stderr)
                continue
            leaf_subject = subject_id(str(match.group("subject")))
            if leaf_subject != subject_id(str(payload["subject"])):
                warnings.append(f"{path}: leaf subject {leaf_subject} != json subject {payload['subject']}")
                print(
                    f"!! {path}: leaf subject {leaf_subject} != json subject {payload['subject']}; using json",
                    file=sys.stderr,
                )
            # Two paths for the same (subject, seed, operator) would be counted as
            # two replicates, silently inflating n and shrinking the residual MS.
            triple = (subject_id(str(payload["subject"])), int(payload["seed"]), str(payload["operator"]))
            if triple in seen_cells:
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
                    "params": int(payload["params"]),
                    "macs": int(payload["macs"]),
                    "operator_params": int(payload["operator_params"]),
                    "operator_macs": int(payload["operator_macs"]),
                    "operator_param_ratio_vs_anchor": payload.get("operator_param_ratio_vs_anchor"),
                    "target_rf": payload.get("target_rf"),
                    "support": payload.get("support"),
                    "session1_opened": bool(payload.get("session1_opened", False)),
                    "screening_only": payload.get("screening_only"),
                    "test": payload.get("test"),
                    "training_seconds": payload.get("training_seconds"),
                    **{key: float(payload[key]) for key in METRIC_DIRECTION},
                }
            )
    return runs, warnings


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


def print_missing(runs: list[dict[str, Any]], min_runs: int) -> list[dict[str, Any]]:
    subjects = sorted({row["subject"] for row in runs})
    operators = sorted({row["operator"] for row in runs})
    counts = {
        (subject, operator): sum(1 for row in runs if row["subject"] == subject and row["operator"] == operator)
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


def print_preference(rows: list[dict[str, Any]], subjects: list[str], operators: list[str], metric: str) -> dict[str, Any]:
    print(f"\nper-subject operator preference ({metric}; {'higher' if METRIC_DIRECTION[metric] == 'higher' else 'lower'} is better)")
    header = f"{'subject':<9}{'winner':<17}{'spread':>9}{'seed-unstable':>15}   ranking (best -> worst)"
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
        # Always max - min, independent of the metric direction, so a bigger
        # spread always means "the operators matter more for this subject".
        spread = max(means.values()) - min(means.values()) if len(ranked) > 1 else 0.0
        # A winner that changes between seeds of the same subject is direct
        # evidence the "preference" is seed noise, not a subject trait.
        seed_winners = []
        for seed in sorted({row["seed"] for row in rows if row["subject"] == subject}):
            seed_rows = [row for row in rows if row["subject"] == subject and row["seed"] == seed]
            if seed_rows:
                seed_winners.append(_best(seed_rows, metric)["operator"])
        unstable = len(set(seed_winners)) > 1
        per_subject[subject] = {
            "means": means,
            "ranking": ranked,
            "spread": spread,
            "winner": ranked[0],
            "seed_winners": seed_winners,
            "seed_unstable": unstable,
        }
        ranking_text = " > ".join(f"{operator}({means[operator]:.4f})" for operator in ranked)
        print(
            f"{subject:<9}{ranked[0]:<17}{spread:>9.4f}{('YES ' + str(seed_winners)) if unstable else 'no':>15}"
            f"   {ranking_text}"
        )
    return per_subject


def print_fairness(rows: list[dict[str, Any]], operators: list[str]) -> dict[str, Any]:
    print("\nfairness vs the dilated anchor (read from the runs, not from config)")
    header = (
        f"{'operator':<17}{'opParams':>10}{'paramRatio':>12}{'paramFlag':>11}"
        f"{'opMACs':>12}{'macRatio':>10}{'macFlag':>10}"
    )
    print(header)
    print("-" * len(header))

    def mean_of(operator: str, key: str) -> float | None:
        values = [row[key] for row in rows if row["operator"] == operator]
        return statistics.mean(values) if values else None

    anchor_params = mean_of(ANCHOR, "operator_params")
    anchor_macs = mean_of(ANCHOR, "operator_macs")
    fairness: dict[str, Any] = {}
    for operator in operators:
        params = mean_of(operator, "operator_params")
        macs = mean_of(operator, "operator_macs")
        param_ratio = params / anchor_params if params is not None and anchor_params else None
        mac_ratio = macs / anchor_macs if macs is not None and anchor_macs else None
        param_flag = ""
        if param_ratio is not None:
            param_flag = "" if abs(param_ratio - 1.0) <= PARAM_TOLERANCE else f"OUT +/-{PARAM_TOLERANCE:.0%}"
        mac_flag = ""
        if mac_ratio is not None:
            mac_flag = (
                "" if abs(mac_ratio - 1.0) <= MAC_FLAG_RATIO - 1.0 else f"dev {mac_ratio:.2f}x"
            )
        fairness[operator] = {
            "operator_params": params,
            "operator_macs": macs,
            "param_ratio_vs_anchor": param_ratio,
            "mac_ratio_vs_anchor": mac_ratio,
            "param_flag": param_flag,
            "mac_flag": mac_flag,
        }
        print(
            f"{operator:<17}{'--' if params is None else f'{params:.0f}':>10}"
            f"{'--' if param_ratio is None else f'{param_ratio:.3f}':>12}{param_flag:>11}"
            f"{'--' if macs is None else f'{macs:.0f}':>12}"
            f"{'--' if mac_ratio is None else f'{mac_ratio:.3f}':>10}{mac_flag:>10}"
        )
    if anchor_params is None:
        print(f"!! anchor '{ANCHOR}' has no runs -- fairness ratios are not interpretable")
    else:
        print(f"param flag = outside +/-{PARAM_TOLERANCE:.0%} of the anchor's parameters (fairness violation)")
        print(
            f"macFlag fires outside {MAC_FLAG_RATIO:.2f}x the anchor's MACs; all five families are matched "
            "by construction, so a flag means a regression rather than a known deviation"
        )
    return fairness


def main() -> int:
    args = parse_args()
    runs, warnings = discover_runs(args.root, args.dataset)
    print(f"operator_v2 variance decomposition | roots {[str(root) for root in args.root]} | dataset {args.dataset}")
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

    # Generation separation.  This is inert on the default root -- every run
    # under run/outputs/operator_v2 is Matched -- so it cannot change the
    # recorded reading.  It exists to catch the one thing that would: an
    # Expressive run that landed in this tree, which the directory walk below
    # would otherwise fold into the archived ANOVA as an unknown operator.
    mix = generation_mix(runs)
    if len(mix) > 1:
        print(f"\ngenerations present: {mix}")
    if not refuse_mixed_generations(runs, combine=args.combine_generations):
        return 2

    # The pilot is only valid while Session 1 stays closed; a run that opened it
    # has a test number the rest of the grid cannot be compared against.
    opened = [row for row in runs if row["session1_opened"]]
    if opened:
        print("\n" + "!" * 78)
        print(f"!! WARNING: Session 1 was opened in {len(opened)} run(s) -- the pilot is only valid")
        print("!! while the test set is closed. Exclude these runs before any decision.")
        for row in opened:
            print(f"!!   {row['leaf']} ({row['path']})")
        print("!" * 78)

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
    per_subject = print_preference(runs, subjects, operators, args.metric)
    fairness = print_fairness(runs, operators)

    verdict = "NO_DATA"
    ratios: dict[str, float | None] = {
        "interaction_over_seed": None,
        "operator_over_seed": None,
        "subject_over_seed": None,
    }
    distinct_winners: list[str] = []
    if anova is None:
        print("\nverdict: not computable -- the design has no residual degrees of freedom")
        verdict = "NOT_COMPUTABLE"
    else:
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
        print("\nverdict")
        if components["interaction"] <= components["seed"]:
            verdict = "WEAK_GLOBAL"
            print(
                "  WEAK_GLOBAL -- global family separability is weak. V_subject x operator "
                f"({components['interaction']:.6f}) does not exceed V_seed ({components['seed']:.6f}) when one "
                "family serves all three bands at once. More seeds will not rescue a zero interaction."
            )
            print(
                "  This is NOT a verdict on the NAS idea, and it does not falsify a per-band family search. "
                "A preference that is band-specific -- low band wanting attention while the high band wants "
                "something else -- averages out across the three bands and is invisible to this design by "
                "construction. The claim that is supported here is only: no family separates globally. "
                "Next step is the band-specific probe (vary one band, pin the other two to the anchor), "
                "which is what a per-band 5^3 family search actually has to beat; see "
                "docs/operator_separability_v2.md."
            )
        elif len(distinct_winners) < 2:
            verdict = "PROCEED_WEAK"
            print(
                "  PROCEED (primary condition met) but the second condition FAILS: "
                f"V_subject x operator ({components['interaction']:.6f}) > V_seed ({components['seed']:.6f}), "
                f"yet every subject's argmax is the same operator ({distinct_winners}). "
                "A single global winner means the search cannot specialise per subject; "
                "NAS is still justified if families differ elsewhere, but not by this evidence."
            )
        else:
            verdict = "PROCEED"
            print(
                "  PROCEED -- V_subject x operator "
                f"({components['interaction']:.6f}) > V_seed ({components['seed']:.6f}) AND the best family "
                f"differs across subjects ({len(distinct_winners)} distinct winners: {distinct_winners}). "
                "A per-subject architecture search has a real, seed-surviving interaction to exploit."
            )
        unstable_subjects = [s for s, entry in per_subject.items() if entry["seed_unstable"]]
        if unstable_subjects:
            print(
                f"  caveat: subject(s) {unstable_subjects} flip their argmax across seeds -- check the seed "
                "spread before attributing those differences to the subject"
            )
        if not anova["balanced"]:
            print(
                "  caveat: cell seed counts differ, so the grid is unbalanced; the Type III SS and the EMS "
                "variance components are approximations -- fix the grid before acting on a borderline ratio"
            )
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
                    "runs": runs,
                    "cells": cells,
                    "missing_cells": missing,
                    "anova": anova,
                    "ratios": {key: (None if value is None or math.isinf(value) else value) for key, value in ratios.items()},
                    "ratios_infinite": {
                        key: bool(value is not None and math.isinf(value)) for key, value in ratios.items()
                    },
                    "per_subject": per_subject,
                    "fairness": fairness,
                    "session1_opened_runs": [row["leaf"] for row in opened],
                    "distinct_winners": distinct_winners,
                    "verdict": verdict,
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
