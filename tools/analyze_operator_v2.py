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

Decision rule
-------------
    V_subject x operator <= V_seed  -> STOP: the family preference does not
        exceed seed noise, so a per-subject architecture search has nothing to
        find.
    V_subject x operator >  V_seed  -> PROCEED, but only if the second
        condition also holds: the best family differs across subjects (>= 2
        distinct per-subject argmax winners).  A large interaction driven by
        two subjects swapping places is still an interaction, but a single
        winner means there is nothing to specialise.

``test`` is null in every pilot run on purpose (Session 1 is never opened);
the primary metric is therefore ``val_best_acc`` (``val_best_nll``, lower is
better, is also accepted).
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

try:  # scipy is only used to decorate F statistics with a p-value
    from scipy import stats as _scipy_stats
except Exception:  # pragma: no cover - exercised on machines without scipy
    _scipy_stats = None

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
#: MACs are *not* matched across operators; attention is expected to be far off.
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
        default="val_best_acc",
        choices=sorted(METRIC_DIRECTION),
        help="metric to decompose (default: val_best_acc)",
    )
    parser.add_argument("--json", type=Path, default=None, help="dump the full analysis here")
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


def _effect_coding(labels: list[int], n_levels: int) -> np.ndarray:
    """Effect coding: +1 on the level, -1 on the reference (last) level."""

    index = np.asarray(labels)
    columns = [(index == level).astype(float) - (index == n_levels - 1).astype(float) for level in range(n_levels - 1)]
    if not columns:
        return np.zeros((len(labels), 0))
    return np.column_stack(columns)


def _sse(design: np.ndarray, y: np.ndarray) -> float:
    """Residual sum of squares of ``y ~ design`` (intercept added by the caller)."""

    matrix = np.column_stack([np.ones(len(y)), design]) if design.shape[1] else np.ones((len(y), 1))
    beta, *_ = np.linalg.lstsq(matrix, y, rcond=None)
    residual = y - matrix @ beta
    return float(residual @ residual)


def two_way_anova(rows: list[dict[str, Any]], metric: str) -> dict[str, Any] | None:
    """Type III two-way ANOVA with interaction, plus EMS variance components.

    Returns ``None`` when the design has no replication (df_residual == 0), since
    then SS_residual is identically zero and the seed-variance scale -- the whole
    yardstick of the decision rule -- does not exist.
    """

    subjects = sorted({row["subject"] for row in rows})
    operators = sorted({row["operator"] for row in rows})
    n_subjects, n_operators = len(subjects), len(operators)
    if n_subjects < 2 or n_operators < 2:
        print(
            f"!! ANOVA needs >= 2 subjects and >= 2 operators, found "
            f"{n_subjects} subjects x {n_operators} operators",
            file=sys.stderr,
        )
        return None

    subject_of = {name: index for index, name in enumerate(subjects)}
    operator_of = {name: index for index, name in enumerate(operators)}
    subj_idx = [subject_of[row["subject"]] for row in rows]
    op_idx = [operator_of[row["operator"]] for row in rows]

    s_code = _effect_coding(subj_idx, n_subjects)
    o_code = _effect_coding(op_idx, n_operators)
    interaction = np.column_stack([s_code[:, a] * o_code[:, b] for a in range(s_code.shape[1]) for b in range(o_code.shape[1])])
    y = np.asarray([row[metric] for row in rows], dtype=float)

    sse_full = _sse(np.column_stack([s_code, o_code, interaction]), y)
    sse_no_interaction = _sse(np.column_stack([s_code, o_code]), y)
    sse_no_operator = _sse(np.column_stack([s_code, interaction]), y)
    sse_no_subject = _sse(np.column_stack([o_code, interaction]), y)
    sse_null = _sse(np.zeros((len(y), 0)), y)

    ss_subject = sse_no_subject - sse_full
    ss_operator = sse_no_operator - sse_full
    ss_interaction = sse_no_interaction - sse_full
    ss_residual = sse_full
    ss_total = sse_null

    df_subject = n_subjects - 1
    df_operator = n_operators - 1
    df_interaction = df_subject * df_operator
    df_residual = len(rows) - n_subjects * n_operators
    if df_residual <= 0:
        print(
            f"!! no residual degrees of freedom ({len(rows)} runs for "
            f"{n_subjects}x{n_operators} cells): every cell has exactly one seed, "
            "so seed variance is not estimable -- add seeds",
            file=sys.stderr,
        )
        return None

    ms = {
        "subject": ss_subject / df_subject,
        "operator": ss_operator / df_operator,
        "interaction": ss_interaction / df_interaction,
        "residual": ss_residual / df_residual,
    }

    # Harmonic mean of the per-cell counts: the standard unbalanced stand-in for
    # the balanced n_rep in the EMS formulas.  With a balanced grid it is exactly
    # the seed count per cell.
    counts = [sum(1 for row in rows if row["subject"] == s and row["operator"] == o) for s in subjects for o in operators]
    filled = [count for count in counts if count > 0]
    n_rep = len(filled) / sum(1.0 / count for count in filled)
    balanced = len(set(filled)) == 1

    raw = {
        "interaction": (ms["interaction"] - ms["residual"]) / n_rep,
        "operator": (ms["operator"] - ms["interaction"]) / (n_subjects * n_rep),
        "subject": (ms["subject"] - ms["interaction"]) / (n_operators * n_rep),
        "seed": ms["residual"],
    }
    components = {key: max(0.0, value) for key, value in raw.items()}
    clamped = [key for key, value in raw.items() if value < 0]

    f_stats = {
        "interaction": ms["interaction"] / ms["residual"],
        "operator": ms["operator"] / ms["interaction"] if ms["interaction"] > 0 else float("inf"),
        "subject": ms["subject"] / ms["interaction"] if ms["interaction"] > 0 else float("inf"),
    }
    p_values = {}
    if _scipy_stats is not None:
        p_values = {
            "interaction": float(_scipy_stats.f.sf(f_stats["interaction"], df_interaction, df_residual)),
            "operator": float(_scipy_stats.f.sf(f_stats["operator"], df_operator, df_interaction)),
            "subject": float(_scipy_stats.f.sf(f_stats["subject"], df_subject, df_interaction)),
        }

    return {
        "subjects": subjects,
        "operators": operators,
        "n_runs": len(rows),
        "n_rep": n_rep,
        "balanced": balanced,
        "cell_counts": {
            f"{subject}/{operator}": sum(1 for row in rows if row["subject"] == subject and row["operator"] == operator)
            for subject in subjects
            for operator in operators
        },
        "ss": {
            "subject": ss_subject,
            "operator": ss_operator,
            "interaction": ss_interaction,
            "residual": ss_residual,
            "total": ss_total,
        },
        "df": {
            "subject": df_subject,
            "operator": df_operator,
            "interaction": df_interaction,
            "residual": df_residual,
        },
        "ms": ms,
        "f": f_stats,
        "p": p_values,
        "components_raw": raw,
        "components": components,
        "clamped": clamped,
        "scipy": _scipy_stats is not None,
    }


def variance_ratio(component: float, seed: float) -> float:
    """Ratio to the seed component; ``inf`` when seed noise is exactly zero."""

    return float("inf") if seed <= 0 else component / seed


def _fmt_ratio(value: float) -> str:
    return "inf (V_seed = 0)" if math.isinf(value) else f"{value:.3f}"


def print_anova(anova: dict[str, Any]) -> None:
    print("\ntwo-way ANOVA with interaction (Type III SS, EMS variance components)")
    if anova["balanced"]:
        print(f"balanced grid, n_rep = {anova['n_rep']:.0f} seeds per cell")
    else:
        print(
            f"UNBALANCED grid -- EMS components use the harmonic-mean n_rep = {anova['n_rep']:.2f} "
            "and are approximate; fix the missing cells before trusting the ratios"
        )
    header = f"{'source':<14}{'SS':>12}{'df':>5}{'MS':>12}{'V(component)':>14}{'F':>10}"
    if anova["scipy"]:
        header += f"{'p':>12}"
    print(header)
    print("-" * len(header))
    for key, label in (
        ("subject", "subject"),
        ("operator", "operator"),
        ("interaction", "subj x op"),
        ("residual", "seed (resid)"),
    ):
        line = (
            f"{label:<14}{anova['ss'][key]:>12.6f}{anova['df'][key]:>5}{anova['ms'][key]:>12.6f}"
            f"{anova['components'][key]:>14.6f}"
        )
        f_value = anova["f"].get(key)
        line += f"{'  --':>10}" if f_value is None else f"{f_value:>10.3f}"
        if anova["scipy"]:
            p_value = anova["p"].get(key)
            line += f"{'  --':>12}" if p_value is None else f"{p_value:>12.4g}"
        suffix = "   <- clamped at 0" if key in anova["clamped"] else ""
        print(line + suffix)
    print(f"{'total':<14}{anova['ss']['total']:>12.6f}{sum(anova['df'].values()):>5}")
    for key in anova["clamped"]:
        print(f"note: V_{key} estimate was negative and is clamped to 0 (effect not detectable at this seed budget)")
    if not anova["scipy"]:
        print("note: scipy not importable -- F p-values omitted, SS/MS/components unaffected")


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
        # MACs are not matched by design: attention is expected around 2.3x.  This
        # is a surfaced, known deviation, not an error.
        mac_flag = ""
        if mac_ratio is not None:
            mac_flag = "" if abs(mac_ratio - 1.0) <= 0.5 else f"dev {mac_ratio:.2f}x"
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
        print("macFlag is informational only: MACs are expected to diverge (attention ~2.3x), not an error")
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
            verdict = "STOP"
            print(
                "  STOP -- do not proceed to NAS. V_subject x operator "
                f"({components['interaction']:.6f}) does not exceed V_seed ({components['seed']:.6f}): "
                "the operator family preference is within seed noise, so a per-subject architecture "
                "search has nothing subject-specific to find. More seeds will not rescue a zero interaction; "
                "a stronger interaction would need a stronger operator family or a harder dataset."
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
