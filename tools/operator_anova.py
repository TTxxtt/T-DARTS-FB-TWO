"""Shared variance decomposition for the operator-separability tools.

Two generations of the pilot are analysed by two entry points
(``analyze_operator_v2.py`` and ``analyze_operator_v2e.py``).  They differ in
their metric scale, their operator set and their decision rule; they do not
differ in the arithmetic.  This module holds the arithmetic, so the two tools
cannot drift apart in the part where a drift would be hardest to notice -- a
number that is merely slightly wrong.

Extracted verbatim from ``analyze_operator_v2.py``.  The Matched tool imports
from here now and its output is unchanged; that was checked by diffing its full
output before and after the move, not by inspection.

Model
-----
With ``y[i, j, k]`` the metric for subject ``i``, operator ``j``, seed ``k``::

    y[i, j, k] = mu + a_i + b_j + (ab)_ij + e_ijk

The sums of squares are **Type III**, from an effect-coded least-squares fit
(+1 for a level, -1 for the reference level, and the product of the two codings
for the interaction): the SS of a term is the increase in residual sum of
squares when that term is dropped from the full model while every other term
stays in.  For a balanced grid this is identical to the textbook two-way ANOVA
closed forms; for an unbalanced grid it is the usual (approximate)
generalisation, which is why unbalanced cells are reported loudly -- they weaken
the ANOVA, they do not merely shrink n.

Variance components use the expected-mean-square (EMS) method, so the components
are on a common "variance of a single run" scale and can be compared against
each other::

    V_seed         = MS_residual
    V_interaction  = (MS_interaction - MS_residual) / n_rep
    V_operator     = (MS_operator    - MS_interaction) / (n_subjects * n_rep)
    V_subject      = (MS_subject     - MS_interaction) / (n_operators * n_rep)

where ``n_rep`` is the number of seeds per cell (harmonic mean of the per-cell
counts when the grid is unbalanced, which is the standard unbalanced EMS
substitute).  Negative estimates are clamped at 0 -- the clamp is reported
whenever it fires, because a clamped component means "no detectable effect",
not "a small negative one".
"""

from __future__ import annotations

import math
import sys
from typing import Any

import numpy as np

try:  # scipy is only used to decorate F statistics with a p-value
    from scipy import stats as _scipy_stats
except Exception:  # pragma: no cover - exercised on machines without scipy
    _scipy_stats = None


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


# ----------------------------------------------------------------------
# generation separation
# ----------------------------------------------------------------------
#: An operator whose name ends in this belongs to the Expressive generation.
EXPRESSIVE_SUFFIX = "_e"

MATCHED = "matched"
EXPRESSIVE = "expressive"


def generation_of(operator: str) -> str:
    """Which pilot generation an operator name belongs to.

    Derived from the name rather than from the output root, so a run that
    landed in the wrong tree is still classified correctly.  That is the
    failure this is here to catch: the Matched analyzer walks every
    ``final_summary.json`` under its root, so an Expressive run written there
    would otherwise be folded into the archived ANOVA as an unknown operator
    without anyone choosing it.
    """
    return EXPRESSIVE if operator.endswith(EXPRESSIVE_SUFFIX) else MATCHED


def generation_mix(runs: list[dict[str, Any]]) -> dict[str, int]:
    """Count runs per generation, for the pooling guard and the header."""

    counts: dict[str, int] = {}
    for row in runs:
        name = generation_of(row["operator"])
        counts[name] = counts.get(name, 0) + 1
    return counts


def refuse_mixed_generations(runs: list[dict[str, Any]], *, combine: bool) -> bool:
    """Return True if analysis may proceed; print and return False otherwise.

    The two generations are not on the same scale -- the Matched result is
    recorded on the raw ``val_best_nll`` scale and the Expressive one
    pre-registers ``log(val_best_nll)`` -- so pooling them by accident produces
    a number that means nothing and looks fine.  Refusing is the default;
    ``combine`` is an explicit opt-in that the caller must pair with an explicit
    metric.
    """

    mix = generation_mix(runs)
    if len(mix) <= 1:
        return True
    if combine:
        return True
    print("!" * 78)
    print("!! REFUSING to analyse runs from more than one generation at once.")
    print(f"!!   found: {mix}")
    print("!! The Matched generation is recorded on the raw val_best_nll scale; the")
    print("!! Expressive generation pre-registers log(val_best_nll).  A single")
    print("!! decomposition over both is not comparable to either recorded result.")
    print("!! Re-run one generation at a time, or pass --combine-generations and an")
    print("!! explicit --metric to accept the mixed scale on purpose.")
    for row in runs:
        print(f"!!   {generation_of(row['operator']):<11} {row['operator']:<20} {row['path']}")
    print("!" * 78)
    return False


def _fmt_ratio(value: float) -> str:
    return "inf (V_seed = 0)" if math.isinf(value) else f"{value:.3f}"


def print_anova(anova: dict[str, Any]) -> None:
    """Render the decomposition table.

    The residual row reads its component from the ``seed`` key: the EMS block
    names the run-to-run component ``seed`` because that is what it estimates,
    while SS/MS/df use ``residual``.  Mapping the two here means this is the one
    place that has to know they are the same thing.
    """

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
    for key, component_key, label in (
        ("subject", "subject", "subject"),
        ("operator", "operator", "operator"),
        ("interaction", "interaction", "subj x op"),
        ("residual", "seed", "seed (resid)"),
    ):
        line = (
            f"{label:<14}{anova['ss'][key]:>12.6f}{anova['df'][key]:>5}{anova['ms'][key]:>12.6f}"
            f"{anova['components'][component_key]:>14.6f}"
        )
        f_value = anova["f"].get(key)
        line += f"{'  --':>10}" if f_value is None else f"{f_value:>10.3f}"
        if anova["scipy"]:
            p_value = anova["p"].get(key)
            line += f"{'  --':>12}" if p_value is None else f"{p_value:>12.4g}"
        suffix = "   <- clamped at 0" if component_key in anova["clamped"] else ""
        print(line + suffix)
    print(f"{'total':<14}{anova['ss']['total']:>12.6f}{sum(anova['df'].values()):>5}")
    for key in anova["clamped"]:
        print(f"note: V_{key} estimate was negative and is clamped to 0 (effect not detectable at this seed budget)")
    if not anova["scipy"]:
        print("note: scipy not importable -- F p-values omitted, SS/MS/components unaffected")
