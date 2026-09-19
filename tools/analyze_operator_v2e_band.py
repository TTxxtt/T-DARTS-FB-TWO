#!/usr/bin/env python
"""Paired analysis of the band-specific mechanism probe.

The question is not "how good is this configuration".  A band probe's absolute
NLL carries the subject's difficulty, that band's difficulty and the seed's
draw all at once, and comparing absolute levels across configurations is how
the two global grids ended up reading a global ordering that never reordered.
So the primary statistic is a **paired difference against the anchor**:

    Delta(s, b, f) = log(NLL_probe(s, b, f)) - log(NLL_anchor(s))

for the same subject and seed.  Only one band changes, and the anchor is
already subtracted, so subject difficulty, band difficulty and the seed's
drawing of initial weights cancel.  ``Delta < 0`` means the replacement helped.

The anchor is not retrained.  Its nine runs (3 subjects x 3 seeds) already
exist as the Expressive ``dilated_e`` arm, and this tool reads them from the
frozen tree at ``run/outputs/operator_v2e``.  Retraining them would put a
second, differently-seeded copy of the same configuration into the comparison
and the pairing would stop being paired.

``log(val_best_nll)`` is the pre-registered primary scale, as in the
Expressive grid.  Raw NLL and accuracy are reported beside it and never
replace it.

What is reported
----------------
The Expressive verdict rule was a ratio against seed variance, and its own
history is why that is not repeated here: the same data reads 0.833 raw and
1.187 in log, so a single ``ratio > 1`` switch is decided by the scale rather
than by the mechanism.  This tool reports four things side by side instead --
effect magnitude, winner diversity, winner stability across seeds, and the
direction consistency of the paired differences -- and :func:`assess_band_probe`
is a pure function of them.

Early-overfit diagnostics are printed and are **not** part of the verdict.  The
``< 50`` epoch threshold was read off the previous 90 runs as a gap in a
bimodal distribution; that makes it usable as a description and unusable as a
selection rule, since anything chosen after seeing the outcome cannot then be
used to filter the outcome.  Every run in the grid enters the analysis.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tdarts import config as C
from tdarts.operator_v2e_band import (
    ANCHOR_FAMILY,
    STAGE_PREFIX,
    VARYING_FAMILIES,
    parse_slug,
)
from tools.operator_anova import two_way_anova

#: Families a cell's winner is chosen from.  The anchor is in the list with a
#: Delta of exactly zero by construction: "no replacement" is a real option, and
#: a stage whose winners are all the anchor has answered its own question.
CANDIDATES: tuple[str, ...] = (ANCHOR_FAMILY,) + VARYING_FAMILIES

#: Diagnostic only -- see the module docstring.  Never a selection criterion and
#: never used to drop a run.
EARLY_EPOCH = 50

#: How many subjects have to carry a stable non-anchor preference before the
#: result is read as a property of the cohort rather than of one subject.  This
#: selects the *label*; it is not a pass condition, so a single-subject signal
#: is still reported as a signal.  See :func:`assess_band_probe`.
MULTI_SUBJECT_EVIDENCE = 2

#: The statement this analysis exists to be able to make.  Printed with every
#: verdict so a reader cannot quote a number without the design it came from.
SESSION1_NOTICE = (
    "本阶段属于 Session-0 architecture-space diagnosis / screening；"
    "Session 1 始终关闭。Stage2 cross-session evaluation 只会在最终搜索方法"
    "和架构选择规则完全冻结后执行。"
)

WEAK_NOTICE = (
    "记录为 temporal mechanism personalization evidence weak。"
    "不要据此强行做 temporal-family NAS；下一阶段再考虑 spatial / electrode / graph 方向。"
)

SINGLE_SUBJECT_NOTICE = (
    "有个体化信号，但在 3-subject pilot 中只出现在一个被试身上："
    "证据不足以直接进入完整 NAS。下一步应当先扩展 subject 验证，而不是据此否定该方向。"
)

#: What each verdict means, in the words the branch above is meant to be read in.
VERDICT_NOTICES = {
    "WEAK_PERSONALIZATION": WEAK_NOTICE,
    "SINGLE_SUBJECT_PERSONALIZATION_SIGNAL": SINGLE_SUBJECT_NOTICE,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--probe-root", type=Path, default=Path(f"run/outputs/{STAGE_PREFIX}"))
    parser.add_argument(
        "--anchor-root",
        type=Path,
        default=Path("run/outputs/operator_v2e"),
        help="frozen Expressive tree the all-dilated anchor is reused from",
    )
    parser.add_argument("--dataset", default="bci42a")
    parser.add_argument("--anchor-operator", default=ANCHOR_FAMILY)
    parser.add_argument("--json", type=Path, default=Path("run/outputs/operator_v2e_band_analysis.json"))
    return parser.parse_args()


# ----------------------------------------------------------------------
# discovery
# ----------------------------------------------------------------------
def _read_summary(leaf: Path) -> dict[str, Any] | None:
    path = leaf / "final_summary.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def discover_anchor(root: Path, dataset: str, operator: str) -> dict[tuple[str, int], dict[str, Any]]:
    """The reused reference runs, keyed by ``(subject, seed)``."""

    base = Path(root) / dataset
    found: dict[tuple[str, int], dict[str, Any]] = {}
    if not base.is_dir():
        return found
    for leaf in sorted(path for path in base.iterdir() if path.is_dir()):
        summary = _read_summary(leaf)
        if summary is None or summary.get("operator") != operator:
            continue
        key = (str(summary["subject"]), int(summary["seed"]))
        if key in found:
            print(f"!! duplicate anchor run for {key}: {leaf}", file=sys.stderr)
        found[key] = {"path": leaf, "summary": summary}
    return found


def discover_probes(root: Path, dataset: str) -> tuple[list[dict[str, Any]], list[str]]:
    """Every run under the probe tree, with its band/family recovered.

    A directory whose summary is unreadable or whose slug is not a probe slug is
    skipped loudly.  Silently ignoring it would shrink the grid without anyone
    choosing to, and a shortened grid is indistinguishable from a finished one
    in the output.
    """

    base = Path(root) / dataset
    runs: list[dict[str, Any]] = []
    skipped: list[str] = []
    if not base.is_dir():
        return runs, [f"{base} does not exist"]
    for leaf in sorted(path for path in base.iterdir() if path.is_dir()):
        summary = _read_summary(leaf)
        if summary is None:
            skipped.append(f"{leaf.name}: no readable final_summary.json")
            continue
        band = summary.get("varying_band")
        family = summary.get("varying_family")
        if band is None or family is None:
            try:
                families = parse_slug(str(summary.get("operator", "")))
            except ValueError as exc:
                skipped.append(f"{leaf.name}: {exc}")
                continue
            varying = [(b, f) for b, f in families.items() if f != ANCHOR_FAMILY]
            band, family = varying[0] if varying else (None, ANCHOR_FAMILY)
        runs.append(
            {
                "path": leaf,
                "subject": str(summary["subject"]),
                "seed": int(summary["seed"]),
                "band": band,
                "family": family,
                "slug": str(summary.get("operator", leaf.name)),
                "val_best_nll": float(summary["val_best_nll"]),
                "val_best_acc": float(summary.get("val_best_acc", float("nan"))),
                "summary": summary,
            }
        )
    return runs, skipped


def run_diagnostics(leaf: Path) -> dict[str, Any] | None:
    """Descriptive training facts.  Diagnostic only -- see the module docstring."""

    path = leaf / "metrics.jsonl"
    if not path.is_file():
        return None
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        return None
    best = min(rows, key=lambda row: row["val_nll"])
    final = rows[-1]
    return {
        "best_epoch": int(best["epoch"]),
        "stop_epoch": int(final["epoch"]),
        "train_nll_at_best": float(best.get("train_nll_eval", float("nan"))),
        "final_train_acc": float(final["train_acc"]),
        "val_best_acc": float(best["val_acc"]),
        "early_overfit_flag": bool(best["epoch"] < EARLY_EPOCH),
    }


# ----------------------------------------------------------------------
# pairing
# ----------------------------------------------------------------------
def build_pairs(
    anchors: dict[tuple[str, int], dict[str, Any]], probes: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[str]]:
    """One paired row per probe run, or a note saying why it could not be paired."""

    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for probe in probes:
        if probe["band"] is None:
            missing.append(f"{probe['path'].name}: all-dilated, already the anchor")
            continue
        key = (probe["subject"], probe["seed"])
        anchor = anchors.get(key)
        if anchor is None:
            missing.append(f"{probe['path'].name}: no anchor run for subject/seed {key}")
            continue
        anchor_nll = float(anchor["summary"]["val_best_nll"])
        probe_nll = probe["val_best_nll"]
        if anchor_nll <= 0 or probe_nll <= 0:
            missing.append(f"{probe['path'].name}: non-positive NLL, log undefined")
            continue
        rows.append(
            {
                "subject": probe["subject"],
                "seed": probe["seed"],
                "band": probe["band"],
                "family": probe["family"],
                "slug": probe["slug"],
                # The pre-registered primary scale.
                "delta_log_nll": math.log(probe_nll) - math.log(anchor_nll),
                "delta_raw_nll": probe_nll - anchor_nll,
                "probe_nll": probe_nll,
                "anchor_nll": anchor_nll,
                "probe_acc": probe["val_best_acc"],
                "anchor_acc": float(anchor["summary"].get("val_best_acc", float("nan"))),
            }
        )
    return rows, missing


def as_deltas(rows: Iterable[dict[str, Any]]) -> dict[tuple[str, str, str], dict[int, float]]:
    out: dict[tuple[str, str, str], dict[int, float]] = {}
    for row in rows:
        out.setdefault((row["subject"], row["band"], row["family"]), {})[row["seed"]] = row["delta_log_nll"]
    return out


# ----------------------------------------------------------------------
# reporting
# ----------------------------------------------------------------------
def _mean(values: list[float]) -> float:
    return statistics.mean(values) if values else float("nan")


def _sd(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def cell_table(
    deltas: dict[tuple[str, str, str], dict[int, float]],
    subjects: list[str],
    bands: tuple[str, ...],
) -> dict[str, Any]:
    """Per subject x band x family: the paired difference, summarised."""

    table: dict[str, Any] = {}
    for subject in subjects:
        for band in bands:
            for family in VARYING_FAMILIES:
                per_seed = deltas.get((subject, band, family), {})
                values = [per_seed[seed] for seed in sorted(per_seed)]
                improved = sum(1 for value in values if value < 0)
                table[f"{subject}/{band}/{family}"] = {
                    "n_seeds": len(values),
                    "mean_delta_log_nll": _mean(values),
                    "sd_delta_log_nll": _sd(values),
                    "median_delta_log_nll": statistics.median(values) if values else float("nan"),
                    "per_seed": {str(seed): per_seed[seed] for seed in sorted(per_seed)},
                    "improved_seeds": improved,
                    # "3/3", "2/3", "1/3" -- how often the replacement helped.
                    "direction_consistency": f"{improved}/{len(values)}" if values else "0/0",
                }
    return table


def _pick(candidates: Iterable[str], value_of) -> str | None:
    """argmin over candidates, first-wins on ties.

    The candidate order puts the anchor first, so an exact tie is read as "no
    replacement", which is the conservative direction: a replacement has to be
    strictly better to be called a winner.
    """

    best: str | None = None
    best_value = None
    for name in candidates:
        value = value_of(name)
        if value is None:
            return None
        if best_value is None or value < best_value:
            best, best_value = name, value
    return best


def winner_table(
    deltas: dict[tuple[str, str, str], dict[int, float]],
    subjects: list[str],
    bands: tuple[str, ...],
    seeds: list[int],
) -> dict[str, Any]:
    """Which family each subject x band cell prefers, and whether it holds up."""

    table: dict[str, Any] = {}
    stable = 0
    total = 0
    for subject in subjects:
        for band in bands:
            def mean_of(family: str):
                if family == ANCHOR_FAMILY:
                    # By construction: the anchor paired against itself.
                    return 0.0
                per_seed = deltas.get((subject, band, family), {})
                if not per_seed:
                    return None
                return _mean([per_seed[seed] for seed in sorted(per_seed)])

            winner = _pick(CANDIDATES, mean_of)
            per_seed_winners = {}
            for seed in seeds:
                def seed_value(family: str):
                    if family == ANCHOR_FAMILY:
                        return 0.0
                    per_seed = deltas.get((subject, band, family), {})
                    return per_seed.get(seed)

                picked = _pick(CANDIDATES, seed_value)
                if picked is not None:
                    per_seed_winners[str(seed)] = picked
            agree = sum(1 for picked in per_seed_winners.values() if picked == winner)
            agrees = len(per_seed_winners) == len(seeds) and agree == len(per_seed_winners)
            if agrees:
                stable += 1
            total += 1
            table[f"{subject}/{band}"] = {
                "winner": winner,
                "per_seed_winners": per_seed_winners,
                "seeds_agreeing": agree,
                "n_seeds": len(per_seed_winners),
                "stable_across_seeds": bool(agrees),
                "mean_delta_by_family": {
                    family: mean_of(family) for family in CANDIDATES
                },
            }
    #: Cells a replacement won.  Computed here rather than in the verdict so the
    #: printed report and the decision read the same list.
    contested = sorted(key for key, row in table.items() if row["winner"] not in (None, ANCHOR_FAMILY))
    contested_stable = [key for key in contested if table[key]["stable_across_seeds"]]
    return {
        "cells": table,
        "stable_cells": stable,
        "total_cells": total,
        "stable_fraction": (stable / total) if total else 0.0,
        "distinct_winners": sorted({row["winner"] for row in table.values() if row["winner"]}),
        "contested_cells": contested,
        "contested_cells_stable_across_seeds": contested_stable,
    }


def band_effect_table(
    deltas: dict[tuple[str, str, str], dict[int, float]],
    subjects: list[str],
    bands: tuple[str, ...],
    seeds: list[int],
) -> dict[str, Any]:
    """Is the *same* family better in one band than in another?

    The spread is compared against the seed noise measured inside those same
    bands.  Without that denominator the table would call the difference between
    two seeds a band effect.
    """

    table: dict[str, Any] = {}
    for subject in subjects:
        for family in VARYING_FAMILIES:
            means = {}
            values = {}
            for band in bands:
                per_seed = deltas.get((subject, band, family), {})
                values[band] = [per_seed[seed] for seed in sorted(per_seed)]
                means[band] = _mean(values[band])
            if any(not values[band] for band in bands):
                continue
            pooled = [value for band in bands for value in values[band]]
            spread = max(means.values()) - min(means.values())
            noise = _sd(pooled)
            best_band = min(means, key=lambda band: means[band])
            per_seed_best = {}
            for seed in seeds:
                if any(seed not in deltas.get((subject, band, family), {}) for band in bands):
                    continue
                per_seed_best[str(seed)] = min(
                    bands, key=lambda band: deltas[(subject, band, family)][seed]
                )
            agreeing = sum(1 for band in per_seed_best.values() if band == best_band)
            meaningful = bool(spread > noise)
            # A spread inside the noise still has an argmin, and reporting it
            # would read as a best band.  It is reported as unknown instead.
            table[f"{subject}/{family}"] = {
                "band_means": means,
                "band_spread": spread,
                "seed_sd_within_bands": noise,
                "exceeds_seed_noise": meaningful,
                "best_band": best_band if meaningful else None,
                "per_seed_best_band": per_seed_best,
                "best_band_seeds_agreeing": agreeing,
                "best_band_stable": bool(
                    meaningful
                    and len(per_seed_best) == len(seeds)
                    and agreeing == len(per_seed_best)
                ),
            }
    return table


def decompose(rows: list[dict[str, Any]], families: tuple[str, ...]) -> dict[str, Any]:
    """Subject / band / subject x band variance components, per family.

    Run on the paired difference, so the subject and seed main effects that the
    pairing already cancelled are not re-counted as structure.  ``operator`` is
    the band axis here; the shared ANOVA is reused rather than re-derived so the
    degrees of freedom and the EMS formulas are the ones already tested.
    """

    out: dict[str, Any] = {}
    for family in families:
        subset = [
            {
                "subject": row["subject"],
                # The shared ANOVA names its second factor "operator"; here it
                # is the band, which is the whole point of the stage.
                "operator": row["band"],
                "delta_log_nll": row["delta_log_nll"],
            }
            for row in rows
            if row["family"] == family
        ]
        if len(subset) < 4:
            out[family] = None
            continue
        existing = {row["subject"] for row in subset}
        existing_bands = {row["operator"] for row in subset}
        if len(existing) < 2 or len(existing_bands) < 2:
            out[family] = None
            continue
        # two_way_anova prints to stderr on an unestimable design; that message
        # is part of the report, so it is left where it is.
        out[family] = two_way_anova(subset, "delta_log_nll")
    return out


# ----------------------------------------------------------------------
# verdict
# ----------------------------------------------------------------------
def assess_band_probe(
    *,
    winners: dict[str, Any],
    effects: dict[str, Any],
    cell_table_: dict[str, Any],
    subjects: list[str],
    bands: tuple[str, ...],
    seeds: list[int],
) -> dict[str, Any]:
    """Map the four reported quantities onto one of four readings.

    Pure: it reads only the tables it is handed, so the conditions can be tested
    on synthetic grids without a run on disk.  Each condition reports its own
    numbers and whether it passed, because "the probe did not proceed" is only
    actionable if the reader can see which of the three failed.
    """

    total_cells = len(subjects) * len(bands)
    complete = (
        winners["total_cells"] == total_cells
        and all(row["n_seeds"] == len(seeds) for row in winners["cells"].values())
        and all(
            row["n_seeds"] == len(seeds)
            for key, row in cell_table_.items()
            if key.split("/")[1] in bands
        )
    )
    if not complete:
        return {
            "verdict": "NOT_COMPUTABLE",
            "reason": "incomplete grid: not every subject x band cell has one run per seed",
            "conditions": {},
            "session1_closed": True,
        }

    # C1  Is there a band effect at all?  Same family, different band, and the
    #     spread has to beat the seed noise measured inside those bands.  One
    #     subject is enough to answer that question; *how many* subjects carry
    #     the effect is a separate question, and it decides the label rather
    #     than this condition.  Counting it here as well would count it twice.
    subjects_with_band_effect = sorted(
        {
            key.split("/")[0]
            for key, row in effects.items()
            if row["exceeds_seed_noise"] and row["best_band_stable"]
        }
    )
    band_effect = bool(subjects_with_band_effect)

    # C2  Do subjects disagree about which family their bands want?  Counting
    #     distinct winners is not enough: "Low wants dynamic_e for everybody"
    #     already produces two distinct winners (dynamic_e and the anchor) while
    #     being a band effect with no subject in it.  What has to differ is a
    #     *signature* -- the band-to-family pattern of one subject against
    #     another -- and that difference has to be real, so at least two
    #     subjects must hold a replacement that survives the seed split.
    #
    #     Either side of a disagreement may be the anchor: "Low wants dynamic_e
    #     for 003 and gated_e for 005" is subject-specific band preference with
    #     no anchor involved, and requiring one side to be the anchor would
    #     discard exactly the strongest form of the result.  What still does not
    #     count is every subject landing on the same family in the same band --
    #     then no pair disagrees, and C2 is false however large the band effect.
    diversity = len(winners["distinct_winners"])
    signature = {
        subject: tuple(
            (winners["cells"].get(f"{subject}/{band}") or {}).get("winner") for band in bands
        )
        for subject in subjects
    }
    disagreeing_pairs = [
        (first, second)
        for index, first in enumerate(subjects)
        for second in subjects[index + 1:]
        if any(
            signature[first][position] != signature[second][position]
            for position in range(len(bands))
        )
    ]
    #: Bands on which some pair of subjects does not share a winner, and which
    #: winners are involved -- reported so a reader can see *what* the
    #: disagreement is rather than only that one exists.
    disagreeing_bands = sorted(
        {
            bands[position]
            for first, second in disagreeing_pairs
            for position in range(len(bands))
            if signature[first][position] != signature[second][position]
        }
    )
    # How many subjects the signal actually spans.  Reported by C2 and used by
    # the label mapping, but not a pass condition: a signal carried by one
    # subject is a different claim from one carried by the whole pilot, not the
    # difference between a result and a non-result.  See the branch below.
    subjects_with_replacement = sorted(
        {
            key.split("/")[0]
            for key, row in winners["cells"].items()
            if row["winner"] not in (None, ANCHOR_FAMILY) and row["stable_across_seeds"]
        }
    )
    personalized = bool(disagreeing_pairs)

    # C3  Is the winner a property of the cell or of the seed?  Two figures are
    #     reported and only one decides.
    #
    #     *contested* cells -- ones a replacement won -- are the deciding figure:
    #     a cell the anchor wins because nothing in it varied is trivially
    #     seed-stable, and averaging those in would let a grid of non-results
    #     report perfect stability.
    #
    #     The overall figure keeps them in, so "this band stably needs no
    #     replacement" stays visible as a fact about the grid rather than
    #     disappearing from the report.
    contested = winners["contested_cells"]
    contested_stable = winners["contested_cells_stable_across_seeds"]
    overall_stable_fraction = winners["stable_fraction"]
    stable_enough = bool(contested) and len(contested_stable) == len(contested)

    conditions = {
        "C1_band_effect": {
            "passed": bool(band_effect),
            "subjects_with_a_stable_band_effect": subjects_with_band_effect,
            "needs_at_least": 1,
        },
        "C2_subject_specific": {
            "passed": bool(personalized),
            "distinct_winners": winners["distinct_winners"],
            "signatures": {subject: list(signature[subject]) for subject in subjects},
            # Signatures that differ nowhere are one signature; that is the
            # "every subject wants the same family in the same band" case.
            "distinct_signatures": len({signature[subject] for subject in subjects}),
            "disagreeing_subject_pairs": [list(pair) for pair in disagreeing_pairs],
            "bands_where_subjects_disagree": disagreeing_bands,
            # Not a pass condition -- see the branch below.  A signal carried by
            # one subject is a real signal with thin evidence, so it gets its own
            # label instead of being rounded down to "no personalization".
            "subjects_with_a_stable_replacement": subjects_with_replacement,
            "n_subjects_with_a_stable_replacement": len(subjects_with_replacement),
        },
        "C3_stable_across_seeds": {
            "passed": bool(stable_enough),
            # The deciding figure: only the cells a replacement won.
            "contested_cell_stability": f"{len(contested_stable)}/{len(contested)}",
            "contested_cells": contested,
            "contested_cells_stable_across_seeds": contested_stable,
            # Descriptive companion, printed so "this band stably needs no
            # replacement" is still visible.  A grid whose cells are all
            # uncontested reads 100% here while deciding nothing, which is why
            # it is not the figure the condition uses.
            "overall_winner_stability": f"{winners['stable_cells']}/{winners['total_cells']}",
            "overall_stable_fraction": overall_stable_fraction,
        },
    }
    # The three conditions each answer one question -- is there band structure,
    # do subjects disagree about it, does it survive the seeds -- and the branch
    # below adds the fourth, which is not a condition: *how much of the pilot
    # carries the signal*.
    #
    # The three-subject design is small enough that a signal carried by one
    # subject is a real finding with thin evidence rather than a non-finding.
    # Rounding it down to "no personalization" would throw away the 003 != 005 =
    # 006 case, which is exactly the kind of result the stage is looking for; it
    # gets its own label so the next step is "run more subjects", not "the idea
    # failed".
    #
    # A cell whose winner is not the anchor but whose paired differences do not
    # even agree in sign across seeds is a coin flip with a label on it, so C3
    # still has to pass before either personalized label applies.
    if band_effect and personalized and stable_enough:
        verdict = (
            "PERSONALIZED_BAND_MECHANISM"
            if len(subjects_with_replacement) >= MULTI_SUBJECT_EVIDENCE
            else "SINGLE_SUBJECT_PERSONALIZATION_SIGNAL"
        )
    elif band_effect:
        verdict = "BAND_EFFECT_WITHOUT_STABLE_PERSONALIZATION"
    elif personalized and stable_enough:
        verdict = "SUBJECT_PREFERENCE_WITHOUT_BAND_STRUCTURE"
    else:
        verdict = "WEAK_PERSONALIZATION"
    return {
        "verdict": verdict,
        "conditions": conditions,
        "session1_closed": True,
        "notice": SESSION1_NOTICE,
        "subjects_with_a_stable_replacement": subjects_with_replacement,
        "multi_subject_evidence_threshold": MULTI_SUBJECT_EVIDENCE,
        "verdict_notice": VERDICT_NOTICES.get(verdict),
    }


# ----------------------------------------------------------------------
# printing
# ----------------------------------------------------------------------
def print_grid(rows: list[dict[str, Any]], skipped: list[str], missing: list[str]) -> None:
    print(f"paired runs: {len(rows)}")
    if skipped:
        print(f"\n!! {len(skipped)} directory/directories could not be read as probe runs:")
        for note in skipped:
            print(f"   {note}")
    if missing:
        print(f"\n!! {len(missing)} run(s) could not be paired against an anchor:")
        for note in missing:
            print(f"   {note}")


def print_cells(table: dict[str, Any], subjects: list[str], bands: tuple[str, ...]) -> None:
    print("\n=== 配对差（primary：log(val_best_nll) 之差，相对同 subject/seed 的 all-dilated 锚）===")
    print("Δ < 0 = 替换该 band 后优于全膨胀卷积；Δ > 0 = 变差")
    print(f"{'受试者':<8}{'band':<7}{'family':<15}{'平均Δ':>10}{'sd':>9}{'中位Δ':>10}{'方向一致':>10}   三个 seed 的原始 Δ")
    for subject in subjects:
        for band in bands:
            for family in VARYING_FAMILIES:
                row = table.get(f"{subject}/{band}/{family}")
                if row is None:
                    continue
                if not row["n_seeds"]:
                    # An empty cell is stated rather than printed as a NaN row:
                    # a grid that is missing runs and a grid whose runs all read
                    # NaN are different problems with the same symptom.
                    print(f"{subject:<8}{band:<7}{family:<15}{'缺':>10}{'':>9}{'':>10}{'0/0':>10}   （该 cell 没有 run）")
                    continue
                raw = "  ".join(f"{row['per_seed'][key]:+.4f}" for key in sorted(row["per_seed"]))
                print(
                    f"{subject:<8}{band:<7}{family:<15}{row['mean_delta_log_nll']:>+10.4f}"
                    f"{row['sd_delta_log_nll']:>9.4f}{row['median_delta_log_nll']:>+10.4f}"
                    f"{row['direction_consistency']:>10}   {raw}"
                )


def print_winners(winners: dict[str, Any], subjects: list[str], bands: tuple[str, ...]) -> None:
    print("\n=== 每个 subject × band 的 winner（候选含锚本身，Δ≡0）===")
    print(f"{'受试者':<8}{'band':<7}{'winner':<15}{'跨seed一致':>12}   逐 seed winner")
    for subject in subjects:
        for band in bands:
            row = winners["cells"].get(f"{subject}/{band}")
            if row is None:
                continue
            per_seed = ", ".join(f"{key}:{value}" for key, value in sorted(row["per_seed_winners"].items()))
            print(
                f"{subject:<8}{band:<7}{str(row['winner']):<15}"
                f"{row['seeds_agreeing']}/{row['n_seeds']:<8}   {per_seed}"
            )
    contested = winners["contested_cells"]
    contested_stable = winners["contested_cells_stable_across_seeds"]
    print(f"\nwinner 多样性（cell 层面）：{winners['distinct_winners']}")
    print("两个稳定性指标同时输出，判定只看第一个：")
    print(f"  非锚 contested-cell 稳定性（判定用）：{len(contested_stable)}/{len(contested)}")
    for key in contested:
        mark = "稳定" if key in contested_stable else "跨 seed 翻转"
        print(f"      {key:<16}{mark}")
    print(f"  overall winner 稳定性（描述用）：{winners['stable_cells']}/{winners['total_cells']} "
          f"= {winners['stable_fraction']:.2f}")
    print("  注：网格里没有 contested cell 时 overall 会读成 100%，却什么都没判；")
    print("      保留 overall 是为了让「某个 band 稳定不需要替换」这个信息不消失。")


def print_band_effects(effects: dict[str, Any], subjects: list[str]) -> None:
    print("\n=== band effect（同一个 family 放在 Low/Mid/High 是否不同）===")
    print("spread 必须超过 band 内部的 seed 噪声，才算 band 效应而不是抽签")
    print("spread 未超过噪声时，最好 band 记为 —（它只是噪声里的最小值）")
    print(f"{'受试者':<8}{'family':<15}{'Low':>10}{'Mid':>10}{'High':>10}{'spread':>10}{'seed噪声':>11}{'最好band':>12}{'跨seed稳定':>12}")
    for subject in subjects:
        for family in VARYING_FAMILIES:
            row = effects.get(f"{subject}/{family}")
            if row is None:
                continue
            means = row["band_means"]
            print(
                f"{subject:<8}{family:<15}{means['Low']:>+10.4f}{means['Mid']:>+10.4f}{means['High']:>+10.4f}"
                f"{row['band_spread']:>10.4f}{row['seed_sd_within_bands']:>11.4f}"
                f"{row['best_band'] or '—':>12}{str(row['best_band_stable']):>12}"
            )


def print_decomposition(decomposition: dict[str, Any]) -> None:
    print("\n=== 方差分解（对配对差做，subject / band / subject×band / seed）===")
    for family, result in decomposition.items():
        if result is None:
            print(f"  {family:<15} 设计不完整，拒绝给分量")
            continue
        components = result["components"]
        residual = components["seed"]
        def ratio(value: float) -> str:
            return "inf" if residual <= 0 else f"{value / residual:.3f}"
        print(
            f"  {family:<15} V_subject/V_seed {ratio(components['subject']):>8}   "
            f"V_band/V_seed {ratio(components['operator']):>8}   "
            f"V_subject×band/V_seed {ratio(components['interaction']):>8}"
        )
        if result["clamped"]:
            print(f"                   （负分量已截断到 0：{sorted(result['clamped'])}）")
        p = result["p"]
        if p:
            print(f"                   p(interaction) = {p['interaction']:.4f}")


def print_diagnostics(probes: list[dict[str, Any]]) -> None:
    print("\n=== 训练诊断（仅诊断，不参与判定，不得用于删 run）===")
    print(f"early_overfit 阈值 = 第 {EARLY_EPOCH} 轮，来自前 90 个 run 的双峰分布间隙；")
    print("它是事后读数，因此只能描述，不能用来筛选。全部 81 个 run 都必须进入正式分析。")
    print(f"{'受试者':<8}{'seed':>10}{'配置':<20}{'bestEpoch':>10}{'stopEpoch':>10}{'最优训练NLL':>13}{'末训练acc':>11}{'早过拟合':>10}")
    for probe in sorted(probes, key=lambda row: (row["subject"], row["seed"], row["slug"])):
        diag = run_diagnostics(probe["path"])
        if diag is None:
            continue
        print(
            f"{probe['subject']:<8}{probe['seed']:>10}{probe['slug']:<20}{diag['best_epoch']:>10}"
            f"{diag['stop_epoch']:>10}{diag['train_nll_at_best']:>13.4f}{diag['final_train_acc']:>11.4f}"
            f"{str(diag['early_overfit_flag']):>10}"
        )


def print_anchor_table(anchors: dict[tuple[str, int], dict[str, Any]], operator: str) -> None:
    print(f"\n=== 复用的 all-dilated 锚（{operator}，来自冻结的 Expressive 树）===")
    print(f"{'受试者':<8}{'seed':>10}{'val_best_nll':>16}{'val_best_acc':>14}{'session1_opened':>18}{'test':>8}")
    for (subject, seed), entry in sorted(anchors.items()):
        summary = entry["summary"]
        print(
            f"{subject:<8}{seed:>10}{float(summary['val_best_nll']):>16.10f}"
            f"{float(summary.get('val_best_acc', float('nan'))):>14.6f}"
            f"{str(summary.get('session1_opened')):>18}{str(summary.get('test')):>8}"
        )


def print_verdict(result: dict[str, Any]) -> None:
    print("\n=== 判定 ===")
    print("C1/C2/C3 各答一个问题；第四个问题「信号覆盖几个被试」决定标签，不是条件：")
    print("  C1∧C2∧C3 且 ≥2 个 subject 有稳定非锚偏好 → PERSONALIZED_BAND_MECHANISM")
    print("  C1∧C2∧C3 但只有 1 个            → SINGLE_SUBJECT_PERSONALIZATION_SIGNAL")
    print("  C1∧(¬C2∨¬C3)                    → BAND_EFFECT_WITHOUT_STABLE_PERSONALIZATION")
    print("  ¬C1∧C2∧C3                       → SUBJECT_PREFERENCE_WITHOUT_BAND_STRUCTURE")
    print("  其余                            → WEAK_PERSONALIZATION")
    print(f"\nverdict: {result['verdict']}")
    for name, condition in result["conditions"].items():
        print(f"  {name}: {'PASS' if condition['passed'] else 'FAIL'}")
        for key, value in condition.items():
            if key != "passed":
                print(f"      {key}: {value}")
    print(f"  subjects_with_a_stable_replacement: {result.get('subjects_with_a_stable_replacement')}")
    print(f"  多被试证据阈值: >= {result.get('multi_subject_evidence_threshold')}")
    if result.get("reason"):
        print(f"  reason: {result['reason']}")
    print(f"\n{SESSION1_NOTICE}")
    if result.get("verdict_notice"):
        print(result["verdict_notice"])


# ----------------------------------------------------------------------
def main() -> int:
    args = parse_args()
    if Path(args.probe_root).resolve() == Path(args.anchor_root).resolve():
        print(
            f"!! --probe-root and --anchor-root are the same tree ({args.probe_root}); "
            f"the pairing would compare the probe against itself",
            file=sys.stderr,
        )
        return 2

    anchors = discover_anchor(args.anchor_root, args.dataset, args.anchor_operator)
    probes, skipped = discover_probes(args.probe_root, args.dataset)
    print(f"anchor runs ({args.anchor_operator}): {len(anchors)}")
    print(f"probe runs: {len(probes)}")
    print(f"\n{SESSION1_NOTICE}\n")
    if not probes:
        print("no probe runs found", file=sys.stderr)
        return 1
    print_anchor_table(anchors, args.anchor_operator)
    print_diagnostics(probes)

    rows, missing = build_pairs(anchors, probes)
    print_grid(rows, skipped, missing)
    if not rows:
        print("no pairings could be formed", file=sys.stderr)
        return 1

    subjects = sorted({row["subject"] for row in rows})
    bands = tuple(C.BANDS)
    seeds = sorted({row["seed"] for row in rows})
    deltas = as_deltas(rows)

    cells = cell_table(deltas, subjects, bands)
    winners = winner_table(deltas, subjects, bands, seeds)
    effects = band_effect_table(deltas, subjects, bands, seeds)
    decomposition = decompose(rows, VARYING_FAMILIES)
    # One value per *run*, not per cell: collapsing the seeds here would report
    # the mean of one seed each instead of the mean of all 81.
    raw_values = [row["delta_raw_nll"] for row in rows]

    print_cells(cells, subjects, bands)
    print_winners(winners, subjects, bands)
    print_band_effects(effects, subjects)
    print_decomposition(decomposition)

    result = assess_band_probe(
        winners=winners, effects=effects, cell_table_=cells, subjects=subjects, bands=bands, seeds=seeds
    )
    print_verdict(result)

    print("\n=== 附加：raw NLL 尺度（sensitivity，不得取代 primary）===")
    raw_mean = _mean(raw_values)
    print(f"全部配对的 raw ΔNLL 均值 = {raw_mean:+.6f}（{len(raw_values)} 条）")

    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(
        json.dumps(
            {
                "primary_scale": "log(val_best_nll)",
                "anchor_operator": args.anchor_operator,
                "anchor_root": str(args.anchor_root),
                "probe_root": str(args.probe_root),
                "session1_closed": True,
                "notice": SESSION1_NOTICE,
                "early_epoch_threshold": EARLY_EPOCH,
                "early_overfit_note": (
                    "diagnostic only; the threshold is post-hoc from the previous 90 runs and "
                    "must not be used as a selection criterion or to drop runs"
                ),
                "subjects": subjects,
                "bands": list(bands),
                "seeds": seeds,
                "paired_runs": len(rows),
                "unpaired": missing,
                "unreadable": skipped,
                "cells": cells,
                "winners": winners,
                "band_effects": effects,
                "decomposition": decomposition,
                # The count travels with the mean so a reader can tell a mean
                # over 81 runs from one over 27 collapsed cells.
                "raw_delta_nll_mean": raw_mean,
                "raw_delta_nll_mean_runs": len(raw_values),
                "anchors": {
                    f"{subject}/{seed}": entry["summary"]
                    for (subject, seed), entry in sorted(anchors.items())
                },
                "assessment": result,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"\n分析已写出：{args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
