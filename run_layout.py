"""Canonical on-disk layout for search and training runs.

Every run lands at::

    <root>/<dataset>/<phase>_s<subject>_seed<seed>[_<arm>]/

and its progress log mirrors the same two levels under the log root::

    <log_root>/<dataset>/<phase>_s<subject>_seed<seed>[_<arm>].log

``phase`` is ``search`` or ``train`` and sits inside the leaf rather than in a
directory level of its own, so a search directory keeps the plain
``search_s003_seed20250901`` spelling that the rest of the tooling has always
used.  ``arm`` is an optional suffix: pass it when both arms write under the
same root, omit it when the root already separates them (the ``run/`` harness
uses ``run/outputs/<arm>/`` as its root).
"""

from __future__ import annotations

from pathlib import Path


def subject_id(subject: str | int) -> str:
    """Normalise a subject identifier to its zero-padded directory form."""

    return f"{int(subject):03d}" if str(subject).isdigit() else str(subject)


def run_leaf(phase: str, subject: str | int, seed: int, arm: str | None = None) -> str:
    """Build the per-run leaf directory name."""

    leaf = f"{phase}_s{subject_id(subject)}_seed{seed}"
    return f"{leaf}_{arm}" if arm else leaf


def allocate(
    *,
    root: Path,
    log_root: Path,
    dataset: str,
    phase: str,
    subject: str | int,
    seed: int,
    arm: str | None = None,
) -> tuple[Path, Path]:
    """Return the ``(run_dir, log_path)`` pair for one run."""

    leaf = run_leaf(phase, subject, seed, arm)
    run_dir = Path(root) / dataset / leaf
    log_path = Path(log_root) / dataset / f"{leaf}.log"
    return run_dir, log_path
