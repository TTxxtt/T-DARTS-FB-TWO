"""BCI-IV-2a data access for Stage-3 architecture search only.

The split intentionally follows the official FBNAS hold-out protocol exactly:
for one subject, Session 0 remains in CSV order, its first ``ceil(0.8 * N)``
trials are training data and the remainder validation data.  Session 1 is never
returned by this module's search split.
"""

from __future__ import annotations

import csv
import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

__all__ = [
    "SearchSplit",
    "BCI42aSearchDataset",
    "load_session0_search_split",
    "load_subject_session",
    "session0_split_index",
]


@dataclass(frozen=True)
class SearchSplit:
    """Immutable Session-0 index split and provenance for one subject."""

    subject: str
    train_indices: tuple[int, ...]
    val_indices: tuple[int, ...]

    @property
    def train_size(self) -> int:
        return len(self.train_indices)

    @property
    def val_size(self) -> int:
        return len(self.val_indices)


def session0_split_index(size: int, train_fraction: float = 0.8) -> int:
    """FBNAS's ``ceil(N * (1 - validationSet))`` boundary."""

    if size < 2:
        raise ValueError(f"need at least two Session-0 trials, got {size}")
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must lie strictly between zero and one")
    boundary = math.ceil(size * train_fraction)
    if boundary == size:
        raise ValueError("split leaves no validation data")
    return boundary


class BCI42aSearchDataset(Dataset):
    """Subset of precomputed ``multiviewPython`` trials for one search role.

    ``preload`` reads every trial into memory up front.  Without it each
    ``__getitem__`` re-opens and re-unpickles the ``.dat`` file: one trial is
    ~774 KB and costs ~1 ms, and a training epoch makes three passes over the
    split (train, the frozen train-loss pass, and validation), so the retrain
    re-read roughly 650 MB per epoch from GPFS and spent close to half its wall
    clock waiting on the filesystem.  The upstream baseline preloads the same
    way -- ``eegDataset.createPartialDataset(..., loadNonLoadedData=True)``,
    called from ``ho.py:269`` -- which is why the FBNAS arm trains about 4x
    faster per epoch on identical data.

    Caching changes only how the tensor is obtained, never its value, so a run
    is numerically identical with or without it.
    """

    def __init__(
        self,
        data_root: str | Path,
        rows: Sequence[dict[str, str]],
        indices: Sequence[int],
        preload: bool = False,
    ):
        self.data_root = Path(data_root).expanduser().resolve()
        self.rows = tuple(rows[index] for index in indices)
        if not self.rows:
            raise ValueError("dataset subset cannot be empty")
        # Sparse: None until a sample has been read once.  With preload it is
        # filled eagerly and __getitem__ never touches the filesystem again.
        self._cache: list[tuple[torch.Tensor, torch.Tensor] | None] = [None] * len(self.rows)
        if preload:
            for index in range(len(self.rows)):
                self._cache[index] = self._load(index)

    def __len__(self) -> int:
        return len(self.rows)

    def _load(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.rows[index]
        relative_path = row["relativeFilePath"]
        with (self.data_root / relative_path).open("rb") as handle:
            trial = pickle.load(handle)
        data = trial["data"]
        if not isinstance(data, torch.Tensor):
            data = torch.as_tensor(data, dtype=torch.float32)
        else:
            data = data.to(dtype=torch.float32)
        if tuple(data.shape) != (22, 1000, 9):
            raise ValueError(
                f"expected trial [22, 1000, 9], got {tuple(data.shape)} in {relative_path}"
            )
        # The Stage-2 model accepts the baseline's five-dimensional layout.
        return data.unsqueeze(0), torch.tensor(int(row["label"]), dtype=torch.long)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        cached = self._cache[index]
        if cached is None:
            cached = self._load(index)
            self._cache[index] = cached
        return cached


def load_session0_search_split(
    data_root: str | Path,
    subject: str | int = "003",
    preload: bool = False,
) -> tuple[BCI42aSearchDataset, BCI42aSearchDataset, SearchSplit]:
    """Load only Session 0 and reproduce FBNAS's ordered 80/20 split."""

    root = Path(data_root).expanduser().resolve()
    labels_path = root / "dataLabels.csv"
    if not labels_path.is_file():
        raise FileNotFoundError(f"missing BCI-IV-2a labels file: {labels_path}")
    subject_text = f"{int(subject):03d}" if str(subject).isdigit() else str(subject)
    with labels_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"relativeFilePath", "label", "subject", "session"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"{labels_path} is not a multiviewPython dataLabels.csv file")
    session0 = [row for row in rows if row["subject"] == subject_text and row["session"] == "0"]
    if not session0:
        raise ValueError(f"no Session-0 samples found for Subject{subject_text}")
    boundary = session0_split_index(len(session0))
    split = SearchSplit(
        subject=subject_text,
        train_indices=tuple(range(boundary)),
        val_indices=tuple(range(boundary, len(session0))),
    )
    return (
        BCI42aSearchDataset(root, session0, split.train_indices, preload=preload),
        BCI42aSearchDataset(root, session0, split.val_indices, preload=preload),
        split,
    )


def load_subject_session(
    data_root: str | Path,
    subject: str | int = "003",
    session: int | str = 1,
    preload: bool = False,
) -> BCI42aSearchDataset:
    """Load all trials from one subject/session without changing CSV order."""

    root = Path(data_root).expanduser().resolve()
    labels_path = root / "dataLabels.csv"
    if not labels_path.is_file():
        raise FileNotFoundError(f"missing BCI-IV-2a labels file: {labels_path}")
    subject_text = f"{int(subject):03d}" if str(subject).isdigit() else str(subject)
    session_text = str(session)
    with labels_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    selected = [row for row in rows if row["subject"] == subject_text and row["session"] == session_text]
    if not selected:
        raise ValueError(f"no Session-{session_text} samples found for Subject{subject_text}")
    return BCI42aSearchDataset(root, selected, tuple(range(len(selected))), preload=preload)
