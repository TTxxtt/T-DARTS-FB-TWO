"""Raw-EEG trials, optionally passed through a frontend.

The rest of the project consumes ``multiviewPython``, whose trials are already
``[22, 1000, 9]`` because the upstream ``filterBank`` transform filtered them.
To search over the filter bank itself the raw ``[22, 1000]`` trials are needed,
and those live in a sibling directory (``rawPython``) with the same
``dataLabels.csv`` layout.

This is deliberately a thin reader, not a second data pipeline: the CSV parsing,
the Session-0 split and the ``SearchSplit`` record come from
:mod:`tdarts.search_data`, so the frequency work and every earlier stage split
the data identically by construction rather than by agreement.
"""

from __future__ import annotations

import csv
import pickle
from pathlib import Path
from typing import Sequence

import torch
from torch.utils.data import Dataset

from tdarts.search_data import SearchSplit, session0_split_index

__all__ = ["RawTrialDataset", "load_raw_session0_split"]

#: Raw trials are one electrode x time matrix, in this exact shape.
RAW_SHAPE = (22, 1000)


def _read_rows(root: Path) -> list[dict]:
    labels_path = root / "dataLabels.csv"
    if not labels_path.is_file():
        raise FileNotFoundError(f"missing BCI-IV-2a labels file: {labels_path}")
    with labels_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"relativeFilePath", "label", "subject", "session"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"{labels_path} does not look like a dataLabels.csv")
    return rows


class RawTrialDataset(Dataset):
    """``[22, 1000]`` trials, or the frontend's ``[9, 22, 1000]`` output.

    With ``frontend=None`` the raw trial is returned untouched, which is what a
    frontend that is itself a module needs; with a frontend the transform is
    applied here so a dataset item is always the tensor the model consumes.
    """

    def __init__(
        self,
        data_root: str | Path,
        rows: Sequence[dict],
        indices: Sequence[int],
        *,
        frontend=None,
        preload: bool = False,
    ) -> None:
        self.data_root = Path(data_root).expanduser().resolve()
        self.rows = tuple(rows[index] for index in indices)
        if not self.rows:
            raise ValueError("dataset subset cannot be empty")
        self.frontend = frontend
        self._cache: list[tuple[torch.Tensor, torch.Tensor] | None] = [None] * len(self.rows)
        if preload:
            for index in range(len(self.rows)):
                self._cache[index] = self._load(index)

    def __len__(self) -> int:
        return len(self.rows)

    def _load(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.rows[index]
        with (self.data_root / row["relativeFilePath"]).open("rb") as handle:
            trial = pickle.load(handle)
        data = trial["data"]
        data = (
            data.to(dtype=torch.float32)
            if isinstance(data, torch.Tensor)
            else torch.as_tensor(data, dtype=torch.float32)
        )
        if tuple(data.shape) != RAW_SHAPE:
            raise ValueError(
                f"expected raw trial {RAW_SHAPE}, got {tuple(data.shape)} in "
                f"{row['relativeFilePath']}; is --data-root pointing at "
                f"rawPython rather than multiviewPython?"
            )
        if self.frontend is not None:
            data = self.frontend(data.unsqueeze(0)).squeeze(0)
        return data, torch.tensor(int(row["label"]), dtype=torch.long)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        cached = self._cache[index]
        if cached is None:
            cached = self._load(index)
            self._cache[index] = cached
        return cached

    def describe(self) -> dict:
        return {
            "data_root": str(self.data_root),
            "trials": len(self.rows),
            "raw_shape": list(RAW_SHAPE),
            "frontend": self.frontend.describe() if self.frontend is not None else None,
        }


def load_raw_session0_split(
    data_root: str | Path,
    subject: str | int = "003",
    *,
    frontend=None,
    preload: bool = False,
) -> tuple[RawTrialDataset, RawTrialDataset, SearchSplit]:
    """Session-0 train/validation split over raw trials.

    The boundary comes from :func:`tdarts.search_data.session0_split_index`, the
    same function the ``multiviewPython`` loader uses, so the two pipelines
    cannot drift into different splits.
    """

    root = Path(data_root).expanduser().resolve()
    subject_text = f"{int(subject):03d}" if str(subject).isdigit() else str(subject)
    rows = _read_rows(root)
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
        RawTrialDataset(root, session0, split.train_indices, frontend=frontend, preload=preload),
        RawTrialDataset(root, session0, split.val_indices, frontend=frontend, preload=preload),
        split,
    )
