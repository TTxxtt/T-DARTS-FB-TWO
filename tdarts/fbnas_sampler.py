"""FBNAS-style search primitives: subnet sampling and candidate enumeration.

This is a behavioural port of the frozen baseline's search machinery
(``FBNAS/codes/centralRepo/utils.py``, ``NAS.py``), kept in ``tdarts`` so the
new arms can be driven from the repo's own data pipeline.  It is deliberately
*mechanical*: the point of the port is that a search space can be proved
identical to upstream's instead of merely asserted to resemble it, so
``tests/test_arm_b_search.py`` compares :func:`random_choice` and
:func:`traverse_choices` against the frozen originals element for element.

What upstream does, and what is reproduced here:

* every training step draws **one** subnet and applies it to the whole batch --
  no architecture parameter, no softmax mixture, no architecture gradient;
* a band's subnet is a *subset* of its four candidates, of size 1..``m`` drawn
  uniformly, then the subset itself drawn uniformly without replacement;
* the final architecture is chosen by enumerating every candidate subset per
  band and taking the Cartesian product, so ``m=1`` gives 4 per band and
  ``m=2`` gives ``C(4,1) + C(4,2) = 10``.

The RNG consumption order matters and is mirrored exactly: per band, first the
cardinality then the subset, bands in ``Low``/``Mid``/``High`` order.
"""

from __future__ import annotations

import random as _random
from itertools import combinations
from typing import Mapping, Sequence

import numpy as np

from tdarts import config as C

__all__ = [
    "CANDIDATE_INDICES",
    "per_band_choices",
    "per_band_index",
    "traverse_choices",
    "choice_index",
    "choice_key",
    "validate_choice",
    "random_choice",
]

#: Candidate slot identifiers inside one band.  Upstream's ``ops = [0, 1, 2, 3]``.
CANDIDATE_INDICES: tuple[int, ...] = (0, 1, 2, 3)


def per_band_choices(m: int) -> tuple[tuple[int, ...], ...]:
    """Every admissible subset of one band's candidates, in upstream's order.

    Sizes run ``1..m`` and, within a size, :func:`itertools.combinations`
    lexicographic order -- so ``m=2`` yields ``(0,), (1,), (2,), (3,), (0,1),
    (0,2), (0,3), (1,2), (1,3), (2,3)``.  This is exactly the ``choice_list``
    that upstream's ``traverse_choice`` builds.
    """

    if m < 1:
        raise ValueError(f"m must be >= 1, got {m}")
    if m > len(CANDIDATE_INDICES):
        raise ValueError(
            f"m must be <= {len(CANDIDATE_INDICES)} (the candidate count), got {m}"
        )
    out: list[tuple[int, ...]] = []
    for size in range(1, m + 1):
        out.extend(combinations(CANDIDATE_INDICES, size))
    return tuple(out)


def per_band_index(m: int, subset: Sequence[int]) -> int:
    """Position of one band's ``subset`` in :func:`per_band_choices`.

    Matches upstream's ``find_choice_index``, including its tolerance for an
    unsorted input (it sorts before looking up).
    """

    return per_band_choices(m).index(tuple(sorted(int(i) for i in subset)))


def traverse_choices(m: int) -> tuple[dict[str, tuple[int, ...]], ...]:
    """The full enumerated search space: every ``(Low, Mid, High)`` combination.

    Iteration order is upstream's triple loop: ``Low`` varies slowest and
    ``High`` fastest, so index ``i`` maps to ``(i // n**2, (i // n) % n, i % n)``
    with ``n = len(per_band_choices(m))``.
    """

    per = per_band_choices(m)
    return tuple(
        {"Low": low, "Mid": mid, "High": high}
        for low in per
        for mid in per
        for high in per
    )


def choice_index(m: int, choice: Mapping[str, Sequence[int]]) -> int:
    """Index of ``choice`` inside :func:`traverse_choices`, or ``-1`` if absent."""

    per = per_band_choices(m)
    n = len(per)
    return (
        per.index(tuple(sorted(choice["Low"]))) * n * n
        + per.index(tuple(sorted(choice["Mid"]))) * n
        + per.index(tuple(sorted(choice["High"])))
    )


def choice_key(choice: Mapping[str, Sequence[int]]) -> tuple[tuple[int, ...], ...]:
    """A hashable, order-insensitive key for a subnet (bands in canonical order)."""

    return tuple(tuple(sorted(int(i) for i in choice[band])) for band in C.BANDS)


def validate_choice(
    choice: Mapping[str, Sequence[int]], *, m: int | None = None
) -> dict[str, tuple[int, ...]]:
    """Normalise and check one subnet, returning bands in canonical order.

    Raises rather than silently repairing: a malformed subnet reaching the
    supernet forward would otherwise be indistinguishable from a legitimate
    one that happens to score badly.
    """

    if not isinstance(choice, Mapping):
        raise TypeError(f"choice must be a mapping of band -> candidate indices, got {type(choice).__name__}")
    if set(choice) != set(C.BANDS):
        raise ValueError(f"choice must name exactly {list(C.BANDS)}, got {sorted(choice)}")

    normalised: dict[str, tuple[int, ...]] = {}
    for band in C.BANDS:
        raw = [int(i) for i in choice[band]]
        if not raw:
            raise ValueError(f"{band}: a subnet needs at least one candidate")
        if len(set(raw)) != len(raw):
            raise ValueError(f"{band}: candidate indices must be distinct, got {raw}")
        if min(raw) < 0 or max(raw) >= len(CANDIDATE_INDICES):
            raise ValueError(
                f"{band}: candidate indices must lie in "
                f"[0, {len(CANDIDATE_INDICES)}), got {raw}"
            )
        if m is not None and len(raw) > m:
            raise ValueError(f"{band}: {len(raw)} candidates exceeds m={m}")
        normalised[band] = tuple(sorted(raw))
    return normalised


def random_choice(
    m: int,
    *,
    rng: np.random.Generator | None = None,
    py_random=_random,
) -> dict[str, tuple[int, ...]]:
    """Draw one subnet, consuming RNG exactly as upstream's ``random_choice``.

    Per band: cardinality from ``numpy.random.randint(1, m+1)``, then the
    subset from ``random.sample(range(4), k)`` -- interleaved band by band.
    The defaults are the *global* numpy and stdlib generators, which is what
    upstream uses; passing a generator is for callers that want isolation.

    ``rng`` accepts either the legacy global module (``np.random``) or a modern
    :class:`numpy.random.Generator`.  The two spell the call differently --
    ``randint`` against ``integers`` -- and both treat ``high`` as exclusive,
    so the draw is the same either way.
    """

    if not isinstance(m, int) or m < 1:
        raise ValueError(f"m must be an integer >= 1, got {m!r}")
    if m > len(CANDIDATE_INDICES):
        raise ValueError(
            f"m must be <= {len(CANDIDATE_INDICES)} (the candidate count), got {m}"
        )
    source = np.random if rng is None else rng
    modern = getattr(source, "integers", None)

    choice: dict[str, tuple[int, ...]] = {}
    for band in C.BANDS:
        if modern is not None:
            size = int(modern(low=1, high=m + 1, size=1)[0])
        else:
            size = int(source.randint(low=1, high=m + 1, size=1)[0])
        choice[band] = tuple(py_random.sample(range(len(CANDIDATE_INDICES)), size))
    return choice
