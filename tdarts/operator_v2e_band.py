"""Band-specific mechanism probe: the configuration vocabulary, in one place.

The Expressive-V2 grid fixed **one** family for all three bands, so a preference
that lives in a *single* band -- Low wanting a gate while High wants attention --
was averaged across the three bands and was structurally invisible.  Both
generations' verdicts said as much: a large global operator effect, no
subject x operator interaction.

This stage asks the narrower question that the previous design could not:
does a subject prefer a *different mechanism in a different band*?

The design is a controlled single-band replacement.  The baseline puts the
anchor in all three bands; each configuration then swaps the family in exactly
one of them and leaves the other two on the anchor.  Swapping two or three at
once would confound the bands with each other, so
:func:`validate_band_families` refuses it rather than trusting the grid to be
written correctly.

Vocabulary, not behaviour
-------------------------
Nothing here builds a module or runs a model.  It is imported by the network,
the entry point, the submit driver, the analyzer and the tests so that the nine
configurations and their slug spelling have exactly one definition.  A slug is
the only thing tying an on-disk run back to a configuration, so two
disagreeing spellings would be a silent mis-pairing in the paired analysis --
the failure this module exists to make impossible.
"""

from __future__ import annotations

from tdarts import config as C

__all__ = [
    "ANCHOR_FAMILY",
    "ANCHOR_SLUG",
    "BAND_FAMILIES",
    "EXCLUDED_FAMILIES",
    "STAGE_PREFIX",
    "VARYING_FAMILIES",
    "config_of",
    "parse_slug",
    "probe_configs",
    "slug_for",
    "validate_band_families",
]

#: The run tree this stage owns.  Every run leaf, log and ledger carries it, and
#: the entry point refuses an output root that does not.  The Expressive grid is
#: frozen under ``operator_v2e``; a band run landing there would be read by that
#: generation's analyzer, which knows nothing about per-band configurations.
STAGE_PREFIX = "operator_v2e_band"

#: The baseline family: one dilated convolution, i.e. the Expressive anchor.
ANCHOR_FAMILY = "dilated_e"

#: Slug of the all-anchor configuration.  It is *not* in the 81-run grid -- its
#: nine runs (3 subjects x 3 seeds) already exist as the Expressive ``dilated_e``
#: arm and are reused as the paired reference.  The spelling exists so the
#: entry point can be pointed at it and checked against that archived arm.
ANCHOR_SLUG = "all_dilated_e"

#: Families allowed to occupy a band in this stage.  ``local_attention_e`` is
#: deliberately absent: it is the worst family on validation NLL in both
#: generations and the one with the highest early-overfit rate, and a probe
#: whose purpose is to detect band-specific preference should not spend a third
#: of its grid on a mechanism already rejected.  See :data:`EXCLUDED_FAMILIES`.
BAND_FAMILIES: tuple[str, ...] = ("dilated_e", "gated_e", "dynamic_e", "band_gated_e")

#: Subset that may be the *varying* band.  ``dilated_e`` is missing by
#: construction: it is the reference, and "replace the anchor with the anchor"
#: is the baseline, not a probe.
VARYING_FAMILIES: tuple[str, ...] = ("dynamic_e", "gated_e", "band_gated_e")

#: Families that exist in ``E_OPERATOR_NAMES`` but are out of scope here, with
#: the reason.  Kept as data so the entry point's error message states the
#: decision rather than just rejecting an input.
EXCLUDED_FAMILIES: dict[str, str] = {
    "local_attention_e": (
        "dropped from the search space after the Expressive grid: worst "
        "validation NLL of the five families and the highest early-overfit "
        "rate (5/9), with the best epoch arriving at a median of 14. It fits "
        "the training set (9/9 reach 100% train accuracy), so this is a "
        "generalisation failure rather than an optimisation one, and the band "
        "probe is not the experiment that would separate those."
    ),
}

#: The varying band, its family, and the slug.  Nine configurations; two or
#: three bands varying is out of scope by design.
_PROBE_GRID: tuple[tuple[str, str], ...] = tuple(
    (band, family) for band in C.BANDS for family in VARYING_FAMILIES
)


def probe_configs() -> tuple[tuple[str, str], ...]:
    """The nine ``(band, family)`` pairs of the single-band replacement grid."""

    return _PROBE_GRID


def config_of(low: str, mid: str, high: str) -> dict[str, str]:
    """Package three band families for a summary or a config file."""

    return {"Low": low, "Mid": mid, "High": high}


def validate_band_families(low: str, mid: str, high: str) -> tuple[str | None, str]:
    """Check a band configuration and return the ``(band, family)`` that varies.

    The band is ``None`` for the all-anchor baseline, which has no varying band.

    Raises ``ValueError`` when a family is out of scope or when more than one
    band leaves the anchor.  The second rule is the design itself: this stage
    measures one band at a time, and a two-band change would attribute a joint
    effect to either band alone.
    """

    families = config_of(low, mid, high)
    for band, family in families.items():
        if family in EXCLUDED_FAMILIES:
            raise ValueError(
                f"{band} family {family!r} is out of scope for this stage: "
                f"{EXCLUDED_FAMILIES[family]}"
            )
        if family not in BAND_FAMILIES:
            raise ValueError(
                f"{band} family {family!r} is not one of {BAND_FAMILIES}; the "
                f"stage is a single-band replacement probe over the Expressive "
                f"candidates that survived the global grid"
            )
    varying = [(band, family) for band, family in families.items() if family != ANCHOR_FAMILY]
    if len(varying) > 1:
        raise ValueError(
            f"exactly one band may differ from {ANCHOR_FAMILY!r}, got "
            f"{varying}; replacing several bands at once confounds the bands "
            f"with each other"
        )
    return varying[0] if varying else (None, ANCHOR_FAMILY)


def slug_for(low: str, mid: str, high: str) -> str:
    """The one-word config name written into the run leaf and the summaries.

    ``all_dilated_e`` for the baseline, ``low_dynamic_e`` / ``high_band_gated_e``
    for a probe.  Only the varying band is spelled out: the other two are the
    anchor by :func:`validate_band_families`, so repeating them would encode the
    same information twice and invite the two copies to disagree.
    """

    band, family = validate_band_families(low, mid, high)
    return ANCHOR_SLUG if band is None else f"{band.lower()}_{family}"


def parse_slug(slug: str) -> dict[str, str]:
    """Recover the three band families from a slug.

    Raises ``ValueError`` on anything that is not a slug this stage produced,
    which is what stops a stray run from being paired against an anchor as if
    it were a probe.
    """

    if slug == ANCHOR_SLUG:
        return config_of(ANCHOR_FAMILY, ANCHOR_FAMILY, ANCHOR_FAMILY)
    parts = slug.split("_", 1)
    if len(parts) != 2:
        raise ValueError(f"{slug!r} is not a band-probe slug")
    band_key, family = parts
    band = next((name for name in C.BANDS if name.lower() == band_key), None)
    if band is None or family not in VARYING_FAMILIES:
        raise ValueError(f"{slug!r} is not a band-probe slug")
    families = config_of(ANCHOR_FAMILY, ANCHOR_FAMILY, ANCHOR_FAMILY)
    families[band] = family
    return families
