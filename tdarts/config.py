"""Path and environment configuration interface for T-DARTS-FB.

The problem this solves: the same code has to run on a laptop and on two
different GPU servers, where data roots, output roots, conda environments and
Python interpreters all differ.  Instead of scattering absolute paths across
training scripts and dataset configs, every machine-specific value lives in one
file (``configs/servers.yaml``) under a named *profile*.

Typical use::

    from tdarts.config import get_profile

    prof = get_profile("serverA")
    data_root = prof.path("data_root")
    out_dir = prof.path("output_root", "bci42a", "subj003", "fold1")

Selecting a profile, in order of precedence:

1. ``explicit`` argument (what a ``--profile`` CLI flag would pass)
2. ``T_DARTS_PROFILE`` environment variable
3. ``auto`` hostname rules in the config file
4. the ``default_profile`` key in the config file

Only the standard library is required.  YAML is used when PyYAML is available;
otherwise a JSON file with the same schema is read instead.
"""

from __future__ import annotations

import dataclasses
import json
import os
import platform
import socket
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

__all__ = [
    "ProfileError",
    "ServerProfile",
    "resolve_profile",
    "get_profile",
    "load_servers_file",
    "find_servers_file",
    "list_profiles",
    "describe",
    "DEFAULT_FILENAMES",
    "ENV_PROFILE",
]

#: Filenames searched for, in order, when no explicit path is given.
DEFAULT_FILENAMES = (
    os.path.join("configs", "servers.yaml"),
    os.path.join("configs", "servers.yml"),
    os.path.join("configs", "servers.json"),
)

ENV_PROFILE = "T_DARTS_PROFILE"
ENV_SERVERS_FILE = "T_DARTS_SERVERS"

# ======================================================================
# Temporal search-space constants
# ======================================================================
# Stage 1 of T-DARTS-FB.  These live here so that no value is duplicated
# across temporal_ops.py, backbone.py and the tests.
#
# Shape convention, matching the official FBNAS multiview input:
#
#     full input   [B, 9, C, T]        C = 22 electrodes, T = 1000 samples
#     one band     [B, 3, C, T]        the 9 filter-bank bands split 3/3/3
#     one path     [B, 6, C, T]        this stage's temporal operator output
#
# The 9 bands are grouped by index into Low / Mid / High:
#
#     Low  = bands 0,1,2      Mid = bands 3,4,5      High = bands 6,7,8
#
# ----------------------------------------------------------------------
# DELIBERATE DEVIATION FROM OFFICIAL FBNAS -- see docs/fbnas_audit.md
# ----------------------------------------------------------------------
# Official FBNAS uses kernel 15 for every band and varies only the dilation
# (1/2/4/8), giving effective receptive fields 15/29/57/113.
#
# In earlier design notes these were instead written as frequency-specific
# kernels (Low 25 / Mid 11 / High 7) with RF_SPACE {25,49,97,193} etc.  Those
# two descriptions cannot both hold: 1 + (25-1)*1 = 25, not 97.
#
# This stage follows the official FBNAS formulation -- one shared kernel of 15
# and one shared RF ladder -- precisely so that Low/Mid/High are symmetric and
# frequency-specific kernels are not yet an extra experimental variable.  That
# makes this stage's RF ladder 15/29/57/113, *not* 25/49/97/193.
# ======================================================================

#: Base kernel length for the dilated / depthwise-separable operators.
#: Identical across bands by design in this stage.
BASE_KERNEL = {
    "Low": 15,
    "Mid": 15,
    "High": 15,
}

#: Dilations tried per band.  Index-aligned with every RF_SPACE entry.
DILATIONS = [1, 2, 4, 8]

#: Target effective receptive fields, computed as 1 + (kernel - 1) * dilation.
#: All three bands share the same ladder in this stage; the dict is kept keyed
#: by band so that frequency-specific ladders can be introduced later without
#: touching call sites.  When that happens, DILATIONS must be recomputed per
#: band rather than shared.
RF_SPACE = {
    "Low": [15, 29, 57, 113],
    "Mid": [15, 29, 57, 113],
    "High": [15, 29, 57, 113],
}

#: The four temporal operator families.
OPERATORS = [
    "dilated",
    "normal",
    "dwsep",
    "lkdw",
]

#: Input channels for one path (one Low/Mid/High group of 3 filter-bank bands).
IN_CHANNELS = 3

#: Output channels produced by one temporal operator.
PATH_CHANNELS = 6

#: Number of parallel temporal paths per band.  Two paths give 2 * PATH_CHANNELS
#: = 12 channels, which is the per-band width the backbone's SCB expects.
NUM_PATHS = 2

#: Band names in the order they are split out of the 9 filter-bank channels.
BANDS = ("Low", "Mid", "High")

#: The declared grid is 4 operators x 4 receptive fields = 16 per band, but the
#: RF15 row contains two aliases: at dilation 1 `dilated` == `normal` and
#: `dwsep` == `lkdw`.  Collapsing them leaves 14 distinct structures per band
#: (2 + 4 + 4 + 4).  See `tdarts.temporal_ops.canonical_candidates`, which
#: derives this rather than hard-coding it.
#
#: Initial value of the DARTS architecture logits.  Zero would be exactly
#: uniform; a small jitter breaks the symmetry between candidates so that
#: gradient descent has a direction to move in.  Stage 2 does not yet update
#: these, so the mixture starts (near-)uniform on purpose.
ALPHA_INIT_SCALE = 1e-3

#: Number of filter-bank bands in the full input, and per Low/Mid/High group.
NUM_BANDS = 9
NUM_BANDS_PER_GROUP = 3

#: EEG electrode count and time length for BCI-IV-2a.
NUM_ELECTRODES = 22
NUM_TIMEPOINTS = 1000

#: Candidate-level normalisation.  BatchNorm2d(PATH_CHANNELS, affine=False) is
#: applied after each operator so that differently-scaled operators do not
#: skew a future softmax architecture gradient.  The receptive-field audit must
#: turn this OFF, because BatchNorm is affine and would otherwise distort a
#: gradient-based RF measurement.
USE_CANDIDATE_NORM = True

#: Backbone dimensions inherited from official FBNAS (do not change here).
#: feature width per band = NUM_PATHS * PATH_CHANNELS = 12
#: SCB input channels    = 3 bands * 12          = 36   (NUM_FEAT below)
#: flat feature size     = NUM_FEAT * SCB_DILATABILITY * STRIDEFACTOR = 2304
NUM_CLASSES = 4
SCB_DILATABILITY = 8
STRIDEFACTOR = 8

#: Per-band feature width after the temporal cells are concatenated.  Derived,
#: but named because the official FBNAS code passes it as `num_Feat=36`.
NUM_FEAT = 3 * NUM_PATHS * PATH_CHANNELS


def band_groups() -> dict:
    """Return the band index ranges for each group.

    ``{'Low': (0, 3), 'Mid': (3, 6), 'High': (6, 9)}``
    """
    step = NUM_BANDS_PER_GROUP
    names = ("Low", "Mid", "High")
    return {name: (i * step, (i + 1) * step) for i, name in enumerate(names)}

# Path-like keys that get ~ and ${VAR} expansion.  Anything else is copied
# verbatim so that non-path settings (conda_env, python, ...) cannot be mangled
# by an accidental environment variable.
_PATH_KEYS = ("data_root", "output_root", "upstream_root")

# Keys an environment variable may override, and their variable names.
_ENV_OVERRIDES = {
    "T_DARTS_DATA_ROOT": "data_root",
    "T_DARTS_OUTPUT_ROOT": "output_root",
    "T_DARTS_CONDA_ENV": "conda_env",
}

# The set of profile keys reachable through a scoped variable such as
# T_DARTS__serverA__DATA_ROOT.
_ENV_OVERRIDE_KEYS = frozenset(_ENV_OVERRIDES.values())

_YAML_IMPORT_ERROR: str | None = None
try:  # pragma: no cover - exercised by which library happens to be installed
    import yaml as _yaml
except Exception as exc:  # pragma: no cover
    _yaml = None
    _YAML_IMPORT_ERROR = str(exc)


class ProfileError(RuntimeError):
    """Raised when the servers file or a requested profile is unusable."""


@dataclass(frozen=True)
class ServerProfile:
    """A fully resolved, machine-specific configuration.

    Attributes
    ----------
    name:
        Profile key this was loaded from.
    host:
        Hostname this profile is meant for, or ``None`` for "any machine".
    kind:
        ``"local"`` or ``"remote"``.  Only used to validate path style and to
        report intent; it does not change how paths are expanded.
    data_root:
        Root directory holding datasets.
    output_root:
        Root directory for run artifacts (checkpoints, logs, metrics).
    upstream_root:
        Where the frozen upstream FBNAS checkout lives, if any.
    conda_env:
        Conda environment name to activate, or ``None``.
    python:
        Interpreter to invoke, or ``None`` to use the current one.
    extras:
        Any additional profile keys, preserved untouched.
    source:
        Path of the file this profile came from.
    """

    name: str
    host: str | None = None
    kind: str = "local"
    data_root: str = ""
    output_root: str = ""
    upstream_root: str | None = None
    conda_env: str | None = None
    python: str | None = None
    extras: Mapping[str, Any] = field(default_factory=dict)
    source: str | None = None

    def __post_init__(self) -> None:
        if not self.data_root:
            raise ProfileError(f"profile {self.name!r}: 'data_root' is required")
        if not self.output_root:
            raise ProfileError(f"profile {self.name!r}: 'output_root' is required")

        if self.kind not in ("local", "remote"):
            raise ProfileError(
                f"profile {self.name!r}: kind must be 'local' or 'remote', "
                f"got {self.kind!r}"
            )

        # A remote profile is meant for a Linux GPU box.  Catching a Windows
        # path here turns a confusing "file not found on a machine you cannot
        # log into" into an immediate, local error.
        if self.kind == "remote":
            for key in _PATH_KEYS:
                value = getattr(self, key)
                if value and not value.startswith("/"):
                    raise ProfileError(
                        f"profile {self.name!r}: kind='remote' requires POSIX "
                        f"absolute paths, but {key}={value!r}. Use "
                        f"kind='local' for Windows paths."
                    )

    # -- path helpers ---------------------------------------------------
    def path(self, *parts: str) -> Path:
        """Join ``parts`` onto ``data_root``.

        With no arguments returns ``data_root`` itself.  The distinction from
        :meth:`output_path` is deliberate: writing a run artifact into the
        dataset directory by accident is a common and annoying mistake.
        """
        return Path(self.data_root).joinpath(*parts)

    def output_path(self, *parts: str) -> Path:
        """Join ``parts`` onto ``output_root``."""
        return Path(self.output_root).joinpath(*parts)

    def require_data_root(self) -> Path:
        """Return ``data_root`` as a :class:`Path`, failing if it is missing.

        Call this at the point where data is actually needed, not at import
        time, so that unit tests on a machine without the dataset still work.
        """
        p = Path(self.data_root)
        if not p.is_dir():
            raise ProfileError(
                f"profile {self.name!r}: data_root does not exist: {p}"
            )
        return p

    def ensure_output_root(self) -> Path:
        """Create ``output_root`` if needed and return it."""
        p = Path(self.output_root)
        p.mkdir(parents=True, exist_ok=True)
        return p

    # -- reporting ------------------------------------------------------
    def describe(self) -> str:
        lines = [
            f"profile      : {self.name}",
            f"source       : {self.source or '(built-in)'}",
            f"kind         : {self.kind}",
            f"host         : {self.host or '(any)'}",
            f"data_root    : {self.data_root}",
            f"output_root  : {self.output_root}",
        ]
        if self.upstream_root:
            lines.append(f"upstream_root: {self.upstream_root}")
        if self.conda_env:
            lines.append(f"conda_env    : {self.conda_env}")
        if self.python:
            lines.append(f"python       : {self.python}")
        for key in sorted(self.extras):
            lines.append(f"{key:<13}: {self.extras[key]}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["extras"] = dict(self.extras)
        return d


# ----------------------------------------------------------------------
# file discovery / loading
# ----------------------------------------------------------------------
def find_servers_file(
    explicit: str | os.PathLike | None = None,
    start: str | os.PathLike | None = None,
    filenames: Iterable[str] = DEFAULT_FILENAMES,
) -> Path:
    """Locate the servers file.

    Order: ``explicit`` path, then ``$T_DARTS_SERVERS``, then walking up from
    ``start`` (default: this file's directory) looking for each candidate
    filename.
    """
    if explicit:
        p = Path(explicit).expanduser()
        if not p.is_file():
            raise ProfileError(f"servers file not found: {p}")
        return p.resolve()

    from_env = os.environ.get(ENV_SERVERS_FILE)
    if from_env:
        p = Path(from_env).expanduser()
        if not p.is_file():
            raise ProfileError(
                f"${ENV_SERVERS_FILE} points at a missing file: {p}"
            )
        return p.resolve()

    here = Path(start) if start else Path(__file__).resolve().parent
    here = here.resolve()
    for directory in (here, *here.parents):
        for name in filenames:
            candidate = directory / name
            if candidate.is_file():
                return candidate

    raise ProfileError(
        "no servers file found. Looked for "
        + ", ".join(repr(n) for n in filenames)
        + f" from {here} upwards. Pass an explicit path or set "
        f"${ENV_SERVERS_FILE}."
    )


def load_servers_file(path: str | os.PathLike) -> dict:
    """Read and validate the top-level structure of a servers file.

    YAML is used when PyYAML is importable, JSON otherwise.  A ``.json``
    extension always forces the JSON reader.
    """
    p = Path(path)
    if not p.is_file():
        raise ProfileError(f"servers file not found: {p}")

    # utf-8-sig strips a BOM when present.  Windows editors add one readily and
    # both YAML and JSON parsers choke on it.
    text = p.read_text(encoding="utf-8-sig")
    suffix = p.suffix.lower()

    if not text.strip():
        raise ProfileError(f"{p}: file is empty")

    if suffix == ".json":
        try:
            raw = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ProfileError(f"{p}: invalid JSON: {exc}") from exc
    else:
        if _yaml is None:
            # The error PyYAML would have raised is rarely the useful part;
            # say what to do instead.
            raise ProfileError(
                f"{p} is YAML but PyYAML is not installed "
                f"({_YAML_IMPORT_ERROR}). Either install it "
                f"(`pip install pyyaml`) or provide the same content as "
                f"configs/servers.json, which needs no dependency."
            )
        try:
            raw = _yaml.safe_load(text)
        except Exception as exc:
            raise ProfileError(f"{p}: invalid YAML: {exc}") from exc

    if raw is None:
        raise ProfileError(f"{p}: file is empty")
    if not isinstance(raw, dict):
        raise ProfileError(
            f"{p}: top level must be a mapping, got {type(raw).__name__}"
        )
    if "profiles" not in raw:
        raise ProfileError(f"{p}: missing required top-level key 'profiles'")
    if not isinstance(raw["profiles"], dict) or not raw["profiles"]:
        raise ProfileError(f"{p}: 'profiles' must be a non-empty mapping")
    for name, body in raw["profiles"].items():
        if not isinstance(body, dict):
            raise ProfileError(
                f"{p}: profile {name!r} must be a mapping, got "
                f"{type(body).__name__}"
            )

    raw["_source"] = str(p.resolve())
    return raw


def list_profiles(path: str | os.PathLike | None = None) -> list[str]:
    """Return the profile names defined in the servers file."""
    data = load_servers_file(find_servers_file(path))
    return sorted(data["profiles"])


# ----------------------------------------------------------------------
# profile resolution
# ----------------------------------------------------------------------
def _auto_detect(profiles: Mapping[str, dict]) -> tuple[str | None, str]:
    """Match the current hostname against ``auto`` rules.

    A rule may be an exact hostname or a suffix prefixed with ``*``.
    Returns ``(profile_name_or_None, human_readable_explanation)``.
    """
    try:
        hostname = socket.gethostname().lower()
    except Exception:  # pragma: no cover - gethostname is effectively total
        return None, "hostname unavailable"

    tried = []
    for name, body in profiles.items():
        rule = body.get("auto")
        if not rule:
            continue
        candidates = rule if isinstance(rule, list) else [rule]
        for cand in candidates:
            cand = str(cand).lower()
            if cand.startswith("*"):
                matched = hostname.endswith(cand[1:])
            else:
                matched = hostname == cand
            tried.append(f"{name}:{cand}")
            if matched:
                return name, f"hostname {hostname!r} matched {cand!r}"
    if tried:
        return None, f"hostname {hostname!r} matched none of {', '.join(tried)}"
    return None, f"no 'auto' rules defined (hostname {hostname!r})"


def resolve_profile(
    explicit: str | None = None,
    servers_file: str | os.PathLike | None = None,
) -> tuple[str, dict, str]:
    """Pick a profile and return ``(name, raw_profile_dict, source_path)``.

    Precedence: ``explicit`` > ``$T_DARTS_PROFILE`` > ``auto`` hostname rules >
    ``default_profile``.
    """
    source = find_servers_file(servers_file)
    data = load_servers_file(source)
    profiles = data["profiles"]

    requested = explicit or os.environ.get(ENV_PROFILE) or None
    if requested:
        if requested not in profiles:
            raise ProfileError(
                f"profile {requested!r} not found in {source}. "
                f"Available: {', '.join(sorted(profiles))}"
            )
        return requested, profiles[requested], str(source)

    detected, why = _auto_detect(profiles)
    if detected:
        return detected, profiles[detected], str(source)

    fallback = data.get("default_profile")
    if fallback:
        if fallback not in profiles:
            raise ProfileError(
                f"{source}: default_profile {fallback!r} is not defined in "
                f"'profiles'. Available: {', '.join(sorted(profiles))}"
            )
        return fallback, profiles[fallback], str(source)

    raise ProfileError(
        f"no profile selected and no default_profile set in {source}. "
        f"({why}) Available: {', '.join(sorted(profiles))}. "
        f"Pass one explicitly or set ${ENV_PROFILE}."
    )


def _expand_paths(body: Mapping[str, Any]) -> dict:
    """Expand ~ and ${VAR} in path-like keys only."""
    out = dict(body)
    for key in _PATH_KEYS:
        value = out.get(key)
        if isinstance(value, str) and value:
            out[key] = os.path.expandvars(os.path.expanduser(value))
    return out


def _apply_env_overrides(body: dict, name: str) -> dict:
    """Apply ``T_DARTS_*`` overrides, honouring an optional profile scope.

    ``T_DARTS_DATA_ROOT`` applies to whatever profile is active;
    ``T_DARTS__serverA__DATA_ROOT`` applies only when ``serverA`` is active.
    """
    out = dict(body)
    for var, key in _ENV_OVERRIDES.items():
        value = os.environ.get(var)
        if value:
            out[key] = os.path.expandvars(os.path.expanduser(value))

    # Match case-insensitively on both sides: Windows upper-cases environment
    # variable names, so a profile named `serverA` arrives as
    # T_DARTS__SERVERA__DATA_ROOT while the config key is `serverA`.
    scoped_prefix = f"T_DARTS__{name}__".upper()
    for var, value in os.environ.items():
        if not var.upper().startswith(scoped_prefix):
            continue
        key = var[len(scoped_prefix):].lower()
        if key in _ENV_OVERRIDE_KEYS:
            out[key] = os.path.expandvars(os.path.expanduser(value))
    return out


_RESERVED = {
    "name", "host", "kind", "source", "auto", "description",
    "_generated_from",
    *_PATH_KEYS, "conda_env", "python",
}


def get_profile(
    explicit: str | None = None,
    servers_file: str | os.PathLike | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> ServerProfile:
    """Resolve and build the :class:`ServerProfile` for this machine.

    Parameters
    ----------
    explicit:
        Profile name, as a ``--profile`` CLI flag would supply.
    servers_file:
        Explicit path to the servers file.
    overrides:
        Highest-precedence key/value overrides, applied last.  Intended for
        tests and for CLI flags that beat the environment.
    """
    name, body, source = resolve_profile(explicit, servers_file)
    body = _expand_paths(body)
    body = _apply_env_overrides(body, name)
    if overrides:
        body = {**body, **dict(overrides)}

    host = body.get("host")
    if host is None:
        try:
            host = socket.gethostname()
        except Exception:  # pragma: no cover
            host = None

    extras = {k: v for k, v in body.items() if k not in _RESERVED}

    return ServerProfile(
        name=name,
        host=host,
        kind=body.get("kind", "local"),
        data_root=body.get("data_root", ""),
        output_root=body.get("output_root", ""),
        upstream_root=body.get("upstream_root"),
        conda_env=body.get("conda_env"),
        python=body.get("python"),
        extras=extras,
        source=source,
    )


def describe(
    explicit: str | None = None,
    servers_file: str | os.PathLike | None = None,
) -> str:
    """Human-readable summary, including how the profile was chosen."""
    try:
        resolve_profile(explicit, servers_file)
    except ProfileError as exc:
        return f"<error> {exc}"

    profile = get_profile(explicit, servers_file)

    if explicit:
        how = "explicit argument"
    elif os.environ.get(ENV_PROFILE):
        how = f"${ENV_PROFILE}"
    else:
        how = "hostname auto rules / default_profile"

    header = (
        f"host platform: {platform.system()} {platform.release()}\n"
        f"resolved via : {how}"
    )
    return header + "\n" + profile.describe()


if __name__ == "__main__":  # pragma: no cover
    from tdarts.cli import main

    raise SystemExit(main())
