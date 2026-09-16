"""T-DARTS-FB: Temporal DARTS research code.

Stage 1 provides the temporal operator pool and the fixed backbone:

* :mod:`tdarts.config`      -- machine profiles *and* the search-space constants
* :mod:`tdarts.temporal_ops` -- 4 operators x 4 receptive fields
* :mod:`tdarts.backbone`    -- SCB / LogVar / classifier, unchanged from FBNAS
* :mod:`tdarts.init_utils`  -- name-addressed deterministic initialisation

The official FBNAS baseline is vendored, unmodified, under ``FBNAS/`` and is not
imported from here.

Configuration entry point for machine-specific paths is
:func:`tdarts.config.get_profile`.
"""

from tdarts.config import (  # noqa: F401
    ProfileError,
    ServerProfile,
    describe,
    get_profile,
    list_profiles,
    resolve_profile,
)

__version__ = "0.1.0"

__all__ = [
    "ProfileError",
    "ServerProfile",
    "describe",
    "get_profile",
    "list_profiles",
    "resolve_profile",
    "__version__",
]
