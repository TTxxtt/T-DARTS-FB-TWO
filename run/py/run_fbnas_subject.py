#!/usr/bin/env python
"""Run one subject of the official FBNAS pipeline from outside the frozen baseline.

This is the ONLY place that imports the vendored baseline.  It exists because
``ho.py`` cannot be invoked as a script: its ``__main__`` block reads positional
argv (ho.py:431 uses ``sys.argv[1:]``), and those positionals are still sitting
in ``sys.argv`` when ``nas_phase()`` -> ``NAS.get_args()`` -> ``argparse
.parse_args()`` (NAS.py:55) runs, which rejects them with SystemExit(2).

Set ``SUB`` to the index into ``sorted(set(subject_id))``, i.e. 0..8 for the nine
BCI-IV-2a subjects.
"""

import os
import sys
from pathlib import Path

# --- MUST precede `import ho` -------------------------------------------
# Importing from inside FBNAS/ otherwise drops __pycache__/*.pyc into the
# frozen baseline, which trips
# tests/test_fbnas_compatibility.py::test_no_extra_files_were_written_into_the_baseline.
# Same idiom as test_fbnas_compatibility.py:69-79.  Setting the flag (rather
# than restoring it in a finally) also covers every module ho.py imports.
sys.dont_write_bytecode = True
# ------------------------------------------------------------------------

# run/py/run_fbnas_subject.py -> run/py -> run -> the repository root.  Deriving
# this keeps the wrapper portable across the cluster and a local checkout
# instead of pinning one absolute path.
REPO = Path(os.environ.get("TDARTS_REPO", Path(__file__).resolve().parents[2]))

# ho.py:19 puts FBNAS/codes/centralRepo on sys.path itself once imported.
sys.path.insert(0, str(REPO / "FBNAS" / "codes" / "classify"))

import ho  # noqa: E402


def main() -> int:
    if "SUB" not in os.environ:
        raise SystemExit("need SUB in the environment (0..8, one BCI-IV-2a subject)")
    try:
        sub = int(os.environ["SUB"])  # 0..8, index into sorted(set(subject_id))
    except ValueError:
        raise SystemExit(f"SUB must be an integer 0..8, got {os.environ['SUB']!r}") from None
    if not 0 <= sub <= 8:
        raise SystemExit(f"SUB must be 0..8, got {sub}")
    # subTorun becomes range(t[0], t[1]) inside ho() (ho.py:250), so it is a
    # half-open [start, end) pair, NOT a list of subject ids.  A single-element
    # list raises IndexError there.
    ho.ho(
        datasetId=0,
        network="FBNASNet",
        nGPU="0",
        subTorun=[sub, sub + 1],
        AugumentEnable=False,
        centerLossEnable=False,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
