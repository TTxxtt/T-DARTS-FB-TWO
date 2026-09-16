"""Compare the mini YAML parser against PyYAML on the real servers file.

The hand-written parser in tools/sync_servers_config.py exists so that the JSON
mirror can be regenerated without PyYAML.  That is only safe if it agrees with
a real YAML parser, so this asserts exactly that.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import sync_servers_config as sync  # noqa: E402

try:
    import yaml
except ImportError:
    print("SKIP: PyYAML not installed, cannot cross-check")
    raise SystemExit(0)

text = (ROOT / "configs" / "servers.yaml").read_text(encoding="utf-8")

reference = yaml.safe_load(text)
mine = sync.parse_yaml_mapping(text)

if reference == mine:
    print("MATCH: mini parser agrees with PyYAML")
    print(json.dumps(mine, indent=2, ensure_ascii=False)[:600])
    raise SystemExit(0)

print("MISMATCH")
import difflib  # noqa: E402

ref_lines = json.dumps(reference, indent=2, ensure_ascii=False).splitlines()
mine_lines = json.dumps(mine, indent=2, ensure_ascii=False).splitlines()
for line in difflib.unified_diff(
    ref_lines, mine_lines, "PyYAML", "mini-parser", lineterm="", n=2
):
    print(line)
raise SystemExit(1)
