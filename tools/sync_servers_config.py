"""Keep configs/servers.yaml and configs/servers.json in sync.

Both files ship so that the profile interface works whether or not PyYAML is
installed.  This script reads the YAML with a tiny, deliberately limited parser
(so it needs no third-party dependency) and writes the JSON mirror.

Usage::

    python tools/sync_servers_config.py           # rewrite the JSON mirror
    python tools/sync_servers_config.py --check    # fail if they differ

The parser supports exactly the subset used by configs/servers.yaml:

* ``key: value`` mappings at consistent indentation
* ``key:`` followed by a more-indented block
* ``- item`` sequences (inline ``[a, b]`` lists are also accepted)
* scalars: quoted strings, ints, floats, ``true``/``false``/``null``/``~``
* ``#`` comments and blank lines

It is not a general YAML parser and will raise on anything it does not
understand rather than guessing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
YAML_PATH = ROOT / "configs" / "servers.yaml"
JSON_PATH = ROOT / "configs" / "servers.json"


class YamlSubsetError(ValueError):
    """Raised when the file uses YAML this limited parser does not support."""


def _strip_comment(line: str) -> str:
    """Remove a trailing comment, respecting quotes."""
    out = []
    quote = None
    for ch in line:
        if quote:
            out.append(ch)
            if ch == quote:
                quote = None
        elif ch in "\"'":
            quote = ch
            out.append(ch)
        elif ch == "#":
            break
        else:
            out.append(ch)
    return "".join(out).rstrip()


def _scalar(token: str) -> Any:
    token = token.strip()
    if not token:
        return None
    if token[0] in "\"'" and token[-1] == token[0] and len(token) >= 2:
        return token[1:-1]
    low = token.lower()
    if low in ("null", "~", "none"):
        return None
    if low == "true":
        return True
    if low == "false":
        return False
    if token.startswith("[") and token.endswith("]"):
        inner = token[1:-1].strip()
        if not inner:
            return []
        return [_scalar(part) for part in inner.split(",")]
    try:
        return int(token)
    except ValueError:
        pass
    try:
        return float(token)
    except ValueError:
        pass
    return token


def parse_yaml_mapping(text: str) -> dict:
    """Parse the supported YAML subset into plain Python data."""

    # (indent, content) for every significant line.
    lines: list[tuple[int, str]] = []
    for raw in text.splitlines():
        stripped = _strip_comment(raw)
        if not stripped.strip():
            continue
        indent = len(stripped) - len(stripped.lstrip(" "))
        lines.append((indent, stripped.strip()))

    if not lines:
        raise YamlSubsetError("file is empty")

    pos = 0

    def parse_block(indent: int, allow_sequence: bool) -> Any:
        nonlocal pos
        if pos >= len(lines):
            return None

        if lines[pos][1].startswith("- "):
            if not allow_sequence:
                raise YamlSubsetError(
                    f"line {pos + 1}: unexpected sequence item "
                    f"{lines[pos][1]!r}"
                )
            items = []
            while pos < len(lines):
                cur_indent, content = lines[pos]
                if cur_indent != indent or not content.startswith("- "):
                    break
                items.append(_scalar(content[2:]))
                pos += 1
            return items

        result: dict[str, Any] = {}
        while pos < len(lines):
            cur_indent, content = lines[pos]
            if cur_indent < indent:
                break
            if cur_indent > indent:
                raise YamlSubsetError(
                    f"line {pos + 1}: unexpected indentation in "
                    f"{content!r}"
                )
            if content.startswith("- "):
                break
            if ":" not in content:
                raise YamlSubsetError(
                    f"line {pos + 1}: expected 'key: value', got {content!r}"
                )
            key, _, rest = content.partition(":")
            key = key.strip()
            rest = rest.strip()
            pos += 1
            if rest:
                result[key] = _scalar(rest)
            else:
                # Nested block or sequence, if it is more indented.
                if pos < len(lines) and lines[pos][0] > cur_indent:
                    nested_indent = lines[pos][0]
                    result[key] = parse_block(nested_indent, True)
                elif (
                    pos < len(lines)
                    and lines[pos][0] == cur_indent
                    and lines[pos][1].startswith("- ")
                ):
                    result[key] = parse_block(cur_indent, True)
                else:
                    result[key] = None
        return result

    return parse_block(lines[0][0], False)


def to_json_text(data: dict) -> str:
    """Render the JSON mirror.

    The note about this file being generated lives in the YAML itself (as the
    ``_generated_from`` key) rather than being injected here, so that parsing
    the YAML and parsing the JSON yield identical structures.
    """
    return json.dumps(data, indent=2, ensure_ascii=False) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--check",
        action="store_true",
        help="exit non-zero if the JSON mirror is out of date",
    )
    args = ap.parse_args(argv)

    yaml_text = YAML_PATH.read_text(encoding="utf-8-sig")
    try:
        parsed = parse_yaml_mapping(yaml_text)
    except YamlSubsetError as exc:
        print(f"error: {YAML_PATH}: {exc}", file=sys.stderr)
        return 2

    expected = to_json_text(parsed)

    if args.check:
        if not JSON_PATH.is_file():
            print(f"error: {JSON_PATH} is missing", file=sys.stderr)
            return 1
        if JSON_PATH.read_text(encoding="utf-8-sig") != expected:
            print(
                f"error: {JSON_PATH} is out of date with {YAML_PATH}. "
                f"Run: python tools/sync_servers_config.py",
                file=sys.stderr,
            )
            return 1
        print("configs/servers.json is in sync")
        return 0

    JSON_PATH.write_text(expected, encoding="utf-8", newline="\n")
    print(f"wrote {JSON_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
