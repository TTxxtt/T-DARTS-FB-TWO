"""Command line interface for T-DARTS-FB.

Currently exposes a single command, ``profile``, which answers the question
"which machine-specific configuration is active here?" without having to read
any Python.

    t-darts-profile --list
    t-darts-profile
    t-darts-profile --profile serverB
"""

from __future__ import annotations

import argparse
import sys

from tdarts import config as _config

__all__ = ["main", "build_parser"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="t-darts-profile",
        description=(
            "Show the active T-DARTS-FB server profile. "
            "Selection order: --profile, $T_DARTS_PROFILE, hostname auto rules, "
            "then default_profile."
        ),
    )
    parser.add_argument(
        "--profile", default=None, help="profile name to resolve"
    )
    parser.add_argument(
        "--servers-file", default=None, help="path to the servers config file"
    )
    parser.add_argument(
        "--list", action="store_true", help="list available profile names"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.list:
            for name in _config.list_profiles(args.servers_file):
                print(name)
        elif args.profile is not None:
            # Resolve explicitly rather than via describe(), which deliberately
            # swallows ProfileError into a string and so would make this always
            # exit 0 -- wrong for a CLI that shell scripts check.
            print(_config.describe(args.profile, args.servers_file))
            _config.resolve_profile(args.profile, args.servers_file)
        else:
            print(_config.describe(None, args.servers_file))
            _config.resolve_profile(None, args.servers_file)
    except _config.ProfileError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
