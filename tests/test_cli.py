"""Tests for the `t-darts-profile` command line interface.

The important property here is the exit code: a CLI that reports success after
failing to find the requested profile would silently mask typos in shell
scripts and CI jobs.
"""

from __future__ import annotations

import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tdarts import cli  # noqa: E402
from tdarts import config as cfg  # noqa: E402

SERVERS = json.dumps(
    {
        "default_profile": "alpha",
        "profiles": {
            "alpha": {
                "kind": "local",
                "data_root": "/data/alpha",
                "output_root": "/out/alpha",
                "conda_env": "eegA",
            },
            "beta": {
                "kind": "remote",
                "data_root": "/gpfs/beta/data",
                "output_root": "/gpfs/beta/out",
            },
        },
    }
)


class TestCli(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        self.servers = self.tmp / "servers.json"
        self.servers.write_text(SERVERS, encoding="utf-8")

        clean = {k: v for k, v in os.environ.items() if not k.startswith("T_DARTS")}
        patcher = mock.patch.dict(os.environ, clean, clear=True)
        patcher.start()
        self.addCleanup(patcher.stop)

    def run_cli(self, *argv: str):
        """Invoke main() and capture stdout/stderr plus the exit code."""
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = cli.main([*argv])
        return code, out.getvalue(), err.getvalue()

    def base(self) -> list[str]:
        return ["--servers-file", str(self.servers)]

    def test_list_prints_profile_names(self):
        code, out, _ = self.run_cli("--list", *self.base())
        self.assertEqual(code, 0)
        self.assertEqual(out.split(), ["alpha", "beta"])

    def test_default_resolution_succeeds(self):
        code, out, _ = self.run_cli(*self.base())
        self.assertEqual(code, 0)
        self.assertIn("alpha", out)

    def test_explicit_profile_succeeds(self):
        code, out, _ = self.run_cli("--profile", "beta", *self.base())
        self.assertEqual(code, 0)
        self.assertIn("beta", out)
        self.assertIn("/gpfs/beta/data", out)

    def test_unknown_profile_exits_nonzero(self):
        """The regression guard: a bad --profile must not report success."""
        code, _, err = self.run_cli("--profile", "nope", *self.base())
        self.assertEqual(code, 1)
        self.assertIn("nope", err)

    def test_missing_servers_file_exits_nonzero(self):
        code, _, err = self.run_cli(
            "--servers-file", str(self.tmp / "absent.json")
        )
        self.assertEqual(code, 1)
        self.assertIn("not found", err)

    def test_broken_config_exits_nonzero(self):
        bad = self.tmp / "bad.json"
        bad.write_text("{ not json", encoding="utf-8")
        code, _, err = self.run_cli("--servers-file", str(bad))
        self.assertEqual(code, 1)
        self.assertTrue(err.strip())

    def test_list_with_bad_file_exits_nonzero(self):
        code, _, err = self.run_cli(
            "--list", "--servers-file", str(self.tmp / "absent.json")
        )
        self.assertEqual(code, 1)
        self.assertIn("not found", err)

    def test_env_profile_is_honoured(self):
        with mock.patch.dict(os.environ, {"T_DARTS_PROFILE": "beta"}):
            code, out, _ = self.run_cli(*self.base())
        self.assertEqual(code, 0)
        self.assertIn("beta", out)

    def test_help_exits_zero(self):
        with self.assertRaises(SystemExit) as ctx:
            with redirect_stdout(io.StringIO()):
                cli.main(["--help"])
        self.assertEqual(ctx.exception.code, 0)

    def test_shipped_config_is_reachable_from_cli(self):
        root = Path(__file__).resolve().parent.parent
        code, out, _ = self.run_cli(
            "--servers-file", str(root / "configs" / "servers.json")
        )
        self.assertEqual(code, 0)
        self.assertIn("local", out)

    def test_describe_swallows_errors_but_resolve_raises(self):
        """Document the deliberate difference between the two entry points."""
        text = cfg.describe("nope", self.servers)
        self.assertTrue(text.startswith("<error>"))
        with self.assertRaises(cfg.ProfileError):
            cfg.resolve_profile("nope", self.servers)


if __name__ == "__main__":
    unittest.main(verbosity=2)
