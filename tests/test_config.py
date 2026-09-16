"""Tests for the server-profile configuration interface.

Run with either of::

    python -m unittest discover -s tests -v
    python -m pytest tests -q

The suite deliberately passes both with and without PyYAML installed: the
library prefers YAML but falls back to JSON, so both branches are asserted
conditionally rather than assumed.
"""

from __future__ import annotations

import json
import os
import socket
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tdarts import config as cfg  # noqa: E402
from tdarts.config import ProfileError, ServerProfile, get_profile  # noqa: E402

HAS_YAML = cfg._yaml is not None

VALID_YAML = """\
default_profile: alpha

profiles:
  alpha:
    kind: local
    data_root: /data/alpha
    output_root: /out/alpha
    conda_env: eegA
  beta:
    kind: remote
    data_root: /gpfs/beta/data
    output_root: /gpfs/beta/out
    conda_env: eegB
    scheduler: slurm
"""

VALID_JSON = json.dumps(
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
                "conda_env": "eegB",
                "scheduler": "slurm",
            },
        },
    }
)


class TempServersFile(unittest.TestCase):
    """Base class providing a scratch directory and a clean environment."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

        # Isolate from the developer's real environment.
        clean = {
            k: v
            for k, v in os.environ.items()
            if not k.startswith("T_DARTS")
        }
        patcher = mock.patch.dict(os.environ, clean, clear=True)
        patcher.start()
        self.addCleanup(patcher.stop)

    def write(self, name: str, text: str) -> Path:
        p = self.tmp / name
        p.write_text(text, encoding="utf-8")
        return p

    def write_yaml(self, text: str = VALID_YAML) -> Path:
        return self.write("servers.yaml", text)

    def write_json(self, text: str = VALID_JSON) -> Path:
        return self.write("servers.json", text)


class TestLoading(TempServersFile):
    def test_yaml_loads_when_pyyaml_present(self):
        path = self.write_yaml()
        if not HAS_YAML:
            with self.assertRaises(ProfileError) as ctx:
                cfg.load_servers_file(path)
            self.assertIn("PyYAML", str(ctx.exception))
            self.assertIn("servers.json", str(ctx.exception))
            return
        data = cfg.load_servers_file(path)
        self.assertEqual(sorted(data["profiles"]), ["alpha", "beta"])
        self.assertEqual(data["_source"], str(path.resolve()))

    def test_json_loads_without_pyyaml(self):
        # The JSON reader must never depend on PyYAML.
        path = self.write_json()
        data = cfg.load_servers_file(path)
        self.assertEqual(sorted(data["profiles"]), ["alpha", "beta"])
        self.assertEqual(data["default_profile"], "alpha")

    def test_empty_file_rejected(self):
        path = self.write_json("")
        with self.assertRaises(ProfileError) as ctx:
            cfg.load_servers_file(path)
        self.assertIn("empty", str(ctx.exception))

    def test_missing_profiles_key_rejected(self):
        path = self.write_json(json.dumps({"default_profile": "alpha"}))
        with self.assertRaises(ProfileError) as ctx:
            cfg.load_servers_file(path)
        self.assertIn("profiles", str(ctx.exception))

    def test_empty_profiles_rejected(self):
        path = self.write_json(json.dumps({"profiles": {}}))
        with self.assertRaises(ProfileError) as ctx:
            cfg.load_servers_file(path)
        self.assertIn("non-empty", str(ctx.exception))

    def test_non_mapping_profile_rejected(self):
        path = self.write_json(json.dumps({"profiles": {"alpha": ["nope"]}}))
        with self.assertRaises(ProfileError) as ctx:
            cfg.load_servers_file(path)
        self.assertIn("must be a mapping", str(ctx.exception))

    def test_missing_file_rejected(self):
        with self.assertRaises(ProfileError) as ctx:
            cfg.load_servers_file(self.tmp / "does-not-exist.json")
        self.assertIn("not found", str(ctx.exception))

    def test_list_profiles(self):
        path = self.write_json()
        self.assertEqual(cfg.list_profiles(path), ["alpha", "beta"])


class TestResolution(TempServersFile):
    def test_explicit_profile_wins(self):
        path = self.write_json()
        prof = get_profile("beta", path)
        self.assertEqual(prof.name, "beta")
        self.assertEqual(prof.kind, "remote")

    def test_env_var_selects_profile(self):
        path = self.write_json()
        with mock.patch.dict(os.environ, {cfg.ENV_PROFILE: "beta"}):
            prof = get_profile(None, path)
        self.assertEqual(prof.name, "beta")

    def test_explicit_beats_env_var(self):
        path = self.write_json()
        with mock.patch.dict(os.environ, {cfg.ENV_PROFILE: "alpha"}):
            prof = get_profile("beta", path)
        self.assertEqual(prof.name, "beta")

    def test_default_profile_used_when_nothing_selected(self):
        path = self.write_json()
        # No `auto` rules in this file, so detection cannot fire.
        prof = get_profile(None, path)
        self.assertEqual(prof.name, "alpha")

    def test_unknown_profile_lists_alternatives(self):
        path = self.write_json()
        with self.assertRaises(ProfileError) as ctx:
            get_profile("gamma", path)
        msg = str(ctx.exception)
        self.assertIn("gamma", msg)
        self.assertIn("alpha", msg)
        self.assertIn("beta", msg)

    def test_missing_default_profile_rejected(self):
        path = self.write_json(
            json.dumps(
                {
                    "default_profile": "ghost",
                    "profiles": {
                        "alpha": {"data_root": "/d", "output_root": "/o"}
                    },
                }
            )
        )
        with self.assertRaises(ProfileError) as ctx:
            get_profile(None, path)
        self.assertIn("ghost", str(ctx.exception))

    def test_no_selection_available_raises(self):
        path = self.write_json(
            json.dumps(
                {"profiles": {"alpha": {"data_root": "/d", "output_root": "/o"}}}
            )
        )
        with self.assertRaises(ProfileError) as ctx:
            get_profile(None, path)
        self.assertIn("no profile selected", str(ctx.exception))

    def test_auto_detects_current_hostname(self):
        host = socket.gethostname()
        path = self.write_json(
            json.dumps(
                {
                    "default_profile": "wrong",
                    "profiles": {
                        "wrong": {"data_root": "/wrong", "output_root": "/wrong"},
                        "right": {
                            "data_root": "/right",
                            "output_root": "/right",
                            "auto": [host.lower()],
                        },
                    },
                }
            )
        )
        prof = get_profile(None, path)
        self.assertEqual(prof.name, "right")

    def test_auto_supports_wildcard_suffix(self):
        host = socket.gethostname()
        # Build a suffix rule that must match the real hostname.
        suffix = host[-4:].lower()
        path = self.write_json(
            json.dumps(
                {
                    "profiles": {
                        "wild": {
                            "data_root": "/wild",
                            "output_root": "/wild",
                            "auto": [f"*{suffix}"],
                        }
                    }
                }
            )
        )
        self.assertEqual(get_profile(None, path).name, "wild")

    def test_auto_beats_default_profile(self):
        host = socket.gethostname()
        path = self.write_json(
            json.dumps(
                {
                    "default_profile": "fallback",
                    "profiles": {
                        "fallback": {
                            "data_root": "/fallback",
                            "output_root": "/fallback",
                        },
                        "detected": {
                            "data_root": "/detected",
                            "output_root": "/detected",
                            "auto": host.lower(),
                        },
                    },
                }
            )
        )
        self.assertEqual(get_profile(None, path).name, "detected")

    def test_servers_file_via_environment(self):
        path = self.write_json()
        with mock.patch.dict(os.environ, {cfg.ENV_SERVERS_FILE: str(path)}):
            prof = get_profile("alpha")
        self.assertEqual(prof.name, "alpha")
        self.assertEqual(prof.source, str(path.resolve()))

    def test_env_servers_file_pointing_nowhere_is_reported(self):
        with mock.patch.dict(
            os.environ, {cfg.ENV_SERVERS_FILE: str(self.tmp / "nope.json")}
        ):
            with self.assertRaises(ProfileError) as ctx:
                get_profile()
            self.assertIn("missing file", str(ctx.exception))

    def test_shipped_config_file_is_valid(self):
        """configs/servers.yaml must parse and expose every profile."""
        root = Path(__file__).resolve().parent.parent
        found = cfg.find_servers_file(start=root)
        if not HAS_YAML:
            # Without PyYAML the loader is expected to say so and to point at
            # the dependency-free JSON alternative.
            with self.assertRaises(ProfileError) as ctx:
                cfg.load_servers_file(found)
            self.assertIn("PyYAML", str(ctx.exception))
            self.assertIn("servers.json", str(ctx.exception))
            return
        data = cfg.load_servers_file(found)
        self.assertEqual(sorted(data["profiles"]), ["local", "serverA", "serverB"])
        self.assertEqual(data["default_profile"], "local")

    def test_shipped_json_config_is_always_readable(self):
        """The dependency-free JSON mirror must work with no extra installs."""
        root = Path(__file__).resolve().parent.parent
        path = root / "configs" / "servers.json"
        self.assertTrue(path.is_file(), f"missing {path}")
        data = cfg.load_servers_file(path)
        self.assertEqual(sorted(data["profiles"]), ["local", "serverA", "serverB"])
        self.assertEqual(data["default_profile"], "local")

    def test_shipped_yaml_and_json_do_not_drift(self):
        """The two shipped config files must describe the same profiles.

        Both files exist so the interface works with or without PyYAML; this
        guards against editing one and forgetting the other.

        The JSON mirror is generated by tools/sync_servers_config.py, whose
        mini YAML reader is imported here so that the comparison does not
        itself depend on PyYAML.
        """
        root = Path(__file__).resolve().parent.parent
        sys.path.insert(0, str(root / "tools"))
        try:
            import sync_servers_config as sync
        finally:
            sys.path.pop(0)

        from_yaml = sync.parse_yaml_mapping(
            (root / "configs" / "servers.yaml").read_text(encoding="utf-8")
        )
        from_json = json.loads(
            (root / "configs" / "servers.json").read_text(encoding="utf-8")
        )
        self.assertEqual(from_yaml, from_json)

    def test_shipped_local_profile_is_usable(self):
        root = Path(__file__).resolve().parent.parent
        # Point at the JSON mirror so this works without PyYAML.
        prof = get_profile("local", root / "configs" / "servers.json")
        self.assertEqual(prof.kind, "local")
        self.assertTrue(prof.data_root)
        self.assertTrue(prof.output_root)
        self.assertIsInstance(prof.path("x"), Path)


class TestPathHandling(TempServersFile):
    def test_path_joins_onto_data_root(self):
        prof = get_profile("alpha", self.write_json())
        self.assertEqual(prof.path("bci42a", "subj003").as_posix(), "/data/alpha/bci42a/subj003")

    def test_path_with_no_parts_is_data_root(self):
        prof = get_profile("alpha", self.write_json())
        # as_posix() so the expectation is identical on Windows and Linux.
        self.assertEqual(prof.path().as_posix(), "/data/alpha")

    def test_output_path_joins_onto_output_root(self):
        prof = get_profile("alpha", self.write_json())
        self.assertEqual(
            prof.output_path("run1", "best.pt").as_posix(), "/out/alpha/run1/best.pt"
        )

    def test_tilde_and_env_vars_expanded(self):
        path = self.write_json(
            json.dumps(
                {
                    "profiles": {
                        "e": {
                            "data_root": "~/data/${MY_SUBDIR}",
                            "output_root": "$MY_OUT",
                        }
                    }
                }
            )
        )
        with mock.patch.dict(
            os.environ, {"MY_SUBDIR": "inner", "MY_OUT": "/expanded/out"}
        ):
            prof = get_profile("e", path)
        self.assertNotIn("~", prof.data_root)
        self.assertNotIn("${", prof.data_root)
        # Compare via pathlib so this holds on both / and \ separators.
        self.assertEqual(Path(prof.data_root).parts[-2:], ("data", "inner"))
        self.assertEqual(prof.output_root, "/expanded/out")

    def test_non_path_values_are_not_expanded(self):
        """A conda env named like a variable must survive verbatim."""
        path = self.write_json(
            json.dumps(
                {
                    "profiles": {
                        "e": {
                            "data_root": "/d",
                            "output_root": "/o",
                            "conda_env": "$NOT_A_PATH",
                        }
                    }
                }
            )
        )
        with mock.patch.dict(os.environ, {"NOT_A_PATH": "/should/not/appear"}):
            prof = get_profile("e", path)
        self.assertEqual(prof.conda_env, "$NOT_A_PATH")

    def test_overrides_have_highest_precedence(self):
        path = self.write_json()
        prof = get_profile("alpha", path, overrides={"data_root": "/override"})
        self.assertEqual(prof.data_root, "/override")

    def test_global_env_override(self):
        path = self.write_json()
        with mock.patch.dict(os.environ, {"T_DARTS_DATA_ROOT": "/from/env"}):
            prof = get_profile("alpha", path)
        self.assertEqual(prof.data_root, "/from/env")

    def test_scoped_env_override_only_applies_to_named_profile(self):
        path = self.write_json()
        with mock.patch.dict(os.environ, {"T_DARTS__beta__DATA_ROOT": "/only/beta"}):
            self.assertEqual(get_profile("beta", path).data_root, "/only/beta")
            self.assertEqual(get_profile("alpha", path).data_root, "/data/alpha")

    def test_extras_are_preserved(self):
        prof = get_profile("beta", self.write_json())
        self.assertEqual(prof.extras.get("scheduler"), "slurm")


class TestValidation(TempServersFile):
    def test_missing_data_root_rejected(self):
        path = self.write_json(
            json.dumps({"profiles": {"x": {"output_root": "/o"}}})
        )
        with self.assertRaises(ProfileError) as ctx:
            get_profile("x", path)
        self.assertIn("data_root", str(ctx.exception))

    def test_missing_output_root_rejected(self):
        path = self.write_json(
            json.dumps({"profiles": {"x": {"data_root": "/d"}}})
        )
        with self.assertRaises(ProfileError) as ctx:
            get_profile("x", path)
        self.assertIn("output_root", str(ctx.exception))

    def test_bad_kind_rejected(self):
        path = self.write_json(
            json.dumps(
                {
                    "profiles": {
                        "x": {
                            "data_root": "/d",
                            "output_root": "/o",
                            "kind": "cluster",
                        }
                    }
                }
            )
        )
        with self.assertRaises(ProfileError) as ctx:
            get_profile("x", path)
        self.assertIn("kind", str(ctx.exception))

    def test_remote_profile_rejects_windows_paths(self):
        """Catch a Windows path pasted into a remote profile immediately."""
        path = self.write_json(
            json.dumps(
                {
                    "profiles": {
                        "x": {
                            "kind": "remote",
                            "data_root": "D:/data",
                            "output_root": "/o",
                        }
                    }
                }
            )
        )
        with self.assertRaises(ProfileError) as ctx:
            get_profile("x", path)
        self.assertIn("POSIX", str(ctx.exception))

    def test_local_profile_accepts_windows_paths(self):
        path = self.write_json(
            json.dumps(
                {
                    "profiles": {
                        "x": {
                            "kind": "local",
                            "data_root": "D:/data",
                            "output_root": "D:/out",
                        }
                    }
                }
            )
        )
        self.assertEqual(get_profile("x", path).data_root, "D:/data")

    def test_require_data_root_reports_missing_directory(self):
        path = self.write_json(
            json.dumps(
                {
                    "profiles": {
                        "x": {
                            "data_root": str(self.tmp / "absent"),
                            "output_root": "/o",
                        }
                    }
                }
            )
        )
        prof = get_profile("x", path)
        with self.assertRaises(ProfileError) as ctx:
            prof.require_data_root()
        self.assertIn("does not exist", str(ctx.exception))

    def test_ensure_output_root_creates_directory(self):
        target = self.tmp / "nested" / "outputs"
        path = self.write_json(
            json.dumps(
                {
                    "profiles": {
                        "x": {
                            "data_root": "/d",
                            "output_root": str(target),
                        }
                    }
                }
            )
        )
        prof = get_profile("x", path)
        self.assertFalse(target.exists())
        self.assertEqual(prof.ensure_output_root(), target)
        self.assertTrue(target.is_dir())

    def test_require_data_root_succeeds_when_present(self):
        data = self.tmp / "data"
        data.mkdir()
        path = self.write_json(
            json.dumps(
                {
                    "profiles": {
                        "x": {
                            "data_root": str(data),
                            "output_root": "/o",
                        }
                    }
                }
            )
        )
        self.assertEqual(get_profile("x", path).require_data_root(), data)


class TestReporting(TempServersFile):
    def test_describe_contains_key_fields(self):
        text = cfg.describe("beta", self.write_json())
        for token in ("beta", "remote", "/gpfs/beta/data", "eegB"):
            self.assertIn(token, text)

    def test_describe_reports_errors_instead_of_raising(self):
        text = cfg.describe("ghost", self.write_json())
        self.assertTrue(text.startswith("<error>"))

    def test_describe_marks_provenance(self):
        path = self.write_json()
        with mock.patch.dict(os.environ, {cfg.ENV_PROFILE: "beta"}):
            text = cfg.describe(None, path)
        self.assertIn("T_DARTS_PROFILE", text)

    def test_to_dict_round_trips(self):
        prof = get_profile("beta", self.write_json())
        d = prof.to_dict()
        self.assertEqual(d["name"], "beta")
        self.assertEqual(d["extras"]["scheduler"], "slurm")


if __name__ == "__main__":
    unittest.main(verbosity=2)
