"""Mirror a GitHub repository into a local directory, byte for byte.

Used to vendor the official FBNAS baseline into ``FBNAS/`` so that the
baseline can be diffed against and audited offline.

By default only text-like files are fetched, and very large files are skipped.
Pass ``--all`` to fetch every blob in the tree, which is what the baseline
mirror uses: it keeps ``.pth`` network-initialisation checkpoints and the
pycache ``.pyc`` files that ship in the upstream repo, so that "the complete
official source is present" is verifiable rather than asserted.

Usage::

    python tools/fetch_upstream.py <owner/repo> <ref> <dest_dir> [--all]
    python tools/fetch_upstream.py wang1239435478/FBNAS-master main FBNAS --all

A ``_MANIFEST.json`` is written recording every blob path, size and git SHA so
the mirror can be checked later (see ``tools/verify_baseline.py``).
"""

from __future__ import annotations

import base64
import json
import os
import sys
import time
import urllib.error
import urllib.request

TEXT_EXT = {
    ".py", ".txt", ".md", ".yaml", ".yml", ".json", ".cfg", ".toml", ".ini",
    ".sh", ".bat", ".ps1", ".rst", ".tex", ".bib", ".csv", ".gitignore",
}

#: Text files larger than this are skipped unless --all is given.
MAX_BYTES = 3_000_000

#: Hard ceiling applied even under --all, so a stray huge blob cannot blow up
#: the repository. The largest upstream blob is ~70 KB.
ABSOLUTE_MAX_BYTES = 50_000_000


def _github_token() -> str | None:
    """Reuse the stored GitHub credential if one is available."""
    try:
        import keyring

        cred = keyring.get_credential("git:https://github.com", None)
        if cred:
            return cred.password
    except Exception:
        pass
    return None


def gh(url: str, token: str | None = None):
    headers = {
        "User-Agent": "dsh-research",
        "Accept": "application/vnd.github+json",
    }
    if token:
        headers["Authorization"] = "Bearer " + token
    req = urllib.request.Request(url, headers=headers)
    last: Exception | None = None
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.load(r)
        except urllib.error.HTTPError as exc:
            last = exc
            if exc.code in (403, 429) and attempt < 3:
                time.sleep(2 * (attempt + 1))
                continue
            raise
        except Exception as exc:  # transient network problem
            last = exc
            if attempt < 3:
                time.sleep(2 * (attempt + 1))
                continue
            raise
    raise RuntimeError(f"unreachable: {last}")


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    fetch_all = "--all" in args
    args = [a for a in args if a != "--all"]
    if len(args) != 3:
        print(__doc__)
        return 2

    slug, ref, dest = args
    token = _github_token()

    tree = gh(f"https://api.github.com/repos/{slug}/git/trees/{ref}?recursive=1", token)
    if tree.get("truncated"):
        print("WARNING: GitHub truncated the tree listing; mirror may be incomplete")

    blobs = [n for n in tree["tree"] if n["type"] == "blob"]
    print(f"repo          : {slug}@{ref}")
    print(f"tree sha      : {tree['sha']}")
    print(f"total blobs   : {len(blobs)}")
    print(f"mode          : {'complete (--all)' if fetch_all else 'text only'}")

    os.makedirs(dest, exist_ok=True)
    saved, skipped, failed = 0, [], []

    for n in blobs:
        path, size = n["path"], n.get("size", 0)
        ext = os.path.splitext(path)[1].lower()

        if size > ABSOLUTE_MAX_BYTES:
            skipped.append((path, size, "exceeds absolute size ceiling"))
            continue
        if not fetch_all and (ext not in TEXT_EXT or size > MAX_BYTES):
            skipped.append((path, size, "binary or large; use --all to include"))
            continue

        target = os.path.join(dest, path.replace("/", os.sep))
        os.makedirs(os.path.dirname(target), exist_ok=True)
        try:
            blob = gh(
                f"https://api.github.com/repos/{slug}/git/blobs/{n['sha']}", token
            )
            raw = base64.b64decode(blob["content"])
            with open(target, "wb") as f:
                f.write(raw)
            if len(raw) != size:
                failed.append((path, f"size mismatch: wrote {len(raw)}, index says {size}"))
                continue
            saved += 1
        except Exception as exc:
            failed.append((path, repr(exc)[:140]))

    manifest = {
        "repo": slug,
        "ref": ref,
        "tree_sha": tree["sha"],
        "complete": fetch_all and not skipped and not failed,
        "total_blobs": len(blobs),
        "saved": saved,
        "skipped": [{"path": p, "size": s, "reason": r} for p, s, r in skipped],
        "failed": [{"path": p, "error": e} for p, e in failed],
        "blobs": [
            {"path": n["path"], "size": n.get("size"), "sha": n["sha"]}
            for n in blobs
        ],
    }
    with open(os.path.join(dest, "_MANIFEST.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"saved={saved} skipped={len(skipped)} failed={len(failed)}")
    print(f"complete={manifest['complete']}")
    for p, s, r in skipped:
        print(f"  skipped: {p} ({s} bytes) - {r}")
    for p, e in failed:
        print(f"  FAILED : {p} - {e}")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
