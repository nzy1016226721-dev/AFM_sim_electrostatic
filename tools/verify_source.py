"""Verify the source-only payload against its SHA-256 manifest (standard library).

Maintainers may use --write when preparing a reviewed candidate. Preserve a
dated historical manifest first. This is integrity checking, not numerical proof.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = "docs/SOURCE_MANIFEST.json"
EXCLUDED_DIRS = {".git", "outputs", "output", "archive_old", "__pycache__",
                 ".pytest_cache", ".numba_cache", ".venv", "venv", "node_modules"}


def records(root):
    result = []
    for here, dirs, names in os.walk(root):
        dirs[:] = sorted(name for name in dirs if name not in EXCLUDED_DIRS
                         and not name.startswith(("output_", ".afm_")))
        for name in dirs + names:
            path = Path(here) / name
            if path.is_symlink() or getattr(path, "is_junction", lambda: False)():
                raise ValueError("Source payload contains a link: " + str(path))
        for name in sorted(names):
            path = Path(here) / name
            relative = path.relative_to(root).as_posix()
            if relative == MANIFEST or path.suffix in {".pyc", ".pyo"}:
                continue
            data = path.read_bytes()
            result.append({"path": relative, "bytes": len(data),
                           "sha256": hashlib.sha256(data).hexdigest()})
    return sorted(result, key=lambda row: row["path"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="regenerate the candidate manifest")
    parser.add_argument("--base", default="8c4819d6f2ac2797774303502a740a50dd9858c0")
    args = parser.parse_args()
    path = ROOT / MANIFEST
    actual = records(ROOT)
    if args.write:
        value = {"format": "afm-standalone-source-manifest-v4", "snapshot_date": "2026-10-26",
                 "base_git_commit": args.base, "scope": "Standalone source-only unpublished upload candidate",
                 "self_excluded": MANIFEST, "files": actual}
        path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8", newline="\n")
        print(json.dumps({"written": MANIFEST, "files": len(actual)}))
        return 0
    expected = json.loads(path.read_text(encoding="utf-8"))["files"]
    expected = {row["path"]: (row["bytes"], row["sha256"]) for row in expected}
    actual = {row["path"]: (row["bytes"], row["sha256"]) for row in actual}
    differences = sorted(name for name in expected.keys() | actual.keys()
                         if expected.get(name) != actual.get(name))
    print(json.dumps({"passed": not differences, "files_checked": len(actual),
                      "manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                      "differences": differences}, indent=2))
    return int(bool(differences))


if __name__ == "__main__":
    raise SystemExit(main())
