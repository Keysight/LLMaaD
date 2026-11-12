#!/usr/bin/env python3
"""Check ./adapter against PROVENANCE.json without loading a model.

Run this on the demo machine before the talk. It needs nothing but the Python
standard library, so it works on a laptop with no ML stack installed and tells
you in a few seconds whether the artifact survived the copy.

  python3 verify_adapter.py
  python3 verify_adapter.py --adapter /some/other/adapter --json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_digest(path: Path) -> str:
    """Hash a directory as sorted (relative path, file hash) pairs.

    Same algorithm as slm_lab.train_lora._tree_digest, so the result is directly
    comparable to the training run's manifest. A renamed file changes the digest
    even if every byte of content is identical -- that is intended.
    """
    digest = hashlib.sha256()
    for file_path in sorted(item for item in path.rglob("*") if item.is_file()):
        digest.update(file_path.relative_to(path).as_posix().encode())
        digest.update(sha256_file(file_path).encode())
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", type=Path, default=HERE / "adapter")
    parser.add_argument("--provenance", type=Path, default=HERE / "PROVENANCE.json")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    args = parser.parse_args()

    if not args.adapter.is_dir():
        print(f"adapter directory not found: {args.adapter}", file=sys.stderr)
        return 2
    expected = json.loads(args.provenance.read_text(encoding="utf-8"))["adapter_sha256"]
    actual = tree_digest(args.adapter)
    files = sorted(p for p in args.adapter.rglob("*") if p.is_file())
    ok = expected == actual

    if args.json:
        print(json.dumps({"ok": ok, "expected": expected, "actual": actual,
                          "files": len(files)}, indent=2))
    else:
        for path in files:
            print(f"  {sha256_file(path)[:16]}  {path.relative_to(args.adapter)}")
        print(f"\n  tree digest  {actual}")
        print(f"  expected     {expected}")
        print("\n  OK — this is the trained artifact." if ok else
              "\n  MISMATCH — this is NOT the trained artifact. Do not demo it.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
