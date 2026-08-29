#!/usr/bin/env python3
"""
merge_chunks.py — Merge 5 turbo 50-100 chunk JSONs into one archival file.

Usage
-----
  .llmaad/bin/python3 results_against_attacks/autodan_results/scripts/merge_chunks.py
"""

import json, re
from collections import defaultdict
from pathlib import Path

RUN_RESULTS = Path(__file__).resolve().parents[1] / "run_results"
CHUNK_RE    = re.compile(r"(turbo|reasoning)_(\w+)_S2S3_(\w+)_r(\d+)-(\d+)_\d+_\d+\.json$")
OUT_NAME    = "turbo_use_strategy_S2S3_algo1_n50-100.json"


def main():
    chunks = []
    for f in RUN_RESULTS.rglob("*.json"):
        m = CHUNK_RE.search(f.name)
        if m:
            attack, mutation, algo, start, end = m.groups()
            if attack == "turbo" and int(start) >= 50:
                chunks.append((f, int(start)))

    chunks.sort(key=lambda x: x[1])
    print(f"Found {len(chunks)} chunks for turbo 50-100:")
    for f, off in chunks:
        print(f"  offset={off}  {f.name}")

    buckets = defaultdict(dict)
    meta = {}
    for fpath, offset in chunks:
        data = json.loads(fpath.read_text(encoding="utf-8"))
        if not meta:
            meta = {k: v for k, v in data.items() if k != "results"}
        for sc_key, block in data.get("results", {}).items():
            rows = block if isinstance(block, list) else block.get("results", [])
            for row in rows:
                idx = (row.get("index", 0) or 0) + offset
                row["prompt_id"] = idx
                buckets[sc_key][idx] = row

    merged_results = {
        k: [v for _, v in sorted(d.items())]
        for k, d in sorted(buckets.items())
    }

    out = {**meta, "results": merged_results}
    out_path = RUN_RESULTS / OUT_NAME
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    for sc, rows in merged_results.items():
        print(f"  {sc}: {len(rows)} prompts")
    print(f"\n✓ Written → {out_path}")


if __name__ == "__main__":
    main()
