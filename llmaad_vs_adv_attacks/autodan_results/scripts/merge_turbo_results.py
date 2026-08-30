#!/usr/bin/env python3
"""
merge_turbo_results.py — Merge chunked turbo/reasoning scenario JSONs into ordered CSV + summary JSON

When --tag contains "turbo" or "reasoning", dirs and output prefix are auto-routed:
  turbo    → in: run_results/autodan_turbo/chunks/   out: run_results/autodan_turbo/turbo_merged.*
  reasoning→ in: run_results/autodan_reasoning/chunks/ out: run_results/autodan_reasoning/reasoning_merged.*

Usage
-----
# Merge turbo chunks (auto-routes to autodan_turbo/)
.llmaad/bin/python3 results_against_attacks/autodan_results/scripts/merge_turbo_results.py --tag turbo

# Merge reasoning chunks
.llmaad/bin/python3 results_against_attacks/autodan_results/scripts/merge_turbo_results.py --tag reasoning

# Dry-run: show what would be merged without writing
.llmaad/bin/python3 results_against_attacks/autodan_results/scripts/merge_turbo_results.py --tag turbo --dry_run

# Override output dir/prefix explicitly
.llmaad/bin/python3 results_against_attacks/autodan_results/scripts/merge_turbo_results.py \
    --tag turbo --out_prefix turbo_use_strategy_S2S3_algo1_n100
"""

import argparse, csv, json, sys
from pathlib import Path
from collections import defaultdict

REPO        = Path(__file__).resolve().parents[3]
RUN_RESULTS = REPO / "results_against_attacks" / "autodan_results" / "run_results"
IN_DIR      = RUN_RESULTS
OUT_DIR     = RUN_RESULTS  # overridden per-attack below in main()

GREEN = "\033[92m"; RED = "\033[91m"; CYAN = "\033[96m"
BOLD  = "\033[1m";  RESET = "\033[0m"

# Canonical column order for CSV
CSV_COLS = [
    "prompt_id", "index", "goal", "mutation", "scenario",
    "jailbreak_prompt",
    # S2 columns
    "raw_response", "final_response", "toxicity_score", "was_blocked",
    # S3 columns
    "misdirected_output", "nlp_toxicity", "actual_asr",
    # shared judge (TurboScorer only)
    "turbo_score", "turbo_jailbroken", "elapsed_sec",
    "attacker_refused", "error",
]


def load_chunk_files(in_dir: Path, tag: str = "") -> list[Path]:
    files = sorted(in_dir.rglob("*.json"))
    # exclude run manifests and already-merged outputs
    files = [f for f in files if not f.name.startswith("run_log_")]
    if tag:
        files = [f for f in files if tag in f.name]
    return files


def merge(files: list[Path]) -> dict:
    """
    Returns {mutation|scenario: [sorted rows]} keeping index order across chunks.
    Duplicate (index, mutation, scenario) entries: last-write wins.
    """
    # key → {index: row}
    buckets: dict[str, dict[int, dict]] = defaultdict(dict)
    meta_seen = {}

    for f in files:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"  {RED}[skip]{RESET} {f.name}  — {e}")
            continue

        for key, block in data.get("results", {}).items():
            # chunk files: block = {"results": [...], "metrics": {...}}
            # merged/summary files: block is already a list
            rows_iter = block if isinstance(block, list) else block.get("results", [])
            for row in rows_iter:
                # prefer prompt_id (global CSV row) over index (local chunk counter)
                idx = row.get("prompt_id", row.get("index"))
                if idx is None:
                    continue
                # tag each row with which file it came from
                row["_src"] = f.name
                buckets[key][idx] = row

        # keep args from first file seen
        if "args" not in meta_seen:
            meta_seen["args"] = data.get("args", {})
        meta_seen.setdefault("files", []).append(f.name)

    # Sort each bucket by index
    merged = {}
    for key, idx_map in sorted(buckets.items()):
        merged[key] = [idx_map[i] for i in sorted(idx_map)]

    return merged, meta_seen


def compute_metrics(rows: list[dict], scenario_key: str) -> dict:
    n = len(rows)
    if n == 0:
        return {}
    if "S2" in scenario_key:
        asr        = sum(r.get("turbo_jailbroken", False) for r in rows) / n
        block_rate = sum(r.get("was_blocked", False) for r in rows) / n
        return {"n": n, "asr_after_block": round(asr, 4), "block_rate": round(block_rate, 4)}
    if "S3" in scenario_key:
        turbo_win = sum(r.get("attack_succeeded", r.get("turbo_jailbroken", False)) for r in rows) / n
        act_asr   = sum(
            r.get("actual_asr", False) or
            any(e.get("actual_asr") for e in r.get("epochs", []))
            for r in rows
        ) / n
        tox_vals  = [r.get("avg_nlp_toxicity", r.get("nlp_toxicity", 0)) for r in rows]
        avg_tox   = sum(tox_vals) / len(tox_vals) if tox_vals else 0.0
        return {"n": n, "turbo_win_asr": round(turbo_win, 4), "actual_asr": round(act_asr, 4), "avg_tox": round(avg_tox, 6)}
    return {"n": n}


def write_csv(merged: dict, out_path: Path):
    rows_flat = []
    for key, rows in merged.items():
        mutation, scenario = key.split("|", 1)
        for r in rows:
            flat = {"mutation": mutation, "scenario": scenario}
            flat.update(r)
            rows_flat.append(flat)

    # Sort: by mutation, scenario, then index to preserve order
    MUTATION_ORDER = {"warm_up": 0, "use_strategy": 1, "find_new_strategy": 2}
    rows_flat.sort(key=lambda r: (
        MUTATION_ORDER.get(r.get("mutation", ""), 99),
        r.get("scenario", ""),
        r.get("index", 0),
    ))

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        # include all keys seen across rows
        all_keys = list(dict.fromkeys(
            CSV_COLS + [k for r in rows_flat for k in r if k not in CSV_COLS and not k.startswith("_")]
        ))
        writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows_flat)

    return len(rows_flat)


def write_summary_json(merged: dict, meta: dict, out_path: Path):
    summary_table = []
    for key, rows in merged.items():
        mutation, scenario = key.split("|", 1)
        metrics = compute_metrics(rows, scenario)
        summary_table.append({"mutation": mutation, "scenario": scenario, **metrics})

    payload = {
        "merge_args": meta.get("args", {}),
        "source_files": meta.get("files", []),
        "summary_table": summary_table,
        "results": {key: rows for key, rows in merged.items()},
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return summary_table


def print_summary(summary_table: list, merged: dict):
    print(f"\n{BOLD}{'─'*65}")
    print(f"  {'Key':40}  {'Metric':25}")
    print(f"{'─'*65}{RESET}")
    for row in summary_table:
        key = f"{row['mutation']} | {row['scenario']}"
        n   = row.get("n", "?")
        if "turbo_win_asr" in row:
            m = f"turbo_win={row['turbo_win_asr']:.0%}  actual_asr={row['actual_asr']:.0%}  tox={row['avg_tox']:.4f}  n={n}"
        elif "asr_after_block" in row:
            m = f"slip={row['asr_after_block']:.0%}  block={row['block_rate']:.0%}  n={n}"
        else:
            m = f"n={n}"
        print(f"  {key:40}  {m}")
    print(f"{BOLD}{'─'*65}{RESET}")


def main():
    p = argparse.ArgumentParser(description="Merge chunked turbo/reasoning scenario JSONs")
    p.add_argument("--tag",     default="", help="Filter source files by name substring; also auto-routes dirs (turbo/reasoning)")
    p.add_argument("--in_dir",  default=None, help="Source directory (auto from --tag if omitted)")
    p.add_argument("--out_dir", default=None, help="Output directory (auto from --tag if omitted)")
    p.add_argument("--out_prefix", default=None,
                   help="Output filename prefix (auto from --tag if omitted)")
    p.add_argument("--dry_run", action="store_true",
                   help="Print what would be merged without writing files")
    args = p.parse_args()

    tag = args.tag.lower()
    if "turbo" in tag:
        chunks_dir = RUN_RESULTS / "autodan_turbo" / "chunks"
        auto_in    = chunks_dir if chunks_dir.exists() else IN_DIR
        auto_out   = RUN_RESULTS / "autodan_turbo"
        auto_pfx   = "turbo_merged"
    elif "reasoning" in tag:
        chunks_dir = RUN_RESULTS / "autodan_reasoning" / "chunks"
        auto_in    = chunks_dir if chunks_dir.exists() else IN_DIR
        auto_out   = RUN_RESULTS / "autodan_reasoning"
        auto_pfx   = "reasoning_merged"
    else:
        auto_in    = IN_DIR
        auto_out   = OUT_DIR
        auto_pfx   = "turbo_merged"

    in_dir     = Path(args.in_dir)     if args.in_dir     else auto_in
    out_dir    = Path(args.out_dir)    if args.out_dir    else auto_out
    out_prefix = args.out_prefix       if args.out_prefix else auto_pfx

    print(f"\n  in_dir  : {in_dir}")
    print(f"  out_dir : {out_dir}")
    print(f"  prefix  : {out_prefix}")

    files = load_chunk_files(in_dir, args.tag)
    if not files:
        print(f"{RED}No JSON files found in {in_dir} (tag={args.tag!r}){RESET}")
        sys.exit(1)

    print(f"\n{BOLD}Merging {len(files)} file(s):{RESET}")
    for f in files:
        print(f"  {f.name}")

    merged, meta = merge(files)

    total_rows = sum(len(v) for v in merged.values())
    keys       = list(merged.keys())
    print(f"\n  Keys found: {keys}")
    print(f"  Total rows: {total_rows}")

    if args.dry_run:
        print(f"\n{CYAN}[dry_run] No files written.{RESET}")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path  = out_dir / f"{out_prefix}.csv"
    json_path = out_dir / f"{out_prefix}.json"

    n_rows   = write_csv(merged, csv_path)
    summary  = write_summary_json(merged, meta, json_path)

    print_summary(summary, merged)
    print(f"\n  {GREEN}CSV  → {csv_path}  ({n_rows} rows){RESET}")
    print(f"  {GREEN}JSON → {json_path}{RESET}\n")


if __name__ == "__main__":
    main()
