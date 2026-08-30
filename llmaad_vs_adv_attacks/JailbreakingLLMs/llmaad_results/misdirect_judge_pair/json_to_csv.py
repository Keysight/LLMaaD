"""
json_to_csv.py  —  Convert misdirect_judge_pair JSON results to CSV files.

Usage:
    python json_to_csv.py              # generates all CSVs in this directory
    python json_to_csv.py --input <file.json>   # single file
"""

import argparse
import csv
import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
JB_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../.."))   # JailbreakingLLMs/

# ── Source JSON files ──────────────────────────────────────────────────────────
SOURCES = {
    "misdirect": os.path.join(
        JB_ROOT,
        "harmbench_results/llamaguard/pair_llamaguard_misdirect_gpt4atk_vicuna_50p.json",
    ),
    "block": os.path.join(
        JB_ROOT,
        "harmbench_results/llamaguard/pair_llamaguard_block_vicuna_50p.json",
    ),
}

# ── Output CSV filenames ───────────────────────────────────────────────────────
OUTPUTS = {
    "misdirect": os.path.join(SCRIPT_DIR, "pair_llamaguard_misdirect_vicuna_50p.csv"),
    "block":     os.path.join(SCRIPT_DIR, "pair_llamaguard_block_vicuna_50p.csv"),
    "summary":   os.path.join(SCRIPT_DIR, "pair_llamaguard_summary.csv"),
}


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def misdirect_to_csv(data, out_path):
    """Per-prompt CSV for the misdirect run."""
    fieldnames = [
        "index", "goal", "category",
        "outcome", "outcome_iteration", "outcome_stream",
        "total_misdirected", "total_passed", "misdirect_failed",
        "elapsed_sec",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in data["results"]:
            writer.writerow({
                "index":              r["index"] + 1,
                "goal":               r["goal"],
                "category":           r.get("category", "unknown"),
                "outcome":            r["outcome"],
                "outcome_iteration":  r.get("outcome_iteration", ""),
                "outcome_stream":     r.get("outcome_stream", ""),
                "total_misdirected":  r.get("total_misdirected", 0),
                "total_passed":       r.get("total_passed", 0),
                "misdirect_failed":   r.get("misdirect_failed", 0),
                "elapsed_sec":        r.get("elapsed_sec", ""),
            })
    print(f"  Written: {out_path}")


def block_to_csv(data, out_path):
    """Per-prompt CSV for the block run."""
    fieldnames = [
        "index", "goal", "category",
        "outcome", "outcome_iteration", "outcome_stream",
        "elapsed_sec",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in data["results"]:
            writer.writerow({
                "index":             r["index"] + 1,
                "goal":              r["goal"],
                "category":         r.get("category", "unknown"),
                "outcome":          r["outcome"],
                "outcome_iteration": r.get("outcome_iteration", ""),
                "outcome_stream":    r.get("outcome_stream", ""),
                "elapsed_sec":      r.get("elapsed_sec", ""),
            })
    print(f"  Written: {out_path}")


def summary_to_csv(misdirect_data, block_data, out_path):
    """Aggregated 2-row summary CSV comparing misdirect vs block."""
    def asr(counts, total):
        jb = counts.get("misdirected_jailbreak", 0) + counts.get("jailbroken", 0) + counts.get("attack_jailbreak", 0)
        return round(jb / total * 100, 1) if total else 0.0

    rows = []
    for label, data in [("LLMAAD misdirect (algo1q)", misdirect_data), ("LlamaGuard block", block_data)]:
        counts = data.get("counts", {})
        total  = data.get("total_prompts", len(data.get("results", [])))
        jb     = (
            counts.get("misdirected_jailbreak", 0)
            + counts.get("jailbroken", 0)
            + counts.get("attack_jailbreak", 0)
        )
        defended = counts.get("defended", 0)
        rows.append({
            "defense_mode":           label,
            "total_prompts":          total,
            "jailbroken":             jb,
            "defended":               defended,
            "asr_pct":                asr(counts, total),
            "defense_success_pct":    round(defended / total * 100, 1) if total else 0.0,
            "judge":                  data["config"].get("judge_model", ""),
            "llamaguard_model":       data["config"].get("llamaguard_model", ""),
            "attack_model":           data["config"].get("attack_model", ""),
            "target_model":           data["config"].get("target_model", ""),
            "total_misdirected_calls": data.get("total_misdirected", ""),
        })

    fieldnames = [
        "defense_mode", "total_prompts", "jailbroken", "defended",
        "asr_pct", "defense_success_pct",
        "judge", "llamaguard_model", "attack_model", "target_model",
        "total_misdirected_calls",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Written: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert misdirect_judge_pair JSONs to CSV")
    parser.add_argument("--input", default=None, help="Single JSON input file (auto-detects mode)")
    args = parser.parse_args()

    if args.input:
        with open(args.input) as f:
            data = json.load(f)
        mode = data.get("defense_mode", "unknown")
        out = os.path.join(SCRIPT_DIR, os.path.basename(args.input).replace(".json", ".csv"))
        if mode == "detect-misdirect":
            misdirect_to_csv(data, out)
        else:
            block_to_csv(data, out)
        return

    # Default: generate all
    print("Generating CSVs for misdirect_judge_pair...")

    misdirect_data = load_json(SOURCES["misdirect"])
    block_data     = load_json(SOURCES["block"])

    misdirect_to_csv(misdirect_data, OUTPUTS["misdirect"])
    block_to_csv(block_data,         OUTPUTS["block"])
    summary_to_csv(misdirect_data, block_data, OUTPUTS["summary"])

    print("\nDone.")


if __name__ == "__main__":
    main()
