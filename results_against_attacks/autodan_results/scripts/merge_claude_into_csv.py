#!/usr/bin/env python3
"""
merge_claude_into_csv.py — Inject claude_judge column from a *_claude_judged.json
into the corresponding best-epoch CSVs.

Usage
-----
  .llmaad/bin/python3 results_against_attacks/autodan_results/scripts/merge_claude_into_csv.py \
      results_against_attacks/autodan_results/run_results/turbo_use_strategy_S2S3_algo1_n50_claude_judged.json

  .llmaad/bin/python3 results_against_attacks/autodan_results/scripts/merge_claude_into_csv.py \
      results_against_attacks/autodan_results/run_results/reasoning_vanilla_S2S3_algo1_n50_claude_judged.json
"""

import csv, json, re, sys
from pathlib import Path

RUN_RESULTS = Path(__file__).resolve().parents[1] / "run_results"

# Maps scenario key suffix → CSV suffix to look for
SCENARIO_TO_CSV_SUFFIX = {
    "S2": "S2_best_epoch",
    "S3": "S3_best_epoch",
}


def scenario_tag(key):
    if "S2" in key:
        return "S2"
    if "S3" in key:
        return "S3"
    return None


def find_csv(attack, mutation, tag, n_range):
    """
    Look for a CSV matching: {attack}_{mutation}_{tag}_best_epoch_{n_range}p.csv
    Falls back to any matching attack+mutation+tag CSV if range not found.
    """
    patterns = [
        f"{attack}_{mutation}_{tag}_best_epoch_{n_range}p.csv",
        f"{attack}_{mutation}_{tag}_best_epoch_50p.csv",
    ]
    for pat in patterns:
        p = RUN_RESULTS / pat
        if p.exists():
            return p
    # glob fallback
    candidates = sorted(RUN_RESULTS.glob(f"{attack}_{mutation}_{tag}_best_epoch*.csv"))
    return candidates[0] if candidates else None


def load_claude_cases(judged_json_path):
    """Returns dict: scenario_key → {index: claude_case}"""
    data = json.loads(Path(judged_json_path).read_text(encoding="utf-8"))
    out = {}
    for sc_key, rows in data.get("results", {}).items():
        mapping = {}
        for r in rows:
            idx = r.get("prompt_id", r.get("index"))
            if idx is not None:
                mapping[int(idx)] = r.get("claude_case", "")
        out[sc_key] = mapping
    return out


def inject_column(csv_path, idx_to_case, col_name="claude_judge"):
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fieldnames = reader.fieldnames or []
        for row in reader:
            rows.append(row)

    if col_name in fieldnames:
        new_fields = fieldnames  # update in place
    else:
        new_fields = list(fieldnames) + [col_name]

    updated = 0
    for row in rows:
        idx = int(row.get("index", -1))
        row[col_name] = idx_to_case.get(idx, "")
        if row[col_name]:
            updated += 1

    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=new_fields)
        writer.writeheader()
        writer.writerows(rows)

    print(f"  ✓ {csv_path.name}  ({updated}/{len(rows)} rows filled)")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    judged_path = Path(sys.argv[1])
    if not judged_path.exists():
        print(f"ERROR: {judged_path} not found")
        sys.exit(1)

    # Parse attack + mutation from filename
    # e.g. turbo_use_strategy_S2S3_algo1_n50_claude_judged.json
    fname = judged_path.name
    m = re.match(r"(turbo|reasoning)_(\w+)_S2S3_\w+_n(\d+(?:-\d+)?)_claude_judged\.json", fname)
    if not m:
        print(f"ERROR: cannot parse attack/mutation from filename: {fname}")
        sys.exit(1)

    attack, mutation, n_range = m.groups()

    print(f"\nFile : {fname}")
    print(f"Attack: {attack} | Mutation: {mutation} | Range: 0-{n_range}\n")

    cases_by_scenario = load_claude_cases(judged_path)

    for sc_key, idx_to_case in cases_by_scenario.items():
        tag = scenario_tag(sc_key)
        if not tag:
            print(f"  SKIP {sc_key} (unrecognised scenario tag)")
            continue

        csv_path = find_csv(attack, mutation, tag, n_range)
        if not csv_path:
            print(f"  SKIP {sc_key} — no matching CSV found for {attack}_{mutation}_{tag}")
            continue

        inject_column(csv_path, idx_to_case)

    print("\nDone.")


if __name__ == "__main__":
    main()
