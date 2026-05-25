#!/usr/bin/env python3
"""Convert claude-judged result JSONs to CSV — one row per prompt (best attempt only).

Supports two formats (auto-detected by iteration structure):
  - GPTFuzz: flat iterations with seed_prompt / mutation_technique / final_prompt_to_target
  - PAIR:    iterations with streams, attacker_prompt / attacker_improvement

Row selection:
  - case == 'jailbroken':  winning iteration/stream (attack bypassed defense)
  - case == 'misdirected': winning misdirected iteration/stream
  - else (defended/other): iteration/stream with highest judge score

Usage:
  python json_to_csv.py <input.json> [output.csv]
  python json_to_csv.py  # walks current directory
"""
import json, csv, os, sys


GPTFUZZ_COLS = [
    "index", "question",
    "seed_prompt", "mutation_technique", "final_prompt_to_target",
    "llamaguard_decision", "jailbroken", "target_responses",
    "case", "total_iterations",
]

PAIR_COLS = [
    "index", "question",
    "attacker_prompt", "attacker_improvement",
    "llamaguard_decision", "jailbroken", "target_responses",
    "case", "total_iterations",
]


def is_gptfuzz(data):
    for r in data.get("results", []):
        for it in r.get("iterations", []):
            return "streams" not in it
    return False


# ── GPTFuzz helpers ───────────────────────────────────────────────────────────

def gptfuzz_pick_iteration(result):
    case = result.get("case", "")

    if case == "jailbroken":
        for it in result["iterations"]:
            if it.get("jailbroken") and not it.get("misdirected"):
                return it

    if case == "misdirected":
        best = None
        for it in result["iterations"]:
            if it.get("misdirected") and it.get("jailbroken"):
                best = it
        if best:
            return best
        # fallback: any misdirected iteration
        for it in result["iterations"]:
            if it.get("misdirected"):
                return it

    # Fallback: highest judge score
    best_it, best_score = result["iterations"][0], -1
    for it in result["iterations"]:
        score = max(it.get("judge_results") or [0])
        if score > best_score:
            best_score = score
            best_it = it
    return best_it


def gptfuzz_row(result):
    case = result.get("case", "")
    it = gptfuzz_pick_iteration(result)

    trs = it.get("target_responses") or []
    if case == "misdirected":
        response = it.get("misdirected_response") or (trs[0] if trs else "")
    else:
        response = trs[0] if trs else ""

    return {
        "index":                result["index"],
        "question":             result["question"],
        "seed_prompt":          it.get("seed_prompt", ""),
        "mutation_technique":   it.get("mutation_technique", ""),
        "final_prompt_to_target": it.get("final_prompt_to_target", ""),
        "llamaguard_decision":  it.get("llamaguard_decision", ""),
        "jailbroken":           result.get("jailbroken", False),
        "target_responses":     response,
        "case":                 case,
        "total_iterations":     result.get("total_iterations", len(result["iterations"])),
    }


# ── PAIR helpers ──────────────────────────────────────────────────────────────

def pair_pick_stream(result):
    oi = result.get("outcome_iteration")
    os_ = result.get("outcome_stream")

    # Use the outcome stream when there is one (jailbroken / misdirected)
    if oi is not None and os_ is not None:
        for it in result["iterations"]:
            if it["iteration"] == oi:
                for s in it["streams"]:
                    if str(s["stream"]) == str(os_):
                        return s

    # Fallback: highest judge score across all streams
    best_s, best_score = None, -1
    for it in result["iterations"]:
        for s in it["streams"]:
            score = s.get("judge_score") or 0
            if score > best_score:
                best_score = score
                best_s = s
    return best_s or result["iterations"][0]["streams"][0]


def pair_row(result):
    case = result.get("case", "")
    s = pair_pick_stream(result)

    if case == "misdirected":
        response = s.get("final_response") or s.get("target_response", "")
    else:
        response = s.get("target_response", "")

    return {
        "index":              result["index"],
        "question":           result.get("goal", result.get("question", "")),
        "attacker_prompt":    s.get("attacker_prompt", ""),
        "attacker_improvement": s.get("attacker_improvement", ""),
        "llamaguard_decision": s.get("llamaguard_decision", ""),
        "jailbroken":         result.get("outcome", "defended") != "defended",
        "target_responses":   response,
        "case":               case,
        "total_iterations":   len(result["iterations"]),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def convert_file(json_path, csv_path=None):
    with open(json_path) as f:
        data = json.load(f)

    if not data.get("results"):
        print(f"  Skipped (no results): {json_path}")
        return

    gptfuzz = is_gptfuzz(data)
    cols = GPTFUZZ_COLS if gptfuzz else PAIR_COLS
    make_row = gptfuzz_row if gptfuzz else pair_row

    rows = [make_row(r) for r in data["results"]]

    if csv_path is None:
        csv_path = json_path.rsplit(".", 1)[0] + ".csv"

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    from collections import Counter
    cases = Counter(r["case"] for r in rows)
    fmt = "GPTFuzz" if gptfuzz else "PAIR"
    print(f"[{fmt}] {len(rows)} rows → {os.path.basename(csv_path)} | {dict(cases)}")
    return csv_path


if __name__ == "__main__":
    if len(sys.argv) >= 2:
        inp = sys.argv[1]
        out = sys.argv[2] if len(sys.argv) >= 3 else None
        convert_file(inp, out)
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        converted = []
        for root, dirs, files in os.walk(script_dir):
            for fname in sorted(files):
                if fname.endswith(".json"):
                    converted.append(convert_file(os.path.join(root, fname)))
        print(f"\nDone: {len(converted)} files converted.")
