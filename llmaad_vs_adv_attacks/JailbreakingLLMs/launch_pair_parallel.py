#!/usr/bin/env python3
"""
launch_pair_parallel.py — xargs-based parallel launcher for PAIR scenario runs.

Splits the prompt dataset into chunks and runs them as separate processes via
xargs -P, mirroring the pattern used by autodan_results/scripts/launch_parallel.py.

Supported modes
---------------
  block             run_pair_detect_block_parallel.py            (standard judge)
  block-hardened    run_pair_detect_block_parallel_hardened_judge.py
  misdirect         run_pair_detect_misdirect_parallel.py        (standard judge)
  misdirect-hardened run_pair_detect_misdirect_parallel_hardened_judge.py

Usage
-----
# detect-block, hardened judge, 50 prompts, 25 per chunk (2 processes)
.llmaad/bin/python3 llmaad_vs_adv_attacks/JailbreakingLLMs/launch_pair_parallel.py \\
    --mode block-hardened --total 50 --chunk-size 25

# detect-misdirect, standard judge, 50 prompts
.llmaad/bin/python3 llmaad_vs_adv_attacks/JailbreakingLLMs/launch_pair_parallel.py \\
    --mode misdirect --total 50

# Dry run — print commands without executing
.llmaad/bin/python3 llmaad_vs_adv_attacks/JailbreakingLLMs/launch_pair_parallel.py \\
    --mode block-hardened --total 50 --dry-run
"""

import argparse
import datetime
import json
import os
import re
import select
import subprocess
import sys
import time
from pathlib import Path

HERE    = Path(__file__).resolve().parent
REPO    = HERE.parents[1]
# The PAIR scripts require the .llmenv environment (has pandas, requests, litellm).
# jailbreakbench is not installed — always use --not-jailbreakbench + the custom CSV.
PYTHON  = REPO / ".llmaad" / "bin" / "python3"
DATASET = HERE / "data" / "harmful_behaviors_custom.csv"
LOG_DIR = HERE / "logs"

SCRIPTS = {
    "block":              HERE / "run_pair_detect_block_parallel.py",
    "block-hardened":     HERE / "run_pair_detect_block_parallel_hardened_judge.py",
    "misdirect":          HERE / "run_pair_detect_misdirect_parallel.py",
    "misdirect-hardened": HERE / "run_pair_detect_misdirect_parallel_hardened_judge.py",
}

OUT_SUBDIRS = {
    "block":              "llmaad_results/detect_and_block",
    "block-hardened":     "llmaad_results/detect_and_block",
    "misdirect":          "llmaad_results/detect_and_misdirect",
    "misdirect-hardened": "llmaad_results/detect_and_misdirect",
}

OUT_STEMS = {
    "block":              "pair_llamaguard_block_vicuna",
    "block-hardened":     "pair_llamaguard_block_vicuna_hardened_judge",
    "misdirect":          "pair_llamaguard_misdirect_vicuna",
    "misdirect-hardened": "pair_llamaguard_misdirect_vicuna_hardened_judge",
}

GREEN = "\033[92m"; RED = "\033[91m"; YELLOW = "\033[93m"
CYAN  = "\033[96m"; BOLD = "\033[1m"; RESET  = "\033[0m"; DIM = "\033[2m"


# ── Helpers ────────────────────────────────────────────────────────────────

def _strip_ansi(text: str) -> str:
    return re.sub(r'\x1b\[[0-9;]*m', '', text)


def _last_line(log_path: Path) -> str:
    try:
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        for line in reversed(lines):
            clean = _strip_ansi(line).strip()
            if clean:
                return clean[:90]
        return "(loading...)"
    except FileNotFoundError:
        return "(not started)"
    except Exception:
        return "(unreadable)"


def _count_completed(log_path: Path) -> tuple:
    """Return (completed, total) from lines like [07/50]."""
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
        matches = re.findall(r'\[(\d+)/(\d+)\]', _strip_ansi(text))
        if matches:
            last = matches[-1]
            return int(last[0]), int(last[1])
    except Exception:
        pass
    return 0, 0


class _LogWatcher:
    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.done_ok  = False
        self._rc      = None

    def refresh(self):
        try:
            text = self.log_path.read_text(encoding="utf-8", errors="replace")
            if "[DONE]" in text:
                self.done_ok = True
                self._rc = 0
            elif "[FAIL]" in text:
                self._rc = 1
        except Exception:
            pass

    def poll(self):
        return self._rc


def _nonblocking_readlines(pipe):
    lines = []
    while True:
        ready, _, _ = select.select([pipe], [], [], 0)
        if not ready:
            break
        line = pipe.readline()
        if not line:
            break
        lines.append(line)
    return lines


def _print_dashboard(chunks: list, elapsed: float):
    print(f"\r\033[{len(chunks) + 4}A", end="")
    print(f"{BOLD}{'─'*70}{RESET}")
    print(f"  {BOLD}PAIR Chunk Status{RESET}  ({elapsed:.0f}s elapsed)")
    print(f"{'─'*70}")
    for c in chunks:
        done, total = _count_completed(c["log"])
        pct = f"{done}/{total}" if total else "0/?"
        last = _last_line(c["log"])
        if c["watcher"].done_ok:
            icon = f"{GREEN}✓{RESET}"
        elif c["watcher"].poll() == 1:
            icon = f"{RED}✗{RESET}"
        else:
            icon = f"{GREEN}▶{RESET}"
        print(f"  {icon} {c['label']:28}  [{pct:>10}]  {DIM}{last}{RESET}")
    print(f"{'─'*70}")
    print(f"  Logs: tail -f {chunks[0]['log'].parent}/*.log")


# ── Chunk builder ──────────────────────────────────────────────────────────

def build_chunks(mode: str, total: int, chunk_size: int,
                 start_index: int, ts: str,
                 extra_args: list) -> list:
    script   = SCRIPTS[mode]
    out_sub  = OUT_SUBDIRS[mode]
    stem     = OUT_STEMS[mode]
    log_dir  = LOG_DIR / f"pair_{mode}_{ts}"
    log_dir.mkdir(parents=True, exist_ok=True)

    chunks = []
    start  = start_index
    end_of = start_index + total
    while start < end_of:
        end  = min(start + chunk_size, end_of)
        size = end - start
        label = f"{mode} [{start}–{end}]"
        log   = log_dir / f"pair_{mode}_{start}_{end}.log"

        out_dir = HERE / out_sub
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"{stem}_{size}p_chunk_{start}_{end}.json"

        cmd = [
            str(PYTHON), str(script),
            "--start-index", str(start),
            "--num-prompts", str(end),   # goals[:end] then skip i<start
            "--output", str(out_file),
            "--not-jailbreakbench",
            "--dataset", str(DATASET),
        ] + extra_args

        chunks.append({
            "label":   label,
            "cmd":     cmd,
            "log":     log,
            "out":     out_file,
            "watcher": _LogWatcher(log),
        })
        start = end
    return chunks


# ── Runner ─────────────────────────────────────────────────────────────────

def run_chunks(chunks: list, dry_run: bool, n_parallel: int) -> bool:
    if dry_run:
        print(f"\n{BOLD}[dry-run] Commands that would run:{RESET}")
        for c in chunks:
            print(f"\n  # {c['label']}")
            print(f"  {' '.join(str(a) for a in c['cmd'])}")
            print(f"  # log → {c['log']}")
        return True

    workers   = n_parallel if n_parallel > 0 else len(chunks)
    cmds_file = chunks[0]["log"].parent / "_xargs_cmds.txt"
    with open(cmds_file, "w") as f:
        for c in chunks:
            line = " ".join([str(c["log"])] + [str(a) for a in c["cmd"]])
            f.write(line + "\n")

    print(f"\n{BOLD}Launching {len(chunks)} chunk(s) via xargs -P {workers}...{RESET}")
    for c in chunks:
        print(f"  ▶ {c['label']}  →  {c['log'].name}")

    xargs_cmd = [
        "xargs", f"-P{workers}", "-L", "1",
        "bash", "-c",
        'log=$1; shift; "$@" > "$log" 2>&1; rc=$?; '
        'if [ "$rc" = "0" ]; then echo "[DONE] $log"; echo "[DONE]" >> "$log"; '
        'else echo "[FAIL] $log"; echo "[FAIL]" >> "$log"; fi; exit $rc',
        "_",
    ]

    print(f"\n  Workers: {workers}  |  Logs: tail -f {chunks[0]['log'].parent}/*.log\n")
    print("\n" * (len(chunks) + 4))

    t0 = time.time()
    try:
        proc = subprocess.Popen(
            xargs_cmd,
            stdin=open(cmds_file),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        while proc.poll() is None:
            for line in _nonblocking_readlines(proc.stdout):
                print(f"  {line.rstrip()}")
            for c in chunks:
                c["watcher"].refresh()
            _print_dashboard(chunks, time.time() - t0)
            time.sleep(10)

        for line in proc.stdout:
            print(f"  {line.rstrip()}")
        for c in chunks:
            c["watcher"].refresh()
        _print_dashboard(chunks, time.time() - t0)

    except KeyboardInterrupt:
        print(f"\n{YELLOW}Interrupted — killing xargs...{RESET}")
        proc.terminate()
        return False

    rc          = proc.returncode
    failed      = [c for c in chunks if not c["watcher"].done_ok]
    print(f"\n{'─'*70}")
    if rc != 0 or failed:
        print(f"  {RED}{len(failed)} chunk(s) failed.{RESET}")
        for c in failed:
            print(f"    {c['label']}")
    else:
        print(f"  {GREEN}All {len(chunks)} chunk(s) completed.{RESET}")
    return rc == 0 and not failed


# ── Merge ──────────────────────────────────────────────────────────────────

def merge_chunks(chunks: list, mode: str, total: int, ts: str):
    """Combine per-chunk JSON output files into a single merged JSON."""
    stem    = OUT_STEMS[mode]
    out_dir = HERE / OUT_SUBDIRS[mode]
    merged_path = out_dir / f"{stem}_{total}p_merged_{ts}.json"

    all_results  = []
    counts       = {}
    total_prompts = 0

    for c in chunks:
        if not c["out"].exists():
            print(f"  {YELLOW}Missing chunk output: {c['out'].name}{RESET}")
            continue
        with open(c["out"]) as f:
            data = json.load(f)
        chunk_results = data.get("results", [])
        all_results.extend(chunk_results)
        total_prompts += data.get("completed", len(chunk_results))
        for k, v in data.get("counts", {}).items():
            counts[k] = counts.get(k, 0) + v

    summary = {
        "description":   f"PAIR {mode} — merged from {len(chunks)} chunks",
        "defense_mode":  mode,
        "total_prompts": total_prompts,
        "completed":     len(all_results),
        "counts":        counts,
        "merged_from":   [str(c["out"].name) for c in chunks],
        "results":       all_results,
    }

    with open(merged_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n  {GREEN}Merged → {merged_path.relative_to(REPO)}{RESET}")
    print(f"  Counts: {counts}")
    print(f"  Total:  {len(all_results)} results")
    return merged_path


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="PAIR parallel launcher — xargs chunk runner"
    )
    p.add_argument("--mode", required=True,
                   choices=list(SCRIPTS.keys()),
                   help="Which PAIR script variant to run")
    p.add_argument("--total",      type=int, required=True,
                   help="Total number of prompts to process")
    p.add_argument("--chunk-size", type=int, default=25,
                   help="Prompts per chunk / process (default: 25)")
    p.add_argument("--start-index", type=int, default=0,
                   help="Dataset start index (default: 0)")
    p.add_argument("--parallel",   type=int, default=0,
                   help="xargs -P worker count (default: number of chunks)")
    p.add_argument("--no-merge",   action="store_true",
                   help="Skip merging chunk outputs at the end")
    p.add_argument("--dry-run",    action="store_true",
                   help="Print commands without executing")

    # Pass-through args forwarded verbatim to child scripts
    p.add_argument("--n-streams",          type=int, default=None)
    p.add_argument("--n-iterations",       type=int, default=None)
    p.add_argument("--attack-model",       default=None)
    p.add_argument("--attack-ip",          default=None)
    p.add_argument("--attack-port",        type=int, default=None)
    p.add_argument("--target-model",       default=None)
    p.add_argument("--target-ip",          default=None)
    p.add_argument("--target-port",        type=int, default=None)
    p.add_argument("--judge-model",        default=None)
    p.add_argument("--judge-ip",           default=None)
    p.add_argument("--judge-port",         type=int, default=None)
    p.add_argument("--llamaguard-url",     default=None)
    p.add_argument("--llamaguard-model",   default=None)
    p.add_argument("--reshaper-model",     default=None)
    p.add_argument("--reshaper-ip",        default=None)
    p.add_argument("--reshaper-port",      type=int, default=None)

    args = p.parse_args()
    ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Build extra args list (only pass explicitly provided values through)
    extra = []
    for flag, val in [
        ("--n-streams",          args.n_streams),
        ("--n-iterations",       args.n_iterations),
        ("--attack-model",       args.attack_model),
        ("--attack-ip",          args.attack_ip),
        ("--attack-port",        args.attack_port),
        ("--target-model",       args.target_model),
        ("--target-ip",          args.target_ip),
        ("--target-port",        args.target_port),
        ("--judge-model",        args.judge_model),
        ("--judge-ip",           args.judge_ip),
        ("--judge-port",         args.judge_port),
        ("--llamaguard-url",     args.llamaguard_url),
        ("--llamaguard-model",   args.llamaguard_model),
        ("--reshaper-model",     args.reshaper_model),
        ("--reshaper-ip",        args.reshaper_ip),
        ("--reshaper-port",      args.reshaper_port),
    ]:
        if val is not None:
            extra += [flag, str(val)]

    print(f"\n{BOLD}{'═'*70}")
    print(f"  PAIR launcher  |  mode: {args.mode}  |  {args.total} prompts × {args.chunk_size}/chunk")
    print(f"  Script: {SCRIPTS[args.mode].name}")
    print(f"{'═'*70}{RESET}")

    chunks = build_chunks(
        mode=args.mode, total=args.total, chunk_size=args.chunk_size,
        start_index=args.start_index, ts=ts, extra_args=extra,
    )

    ok = run_chunks(chunks, dry_run=args.dry_run, n_parallel=args.parallel)

    if ok and not args.dry_run and not args.no_merge:
        merge_chunks(chunks, mode=args.mode, total=args.total, ts=ts)


if __name__ == "__main__":
    main()
