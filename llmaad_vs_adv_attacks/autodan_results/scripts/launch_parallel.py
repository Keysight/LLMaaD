#!/usr/bin/env python3
"""
launch_parallel.py — Python launcher for parallel AutoDAN scenario runs.
Replaces run_parallel_500_turbo.sh and run_parallel_500_reasoning.sh.

Features:
  - Spawns chunk sub-processes with full argument control
  - Live progress dashboard (refreshes every 10s)
  - Per-chunk last-line status pulled from logs
  - Auto-merge at the end via merge_turbo_results.py

Usage
-----
# AutoDAN-Turbo, S2+S3, 100 prompts (gpt-oss attacker, vicuna target — new default)
.llmaad/bin/python3 results_against_attacks/autodan_results/scripts/launch_parallel.py \
    --attack turbo --scenarios 2 3 --total 100

# AutoDAN-Reasoning, S3 only, 500 prompts
.llmaad/bin/python3 results_against_attacks/autodan_results/launch_parallel.py \
    --attack reasoning --scenarios 3 --total 500

# Both turbo + reasoning in one call (serial, one after the other)
.llmaad/bin/python3 results_against_attacks/autodan_results/launch_parallel.py \
    --attack all --scenarios 2 3 --total 100

# Dry run — print commands, don't execute
.llmaad/bin/python3 results_against_attacks/autodan_results/launch_parallel.py \
    --attack turbo --dry_run

# Status check on already-running chunks
.llmaad/bin/python3 results_against_attacks/autodan_results/launch_parallel.py --status
"""

import argparse, datetime, json, os, re, subprocess, sys, time
from pathlib import Path

REPO    = Path(__file__).resolve().parents[3]
PYTHON  = REPO / ".llmaad" / "bin" / "python3"
_AD     = REPO / "llmaad_vs_adv_attacks" / "autodan_results"
SCRIPTS = {
    "turbo":     _AD / "run_turbo_scenarios.py",
    "reasoning": _AD / "run_reasoning_scenarios.py",
}
MERGE   = _AD / "scripts" / "merge_turbo_results.py"
LOG_DIRS = {
    "turbo":     _AD / "turbo_scenarios" / "logs",
    "reasoning": _AD / "reasoning_scenarios" / "logs",
}
RUN_RESULTS = _AD / "run_results"
OUT_DIRS = {
    "turbo":     RUN_RESULTS,
    "reasoning": RUN_RESULTS,
}

# ANSI
GREEN  = "\033[92m"; RED   = "\033[91m"; YELLOW = "\033[93m"
CYAN   = "\033[96m"; BOLD  = "\033[1m";  RESET  = "\033[0m"; DIM = "\033[2m"

# ── Default model configs ──────────────────────────────────────────────────

DEFAULTS = {
    "attacker_model":  "gemma",
    "target_model":    "abliterated",
    "reshaping_model": "abliterated",
    "scorer_model":    "oss120b",
    "chunk_size":      50,
}
TURBO_DEFAULTS     = {"mutation": "use_strategy", "algo": "algo1q"}
REASONING_DEFAULTS = {"method": "vanilla",        "algo": "algo1",
                      "best_of_n_N": 4, "beam_W": 4, "beam_C": 3, "beam_K": 10}


# ── Helpers ────────────────────────────────────────────────────────────────

def _strip_ansi(text: str) -> str:
    return re.sub(r'\x1b\[[0-9;]*m', '', text)


def _last_meaningful_line(log_path: Path) -> str:
    """Return last non-empty, non-progress-bar line from a log file."""
    try:
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        for line in reversed(lines):
            clean = _strip_ansi(line).strip()
            if clean and "it/s]" not in clean and "Loading weights" not in clean:
                return clean[:90]
        return "(loading...)"
    except FileNotFoundError:
        return "(not started)"
    except Exception:
        return "(unreadable)"


def _count_completed(log_path: Path) -> tuple[int, int]:
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


def _read_progress_lock(log_path: Path) -> dict | None:
    """Read the progress lock file written by run_*_scenarios.py for this chunk."""
    m = re.search(r'_(\d+)_(\d+)\.log$', log_path.name)
    if not m:
        return None
    lock = log_path.parent / f"progress_{m.group(1)}_{m.group(2)}.json"
    try:
        return json.loads(lock.read_text())
    except Exception:
        return None


def _print_dashboard(chunks: list[dict], elapsed: float):
    print(f"\r\033[{len(chunks)+5}A", end="")  # move cursor up
    print(f"{BOLD}{'─'*70}{RESET}")
    print(f"  {BOLD}Chunk Status{RESET}  ({elapsed:.0f}s elapsed)")
    print(f"{'─'*70}")
    for c in chunks:
        alive = c["proc"].poll() is None if c["proc"] else False
        done, total = _count_completed(c["log"])
        lock  = _read_progress_lock(c["log"])
        if lock:
            cur  = lock.get("current", done)
            tot  = lock.get("total", total) or total
            pid  = lock.get("prompt_id", "?")
            goal = lock.get("goal", "")[:60]
            scen = lock.get("scenario", "")
            step = lock.get("step", "")
            tag  = f"{scen}:{step}" if step else scen
            pct  = f"{cur}/{tot} ►{tag}"
            goal_line = f"#{pid} \"{goal}\""
        else:
            pct  = f"{done}/{total}" if total else "0/?"
            goal_line = _last_meaningful_line(c["log"])
        if c["proc"] and c["proc"].done_ok:
            icon = f"{GREEN}✓{RESET}"
        elif c["proc"] and c["proc"]._rc == 1:
            icon = f"{RED}✗{RESET}"
        elif alive:
            icon = f"{GREEN}▶{RESET}"
        else:
            icon = f"{RED}✗{RESET}"
        print(f"  {icon} {c['label']:22}  [{pct:>14}]  {DIM}{goal_line}{RESET}")
    print(f"{'─'*70}")
    print(f"  Logs: tail -f {chunks[0]['log'].parent}/*.log")


# ── Chunk builder ──────────────────────────────────────────────────────────

def build_chunks(attack: str, total: int, chunk_size: int, scenarios: list[str],
                 attacker: str, target: str, reshaper: str, scorer: str,
                 mutation: str = None, method: str = None,
                 algo: str = "algo1", max_epochs: int = 0,
                 llamaguard_ip: str = os.getenv("LLAMAGUARD_IP", "localhost"),
                 llamaguard_port: int = int(os.getenv("LLAMAGUARD_PORT", 8001)),
                 start_offset: int = 0, **kwargs) -> list[dict]:
    script  = SCRIPTS[attack]
    ts      = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_tag = f"{attack}_{mutation or method or 'all'}_{ts}"
    log_dir = LOG_DIRS[attack] / run_tag
    log_dir.mkdir(parents=True, exist_ok=True)

    prefix = "turbo" if attack == "turbo" else "reasoning"
    chunks = []
    end_offset = start_offset + total
    start  = start_offset
    while start < end_offset:
        end = min(start + chunk_size, end_offset)
        log = log_dir / f"{prefix}_chunk_{start}_{end}.log"

        cmd = [str(PYTHON), str(script),
               "--attacker_model",  attacker,
               "--target_model",    target,
               "--reshaping_model", reshaper,
               "--scorer_model",    scorer,
               "--scenarios"] + scenarios + [
               "--algo", algo,
               "--range", str(start), str(end)]

        if attack == "turbo" and mutation:
            cmd += ["--mutation", mutation]
        if attack == "turbo" and max_epochs:
            cmd += ["--max_epochs", str(max_epochs)]
        if attack == "turbo":
            cmd += ["--llamaguard_ip", llamaguard_ip, "--llamaguard_port", str(llamaguard_port)]
            cmd += ["--lock_dir", str(log_dir)]
        if attack == "reasoning" and method:
            cmd += ["--method", method]
            for k, v in kwargs.items():
                cmd += [f"--{k}", str(v)]

        chunks.append({
            "label": f"{prefix} [{start}–{end}]",
            "cmd":   cmd,
            "log":   log,
            "proc":  None,
        })
        start = end
    return chunks


# ── Runner ─────────────────────────────────────────────────────────────────

def run_chunks(chunks: list[dict], dry_run: bool = False, n_parallel: int = 0):
    if dry_run:
        print(f"\n{BOLD}[dry_run] Commands that would run:{RESET}")
        for c in chunks:
            print(f"\n  # {c['label']}")
            print(f"  {' '.join(c['cmd'])}")
            print(f"  # → {c['log']}")
        return True

    # Determine worker count: default = len(chunks) so all chunks run in parallel
    workers = n_parallel if n_parallel > 0 else len(chunks)

    # Write a commands file for xargs — one wrapper script call per line.
    # Format: <log_path> <cmd...>
    # The inner bash snippet redirects stdout+stderr to <log_path>.
    log_dir = chunks[0]["log"].parent
    cmds_file = log_dir / "_xargs_cmds.txt"
    with open(cmds_file, "w") as f:
        for c in chunks:
            quoted_log = str(c["log"])
            # Each xargs line: LOG ARG1 ARG2 ... — handled by the wrapper below
            line = " ".join([quoted_log] + [str(a) for a in c["cmd"]])
            f.write(line + "\n")

    print(f"\n{BOLD}Launching {len(chunks)} chunk(s) via xargs -P {workers}...{RESET}")
    print(f"  Commands file: {cmds_file}")
    for c in chunks:
        print(f"  ▶ {c['label']}  →  {c['log'].name}")

    # xargs reads each line: first token = log path, rest = python command.
    # bash -c extracts log ($1) then runs the remainder ($2 $3 ...) redirected.
    # [DONE]/[FAIL] markers are written to BOTH xargs stdout AND the log file so
    # _LogWatcher.refresh() can detect completion by reading the log.
    xargs_cmd = [
        "xargs", f"-P{workers}", "-L", "1",
        "bash", "-c",
        'log=$1; shift; "$@" > "$log" 2>&1; rc=$?; '
        'if [ "$rc" = "0" ]; then echo "[DONE] $log"; echo "[DONE]" >> "$log"; '
        'else echo "[FAIL] $log"; echo "[FAIL]" >> "$log"; fi; exit $rc',
        "_",
    ]

    print(f"\n  xargs workers: {workers}  |  Logs: tail -f {log_dir}/*.log\n")

    # Print blank lines so dashboard cursor math works
    print("\n" * (len(chunks) + 5))

    t0 = time.time()
    try:
        xargs_proc = subprocess.Popen(
            xargs_cmd,
            stdin=open(cmds_file),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        # Fake proc objects so _print_dashboard can poll them
        # xargs handles parallelism; we just monitor logs
        for c in chunks:
            c["proc"] = _LogWatcher(c["log"])

        while xargs_proc.poll() is None:
            # Drain any xargs output lines ([DONE]/[FAIL] echoes)
            for line in _nonblocking_readlines(xargs_proc.stdout):
                print(f"  {line.rstrip()}")
            for c in chunks:
                c["proc"].refresh()
            _print_dashboard(chunks, time.time() - t0)
            time.sleep(10)

        # Final drain + dashboard — must refresh watchers AFTER draining stdout
        # so chunks that wrote [DONE] just before xargs exited are not missed.
        for line in xargs_proc.stdout:
            print(f"  {line.rstrip()}")
        for c in chunks:
            c["proc"].refresh()
        _print_dashboard(chunks, time.time() - t0)

    except KeyboardInterrupt:
        print(f"\n{YELLOW}Interrupted — killing xargs...{RESET}")
        xargs_proc.terminate()
        return False

    rc = xargs_proc.returncode
    failed_logs = [c for c in chunks if not c["proc"].done_ok]
    print(f"\n{'─'*70}")
    if rc != 0 or failed_logs:
        print(f"  {RED}{len(failed_logs)} chunk(s) appear to have failed.{RESET}")
        for c in failed_logs:
            print(f"    {c['label']}")
    else:
        print(f"  {GREEN}All {len(chunks)} chunk(s) completed successfully.{RESET}")
    return rc == 0 and not failed_logs


class _LogWatcher:
    """Thin wrapper that mimics subprocess poll() for dashboard compat."""
    def __init__(self, log_path: Path):
        self.log_path  = log_path
        self.done_ok   = False
        self._rc: int | None = None

    def refresh(self):
        try:
            text = self.log_path.read_text(encoding="utf-8", errors="replace")
            if "[DONE]" in text or "Saved →" in text or "Summary Table" in text:
                self.done_ok = True
                self._rc = 0
            elif "[FAIL]" in text:
                self._rc = 1
        except Exception:
            pass

    def poll(self) -> int | None:
        return self._rc


def _nonblocking_readlines(pipe):
    """Read all available lines from a pipe without blocking."""
    import select
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


# ── Status-only mode ───────────────────────────────────────────────────────

def show_status():
    print(f"\n{BOLD}Live Chunk Status{RESET}\n")
    found = False
    for attack in ["turbo", "reasoning"]:
        log_dir = LOG_DIRS[attack]
        logs    = sorted(log_dir.rglob("*.log")) if log_dir.exists() else []
        if not logs:
            continue
        found = True
        print(f"  {BOLD}{attack.upper()}{RESET}")
        for log in logs:
            done, total = _count_completed(log)
            lock = _read_progress_lock(log)
            if lock:
                cur  = lock.get("current", done)
                tot  = lock.get("total", total) or total
                pid  = lock.get("prompt_id", "?")
                goal = lock.get("goal", "")[:60]
                scen = lock.get("scenario", "")
                mut  = lock.get("mutation", "")
                step = lock.get("step", "")
                tag  = f"{scen}:{step}" if step else scen
                pct  = f"{cur}/{tot} ►{tag}"
                last = f"#{pid} [{mut}] \"{goal}\""
            else:
                pct  = f"{done}/{total}" if total else "0/?"
                last = _last_meaningful_line(log)
            print(f"    {log.name:35}  [{pct:>14}]  {DIM}{last}{RESET}")
        print()
    if not found:
        print(f"  {YELLOW}No log files found yet.{RESET}")


# ── Auto-merge ─────────────────────────────────────────────────────────────

def _write_run_log(attack: str, args, chunks: list[dict]):
    """Write a JSON run manifest to the attack output directory at start of every run."""
    out_dir = OUT_DIRS[attack]
    out_dir.mkdir(parents=True, exist_ok=True)
    ts  = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log = {
        "ts":         ts,
        "attack":     attack,
        "scenarios":  args.scenarios,
        "total":      args.total,
        "chunk_size": args.chunk_size,
        "n_chunks":   len(chunks),
        "mutation":   getattr(args, "mutation", None),
        "method":     getattr(args, "method",   None),
        "algo":       args.algo,
        "models": {
            "attacker": args.attacker,
            "target":   args.target,
            "reshaper": args.reshaper,
        },
        "chunks": [
            {"label": c["label"], "log": c["log"].name}
            for c in chunks
        ],
    }
    path = out_dir / f"run_log_{attack}_{ts}.json"
    path.write_text(json.dumps(log, indent=2), encoding="utf-8")
    print(f"  Run log → {path.relative_to(Path(__file__).resolve().parents[3])}")


def auto_merge(attack: str, tag: str = "", out_prefix: str = ""):
    out_dir = OUT_DIRS[attack]
    if not out_prefix:
        out_prefix = f"{attack}_merged"
    tag_display = tag or "all"
    print(f"\n  Merging {attack} results → {out_prefix}  (tag={tag_display!r})...")
    cmd = [str(PYTHON), str(MERGE),
           "--in_dir",  str(out_dir),
           "--out_dir", str(out_dir),
           "--out_prefix", out_prefix]
    if tag:
        cmd += ["--tag", tag]
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(f"  {RED}Merge error:{RESET} {result.stderr[:200]}")


# ── Argument parser ─────────────────────────────────────────────────────────

def build_parser():
    p = argparse.ArgumentParser(description="Parallel AutoDAN scenario launcher")

    p.add_argument("--attack", choices=["turbo", "reasoning", "all"], default="turbo",
                   help="Which attack script to launch (default: turbo)")
    p.add_argument("--total",      type=int, default=None, help="Total prompts to run (required unless --range/--status)")
    p.add_argument("--range",      type=int, nargs=2, metavar=("START", "END"), default=None,
                   help="Prompt index range, e.g. --range 20 50. Overrides --total and sets start offset.")
    p.add_argument("--chunk_size", type=int, default=50,   help="Prompts per chunk (default: 50)")
    p.add_argument("--scenarios",  nargs="+", default=["2", "3"],
                   help="Scenarios to run: 2, 3, or both (default: 2 3)")

    # Model config
    p.add_argument("--attacker",  default=DEFAULTS["attacker_model"])
    p.add_argument("--target",    default=DEFAULTS["target_model"])
    p.add_argument("--reshaper",  default=DEFAULTS["reshaping_model"])
    p.add_argument("--scorer",    default=DEFAULTS["scorer_model"],
                   help="Attacker judge/scorer model (default: gemma)")

    # Turbo-specific
    p.add_argument("--mutation",  default=TURBO_DEFAULTS["mutation"],
                   help="Turbo mutation strategy (default: use_strategy)")

    # Reasoning-specific
    p.add_argument("--method",    default=REASONING_DEFAULTS["method"],
                   help="Reasoning attack method (default: vanilla)")
    p.add_argument("--best_of_n_N", type=int, default=REASONING_DEFAULTS["best_of_n_N"])
    p.add_argument("--beam_W",      type=int, default=REASONING_DEFAULTS["beam_W"])
    p.add_argument("--beam_C",      type=int, default=REASONING_DEFAULTS["beam_C"])
    p.add_argument("--beam_K",      type=int, default=REASONING_DEFAULTS["beam_K"])

    # Shared
    p.add_argument("--algo",     default="algo1")
    p.add_argument("--max_epochs", type=int, default=0,
                   help="AutoDAN-Turbo S3 epochs per prompt (0 = use script default=20)")
    p.add_argument("--llamaguard_ip",   default=os.getenv("LLAMAGUARD_IP", "localhost"),
                   help="LlamaGuard vLLM server IP (env: LLAMAGUARD_IP, default: localhost)")
    p.add_argument("--llamaguard_port", type=int, default=int(os.getenv("LLAMAGUARD_PORT", 8001)),
                   help="LlamaGuard vLLM server port (env: LLAMAGUARD_PORT, default: 8001)")
    p.add_argument("--parallel", type=int, default=0,
                   help="xargs -P worker count (default: min(chunks, 4))")
    p.add_argument("--dry_run",  action="store_true")
    p.add_argument("--status",   action="store_true", help="Show live log status and exit")
    p.add_argument("--no_merge", action="store_true", help="Skip auto-merge after completion")
    return p


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    p    = build_parser()
    args = p.parse_args()

    if args.status:
        show_status()
        return

    range_start = 0
    if args.range is not None:
        range_start, range_end = args.range
        args.total = range_end - range_start
    elif args.total is None:
        p.error("--total or --range is required (e.g. --total 50 or --range 20 50).")

    attacks = ["turbo", "reasoning"] if args.attack == "all" else [args.attack]

    for attack in attacks:
        print(f"\n{BOLD}{'═'*70}")
        print(f"  AutoDAN-{attack.capitalize()} | S{'+S'.join(args.scenarios)} | "
              f"{args.total} prompts × {args.chunk_size}/chunk")
        print(f"  Attacker: {args.attacker}  Target: {args.target}  Reshaper: {args.reshaper}  Scorer: {args.scorer}")
        print(f"{'═'*70}{RESET}")

        extra = {}
        if attack == "reasoning":
            extra = dict(best_of_n_N=args.best_of_n_N, beam_W=args.beam_W,
                         beam_C=args.beam_C, beam_K=args.beam_K)

        chunks = build_chunks(
            attack=attack, total=args.total, chunk_size=args.chunk_size,
            scenarios=args.scenarios, attacker=args.attacker,
            target=args.target, reshaper=args.reshaper, scorer=args.scorer, algo=args.algo,
            mutation=args.mutation, method=args.method,
            max_epochs=args.max_epochs,
            llamaguard_ip=args.llamaguard_ip, llamaguard_port=args.llamaguard_port,
            start_offset=range_start,
            **extra,
        )

        if not args.dry_run:
            _write_run_log(attack, args, chunks)
        ok = run_chunks(chunks, dry_run=args.dry_run, n_parallel=args.parallel)

        if ok and not args.dry_run and not args.no_merge:
            scen_tag   = "S" + "S".join(sorted(args.scenarios))
            mut_tag    = args.method if attack == "reasoning" else args.mutation
            merge_tag  = f"{attack}_{mut_tag}_{scen_tag}_{args.algo}"
            out_prefix = f"{merge_tag}_n{args.total}"
            auto_merge(attack, tag=merge_tag, out_prefix=out_prefix)


if __name__ == "__main__":
    main()
