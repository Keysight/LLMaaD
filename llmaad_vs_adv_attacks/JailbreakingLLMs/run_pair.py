#!/usr/bin/env python3
"""
run_pair.py — Unified launcher for all PAIR attack modes.

Dispatches to one of five underlying scripts based on --mode and --judge:

  Mode              Judge       Script
  ──────────────    ─────────   ──────────────────────────────────────────────
  baseline          standard    run_pair_baseline_parallel.py
  detect-block      standard    run_pair_detect_block_parallel.py
  detect-block      hardened    run_pair_detect_block_parallel_hardened_judge.py
  detect-misdirect  standard    run_pair_detect_misdirect_parallel.py
  detect-misdirect  hardened    run_pair_detect_misdirect_parallel_hardened_judge.py

Usage
-----
# Detect-block, vicuna target, 50 prompts
python3 run_pair.py --mode detect-block --target-ip 10.36.129.2 --num-prompts 50

# Detect-misdirect with hardened judge, abliterated target
python3 run_pair.py --mode detect-misdirect --judge hardened \\
    --target-model mlabonne/NeuralDaredevil-8B-abliterated \\
    --target-ip 10.36.129.1 --reshaper-ip 10.36.129.1 \\
    --num-prompts 50

# Baseline (no defense), dry run
python3 run_pair.py --mode baseline --dry-run

# All modes in sequence for both targets
python3 run_pair.py --mode all --target-ip 10.36.129.2 --num-prompts 50
"""

import argparse, subprocess, sys, time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PYTHON     = sys.executable

SCRIPTS = {
    ("baseline",         "standard"): SCRIPT_DIR / "run_pair_baseline_parallel.py",
    ("detect-block",     "standard"): SCRIPT_DIR / "run_pair_detect_block_parallel.py",
    ("detect-block",     "hardened"): SCRIPT_DIR / "run_pair_detect_block_parallel_hardened_judge.py",
    ("detect-misdirect", "standard"): SCRIPT_DIR / "run_pair_detect_misdirect_parallel.py",
    ("detect-misdirect", "hardened"): SCRIPT_DIR / "run_pair_detect_misdirect_parallel_hardened_judge.py",
}

ALL_MODES = ["baseline", "detect-block", "detect-misdirect"]

GREEN = "\033[92m"; RED = "\033[91m"; BOLD = "\033[1m"; RESET = "\033[0m"


def build_cmd(mode: str, judge: str, args) -> list[str]:
    script = SCRIPTS[(mode, judge)]
    target_model = args.target_model

    # auto-derive a clean target name for output filename
    target_short = (
        "vicuna"      if "vicuna"      in target_model.lower() else
        "abliterated" if "abliterated" in target_model.lower() else
        "qwen3"       if "qwen3"       in target_model.lower() else
        target_model.split("/")[-1].lower()
    )
    judge_tag  = "_hardened" if judge == "hardened" else ""
    mode_short = mode.replace("detect-", "").replace("baseline", "baseline")
    out_dir    = args.out_dir or f"llmaad_results/{mode}"
    out_file   = (
        f"{out_dir}/pair_llamaguard_{mode_short}{judge_tag}_{target_short}"
        f"_{args.num_prompts}p.json"
    )

    cmd = [
        PYTHON, str(script),
        "--attack-model",        args.attack_model,
        "--attack-ip",           args.attack_ip,
        "--attack-port",         str(args.attack_port),
        "--target-model",        target_model,
        "--target-ip",           args.target_ip,
        "--target-port",         str(args.target_port),
        "--judge-model",         args.judge_model,
        "--judge-ip",            args.judge_ip,
        "--judge-port",          str(args.judge_port),
        "--parallel-workers",    str(args.parallel_workers),
        "--n-streams",           str(args.n_streams),
        "--n-iterations",        str(args.n_iterations),
        "--output",              out_file,
    ]

    # dataset / range args (not present on baseline)
    if mode != "baseline":
        cmd += [
            "--dataset",      args.dataset,
            "--num-prompts",  str(args.num_prompts),
            "--start-index",  str(args.start_index),
        ]

    # LlamaGuard (all non-baseline modes)
    if mode != "baseline":
        cmd += [
            "--llamaguard-model", args.llamaguard_model,
            "--llamaguard-url",   args.llamaguard_url,
        ]

    # Reshaper (misdirect modes only)
    if mode == "detect-misdirect":
        cmd += [
            "--reshaper-model", args.reshaper_model,
            "--reshaper-ip",    args.reshaper_ip,
            "--reshaper-port",  str(args.reshaper_port),
        ]

    if args.verbosity:
        cmd += ["-" + "v" * args.verbosity]

    return cmd


def run_mode(mode: str, judge: str, args) -> bool:
    script = SCRIPTS.get((mode, judge))
    if not script or not script.exists():
        print(f"  {RED}[skip] script not found for mode={mode} judge={judge}{RESET}")
        return False

    cmd = build_cmd(mode, judge, args)

    label = f"{mode}  [{judge} judge]"
    print(f"\n{BOLD}{'─'*65}")
    print(f"  PAIR | {label}")
    print(f"  Target:  {args.target_model} @ {args.target_ip}:{args.target_port}")
    print(f"  Attacker:{args.attack_model} @ {args.attack_ip}:{args.attack_port}")
    print(f"  Judge:   {args.judge_model}  @ {args.judge_ip}:{args.judge_port}")
    if mode != "baseline":
        print(f"  LlamaGuard: {args.llamaguard_url}")
    if mode == "detect-misdirect":
        print(f"  Reshaper: {args.reshaper_model} @ {args.reshaper_ip}:{args.reshaper_port}")
    print(f"{'─'*65}{RESET}")

    if args.dry_run:
        print(f"  [dry-run] {' '.join(cmd)}")
        return True

    t0 = time.time()
    result = subprocess.run(cmd)
    elapsed = time.time() - t0
    ok = result.returncode == 0
    status = f"{GREEN}DONE{RESET}" if ok else f"{RED}FAILED (rc={result.returncode}){RESET}"
    print(f"\n  {status}  {elapsed/60:.1f} min")
    return ok


def build_parser():
    p = argparse.ArgumentParser(
        description="Unified PAIR attack launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    p.add_argument("--mode", default="detect-block",
                   choices=["baseline", "detect-block", "detect-misdirect", "all"],
                   help="Attack/defense mode (default: detect-block)")
    p.add_argument("--judge", default="standard", choices=["standard", "hardened", "both"],
                   help="Judge variant: standard, hardened, or both (default: standard)")

    # Models
    p.add_argument("--attack-model",   default="mlabonne/NeuralDaredevil-8B-abliterated")
    p.add_argument("--attack-ip",      default="10.36.129.1")
    p.add_argument("--attack-port",    type=int, default=8000)
    p.add_argument("--target-model",   default="lmsys/vicuna-13b-v1.5")
    p.add_argument("--target-ip",      default="10.36.129.2")
    p.add_argument("--target-port",    type=int, default=8000)
    p.add_argument("--judge-model",    default="openai/gpt-oss-120b")
    p.add_argument("--judge-ip",       default="10.36.129.6")
    p.add_argument("--judge-port",     type=int, default=8000)
    p.add_argument("--reshaper-model", default="mlabonne/NeuralDaredevil-8B-abliterated")
    p.add_argument("--reshaper-ip",    default="10.36.129.1")
    p.add_argument("--reshaper-port",  type=int, default=8000)

    # LlamaGuard
    p.add_argument("--llamaguard-model", default="meta-llama/Llama-Guard-3-8B")
    p.add_argument("--llamaguard-url",   default="http://10.36.129.1:8001/v1",
                   help="LlamaGuard vLLM endpoint (default: http://10.36.129.1:8001/v1)")

    # Dataset / run config
    p.add_argument("--dataset",          default="data/harmful_behaviors_custom.csv")
    p.add_argument("--num-prompts",      type=int, default=50)
    p.add_argument("--start-index",      type=int, default=0)
    p.add_argument("--n-streams",        type=int, default=10)
    p.add_argument("--n-iterations",     type=int, default=5)
    p.add_argument("--parallel-workers", type=int, default=10)

    # Output
    p.add_argument("--out-dir", dest="out_dir", default=None,
                   help="Output directory (auto-derived from mode if not set)")

    p.add_argument("--dry-run",   action="store_true", help="Print commands without executing")
    p.add_argument("-v", "--verbosity", action="count", default=0)

    return p


def main():
    args = build_parser().parse_args()

    modes   = ALL_MODES if args.mode == "all" else [args.mode]
    judges  = (["standard", "hardened"] if args.judge == "both"
               else [args.judge])
    # baseline only has standard judge
    combos  = [
        (m, j) for m in modes for j in judges
        if not (m == "baseline" and j == "hardened")
    ]

    print(f"\n{BOLD}PAIR Launcher — {len(combos)} run(s){RESET}")
    print(f"  Target:  {args.target_model}")
    print(f"  Prompts: {args.num_prompts}  (start={args.start_index})")

    results = {}
    for mode, judge in combos:
        ok = run_mode(mode, judge, args)
        results[(mode, judge)] = ok

    print(f"\n{BOLD}{'─'*65}\n  Summary{RESET}")
    all_ok = True
    for (mode, judge), ok in results.items():
        status = f"{GREEN}OK{RESET}" if ok else f"{RED}FAILED{RESET}"
        print(f"  {mode:20}  [{judge:8}]  {status}")
        all_ok = all_ok and ok

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
