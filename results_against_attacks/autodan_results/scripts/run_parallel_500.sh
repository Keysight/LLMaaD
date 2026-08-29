#!/usr/bin/env bash
# run_parallel_500.sh — Run turbo scenarios on 500 prompts using 5 parallel range chunks
#
# Strategy:
#   Split 500 prompts into 5 chunks of 100.
#   Run each chunk in its own background process (or tmux pane).
#   Each chunk runs all 3 mutations × requested scenarios.
#   All chunks write to turbo_scenarios/ with a timestamped filename.
#   After all chunks finish, run merge_turbo_results.py to combine + export CSV.
#
# Usage:
#   # S3 only (CMPE misdirect), all mutations
#   bash results_against_attacks/autodan_results/run_parallel_500.sh --scenarios 3
#
#   # S2 + S3, all mutations
#   bash results_against_attacks/autodan_results/run_parallel_500.sh --scenarios "2 3"
#
#   # Dry run: print commands without executing
#   bash results_against_attacks/autodan_results/run_parallel_500.sh --dry_run
#
# Logs:
#   Each chunk logs to turbo_scenarios/logs/chunk_<start>_<end>.log

set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
SCRIPT="$REPO/results_against_attacks/autodan_results/run_turbo_scenarios.py"
MERGE="$REPO/results_against_attacks/autodan_results/merge_turbo_results.py"
LOG_DIR="$REPO/results_against_attacks/autodan_results/turbo_scenarios/logs"
PYTHON="$REPO/.llmaad/bin/python3"

# ── Defaults ────────────────────────────────────────────────────────────────
ATTACKER_MODEL="abliterated"
TARGET_MODEL="abliterated"
RESHAPING_MODEL="abliterated"
SCENARIOS="3"          # "2", "3", or "2 3"
ALGO="algo1"
MUTATION="all"
CHUNK_SIZE=100
TOTAL=500
DRY_RUN=false

# ── Arg parsing ──────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --scenarios)    SCENARIOS="$2";         shift 2 ;;
        --algo)         ALGO="$2";              shift 2 ;;
        --mutation)     MUTATION="$2";          shift 2 ;;
        --chunk_size)   CHUNK_SIZE="$2";        shift 2 ;;
        --total)        TOTAL="$2";             shift 2 ;;
        --attacker)     ATTACKER_MODEL="$2";    shift 2 ;;
        --target)       TARGET_MODEL="$2";      shift 2 ;;
        --reshaper)     RESHAPING_MODEL="$2";   shift 2 ;;
        --dry_run)      DRY_RUN=true;           shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

mkdir -p "$LOG_DIR"

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  AutoDAN-Turbo Parallel 500-Prompt Run"
echo "═══════════════════════════════════════════════════════════════"
echo "  Attacker : $ATTACKER_MODEL"
echo "  Target   : $TARGET_MODEL"
echo "  Reshaper : $RESHAPING_MODEL"
echo "  Scenarios: $SCENARIOS"
echo "  Mutations: $MUTATION"
echo "  Algo     : $ALGO"
echo "  Total    : $TOTAL prompts in chunks of $CHUNK_SIZE"
echo "  Dry run  : $DRY_RUN"
echo "───────────────────────────────────────────────────────────────"

PIDS=()

start=0
while (( start < TOTAL )); do
    end=$(( start + CHUNK_SIZE ))
    (( end > TOTAL )) && end=$TOTAL

    LOG="$LOG_DIR/chunk_${start}_${end}.log"

    CMD="$PYTHON $SCRIPT \
        --attacker_model $ATTACKER_MODEL \
        --target_model   $TARGET_MODEL \
        --reshaping_model $RESHAPING_MODEL \
        --mutation $MUTATION \
        --scenarios $SCENARIOS \
        --algo $ALGO \
        --range $start $end"

    echo "  [chunk $start–$end]  → $LOG"

    if [[ "$DRY_RUN" == "false" ]]; then
        # Run in background, redirect stdout+stderr to log
        eval "$CMD" > "$LOG" 2>&1 &
        PIDS+=($!)
    else
        echo "    CMD: $CMD"
    fi

    start=$end
done

if [[ "$DRY_RUN" == "true" ]]; then
    echo ""
    echo "  [dry_run] No processes started."
    exit 0
fi

echo ""
echo "  Started ${#PIDS[@]} parallel chunk(s). PIDs: ${PIDS[*]}"
echo "  Waiting for all to finish..."
echo "  (Follow logs: tail -f $LOG_DIR/chunk_*.log)"
echo ""

# Wait for all chunks and collect exit codes
FAILED=0
for pid in "${PIDS[@]}"; do
    if ! wait "$pid"; then
        echo "  [WARN] PID $pid exited with error"
        FAILED=$(( FAILED + 1 ))
    fi
done

echo "═══════════════════════════════════════════════════════════════"
if (( FAILED > 0 )); then
    echo "  DONE with $FAILED failed chunk(s). Check logs in $LOG_DIR"
else
    echo "  ALL CHUNKS COMPLETE."
fi
echo "═══════════════════════════════════════════════════════════════"

# ── Auto-merge ────────────────────────────────────────────────────────────
echo ""
echo "  Merging results..."
"$PYTHON" "$MERGE" --out_prefix "turbo_abliterated_500"
