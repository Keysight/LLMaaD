#!/usr/bin/env bash
# run_parallel_500_turbo.sh — AutoDAN-Turbo × S2+S3 parallel run via xargs -P
#
# xargs -P spawns N workers simultaneously, each handling one chunk.
# Faster and simpler than manual PID tracking.
#
# Usage:
#   bash results_against_attacks/autodan_results/run_parallel_500_turbo.sh
#   bash results_against_attacks/autodan_results/run_parallel_500_turbo.sh --total 500 --chunk 100
#   bash results_against_attacks/autodan_results/run_parallel_500_turbo.sh --dry_run

set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
SCRIPT="$REPO/results_against_attacks/autodan_results/run_turbo_scenarios.py"
MERGE="$REPO/results_against_attacks/autodan_results/merge_turbo_results.py"
LOG_DIR="$REPO/results_against_attacks/autodan_results/turbo_scenarios/logs"
PYTHON="$REPO/.llmaad/bin/python3"

# ── Config ───────────────────────────────────────────────────────────────────
ATTACKER_MODEL="vicuna"
TARGET_MODEL="vicuna"
RESHAPING_MODEL="abliterated"
MUTATION="use_strategy"
ALGO="algo1"
SCENARIOS="2 3"
TOTAL=100
CHUNK=50
PARALLEL=4        # xargs -P workers (set ≤ vLLM server concurrency)
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --total)    TOTAL="$2";    shift 2 ;;
        --chunk)    CHUNK="$2";    shift 2 ;;
        --parallel) PARALLEL="$2"; shift 2 ;;
        --mutation) MUTATION="$2"; shift 2 ;;
        --dry_run)  DRY_RUN=true;  shift ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

mkdir -p "$LOG_DIR"

# ── Build newline-separated "start end" pairs for xargs ──────────────────────
PAIRS=$("$PYTHON" -c "
cs=$CHUNK
for s in range(0, $TOTAL, cs):
    print(s, min(s+cs, $TOTAL))
")

N_CHUNKS=$(echo "$PAIRS" | wc -l)

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  AutoDAN-Turbo | Mutation: $MUTATION | S2+S3"
echo "  Attacker: $ATTACKER_MODEL  Target: $TARGET_MODEL  Reshaper: $RESHAPING_MODEL"
echo "  $TOTAL prompts → $N_CHUNKS chunks of $CHUNK  |  xargs -P $PARALLEL"
echo "═══════════════════════════════════════════════════════════════"
echo "$PAIRS" | while read s e; do
    echo "  chunk $s–$e  →  $LOG_DIR/turbo_chunk_${s}_${e}.log"
done
echo ""

[[ "$DRY_RUN" == "true" ]] && { echo "[dry_run] Exiting."; exit 0; }

# ── Run via xargs -P ─────────────────────────────────────────────────────────
# Each line "start end" is passed as $1 $2 to the inner bash -c.
echo "$PAIRS" | xargs -P "$PARALLEL" -L 1 bash -c '
    start=$1; end=$2
    log="'"$LOG_DIR"'/turbo_chunk_${start}_${end}.log"
    echo "[START] chunk $start–$end → $log"
    '"$PYTHON $SCRIPT"' \
        --attacker_model  '"$ATTACKER_MODEL"' \
        --target_model    '"$TARGET_MODEL"' \
        --reshaping_model '"$RESHAPING_MODEL"' \
        --mutation        '"$MUTATION"' \
        --scenarios       '"$SCENARIOS"' \
        --algo            '"$ALGO"' \
        --range           $start $end \
        > "$log" 2>&1
    echo "[DONE]  chunk $start–$end (exit $?)"
' _

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  All chunks complete. Merging results..."
echo "═══════════════════════════════════════════════════════════════"
"$PYTHON" "$MERGE" --out_prefix "turbo_vicuna_s23_${TOTAL}"
