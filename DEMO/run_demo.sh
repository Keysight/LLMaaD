#!/usr/bin/env bash
# Launcher for the stage demo. Picks the right base-model source for wherever
# you are: inside the DGX lab container the pointer file exists and the weights
# are already local; anywhere else it falls back to the Hub id.
#
#   ./run_demo.sh                 interactive
#   ./run_demo.sh --walkthrough   scripted stage sequence
#   DEMO_BASE=/path/to/checkpoint ./run_demo.sh
#
# Any extra arguments are passed straight through to demo.py.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
POINTER="${SLM_MODEL_POINTER:-/lab/state/current-model.json}"

# Prefer a venv over the system interpreter. macOS ships Python 3.9 with a
# user-site transformers whose huggingface-hub pin is already broken, so
# defaulting to `python3` here fails before the model ever loads.
if [[ -n "${PYTHON:-}" ]]; then
  :
elif [[ -x "$HERE/.venv/bin/python" ]]; then
  PYTHON="$HERE/.venv/bin/python"
elif [[ -x "$HERE/../.venv/bin/python" ]]; then
  PYTHON="$HERE/../.venv/bin/python"
else
  PYTHON="python3"
fi

ARGS=()
if [[ -n "${DEMO_BASE:-}" ]]; then
  ARGS+=(--base "$DEMO_BASE")
elif [[ -f "$POINTER" ]]; then
  ARGS+=(--pointer "$POINTER")
else
  ARGS+=(--base Qwen/Qwen2.5-3B-Instruct)
  echo "No local checkpoint pointer at $POINTER — resolving the base model from the Hub." >&2
  echo "First run downloads ~6 GB. Do that before you are on stage, not during." >&2
fi

exec "$PYTHON" "$HERE/demo.py" "${ARGS[@]}" "$@"
