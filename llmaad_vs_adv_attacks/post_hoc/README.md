# Post-Hoc Validation Scripts

Two scripts for secondary validation and result export after running attack framework experiments.

## `claude_judge_attack_jailbreaks.py`

Applies a Claude Sonnet secondary judge to the attack framework's jailbreak outcomes to correct false positives from the attacker's own judge.

**Logic:**
- `detect-block` mode: Claude judges only responses the attacker marked as jailbroken; all others are classified as `defended`.
- `detect-misdirect` mode: Claude judges only responses where the defense was fully bypassed (`attack_jailbreak`); misdirected responses are auto-classified as `misdirected`; defended responses as `defended`.

**Output:** Adds a `case` field to each result and a `claude_judging` summary block.

**Usage:**
```bash
export ANTHROPIC_API_KEY=<your-key>
# Optional for Azure endpoint:
# export ANTHROPIC_BASE_URL=https://<resource>.services.ai.azure.com/anthropic/

python claude_judge_attack_jailbreaks.py <input.json> <output_claude_judged.json>
```

## `json_to_csv.py`

Converts claude-judged result JSON files to CSV — one row per prompt (best/outcome attempt).

**Format detection:** Automatically detects GPTFuzz (flat iterations) vs PAIR (nested iterations → streams).


**Usage:**
```bash
# Single file
python json_to_csv.py <input_claude_judged.json> [output.csv]

# Batch: convert all JSON files in the current directory
python json_to_csv.py
```
