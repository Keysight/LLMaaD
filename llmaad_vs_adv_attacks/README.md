# End-to-End Adversarial Evaluation Harness

This directory contains the evaluation harness for testing the LLMaaD detect-and-misdirect defense against three state-of-the-art model-guided attack frameworks: **AutoDAN-Turbo**, **GPTFuzz**, and **PAIR**.

## Directory Structure

```
llmaad_vs_adv_attacks/
├── autodan_results/                          # AutoDAN-Turbo integration
│   ├── run_turbo_scenarios.py                # AutoDAN-Turbo S2/S3 scenario runner
│   ├── run_reasoning_scenarios.py            # AutoDAN-Reasoning S2/S3 runner (experimental)
│   ├── scripts/
│   │   ├── launch_parallel.py               # Parallel chunk launcher (turbo + reasoning)
│   │   └── merge_turbo_results.py           # Merge chunk JSONs → CSV + summary JSON
│   ├── llmaad_results/
│   │   ├── turbo/                           # Final turbo results (n=50, claude-judged)
│   │   └── reasoning/                       # Reasoning results + ISSUES doc
│   └── REFERENCES.md                         # Attribution for AutoDAN assets used
├── GPTFuzz/                                  # GPTFuzz framework (vendored, MIT) + integration
│   ├── gptfuzz_llmaad_parallel.py            # Main integration script (parallel, 50 prompts)
│   ├── llmaad_results/                       # Final experiment results (claude-judged)
│   └── [upstream GPTFuzz code]
├── JailbreakingLLMs/                         # PAIR framework (vendored, MIT) + integration
│   ├── run_pair.py                           # Unified PAIR launcher (all 5 modes)
│   ├── run_pair_baseline_parallel.py         # PAIR baseline (no defense)
│   ├── run_pair_detect_block_parallel.py     # PAIR + detect-and-block
│   ├── run_pair_detect_misdirect_parallel.py # PAIR + detect-and-misdirect (CMPE)
│   ├── run_pair_detect_block_parallel_hardened_judge.py
│   ├── run_pair_detect_misdirect_parallel_hardened_judge.py
│   ├── system_prompts.py                     # Standard + misdirection-aware judge prompts
│   ├── llmaad_results/                       # Final experiment results (claude-judged)
│   └── [upstream JailbreakingLLMs code]
├── post_hoc/                                 # Post-hoc validation and analysis
│   ├── claude_judge_autodan.py               # Claude judge for AutoDAN results
│   └── claude_judge_attack_jailbreaks.py     # Claude judge for GPTFuzz/PAIR results
└── final_results.md                          # Summary tables (paper §VI-B)
```

## Model Configuration

All experiments use the following model stack (served locally via vLLM):

| Role | Model |
|------|-------|
| Victim (primary) | lmsys/vicuna-13b-v1.5 |
| Victim (secondary) | mlabonne/NeuralDaredevil-8B-abliterated |
| Attacker / Mutator | gemma1-7b-it (AutoDAN) · mlabonne/NeuralDaredevil-8B-abliterated (PAIR/GPTFuzz) |
| Scorer / Judge | openai/gpt-oss-120b |
| Defender (detection) | meta-llama/Llama-Guard-3-8B |
| Reshaper (CMPE) | mlabonne/NeuralDaredevil-8B-abliterated |

Model endpoints (IPs and ports) are configured via environment variables at runtime — see each script's `--help` output or the `TARGET_MODEL_MAP` / argument defaults inside `run_turbo_scenarios.py` and `run_reasoning_scenarios.py` for the variable names.

## Running AutoDAN Experiments

### Scripts

| Script | Purpose |
|--------|---------|
| `autodan_results/run_turbo_scenarios.py` | AutoDAN-Turbo S2/S3 runner — handles warm_up, use_strategy, find_new_strategy mutations; integrates LlamaGuard + CMPE algo1q; saves strategy library after each S3 run |
| `autodan_results/run_reasoning_scenarios.py` | AutoDAN-Reasoning S2/S3 runner — implements vanilla, best-of-N, and beam-search attack methods against the same S2/S3 scenario architecture |
| `autodan_results/scripts/launch_parallel.py` | Splits the prompt range into chunks and launches scenario runners in parallel subprocesses; supports both turbo and reasoning attacks |
| `autodan_results/scripts/merge_turbo_results.py` | Merges per-chunk JSON outputs into a single result file with summary table and CSV export |

Both scenario runners accept `--help` for the full argument list. Key shared flags:

```
--target        victim model alias (vicuna | abliterated | qwen3-abliterated | oss120b)
--scenarios     which scenarios to run (2 = detect-block, 3 = detect-misdirect)
--range         prompt index range, e.g. 0 50
--algo          CMPE algorithm for S3 (algo1 | algo1q | algo2)  [default: algo1q]
--scorer_model  scoring/judge model alias                        [default: oss120b]
--out_dir       output directory for result JSONs
```

### Running via the parallel launcher (recommended)

Run from the **repo root**:

```bash
# AutoDAN-Turbo — S2 (detect-block), vicuna target, 50 prompts, chunks of 5
.llmaad/bin/python3 llmaad_vs_adv_attacks/autodan_results/scripts/launch_parallel.py \
    --attack turbo --mutation use_strategy --scenarios 2 --total 50 \
    --target vicuna --chunk_size 5

# AutoDAN-Turbo — S3 (detect-misdirect / CMPE), vicuna target
.llmaad/bin/python3 llmaad_vs_adv_attacks/autodan_results/scripts/launch_parallel.py \
    --attack turbo --mutation use_strategy --scenarios 3 --total 50 \
    --target vicuna --chunk_size 5

# Repeat with --target abliterated for the second victim model

# AutoDAN-Reasoning — S2 + S3, vicuna target
.llmaad/bin/python3 llmaad_vs_adv_attacks/autodan_results/scripts/launch_parallel.py \
    --attack reasoning --scenarios 2 3 --total 50 --target vicuna --chunk_size 5

# Dry-run: print chunk commands without executing
.llmaad/bin/python3 llmaad_vs_adv_attacks/autodan_results/scripts/launch_parallel.py \
    --attack turbo --dry_run
```

### Running a single scenario directly

```bash
# Turbo S3, prompts 0–10, vicuna target
.llmaad/bin/python3 llmaad_vs_adv_attacks/autodan_results/run_turbo_scenarios.py \
    --mutation use_strategy --scenarios 3 --range 0 10 --target vicuna

# Reasoning S2, prompts 0–50, abliterated target
.llmaad/bin/python3 llmaad_vs_adv_attacks/autodan_results/run_reasoning_scenarios.py \
    --method vanilla --scenarios 2 --range 0 50 --target abliterated
```

Results land in `autodan_results/run_results/`. Apply Claude judging after:

```bash
export ANTHROPIC_FOUNDRY_API_KEY=<key>
export ANTHROPIC_FOUNDRY_RESOURCE=<resource-name>

.llmaad/bin/python3 llmaad_vs_adv_attacks/post_hoc/claude_judge_autodan.py \
    autodan_results/run_results/<result>.json \
    autodan_results/llmaad_results/turbo/<result>_claude_judged.json
```

> **Strategy library:** `lifelong_strategy_library.pkl` is required for `use_strategy` and `find_new_strategy` mutations but is not tracked in this repo (>100 MB). See [`autodan_results/REFERENCES.md`](autodan_results/REFERENCES.md) for the download command.

> **AutoDAN-Reasoning:** Two experimental runs were conducted but could not be used for the final paper due to invalid threat model configurations and a merge bug. See [`autodan_results/llmaad_results/reasoning/ISSUES_gemma_attacker_judge.md`](autodan_results/llmaad_results/reasoning/ISSUES_gemma_attacker_judge.md) for details.

## Running GPTFuzz Experiments

```bash
cd GPTFuzz

# detect-and-block
python gptfuzz_llmaad_parallel.py --defense-mode detect-block \
    --victim vicuna --n-prompts 50

# detect-and-misdirect
python gptfuzz_llmaad_parallel.py --defense-mode detect-misdirect \
    --victim vicuna --n-prompts 50
```

## Running PAIR Experiments

Use `run_pair.py` to launch any of the five PAIR variants:

```bash
cd JailbreakingLLMs

# Detect-and-block (standard judge), vicuna target
python run_pair.py --mode detect-block --num-prompts 50

# Detect-and-misdirect (standard judge)
python run_pair.py --mode detect-misdirect --num-prompts 50

# Detect-and-misdirect (misdirection-hardened judge)
python run_pair.py --mode detect-misdirect --judge hardened --num-prompts 50

# Abliterated target (set --target-ip/--target-port to match your vLLM serving setup)
python run_pair.py --mode detect-misdirect \
    --target-model mlabonne/NeuralDaredevil-8B-abliterated \
    --num-prompts 50

# Dry-run all modes to preview commands
python run_pair.py --mode all --judge both --dry-run
```

Individual scripts (`run_pair_detect_block_parallel.py`, etc.) can also be called directly with the same arguments.

## Post-Hoc Validation

After collecting raw results, apply secondary Claude judging:

```bash
export ANTHROPIC_FOUNDRY_API_KEY=<key>
export ANTHROPIC_FOUNDRY_RESOURCE=<resource-name>

# AutoDAN results
.llmaad/bin/python3 post_hoc/claude_judge_autodan.py \
    autodan_results/run_results/<result>.json \
    autodan_results/llmaad_results/turbo/<result>_claude_judged.json

# GPTFuzz / PAIR results
.llmaad/bin/python3 post_hoc/claude_judge_attack_jailbreaks.py \
    JailbreakingLLMs/llmaad_results/<result>.json \
    JailbreakingLLMs/llmaad_results/<result>_claude_judged.json
```

## Results

Validated experiment results are stored in:
- `autodan_results/llmaad_results/turbo/` — 4 JSON + 4 CSV files (vicuna + abliterated, S2 + S3, n=50)
- `GPTFuzz/llmaad_results/` — detect-block + detect-misdirect results for vicuna + abliterated
- `JailbreakingLLMs/llmaad_results/` — detect-block + detect-misdirect + hardened judge results

Summary tables (with column definitions) are in [`final_results.md`](final_results.md).
