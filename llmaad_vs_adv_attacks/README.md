# End-to-End Adversarial Evaluation Harness

This directory contains the evaluation harness for testing the LLMaaD detect-and-misdirect defense against three state-of-the-art model-guided attack frameworks: **AutoDAN** (Turbo + Reasoning), **GPTFuzz**, and **PAIR**.

## Directory Structure

```
llmaad_vs_adv_attacks/
├── autodan_results/                          # AutoDAN-Turbo + AutoDAN-Reasoning integration
│   ├── run_turbo_scenarios.py                # AutoDAN-Turbo S2/S3 scenario runner
│   ├── run_reasoning_scenarios.py            # AutoDAN-Reasoning S2/S3 scenario runner
│   ├── scripts/
│   │   ├── launch_parallel.py               # Parallel chunk launcher
│   │   └── merge_turbo_results.py           # Merge chunks → CSV + summary JSON
│   ├── final_results/                        # Final experiment results (n=100, claude-judged)
│   └── REFERENCES.md                         # Attribution for AutoDAN assets used
├── GPTFuzz/                                  # GPTFuzz framework (vendored, MIT) + our integration
│   ├── gptfuzz_llmaad_parallel.py            # Main integration script (parallel, 50 prompts)
│   ├── llmaad_results/                       # Final experiment results (claude-judged)
│   └── [upstream GPTFuzz code]
├── JailbreakingLLMs/                         # PAIR framework (vendored, MIT) + our integration
│   ├── run_pair_detect_block_parallel.py     # PAIR with detect-and-block defense
│   ├── run_pair_detect_misdirect_parallel.py # PAIR with detect-and-misdirect defense
│   ├── system_prompts.py                     # Victim model system prompts
│   ├── llmaad_results/                       # Final experiment results (claude-judged)
│   └── [upstream JailbreakingLLMs code]
├── post_hoc/                                 # Post-hoc validation and analysis
│   ├── claude_judge_attack_jailbreaks.py     # Secondary LLM judge (Claude Sonnet)
│   └── json_to_csv.py                        # Export results to CSV
└── final_results.md                          # Summary tables (paper VI-B)
```

## Model Configuration

All experiments use the following model stack (served locally via vLLM):

| Role | Model |
|------|-------|
| Victim (primary) | lmsys/vicuna-13b-v1.5 |
| Victim (secondary) | mlabonne/NeuralDaredevil-8B-abliterated |
| Mutator / Attacker | gpt-3.5-turbo (GPTFuzz/PAIR) · mlabonne/NeuralDaredevil-8B-abliterated (AutoDAN) |
| Defender | meta-llama/Llama-Guard-3-8B |
| Reshaper (CMPE) | mlabonne/NeuralDaredevil-8B-abliterated |
| Scorer (AutoDAN) | gemma1-7b-it |

## Running AutoDAN Experiments

```bash
# AutoDAN-Turbo: S2 (detect-block) + S3 (detect-misdirect), 100 prompts
.llmaad/bin/python3 autodan_results/scripts/launch_parallel.py \
    --attack turbo --mutation use_strategy --scenarios 2 3 --total 100

# AutoDAN-Reasoning: S2 + S3, 100 prompts
.llmaad/bin/python3 autodan_results/scripts/launch_parallel.py \
    --attack reasoning --scenarios 2 3 --total 100

# Merge results after run
.llmaad/bin/python3 autodan_results/scripts/merge_turbo_results.py \
    --in_dir autodan_results/final_results \
    --out_dir autodan_results/final_results \
    --out_prefix turbo_use_strategy_S2_detect_block_n100 \
    --tag turbo_use_strategy_S2_detect_block
```

See [`autodan_results/REFERENCES.md`](autodan_results/REFERENCES.md) for attribution of AutoDAN assets used.

## Running GPTFuzz Experiments

```bash
cd GPTFuzz
export OPENAI_API_KEY=<your-key>

# detect-and-block
python gptfuzz_llmaad_parallel.py --defense-mode detect-block --victim vicuna --n-prompts 50

# detect-and-misdirect
python gptfuzz_llmaad_parallel.py --defense-mode detect-misdirect --victim vicuna --n-prompts 50
```

## Running PAIR Experiments

```bash
cd JailbreakingLLMs
export OPENAI_API_KEY=<your-key>

# detect-and-block
python run_pair_detect_block_parallel.py --victim vicuna --n-prompts 50

# detect-and-misdirect
python run_pair_detect_misdirect_parallel.py --victim vicuna --n-prompts 50
```

## Post-Hoc Validation

After collecting raw results, apply secondary Claude judging and export to CSV:

```bash
export ANTHROPIC_API_KEY=<your-key>
# Optional for Azure endpoint:
# export ANTHROPIC_BASE_URL=https://<resource>.services.ai.azure.com/anthropic/

# Step 1: Claude judge
python post_hoc/claude_judge_attack_jailbreaks.py \
  GPTFuzz/llmaad_results/llamaguard/<result>.json \
  GPTFuzz/llmaad_results/llamaguard/claude_judged/<result>_claude_judged.json

# Step 2: Export to CSV
python post_hoc/json_to_csv.py \
  GPTFuzz/llmaad_results/llamaguard/claude_judged/<result>_claude_judged.json
```

See [`post_hoc/README.md`](post_hoc/README.md) for details.

## Results

Validated experiment results are stored in:
- `autodan_results/final_results/` — 8 files (2 attacks × 2 defense modes × n=100, JSON + CSV)
- `GPTFuzz/llmaad_results/llamaguard/claude_judged/` — 4 JSON files (2 models × 2 defense modes)
- `JailbreakingLLMs/llmaad_results/llamaguard/claude_judged/` — 4 JSON files

Summary tables are in [`final_results.md`](final_results.md).
