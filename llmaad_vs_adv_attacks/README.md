# End-to-End Adversarial Evaluation Harness

This directory contains the evaluation harness for testing the LLMaaD detect-and-misdirect defense against two state-of-the-art model-guided attack frameworks: **GPTFuzz** and **PAIR**.

## Directory Structure

```
llmaad_vs_adv_attacks/
├── GPTFuzz/                          # GPTFuzz framework (vendored, MIT) + our integration
│   ├── gptfuzz_llmaad_parallel.py    # Main integration script (parallel, 50 prompts)
│   ├── custom_scripts/               # Batch runners, baselines, test scripts
│   ├── ATTACK_REPORT.md              # Setup and usage documentation
│   ├── llmaad_results/llamaguard/    # Final experiment results (claude-judged)
│   └── [upstream GPTFuzz code]
├── JailbreakingLLMs/                 # PAIR framework (vendored, MIT) + our integration
│   ├── run_pair_detect_block_parallel.py     # PAIR with detect-and-block defense
│   ├── run_pair_detect_misdirect_parallel.py # PAIR with detect-and-misdirect defense
│   ├── system_prompts.py             # Victim model system prompts
│   ├── custom_scripts/               # Batch runners and utilities
│   ├── PAIR_ATTACK.md                # Setup and usage documentation
│   ├── llmaad_results/llamaguard/    # Final experiment results (claude-judged)
│   └── [upstream JailbreakingLLMs code]
├── post_hoc/                         # Post-hoc validation and analysis
│   ├── claude_judge_attack_jailbreaks.py  # Secondary LLM judge (Claude Sonnet)
│   └── json_to_csv.py                    # Export results to CSV
└── final_results.md                  # Summary tables (paper §VI-B)
```

## Model Configuration

All experiments use the following model stack (served locally via vLLM):

| Role | Model | Endpoint |
|------|-------|---------|
| Victim (primary) | lmsys/vicuna-13b-v1.5 | `http://localhost:8000/v1` |
| Victim (secondary) | mlabonne/NeuralDaredevil-8B-abliterated | `http://localhost:8001/v1` |
| Mutator / Attacker | gpt-3.5-turbo | OpenAI API |
| Defender | meta-llama/Llama-Guard-3-8B | `http://localhost:8002/v1` |
| Reshaper (CMPE) | mlabonne/NeuralDaredevil-8B-abliterated | `http://localhost:8001/v1` |

## Running GPTFuzz Experiments

See [`GPTFuzz/ATTACK_REPORT.md`](GPTFuzz/ATTACK_REPORT.md) for full instructions.

```bash
cd GPTFuzz
export OPENAI_API_KEY=<your-key>

# detect-and-block
python gptfuzz_llmaad_parallel.py --defense-mode detect-block --victim vicuna --n-prompts 50

# detect-and-misdirect
python gptfuzz_llmaad_parallel.py --defense-mode detect-misdirect --victim vicuna --n-prompts 50
```

## Running PAIR Experiments

See [`JailbreakingLLMs/PAIR_ATTACK.md`](JailbreakingLLMs/PAIR_ATTACK.md) for full instructions.

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
- `GPTFuzz/llmaad_results/llamaguard/claude_judged/` — 4 JSON files (2 models × 2 defense modes)
- `JailbreakingLLMs/llmaad_results/llamaguard/claude_judged/` — 4 JSON files

Summary tables are in [`final_results.md`](final_results.md).
