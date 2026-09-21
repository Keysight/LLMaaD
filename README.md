# LLMaaD: LLM-Misdirection-as-a-Defense-Strategy

**Misdirection as a Defense Against Adversarial AI Agents**

> Research artifact for the paper submitted to IEEE ACSAC 2026.

## Abstract

As generative AI is integrated into agentic applications, defenses increasingly combine model-level safeguards and external security controllers to monitor and mediate agent interactions. At the same time, attackers are adopting model-guided automation to scale multi-step probing, semantic prompt refinement, and response evaluation beyond manual effort. We analyze this setting probabilistically and derive asymptotic bounds on attacker success rate (ASR) as a function of response evaluation error rates. The analysis shows that conventional detect-and-block defense strategies can allow ASR to approach one as the attacker query budget grows. To address this limitation, we propose a **detect-and-misdirect** defense strategy that replaces predictable refusal responses with controlled, non-operational responses designed to corrupt the attacker's automated response evaluation process through misdirection-induced false-positives. We show that such false-positives result in a bounded asymptotic ASR. We instantiate this idea through **Contextual Misdirection via Progressive Engagement (CMPE)**, a lightweight conversational misdirection method, and evaluate it on jailbreak benchmarks against multiple automated judge models. CMPE substantially increases attacker-side false-positives and reduces estimated ASR by up to two orders of magnitude across simulated attacker-defender judge configurations. We further evaluate CMPE against representative model-guided attack frameworks, PAIR and GPTFuzz, where it substantially reduces verified attack success rate and induces premature termination.

---

## Repository Structure

| Directory | Description | Paper Section |
|-----------|-------------|---------------|
| `prompt_reshaping/` | Core LLMaaD defense framework — CMPE algorithm, detectors, LLM clients, CLI | IV, V |
| `llmaad_vs_adv_attacks/GPTFuzz/` | GPTFuzz attack framework (vendored, MIT) + our integration scripts | VI-B |
| `llmaad_vs_adv_attacks/JailbreakingLLMs/` | PAIR attack framework (vendored, MIT) + our integration scripts | VI-B |
| `llmaad_vs_adv_attacks/autodan_results/` | AutoDAN-Turbo and AutoDAN-Reasoning attack integration scripts + results | VI-B |
| `llmaad_vs_adv_attacks/post_hoc/` | Post-hoc validation scripts (Claude judge + CSV export) | VI-B |
| `llmaad_vs_adv_attacks/final_results.md` | Summary tables of end-to-end evaluation results | VI-B |

---

## Installation

```bash
git clone https://github.com/<org>/llmaad.git
cd llmaad
pip install -r requirements.txt
```

### Additional Requirements

- **vLLM** (recommended for serving victim and reshaper models locally):
  ```bash
  pip install vllm
  ```
- **Llama-Guard-3-8B** for detection: served via vLLM endpoint.
- **NeuralDaredevil-8B-abliterated** for reshaping: served via vLLM endpoint.
- **OpenAI API key** for GPTFuzz and PAIR mutator/attacker model (`gpt-3.5-turbo`).

---

## Quick Start: Running the Defense

```bash
# Run CMPE misdirection on a single harmful prompt
python -m prompt_reshaping.cli \
  --algo algo1q \
  --victim-url http://localhost:8000/v1 \
  --victim-model vicuna-13b \
  --defender-url http://localhost:8001/v1 \
  --defender-model Llama-Guard-3-8B \
  --reshaper-url http://localhost:8002/v1 \
  --reshaper-model NeuralDaredevil-8B-abliterated \
  --mode detect-misdirect
```

See [`prompt_reshaping/README.md`](prompt_reshaping/README.md)
---

## Reproducing the End-to-End Evaluation

### GPTFuzz

```bash
cd llmaad_vs_adv_attacks/GPTFuzz
python gptfuzz_llmaad_parallel.py \
  --defense-mode detect-misdirect \
  --victim vicuna \
  --n-prompts 50
```

### PAIR

```bash
cd llmaad_vs_adv_attacks/JailbreakingLLMs
python run_pair_detect_misdirect_parallel.py \
  --victim vicuna \
  --n-prompts 50
```

### Post-Hoc Validation

After running experiments, use the post-hoc scripts to apply secondary Claude judging and export results:

```bash
export ANTHROPIC_API_KEY=<your-key>
# Optional for Azure endpoint:
# export ANTHROPIC_BASE_URL=https://<your-resource>.services.ai.azure.com/anthropic/

python llmaad_vs_adv_attacks/post_hoc/claude_judge_attack_jailbreaks.py \
  <input_results.json> <output_judged.json>

python llmaad_vs_adv_attacks/post_hoc/json_to_csv.py \
  <output_judged.json>
```

### AutoDAN-Turbo

Use the parallel launcher from the repo root (`--target` selects the victim model):

```bash
# S2 — detect-and-block, vicuna target, 50 prompts
.llmaad/bin/python3 llmaad_vs_adv_attacks/autodan_results/scripts/launch_parallel.py \
    --attack turbo --mutation use_strategy --scenarios 2 --total 50 --target vicuna

# S3 — detect-and-misdirect (CMPE), vicuna target
.llmaad/bin/python3 llmaad_vs_adv_attacks/autodan_results/scripts/launch_parallel.py \
    --attack turbo --mutation use_strategy --scenarios 3 --total 50 --target vicuna

# Repeat with --target abliterated for the second victim model
```

Results land in `llmaad_vs_adv_attacks/autodan_results/run_results/`; Claude-judged finals are in
`llmaad_vs_adv_attacks/autodan_results/llmaad_results/turbo/`.

### PAIR

A unified launcher covers all five PAIR script variants:

```bash
cd llmaad_vs_adv_attacks/JailbreakingLLMs

# Detect-and-block (standard judge)
python run_pair.py --mode detect-block --num-prompts 50

# Detect-and-misdirect (standard judge)
python run_pair.py --mode detect-misdirect --num-prompts 50

# Detect-and-misdirect (hardened judge)
python run_pair.py --mode detect-misdirect --judge hardened --num-prompts 50

# All modes in sequence, dry-run first
python run_pair.py --mode all --judge both --dry-run
```

Post-hoc validation is the same as above (Claude judge + CSV export).


---

## Results Summary

Full results are in [`llmaad_vs_adv_attacks/final_results.md`](llmaad_vs_adv_attacks/final_results.md).

> **Positive Exits = MI FP + FP + True Pos** — columns reflect the attacker's perspective; see [`final_results.md`](llmaad_vs_adv_attacks/final_results.md) for full column descriptions.

### GPTFuzz (50 prompts, max 50 iterations)

| Model       | Defense Strategy       | Positive Exits | MI FP | Exhausted | FP | True Positives | Avg Iterations |
| ----------- | ---------------------- | -------------- | ----- | --------- | -- | -------------- | -------------- |
| Vicuna      | LLaMA-Guard3-8B        | 23             | —     | 27        | 13 | 10             | 37.5           |
| Vicuna      | LLaMA-Guard3-8B + CMPE | 48             | 44    | 2         | 4  | 0              | 6.8            |
| Abliterated | LLaMA-Guard3-8B        | 22             | —     | 28        | 14 | 8              | 34.0           |
| Abliterated | LLaMA-Guard3-8B + CMPE | 48             | 42    | 2         | 5  | 1              | 6.4            |


### PAIR (50 prompts, max 5 iterations × 10 streams)

| Model        | Defense Strategy       | Positive Exits | MI FP | Exhausted | FP | True Positives | Avg Streams |
| ------------ | ---------------------- | -------------- | ----- | --------- | -- | -------------- | ----------- |
| Vicuna       | LLaMA-Guard3-8B        | 8              | —     | 42        | 3  | 5              | 45.4        |
| Vicuna       | LLaMA-Guard3-8B + CMPE | 43             | 43    | 7         | 0  | 0              | 20.4        |
| Abliterated  | LLaMA-Guard3-8B        | 12             | —     | 38        | 7  | 5              | 41.4        |
| Abliterated* | LLaMA-Guard3-8B + CMPE | 47             | 47    | 3         | 0  | 0              | 15.8        |


### AutoDAN-Turbo (`use_strategy` mutation, n=50 per target, algo1q)

Attacker: gemma1-7b-it | Scorer: openai/gpt-oss-120b | Max epochs: 20 | Break score: 8.5

| Model       | Defense Strategy       | Total | Positive Exits | MI FP | Exhausted | FP | True Positives | Avg Iterations |
| ----------- | ---------------------- | ----- | -------------- | ----- | --------- | -- | -------------- | -------------- |
| Vicuna      | LLaMA-Guard3-8B        | 50    | 6              | —     | 44        | 3  | 3              | 18.8           |
| Vicuna      | LLaMA-Guard3-8B + CMPE | 50    | 5              | 5     | 45        | 0  | 0              | 19.2           |
| Abliterated | LLaMA-Guard3-8B        | 50    | 13             | —     | 37        | 5  | 8              | 16.3           |
| Abliterated | LLaMA-Guard3-8B + CMPE | 50    | 26             | 26    | 24        | 0  | 0              | 13.0           |


### PAIR — Misdirection-Hardened Judge

| Model  | Defense Strategy       | Total | Positive Exits | MI FP | Exhausted | FP | True Positives | Avg Streams |
| ------ | ---------------------- | ----- | -------------- | ----- | --------- | -- | -------------- | ----------- |
| Vicuna | LLaMA-Guard3-8B        | 50    | 8              | —     | 42        | 3  | 5              | 45.8        |
| Vicuna | LLaMA-Guard3-8B + CMPE | 50    | 37             | 35    | 13        | 0  | 2              | 25.8        |
