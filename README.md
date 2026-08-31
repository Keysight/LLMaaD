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

```bash
cd llmaad_vs_adv_attacks/autodan_results
# Detect-and-block baseline (S2)
python run_turbo_scenarios.py --mutation use_strategy --scenarios 2 --range 0 100

# Detect-and-misdirect / CMPE (S3)
python run_turbo_scenarios.py --mutation use_strategy --scenarios 3 --range 0 100
```

Results are written to `autodan_results/final_results/turbo_use_strategy_S{2,3}_detect_{block,misdirect}_n100.{json,csv}`.

### AutoDAN-Reasoning

```bash
cd llmaad_vs_adv_attacks/autodan_results
# Detect-and-block baseline (S2)
python run_reasoning_scenarios.py --method vanilla --scenarios 2 --range 0 100

# Detect-and-misdirect / CMPE (S3)
python run_reasoning_scenarios.py --method vanilla --scenarios 3 --range 0 100
```

Results are written to `autodan_results/final_results/reasoning_vanilla_S{2,3}_detect_{block,misdirect}_n100.{json,csv}`.

### PAIR with Misdirection-Hardened Judge

The standard PAIR judge can be replaced with a hardened variant that penalises misdirected
cooperative responses (i.e., responses that look helpful but contain no actionable information).
Use `run_pair_detect_misdirect_parallel_hardened_judge.py` for this configuration:

```bash
cd llmaad_vs_adv_attacks/JailbreakingLLMs
python run_pair_detect_misdirect_parallel_hardened_judge.py \
  --victim vicuna \
  --n-prompts 50
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


### AutoDAN-Turbo (`use_strategy` mutation, n=100)

Victim: lmsys/vicuna-13b-v1.5 | Scorer: gemma1-7b-it | Max epochs: 20 | Break score: 8.5

| Defense Strategy       | Total | Positive Exits | MI FP | Exhausted | FP | True Positives | Avg Iterations |
| ---------------------- | ----- | -------------- | ----- | --------- | -- | -------------- | -------------- |
| LLaMA-Guard3-8B        | 100   | 99             | —     | 1         | 85 | 14             | 2.6            |
| LLaMA-Guard3-8B + CMPE | 100   | 45             | 45    | 55        | 0  | 0              | 11.2           |


### AutoDAN-Reasoning (`vanilla` method, n=100)

Victim: lmsys/vicuna-13b-v1.5 | Scorer: gemma1-7b-it | Max epochs: 10 | Break score: 8.5

| Defense Strategy       | Total | Positive Exits | MI FP | Exhausted | FP | True Positives | Avg Iterations |
| ---------------------- | ----- | -------------- | ----- | --------- | -- | -------------- | -------------- |
| LLaMA-Guard3-8B        | 100   | 57             | —     | 43        | 52 | 5              | 5.8            |
| LLaMA-Guard3-8B + CMPE | 100   | 16             | 7     | 84        | 9  | 0              | 9.8            |


### PAIR — Misdirection-Hardened Judge (Claude Sonnet 4.6, Vicuna Target, 50 prompts)

> Source: `JailbreakingLLMs/llmaad_results/misdirect_judge_pair/`

| Model  | Defense Strategy       | Total | Positive Exits | MI FP | Exhausted | FP | True Positives | Avg Streams |
| ------ | ---------------------- | ----- | -------------- | ----- | --------- | -- | -------------- | ----------- |
| Vicuna | LLaMA-Guard3-8B        | 50    | 8              | —     | 42        | 3  | 5              | 45.8        |
| Vicuna | LLaMA-Guard3-8B + CMPE | 50    | 37             | 35    | 13        | 0  | 2              | 25.8        |
