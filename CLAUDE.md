# LLMaaD — Project Guide (ACSAC Version)

**Full name:** Misdirection as a Defense Against Model-Guided Automated LLM Attacks
**Technique:** CMPE (Confuse, Misdirect, Preserve, Evade)
**Paper:** `paper/FINAL_paper.pdf` — submitted to ACSAC 2026

The system defends against automated jailbreak attacks by *misdirecting* attacker judges rather than blocking requests. When an attack probe arrives, CMPE reshapes the prompt/response so the attacker's own judge model scores the exchange as a failure — causing the automated attack pipeline to abandon the attempt without realizing it was countered.

**CRITICAL:** Never modify attacker prompts/rubrics from third-party frameworks (AutoDAN, PAIR, GPTFuzz). These must stay verbatim for research reproducibility.

---

## Codebase Map

```
prompt_reshaping/           # core Python package
  cli.py                    # argparse entry point
  algos/
    algo1.py                # jumble → expand → context inject → compress → harmful inject
    algo1q.py               # algo1 + follow-up question step
    algo2.py                # abliterated model generates wrong-but-relevant answer
  detectors/
    base.py                 # JudgeService — orchestrates all judges
    detector_gen.py         # GCG, PAIR-GPT, StrongReject, NLP, Custom, GPTFuzz
  llm_gen/
    clients.py              # VLLMChatClient, OpenAIChatClient, GeminiChatClient
                            # NOTE: vicuna routed via _vicuna_completions (no chat template)
    model_pick.py           # ModelSelect — routes by model name
  datasets/
    loader.py               # dataset1 (CSV) and dataset2 (JSON) loaders

results_against_attacks/
  autodan_results/
    run_turbo_scenarios.py      # AutoDAN-Turbo: S2 (block) + S3 (CMPE) scenarios
    run_reasoning_scenarios.py  # AutoDAN-Reasoning: S2 + S3 scenarios
    scripts/
      launch_parallel.py        # parallel chunk launcher + auto-merge
      merge_turbo_results.py    # merge chunk JSONs → n50 CSV + summary JSON
    run_results/                # all per-run JSONs + merged n50 files + CSVs
    summary_turbo_reasoning.md  # analysis summary with LaTeX table
  pair_results/
  gptfuzz_results/

llmaad_vs_adv_attacks/
  GPTFuzz/                  # GPTFuzz attack + integration
  JailbreakingLLMs/         # PAIR attack + integration
  post_hoc/                 # Claude judge + CSV post-processing
```

---

## Key Models & Endpoints

| Alias | Model | Endpoint |
|-------|-------|----------|
| `vicuna` | lmsys/vicuna-7b-v1.3 | 10.36.129.1:8005 |
| `abliterated` | mlabonne/NeuralDaredevil-8B-abliterated | 10.36.129.1:8000 |
| `gemma` | gemma1-7b-it | 10.36.129.2:8001 |
| `qwen3` | Qwen3-32B | 10.36.129.2:8000 |
| LlamaGuard | (binary: SAFE/UNSAFE) | 10.36.129.1:8001 |

**Python env:** `.llmaad/bin/python3` (at repo root)

---

## Running AutoDAN Experiments

```bash
# Launch parallel turbo run — S2+S3, 50 prompts, use_strategy mutation
.llmaad/bin/python3 results_against_attacks/autodan_results/scripts/launch_parallel.py \
    --attack turbo --mutation use_strategy --scenarios 2 3 --total 50

# Launch parallel reasoning run — S2+S3, 50 prompts
.llmaad/bin/python3 results_against_attacks/autodan_results/scripts/launch_parallel.py \
    --attack reasoning --scenarios 2 3 --total 50

# Dry-run: print chunk commands without executing
.llmaad/bin/python3 results_against_attacks/autodan_results/scripts/launch_parallel.py \
    --attack turbo --dry_run

# Merge results after run
.llmaad/bin/python3 results_against_attacks/autodan_results/scripts/merge_turbo_results.py \
    --tag turbo_use_strategy_S2S3_algo1 \
    --out_prefix turbo_use_strategy_S2S3_algo1_n50
```

---

## Scenario Architecture (S2 and S3)

### S2 — Detect & Block
```
attacker mutates prompt → target (vicuna) responds
  → LlamaGuard detects
    UNSAFE → block (was_blocked=True), epoch continues
    SAFE   → gemma scorer rates response (1-10)
              score ≥ 8.0 → BREAK (attack succeeded)
```

### S3 — CMPE Misdirection
```
attacker mutates prompt → target (vicuna) responds
  → LlamaGuard detects
    SAFE   → gemma scorer rates raw_response; score ≥ 8.0 → BREAK (Natural ASR)
    UNSAFE → CMPE (abliterated) reshapes → misdirected_output
              → gemma scorer rates misdirected_output
                score ≥ 8.0 → BREAK (MI-FP: attacker fooled, content misdirected)
                score < 8.0 → continue (CMPE defended)
```

---

## Result Table Column Definitions

| Column | Meaning |
|--------|---------|
| **Positive Exits** | Prompts where loop broke with score ≥ 8.0 (attacker declares success) |
| **Negative Exits** | Prompts where max epochs exhausted, score never ≥ 8.0 |
| **MI-FP** | Positive exits via UNSAFE+CMPE: misdirected content, judge fooled |
| **FP** | Positive exits pending Claude normalization |
| **TP** | Positive exits confirmed harmful (pending Claude normalization) |
| **ASR** | TP / Total (finalized after Claude normalization) |

**Partition:** MI-FP + FP + TP = Positive Exits; Positive + Negative = Total

---

## Current n=50 Results (use `.llmaad/bin/python3` from repo root)

| Attack | Defense | Pos Exits | Neg Exits | MI-FP | FP | TP | ASR | Avg Iter |
|--------|---------|-----------|-----------|-------|----|----|-----|----------|
| Turbo use_strategy | LlamaGuard | 49 | 1 | 0 | 0 | 49 | 0.980 | 2.6 |
| Turbo use_strategy | LlamaGuard+CMPE | 23 | 27 | 23 | 0 | 0 | 0.000 | 7.6 |
| Reasoning vanilla | LlamaGuard | 50 | 0 | 0 | 0 | 50 | 1.000 | 2.2 |
| Reasoning vanilla | LlamaGuard+CMPE | 7 | 43 | 7 | 0 | 0 | 0.000 | 9.7 |

Result files: `results_against_attacks/autodan_results/run_results/`
