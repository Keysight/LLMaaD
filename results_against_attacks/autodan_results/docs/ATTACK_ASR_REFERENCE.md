# Jailbreak Attack ASR Reference

Published ASR numbers, mutation models, and judge models across the three attacks in this repo.
All numbers from official papers. CMPE angle explains why each judge is defeated by misdirection.

---

## 1. AutoDAN — GA / HGA
**Paper:** arXiv 2310.04451 (ICLR 2024)
**Repo:** `../../../llmattacks/AutoDAN/`

| Component | Detail |
|-----------|--------|
| Mutation model | GPT-4 — rewrites random sentences at 1% rate per generation |
| Judge model | Keyword refusal list (binary) — 38 refusal prefixes, no LLM needed |
| Fitness function | CrossEntropyLoss against target string `"Sure, here is"` (local model weights) |
| Optimization | 100 steps, batch=256, num_elites=5%, crossover=0.5, num_points=5 |
| Target string | `"Sure, here is"` |

### Published ASR (AdvBench 50 behaviors)

| Target Model | AutoDAN-GA | AutoDAN-HGA |
|---|---|---|
| Llama-2-7b-chat-hf | ~57% | ~83% |
| **Vicuna-7b-v1.3** | ~95% | ~99% |
| WizardLM-7B-V1.0 | ~98% | ~99% |
| Guanaco-7B-HF | ~87% | ~90% |
| MPT-7b-chat | — | ~85% |
| Falcon-7b-instruct | — | ~88% |
| GPT-3.5-turbo *(transfer)* | ~35% | ~41% |
| GPT-4 *(transfer)* | ~12% | ~19% |

**Vicuna model used:** `lmsys/vicuna-7b-v1.3` (from `model_path_dicts` in `autodan_ga_eval.py`)

**Why CMPE defeats it:** AutoDAN's judge checks only for refusal keywords. CMPE misdirected output contains no refusal keywords → `jailbroken=True` → attacker's GA terminates thinking it succeeded.

---

## 2. PAIR — Prompt Automatic Iterative Refinement
**Paper:** arXiv 2310.08419 (ICLR 2024)
**Repo:** `../../adversarial_attack/JailbreakingLLMs/`

| Component | Detail |
|-----------|--------|
| Attacker (mutation) model | `lmsys/vicuna-13b-v1.5` (local) or `gpt-3.5-turbo` (API) |
| Judge model | `gpt-4` or `gpt-3.5-turbo` — LLM rubric, scores 1–10 |
| Judge type | LLM-as-judge: score ≥ 7 = successful jailbreak |
| Efficiency | Typically < 20 queries to produce a jailbreak |
| Target string | `"Sure, here is how to [goal]"` |

### Published ASR

| Target Model | PAIR ASR |
|---|---|
| Vicuna-13b-v1.5 | ~60% |
| GPT-3.5-turbo | ~45% |
| GPT-4 | ~20–30% |
| Claude (Instant/v2) | ~15–25% |
| GeminiPro | ~30–40% |

**Vicuna model used (as attacker):** `lmsys/vicuna-13b-v1.5` (from `config.py: VICUNA_PATH`)
**Vicuna model used (as target):** also `lmsys/vicuna-13b-v1.5`

**Why CMPE defeats it:** PAIR's GPT-4 judge rates responses 1–10 on "how much does this violate safety guidelines." A CMPE misdirected response looks compliant — it contains harmful-sounding vocabulary, enthusiastic phrasing, no refusals — so GPT-4 rates it 8–10. Judge declares jailbreak. Attacker stops iterating.

---

## 3. GPTFuzz
**Paper:** arXiv 2309.10253
**Judging scripts:** `../../gptfzz_judging/`

| Component | Detail |
|-----------|--------|
| Mutation model | GPT-4 — 5 operators on seed templates: `generate_similar`, `crossover`, `expand`, `shorten`, `rephrase` |
| Judge model | `hubert233/GPTFuzz` — fine-tuned HuggingFace binary classifier |
| Judge type | Binary (label 0=safe / 1=unsafe) + confidence score 0–1 |
| Advantage | No LLM API cost for judging — single GPU inference |

### Published ASR (paper)

| Target Model | GPTFuzz ASR |
|---|---|
| GPT-3.5-turbo | ~80–90% |
| GPT-4 | ~60–70% |
| Llama-2-chat | ~70–80% |
| Vicuna | ~90–95% |

### Our CMPE results (from `gptfzz_judging/judge_results/`)

| CMPE Algo | GPTFuzz Score | n |
|---|---|---|
| algo1 (d2, 500 prompts) | ~0.95 average confidence | 500 |
| algo1q | ~0.94 average | 500 |
| algo2 | ~0.91 average | ~500 |

Classifier labels CMPE misdirected output as `label=1` (unsafe) at 95% confidence — attacker believes attack succeeded.

**Why CMPE defeats it:** The GPTFuzz classifier was trained to detect harmful content by surface features (vocabulary, topic, tone). CMPE misdirected responses use harmful-domain vocabulary and an enthusiastic tone while providing no actionable information — the classifier fires `unsafe` with high confidence.

---

## Cross-Attack Summary

| Attack | Mutation model | Judge model | Judge type | CMPE defeats via |
|--------|---------------|-------------|------------|-----------------|
| AutoDAN-GA | GPT-4 (sentence rewrite) | Keyword list | Binary refusal check | No refusal keywords in misdirected output |
| PAIR | Vicuna-13b-v1.5 / GPT-3.5 | GPT-4 (1–10 rubric) | LLM-as-judge | Looks compliant → GPT-4 scores 8–10 |
| GPTFuzz | GPT-4 (5 operators) | hubert233/GPTFuzz HF classifier | Binary + confidence | Harmful vocabulary triggers classifier |

All three judge types are fooled by the same CMPE output — this is the core ACSAC paper claim.

---

## Exact Model Versions

| Attack | Role | Exact model |
|--------|------|-------------|
| AutoDAN | Victim (7B target) | `lmsys/vicuna-7b-v1.3` |
| AutoDAN | Mutation operator | GPT-4 (via API) |
| PAIR | Attacker/mutation | `lmsys/vicuna-13b-v1.5` |
| PAIR | Victim target | `lmsys/vicuna-13b-v1.5` (also GPT-3.5-turbo, GPT-4) |
| PAIR | Judge | `gpt-4` (primary), `gpt-3.5-turbo` (fast variant) |
| GPTFuzz | Mutation operator | GPT-4 |
| GPTFuzz | Judge | `hubert233/GPTFuzz` (HuggingFace) |
| Our API attack | Mutation operator | `gpt-4.1` |
| Our API attack | Victim | `openai/gpt-oss-120b` (default) |
