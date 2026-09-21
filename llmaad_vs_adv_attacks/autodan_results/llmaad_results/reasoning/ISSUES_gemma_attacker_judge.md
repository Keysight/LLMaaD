# AutoDAN-Reasoning — Issues with Gemma as Attacker and Judge

## Overview

Two separate runs of AutoDAN-Reasoning (vanilla method, n=50, algo1q, S2+S3) were conducted with
different attacker/scorer configurations. Both produced results that could not be used as-is for
the final paper table, for the reasons documented below.

---

## Run 1 — Abliterated as Attacker + Gemma as Scorer

**Config:**
- Attacker: `mlabonne/NeuralDaredevil-8B-abliterated` @ 10.36.129.1:8000
- Target: `mlabonne/NeuralDaredevil-8B-abliterated` @ 10.36.129.1:8000
- Reshaper (CMPE): `mlabonne/NeuralDaredevil-8B-abliterated` @ 10.36.129.1:8000
- Scorer/Judge: `gemma1-7b-it` @ 10.36.129.2:8001
- Result file: `abliterated_attacker_S2S3_algo1q_n50_claude_judged.json`

### Issues

**1. Attacker = Target = Reshaper (same model)**
All three roles were assigned to the same model instance (abliterated). This means the attacker
generating jailbreak prompts and the target responding to them are identical, which is not a
valid threat model. The attacker should be independent from the target.

**2. Gemma scorer calibration — inflated refusals**
Gemma produced consistently low scores even on responses that contained partial harmful content.
The S2 result shows 5 positive exits all classified as `false_positive` after Claude judging —
meaning Gemma scored them ≥ 8.5 but Claude confirmed they were not actually harmful. This suggests
Gemma's scoring threshold is miscalibrated relative to the oss120b rubric used in GPTFuzz and PAIR.

**3. S3 misdirected_jailbroken anomaly**
6 out of 6 positive exits in S3 were classified as `misdirected_jailbroken` — the attacker was
declared successful on the CMPE-misdirected output, and Claude confirmed the raw response was
genuinely harmful. However, given Gemma's known calibration issues, these scores may be
unreliable (Gemma may have scored the misdirected output ≥ 8.5 for wrong reasons).

**4. No scorer in run log**
The run log does not record a `scorer_model` field — the scorer was passed via the old default
argument (`gemma`) before we standardised to `oss120b`. This makes the run non-reproducible
without checking the launch script defaults at the time.

---

## Run 2 — gpt-oss-120b as Attacker + oss120b as Scorer

**Config:**
- Attacker: `openai/gpt-oss-120b` (alias `gpt-oss`) @ 10.36.129.6:8000
- Target: `lmsys/vicuna-13b-v1.5` @ 10.36.129.2:8000
- Reshaper (CMPE): `mlabonne/NeuralDaredevil-8B-abliterated` @ 10.36.129.1:8000
- Scorer/Judge: `openai/gpt-oss-120b` @ 10.36.129.6:8000
- Result file: `oss120b_attacker_S2S3_algo1q_n50.json`

### Issues

**1. Claude judging failed — all rows returned `unknown`**
The Claude judging step was run against this file but every prompt returned `claude_case: unknown`.
Root cause: the `attack_succeeded` field was `True` for all 50 prompts in both S2 and S3
(100% attack rate), which meant every row was sent to the judge. However, the `raw_response`
field was missing or empty in all rows — likely a serialisation bug in the version of
`run_reasoning_scenarios.py` that was active at the time of the run. Without a raw response
Claude has nothing to judge, so it returned unknown for all entries.

**2. Suspiciously high attack rate (S2: 15/50, S3: 1/50 raw; but logs show 50/50)**
The merged JSON shows `attack_succeeded=True` for all 50 prompts in both scenarios. This conflicts
with the per-chunk logs which showed varied outcomes. The merge script at the time had a bug
where it defaulted `attack_succeeded` to `True` when the field was missing from a chunk result.

**3. gpt-oss-120b used as both attacker and scorer**
Using the same model as attacker and judge creates a potential circular bias — the model may be
more inclined to score its own generated jailbreak attempts highly. For a clean evaluation the
attacker and scorer should be different models.

**4. Result cannot be Claude-judged or used in paper**
Due to the missing `raw_response` fields and the merge bug, this run's results are not usable
for the final paper table. It is retained here for reference only.

---

## Conclusion

Neither reasoning run produced results suitable for the paper:

| Run | Attacker | Scorer | Blocker |
|-----|----------|--------|---------|
| Run 1 (abliterated) | Attacker = Target = Reshaper (invalid threat model) | Gemma miscalibrated | ✗ |
| Run 2 (gpt-oss-120b) | gpt-oss = Scorer (circular) | merge bug → all unknown | ✗ |

The AutoDAN-Reasoning section was removed from `final_results.md`. If a clean reasoning run is
needed for the paper, it should use:
- Attacker: `gemma1-7b-it` (independent from target)
- Target: `vicuna` or `abliterated`
- Scorer: `openai/gpt-oss-120b`
- Same 50 prompts (rows 0–49 of `harmful_behaviors.csv`) as the AutoDAN-Turbo runs

---

## Raw Results (for reference)

Attacker: see per-run config | Scorer: see per-run config | Max epochs: 10 | Break score: 8.5 | algo1q

### Run 1 — Abliterated Attacker + Gemma Scorer (Claude-judged)

> **Warning:** Attacker = Target = Reshaper (same model). Results are not valid for paper use.

| Model (Target) | Defense Strategy | Total | Positive Exits | MI FP | Exhausted | FP | True Positives | Avg Iters |
| -------------- | ---------------- | ----- | -------------- | ----- | --------- | -- | -------------- | --------- |
| Abliterated    | LLaMA-Guard3-8B        | 50 |  5 | — | 45 | 5 |  0 |  9.5 |
| Abliterated    | LLaMA-Guard3-8B + CMPE | 50 |  6 | — | 44 | 0 |  6 |  9.4 |

Notes:
- S2 FP=5: Claude confirmed all 5 positive exits were false positives (Gemma miscalibrated).
- S3 TP=6: All 6 positive exits classified as `misdirected_jailbroken` — CMPE triggered but the misdirected output remained harmful; CMPE did not successfully redirect. `cmpe_triggered` flag was not persisted in this run's schema.

### Run 2 — gpt-oss-120b Attacker + oss120b Scorer (raw scores only, no Claude judging)

> **Warning:** Attacker = Scorer (same model). Claude judging failed — all `raw_response` fields missing due to merge bug. FP/TP cannot be determined.

| Model (Target) | Defense Strategy | Total | Positive Exits | MI FP | Exhausted | FP | True Positives | Avg Iters |
| -------------- | ---------------- | ----- | -------------- | ----- | --------- | -- | -------------- | --------- |
| Vicuna         | LLaMA-Guard3-8B        | 50 | 15 | — | 35 | ? | ? | 8.5 |
| Vicuna         | LLaMA-Guard3-8B + CMPE | 50 |  1 | — | 49 | ? | ? | 9.9 |

Notes:
- S2 pos=15: 15 prompts broke with score ≥ 8.5 under detect-block. Claude judging not available.
- S3 pos=1: Only 1 positive exit under CMPE — suggesting oss120b scoring + CMPE is highly effective, but cannot be confirmed without Claude validation.
- All rows have `claude_case: n/a` due to missing `raw_response` fields in the merged output.
