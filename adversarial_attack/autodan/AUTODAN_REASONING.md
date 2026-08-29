# AutoDAN-Reasoning — Attack Strategy Reference

**Paper:** [AutoDAN-Reasoning: Enhancing Strategies Exploration based Jailbreak Attacks with Test-Time Scaling](https://arxiv.org/abs/2510.05379)  
**Repo:** `adversarial_attack/AutoDAN-Reasoning/`

---

## Core Idea

AutoDAN-Reasoning **extends AutoDAN-Turbo** with **test-time scaling** — it takes the pre-trained strategy library from AutoDAN-Turbo-R and applies two scaling methods at inference time to extract more attack power from the library without additional training.

Key insight: vanilla AutoDAN-Turbo samples one strategy and generates one jailbreak prompt. AutoDAN-Reasoning generates **multiple candidates** or explores **strategy combinations**, selecting the best by score. This trades compute for higher attack success rates.

---

## Three Attack Methods

### 1. Vanilla (AutoDAN-Turbo Baseline)
The unmodified AutoDAN-Turbo-R test mode. One strategy retrieved → one prompt generated → scored.

- **Epochs:** up to 150, stops at score ≥ 8.5
- **Per epoch:** retrieve strategy from library based on previous response embedding → generate one prompt → score

### 2. Best-of-N (Algorithm 1)
For each iteration, generate **N candidate prompts** from the retrieved strategies and select the highest-scoring one.

```
For each epoch:
  retrieve strategy_list from library (based on prev response)
  generate N prompts with same strategies [prev_attempt feedback included]
  send each to target → score each
  keep the best (highest score)
  if best_score >= 8.5: stop
```

**Parameters:** `--best_of_n N` (default 4)  
**Gain vs vanilla:** More chances to generate a successful prompt per epoch without exploring new strategies.

### 3. Beam Search (Algorithm 2)
Explores **combinations of strategies** from a larger pool of K, maintaining a beam of W most promising combinations and expanding to depth C.

```
For each epoch (j > 0):
  retrieve top-K strategies (return_all=True, no score filter)
  
  # Initialize beam with top-W single strategies
  beam = [evaluate(combo=[strategy_i]) for i in 0..W-1]
  
  # Expand beam iteratively up to depth C
  for depth in 2..C:
    candidates = []
    for each beam_entry:
      for each unused_strategy in strategy_pool:
        new_combo = beam_entry.combo + [strategy]
        candidates.append(evaluate(new_combo))
    beam = top-W(candidates by score)
  
  best = max(beam by score)
  use best.prompt, best.score
  if best.score >= 8.5: stop
```

**Parameters:**
- `--beam_width W` (default 4) — beam size at each depth
- `--beam_depth C` (default 3) — max combination length
- `--beam_k K` (default 10) — strategy pool size

**Gain vs vanilla:** Discovers synergistic strategy combinations that individually fail but together succeed. Achieves up to **+15.6 pp ASR** on Llama-3.1-70B, ~60% relative improvement on GPT-o4-mini.

---

## Enhanced Attacker with Previous Attempt Feedback

AutoDAN-Reasoning introduces `AttackerAutoDANReasoning` which augments every strategy-based attack with the **previous attempt's results**:

```python
prev_attempt = {
    'prompt': previous_jailbreak_prompt,
    'response': target_response,
    'score': numerical_score,
}
```

The attacker is instructed to:
- **Score < 5:** Try a significantly different approach
- **Score 5–8:** Refine and improve the previous prompt
- **Score > 8:** Make subtle improvements

This creates an **iterative optimization loop** on top of strategy retrieval.

---

## Framework Architecture

```
framework/              # Inherited from AutoDAN-Turbo (unchanged)
  attacker.py           # Original conditional-generate attacker (for baseline)
  scorer.py             # Scorer with 1-10 rubric
  summarizer.py         # Strategy extraction from prompt pairs
  retrival.py           # FAISS search — extended with return_all=True for beam
  library.py            # Strategy storage

framework_r/            # AutoDAN-Turbo-R (DeepSeek-R1 based)
  attacker_reasoning_model.py  # Attacker using generate(system, user) interface
  scorer_reasoning_model.py    # Scorer using generate(system, user) interface
  summarizer_reasoning_model.py

framework_autodan_reasoning/   # Test-time scaling (NEW in AutoDAN-Reasoning)
  attacker_autodan_reasoning.py  # Enhanced attacker with prev_attempt feedback
  attacker_best_of_n.py          # Best-of-N wrapper
  attacker_beam_search.py        # Beam Search wrapper

pipeline_autodan_reasoning.py   # AutoDANReasoning class (extends pipeline.py)
```

---

## Key Differences vs AutoDAN-Turbo

| Aspect | AutoDAN-Turbo | AutoDAN-Reasoning |
|--------|---------------|-------------------|
| Strategy generation | Conditional generation (prefix-forced) | Direct generation (no prefix forcing) |
| Attacker feedback | None | Previous prompt/response/score |
| Attack candidates | 1 per epoch | N (Best-of-N) or W×depth (Beam Search) |
| Strategy selection | Sequential retrieval | Combinatorial beam search |
| Training required | Yes (warm-up + lifelong) | No (uses pre-trained library) |
| Primary model | HuggingFace models | DeepSeek-R1 / any chat model |

---

## Retrieval Extension for Beam Search

The `Retrieval.pop` method in AutoDAN-Reasoning adds a `return_all=True` flag:
- `return_all=False` (vanilla): Score-filtered selection (≥5 → single, 2–5 → up to k, <2 → avoid list)
- `return_all=True` (beam search): Returns raw top-K by embedding similarity, no score filtering

This gives beam search access to a larger unfiltered pool to explore combinations from.

---

## Pre-trained Strategy Libraries

Located at `logs_r/` in both repos:
- `warm_up_strategy_library.pkl` — ~20 strategies from warm-up
- `lifelong_strategy_library.pkl` — 30+ strategies from full training

AutoDAN-Reasoning's `logs_r/` uses the same format as AutoDAN-Turbo-R. Both are valid starting points for test-time scaling evaluation.

---

## CMPE Defense Target

AutoDAN-Reasoning is a **harder attacker** than vanilla AutoDAN-Turbo for our CMPE defense:

- **Best-of-N** means CMPE must fool the scorer N times in the same epoch to prevent any candidate from succeeding
- **Beam Search** explores strategy combinations — CMPE must misdirect across diverse multi-strategy prompts
- **Prev-attempt feedback** means the attacker can adapt if CMPE partially succeeds

These make the scorer-confusion loop more robust and harder to consistently fool with a single static misdirection strategy.

See `results_against_attacks/autodan_results/eval_autodan.py` for the integrated evaluation against all three methods.
