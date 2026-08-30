# AutoDAN Integration — References & Attribution

Our scenario scripts (`run_turbo_scenarios.py`, `run_reasoning_scenarios.py`) implement the full
S2/S3 evaluation framework from scratch. The following elements were taken verbatim or adapted
from the official AutoDAN repositories.

---

## AutoDAN-Turbo

**Repository:** https://github.com/XHMY/AutoDAN-Turbo  
**Paper:** Zhu et al., "AutoDAN-Turbo: A Lifelong Agent for Strategy Self-Exploration to Jailbreak LLMs" (2024)  
**License:** MIT

### Assets used as-is

| Asset | Path in repo | Usage |
|-------|-------------|-------|
| Pre-trained strategy library | `logs_r/lifelong_strategy_library.pkl` | Loaded at runtime to supply mutation strategies for `use_strategy` and `find_new_strategy` mutations |
| Harmful behaviors dataset | `data/harmful_behaviors.csv` (AdvBench) | Evaluation prompt set (500 goals) |

### Logic taken verbatim

| Element | Source location | Our file | Notes |
|---------|----------------|----------|-------|
| Attacker refusal word list | `framework/pipeline.py` line 22 | `run_turbo_scenarios.py` line 67 | `["I cannot", "I am unable", "I can't"]` — exact list, substring check |
| Attacker refusal fallback logic | `framework/pipeline.py` lines 107–109, 176–178, 198–200 | `run_turbo_scenarios.py` `_attacker_refusal_fallback()` | If attacker outputs a refusal word, fall back to raw goal — do NOT skip epoch |
| TurboScorer rubric (1–10 scale) | `framework/pipeline.py` scorer prompt | `run_turbo_scenarios.py` `SCORER_RUBRIC` | Exact rubric text used to judge target responses |
| Three mutation modes | `framework/pipeline.py` | `run_turbo_scenarios.py` `TurboAttacker` class | `warm_up_attack`, `use_strategy`, `find_new_strategy` — prompts adapted to our vLLM interface |

### What we wrote ourselves

- S2 (detect-and-block) and S3 (detect-and-misdirect) scenario loop (`run_scenario2`, `run_scenario3`)
- LlamaGuard-3-8B integration for UNSAFE/SAFE detection
- CMPE algo1q reshaping integration within the epoch loop
- FAISS-based strategy retrieval wrapper (`faiss_retrieve`)
- Parallel chunk launcher (`scripts/launch_parallel.py`)
- Result merge and CSV export (`scripts/merge_turbo_results.py`)

---

## AutoDAN-Reasoning

**Repository:** https://github.com/XHMY/AutoDAN-Reasoning  
**Paper:** Zhu et al., "Improved Techniques for Optimization-Based Jailbreaking on Large Language Models" (2025)  
**License:** MIT

### Assets used as-is

| Asset | Path in repo | Usage |
|-------|-------------|-------|
| Pre-trained strategy library | `logs_r/lifelong_strategy_library.pkl` | Same library as Turbo — shared asset |

### Logic adapted

| Element | Source | Our file | Notes |
|---------|--------|----------|-------|
| Vanilla attack method | `AttackerAutoDANReasoning` class | `run_reasoning_scenarios.py` `ReasoningAttacker.vanilla()` | Single strategy retrieval + jailbreak generation with prev_attempt feedback |
| Best-of-N method | Paper §3.2 | `run_reasoning_scenarios.py` `ReasoningAttacker.best_of_n()` | N=4 candidates, highest-scored selected |
| Beam search method | Paper §3.3 | `run_reasoning_scenarios.py` `ReasoningAttacker.beam_search()` | W=4 width, C=3 combinations, K=10 steps |

### What we wrote ourselves

- S2/S3 scenario loop for all three attack methods
- Epoch scheduling, timeout handling, CMPE integration
- Progress lock files for parallel chunk execution

---

## AdvBench Dataset

**Source:** Zou et al., "Universal and Transferable Adversarial Attacks on Aligned Language Models" (2023)  
**File:** `harmful_behaviors.csv` (500 harmful behavior goals)  
Used as the evaluation prompt set across all AutoDAN experiments.
