# Merge Plan: acsac_dgx2 → acsac (Final Paper Version)

## Context Summary

| Repo | State | Last touched |
|------|-------|-------------|
| `acsac` | Active main repo. Has AutoDAN (Turbo+Reasoning, n=100, fully Claude-normalized), PAIR, GPTFuzz, hardened-judge variants, full `RESULTS_TRACKER.md`, `claude_judge_autodan.py` (the misdirection judge), `patch_algo1q` infrastructure. | Aug 25–26, 2026 |
| `acsac_dgx2` | Earlier snapshot + DGX2 experiment outputs. Unique asset: `all_results/algo1q/v5_results/` — 7 JSON files from algo1q runs on 50 prompts. Does NOT have `claude_judge_autodan.py` or any AutoDAN infrastructure. | May 22–26, 2026 |

The "new misdirection judge" is `claude_judge_autodan.py` (already in `acsac`). The "algo1q 50p results" are the 7 JSONs in `dgx2/all_results/algo1q/v5_results/` — run on DGX2 but never normalized or incorporated into the main tracker.

---

## Phase 1 — Import dgx2's unique algo1q results

### 1a. Copy v5_results into acsac

- **Source:** `acsac_dgx2/llmaad/all_results/algo1q/v5_results/*.json` (7 files)
- **Destination:** `acsac/all_results/algo1q/v5_results/` (directory doesn't exist yet in acsac)

### 1b. Inspect the 7 JSONs to determine schema

- If structured attack results (with `raw_response`, `score`, `was_unsafe`, etc.) → compatible with `claude_judge_autodan.py` directly
- If raw reshape artifacts (intermediate CMPE computation) → need to be wrapped in the attack result schema first, or documented as ablation data
- **Action:** read one file's keys/structure before proceeding

### 1c. Run `claude_judge_autodan.py` on v5_results

- Produces `v5_results_claude_judged.json`
- Classifies each prompt: `misdirected_fp` / `misdirected_wrong` / `jailbroken` / `false_positive`
- Output feeds MI-FP / FP / TP / ASR columns in the tracker

---

## Phase 2 — Bring in dgx2 scripts missing from acsac

Three scripts exist in dgx2 but not in acsac. Assess each:

| Script | Location in dgx2 | Decision |
|--------|-----------------|----------|
| `adapt_25p_schema.py` | `llmaad_vs_adv_attacks/JailbreakingLLMs/` | Copy in — schema adapter, useful reference |
| `run_pair_baseline_parallel.py` | same | Copy in — baseline variant not present in acsac |
| `run_pair_detect_misdirect_algo1_parallel.py` | same | Likely superseded by acsac's `run_pair_detect_misdirect_parallel.py` — compare before deciding; may trash |

---

## Phase 3 — Keep acsac versions for all modified files (no merges needed)

These files differ between repos but **acsac's version is strictly better**:

| File | Why acsac wins |
|------|----------------|
| `prompt_reshaping/algos/algo1.py` | Cleaner (no `log_action` noise), updated expansion prompt API |
| `prompt_reshaping/llm_gen/clients.py` | Critical Vicuna routing fix (`_vicuna_completions`) that dgx2 lacks |
| `prompt_reshaping/llm_gen/model_pick.py` | Adds vicuna/qwen3/gemma/llamaguard endpoints |
| `prompt_reshaping/detectors/detector_gen.py` | Adds `LlamaGuardDetector` class, offline mode |
| PAIR integration scripts | acsac has hardened-judge variants dgx2 doesn't |

`prompt_reshaping/artifacts/logging.py` is identical between repos — no action needed.

---

## Phase 4 — Documentation updates

### 4a. `results_against_attacks/autodan_results/run_results/RESULTS_TRACKER.md`

Add new section **"AutoDAN + algo1q (DGX2 runs)"** with 50p results once Claude-normalized. Format: same table schema as existing runs A–H, labelled Run I or similar. Expected outcome: ASR=0.000 for CMPE (consistent with algo1 results), reinforcing the paper's claim across both algo variants.

### 4b. `CLAUDE.md`

- Add algo1q row to the current n=50 results table
- Update scenario architecture note to mention algo1q validated separately on DGX2

### 4c. `README.md`

- Add algo1q results row to the main results table
- Note: "algo1q evaluated on DGX2 (n=50)" in the evaluation section

### 4d. `results_against_attacks/autodan_results/summary_turbo_reasoning.md`

Covers algo1 runs only — either extend to algo1q here or add a parallel `summary_algo1q.md`.

### 4e. `paper/experiment_results.tex`

Add algo1q data to the relevant LaTeX table. Requires normalized numbers from Phase 1c first.

---

## Phase 5 — Paper-readiness checklist

| Item | Status |
|------|--------|
| AutoDAN Turbo n=100 (algo1) | ✅ Claude-normalized, ASR=0.000 for CMPE |
| AutoDAN Reasoning n=100 (algo1) | ✅ Claude-normalized, ASR=0.000 for CMPE |
| PAIR 50p (algo1, both models) | ✅ Claude-judged, all 4 configs |
| GPTFuzz 50p (algo1, both models) | ✅ Claude-judged, all 4 configs |
| algo1q 50p (DGX2 runs) | ⏳ Pending — Phase 1 above |
| Hardened judge PAIR variants | Results in run logs — check if normalization needed |
| LaTeX tables updated | ⏳ After algo1q normalization |
| `final_results.md` synced | ✅ Identical between both repos |

---

## Execution Order

```
1. cp dgx2 v5_results → acsac/all_results/algo1q/v5_results/
2. Read one v5 JSON → confirm schema
3. Run claude_judge_autodan.py on them → produce _claude_judged.json
4. Update RESULTS_TRACKER.md with algo1q numbers
5. Copy the 2–3 unique dgx2 scripts → assess/trash run_pair_detect_misdirect_algo1_parallel.py
6. Update README.md + CLAUDE.md + experiment_results.tex
7. Final paper submission check
```
