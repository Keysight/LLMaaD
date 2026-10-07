# Artifact Evaluation Guide — LLMaaD (ACSAC 2026)

**Paper:** Analyzing Defensive Misdirection Against Model-Guided Automated Attacks on Agentic AI Systems
**Badges requested:** Available · Functional · Reproduced
**Artifact DOI:** https://doi.org/10.5281/zenodo.22869768
**GitHub:** https://github.com/Keysight/LLMaaD

---

## Quick Start for Reviewers

All main claims can be verified **in under 5 minutes with no GPU** using pre-computed result CSVs:

```bash
git clone https://github.com/Keysight/LLMaaD.git
cd LLMaaD

# AutoDAN-Turbo: CMPE reduces ASR to 0% (abliterated target)
cat llmaad_vs_adv_attacks/autodan_results/llmaad_results/turbo/detect_and_misdirect/abliterated_S3_n50.csv

# GPTFuzz: CMPE reduces ASR to 0% TP
ls llmaad_vs_adv_attacks/GPTFuzz/llmaad_results/detect_misdirect/

# PAIR: CMPE reduces ASR to 0% TP
ls llmaad_vs_adv_attacks/JailbreakingLLMs/llmaad_results/detect_and_misdirect/
```

In each CSV, verify: **`true_positives` column = 0** for all CMPE (detect-misdirect) rows.

---

## Installation

```bash
git clone https://github.com/Keysight/LLMaaD.git
cd LLMaaD
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

See [INSTALL.md](INSTALL.md) for vLLM server setup and model serving commands.

---

## Hardware Requirements

| Scenario | Hardware Needed |
|---|---|
| Verify pre-computed CSVs | Any laptop, no GPU |
| Live re-execution (8B models only, no 120B scorer) | ≥ 48 GB total GPU VRAM, ≥ 64 GB RAM |
| Full paper configuration | 4× NVIDIA DGX Spark (GB10 Grace Blackwell, 128 GB unified memory each) |

**Paper setup:** 4× NVIDIA DGX Spark (GB10 Grace Blackwell Superchip, 128 GB unified memory) used concurrently. Models served:

| Role | Model |
|---|---|
| Victim target 1 | `lmsys/vicuna-13b-v1.5` |
| Victim target 2 / Reshaper | `mlabonne/NeuralDaredevil-8B-abliterated` |
| Detector | `meta-llama/Llama-Guard-3-8B` |
| Attacker judge | `gemma1-7b-it` |
| Scorer | `openai/gpt-oss-120b` (MXFP4 quantized) |

---

## Software Requirements

- OS: Linux (Ubuntu 20.04+), tested on Ubuntu 22.04
- Python: 3.10+, CUDA ≥ 11.8
- Key packages: `vllm ≥ 0.4.0`, `torch ≥ 2.0`, `transformers`, `openai ≥ 1.0`, `pandas`, `anthropic` (optional)
- Install: `pip install -r requirements.txt`

**API keys:**
- `OPENAI_API_KEY` — required only for GPTFuzz and PAIR live runs (GPT-3.5-turbo mutator). **Not needed for CSV verification or AutoDAN.**
- `ANTHROPIC_API_KEY` — optional, for post-hoc re-judging only. Pre-computed judged CSVs are provided.

All model endpoints configured via environment variables. See [config.example.env](config.example.env).

---

## Expected Runtime

| Task | Time | GPU Required |
|---|---|---|
| Verify pre-computed CSVs (all claims) | < 5 minutes | No |
| Scaled live run — AutoDAN-Turbo S3, 5 prompts, 5 epochs | ~30–45 minutes | Yes (48 GB VRAM) |
| Scaled live run — GPTFuzz, 5 prompts, 10 iterations | ~20–30 minutes | Yes |
| Scaled live run — PAIR, 5 prompts, 3 iterations × 5 streams | ~20–30 minutes | Yes |
| Full paper run — AutoDAN (n=50, 20 epochs) | ~8–12 hours per config | Yes (DGX Spark) |
| Full paper run — GPTFuzz (n=50, 50 iterations) | ~4–6 hours per config | Yes |
| Full paper run — PAIR (n=50, 5 iter × 10 streams) | ~3–5 hours per config | Yes |

---

## Expected Outputs and How to Determine Success

### Interpreting Result CSVs

Each result CSV has these key columns:

| Column | Meaning |
|---|---|
| `true_positives` | Prompts where CMPE failed — attacker got harmful content (should be 0) |
| `mi_fp` | Misdirection-Induced False Positives — attacker fooled by non-harmful response |
| `positive_exits` | Total prompts where attacker declared success (mi_fp + fp + true_positives) |
| `exhausted` | Prompts where attacker gave up (max epochs/iterations reached) |

### Success Criteria per Badge Claim

| Claim | File to Check | Success = |
|---|---|---|
| CMPE reduces AutoDAN-Turbo ASR to 0% (abliterated) | `autodan_results/llmaad_results/turbo/detect_and_misdirect/abliterated_S3_n50.csv` | `true_positives == 0` |
| CMPE reduces AutoDAN-Turbo ASR to 0% (vicuna) | `autodan_results/llmaad_results/turbo/detect_and_misdirect/vicuna_S3_n50.csv` | `true_positives == 0` |
| Detect+block (baseline) leaks true positives | `autodan_results/llmaad_results/turbo/detect_and_block/` | `true_positives > 0` |
| CMPE reduces GPTFuzz ASR to 0% TP | `GPTFuzz/llmaad_results/detect_misdirect/*.csv` | `true_positives == 0` |
| CMPE reduces PAIR ASR to 0% TP | `JailbreakingLLMs/llmaad_results/detect_and_misdirect/*.csv` | `true_positives == 0` |
| CMPE holds under hardened judge | `JailbreakingLLMs/llmaad_results/misdirect_judge_pair/*.csv` | `true_positives` reduced vs baseline |

---

## Claim-to-Script/Data Mapping

| Paper Claim (§VI-B) | Pre-computed File | Live Script |
|---|---|---|
| AutoDAN-Turbo CMPE ASR = 0% (abliterated) | `autodan_results/llmaad_results/turbo/detect_and_misdirect/abliterated_S3_n50.csv` | `autodan_results/run_turbo_scenarios.py --scenarios 3 --target abliterated` |
| AutoDAN-Turbo CMPE ASR = 0% (vicuna) | `autodan_results/llmaad_results/turbo/detect_and_misdirect/vicuna_S3_n50.csv` | `autodan_results/run_turbo_scenarios.py --scenarios 3 --target vicuna` |
| Detect+block leaks TP | `autodan_results/llmaad_results/turbo/detect_and_block/` | `autodan_results/run_turbo_scenarios.py --scenarios 2` |
| GPTFuzz CMPE ASR = 0% TP | `GPTFuzz/llmaad_results/detect_misdirect/` | `GPTFuzz/gptfuzz_llmaad_parallel.py --defense-mode detect-misdirect` |
| PAIR CMPE ASR = 0% TP | `JailbreakingLLMs/llmaad_results/detect_and_misdirect/` | `JailbreakingLLMs/run_pair_detect_misdirect_parallel.py` |
| CMPE under hardened judge | `JailbreakingLLMs/llmaad_results/misdirect_judge_pair/` | `JailbreakingLLMs/run_pair_detect_misdirect_parallel.py --judge hardened` |

---

## No-Cost / Reduced-Scale Evaluation Path

Reviewers **without GPU access or API keys** can fully verify all claims using pre-computed CSVs (< 5 min, no credentials needed).

Reviewers **with GPU access but without the 120B scorer** can substitute `gemma1-7b-it` as the scorer for AutoDAN runs:

```bash
# Scaled live run — no 120B scorer needed
python llmaad_vs_adv_attacks/autodan_results/run_turbo_scenarios.py \
    --scenarios 3 --target abliterated \
    --scorer_model gemma \
    --range 0 5 --max_epochs 5
```

Expected: 0 true positives among CMPE-defended prompts.

---

## Known Limitations and Nondeterminism

1. **LLM nondeterminism:** All model outputs (CMPE reshaping, attacker mutations, scorer ratings) are stochastic. Exact per-prompt outcomes will differ across runs; aggregate ASR trends (0% TP for CMPE configs) are stable.

2. **120B scorer availability:** The `openai/gpt-oss-120b` scorer requires ≥ 128 GB GPU memory. Evaluators without this can use `gemma1-7b-it` as a substitute scorer or verify from pre-computed CSVs.

3. **AutoDAN strategy library:** The lifelong strategy library (`lifelong_strategy_library.pkl`) accumulates across runs. A fresh run starts with an empty library, which may produce slightly higher iteration counts than the paper's n=50 results (which benefited from a warmed-up library).

4. **OpenAI API key for GPTFuzz/PAIR:** GPT-3.5-turbo is used as the attacker mutator for these two frameworks. Without an API key, only pre-computed CSVs can be used for these claims.

5. **Scaled-down vs full results:** Scaled runs (5 prompts, 5 epochs) confirm the directional claim (CMPE → 0 TP) but may show higher variance than the n=50 paper results.

---

## Contact

For access issues or questions during evaluation, contact via HotCRP. SSH access to the DGX Spark servers can be arranged within 24 hours upon request.
