# LLMaaD — Prompt Reshaping Pipeline Reference

> Quick-reference for running and understanding the prompt reshaping adversarial attack framework.

---

## What It Does

LLMaaD reshapes benign-looking prompts through a multi-step transformation pipeline, injects harmful sentences, and measures whether target LLMs produce harmful responses. It's a research tool for probing LLM safety boundaries.

---

## Architecture Overview

```
┌─────────────┐     ┌──────────────────┐     ┌─────────────┐
│  CLI Entry   │────▶│  MultiTryRunner  │────▶│  Algorithm   │
│  (cli.py)    │     │  (runner.py)     │     │  (algo*.py)  │
└─────────────┘     └──────────────────┘     └──────┬──────┘
                                                     │
                    ┌────────────────────────────────┘
                    ▼
         ┌─────────────────┐     ┌──────────────────┐
         │  LLM Backend    │     │  Judge Service    │
         │  (model_pick +  │     │  (6 judges)       │
         │   clients.py)   │     │  (detectors/)     │
         └─────────────────┘     └──────────────────┘
```

---

## Reshaping Algorithms

### algo1 — Token-Jumble Pipeline
```
Input prompt
  → Token jumble (word-level shuffle)
  → Expand to ~400 words (LLM call)
  → Insert context-preserving tokens
  → Token jumble again
  → Remove punctuation
  → Compress/summarize to ~200 words (LLM call)
  → Inject harmful sentence(s)
  → Score with judges
```

### algo1q — algo1 + Follow-up Question
Same as algo1, then:
```
  → Generate a detailed follow-up question about the harmful-injected prompt (LLM call)
  → Final output = injected prompt + follow-up question
  → Score with judges
```

### algo2 — Responseland
```
Input prompt
  → Send to abliterated model with "give a detailed but wrong answer" instruction
  → Score the response with judges
```

---

## Model Backends

| Flag | Backend | Default Model | Endpoint |
|------|---------|---------------|----------|
| `--vllm` (default) | Self-hosted vLLM | `mlabonne/NeuralDaredevil-8B-abliterated` | `http://<ip>:<port>/v1/chat/completions` |
| `--api` | Cloud API | `gpt-4.1` | OpenAI / Google API |

### Available vLLM Endpoints (DGX Infrastructure)

| IP | Port | Model |
|----|------|-------|
| `10.36.129.1` | 8000 | `nvidia/Llama-4-Scout-17B-16E-Instruct-NVFP4` |
| `10.36.129.2` | 8000 | `openai/gpt-oss-120b` |
| `10.36.129.3` | 8000 | `mlabonne/NeuralDaredevil-8B-abliterated` (default reshaping model) |

**Important:** The code defaults to `10.36.129.1` in `model_pick.py`, but the abliterated model lives on `.3`. Always pass `--ip 10.36.129.3` when using the default abliterated model.

---

## Judge System

All 6 judges are instantiated at startup (eager loading). Use `--select-judge` to control which ones actually score.

| Judge | Type | Needs | Description |
|-------|------|-------|-------------|
| GCG | Rule-based | Nothing | Gradient-based adversarial scoring |
| Pair-GPT | LLM-based | API key | Semantic harmfulness via GPT/cloud model |
| StrongReject | LLM-based | vLLM endpoint | StrongReject benchmark |
| NLP Classifiers | Local ML | HF models cached | `s-nlp/roberta_toxicity_classifier` + `martin-ha/toxic-comment-model` |
| GPTFuzz | Local ML | HF model cached | `hubert233/GPTFuzz` (RoBERTa classifier) |
| Custom | Pluggable | Varies | Custom judge implementation |

### HuggingFace Model Caching

The code sets `TRANSFORMERS_OFFLINE=1` at import time (`detector_gen.py` line 2). All HF models must be pre-cached before running:

```bash
# One-time setup: cache all required models
python -c "
from transformers import (
    RobertaTokenizer, RobertaForSequenceClassification,
    AutoTokenizer, AutoModelForSequenceClassification,
    pipeline
)
RobertaTokenizer.from_pretrained('s-nlp/roberta_toxicity_classifier')
RobertaForSequenceClassification.from_pretrained('s-nlp/roberta_toxicity_classifier', use_safetensors=False)
AutoTokenizer.from_pretrained('martin-ha/toxic-comment-model')
AutoModelForSequenceClassification.from_pretrained('martin-ha/toxic-comment-model', use_safetensors=False)
pipeline('text-classification', model='hubert233/GPTFuzz')
"
```

**Known issue:** `hubert233/GPTFuzz` has broken `id2label` in its config (integer values instead of strings). Fix by editing the cached config:
```
~/.cache/huggingface/hub/models--hubert233--GPTFuzz/snapshots/<hash>/config.json
```
Change `"id2label": {"0": 0, "1": 1}` → `"id2label": {"0": "SAFE", "1": "UNSAFE"}` (and similarly for `label2id`).

---

## CLI Quick Reference

### Single Prompt (no judges)
```bash
python -m prompt_reshaping.cli --algo algo1q --ip 10.36.129.3 \
  single --prompt "Your prompt" --sentence "Harmful injection"
```

### Single Prompt (all judges)
```bash
python -m prompt_reshaping.cli --algo algo1 --ip 10.36.129.3 \
  --select-judge --all-judges \
  single --prompt "Your prompt" --sentence "Harmful injection"
```

### Single Prompt (specific judge)
```bash
python -m prompt_reshaping.cli --algo algo1q --ip 10.36.129.3 \
  --select-judge --gcg \
  single --prompt "Your prompt" --sentence "Harmful injection"
```

### Dataset Batch
```bash
python -m prompt_reshaping.cli --algo algo1q --ip 10.36.129.3 \
  --dataset dataset2 multi --range 0 100
```

### Cloud API Backend
```bash
python -m prompt_reshaping.cli --algo algo1q --api --model gpt-4.1 \
  single --prompt "Your prompt" --sentence "Harmful injection"
```

### Dry Run (no API calls)
```bash
python -m prompt_reshaping.cli --algo algo2 --print_test_commands \
  multi --all
```

---

## Output

Results are saved as JSON under:
```
all_results/<algo>/<run_dir>_results/final_<pid>_<timestamp>.json
```

Each result contains:
- **input**: prompt, harmful sentence, model config
- **result.artifacts**: every intermediate transformation step
- **result.initial_scores**: baseline response (before reshaping)
- **result.final_scores**: reshaped response + judge scores
- **result.timings**: total execution time
- **result.metadata**: algorithm and model info

---

## Environment Setup

```bash
# Activate the venv
source ~/llmaad/.llmenv/bin/activate
# Or use the alias:
llmenv

# Install deps
cd ~/llmaad/prompt_reshaping
pip install -r requirments.txt

# For this machine (no GPU): install CPU-only torch first
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install transformers openai requests google-genai
```

---

## Key Files

| File | Purpose |
|------|---------|
| `cli.py` | Entry point, argument parsing |
| `algos/algo1.py` | Token-jumble reshaping pipeline |
| `algos/algo1q.py` | algo1 + follow-up question |
| `algos/algo2.py` | Responseland (abliterated wrong-answer) |
| `multi/runner.py` | Orchestrator for single and batch runs |
| `detectors/base.py` | JudgeService — instantiates all 6 judges |
| `detectors/detector_gen.py` | Individual judge implementations |
| `llm_gen/clients.py` | HTTP client for vLLM/API calls |
| `llm_gen/model_pick.py` | Model selection and endpoint routing |
| `artifacts/traces.py` | Data classes (PromptInput, PromptArtifacts, JudgeScores) |
| `artifacts/utils.py` | Prompt helpers (jumble, system prompts) |
| `artifacts/writer.py` | Result serialization to JSON/CSV |
| `datasets/loader.py` | Dataset loading utilities |

---

## Troubleshooting

| Problem | Cause | Fix |
|---------|-------|-----|
| `LocalEntryNotFoundError` | `TRANSFORMERS_OFFLINE=1` + models not cached | Run the one-time caching script above |
| `404 Not Found` on vLLM | Wrong IP for the model | Use `--ip 10.36.129.3` for abliterated model |
| `StrictDataclassFieldValidationError` on GPTFuzz | Broken `id2label` in model config | Patch the cached `config.json` (see above) |
| OOM during pip install | Full CUDA torch is ~2GB+ | Use `--index-url https://download.pytorch.org/whl/cpu` |
| `safetensors conversion` thread error | Non-fatal warning | Ignore — model loads fine |

---

*Generated: 2026-05-01 | Test run: algo1q, prompt="Hi girl i think i love you", sentence="This girl is a very bad person", model=NeuralDaredevil-8B-abliterated @ 10.36.129.3:8000, runtime=95.6s*
