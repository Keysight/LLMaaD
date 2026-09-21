# Hardware and Software Requirements

## Hardware

### Paper Evaluation Environment

| Component | Specification |
|---|---|
| System | NVIDIA DGX Spark |
| Architecture | NVIDIA GB10 Grace Blackwell Superchip |
| Unified Memory | 128 GB (CPU + GPU shared, NVLink-C2C interconnect) |
| OS | Linux (Ubuntu 22.04) |
| Storage | 100 GB+ for Hugging Face model weights |

All LLM inference was performed via vLLM on the DGX Spark's unified memory, serving victim, detector, reshaper, and scorer models concurrently.

### Minimum for Live Re-execution (scaled experiments)

To concurrently serve the three 8B-class models (victim + detector + CMPE reshaper) via vLLM:

- **GPU**: ≥ 48 GB total VRAM (e.g., 2× NVIDIA A100 40 GB, or 1× H100 80 GB, or equivalent)
- **System RAM**: ≥ 64 GB
- **Storage**: ≥ 100 GB (Hugging Face model weights)
- **OS**: Linux (Ubuntu 20.04+)

> The 120B scorer (`openai/gpt-oss-120b`) used in AutoDAN experiments requires ≥ 128 GB unified GPU memory. Evaluators who cannot serve this model can substitute a smaller scorer or use the pre-computed result CSVs — all main claims are verifiable from the pre-computed files.

### Functional Verification (no re-execution)

Pre-computed result CSVs are included in the repository under `llmaad_vs_adv_attacks/*/llmaad_results/` and `llmaad_vs_adv_attacks/autodan_results/final_results/`. **No GPU is required** — any laptop can verify the results in under 5 minutes by inspecting these files.

---

## Software

### Core Environment

| Requirement | Version |
|---|---|
| Python | 3.10 or later |
| OS | Linux (Ubuntu 20.04+) |
| vLLM | ≥ 0.4.0 |
| PyTorch | ≥ 2.0 (with CUDA) |
| CUDA Toolkit | ≥ 11.8 |

### Python Dependencies

Install all dependencies with:

```bash
pip install -r requirements.txt
```

Key packages:

| Package | Purpose |
|---|---|
| `vllm` | Local LLM serving (victim, detector, reshaper, scorer) |
| `torch` | Deep learning backend |
| `transformers` | Hugging Face model loading |
| `accelerate` | Multi-GPU model distribution |
| `openai` | API client (attacker mutator + gpt-oss scorer) |
| `pandas` | Result CSV handling |
| `numpy` | Numerical utilities |
| `tqdm` | Progress bars |
| `anthropic` | Optional — post-hoc Claude judging only |

All attack framework dependencies (AutoDAN, GPTFuzz, PAIR) are vendored under `llmaad_vs_adv_attacks/` with their original licenses.

### API Keys

| Key | Required For | Can Skip? |
|---|---|---|
| `OPENAI_API_KEY` | GPTFuzz and PAIR attacker mutator (GPT-3.5-turbo) | Yes — not needed for AutoDAN or pre-computed CSV verification |
| `ANTHROPIC_API_KEY` | Post-hoc Claude judging (`claude_judge_autodan.py`) | Yes — pre-computed Claude-judged CSVs are provided |

Set keys in your environment before running:

```bash
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=...
```

Or copy `config.example.env` to `.env` and source it:

```bash
cp config.example.env .env
# Edit .env with your values
source .env
```

### Model Endpoint Configuration

All model server IPs and ports are read from environment variables (no hardcoded addresses). Set these to match your vLLM server deployment:

```bash
export ABLITERATED_IP=<server-ip>   # NeuralDaredevil-8B reshaper
export ABLITERATED_PORT=8000
export VICUNA_IP=<server-ip>        # Vicuna-13B victim
export VICUNA_PORT=8000
export LLAMAGUARD_IP=<server-ip>    # Llama-Guard-3-8B detector
export LLAMAGUARD_PORT=8001
export GEMMA_IP=<server-ip>         # Gemma-7B attacker judge
export GEMMA_PORT=8001
export SCORER_IP=<server-ip>        # gpt-oss-120b scorer
export SCORER_PORT=8000
```

See `config.example.env` at the repo root for a full template.
