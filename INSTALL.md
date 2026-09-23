# Installation Guide

## Requirements

See [REQUIREMENTS.md](REQUIREMENTS.md) for full hardware and software specifications.

## 1. Clone the Repository

```bash
git clone https://github.com/Keysight/LLMaaD.git
cd LLMaaD
```

## 2. Set Up Python Environment

Python 3.10 or later is required.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 3. Configure Model Endpoints

Copy the example config and fill in your vLLM server IPs:

```bash
cp config.example.env .env
# Edit .env with your server addresses
source .env
```

All model IPs and ports are read from environment variables — no hardcoded addresses.

## 4. Start vLLM Model Servers

Launch each model on a separate GPU node. Example commands:

```bash
# Victim target — Vicuna-13B
python -m vllm.entrypoints.openai.api_server \
    --model lmsys/vicuna-13b-v1.5 \
    --host 0.0.0.0 --port 8000

# Victim target — NeuralDaredevil-8B-abliterated (also used as reshaper)
python -m vllm.entrypoints.openai.api_server \
    --model mlabonne/NeuralDaredevil-8B-abliterated \
    --host 0.0.0.0 --port 8000

# Detector — Llama-Guard-3-8B
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-Guard-3-8B \
    --host 0.0.0.0 --port 8001

# Attacker judge — Gemma-7B
python -m vllm.entrypoints.openai.api_server \
    --model gemma1-7b-it \
    --host 0.0.0.0 --port 8001

# Scorer — gpt-oss-120b (MXFP4 quantized)
python -m vllm.entrypoints.openai.api_server \
    --model openai/gpt-oss-120b \
    --quantization mxfp4 \
    --host 0.0.0.0 --port 8000
```

## 5. Smoke Test

Verify a model endpoint is live:

```bash
curl http://$ABLITERATED_IP:$ABLITERATED_PORT/v1/models
```

## 6. Functional Verification (No GPU Required)

All main claims can be verified from pre-computed result CSVs in under 5 minutes — no model servers needed:

```bash
# Inspect AutoDAN-Turbo results
cat llmaad_vs_adv_attacks/autodan_results/final_results/turbo/abliterated_S3_algo1q_n50_claude_judged.csv

# Inspect GPTFuzz results
ls llmaad_vs_adv_attacks/GPTFuzz/llmaad_results/detect_misdirect/

# Inspect PAIR results
ls llmaad_vs_adv_attacks/JailbreakingLLMs/llmaad_results/detect_and_misdirect/
```

See [README.md](README.md) for full experiment reproduction commands.
