# Datasets

This directory contains datasets used for training, evaluation, and testing of the LLMaaD defense framework.

## Included Files

### `harmful_behaviors.csv`
A subset of the AdvBench dataset (Zou et al., 2023) containing harmful behavior prompts used to evaluate jailbreak attack success rates.

- **Source:** [AdvBench](https://github.com/llm-attacks/llm-attacks/tree/main/data/advbench) — Zou et al., "Universal and Transferable Adversarial Attacks on Aligned Language Models", arXiv 2023.
- **Usage:** Evaluation of GPTFuzz and PAIR attack frameworks against the LLMaaD defense.

### `harmful_sentences_gen_latest.json`
Pre-generated harmful sentence completions used as seed data for the sentence-generation component of the reshaping pipeline.

- **Usage:** Input to `sentence_gen/generator.py` for bootstrapping misdirection candidates.

## Excluded Files

The following files are excluded from this repository as they contain generated model outputs to harmful prompts:

- `abliterated_resp.json` — Pre-generated responses from the abliterated model to harmful queries (used as Responseland seeds for Algorithm 2). Generate by running the abliterated model against `harmful_behaviors.csv`.
- `toxic_data.json` — Supplementary toxic content dataset. Obtain from the original source or generate using the provided scripts.
