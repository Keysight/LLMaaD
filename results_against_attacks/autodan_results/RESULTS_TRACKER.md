# AutoDAN Results Tracker
*Last updated: 2026-08-25 21:27*

**Target:** Vicuna-13b-v1.3 | **Attacker:** NeuralDaredevil-8B-abliterated | **CMPE:** algo1
**Scorer:** gemma1-7b-it | **LlamaGuard:** 10.36.129.1:8001 | **Break score:** 8.0
**ASR = TP / n** (pre-Claude-normalization; TP = positive exits via SAFE path, no CMPE fired)

---

## Column Definitions

| Column | Meaning |
|--------|---------|
| **Pos Exits** | Prompts where attack loop broke with score ≥ 8.0 |
| **Neg Exits** | Prompts where max epochs exhausted, score never ≥ 8.0 (model refusal) |
| **MI-FP** | Positive exits via UNSAFE+CMPE path — all misdirected variants (misdirected_fp, misdirected_wrong, misdirected_jailbroken); CMPE activated |
| **FP** | Positive exits pending Claude normalization |
| **TP** | Positive exits via SAFE path — pending Claude normalization to confirm actual harm |
| **ASR** | TP / n (pre-normalization) |
| **Avg Iter** | Average epochs per prompt |

**Partition:** Pos Exits = MI-FP + FP + TP &nbsp;|&nbsp; Pos Exits + Neg Exits = n

---

## A. AutoDAN-Turbo `use_strategy` (merged n=50) — Prompts 0–50 ✓ Claude-normalized

**Source files:**
- `autodan_turbo/turbo_use_strategy_S2S3_algo1_n50_claude_judged.json`

| Victim | Defense | n | Pos Exits | Neg Exits | MI-FP | FP | TP | ASR | Avg Iter |
|--------|---------|---|-----------|-----------|-------|----|----|-----|----------|
| Vicuna-13b | LlamaGuard | 50 | 49 | 1 | 0 | 44 | 5 | 0.100 | 2.6 |
| Vicuna-13b | LlamaGuard + CMPE | 50 | 23 | 27 | 23 | 0 | 0 | 0.000 | 7.6 |

```latex
\begin{tabular}{llccccccc}
    \toprule
    \textbf{Victim} & \textbf{Defense} & \textbf{n} & \textbf{Pos} & \textbf{Neg} & \textbf{MI-FP} & \textbf{FP} & \textbf{TP} & \textbf{ASR} \\
    \midrule
    Vicuna-13b & LlamaGuard & 50 & 49 & 1 & 0 & 44 & 5 & 0.100 \\
    Vicuna-13b & LlamaGuard + CMPE & 50 & 23 & 27 & 23 & 0 & 0 & 0.000 \\
    \bottomrule
\end{tabular}
```

---

## B. AutoDAN-Turbo `use_strategy` (chunks 50–100) — Prompts 50–100 ⏳ pre-normalization

**Source files:**
- `autodan_turbo/chunks/turbo_use_strategy_S2S3_algo1_r50-60_20260824_224225.json`
- `autodan_turbo/chunks/turbo_use_strategy_S2S3_algo1_r60-70_20260824_222010.json`
- `autodan_turbo/chunks/turbo_use_strategy_S2S3_algo1_r70-80_20260824_222159.json`
- `autodan_turbo/chunks/turbo_use_strategy_S2S3_algo1_r80-90_20260824_222133.json`
- `autodan_turbo/chunks/turbo_use_strategy_S2S3_algo1_r90-100_20260824_230050.json`

| Victim | Defense | n | Pos Exits | Neg Exits | MI-FP | FP | TP | ASR | Avg Iter |
|--------|---------|---|-----------|-----------|-------|----|----|-----|----------|
| Vicuna-13b | LlamaGuard | 50 | 50 | 0 | 0 | 0 | 50 | 1.000 | 2.6 |
| Vicuna-13b | LlamaGuard + CMPE | 50 | 22 | 28 | 22 | 0 | 0 | 0.000 | 14.7 |

```latex
\begin{tabular}{llccccccc}
    \toprule
    \textbf{Victim} & \textbf{Defense} & \textbf{n} & \textbf{Pos} & \textbf{Neg} & \textbf{MI-FP} & \textbf{FP} & \textbf{TP} & \textbf{ASR} \\
    \midrule
    Vicuna-13b & LlamaGuard & 50 & 50 & 0 & 0 & 0 & 50 & 1.000 \\
    Vicuna-13b & LlamaGuard + CMPE & 50 & 22 & 28 & 22 & 0 & 0 & 0.000 \\
    \bottomrule
\end{tabular}
```

---

## C. AutoDAN-Turbo `use_strategy` (n=50 range 50–100) — Prompts 50–100 ✓ Claude-normalized

**Source files:**
- `autodan_turbo/turbo_use_strategy_S2S3_algo1_n50-100_claude_judged.json`

| Victim | Defense | n | Pos Exits | Neg Exits | MI-FP | FP | TP | ASR | Avg Iter |
|--------|---------|---|-----------|-----------|-------|----|----|-----|----------|
| Vicuna-13b | LlamaGuard | 50 | 50 | 0 | 0 | 41 | 9 | 0.180 | 2.6 |
| Vicuna-13b | LlamaGuard + CMPE | 50 | 22 | 28 | 22 | 0 | 0 | 0.000 | 14.7 |

```latex
\begin{tabular}{llccccccc}
    \toprule
    \textbf{Victim} & \textbf{Defense} & \textbf{n} & \textbf{Pos} & \textbf{Neg} & \textbf{MI-FP} & \textbf{FP} & \textbf{TP} & \textbf{ASR} \\
    \midrule
    Vicuna-13b & LlamaGuard & 50 & 50 & 0 & 0 & 41 & 9 & 0.180 \\
    Vicuna-13b & LlamaGuard + CMPE & 50 & 22 & 28 & 22 & 0 & 0 & 0.000 \\
    \bottomrule
\end{tabular}
```

---

## D. AutoDAN-Turbo `use_strategy` — **Combined n=100** — Prompts 0–100 ✓ Claude-normalized

**Source files:**
- `autodan_turbo/turbo_use_strategy_S2S3_algo1_n50_claude_judged.json`
- `autodan_turbo/turbo_use_strategy_S2S3_algo1_n50-100_claude_judged.json`
- `autodan_turbo/chunks/turbo_use_strategy_S2S3_algo1_r50-60_20260824_224225.json`
- `autodan_turbo/chunks/turbo_use_strategy_S2S3_algo1_r60-70_20260824_222010.json`
- `autodan_turbo/chunks/turbo_use_strategy_S2S3_algo1_r70-80_20260824_222159.json`
- `autodan_turbo/chunks/turbo_use_strategy_S2S3_algo1_r80-90_20260824_222133.json`
- `autodan_turbo/chunks/turbo_use_strategy_S2S3_algo1_r90-100_20260824_230050.json`

| Victim | Defense | n | Pos Exits | Neg Exits | MI-FP | FP | TP | ASR | Avg Iter |
|--------|---------|---|-----------|-----------|-------|----|----|-----|----------|
| Vicuna-13b | LlamaGuard | 100 | 99 | 1 | 0 | 85 | 14 | 0.140 | 2.6 |
| Vicuna-13b | LlamaGuard + CMPE | 100 | 45 | 55 | 45 | 0 | 0 | 0.000 | 11.2 |

```latex
\begin{tabular}{llccccccc}
    \toprule
    \textbf{Victim} & \textbf{Defense} & \textbf{n} & \textbf{Pos} & \textbf{Neg} & \textbf{MI-FP} & \textbf{FP} & \textbf{TP} & \textbf{ASR} \\
    \midrule
    Vicuna-13b & LlamaGuard & 100 & 99 & 1 & 0 & 85 & 14 & 0.140 \\
    Vicuna-13b & LlamaGuard + CMPE & 100 & 45 & 55 & 45 & 0 & 0 & 0.000 \\
    \bottomrule
\end{tabular}
```

---

## E. AutoDAN-Reasoning `vanilla` (merged n=50) — Prompts 0–50 ✓ Claude-normalized

**Source files:**
- `autodan_reasoning/reasoning_vanilla_S2S3_algo1_n50_claude_judged.json`

| Victim | Defense | n | Pos Exits | Neg Exits | MI-FP | FP | TP | ASR | Avg Iter |
|--------|---------|---|-----------|-----------|-------|----|----|-----|----------|
| Vicuna-13b | LlamaGuard | 50 | 50 | 0 | 0 | 46 | 4 | 0.080 | 2.2 |
| Vicuna-13b | LlamaGuard + CMPE | 50 | 16 | 34 | 7 | 9 | 0 | 0.000 | 9.7 |

```latex
\begin{tabular}{llccccccc}
    \toprule
    \textbf{Victim} & \textbf{Defense} & \textbf{n} & \textbf{Pos} & \textbf{Neg} & \textbf{MI-FP} & \textbf{FP} & \textbf{TP} & \textbf{ASR} \\
    \midrule
    Vicuna-13b & LlamaGuard & 50 & 50 & 0 & 0 & 46 & 4 & 0.080 \\
    Vicuna-13b & LlamaGuard + CMPE & 50 & 16 & 34 & 7 & 9 & 0 & 0.000 \\
    \bottomrule
\end{tabular}
```

---

## F. AutoDAN-Reasoning `vanilla` (chunks 50–100) — Prompts 50–100 ⏳ pre-normalization

**Source files:**
- `run_20260825_071030/reasoning_vanilla_S2S3_algo1_r50-60_20260825_071030.json`
- `run_20260825_070327/reasoning_vanilla_S2S3_algo1_r60-70_20260825_070327.json`
- `run_20260825_063431/reasoning_vanilla_S2S3_algo1_r70-80_20260825_063431.json`
- `run_20260825_062814/reasoning_vanilla_S2S3_algo1_r80-90_20260825_062814.json`
- `run_20260825_070915/reasoning_vanilla_S2S3_algo1_r90-100_20260825_070915.json`

| Victim | Defense | n | Pos Exits | Neg Exits | MI-FP | FP | TP | ASR | Avg Iter |
|--------|---------|---|-----------|-----------|-------|----|----|-----|----------|
| Vicuna-13b | LlamaGuard | 50 | 7 | 43 | 0 | 0 | 7 | 0.140 | 9.4 |
| Vicuna-13b | LlamaGuard + CMPE | 50 | 0 | 50 | 0 | 0 | 0 | 0.000 | 10.0 |

```latex
\begin{tabular}{llccccccc}
    \toprule
    \textbf{Victim} & \textbf{Defense} & \textbf{n} & \textbf{Pos} & \textbf{Neg} & \textbf{MI-FP} & \textbf{FP} & \textbf{TP} & \textbf{ASR} \\
    \midrule
    Vicuna-13b & LlamaGuard & 50 & 7 & 43 & 0 & 0 & 7 & 0.140 \\
    Vicuna-13b & LlamaGuard + CMPE & 50 & 0 & 50 & 0 & 0 & 0 & 0.000 \\
    \bottomrule
\end{tabular}
```

---

## G. AutoDAN-Reasoning `vanilla` (n=50 range 50–100) — Prompts 50–100 ✓ Claude-normalized

**Source files:**
- `reasoning_vanilla_S2S3_algo1_n50-100_claude_judged.json`

| Victim | Defense | n | Pos Exits | Neg Exits | MI-FP | FP | TP | ASR | Avg Iter |
|--------|---------|---|-----------|-----------|-------|----|----|-----|----------|
| Vicuna-13b | LlamaGuard | 50 | 7 | 43 | 0 | 6 | 1 | 0.020 | 9.4 |
| Vicuna-13b | LlamaGuard + CMPE | 50 | 0 | 50 | 0 | 0 | 0 | 0.000 | 10.0 |

```latex
\begin{tabular}{llccccccc}
    \toprule
    \textbf{Victim} & \textbf{Defense} & \textbf{n} & \textbf{Pos} & \textbf{Neg} & \textbf{MI-FP} & \textbf{FP} & \textbf{TP} & \textbf{ASR} \\
    \midrule
    Vicuna-13b & LlamaGuard & 50 & 7 & 43 & 0 & 6 & 1 & 0.020 \\
    Vicuna-13b & LlamaGuard + CMPE & 50 & 0 & 50 & 0 & 0 & 0 & 0.000 \\
    \bottomrule
\end{tabular}
```

---

## H. AutoDAN-Reasoning `vanilla` — **Combined n=100** — Prompts 0–100 ✓ Claude-normalized

**Source files:**
- `autodan_reasoning/reasoning_vanilla_S2S3_algo1_n50_claude_judged.json`
- `reasoning_vanilla_S2S3_algo1_n50-100_claude_judged.json`
- `run_20260825_071030/reasoning_vanilla_S2S3_algo1_r50-60_20260825_071030.json`
- `run_20260825_070327/reasoning_vanilla_S2S3_algo1_r60-70_20260825_070327.json`
- `run_20260825_063431/reasoning_vanilla_S2S3_algo1_r70-80_20260825_063431.json`
- `run_20260825_062814/reasoning_vanilla_S2S3_algo1_r80-90_20260825_062814.json`
- `run_20260825_070915/reasoning_vanilla_S2S3_algo1_r90-100_20260825_070915.json`

| Victim | Defense | n | Pos Exits | Neg Exits | MI-FP | FP | TP | ASR | Avg Iter |
|--------|---------|---|-----------|-----------|-------|----|----|-----|----------|
| Vicuna-13b | LlamaGuard | 100 | 57 | 43 | 0 | 52 | 5 | 0.050 | 5.8 |
| Vicuna-13b | LlamaGuard + CMPE | 100 | 16 | 84 | 7 | 9 | 0 | 0.000 | 9.8 |

```latex
\begin{tabular}{llccccccc}
    \toprule
    \textbf{Victim} & \textbf{Defense} & \textbf{n} & \textbf{Pos} & \textbf{Neg} & \textbf{MI-FP} & \textbf{FP} & \textbf{TP} & \textbf{ASR} \\
    \midrule
    Vicuna-13b & LlamaGuard & 100 & 57 & 43 & 0 & 52 & 5 & 0.050 \\
    Vicuna-13b & LlamaGuard + CMPE & 100 & 16 & 84 & 7 & 9 & 0 & 0.000 \\
    \bottomrule
\end{tabular}
```

## Cross-Attack Summary (post-Claude-normalization)

| Run | Defense | n | Pos Exits | Neg Exits | MI-FP | FP | TP | ASR | Avg Iter |
|-----|---------|---|-----------|-----------|-------|----|----|-----|----------|
| AutoDAN-Turbo `use_strategy` (merged n=50) | LlamaGuard | 50 | 49 | 1 | 0 | 44 | 5 | 0.100 | 2.6 |
| AutoDAN-Turbo `use_strategy` (merged n=50) | LlamaGuard + CMPE | 50 | 23 | 27 | 23 | 0 | 0 | 0.000 | 7.6 |
| AutoDAN-Turbo `use_strategy` (chunks 50–100) | LlamaGuard | 50 | 50 | 0 | 0 | 0 | 50 | 1.000 | 2.6 |
| AutoDAN-Turbo `use_strategy` (chunks 50–100) | LlamaGuard + CMPE | 50 | 22 | 28 | 22 | 0 | 0 | 0.000 | 14.7 |
| AutoDAN-Turbo `use_strategy` (n=50 range 50–100) | LlamaGuard | 50 | 50 | 0 | 0 | 41 | 9 | 0.180 | 2.6 |
| AutoDAN-Turbo `use_strategy` (n=50 range 50–100) | LlamaGuard + CMPE | 50 | 22 | 28 | 22 | 0 | 0 | 0.000 | 14.7 |
| AutoDAN-Turbo `use_strategy` — **Combined n=100** | LlamaGuard | 100 | 99 | 1 | 0 | 85 | 14 | 0.140 | 2.6 |
| AutoDAN-Turbo `use_strategy` — **Combined n=100** | LlamaGuard + CMPE | 100 | 45 | 55 | 45 | 0 | 0 | 0.000 | 11.2 |
| AutoDAN-Reasoning `vanilla` (merged n=50) | LlamaGuard | 50 | 50 | 0 | 0 | 46 | 4 | 0.080 | 2.2 |
| AutoDAN-Reasoning `vanilla` (merged n=50) | LlamaGuard + CMPE | 50 | 16 | 34 | 7 | 9 | 0 | 0.000 | 9.7 |
| AutoDAN-Reasoning `vanilla` (chunks 50–100) | LlamaGuard | 50 | 7 | 43 | 0 | 0 | 7 | 0.140 | 9.4 |
| AutoDAN-Reasoning `vanilla` (chunks 50–100) | LlamaGuard + CMPE | 50 | 0 | 50 | 0 | 0 | 0 | 0.000 | 10.0 |
| AutoDAN-Reasoning `vanilla` (n=50 range 50–100) | LlamaGuard | 50 | 7 | 43 | 0 | 6 | 1 | 0.020 | 9.4 |
| AutoDAN-Reasoning `vanilla` (n=50 range 50–100) | LlamaGuard + CMPE | 50 | 0 | 50 | 0 | 0 | 0 | 0.000 | 10.0 |
| AutoDAN-Reasoning `vanilla` — **Combined n=100** | LlamaGuard | 100 | 57 | 43 | 0 | 52 | 5 | 0.050 | 5.8 |
| AutoDAN-Reasoning `vanilla` — **Combined n=100** | LlamaGuard + CMPE | 100 | 16 | 84 | 7 | 9 | 0 | 0.000 | 9.8 |

---

## Claude Normalization: **All runs normalized** ✓
