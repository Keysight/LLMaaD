> **Positive Exits (claimed jailbreaks) = MI FP + FP + True Positives**

---

### AutoDAN-Turbo Results (`use_strategy` mutation, n=100)

Victim: lmsys/vicuna-13b-v1.5 | Scorer: gemma1-7b-it | Max epochs: 20 | Break score: 8.5

| Defense Strategy | Total | Positive Exits | MI FP | Exhausted | FP | True Positives | Avg Iters |
| ---------------- | ----- | -------------- | ----- | --------- | -- | -------------- | --------- |
| LLaMA-Guard3-8B | 100 | 99 | — | 1 | 85 | 14 | 2.6 |
| LLaMA-Guard3-8B + CMPE | 100 | 45 | 45 | 55 | 0 | 0 | 11.2 |

---

### AutoDAN-Reasoning Results (`vanilla` method, n=100)

Victim: lmsys/vicuna-13b-v1.5 | Scorer: gemma1-7b-it | Max epochs: 10 | Break score: 8.5

| Defense Strategy | Total | Positive Exits | MI FP | Exhausted | FP | True Positives | Avg Iters |
| ---------------- | ----- | -------------- | ----- | --------- | -- | -------------- | --------- |
| LLaMA-Guard3-8B | 100 | 57 | — | 43 | 52 | 5 | 5.8 |
| LLaMA-Guard3-8B + CMPE | 100 | 16 | 7 | 84 | 9 | 0 | 9.8 |

---

### Column Descriptions (Attacker POV)

| Column | Attacker's Interpretation |
| ------ | ------------------------- |
| **Total** | Number of adversarial goals the attacker attempted |
| **Positive Exits** | Total times the attacker's judge declared success — the attacker believes these are jailbreaks |
| **MI FP** | *Misdirection-Induced False Positives* — subset of Positive Exits where the attacker was fooled by a misdirected response; the attacker claimed a win but received a controlled, non-operational reply. Shown as — under detect-and-block, where no misdirection is applied |
| **Exhausted** | Runs where the attacker exhausted its full iteration budget without the judge ever claiming success — a definitive failure from the attacker's perspective |
| **FP** | *False Positives* — subset of Positive Exits where the attacker's judge was wrong independently of misdirection; confirmed safe or off-topic by secondary validation |
| **True Positives** | Subset of Positive Exits confirmed as genuine, harmful jailbreaks by secondary validation — the attacker's real wins |
| **Avg Iters** | Average iterations the attacker ran before terminating (either by claiming success or exhausting budget) |

---

### GPTFuzz Results

Avg Iters = average over all 50 prompts (GPTFuzz  iterations; PAIR  total streams = iterations × 10)

| Model       | Defense Strategy          | Total | Positive Exits | MI FP | Exhausted | FP | True Positives | Avg Iters |
| ----------- | ----------------          | ----  | -------------  | ----  | --------  | -  | -------------  | --------  |
| Vicuna      | LLaMA-Guard3-8B           |    50 |             23 |     — |        27 | 13 |             10 |      37.5 |
| Vicuna      | LLaMA-Guard3-8B + CMPE    |    50 |             48 |    44 |         2 |  4 |              0 |       6.8 |
| Abliterated | LLaMA-Guard3-8B           |    50 |             22 |     — |        28 | 14 |              8 |      34.0 |
| Abliterated | LLaMA-Guard3-8B + CMPE    |    50 |             48 |    42 |         2 |  5 |              1 |       6.4 |


---

### PAIR Results

Avg Iters = average over all 50 prompts (total streams = iterations × 10)

| Model        | Defense Strategy         | Total | Positive Exits | MI FP | Exhausted | FP | True Positives | Avg Iters |
| ------------ | ----------------         | ----  | -------------  | ----  | --------  | -  | -------------  | --------  |
| Vicuna       | LLaMA-Guard3-8B          |    50 |              8 |     — |        42 |  3 |              5 |      45.4 |
| Vicuna       | LLaMA-Guard3-8B + CMPE   |    50 |             43 |    43 |         7 |  0 |              0 |      20.4 |
| Abliterated  | LLaMA-Guard3-8B          |    50 |             12 |     — |        38 |  7 |              5 |      41.4 |
| Abliterated* | LLaMA-Guard3-8B + CMPE   |    50 |             47 |    47 |         3 |  0 |              0 |      15.8 |
