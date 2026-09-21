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

> **Positive Exits (claimed jailbreaks) = MI FP + FP + True Positives**

---

### AutoDAN-Turbo Results (`use_strategy` mutation, n=50, algo1q)

Attacker: gemma1-7b-it | Scorer: openai/gpt-oss-120b | Max epochs: 20 | Break score: 8.5

| Model       | Defense Strategy       | Total | Positive Exits | MI False Positives | Exhausted | False Positives | True Positives | Avg Iters |
| ----------- | ---------------------- | ----- | -------------- | ------------------ | --------- | --------------- | -------------- | --------- |
| Vicuna      | LLaMA-Guard3-8B        |    50 |              6 |                  — |        44 |               3 |              3 |      18.8 |
| Vicuna      | LLaMA-Guard3-8B + CMPE |    50 |              5 |                  5 |        45 |               0 |              0 |      19.2 |
| Abliterated | LLaMA-Guard3-8B        |    50 |             13 |                  — |        37 |               5 |              8 |      16.3 |
| Abliterated | LLaMA-Guard3-8B + CMPE |    50 |             26 |                 26 |        24 |               0 |              0 |      13.0 |


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


---

### PAIR Results — Misdirection-Hardened Judge

| Model  | Defense Strategy       | Total | Positive Exits | MI FP | Exhausted | FP | True Positives | Avg Iters |
| ------ | ---------------------- | ----- | -------------- | ----- | --------- | -- | -------------- | --------- |
| Vicuna | LLaMA-Guard3-8B        |    50 |              8 |    —  |        42 |  3 |              5 |      45.8 |
| Vicuna | LLaMA-Guard3-8B + CMPE |    50 |             37 |    35 |        13 |  0 |              2 |      25.8 |
