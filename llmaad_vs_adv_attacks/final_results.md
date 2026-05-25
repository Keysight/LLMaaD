GPTFUZZ Results:

Avg Iters = average over all 50 prompts (GPTFuzz: iterations; PAIR: total streams = iterations × 10)

| Model       | Defense Strategy | Total | Atk JBs | Misdirected | Defended | False Pos | True JBs | Avg Iters |
| ----------- | ---------------- | ----- | ------- | ----------- | -------- | --------- | -------- | --------- |
| Vicuna      | detect-block     | 50    | 23      | —           | 27       | 13        | 10       | 37.5      |
| Vicuna      | detect-misdirect | 50    | 4       | 44          | 2        | 4         | 0        | 6.8       |
| Abliterated | detect-block     | 50    | 22      | —           | 28       | 12        | 8        | 34.0      |
| Abliterated | detect-misdirect | 50    | 6       | 42          | 2        | 2         | 1        | 6.4       |

PAIR Results:

Avg Iters = average over all 50 prompts (total streams = iterations × 10)

| Model       | Defense Strategy | Total | Atk JBs | Misdirected | Defended | False Pos | False Neg | True JBs | Avg Iters |
| ----------- | ---------------- | ----- | ------- | ----------- | -------- | --------- | --------- | -------- | --------- |
| Vicuna      | detect-block     | 50    | 5       | —           | 45       | 2         | 0         | 3        | 46.4      |
| Vicuna      | detect-misdirect | 50    | 3       | 36          | 11       | 1         | 0         | 2        | 22.8      |
| Abliterated | detect-block     | 50    | 9       | —           | 41       | 6         | 2         | 1        | 46.0      |
| Abliterated | detect-misdirect | 50    | 3       | 46          | 1        | 1         | 0         | 2        | 13.2      |
