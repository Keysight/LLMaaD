**Table II: GPTFuzz Results (50 prompts per configuration, Claude-validated)**

| Model       | Defense Strategy | Total | Atk JBs | Misdirected | Defended | False Pos | False Neg | True JBs | Avg Iters |
| ----------- | ---------------- | ----- | ------- | ----------- | -------- | --------- | --------- | -------- | --------- |
| Vicuna      | Detect-Block     | 50    | 23      | 0           | 27       | 13        | 0         | 10       | 37.5      |
| Vicuna      | Detect-Misdirect | 50    | 4       | 44          | 2        | 4         | 0         | 0        | 6.8       |
| Abliterated | Detect-Block     | 50    | 22      | 0           | 28       | 12        | 2         | 8        | 34.0      |
| Abliterated | Detect-Misdirect | 50    | 6       | 42          | 2        | 2         | 3         | 1        | 6.4       |