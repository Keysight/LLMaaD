# Third-Party Licenses

This repository includes vendored copies of two open-source attack frameworks. Both are MIT-licensed and have been modified to interface with the LLMaaD defense. The original licenses are reproduced below.

---

## GPTFuzz

**Original repository:** https://github.com/sherdencooper/GPTFuzz

**Paper:** Jiahao Yu, Xingwei Lin, Zheng Yu, Xinyu Xing. "GPTFUZZER: Red Teaming Large Language Models with Auto-Generated Jailbreak Prompts." Usenix Security 2024.

**License:**

```
MIT License

Copyright (c) [2023] [Jiahao YU]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

**Files added by LLMaaD authors** (located in `llmaad_vs_adv_attacks/GPTFuzz/`):
- `gptfuzz_llmaad_parallel.py` — parallel integration script with LLMaaD defense hooks
- `custom_scripts/` — experiment runner scripts and batch evaluation utilities
- `ATTACK_REPORT.md` — experiment documentation
- `llmaad_results/` — experiment results from LLMaaD evaluation

---

## JailbreakingLLMs (PAIR)

**Original repository:** https://github.com/patrickrchao/JailbreakingLLMs

**Paper:** Patrick Chao, Alexander Robey, Edgar Dobriban, Hamed Hassani, George J. Pappas, Eric Wong. "Jailbreaking Black Box Large Language Models in Twenty Queries." arXiv 2023.

**License:**

```
MIT License

Copyright (c) 2023 PAIR Team

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

**Files added by LLMaaD authors** (located in `llmaad_vs_adv_attacks/JailbreakingLLMs/`):
- `run_pair_detect_block_parallel.py` — parallel PAIR runner with detect-and-block defense
- `run_pair_detect_misdirect_parallel.py` — parallel PAIR runner with detect-and-misdirect defense
- `system_prompts.py` — victim model system prompt definitions
- `custom_scripts/` — experiment runner scripts and batch evaluation utilities
- `PAIR_ATTACK.md` — experiment documentation
- `llmaad_results/` — experiment results from LLMaaD evaluation
