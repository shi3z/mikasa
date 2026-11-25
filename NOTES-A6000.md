# mikasa on the RTX A6000 (sm_86) - measurements

Build: `nvcc -O3 -std=c++17 -arch=sm_86 -I src -o mikasa src/inkling_model.cu src/gguf.cpp` (CUDA 12.8, 3 s).
Model: `Inkling-2-Mini-UD-Q4_K_XL.gguf` (unsloth, 17.6 GB), tokens from the official tokenizer.json + chat_template.jinja (`tools/tok.py`).

## Correctness (semantic only; not yet bit-compared with a reference)
Prompt "日本語で一文だけ自己紹介して。" (26 tokens), 160 greedy tokens: a coherent English thinking, then
`<|end_message|><|message_model|><|content_text|>私はInklingで、Thinking Machines Labが開発したAIアシスタントです。<|end_message|><|content_model_end_sampling|>`.
No stop logic yet: it keeps generating after the end token.

## Speed (decode, batch 1, CUDA Graph replay, `--gen 128 --reps 5`)
| condition | GPU ms/token | tok/s |
|---|---:|---:|
| exclusive GPU (llama-server SIGSTOPped), 3 runs | 11.05 - 11.11 | 90.0 - 90.5 |
| shared with a running llama-server job (NOT valid) | 18.0 - 18.6 | 54 - 56 |
| llama.cpp (mainline + PR 32704), Q4_K_M, same GPU | ~20 | ~50 |
Upstream (RTX 3090): 9.80 ms. The A6000 has 768 GB/s against 936 GB/s: 9.8 x 936/768 = 11.9 ms expected, 11.07 measured.
Model load: 9 s. Host-side launch: 2.2 ms/token, hidden behind the 11 ms of GPU time.
