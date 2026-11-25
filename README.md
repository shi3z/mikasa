# Mikasa — Inkling-2 Mini Inference Engine

A single-file-stack CUDA inference engine for Inkling-2 Mini on the RTX 3090 (sm_86).
It memory-maps the GGUF model (`unsloth/Inkling-2-Mini-GGUF`, `Inkling-2-Mini-UD-Q4_K_XL.gguf`)
and runs each decode step as one CUDA Graph. llama.cpp is used only as a numerical reference
(logits and intermediate tensors); it is not used at runtime.

## Status

| Version | ms/token | tok/s | Notes |
|---|---:|---:|---|
| Initial CUDA Graph build | 14.09 | 71 | |
| `baseline/ctrl_fused_12p40` | 12.40 | 80.6 | |
| `baseline/pack4_11p86` | 11.86 | 84.3 | NVFP4 expert GEMV row-packing |
| `baseline/fused_11p18` | 11.18 | 89.4 | D2D copy removal, mix/conv fusion |
| `baseline/small_kernel_9p85` | **9.80–9.87** | **101–102** | Small-kernel fusion, register-resident RMS |

Measured with `--gen 128 --reps 5` (RTX 3090, CUDA Graph replay).
The 128-token greedy output matches the reference exactly. Logits differ from the reference by at most 7.6e-6, and the argmax matches at every position.

## Profile (RTX A6000, Nsight Systems, 2026-11-25)

`nsys profile --cuda-graph-trace=node --trace=cuda ./mikasa Inkling-2-Mini-UD-Q4_K_XL.gguf out --gen 64 --reps 2`, GPU otherwise idle.
Under the profiler: 10,950 us/token (91.3 tok/s); summed kernel time 10,679 us/token. Per-kernel table: [`profile/nsys_kernel_summary.csv`](profile/nsys_kernel_summary.csv).

| kernel (per token) | calls | us | share |
|---|---:|---:|---:|
| `gemv_q8_rows<2>` (Q8_0 attention / dense projections) | 241 | 3,334 | 31.2 % |
| `nvfp4_gemv_pack4<72>` + `<36>` (NVFP4 experts) | 88 | 3,090 | 29.0 % |
| `k_attn_decode_gen` | 48 | 592 | 5.5 % |
| `k_mix_scores2` | 144 | 578 | 5.4 % |
| `q8_experts_rows2` | 8 | 431 | 4.0 % |
| router: `gemv_f32_row9` + `k_router_top8` | 96 | 404 | 3.8 % |

The two GEMV families (Q8_0 projections and NVFP4 experts) are 60 % of the step. The MoE router (128 experts, top-8, 48 applications per token) is 3.8 %:
even a free router would give at most +3.8 % tokens/s. [`profile/router_bench.cu`](profile/router_bench.cu) times the exact router kernels in a CUDA graph (409 us per token, matching
the trace) and a fused variant (`profile/fused.cuh`: one kernel, last-block-done ticket, one-warp top-8; identical ids/weights on 300 random trials) that is 7.46 us per application instead of 8.52,
saving 51 us per token (0.46 %). Replacing the router with RT Core BVH traversal (Spectral-AI) is not worthwhile here: its best published latency (19.1 us per batch on an RTX 5070 Ti)
is above the current 8.5 us per application, and a distilled BVH router matches the original top-8 only ~96.6 % of the time. These numbers come from one profiling run.

## Batched forward and parallel requests (`forward_batch`, A6000)

`forward_batch` (in `src/batch_host.inc`, kernels in `src/batch_kernels*.cuh`) runs B tokens in one pass and shares the weight reads: either B consecutive tokens of one sequence
(prefill, speculative-decoding verification, with `batch_rollback(j)` for rejected drafts) or one token each of up to 4 different sequences (parallel requests, own KV cache and conv state per slot).
Validated against sequential `forward_token` (`./mikasa model out --batch-test`): argmax identical at every position of three 326-357 token sequences, max abs logit diff 9e-5, rollback and 2-4 sequence modes included.
`forward_token` and the default decode path are unchanged (128 generated tokens identical). Details, design notes and caveats: [`batch/RESULTS.md`](batch/RESULTS.md).

| same sequence, B tokens | ms per call | ms per token |
|---|---:|---:|
| 1 | 13.19 | 13.19 |
| 4 | 17.13 | 4.28 |
| 8 | 27.25 | 3.41 |
| 16 | 48.45 | 3.03 |

| parallel sequences | ms per step | aggregate tok/s |
|---|---:|---:|
| 1 | 13.20 | 75.8 |
| 2 | 15.61 | 128.2 |
| 4 | 20.24 | 197.6 |

Limits today: context is still MAXT = 1024 positions per sequence; the batch path keeps a second, interleaved copy of the weights (GPU use ~36 GB); one token per sequence per step in multi-sequence mode.

## OpenAI-compatible front end (`server/`)

`server/chatfmt.py` renders the official chat template and parses the model's output stream into `reasoning_content`, `content` and streamed `tool_calls`;
`server/mikasa_server.py` serves `/v1/chat/completions` (stream or not) with an engine adapter (`server/engines.py`). Only a mock engine is wired up so far (protocol tests against the harness client); the real engine adapter is not written yet.

## Layout

- `src/gguf.{hpp,cpp}`: GGUF v3 parser (mmap)
- `src/kernels_common.cuh`: all kernels (NVFP4 / Q8_0 / F32 / BF16 GEMV, RMS norm, mix, conv, attention, loop control)
- `src/inkling_model.cu`: model, `forward_token`, CUDA Graph capture, `main`
- `src/*_bench.cu`, `src/nvfp4_*.cu`, `src/moe_sched.cu`: microbenchmarks for the GEMV variants
- `ref/`: llama.cpp reference dumper (`dump_nodes.cpp`) and numpy reference scripts
- `baseline/`: snapshots of each stage (source and README)
- `DESIGN.md`: design notes

## Build and run

```sh
nvcc -O3 -std=c++17 -arch=sm_86 -I src -o mikasa src/inkling_model.cu src/gguf.cpp
./mikasa Inkling-2-Mini-UD-Q4_K_XL.gguf out_dir --gen 128 --reps 5   # generate and time
./mikasa Inkling-2-Mini-UD-Q4_K_XL.gguf out_dir --dump               # write logits.bin
```

The model file is not included in this repository.
