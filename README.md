# Mikasa — Inkling-2 Mini Inference Engine

A single-file-stack CUDA inference engine for Inkling-2 Mini on the RTX 3090 (sm_86).
It memory-maps the GGUF model (`unsloth/Inkling-2-Mini-GGUF`, `Inkling-2-Mini-UD-Q4_K_XL.gguf`)
and runs each decode step as one CUDA Graph. llama.cpp is used only as a numerical reference
(logits and intermediate tensors); it is not used at runtime.

## Benchmark summary (tokens per second, RTX A6000 48 GB, greedy, CUDA Graph)

Every number below was measured on this machine in the sessions that produced the commits; details, logs and caveats are in the sections that follow.
Decode numbers use synthetic KV contents (cost depends on positions only), prefill numbers use real prompts.

**Decode (generation), tok/s**

| context | 1 sequence | 4 sequences in parallel (aggregate) | per sequence when 4 run |
|---:|---:|---:|---:|
| 300 | 80.1 | 198.8 | 49.7 |
| 1K | 76.6 | 185.9 | 46.5 |
| 8K | 75.9 | 179.2 | 44.8 |
| 32K | 69.7 | 147.6 | 36.9 |
| 64K | 62.3 | 119.2 | 29.8 |

(`--lean --kv16`; 1 sequence = 1000 / ms-per-step. Default single-token path at short context: 82-91 tok/s, see Status and Profile.)

**Prefill (prompt processing), tok/s**

| prompt tokens | old 16-token GEMV path | `--pf-fast 512` | `--pf-fast 1024` |
|---:|---:|---:|---:|
| 512 | 306 | 3,510 | 3,527 |
| 4K | 297 | 3,306 | 3,743 |
| 16K | 279 | 3,082 | 3,509 |
| 32K | 265 | 2,893 | 3,269 |
| 60K | 247 | 2,612-2,652 | 2,916 |

**Speculative decoding (`--spec`, lean mode, greedy, single sequence; plain = 77-80 tok/s), tok/s**

| workload (1000 tokens) | plain | n-gram / copy drafts, batch kernels (output identical to plain) | `--spec-wide` 32-64 row verify (approximate greedy) |
|---|---:|---:|---:|
| whole file with 2 small changes (copy-heavy) | 77 | 216 (2.8x) | **398 (5.2x)** |
| JSONL edit | 77 | 210 (2.7x) | 347 (4.5x) |
| edit as unified diff / write_file tool call | 77 | 112-118 (1.5x) | 112-121 |
| shell command tool calls | 77-82 | 93-182 (1.2-2.3x) | similar |
| new code generation | 79 | 83-95 (1.04-1.2x) | similar |
| reasoning text | 79 | 86-93 (1.1-1.2x) | similar |

**Multi-token verification (same sequence, B tokens per pass, the basis for speculative decoding), tok/s**

| B | 1 | 2 | 4 | 8 | 16 |
|---|---:|---:|---:|---:|---:|
| tok/s | 76 | 141 | 234 | 294 | 330 |

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

| same sequence, B tokens | ms per call | ms per token | tok/s |
|---|---:|---:|---:|
| 1 | 13.19 | 13.19 | 76 |
| 4 | 17.13 | 4.28 | 234 |
| 8 | 27.25 | 3.41 | 294 |
| 16 | 48.45 | 3.03 | 330 |

| parallel sequences | ms per step | aggregate tok/s | per sequence tok/s |
|---|---:|---:|---:|
| 1 | 13.20 | 75.8 | 75.8 |
| 2 | 15.61 | 128.2 | 64.1 |
| 4 | 20.24 | 197.6 | 49.4 |

Limits of the plain batch path: context MAXT = 1024 positions per sequence and a second interleaved weight copy (~36 GB). The long-context mode below removes both.

## Long context and fast prefill (`--lean`, `--pf-fast`)

`--lean --kv16 --nslots 4 --maxctx 65536` runs `forward_batch` with 4 sequence slots of up to 65,536 positions each, without the
original weight copies: 29.3 GB total (FP16 KV on the 8 full-attention applications; 37.9 GB with FP32 KV). SWA applications use a
1040-row ring. Full layers use split-KV attention. Decode step (ms, CUDA graph; KV contents synthetic, cost depends on positions only):

| context | 1 sequence | 4 sequences (one token each) |
|---|---|---|
| 1K | 13.1 | 21.5 |
| 8K | 13.2 | 22.3 |
| 32K | 14.4 | 27.1 |
| 64K | 16.1 | 33.6 |

`--pf-fast PB` (PB = 512 or 1024) prefills with tensor-core GEMMs (mma, reading the existing Q8_0 / NVFP4 weights), grouped MoE tiles
and flash-style attention; logits only for the last token. About 11x faster than the 16-token GEMV prefill: 3,300 tok/s at 4K,
about 2,650 tok/s at 60K (PB=512; 2,900-3,700 at PB=1024; old path 250-300).

Validation (details and logs in `long/`): oracle test passes (argmax equal at every position, max rel 6e-6); needle-in-a-haystack
at 30K and 60K returns the right answer for FP16 and FP32 KV; teacher-forced 32-step continuation after a 4096-token prefill: 0 argmax
mismatches vs the old prefill (relative logit L2 7.6 % mean, about the size of the FP16-KV vs FP32-KV difference of the old path).
Caveats: FP16 KV changes the argmax at ~1.3-1.9 % of positions versus FP32 KV (rounding, not a bug); prefill rounds activations to FP16;
splitting a prompt at a position that is not a multiple of 256 changes logits at noise level or more; `--lean` has no `forward_token` /
`--batch-test` (use the non-lean build for the oracle test); B=32 and position 65535 itself are untested. Reproduce with `tools/run_all.sh`,
`tools/run_prefill_all.sh`.

## Speculative decoding (`--spec`, `--spec-copy`, `--spec-wide`, `--temp`)

Lean-mode generation loop with one CUDA-graph launch and one host sync per step (`src/spec_host.inc`). Drafts come from a lookup over prompt + generated text:
n-gram with the longest backward match, and (`--spec-copy 4`) a copy proposer after [shi3z/xeflash](https://github.com/shi3z/xeflash) (3-gram index, longest match over the 64 latest occurrences,
"sticky" continuation of the previous copy source). Verification = `forward_batch` over [last token + drafts], longest accepted prefix + bonus token, `batch_rollback` for the conv state. The draft width
is chosen per step from measured step costs and online acceptance estimates, so plain decoding is used when drafting does not pay.

* Batch-kernel verification (up to 16 rows): output identical to the plain loop token for token in 44 of 44 runs over 22 workloads.
* `--spec-wide W`: verification passes with more than W rows (up to 64) run on the tensor-core prefill path with its own rollback. Much faster on copy-heavy output (4.5-5.4x), but **approximate greedy**:
  outputs diverged from plain after 90-744 tokens in 13 of 22 workloads (numerics, not a rollback bug).
* `--temp T --seed S`: rejection-sampling verification (accept a draft with probability p(d)), chi-square tested; no top-k/top-p/penalties, not combined with the wide path.
* Not tested: more than one slot with speculation, `--kv32`, wide path beyond ~25K context. One run per cell, synthetic prompts. Full tables, sweeps and the honest list: [`spec/RESULTS_spec.md`](spec/RESULTS_spec.md); reproduce with `tools/run_spec_all.sh`.

## Resident daemon (`mikasa --serve`) and the OpenAI server

`mikasa --serve` (built from the same sources; defaults `--lean --kv16 --nslots 4 --maxctx 65536 --pf-fast 512 --spec 15 --spec-copy 4`, `--spec-wide` off) loads the model once and serves JSON-line requests on a unix socket.
`server/mikasa_server.py --engine daemon --launch-daemon --model GGUF` exposes it as an OpenAI-compatible endpoint (streaming, tool calls, reasoning split, `/props` with `n_ctx` and slots); the web console preset `mikasa` uses port 8767 like the llama-server preset.
Design, protocol, flags, numbers and the list of limits: [`DAEMON.md`](DAEMON.md); reproduce with `tools/run_daemon_tests.sh`.

* One generating request: n-gram/copy speculation (exact greedy). Two or more: plain multi-sequence steps (no speculation). Up to 4 concurrent requests; cancel frees the slot in 13 ms (decode) / 117 ms (26K-token prefill).
* Prefix cache: a request goes to the slot with the longest reusable prefix. Resident (prompt extends the slot's history) or snapshot (conv state + SWA ring at a multiple of 512, 345 MB each, 24 by default) restore: a 6,880-token turn
  prefills in 99 ms instead of 2,195 ms; generated tokens identical to the uncached run (greedy) in all 7 restores tested.
* Only pure temperature sampling (`top_p=1`, `top_k=0`); other values are rejected with HTTP 400.

| through the server / socket | tok/s |
|---|---:|
| 1 stream, chat (T=1.0) | 78-82 |
| 1 stream, copy-heavy edit (greedy, speculation) | 201 |
| 4 concurrent streams, aggregate (short prompts) | 174 (about 44 per stream) |
| prefill, 7K / 65K prompt | 3,150 / 2,540 |

## OpenAI-compatible front end (`server/`)

`server/chatfmt.py` renders the official chat template and parses the model's output stream into `reasoning_content`, `content` and streamed `tool_calls`;
`server/mikasa_server.py` serves `/v1/chat/completions` (stream or not) with an engine adapter (`server/engines.py`). The real engine is the resident daemon (`mikasa --serve`, `server/daemon_engine.py`); see [`DAEMON.md`](DAEMON.md) (protocol, prefix cache, measured results, limits).

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
