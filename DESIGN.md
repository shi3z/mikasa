# Inkling 2 Mini inference engine: design

Goal: a decode engine built into the inkling harness, fast on RTX 3090 (sm_86),
reading the unsloth GGUF UD-Q4_K_XL file. llama.cpp is not used at run time.
prebuilt llama.cpp b12588 stays as the reference for output checks only.

## 1. What the model needs (from the GGUF and inkling2.cpp)

Metadata (general.architecture = inkling2):

| key | value |
|---|---|
| block_count | 48 (24 weight layers, the middle 12 run 3 times) |
| embedding_length | 2304 |
| attention heads | 24 query, 8 kv, head dim 128 (key = value = 128) |
| experts | 128 routed, top 8, FFN 1152 per expert, sigmoid gating, bias for selection |
| expert_weights_scale | 8.0 |
| context | up to 1048576 |
| vocab | padded 201024, real 200058 (logits are cut to the real size) |
| attention | full layers + SWA layers (window 1024), pattern in the GGUF |
| relative position bias | 16 buckets, extent 1024, clip 8.0 |
| qk clip | 22.627 (clip each row to L2 norm), log-scale position for full layers |
| short convolution | kernel 4, dynamic taps on full layers |
| attn res | block size 6; each layer mixes earlier block outputs with learned queries |
| loop | core layers 6..17, 3 passes, gate rank 64, messages per pass |
| final logit softcap | 30.0 |

Tensor types in the file: 0 = F32 (508), 8 = Q8_0 (154), 30 = BF16 (32), 40 = NVFP4 (40).

## 2. Per-token forward pass (decode, one token, batch 1)

Taken from `src/models/inkling2.cpp` (lines 294-670). Per layer `il`:

1. Attention input: `mix(blocks, cur, attn_res_query)` (SWA layers give one input, full layers give Q, K, V inputs). Then the loop message control when `messages` exist (pass > 0 in the core).
2. Norms: SWA uses `attn_norm`; full layers use `q/k/v_inp_norm` (RMS).
3. Projections: Q = wq*in_q, K = conv(wk*in_k), V = conv(wv*in_v), R = attn_rel*in_q.
   - K and V pass a depthwise short conv (kernel 4) with the state of the last 3 inputs; full layers add dynamic taps from `k/v_conv_proj`.
4. Clip Q (per-channel scale `attn_q_clip`), K, R; RMS-norm V; scale Q and R by tau = 1 + alpha*log(max(1,(pos+1)/8192)) on full layers.
5. Attention: `softmax(QK^T/sqrt(d) + rel_bias[R-table]) with sink`. Sink term: `attn_sink_proj * in_q`. Full layers use the whole cache; SWA layers use a window of 1024.
6. Output gate: `out = attn * sigmoid(wqkv_gate * in_q)`, then `wo`, then a conv with `attn_post_conv`.
7. Residual: layers at block starts push `inpSA` into `blocks`, others add it.
8. FFN: `mix(blocks, cur, ffn_res_query)`, optional loop control (message 3), RMS norm, MoE (router sigmoid, top 8 of 128, weights scaled by 8.0), then `ffn_post_conv`, residual add.
9. Final: `mix(blocks, cur, output_res_query)`, RMS norm, `output` (lm head, padded), softcap 30, cut padding.

Embedding: `tok_embd` lookup, then `tok_norm` RMS norm (token input only).

## 3. Where the time goes (measured)

Decode is ~60 tok/s; the per-token budget is ~16.7 ms.

- Memory read per token (estimate): experts 8/128 of the routed weights per layer application, attention, output head (BF16 201024 x 2304 = 0.93 GB read in BF16 by the current build; Q8_0 in other builds). The head alone is a large share, so it is worth a dedicated kernel.
- Measured kernels (nsys, prebuilt): quantized GEMV 31%, fp32 GEMV 8%, small elementwise and norm ops about 25%.
- ncu (Q8_0 GEMV): DRAM 35-62%, occupancy 54%, 64 registers. Underused bandwidth.

## 4. Engine design

### 4.1 Weights
- Read the GGUF directly. Map the file with mmap, copy each tensor to device memory once at load.
- Keep quantized formats as stored: Q8_0 (34 bytes / 32 weights), NVFP4 (block format from llama.cpp `block_nvfp4`), BF16 head, F32 norms.
- Fuse per-layer metadata into a flat table (offsets, shapes, type) built at load. No per-token allocation.

### 4.2 Kernels (decode, batch 1)
- GEMV for each quant type: one warp per output row group, input vector in shared memory (Q8_1 style quantized input), accumulate in int32 then scale. Tune rows per block, threads per block (64/128/256), and loads in flight per thread.
- Fused small ops: rms_norm + scale, bcast add + mul, sigmoid gate + multiply, residual add into the next norm.
- MoE: router GEMV (fp32 weight), top-8 selection on device, then expert GEMVs for the 8 selected experts in one launch (grouped by expert index), then weighted sum.
- Attention: single-query decode kernel over the KV cache with the relative-bias table; SWA layers read only 1024 positions.
- Loop core: the three passes reuse the same weights; the message controllers are small GEMVs with 64-rank bottleneck.
- Sampling: greedy on device (argmax) first; top-k/top-p later.

### 4.3 Launch overhead
- One CUDA graph per decode step (all kernels), captured once and replayed. Graph update when the KV length changes (pass the position in device memory, not as a kernel argument).

### 4.4 Validation
- Reference: prebuilt llama-server (b12588) with greedy decoding (temperature 0) on fixed prompts.
- Per layer: compare hidden states against llama.cpp via dumps (`t_layer_inp`), target max abs diff below 1e-3 in F32 accumulation.
- End to end: identical greedy tokens for 256 tokens on 10 prompts; report mismatches.

### 4.5 Speed targets
- Start: current 60 tok/s. Target by phase: GEMV rewrite 80, graphs and fusion 95, head and MoE grouping 110+.
- Report every change with tok/s, kernel time, DRAM %, occupancy (same prompts as the harness benchmark).

## 5. Phases

| phase | content | exit check |
|---|---|---|
| 0 | GGUF loader, tensor table, device upload | every tensor loaded, checksums match the file |
| 1 | reference forward in F32 on CPU for 1 token (small test) | logits match llama.cpp dump within tolerance |
| 2 | GPU kernels for Q8_0 and NVFP4 GEMV, norms, attention, MoE | per-layer hidden states match |
| 3 | full decode loop, greedy, tokenizer from GGUF | identical 256-token outputs on 10 prompts |
| 4 | CUDA graph, fusion, GEMV tuning | tok/s table vs baseline |
| 5 | harness integration (`inkling` uses the built-in engine) | `inkling ask` works, same output |

## 6. Open questions
- Tokenizer: needs the GGUF tokenizer (pre-tokenizer `inkling2`); reuse the Python side or port.
- Vision input (mmproj): out of scope for phases 0-5.
- DSpark drafter: add after phase 3, same kernels with block size 8.
- NVFP4 layout: confirm the block layout against llama.cpp `ggml-common.h` before writing the kernel.
