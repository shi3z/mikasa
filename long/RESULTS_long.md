# mikasa-long: forward_batch with 64K-token contexts (4 slots)

Work copy of `~/rrr/mikasa-batch` (originals untouched). The previous design notes are kept in `RESULTS_batch_orig.md`.
Build / run (every number below comes from the final binary built with this exact command):

```
/usr/local/cuda/bin/nvcc -O3 -std=c++17 -arch=sm_86 -I src -o mikasa_long src/inkling_model.cu src/gguf.cpp
M=~/rrr/gguf/Inkling-2-Mini-UD-Q4_K_XL.gguf
./mikasa_long $M out --batch-test                                   # old oracle test (non-lean, FP32 KV, alias slot 0), must pass
./mikasa_long $M out --lean --kv16 --prefill data/needle60k.txt --gen-long 600 logs/gen.txt   # lean, FP16 full-layer KV, 4 slots x 65536
./mikasa_long $M out --lean --kv16 --time-decode --time-prefill     # timings
```
Flags (new): `--lean` (see 1b), `--kv16|--kv32`, `--maxctx N` (default 65536), `--nslots N` (default 4), `--ring N`, `--prefill FILE [--slot s] [--pf-b B]`,
`--tail K FILE` (dump logits of the last K prefilled tokens), `--argmax-out FILE`, `--cmp-ref K` (new vs naive attention on the last K tokens),
`--ref-all` (whole run with the naive attention), `--multi-check`, `--gen-long N FILE` (greedy decode), `--time-decode`, `--time-prefill`,
`--cmp-dump A B K` (compare two logit dumps). Scripts: `tools/mkctx.py` (token files), `tools/run_all.sh` (the whole validation chain), `tools/run_*.sh`. Logs in `logs/`.

## 1. Memory
(a) KV layout (`BatchState::kvs`, per application and slot):
- 40 SWA applications: FP32 ring of **1040 rows** (row = pos % 1040), not 1024. Reason: a same-sequence batch writes B <= 16 new rows before attention runs; with exactly 1024 rows the write of position p+15 would overwrite position p-1009, which query p still needs (window p-1023..p). 1040 >= 1024 + 16 - 1 is the minimum safe size. 40 x 4 slots x 2 x 1040 x 4 KB = 1.4 GB.
- 8 full applications: linear [65536][1024] per slot, FP16 (`--kv16`, 2.1 GB per slot incl. K and V, 8.6 GB for 4 slots) or FP32 (`--kv32`, 4.3 GB per slot, 17.2 GB for 4 slots).
(b) `--lean`: weights are uploaded, converted straight to the interleaved layout of the batch kernels and the SoA copy is freed per tensor (`lean_il`); `tmap()` becomes the identity. No `Engine::kcache/vcache`. Works: `forward_batch` for any B = 1..16, `multi` 0 and 1, rollback, prefill, greedy decode, all long-context code. Not available: `forward_token`, the original main path, `--batch-test`, `--batch-bench` (they need the SoA weights / engine KV).

Measured (cudaMemGetInfo, and nvidia-smi peak sampled every 100 ms; card total 50.9 GB incl. ~0.4 GB baseline use):

| configuration | weights | after init_batch (total) | nvidia-smi peak |
|---|---|---|---|
| lean, FP16 full KV, 4 slots x 64K | 18.6 GB | **29.3 GB** | 27967 MiB (= 29.3 GB) |
| lean, FP32 full KV, 4 slots x 64K | 18.6 GB | **37.9 GB** | 36159 MiB (= 37.9 GB) |
| non-lean (SoA + interleaved), FP16 KV, 4 x 64K | 19.0 GB (SoA only) | 46.9 GB | not sampled |

So lean saves ~17.6 GB. FP32 KV for all 4 slots at 64K also fits comfortably (37.9 GB), i.e. the FP16 cache is a speed/memory option, not a necessity (see accuracy in 4).

## 2. Attention (`src/long_kernels.cuh`)
- `kb_kv_write<float|half>`: clip/RMS + store at row pos (full) or pos % ring (SWA) of the slot's cache (slot pointers per application in `KVSlots`, so slot 0 is no longer special).
- `kb_attn_part` + `kb_attn_combine`: split-KV. One block = (query, chunk, kv head), 512 threads, half-warp per key with 16-byte loads (FP16 or FP32), the 3 query heads of a kv head share the K/V read, chunk-local softmax (online-softmax partials m, l, O), then a combine kernel merges the chunk partials together with the zero sink logit (`M = max(0, m_c)`, `den = exp(-M) + sum w_c l_c`) and applies the sigmoid gate. Full layers: chunk = 1024 positions, grid (B, 64, 8) (blocks beyond pos exit immediately; b is the fastest grid index so the queries of one batch share the chunk through L2). SWA layers use the same kernels with window 1024 in chunks of 128 positions (8 chunks, ring addressing); this alone took S=1 decode at 1K from 15.8 to 12.9 ms (the old one-block-per-kv-head kernel was latency-bound with 8 blocks). The old ring kernel `kb_attn2_ring` is kept (`MIKASA_SWA_OLD=1`).
- Causal masking in batches unchanged (query b at `pos_arr[b]`, keys <= pos_arr[b], slot by `slot_arr[b]`); relative-bias rows only for distance < 1024 (zero row beyond, same as llama.cpp `set_input_relative_pos`, extent 1024 from the GGUF key `relative_extent`).
- Other T <= 1024 assumptions: the only ones were the smem score buffer of kb_attn2 and the KV allocation (MAXT). `kb_scale_tau`, router, mixes, convs are position independent. `forward_token`/`k_attn_decode_gen` still have the 1024 limit (only used by the non-lean oracle).
- tau scaling checked against the source of truth: `llama.cpp-inkling/src/models/inkling2.cpp` computes `tau = 1 + alpha * log(max(1, (pos + 1) / floor))` and multiplies Q and the relative-position vector R of the full-attention layers only; the GGUF has `log_scale_floor = 8192`, `log_scale_alpha = 0.1` (also `context_length = 1048576`, `sliding_window = 1024`, `relative_extent = 1024`). `kb_scale_tau` implements exactly this (alpha as float32(0.1)), reads the per-token position, and is applied to full layers only. At position 65535 tau = 1.208. No clamping or other change was needed.
- `kb_attn_ref`: naive reference (FP32, one block per (head, query), all scores in global memory, plain max/sum/PV; ring or linear; SWA and full), selected with `X.ref` (`--ref-all` / `--cmp-ref`).

## 3. Prefill
`prefill(g, X, R, slot, tokens, n, start, chunk, argmax_out, tail_logits, tail_k, stream)` (`src/long_host.inc`): chunks of B = 16 through `forward_batch` (same-sequence graph), positions up to 65535. For slots > 0 the carried conv state of the slot is swapped with slot 0's for the duration of the call (the same-sequence graph only knows slot 0's conv state); KV goes straight to the slot's cache. Verified by `--multi-check`.
Measured (GPU event time, real tokens, B = 16): 4K context 288 tok/s, 16K 279, 30K 269, 60K 250 (FP16 KV); FP32 KV: 231 at 60K. Chunk timing with synthetic KV at position ctx (B = 16): FP16: 308 tok/s at 256, 290 at 8K, 257 at 32K, 226 at 65K; FP32: 307 / 284 / 237 / 196.
B = 32 was NOT attempted: `MB = 16` is hard-wired in many kernels (per-token register arrays, `grp_t` stride 16, gemv template ladder up to 16). It is not a cheap change.

## 4. Validation (all numbers from the final code; `tools/run_all.sh`, logs in `logs/`)
(i) `--batch-test` (non-lean, FP32, slot 0 aliasing the engine KV, so the old oracle still applies): **ORACLE TEST: argmax matches at every position** (`logs/batch_test_final.log`): all B = 1,2,4,8,16 chunks, rollback and multi-sequence S = 2..4 cases, 0 argmax mismatches, max abs diff 1.6e-4 / max rel 6.0e-6 (old: 8.9e-5 / 3.4e-6; the SWA layers now use the split kernel, i.e. a different summation order; all other numbers in the same range).

(ii) Optimised vs naive reference attention, last 8 tokens of the context, same KV prefix (rollback of the batch, rerun with `kb_attn_ref`):

| context | FP16 KV: max abs / max rel, argmax | FP32 KV: max abs / max rel, argmax |
|---|---|---|
| 4096 | 1.6e-4 / 5.9e-6, 0 mismatches | 2.9e-5 / 1.1e-6, 0 |
| 16384 | 3.5e-4 / 1.3e-5, 0 | 7.2e-5 / 2.7e-6, 0 |
| 61440 | 7.7e-5 / 2.7e-6, 0 | 8.5e-5 / 3.0e-6, 0 |

(The contexts are the three probe prompts followed by real text from the llama.cpp docs, `data/ctx*.txt`; no random filler was needed. The reference covers both SWA and full layers.)
Ring correctness: 4096 tokens with ring = 1040 (wraps 4 times) vs ring = 4096 (no wrap): logits of the last 16 positions **bit-identical**, all 4096 argmax identical.
Multi-slot: `--multi-check` (4 slots prefilled with different slices/lengths 3696/3643/3590/3537, then one multi-sequence step vs four single-sequence steps): max diff 0 in all 4 slots (bit-identical).
End-to-end (whole prefill with the naive attention vs optimised, FP32 KV): argmax agreement 100% (4096 positions) and 99.87% (16384; 22 of 16384 differ), but the last-16-position logits differ by up to 0.54 abs (rel 1.8e-2) at 4K and 5.5 abs (rel 0.19) at 16K. This is not a kernel discrepancy (the same-KV comparison above gives 1e-6) but the sensitivity of the network to tiny perturbations accumulated over thousands of positions: FP16 rounding of K only / V only / both at 512 tokens gives 2.5e-2 rel logit change in each case. My hypothesis (NOT verified) is that discrete top-8 router decisions flipping at near-ties amplify the noise. Consequence: end-to-end logit comparisons between two numerically different long runs have a noise floor of this size; they cannot show kernel exactness, the same-KV test does.

FP16 vs FP32 KV (same context, whole prefill, argmax at every position): agreement 98.68% (4096; positions >= 1024: 98.47%), 98.10% (16384; 98.02%), 98.39% (61440; 98.37%). Last-8-token logits: max rel diff 0.14 / 0.32 / 0.053, argmax mismatch at the tail only once (60K, 1 of 8). I.e. FP16 KV costs about 1.6% argmax disagreement on ordinary text, roughly 10x the 0.13% flip rate of two FP32 implementations (16K). Emulating the FP16 rounding in FP32 storage (`MIKASA_KVROUND=3`) gives bit-identical logits to the real FP16 cache, so the FP16 path itself is implemented correctly; the loss is the rounding of K/V. Use `--kv32` where fidelity matters (fits: 37.9 GB for 4 x 64K).

(iii) Needle in a haystack (chat template via `chatfmt.render_prompt`, venv-conv python, "The secret code is 7391." inserted into real documentation text, question "What is the secret code ... Answer with the number only.", greedy decoding, `data/needle*.txt`, `logs/needle_*.text`). All retrieved, final answer `7391`:

| prompt tokens | needle depth | KV | prefill tok/s | result |
|---|---|---|---|---|
| 29806 | 5 % | FP16 | 269 | 7391 |
| 29806 | 50 % | FP16 | 269 | 7391 |
| 59865 | 5 % | FP16 | 250 | 7391 |
| 59865 | 50 % | FP16 | 250 | 7391 |
| 59865 | 5 % | FP32 | 231 | 7391 |

Example output (60K, 5 %, FP16): `<|message_model|><|content_thinking|>The user is asking for the secret code mentioned in the document. I need to find "The secret code is" in the document. Looking at the document, I can find: "The secret code is 7391." So the answer is 7391.<|end_message|><|message_model|><|content_text|>7391<|end_message|><|content_model_end_sampling|>`. Greedy decode of these runs: 14-16 ms/token (FP16) including a host round trip per token.

## 5. Timings (RTX A6000, CUDA graphs, 30 reps, lean, 4 slots allocated, maxctx 65536)
Decode step = one token for each of S sequences (multi-sequence graph, distinct slots). KV contents are synthetic (zeros): the cost depends only on the positions. Position 65000 is used for "64K". `logs/timing_kv16.log`, `logs/timing_kv32.log`.

| context | FP16 S=1 | FP16 S=4 | FP32 S=1 | FP32 S=4 |
|---|---|---|---|---|
| 300 | 12.48 ms | 20.12 | 12.53 | 20.28 |
| 1K | 13.05 | 21.52 | 13.03 | 21.70 |
| 8K | 13.17 | 22.32 | 13.49 | 23.89 |
| 32K | 14.35 | 27.10 | 15.86 | 33.22 |
| 64K | 16.05 | 33.56 | 19.02 | 45.75 |

S=1 at 300 tokens is 12.5 ms vs 13.0 ms for the original batch code (`mikasa_batch`, same machine, measured): no regression. The long-context cost is memory bandwidth: 64K FP16 adds ~3.5 ms per sequence (2.1 GB of KV per token, ~0.6 TB/s effective), FP32 ~6.5 ms.
Prefill (B = 16 chunk): see section 3.

## 6. Unfinished / approximate / honest list
- FP16 full-layer KV is lossy: ~1.6% argmax disagreement with FP32 on ordinary text (section 4). Not a bug (verified by rounding emulation), a precision cost; FP32 is available and fits.
- End-to-end logit equality between different numerics is not achievable at long context (noise floor above); the kernel-level equality is shown on identical KV. The router-flip explanation is a hypothesis.
- The ring has 1040 rows, not 1024 (necessary for B = 16 batches; documented above).
- B = 32 prefill not done. Prefill is 226-308 tok/s (not much better than before; attention adds up to 25% at 64K; the model body is the cost).
- Timings use synthetic zero KV (cost is data independent in these kernels, but this was not independently verified); the prefill/decode numbers inside needle runs use real data and agree (decode 14-16 ms/token at 30-60K).
- The full-attention grid is sized for `maxctx` (blocks beyond the position exit at once); a smaller `--maxctx` is slightly cheaper at short contexts.
- Lean mode cannot run `forward_token`, `--batch-test`, `--batch-bench`; long KV + non-lean fits (46.9 GB) but only with FP16 KV and 4 slots (leaves 4 GB).
- Prefill of a slot > 0 swaps conv states with slot 0 (cheap, verified); mixing several tokens of one sequence with other sequences in a single step is still not supported (inherited limitation).
- Only positions up to 61439 carry real content in the correctness tests; position 65535 itself (last row) and `maxctx` boundary were not exercised with data. Prefill/decode refuse positions >= maxctx.
- Generation is greedy only; no stop-sequence handling beyond the two stop ids 200006 / 199999 in `--gen-long`.
- No git commit (as instructed); old `mb_verify`, `mikasa_batch` binaries from the source copy are untouched.
