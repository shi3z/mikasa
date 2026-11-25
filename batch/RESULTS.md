# mikasa-batch: multi-token forward pass (`forward_batch`)

Work copy of `~/rrr/mikasa/src` (the original is untouched). Build and run:

```
/usr/local/cuda/bin/nvcc -O3 -std=c++17 -arch=sm_86 -I src -o mikasa_batch src/inkling_model.cu src/gguf.cpp
./mikasa_batch ~/rrr/gguf/Inkling-2-Mini-UD-Q4_K_XL.gguf out --batch-test    # oracle comparison (about 75 s)
./mikasa_batch ~/rrr/gguf/Inkling-2-Mini-UD-Q4_K_XL.gguf out --batch-bench   # timings (--bench-b N for one B, --bench-pos P, --probe DIR)
```
Logs of the runs below: `test_final.log`, `bench_final.log`, `prof_B8_breakdown.txt`.
`forward_token` and the old main are unchanged and remain the oracle (the same engine state is shared by both paths).

## Files
- `src/batch_kernels.cuh`: batched small kernels (norms, mix, conv, clip, router, grouping, situ, softcap, argmax, rollback).
- `src/batch_kernels2.cuh`: batched heavy kernels (Q8_0 GEMV, NVFP4 / Q8_0 grouped experts, attention) and the interleaved weight copies.
- `src/batch_host.inc`: `BatchState`, `forward_batch`, `batch_rollback`, the oracle test and the benchmark (included by `inkling_model.cu` before `main`).

## API
`forward_batch(Engine&, BatchState&, int B, int multi, cudaStream_t)`, capturable in a CUDA graph (one graph per (B, multi); `Runner` caches them).
Inputs live in device memory: `X.btok[B]` tokens, `X.pos_arr[B]` positions, `X.slot_arr[B]` sequence slots.
Outputs: `X.logits[B][NVOCAB_PAD]` (softcapped, first 200058 valid) and `X.barg[B]` (argmax per token).
B <= 16, position + B <= MAXT (1024, unchanged).

- `multi = 0` (same-sequence batch): tokens at positions p..p+B-1 of slot 0. State afterwards equals B sequential `forward_token` steps.
- `multi = 1` (parallel requests): token b belongs to its own sequence slot `slot_arr[b]` (distinct slots, 0..3, one token each) at its own position. Slot 0 is the engine's normal KV/state (so slot 0 interoperates with `forward_token`); slots 1..3 have their own KV caches (`kx`, `vx`) and conv states (`Sxk/Sxv/Sxp/Sxf`) per application. The same kernels serve both modes; only the conv-state source, KV base and position come from the per-token arrays. No extra kernel set was needed.

## Design notes
- Per-token buffers got a leading batch dimension: cur, prelude_cur, bank_last, blocks[B][24][E], bank[B][24][E], msgs[B][4][E], inbuf/xs3[B][3][E], q/k/v/r/gate, router, ids/wts[B][8], gu/hbuf/dn[B*8][...], logits[B][201024]. Scores/inverse-rms of the mix use a per-token stride. The stage copies of the original (states_buf) are gone: the mixes read `[bank rows..., bank_last]` directly with per-token strides.
- Conv states (K, V, attn post, ffn post, 48 applications): each application has a history buffer `H[3+16][C]`. The producing GEMV writes its output straight into rows 3..3+B-1. The conv kernel (one thread per channel, loops over b with a register window) first copies the OLD committed 3-row state into H rows 0..2, then computes all B outputs, then stores the new state (last 3 inputs). In multi mode each token uses the state of its slot instead.
- Rollback: `batch_rollback(g, X, j, stream)` sets the committed state of all 192 conv buffers to H rows j..j+2 (= state after j tokens, 0 <= j <= B) with one launch (table of 192 entries, j read from device memory, so it can be graph-captured). Valid until the next `forward_batch`. KV rows >= pos+j are stale but never read before they are rewritten (causal limit). Nothing else is carried between tokens (blocks/bank/msgs are recomputed per token).
- Attention: grid (8 kv heads, B), 512 threads; one block does the 3 query heads of a KV head (K/V rows read once, coalesced), causal limit `pos_arr[b]`, KV base by `slot_arr[b]`.
- MoE: `kb_group` builds per layer the union of the B*8 chosen experts (unique ids ascending, per expert the list of (token, slot)). Expert GEMV blocks are launched for min(128, 8B) expert indices (extra ones exit at once); each unique expert's weights are read once and applied to its tokens (4 tokens at a time, activations of those tokens in shared memory; more than 4 tokens on one expert loop and re-read the weights from L2). Both NVFP4 and Q8_0 expert layers are handled.
- Q8_0 GEMV: `gemvb2_q8<BT,KS,RG>`: lane = row, x staged in shared memory (256-k chunks), weights loaded once and applied to B accumulators.
- Interleaved weight copies (important, ~2x on heavy kernels): the lane-per-row kernels need contiguous warp loads, so at start-up `init_batch` builds a second, interleaved copy of every Q8_0 matrix, the head and all expert tensors on the GPU (`k_repack_il`). The original SoA copy stays for `forward_token`. Cost: roughly doubles the weight footprint of those tensors (the A6000 has room; start-up takes a few seconds longer).
- Numerics: per-token arithmetic of the small kernels is the original one; the heavy GEMVs, attention and the grouped experts use different summation orders (K-split warps, shuffle trees), which is the only source of differences.

## Validation (oracle: sequential `forward_token`, 3 teacher-forced sequences seq1..3 = 326/336/357 tokens)
Same-sequence batches, chunks of B = 1, 2, 4, 8, 16 with state carried between chunks, all positions compared (full 200058-logit vectors):

| sequence | max abs diff | max rel diff (abs / max|logit| of the position) | argmax mismatches |
|---|---|---|---|
| seq1 | 8.9e-5 | 3.4e-6 | 0 / 326 |
| seq2 | 6.7e-5 | 2.5e-6 | 0 / 336 |
| seq3 | 8.6e-5 | 3.1e-6 | 0 / 357 |

(identical numbers for every B: the batch result is independent of B and differs from the sequential kernels only by summation order.)

Rollback: B = 8 batch at p = 40, 120, 200 on each sequence, tokens beyond j replaced by garbage tokens, `batch_rollback(j)` for j = 0, 3, 8, then 6 single `forward_token` steps and a B = 4 batch. 27 cases, 0 argmax mismatches; continuation max rel diff 7.7e-7 (batch-prefix logits of the j real tokens agree to the same level). Negative control (`MIKASA_NEG_CONTROL=1` skips the rollback): continuation errors jump to rel 0.27 and argmax mismatches appear, so the test does detect a wrong state.

Multi-sequence mode: S = 2, 3, 4 slots, prompt files 1, 2, 3 and 1 again (slot 3, other start step), staggered start steps 0/30/70/110 so tokens in one step sit at different positions, B varies 1..4 during the run; each slot compared per position with its own independent sequential run: all positions, argmax mismatches 0, same max abs/rel differences as above (8.9e-5 / 3.4e-6 worst).

Final line of the test: `ORACLE TEST: argmax matches at every position`.

## Timing (RTX A6000, CUDA graph replay, 100 reps, real 300-token context, position 300)
`forward_token` graph (B=1 reference): 12.2 ms/token in this session (the 10.7 ms quoted for the original was measured at shorter context; this run is at position 300).

Same-sequence batch:

| B | ms per call | ms per token | tok/s | speedup vs forward_token |
|---|---|---|---|---|
| 1 | 13.19 | 13.19 | 76 | 0.93x (forward_token is better at B=1) |
| 2 | 14.16 | 7.08 | 141 | 1.73x |
| 4 | 17.13 | 4.28 | 234 | 2.86x |
| 8 | 27.25 | 3.41 | 294 | 3.60x |
| 16 | 48.45 | 3.03 | 330 | 4.05x |

Multi-sequence (S sequences, one token each per step, own KV/state, context 300 each):

| S | ms/step | aggregate tok/s | vs 1 sequence |
|---|---|---|---|
| 1 | 13.20 | 75.8 | 1.00x |
| 2 | 15.61 | 128.2 | 1.69x |
| 3 | 19.13 | 156.8 | 2.07x |
| 4 | 20.24 | 197.6 | 2.61x |

(For reference the single-token decode path gives 82 tok/s at this context: 1000/12.2.)

Breakdown at B = 8 (nsys, `prof_B8_breakdown.txt`, ms per forward_batch call under the profiler, 28 ms total): NVFP4 expert GEMV gate_up 6.5 + down 3.6 (36%); Q8_0 projections 6.4 + head 0.8 (26%); Q8_0 expert layers 2.9 (10%); attention 1.4 (5%); loop-control GEMV + gain/forget 1.7; mix 1.0; remaining small kernels ~4. At B = 1 the heavy kernels are ~9 of 13 ms. The many small kernels grow roughly 4 ms (B=1) -> 5.3 (B=2) -> ~8 (B=8), i.e. about +15-30 % per extra token at small B and +13 %/token overall, so the "< 30 % per token" target is met approximately; the heavy kernels now dominate.

## Unfinished / approximate / honest caveats
- B=1 through `forward_batch` is 8 % slower than `forward_token` (lane-per-row kernels, extra grouping); use `forward_token` for single tokens. At S=1 multi mode costs the same as B=1.
- The heavy kernels are not at the bandwidth limit: the Q8 and expert GEMVs are bound by shared-memory/FMA issue at B >= 4 (about 3x the ideal DRAM time at B=8). The obvious next step is fp16 tensor-core MMA with a hi/lo split of the activations (weights int8 / E2M1 are exact in fp16); not attempted.
- Tokens per expert above 4 (frequent at B = 16 for popular experts) loop in chunks of 4 and re-read the weights from L2; no tuning of NT.
- Interleaved weight copies roughly double GPU memory of the weights (fits on the 48 GB card; not measured exactly).
- Multi mode: each slot's token is a single token per step (no multi-token-per-sequence + multi-sequence mixing), slots limited to 4 (0..3), distinct slots per step. KV memory for 3 extra slots is 1.2 GB.
- tau scaling for positions beyond 8192 (full layers) is untouched by the tests (MAXT = 1024 stays).
- Only the three probe sequences were tested; positions up to ~360 (attention loop and SWA window < 1024 never exercised past the window).
- `--batch-bench` prefill uses the batch path itself (validated), not forward_token.
