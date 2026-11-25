# Tensor-core prefill for the mikasa engine (Inkling-2-Mini, RTX A6000 sm_86)

Work copy `~/rrr/mikasa-prefill` (originals untouched, nothing committed). The old B=16 GEMV prefill is still in the same binary (baseline); the new path is selected with `--pf-fast PB`.
Everything below was measured with the final binary; run `tools/run_prefill_all.sh [build unit val needle timing slow]` to reproduce (logs in `logs/`).

```
nvcc -O3 -std=c++17 -arch=sm_86 -I src -o mikasa_pf src/inkling_model.cu src/gguf.cpp
./mikasa_pf $M out --lean --kv16 --nslots 4 --pf-fast 512 --prefill data/needle60k.txt --gen-long 600 logs/gen.txt
```
New flags: `--pf-fast PB` (chunk size; ring is auto-raised to 1024+PB), `--pf-all` (argmax of every token, validation only), `--pf-split N` (two prefill calls, second continues the state), `--force FILE` / `--gen-logits FILE` (teacher-forced continuation, per-step logits), `--time-depths`, `--layer-cmp N` (layer-by-layer residual comparison vs old path), `--moe-tb`.

## Result: speed (tok/s, GPU event time, real tokens, FP16 full-layer KV, 4 slots x 64K allocated)
Whole-prompt prefill (`logs/timing_prompt.log`, `logs/timing_baseline.log`):

| prompt tokens | baseline (B=16) | PB=256 | PB=512 | PB=1024 | speed-up PB=512 / 1024 |
|---|---|---|---|---|---|
| 256 | ~300 (not run separately, 512: 306) | 2798 | 2719 | 2726 | ~9x |
| 512 | 306 | 2722 | 3510 | 3527 | 11.5x / 11.5x |
| 4096 | 292 | 2545 | 3254 | 3743 | 11.1x / 12.8x |
| 16384 | 279 | 2417 | 3082 | 3509 | 11.0x / 12.6x |
| 32768 | 265 | 2278 | 2893 | 3269 | 10.9x / 12.3x |
| 61440 | 247 | 2081 | 2612 | 2916 | 10.6x / 11.8x |

(256-token prompt = a single partial chunk, dominated by weight streaming; its baseline was not measured separately, ~300 from the 512 run.)
Throughput of one more chunk at depth D with real tokens already in the cache (`logs/timing_depth.log`), tok/s:

| depth | PB=256 | PB=512 | PB=1024 |
|---|---|---|---|
| 0 | 2721 | 3431 | 3875 |
| 4K | 2425 | 3081 | 3509 |
| 16K | 2301 | 2855 | 3187 |
| 32K | 2070 | 2544 | 2831 |
| 60K | 1775 | 2160 | 2361 |

Targets: >=2500 tok/s at short context (met with PB>=512: 3.4-3.9K; PB=256 reaches 2.5-2.8K), >=1500 at 32K+ (met everywhere: 1.8-2.4K at depth 60K, 2.3-3.3K whole 32K/61K prompts). Baseline was 226-308. "About 10x" is reached at PB=512 (10.6-11.5x), 11.8-12.8x at PB=1024; PB=256 is 7-9x (weights are re-streamed per chunk).
Memory (lean, FP16 full KV, 4 slots x 64K): 29.5 GB after init incl. prefill buffers at PB=256, 30.9 GB at PB=1024 (31.1 after a 61K prefill). FP32 full KV would add ~8.6 GB (not run with the fast path except the 4096/60K tests below). Limit 42 GB respected.

## What was built
- `src/pf_gemm.cuh`: tensor-core GEMMs, `mma.sync.m16n8k16` FP16 in / FP32 accumulate. Weight rows are the A operand, tokens the B operand; weights are read straight from the existing interleaved global layout (no extra weight copy, no smem for weights), dequantised in registers: Q8_0 int8->FP16 exactly (scale applied to the FP32 partial per 32-k block), NVFP4 via `prmt` LUT, products value*scale exact in FP16. Activations FP16 via cp.async (3 stages, swizzled smem, k order permuted consistently for A and B). Kernels: `pf_gemm_dense` (fused q/k/v/gate/rel in one launch, wo), grouped MoE `pf_gemm_moe` (NVFP4 and Q8_0 experts, tiles of 32/64 tokens per expert from `pf_group` which sorts the (token,expert) assignments), 128 weight rows per block.
- `src/pf_attn.cuh` (written by a helper agent, reviewed/tested here): flash-style tensor-core attention, 3 GQA heads share K/V tiles, split-KV with zero sink logit, relative bias on tensor cores; ring (SWA, FP32 cache) and linear (FP16/FP32) caches. K/V/Q/P rounded to FP16 inside the mma (FP32 softmax stats).
- `src/pf_ops.cuh`, `src/prefill_host.inc`: batched small kernels, convs parallel over time (state commit identical to sequential), split-K FP32 small GEMMs (router, loop-control, sinks, conv taps), `forward_prefill`, `prefill_fast`. Head/logits/softcap/argmax only for the last token (or last k). Everything else (mix, norms, kv write...) reuses or generalises the existing kernels.
- Ring for SWA layers is 1024+PB rows (chunk writes cannot overwrite keys needed by its first query). Decode continues unchanged on the same caches.

## Validation (all with the final code)
Unit tests (`logs/unit_gemm.log`, `logs/unit_attn.log`): GEMM vs double-precision naive reference on synthetic weights in the real layout, dense and MoE, Q8_0 and NVFP4: rel error 3e-7 (Q8) / 4e-6 (NVFP4, FP32 accumulation of values ~3e3), "ALL OK". Attention vs `kb_attn_ref` (SWA, full FP16, full FP32, pos0 0..32000, T=256/200/37): max abs error 1e-3..2.3e-3 on gated values (FP16 output rounding level), no NaN.

End-to-end vs the existing path. The network is chaotic w.r.t. tiny perturbations (top-8 router near-ties; documented in RESULTS.md), so exact logit equality with a numerically different implementation is impossible. I therefore (a) did a teacher-forced 32-step continuation (the fast path is fed exactly the baseline's 32 greedy tokens after the prompt; argmax and full logits compared at every step), and (b) calibrated against the noise that already exists between the old path with FP16 vs FP32 KV (`logs/val_*.log`):

| prompt | PB | argmax mismatches / steps | relL2 of logits mean / max |
|---|---|---|---|
| 4096 (kv16) | 128,256,512,1024 | 0 / 32 (all four) | 7.6e-2 / 2.0e-1 |
| 4096 (kv32) | 512 | 0 / 32 | 8.4e-2 / 1.8e-1 |
| 3333 | 128..1024 | 1 / 32 (ref top1-top2 gap 0.12, test gap between the two tokens 0.007) | 6.1e-2 / 1.6e-1 |
| 1500 | 128..1024 | 0 / 32 | 8.5e-2 / 4.0e-1 |
| 512 | 128..1024 | 0 / 32 | 4.9e-3 / 2.2e-2 |
| 130 | 128..1024 | 0 / 20 | 1.3e-2 / 3.0e-2 |
| noise floor: old path FP16-KV vs FP32-KV, 4096 | - | 0 / 32 | 7.1e-2 / 1.9e-1 |

So the new path differs from the old one by about as much as the old path differs from itself when the KV cache is FP16 instead of FP32. Whole-prompt next-token agreement at every one of 4096 positions (`logs/val_argmax.log`): fast vs old 98.17% (kv16) / 98.14% (kv32); old kv16 vs old kv32 98.68%. A residual-stream comparison on the first 256 tokens (`--layer-cmp`) shows 2e-4 relative error in layers 0-3 (FP16 rounding level), then discrete jumps (router flips) and growth to 5e-2 at the end; baseline-vs-baseline with only the full-layer K/V rounded to FP16 grows to 2.4e-2, so the new path is ~2x noisier in that metric. I did not isolate which FP16 rounding (activations in the GEMMs, Q/P in attention) contributes most. Results are bit-identical for PB=128/256/512/1024 on the tested prompts (chunk boundaries aligned to 128).
- Not multiples of the chunk: 3333, 1500, 130 above (last chunk partial). Two prefill calls: split at multiples of 256 give bit-identical logits to one call (carried state exact); unaligned splits (1000, 1281, 300) give different results (relL2 up to 0.55 at 3333 tokens, 4e-3..1e-2 at 512 tokens, comparable to the old-vs-new difference at that length, argmax mismatches 1-2 of 32 at near ties). My explanation (attention key tiling depends on the chunk start alignment, so summation order changes, then chaos amplifies) is a hypothesis, not proven.
- Other KV slots (2, 3 of 4): generated tokens identical to baseline (`logs/val_slots.log`).
- Needle (`logs/val_needle.log`, PB=512, kv16 unless noted): 29806 tokens 5%/50% depth -> 7391, 7391; 59865 tokens 5%/50% -> 7391, 7391; 59865 tokens FP32 KV -> 7391. Prefill 2.9K / 2.6K tok/s.
- Not run: a 16K teacher-forced comparison against the old path (the script runs it with `slow`), FP32-KV at 16K+, positions >= 61440 with data.

## What limits speed
Baseline nsys (512 tokens, `logs/nsys_baseline_top.txt`): NVFP4 expert GEMV 39%, Q8 GEMV (projections+head for every token) 24%, Q8 experts 10%: weights re-read every 16 tokens. After (PB=512, 4K prompt, `logs/nsys_fast_pb512_top.txt`): MoE GEMMs 38% (NVFP4 34%, Q8-expert layers 4%), dense Q8 GEMMs 17%, attention ~10% at 4K (grows with depth: 3.3 GFLOP/token at 32K), loop-control 4.5%, mix kernels 6%, split-K small GEMMs 4%, SITU+expert-sum 4%, rest small. The GPU is >95% busy (no launch gaps; no CUDA graph used).
- MoE at PB=256..512 is partly weight-bandwidth bound: all 128 experts are touched every layer (570 MB/layer-app, ~500 GB/s achieved); ablating the mma or the dequantisation did not change kernel time much. Doubling the chunk (1024) halves per-token weight traffic: +15% (3743 vs 3254 at 4K). At PB=1024 it is ~53 TFLOP/s, i.e. ~1/3 of the measured mma peak (154 TFLOP/s with FP32 accumulate), limited by dequant ALU and loads.
- Dense Q8: ~50-60 TFLOP/s, FP32 FMA scale application and 245 registers (2 blocks/SM).
- Attention: ~38 TFLOP/s on long full-attention chunks, 17 TFLOP/s on SWA chunks (0.19 ms vs 0.15 ms target).

## Not done / approximate
- FP16 operands: activations are rounded to FP16 for every projection and expert GEMM, Q/K/V/P to FP16 in attention (SWA K/V FP32 caches are rounded when staged). Noise vs the old path is about the old FP16-vs-FP32-KV noise (above). Router, loop control, sinks, conv taps stay FP32 (split-K, deterministic summation order different from the old kernels).
- Chunk size: the brief suggested 128-512; 1024 is faster but needs ring 2048 rows (more SWA memory, included in the numbers). `--moe-tb 0` auto-selects 64-token expert tiles for chunks >=512 else 32.
- No CUDA graph; no fusion of SITU into the gate_up GEMM; mix/ctrl/rms kernels untuned (~15% of time); `pf_attention` allocates its scratch lazily and holds it in a static (single stream only).
- ncu could not be used (no counter permission), so the efficiency analysis rests on nsys and ablation timings.
- The head runs only for the last token(s); `--pf-all` computes it for all tokens through the old GEMV (validation only, slow).
- Timings are single runs (minimum of 3 for depth numbers); the second GPU user (my helper agent) was finished before any timing in this file.
