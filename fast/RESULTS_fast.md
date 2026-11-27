# RESULTS_fast: faster exact decoding in the lean engine (RTX A6000, 2026-11-26)

Everything here is exact: **every generated token file and every logit dump is bit-identical to the baseline** (commit 360dcc3). Work was done only in `~/rrr/mikasa-fast`, nothing committed.

## Headline
* Lean B=1 step (`forward_batch`, what the daemon and the spec loop run): **12.3 -> 9.9 ms (78.6 -> ~97 tok/s plain, 1.25x)**. The old non-lean `forward_token` graph measures 10.63 ms/token on this machine today (`./mikasa_fast model out --gen 128 --reps 5`), so the lean path is now 7 % faster than it, instead of 15 % slower.
* Verify passes B=2..16: 1.12x - 1.22x faster (table below). The B=5 and B=9 jumps went from +4.8/+6.3 ms to +3.1/+2.6 ms (the remainder is real: every extra token brings about 6-8 new experts = 1.2-1.6 GB of weights, about 2.1 ms at 768 GB/s).
* Spec decoding (`--spec 15 --spec-m 1 --spec-copy 4`), 15 workloads: 1.10x - 1.41x over the old spec numbers, and 1.3x - 3.4x over old plain decoding.
* Long context S=1 decode (ms): 300 ctx 12.6 -> 10.1, 64K ctx 16.1 -> 13.6 (attention at 64K is at the FP16-KV bandwidth floor already).

## Validation (all on the final binary `mikasa_fast`, built with the README command)
| check | result |
|---|---|
| `tools/run_bitexact.sh`: md5 of logits of 48 B=1 decode steps and of the last 32 positions for same-sequence batches B=2,3,5,8,9,13,16 (`--pf-b B`) | identical to baseline in all 8 cases (and the baseline itself gives the same md5 for every B: per-token arithmetic is B-independent; the changes keep that) |
| `tools/run_fast_check.sh`: plain (`--spec 0`) and spec token files, 1000 tokens, 15 workloads (30 files) | all 30 identical to the baseline run **and** to the committed `spec/logs/*.old.txt` |
| `--batch-test` (non-lean oracle, `--probe ~/rrr/spectral-ai-src/spec_probe`) | `ORACLE TEST: argmax matches at every position` (max abs 1.6e-4, rel 6.0e-6 as before) |
| `--multi-check` (4 slots, multi-sequence step vs single) | max diff 0 in all 4 slots (bit-identical, as before) |
| `tools/run_daemon_tests.sh build refs daemon` (exact, cancel, conc, cache, sample, limit) | all pass (21 IDENTICAL checks, cancel 10 ms / 114 ms, daemon refs equal committed logs) |

## What changed (all in `src/batch_*.{cuh,inc}`; arithmetic order per output element is unchanged)
1. **One launch for wq, wgate, wk, wv, wrel (+ the tiny BF16 sink / conv-projection GEMVs as extra blocks)**: `gemvb2_q8m` (`Q8Jobs`). Saves 4-7 launches per layer and overlaps the small matrices (wk/wv ran at 45 % of bandwidth). Largest single gain (about 1.2 ms at B=1).
2. **Q8 GEMV tile choice per B**: BT = B exactly for B<=8 (templates 1..8), for B>=9 two token groups of ceil(B/2). Token groups are *adjacent in the grid* so the second group hits the first group's weights in L2 (previously a second pass re-read DRAM). Removes most of the old BT=4->8->16 cliffs. For BT<=2 a variant stages x once and prefetches 3 blocks deep (`q8_body_full`), wo 957 -> 853 us.
3. **Q8_0 experts** (4 layers): K split over 8 warps with the per-block partials stored and re-accumulated in the original order by one FMA chain (`q8_experts_ks`): 874 -> 555 us. (Bit-exact because the order of `acc += d*t` is kept; FMA contraction matches.)
4. **Horizontal fusion of independent chains**: the two bank mixes + the attention-input mix in one launch pair (`kb_mix_scores_j/out_j`), the 3 loop controls of the attention inputs in one launch per stage (`*_j` kernels), the ffn_inp copy folded into `kb_conv4_res`.
5. **Router**: gemv + top-8 + expert grouping in one kernel for B<=2 (last-block ticket; top-8 by rank counting instead of 8 sequential reductions) 555 -> 388 us.
6. Small: `gemvb_f32_row8` loads hoisted, 4-way `block_sum4` in the mix kernel, multi-block arg-max (39 -> 7 us), Q8-expert loads 4-deep.
Launches per token: 1738 -> 1094 (B=1).

## Time per step vs B (distinct tokens, synthetic zero KV, 40 graph reps; `--time-b`, `TIMEB_DIVERSE=1`)

ctx 300 (ms per forward_batch call):

| B | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| before | 12.3 | 14.3 | 17.1 | 18.7 | 23.4 | 25.4 | 27.6 | 29.6 | 35.9 | 38.6 | 40.7 | 42.9 | 45.2 | 47.8 | 49.5 | 52.0 |
| after | 9.9 | 12.3 | 15.1 | 16.6 | 19.6 | 21.6 | 23.2 | 25.4 | 28.0 | 30.4 | 32.9 | 35.0 | 36.8 | 38.8 | 41.1 | 43.2 |
| speedup | 1.24x | 1.16x | 1.13x | 1.13x | 1.19x | 1.18x | 1.19x | 1.16x | 1.28x | 1.27x | 1.24x | 1.23x | 1.23x | 1.23x | 1.20x | 1.20x |
| after, ms/token | 9.92 | 6.17 | 5.04 | 4.14 | 3.92 | 3.60 | 3.32 | 3.18 | 3.11 | 3.04 | 2.99 | 2.91 | 2.83 | 2.77 | 2.74 | 2.70 |
| increment vs B-1, before / after | - | 2.0/2.4 | 2.8/2.8 | 1.6/1.4 | 4.8/3.1 | 2.0/2.0 | 2.2/1.7 | 2.0/2.2 | 6.3/2.6 | 2.7/2.4 | 2.1/2.5 | 2.2/2.1 | 2.4/1.8 | 2.5/2.0 | 1.7/2.3 | 2.5/2.1 |

ctx 4000 (ms per forward_batch call):

| B | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| before | 12.9 | 14.6 | 17.6 | 19.3 | 24.4 | 27.1 | 29.2 | 31.0 | 38.1 | 40.8 | 43.1 | 45.0 | 48.2 | 50.8 | 53.0 | 55.8 |
| after | 10.5 | 12.6 | 15.3 | 16.8 | 19.8 | 22.8 | 24.6 | 26.5 | 29.8 | 32.2 | 35.0 | 36.8 | 39.4 | 41.6 | 44.2 | 46.7 |
| speedup | 1.23x | 1.17x | 1.15x | 1.15x | 1.23x | 1.18x | 1.19x | 1.17x | 1.28x | 1.26x | 1.23x | 1.22x | 1.22x | 1.22x | 1.20x | 1.20x |
| after, ms/token | 10.52 | 6.28 | 5.11 | 4.21 | 3.96 | 3.81 | 3.51 | 3.31 | 3.32 | 3.22 | 3.18 | 3.07 | 3.03 | 2.97 | 2.95 | 2.92 |
| increment vs B-1, before / after | - | 1.7/2.0 | 3.0/2.8 | 1.6/1.5 | 5.1/3.0 | 2.7/3.1 | 2.1/1.7 | 1.8/1.9 | 7.1/3.3 | 2.7/2.4 | 2.3/2.7 | 2.0/1.9 | 3.1/2.6 | 2.7/2.2 | 2.2/2.6 | 2.7/2.4 |

The old calibration in the spec loop uses the same token repeated; with distinct tokens the costs are practically the same (checked), so the adaptive draft-width selection sees the cheaper steps automatically.

Long-context decode (`--time-decode`, ms/step, 4 slots allocated, maxctx 65536):

| context | S=1 before | S=1 after | S=4 before | S=4 after |
|---:|---:|---:|---:|---:|
| 300 | 12.56 | 10.05 | 20.37 | 18.29 |
| 1024 | 13.09 | 10.59 | 21.72 | 19.64 |
| 8192 | 13.23 | 10.73 | 22.53 | 20.37 |
| 32768 | 14.43 | 11.90 | 27.35 | 25.11 |
| 65000 | 16.12 | 13.58 | 33.78 | 31.54 |

## tok/s per workload (1000 tokens, `--lean --kv16 --nslots 1 --maxctx 8192 --pf-fast 512`; "plain" = `--spec 0`, "spec" = `--spec 15 --spec-m 1 --spec-copy 4`)
| workload | tokens | plain before | plain after | spec before | spec after | spec after/before | spec after / plain before | tokens/step (spec, after) | token files identical |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| edit_diff_cpp_direct | 1000 | 77.4 | 95.8 | 118.9 | 140.3 | 1.18x | 1.81x | 2.13 | yes |
| edit_toolcall_py_direct | 1000 | 78.0 | 95.8 | 113.5 | 131.6 | 1.16x | 1.69x | 2.10 | yes |
| edit_whole_py_direct | 1000 | 77.3 | 95.4 | 104.1 | 122.5 | 1.18x | 1.58x | 1.63 | yes |
| editcopy_file_direct | 1000 | 77.8 | 95.4 | 223.8 | 264.7 | 1.18x | 3.40x | 7.51 | yes |
| editcopy_jsonl_direct | 1000 | 77.3 | 95.4 | 211.8 | 242.1 | 1.14x | 3.13x | 6.36 | yes |
| editcopy_replace_direct | 1000 | 77.5 | 95.8 | 133.3 | 147.2 | 1.10x | 1.90x | 2.58 | yes |
| gen_c_vec_direct | 737 | 79.8 | 98.7 | 83.3 | 102.7 | 1.23x | 1.29x | 1.09 | yes |
| gen_py_logstats_direct | 1000 | 79.2 | 98.3 | 86.7 | 105.7 | 1.22x | 1.33x | 1.17 | yes |
| gen_py_lru_direct | 1000 | 79.1 | 98.1 | 95.7 | 112.3 | 1.17x | 1.42x | 1.30 | yes |
| reason_essay | 1000 | 79.2 | 98.6 | 86.4 | 103.9 | 1.20x | 1.31x | 1.19 | yes |
| reason_math | 1000 | 79.4 | 98.2 | 92.4 | 111.7 | 1.21x | 1.41x | 1.27 | yes |
| reason_puzzle | 1000 | 79.2 | 98.0 | 88.8 | 103.9 | 1.17x | 1.31x | 1.16 | yes |
| shell_find_direct | 37 | 78.5 | 97.5 | 153.7 | 216.7 | 1.41x | 2.76x | 4.00 | yes |
| shell_pipeline_direct | 63 | 80.6 | 101.5 | 94.8 | 110.1 | 1.16x | 1.37x | 1.35 | yes |
| shell_script_direct | 85 | 77.3 | 95.6 | 105.8 | 120.6 | 1.14x | 1.56x | 1.79 | yes |


Noise: the spec path chooses its draft width from timed costs, so the *token path is identical but the step count varies between runs*. On `editcopy_file_direct` the baseline gave 207-221 tok/s in three runs (140-181 steps), the final binary 238 (181 steps, 3 of 3) in an earlier build and 265 (7.5 tokens/step) in the final validation run; `edit_toolcall_py_direct`: baseline 112.6-113.1, after 130.4-131.6 (`logs/final/spec_repeat.txt`). The table is one run per cell; treat differences below about 5 % on the copy-heavy rows as noise.

## nsys: kernel time per step by family (45 graph launches, 1 slot, ctx 300; `tools/nsys_prof.sh`, `tools/nsys_families.py`; CSVs in `prof/`)
B=1:

| kernel family | before us/step | calls | after us/step | calls |
|---|---:|---:|---:|---:|
| Q8 projections / wo / head GEMV | 4572 | 289 | 3308 | 97 |
| NVFP4 expert GEMV | 2679 | 88 | 2831 | 88 |
| Q8_0 expert GEMV | 877 | 8 | 555 | 8 |
| attention (+KV write, clip, conv) | 1012 | 256 | 1029 | 256 |
| block mixes | 919 | 288 | 600 | 192 |
| loop control | 997 | 288 | 732 | 144 |
| router (GEMV, top-8, grouping) | 555 | 144 | 388 | 48 |
| tiny BF16 GEMVs (sinks, conv proj) | 240 | 64 | 0 | 0 |
| other small (norms, copies, conv, situ, moe sum, argmax) | 718 | 309 | 635 | 261 |
| **total** | **12569** | **1734** | **10078** | **1094** |

B=16:

| kernel family | before us/step | calls | after us/step | calls |
|---|---:|---:|---:|---:|
| Q8 projections / wo / head GEMV | 12731 | 289 | 6867 | 97 |
| NVFP4 expert GEMV | 22406 | 88 | 23164 | 88 |
| Q8_0 expert GEMV | 5711 | 8 | 3727 | 8 |
| attention (+KV write, clip, conv) | 3007 | 256 | 2998 | 256 |
| block mixes | 1085 | 288 | 819 | 192 |
| loop control | 2942 | 288 | 2532 | 144 |
| router (GEMV, top-8, grouping) | 1147 | 144 | 1232 | 144 |
| tiny BF16 GEMVs (sinks, conv proj) | 871 | 64 | 0 | 0 |
| other small (norms, copies, conv, situ, moe sum, argmax) | 1883 | 309 | 1857 | 261 |
| **total** | **51784** | **1734** | **43197** | **1190** |


## Why was lean B=1 slower than `forward_token`?
From the first nsys tables (sum of kernel time, gaps between kernels are about 0): the Q8 GEMVs ran as 6 separate launches per layer, three of them (wk, wv, wrel) at 28-45 % of bandwidth (4.57 ms vs about 3.3 ms in forward_token), the Q8 expert kernel was 2x slower than the SoA one (874 vs 431 us: one dependent load per K block, only 144 blocks), mix scores + out took 919 us (fused `k_mix_fused` 578 us), loop control 3 launches, router 3 launches, tiny kernels never overlapped. I did not port the SoA `forward_token` kernels: lean mode has no SoA weights and their summation order differs, which would break bit-identity between B=1 and the verify passes. Instead the same ideas were applied to the interleaved kernels, with the per-element arithmetic kept.

## What did not work (measured, reverted or left off)
* **x staged in chunks / more occupancy for `nvfp4_exp2` (K=2304)**: no change at B=1, +3-6 % slower at B>=9. Reverted.
* **K split of the NVFP4 expert GEMV over 2-4 warp slices with ordered re-accumulation** (`nvfp4_exp_ks`, bit-exact): 4x the warps in flight, identical time. Code kept, disabled (`KS_B = 0`).
* **Fused gate_up -> SITU -> down kernel with a grid barrier** (`nvfp4_moe_fused`, bit-exact, B<=2): 64.8 vs 65.9 us per layer (0.5 %); needs co-resident blocks. Off by default (`MIKASA_MOEFUSE=1`).
* **Byte-pair float2 LUT for NVFP4 decode**: shared-memory bank conflicts, +5 % slower. Reverted.
* **Fusing the attention-input RMSNorm into the Q8 projection kernel, and PDL** (details in `fast/PLAN_fusion.md` section 4, prototype `fast/fusionA1_prototype.diff`): PDL needs `griddepcontrol` (sm_90+), not available on sm_86. Removing `kb_rms_rows3_9` entirely saves only 0.07-0.10 ms per token (1.7 us per launch in the graph, not the 4.8 us ncu duration), and the fused kernel pays that back in its prologue: B=1 9.88 ms fused vs 9.95 unfused vs 9.87 committed. A first version that normalised before the weight prefetch with a second pass over x cost +14 us per projection kernel. Not merged.
* **Two weight rows per thread in `nvfp4_exp2` (x read from shared memory once for 2 rows; `MIKASA_NVR=2`)**: bit-identical, but **slower**: B=16 42.0 -> 45.1 ms, B=8 24.5 -> 26.0 ms, B=4 16.1 -> 16.5 ms (ctx 300). The shared-memory traffic that matters is not the x broadcast (8 LDS.128 per token and 32-k half) but the `kt[]` lookups of the weights (32 LDS per row and half), which this doubles per thread while halving the blocks. Left in the code as an option (default 1 row).
* **Cheaper NVFP4 dequantisation in the prefill GEMM (`pf_nv4_to_h2`, patch: `fast/nv4_dequant_prefill.diff`)**: the ue4m3 scale is built from its bits (`(byte & 0x7F) << 7` read as FP16 = value * 2^-8; the factor 128 and the 0.5 moved into the E2M1 LUT, whose entries are now the high bytes of 128 * {0,1,2,3,4,6,8,12}) instead of float arithmetic + cvt, and the sign is merged by a `prmt` in msb-replicate mode + one `lop3`. Bit-identical products (same exact FP16 value*scale), unit tests "ALL OK", token MD5 of a 4K prefill + 32 tokens identical to the old binary. Micro-benchmark (`pf_gemm_bench`, T=512, gate_up, random routing): TB=16 1.055 -> 0.746 ms, TB=32 0.906 -> 0.669 ms (-26 %), but TB=64 0.740 -> 0.733..0.882 ms (no gain). **End to end no gain**: 4K prefill, PB=512: 3,375 tok/s (old, auto TB) vs 3,361 (new, `--moe-tb 32`), 3,225 (new, auto = TB 64). Reasons: at PB=512..1024 the MoE GEMMs are bound by weight bandwidth (about 500 GB/s) and the mma issue rate, not by the dequantisation ALU (same conclusion as the ablation in `long/RESULTS_prefill.md`); with TB=64 the added hoisting raised `pf_gemm_moe<NV4,64>` from 164 to 231 registers, which lowers occupancy (2 instead of 3 blocks/SM); forcing `__launch_bounds__(128, 4)` on all TB=64 instantiations made the Q8 expert kernel spill and cut the 4K prefill to 1,960 tok/s. Not merged. The decode kernels (B=3..16) were not changed: the dequantised weight is reused for B tokens and the B=1 ablation above already showed < 1 %.
* **L2 prefetch of the next layer's wq in a parallel graph branch** during the small-kernel phase: no gain (+0-2 %). Reverted.
* **Ablation at B=1**: removing the NVFP4 decode, or the FMAs, from `nvfp4_exp2` changes the step time by < 1 %. So the NVFP4 kernels at B=1 are not instruction bound; they sit at about 560 GB/s (head GEMV 692 GB/s) and the cause (about 5-8 us per launch of ramp/tail, or DRAM access pattern) is not understood. Fusing launches, more warps, prefetch did not move it. At B=16 the same kernel is 2.4x above its byte floor (compute/LDS bound).
* Fusing the router for B>2 and keeping Q8 FULL staging for BT 3-4: slower or no gain.
* Host side: the spec loop has no measurable gap (launch 8-14 us, between-steps host 2-3 us, sync wait = GPU time), so no work was needed there. Attention at 64K context is at the FP16 KV bandwidth floor (8 layers x about 0.4 ms).
* `ncu` is not available; wanted counters are in `ncu_wanted.sh`.

## NVFP4 decode without shared-memory LUT, and chunk 1024 (second round)
* **`nv4_dec_word` (batch_kernels2.cuh)**: the decode of `nvfp4_exp2` takes the doubled-E2M1 value from `prmt` (LUT in two registers) + sign by `prmt`/`lop3`, then `cvt` half2 -> float2, instead of 32 `kt[]` LDS per row and half. Same float values, so bit-identical: token MD5 identical for 5 workloads x (plain, speculative), 600 tokens (`--spec 0` and `--spec 15 --spec-m 1 --spec-copy 4`). `--time-b` (ctx 300, ms per call, old -> new): B=4 16.08 -> 16.16, B=8 24.54 -> 24.29, B=12 33.85 -> 33.18, B=16 41.96 -> 40.95 (-2.4 %). tok/s: plain 95.7-99.9 -> 95.9-100.4, speculative copy-heavy 244.8 -> 246.0, other speculative +0.2-1.1 %. A small, exact gain (and the structure the next ncu round suggests: the B>=3 NVFP4 kernels are limited by instruction issue, not DRAM).
* **Prefill chunk 1024 (`--pf-fast 1024`) re-checked with the fast kernels** (`--lean --kv16 --nslots 1 --maxctx 65536`, 32 greedy tokens after the prompt, RTX A6000, GPU time):

| prompt | chunk 512 | chunk 1024 | gain | token file MD5 |
|---:|---:|---:|---:|---|
| 4,096 | 3,334 tok/s | 3,807 tok/s | +14 % | identical |
| 16,384 | 3,134 | 3,552 | +13 % | identical |
| 61,440 | 2,638 | 2,919 | +11 % | identical |

  VRAM (nvidia-smi peak, 1 slot): 20.8 GB -> 21.5 GB (+0.69 GB). The daemon configuration (4 slots x 64K): 29.75 GB -> 30.87 GB (+1.12 GB: the SWA rings of all slots grow to 1024 + PB rows, plus the prefill buffers). Unit tests (`pf_gemm_test`, `pf_attn_test`): ALL OK. Daemon test chain with `--pf-fast 1024 --pf-chunk 1024` (exact, cancel, conc, cache, sample, limit; 5 reference prompts, prefix cache, concurrent requests, cancel during a 26K prefill): every comparison IDENTICAL; 65K-token prefill 2,837 tok/s; cancel during prefill 115 ms. The daemon now defaults to `--pf-fast 1024 --pf-chunk 1024` (while other requests are generating the chunk stays 256).

## Remaining ideas, ranked by expected gain
1. NVFP4 expert kernels for B=3..16 (22.9 of 43 ms at B=16, floor about 9.6): cut instruction count (LUT-free decode, 2 tokens per x load, NT=8 for hot experts). Maybe -3 to -6 ms at B=16, +10 % on copy-heavy spec. Exactness rules out tensor cores.
2. Explain and remove the 560 vs 690 GB/s gap of the big GEMVs (about 190 launches x 5-8 us = 1.0-1.5 ms): ncu on `gemvb2_q8m`, `nvfp4_exp2` (stalls, DRAM page behaviour, TLB) is the next step; then possibly a persistent per-layer kernel.
3. SWA attention (`kb_attn_part` + combine, about 0.56 ms at B=1): fewer barriers, one block per kv-head over all chunks (needs care to keep the softmax order).
4. Remaining small-kernel fusion (conv4_kv with clip_qr, rms into the router, situ into down, fused router for B up to 4): about 0.3-0.4 ms together.
5. Spec policy: the draft-width chooser is noisy (140 vs 181 steps for the same output); using a smoother cost model could gain 5-10 % on copy-heavy prompts.
6. Wide verification for B>16 is still approximate-greedy (unchanged).

## Reproduce
```
nvcc -O3 -std=c++17 -arch=sm_86 -I src -o mikasa_fast src/inkling_model.cu src/gguf.cpp
tools/build_base.sh                          # ./mikasa_base = original kernels + instrumentation only
tools/run_bitexact.sh ./mikasa_fast fast      # logs/bitexact/fast/md5.txt (compare with base)
tools/run_fast_check.sh ./mikasa_fast fast    # tok/s + token files (logs/fast/fast); then: tools/run_fast_check.sh --cmp base fast
TIMEB_DIVERSE=1 ./mikasa_fast MODEL out --lean --kv16 --nslots 1 --maxctx 8192 --time-b
tools/nsys_prof.sh ./mikasa_fast TAG 16       # per-kernel table for one B
```
New flags/env: `--time-b`, `TIMEB_ONLY`, `TIMEB_DIVERSE`, `MIKASA_Q8BT` (tile table), `MIKASA_NOFUSE` (old router path), `MIKASA_MOEFUSE`, `MIKASA_Q8MFULL`.

## FP8 (E4M3) KV cache for the full-attention layers (`--kv8`, opt-in, not bit-exact)

Full-attention K/V rows are stored as E4M3 bytes (SWA rings stay FP32). sm_86 has no FP8 conversion instructions: the store uses
`__nv_cvt_float_to_fp8` (SATFINITE), the load decodes with `prmt` + shift + `hmul2 * 256` (exact E4M3 -> FP16, subnormals included).
Decode `kb_attn_part`, the reference attention and the tensor-core prefill attention (`pf_attn_kernel`, FP8 -> FP16 into the smem tile)
all take the new element type `fp8e4`. The default path (`--kv16`) is unchanged: logits and token files have the same MD5 as before.
A global scale was not needed (V is RMS-normalised, K is norm-clipped).

Speed (RTX A6000, B=1 CUDA graph, `--time-decode`, ms/step): ctx 300 10.11 -> 10.10, 8K 10.78 -> 10.64, 32K 11.93 -> 11.18,
**65K 13.62 -> 12.19 (73 -> 82 tok/s)**. B=16 verify: unchanged to +0.4 ms (software decode cost). Prefill at 61K: 3,032 -> 2,993 tok/s (-1.3%).
VRAM: -1.0 GB per 61K-token slot (-4 GB with 4 slots).

Quality (teacher-forced on the FP16 greedy tokens, 128 steps, logits vs FP16):

| test | KL(FP16 \|\| test) mean / max (nats) | top-1 agree |
|---|---|---|
| FP32 KV (reference of the FP16 rounding itself) | < 1e-4 | 128/128 |
| FP8 K only (sim, FP16 storage rounded through E4M3) | 0.0033 / 0.050 | 126/128 |
| FP8 V only (sim) | 0.0016 / 0.036 | 126/128 |
| FP8 K+V, 16K context (real kernel, = sim 3 bit for bit) | 0.0026 / 0.075 | 128/128 |
| FP8 K+V, 61K context | 0.0079 / 0.083 | 122/128 |

Free-running greedy generation at 16K: 128/128 tokens identical to FP16; at 61K it diverges at token 1 (a near tie, the prompt is raw
documentation text). Needle-in-a-haystack (30K and 60K, depth 0.1 / 0.5 / 0.9, thinking on, 400 tokens): 6/6 answer 7391 with FP8, 6/6 with
FP16; 5 of the 6 outputs are token-identical including the reasoning. The relative L2 of the logits (0.18) is not a good metric: MoE routing
flips make it saturate (FP32 vs FP16 already gives 0.027); use the KL.

Daemon (`--serve --kv8`): `exact` tests (speculation on/off vs the single-sequence FP8 loop) 10/10 IDENTICAL, prefix-cache turns, branches,
cancel, concurrency, sampling and the 65K limit pass. One difference: "resident continuation" (generated tokens are in the cache, written
by the decode kernels; the uncached run re-writes them with the prefill kernels) diverges at token 64, because decode and prefill K/V differ in the last
FP32 bits and E4M3 rounds that to a different byte more often than FP16 does. Cache hits are therefore not exactly reproducible against a recompute with `--kv8`.

Verdict: -1.4 ms at 65K (-10%) and half the KV memory for a small, measurable distribution shift; it is not exact, so it is opt-in. 100 tok/s at 65K
is still out of reach (12.2 ms). Experiment switch: `MIKASA_KV8SIM=1|2|3` (FP16 storage rounded through E4M3 for K / V / both).
