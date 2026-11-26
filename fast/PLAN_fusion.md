# Decode kernel-fusion plan (B=1, ctx 300 profile; target: 65K context at 100 tok/s)

Status: **first measurements done (section 4); the savings below were over-estimated, see section 4.** Plan A1 was prototyped and PDL was tested. Data: ncu over one full B=1 step (1,094 kernel launches of the CUDA graph, `ncu --csv`, metrics
`gpu__time_duration`, `dram__throughput`), file `fast/ncu/ncu_step.csv`, summed per kernel. ncu serialises and flushes caches, so its step total is 13.2 ms against
10.1 ms for the graph; "graph ms" below = ncu us x 0.77. Exactness rule: every fusion must keep each reduction's summation order (bit-identical token files).

## 1. Where the time goes (ncu, per step)
| kernel | calls | ncu us | share | DRAM % | verdict |
|---|---:|---:|---:|---:|---|
| nvfp4_exp2<2304> + <1152> (experts) | 88 | 3,202 | 24.3 % | 78 / 71 | bandwidth bound, near the ceiling |
| gemvb2_q8m<1> (fused q/gate/k/v/rel) | 48 | 2,013 | 15.3 % | 76 | bandwidth bound |
| gemvb2_q8<1,8,1> (wo etc.) | 48 | 1,125 | 8.5 % | **53** | under-utilised: retile |
| gemvb2_q8<1,1,8> (head) | 1 | 716 | 5.4 % | 95 | at the ceiling |
| q8_experts_ks<4> | 8 | 655 | 5.0 % | 61 | some room |
| attention part/combine | 96 | 1,111 | 8.4 % | 33 (ctx 300) | latency bound at ctx 300, bandwidth bound at 65K |
| kb_router_fused | 48 | 554 | 4.2 % | 27 | latency bound (11.5 us) |
| loop-control GEMVs (f32_row8, ctrl_gf) | 96 | 937 | 7.1 % | 22-54 | small, 5-16 us each |
| **small kernels, DRAM < 6 %** (mix x4, rms x4, moe_conv_res, conv4_kv/res, situ, clip_qr, kv_write, scale_tau, copy) | **~590** | **~3,000** | **~23 %** | 1-5 | launch/latency bound: 3-5 us each |

The big weight GEMVs (first five rows) are 59 % of the step and already at 71-78 % of DRAM peak (95 % for the head): not a fusion target. The ~590 small
kernels are 23 % of the step at 1-5 % DRAM: that is where fusion pays.

## 2. Fusions (in priority order; ncu us saved -> graph ms)
| # | fuse | launches removed | est. saving | exactness |
|---|---|---:|---:|---|
| A | RMSNorm (rms_rows9, rms_rows3_9, rms2_rows9[_j]) into the prologue of the next Q8 GEMV / router (every block recomputes the same row-sum in the same order) | ~145 | 600 -> **0.45 ms** | identical code, identical order |
| B | mix_scores2 + mix_out2 (+ the _j variants) into one kernel | ~96 | 410 -> **0.32 ms** | same softmax/reduction order |
| C | conv4_kv + clip_qr + kv_write | ~96 | 340 -> **0.26 ms** | element-wise |
| D | situ into the staging of nvfp4_exp2<1152>; moe_conv_res + conv4_res; attn_combine into the staging of the o-projection | ~144 | 580 -> **0.45 ms** | same arithmetic, no grid barrier needed (producer kernel finished) |
| E | retile gemvb2_q8<1,8,1> (53 % -> ~75 % DRAM) and q8_experts_ks (61 % -> ~72 %) | 0 | 470 -> **0.36 ms** | same order per row |
| F | merge the loop-control f32 GEMVs (row8_j + ctrl_gf_warp_j) | ~24 | 120 -> **0.09 ms** | same order |
| (G) | programmatic dependent launch to overlap the ramp/tail of the ~190 big GEMVs (5-8 us each, ~1 ms): **must be tested first, PDL is documented for sm_90+; may not work on sm_86** | 0 | 0..0.7 ms | n/a |

Total upper bound A-F: about 1.9 ms graph; realistic 60-70 %: **1.2-1.4 ms**. About 590 -> 230 small launches; 1,094 -> about 730 per step.

## 3. Can 65K reach 100 tok/s (< 10 ms/token)?
Measured B=1 (lean, FP16 KV): 10.08 ms at ctx 300, 13.54 ms at 65K -> 74 tok/s.
* Bandwidth floor: weights per token about 1.7 GB (NVFP4 experts) + 2.5 GB (Q8, head) at about 690 GB/s = 6.1 ms; FP16 KV of the 8 full-attention applications at 65K = 8 x 266 MB = 2.1 GB = 3.1 ms. Floor about 9.2 ms; the 65K attention delta (+3.5 ms) is already at that floor.
* Exact fusions A-F: 13.54 - (1.2..1.9) = **11.6-12.3 ms = 81-86 tok/s** at 65K (ctx 300: 8.2-8.9 ms = 112-122 tok/s).
* **100 tok/s at 65K is not reachable with bit-exact fusion alone**, it would sit at 108 % of the bandwidth floor. It needs fewer bytes: FP8 KV for the full-attention layers (-1.5 ms, not bit-exact, flips some argmax tokens like FP16 KV did: ~1-2 %) gives about 10.1-10.8 ms (93-99 tok/s); only with that plus a retile gain near the top of the range does 100 tok/s become borderline. Copy-heavy edits with speculation already exceed 235 tok/s.

## 4. Measured: PDL and fusion A1 (second round)
* **PDL does not work on sm_86.** `cudaTriggerProgrammaticLaunchCompletion` / `cudaGridDependencySynchronize` compile to `griddepcontrol`, which ptxas rejects below sm_90 ("Instruction 'griddepcontrol' requires .target sm_90 or higher"). The launch attribute `programmaticStreamSerialization` is accepted (stream and graph capture) but without the device-side trigger the next kernel starts only after the previous one ended (tested with a 30 us producer). Item G is dropped.
* **A launch is worth about 1.7 us in the graph, not 4.8 us.** Ablation: `kb_rms_rows3_9` (48 launches per step) removed completely (wrong results, timing only): B=1 9.92-9.96 -> 9.84-9.86 ms = **-0.07..-0.10 ms** (1.7 us per launch). The ncu duration of a tiny kernel (3-5 us) includes ramp and tail that the graph overlaps with its neighbours. So the upper bound of A (all RMSNorm launches, ~145) is about 0.25 ms, and of all ~590 small kernels about 1.0 ms if every one of them could be removed at zero cost.
* **A1 prototype (`fast/fusionA1_prototype.diff`)**: RMSNorm of the three attention-input rows done inside `gemvb2_q8m` (x staged once into shared memory, normalised there with the same per-thread order, butterfly and `(v * inv) * w` as `kb_rms_rows3_9`; BF16 sink/conv jobs normalise in registers), B <= 2. First version (norm before the weight prefetch, then a second pass over x): the Q8 kernel went from 38.7 to 52 us (+14 us, two dependent L2 round trips behind the saturated memory pipeline), step +0.4 ms. Final version (one load, norm in shared memory): B=1 9.88 ms fused vs 9.95 unfused vs 9.87 for the committed binary, B=2 12.22 vs 12.22 vs 12.25: **no net gain** (what the launch saves is paid back by the prologue, and the structural changes to the kernel cost about as much as the ablation bound). Not merged.
* **Revised outlook.** Fusing tiny kernels is worth at most 0.3-0.5 ms in total (3-5 % of the step), not the 1.2-1.9 ms of section 2. 65K: 13.54 -> about 13.0-13.2 ms = 76-77 tok/s. The step is dominated by bytes (weights 6.1 ms + 65K KV 3.1 ms at the DRAM ceiling), so the remaining levers are fewer bytes (FP8 KV, -1.5 ms, not exact), better DRAM efficiency of the 53 %/61 % kernels (item E, about 0.3 ms), and speculation.
