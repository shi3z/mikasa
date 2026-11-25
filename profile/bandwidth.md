# Achieved memory bandwidth of the mikasa GEMV kernels (RTX A6000, peak 768 GB/s; 2026-11-25)
Nsight Compute could not be used: ERR_NVGPUCTRPERM (RmProfilingAdminOnly=1, no root). Substitute: nominal weight bytes per call (counted at graph capture, see inkling_model.cu / kernels_common.cuh)
divided by the per-instance kernel duration from the Nsight Systems trace (mikasa_decode.nsys-rep). These are NOT DRAM counter readings.

| kernel / shape | calls per token | avg us | MB per call | GB/s | % of 768 |
|---|---:|---:|---:|---:|---:|
| head (Q8, 201,024 x 2304) | 1 | 699 | 492.1 | 704 | 92 % |
| gemv_q8_rows wo (2304 x 3072) | 48 | 12.74 | 7.52 | 590 | 77 % |
| nvfp4_gemv_pack4<72> gate_up x8 | 44 | 43.24 | 23.89 | 552 | 72 % |
| gemv_q8_rows wq/wgate (3072 x 2304) | 96 | 13.76 | 7.52 | 546 | 71 % |
| nvfp4_gemv_pack4<36> down x8 | 44 | 26.53 | 11.94 | 450 | 59 % |
| gemv_q8_rows wk/wv (1024 x 2304) | 96 | 7.30 | 2.51 | 344 | 45 % |
| gemv_q8_split4 wrel | 48 | 4.32 | 0.94 | 218 | 28 % |
| q8_experts_rows2 (8 calls, 33.8 MB each) | 8 | 53.94 | 33.8 | 628 | 82 % |

Ceiling if every GEMV reached the head kernel's 704 GB/s: about 1.7 ms of 10.95 ms per token saved = +18 % tokens/s. Half of that (+8-9 %) is a more realistic target;
the small kernels (wk/wv 2.5 MB, wrel 0.9 MB) are latency/launch bound and will not reach 700 GB/s.

## Nsight Compute (counters, run under sudo; 2026-11-25)
DRAM bytes read per call equal the nominal weight bytes (gate_up 23.95 MB vs 23.89 nominal, down 12.01 vs 11.94, head 492.5 vs 492.1): the table above is valid.
| kernel | us (ncu, caches flushed) | DRAM % peak | ALU pipe % | issue slot % | warps active % | waves/SM | L2 hit |
|---|---:|---:|---:|---:|---:|---:|---:|
| head Q8 | 702.9 | 97 | 12 | 18 | 64 | 37.4 | 2 % |
| nvfp4 gate_up<72> | 52.2 | 69 | 35 | 41 | 67 | 1.5 | 9 % |
| nvfp4 down<36> | 29.7 | 65 | 37 | 44 | 93 | 1.1 | 21 % |
| q8 wq/wgate | 16.0 | 75 | 10 | 16 | 40 | 0.6 | 12 % |
| q8 wk/wv | 8.7 | 42 | 6 | 10 | 12 | 0.2 | 27 % |
Reading: INT32/ALU is at 35-37 % and issue slots under 45 %, so ALU contention is not the limiter; dominant stall is long scoreboard (DRAM latency).
gate_up has 1.5 and down 1.1 waves: the last, mostly empty wave is a quantization loss. Slot sweep (nvbench.cu): baseline gate_up 509 GB/s at 8 slots, 626 GB/s at 32; down 438 -> 584.
Even at 32-64 slots the kernel stays at ~80 % of peak (head kernel: 92-97 %).
Streaming loads (ld.global.nc.L1::no_allocate), bit-identical: microbenchmark gate_up -5.6 %, down -3 %; end to end 11,045 -> 10,985 us/token (-0.5 %, within run-to-run noise; 3 runs, same sign).
k_situ (the SwiGLU kernel) costs 1.79 us x 48 = 86 us per token, so fusing gate/up with SwiGLU saves at most 0.8 %.
