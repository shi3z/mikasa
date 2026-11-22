# Mikasa (三笠) — Inkling-2 Mini Inference Engine

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
