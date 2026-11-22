# 三笠 (Mikasa) — Inkling-2 Mini 推論エンジン (RTX 3090 / sm_86)

単一 CUDA 実装の Inkling-2 Mini 推論エンジンです。`unsloth/Inkling-2-Mini-GGUF` の
`Inkling-2-Mini-UD-Q4_K_XL.gguf` を mmap で読み、decode 1 トークンを 1 つの CUDA Graph で実行します。
llama.cpp は参照（ロジットと中間値の比較）にのみ使い、実行時には使いません。

## 現状

| 版 | ms/token | tok/s | 備考 |
|---|---:|---:|---|
| 初期 CUDA Graph 版 | 14.09 | 71 | |
| `baseline/ctrl_fused_12p40` | 12.40 | 80.6 | |
| `baseline/pack4_11p86` | 11.86 | 84.3 | NVFP4 expert GEMV の行パック化 |
| `baseline/fused_11p18` | 11.18 | 89.4 | D2D コピー削除・mix/conv 融合 |
| `baseline/small_kernel_9p85` | **9.80〜9.87** | **101〜102** | 小カーネル融合・register 常駐 RMS |

計測: `--gen 128 --reps 5`（RTX 3090、CUDA Graph 再生）。
128 トークン生成は参照出力と完全一致、logits は参照との最大差 7.6e-6、argmax は全位置一致。

## 構成

- `src/gguf.{hpp,cpp}`: GGUF v3 パーサ（mmap）
- `src/kernels_common.cuh`: 全カーネル（NVFP4 / Q8_0 / F32 / BF16 GEMV、RMS、mix、conv、attention、loop control）
- `src/inkling_model.cu`: モデル本体、`forward_token`、CUDA Graph 構築、`main`
- `src/*_bench.cu`, `src/nvfp4_*.cu`, `src/moe_sched.cu`: マイクロベンチマーク（各 GEMV 方式の比較）
- `ref/`: llama.cpp 参照用のダンパ（`dump_nodes.cpp`）と numpy リファレンス
- `baseline/`: 各段階のスナップショット（ソースと README）
- `DESIGN.md`: 設計メモ

## ビルド・実行

```sh
nvcc -O3 -std=c++17 -arch=sm_86 -I src -o mikasa src/inkling_model.cu src/gguf.cpp
./mikasa Inkling-2-Mini-UD-Q4_K_XL.gguf out_dir --gen 128 --reps 5   # 生成と計測
./mikasa Inkling-2-Mini-UD-Q4_K_XL.gguf out_dir --dump               # logits.bin を書き出し
```

モデルファイルはリポジトリに含みません。
