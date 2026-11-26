#!/bin/bash
# ncu command lines wanted (run with sudo by the coordinator). Each profiles a few launches of one dominant kernel family.
# TIMEB_ONLY=<B> makes --time-b run only that batch size (TIMEB_DIVERSE=1: distinct tokens). Metrics: DRAM throughput, occupancy, waves, stall reasons.
M=$HOME/rrr/gguf/Inkling-2-Mini-UD-Q4_K_XL.gguf
A="$M out --lean --kv16 --nslots 1 --maxctx 8192 --time-b"
SEC="--section SpeedOfLight --section Occupancy --section LaunchStats --section WarpStateStats --section MemoryWorkloadAnalysis"
for B in 1 4 16; do
  export TIMEB_DIVERSE=1 TIMEB_ONLY=$B
  /usr/local/cuda/bin/ncu $SEC -k regex:gemvb2_q8m  --launch-skip 10 --launch-count 3 ./mikasa_fast $A
  /usr/local/cuda/bin/ncu $SEC -k regex:"gemvb2_q8<" --launch-skip 10 --launch-count 3 ./mikasa_fast $A
  /usr/local/cuda/bin/ncu $SEC -k regex:nvfp4_exp2 --launch-skip 10 --launch-count 4 ./mikasa_fast $A
  /usr/local/cuda/bin/ncu $SEC -k regex:q8_experts_ks --launch-skip 2 --launch-count 2 ./mikasa_fast $A
  /usr/local/cuda/bin/ncu $SEC -k regex:kb_attn_part --launch-skip 10 --launch-count 3 ./mikasa_fast $A
done
