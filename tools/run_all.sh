#!/bin/bash
# full validation chain with the final binary. Logs: logs/
cd "$(dirname "$0")/.." ; M=~/rrr/gguf/Inkling-2-Mini-UD-Q4_K_XL.gguf; rm -f logs/all_done
./mikasa_long $M out --batch-test > logs/batch_test_final.log 2>&1
./mikasa_long $M out --lean --kv16 --prefill data/ctx4096.txt --multi-check > logs/multi_check_kv16.log 2>&1
tools/run_val.sh
tools/run_val2.sh
tools/run_val3.sh
for n in needle30k needle30k_mid needle60k needle60k_mid; do tools/run_needle.sh $n kv16 600; done
tools/run_needle.sh needle60k kv32 600
echo done > logs/all_done
