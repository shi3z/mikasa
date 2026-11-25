#!/bin/bash
# needle-in-a-haystack: usage run_needle.sh name [kv16|kv32] [gen]
cd "$(dirname "$0")/.." ; M=~/rrr/gguf/Inkling-2-Mini-UD-Q4_K_XL.gguf
n=$1; kv=${2:-kv16}; gen=${3:-600}
./mikasa_long $M out --lean --$kv --nslots 1 --maxctx 65536 --prefill data/$n.txt --gen-long $gen logs/gen_${n}_$kv.txt > logs/needle_${n}_$kv.log 2>&1
${PYTHON:-python3} tools/tok.py decode "$(tr '\n' ',' < logs/gen_${n}_$kv.txt)" > logs/needle_${n}_$kv.text 2>/dev/null
