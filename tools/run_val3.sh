#!/bin/bash
cd "$(dirname "$0")/.." ; M=~/rrr/gguf/Inkling-2-Mini-UD-Q4_K_XL.gguf
for N in 4096 16384; do
./mikasa_long $M out --lean --kv32 --nslots 1 --maxctx $N --ref-all --prefill data/ctx$N.txt --argmax-out logs/arg_refall32_$N.txt --tail 16 logs/tail_refall32_$N.bin > logs/refall32_$N.log 2>&1
./mikasa_long $M out --lean --kv32 --nslots 1 --maxctx $N --prefill data/ctx$N.txt --argmax-out logs/arg_opt32_$N.txt --tail 16 logs/tail_opt32_$N.bin > logs/opt32_$N.log 2>&1
python3 tools/cmp_argmax.py logs/arg_refall32_$N.txt logs/arg_opt32_$N.txt > logs/refallcmp32_$N.log
./mikasa_long $M out --cmp-dump logs/tail_refall32_$N.bin logs/tail_opt32_$N.bin 16 >> logs/refallcmp32_$N.log
done
echo done > logs/val3_done
