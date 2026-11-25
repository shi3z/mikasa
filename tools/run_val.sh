#!/bin/bash
# long-context validation runs (sequential). logs in logs/
cd "$(dirname "$0")/.." ; mkdir -p logs; M=~/rrr/gguf/Inkling-2-Mini-UD-Q4_K_XL.gguf
for N in 4096 16384 61440; do
  ./mikasa_long $M out --lean --kv16 --prefill data/ctx$N.txt --cmp-ref 8 --argmax-out logs/arg16_$N.txt --tail 8 logs/tail16_$N.bin > logs/val16_$N.log 2>&1
  ./mikasa_long $M out --lean --kv32 --nslots 1 --prefill data/ctx$N.txt --cmp-ref 8 --argmax-out logs/arg32_$N.txt --tail 8 logs/tail32_$N.bin > logs/val32_$N.log 2>&1
  python3 tools/cmp_argmax.py logs/arg16_$N.txt logs/arg32_$N.txt 1024 > logs/agree_$N.log
  ./mikasa_long $M out --cmp-dump logs/tail16_$N.bin logs/tail32_$N.bin 8 >> logs/agree_$N.log
done
echo done > logs/val_done
