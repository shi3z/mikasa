#!/bin/bash
cd "$(dirname "$0")/.." ; M=~/rrr/gguf/Inkling-2-Mini-UD-Q4_K_XL.gguf
# ring (1040 rows, wraps) vs linear SWA storage (ring 4096 = no wrap), 4K and 16K contexts, FP16 full KV
./mikasa_long $M out --lean --kv16 --nslots 1 --maxctx 4096 --ring 4096 --prefill data/ctx4096.txt --argmax-out logs/arg_lin_4096.txt --tail 16 logs/tail_lin_4096.bin > logs/lin4096.log 2>&1
./mikasa_long $M out --lean --kv16 --nslots 1 --maxctx 4096 --prefill data/ctx4096.txt --argmax-out logs/arg_ring_4096.txt --tail 16 logs/tail_ring_4096.bin > logs/ring4096.log 2>&1
python3 tools/cmp_argmax.py logs/arg_lin_4096.txt logs/arg_ring_4096.txt > logs/ringcmp_4096.log
./mikasa_long $M out --cmp-dump logs/tail_lin_4096.bin logs/tail_ring_4096.bin 16 >> logs/ringcmp_4096.log
# end-to-end naive reference attention (all layers, whole prefill) vs optimised path
for N in 4096 16384; do
./mikasa_long $M out --lean --kv16 --nslots 1 --maxctx $N --ref-all --prefill data/ctx$N.txt --argmax-out logs/arg_refall_$N.txt --tail 16 logs/tail_refall_$N.bin > logs/refall$N.log 2>&1
./mikasa_long $M out --lean --kv16 --nslots 1 --maxctx $N --prefill data/ctx$N.txt --argmax-out logs/arg_opt_$N.txt --tail 16 logs/tail_opt_$N.bin > logs/opt$N.log 2>&1
python3 tools/cmp_argmax.py logs/arg_refall_$N.txt logs/arg_opt_$N.txt > logs/refallcmp_$N.log
./mikasa_long $M out --cmp-dump logs/tail_refall_$N.bin logs/tail_opt_$N.bin 16 >> logs/refallcmp_$N.log
done
echo done > logs/val2_done
