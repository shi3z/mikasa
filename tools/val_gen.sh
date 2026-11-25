#!/bin/bash
# usage: val_gen.sh prompt.txt kv16|kv32 PB   -> teacher-forced 32-step continuation comparison (baseline vs fast prefill)
cd "$(dirname "$0")/.." || exit 1; M=~/rrr/gguf/Inkling-2-Mini-UD-Q4_K_XL.gguf
f=$1; kv=${2:-kv16}; pb=${3:-256}; tag=$(basename $f .txt)_${kv}_$pb
mkdir -p logs
./mikasa_pf $M out --lean --$kv --nslots 1 --prefill $f --gen-long 32 logs/vg_base_$tag.txt --gen-logits logs/vg_base_$tag.bin 2>&1 | grep -E "^prefill:" 
./mikasa_pf $M out --lean --$kv --nslots 1 --pf-fast $pb --prefill $f --gen-long 32 logs/vg_fast_$tag.txt --force logs/vg_base_$tag.txt --gen-logits logs/vg_fast_$tag.bin 2>&1 | grep -E "^prefill:|fast prefill"
python3 tools/cmp_gen.py logs/vg_fast_$tag.bin logs/vg_base_$tag.bin
