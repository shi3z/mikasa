#!/bin/bash
# wide (tensor-core prefill path) verification sweep: K, switch point W (--spec-wide), copy proposer.  usage: tools/spec_wide_sweep.sh [workloads...]
cd "$(dirname "$0")/.." || exit 1
M=${MODEL:-~/rrr/gguf/Inkling-2-Mini-UD-Q4_K_XL.gguf}; C="--lean --kv16 --nslots 1 --maxctx 8192 --pf-fast 512"
WL=${*:-"editcopy_file_direct editcopy_jsonl_direct editcopy_replace_direct"}
for w in $WL; do
  [ -f logs/spec/cs_$w.old.txt ] || ./mikasa_spec $M out $C --prefill data/spec/$w.txt --gen-long 1000 logs/spec/cs_$w.old.txt > logs/spec/cs_$w.old.log 2>&1
  echo "$w: plain $(grep -o '[0-9.]* ms/token' logs/spec/cs_$w.old.log | head -1) ($(wc -l < logs/spec/cs_$w.old.txt) tokens)"
  for cfg in "--spec 15 --spec-copy 4" "--spec 15 --spec-copy 4 --spec-wide 8" "--spec 31 --spec-copy 4 --spec-wide 8" "--spec 63 --spec-copy 4 --spec-wide 8" "--spec 63 --spec-copy 4 --spec-wide 16" "--spec 63 --spec-copy 4 --spec-wide 4" "--spec 63 --spec-m 1 --spec-wide 8"; do
    tag=$(echo $cfg | tr -d ' -'); ./mikasa_spec $M out $C --prefill data/spec/$w.txt --gen-long 1000 logs/spec/ws_$w.$tag.txt $cfg > logs/spec/ws_$w.$tag.log 2>&1
    r=$(${PYTHON:-python3} tools/cmp_tokens.py logs/spec/cs_$w.old.txt logs/spec/ws_$w.$tag.txt x | sed 's/.*: //')
    echo "  $cfg: $(grep -E '^spec: K' logs/spec/ws_$w.$tag.log | grep -o '[0-9.]* tok/s.*') | $r"
  done
done
