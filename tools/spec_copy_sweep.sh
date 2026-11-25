#!/bin/bash
# copy-proposer sweep on copy-heavy edit workloads.  usage: tools/spec_copy_sweep.sh [workloads...]
cd "$(dirname "$0")/.." || exit 1
M=${MODEL:-~/rrr/gguf/Inkling-2-Mini-UD-Q4_K_XL.gguf}; C="--lean --kv16 --nslots 1 --maxctx 8192 --pf-fast 512"
WL=${*:-"editcopy_file_direct editcopy_jsonl_direct editcopy_replace_direct"}
for w in $WL; do
  ./mikasa_spec $M out $C --prefill data/spec/$w.txt --gen-long 1000 logs/spec/cs_$w.old.txt > logs/spec/cs_$w.old.log 2>&1
  echo "$w: plain $(grep -o '[0-9.]* ms/token' logs/spec/cs_$w.old.log | head -1) ($(wc -l < logs/spec/cs_$w.old.txt) tokens)"
  for cfg in "--spec 15 --spec-m 1" "--spec 4 --spec-copy 4" "--spec 8 --spec-copy 4" "--spec 15 --spec-copy 4" "--spec 15 --spec-copy 2" "--spec 15 --spec-copy 4 --spec-fixed"; do
    tag=$(echo $cfg | tr -d ' -'); ./mikasa_spec $M out $C --prefill data/spec/$w.txt --gen-long 1000 logs/spec/cs_$w.$tag.txt $cfg > logs/spec/cs_$w.$tag.log 2>&1
    same=$(cmp -s logs/spec/cs_$w.$tag.txt logs/spec/cs_$w.old.txt && echo identical || echo DIFFERENT)
    echo "  $cfg: $(grep -E '^spec: K' logs/spec/cs_$w.$tag.log | grep -o '[0-9.]* tok/s.*') | $(grep -E 'copy prop' logs/spec/cs_$w.$tag.log | cut -c7-60) | $same"
  done
done
