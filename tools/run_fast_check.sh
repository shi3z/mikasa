#!/bin/bash
# Fast-path check: plain (--spec 0) and spec (--spec 15 --spec-copy 4 --spec-m 1) decode of several workloads with a given binary.
# usage: tools/run_fast_check.sh BIN TAG [workloads...]     -> logs/fast/TAG/*.txt (tokens), *.log, summary.txt (tok/s + md5)
# Compare tags with: tools/run_fast_check.sh --cmp TAGA TAGB
cd "$(dirname "$0")/.." || exit 1
if [ "$1" = "--cmp" ]; then
  bad=0
  for f in logs/fast/$2/*.txt; do b=$(basename $f); [ $b = summary.txt ] && continue
    if cmp -s $f logs/fast/$3/$b; then echo "same  $b"; else echo "DIFF  $b"; bad=1; fi; done
  [ $bad = 0 ] && echo "ALL IDENTICAL" || echo "MISMATCH"; exit $bad
fi
BIN=$1; TAG=$2; shift 2
M=${MODEL:-~/rrr/gguf/Inkling-2-Mini-UD-Q4_K_XL.gguf}
GEN=${GEN:-1000}
C="--lean --kv16 --nslots 1 --maxctx 8192 --pf-fast 512"
WL=${*:-"editcopy_file_direct edit_toolcall_py_direct gen_py_lru_direct gen_c_vec_direct shell_script_direct reason_math edit_diff_cpp_direct"}
D=logs/fast/$TAG; mkdir -p $D; : > $D/summary.txt
for w in $WL; do
  for c in plain spec; do
    x=""; [ $c = spec ] && x="--spec 15 --spec-m 1 --spec-copy 4"; [ $c = plain ] && x="--spec 0"
    $BIN $M out $C --prefill spec/data/$w.txt --gen-long $GEN $D/$w.$c.txt $x > $D/$w.$c.log 2>&1
    echo "$w $c $(grep -hE 'generated' $D/$w.$c.log | tr '\n' ' ') md5=$(md5sum < $D/$w.$c.txt | cut -c1-8)" | tee -a $D/summary.txt
  done
done
