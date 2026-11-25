#!/bin/bash
# sweep of the n-gram order M (minimum match) and candidate choice, K<=8 adaptive.  usage: tools/spec_msweep.sh [workloads...]
cd "$(dirname "$0")/.." || exit 1
M=${MODEL:-~/rrr/gguf/Inkling-2-Mini-UD-Q4_K_XL.gguf}; C="--lean --kv16 --nslots 1 --maxctx 8192 --pf-fast 512"
WL=${*:-"gen_py_logstats_direct edit_whole_py_direct edit_diff_cpp_direct edit_toolcall_py_direct reason_math reason_essay"}
for w in $WL; do
  for cfg in "1 --spec-first" "2 --spec-first" "1" "2" "3" "4" "6" "3 --spec-fixed"; do
    set -- $cfg; m=$1; shift
    ./mikasa_spec $M out $C --prefill data/spec/$w.txt --gen-long 1000 logs/spec/ms_$w.$m$*.txt --spec 8 --spec-m $m "$@" 2>&1 | grep "^spec: K" | sed "s|^spec: K=8 M=\([0-9]*\) \(.*\)temp.*: generated \([0-9]*\) tokens in [0-9.]* s = \([0-9.]*\) tok/s.*tokens/step \([0-9.]*\)|$w M=$m $* \2 \4 tok/s, tokens/step \5|"
  done
done
