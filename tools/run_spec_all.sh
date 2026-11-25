#!/bin/bash
# Speculative decoding benchmark chain.  usage: tools/run_spec_all.sh [build] [prompts] [bench] [exact] [oracle] [sampler] [trials] [sample] [msweep] [table]   (default: all; ~1.5 h)
# Needs ~/rrr/venv-conv/bin/python (tokenizers/transformers) for prompts and decoding.  Logs in logs/spec/, prompts in data/spec/.
cd "$(dirname "$0")/.." || exit 1
M=${MODEL:-~/rrr/gguf/Inkling-2-Mini-UD-Q4_K_XL.gguf}
NVCC=/usr/local/cuda/bin/nvcc
PY=${PYTHON:-~/rrr/venv-conv/bin/python}
GEN=${GEN:-1000}
C="--lean --kv16 --nslots 1 --maxctx 8192 --pf-fast 512"
WL=${WL:-"gen_py_logstats_direct gen_py_lru_direct gen_c_vec_direct edit_whole_py_direct edit_diff_cpp_direct edit_toolcall_py_direct shell_script_direct shell_find_direct shell_pipeline_direct
 gen_py_logstats gen_c_vec edit_whole_py edit_toolcall_py shell_script shell_find reason_puzzle reason_essay reason_math edit_whole_py_ctx24k_direct editcopy_file_direct editcopy_jsonl_direct editcopy_replace_direct"}
stages="$*"; [ -z "$stages" ] && stages="build prompts bench exact oracle sampler trials sample msweep copysweep widesweep table"
has() { [[ " $stages " == *" $1 "* ]]; }
mkdir -p logs/spec data/spec
if has build; then $NVCC -O3 -std=c++17 -arch=sm_86 -I src -o mikasa_spec src/inkling_model.cu src/gguf.cpp 2>&1 | grep -E "error"; echo built; fi
if has prompts; then $PY tools/mkspec_prompts.py 2>&1 | grep -E "tokens|Error"; fi
run() { # name cfgname extra-args...
  local w=$1 c=$2; shift 2
  local cc=$C; [[ $w == *ctx24k* ]] && cc="--lean --kv16 --nslots 1 --maxctx 32768 --pf-fast 512"
  ./mikasa_spec $M out $cc --prefill data/spec/$w.txt --gen-long $GEN logs/spec/$w.$c.txt "$@" > logs/spec/$w.$c.log 2>&1
}
if has bench; then
  for w in $WL; do
    echo "== $w"
    run $w old                              # old greedy loop (forward_batch B=1 graph, host round trip per token)
    run $w p0 --spec 0                      # new loop, no drafts
    run $w s8 --spec 8 --spec-m 1                                        # n-gram drafts (M=1, longest match), batch kernels, K<=8
    run $w s15 --spec 15 --spec-m 1 --spec-copy 4                        # + copy proposer (3-gram, longest match, sticky), K<=15 (16 rows, batch kernels)
    run $w w63 --spec 63 --spec-m 1 --spec-copy 4 --spec-wide 8          # up to 64 rows; verify passes with > 8 rows run on the tensor-core prefill path
    grep -hE "generated" logs/spec/$w.old.log | sed "s/^/  old: /"; grep -hE "^spec: K" logs/spec/$w.p0.log logs/spec/$w.s8.log | sed "s/^/  /"
  done
fi
if has exact; then
  echo "== exactness (token files vs old loop)"; : > logs/spec/exact.txt
  for w in $WL; do for c in p0 s8 s15 w63; do
    [ -f logs/spec/$w.$c.txt ] && $PY tools/cmp_tokens.py logs/spec/$w.old.txt logs/spec/$w.$c.txt $w.$c | tee -a logs/spec/exact.txt
  done; done
fi
if has oracle; then
  echo "== oracle drafts (reference output as draft, fixed width): any rejection = numerical disagreement of the verify kernels with the B=1 decode"; : > logs/spec/oracle.txt
  for w in $WL; do
    run $w oracle --spec 7 --spec-fixed --spec-oracle logs/spec/$w.old.txt                       # batch kernels, 8 rows
    echo "$w (8 rows, batch kernels): $(grep -h 'drafted' logs/spec/$w.oracle.log)" | tee -a logs/spec/oracle.txt
    run $w oraclew --spec 31 --spec-fixed --spec-wide 8 --spec-oracle logs/spec/$w.old.txt        # 32 rows, tensor-core path
    echo "$w (32 rows, tensor-core path): $(grep -h 'drafted' logs/spec/$w.oraclew.log)" | tee -a logs/spec/oracle.txt
  done
fi
if has sampler; then
  ./mikasa_spec $M out $C --prefill data/spec/gen_py_lru_direct.txt --temp 1.0 --gen-long 1 /dev/null --spec 4 --sampler-test 2>&1 | grep -E "sampler|draft" | tee logs/spec/sampler_test.log
  ./mikasa_spec $M out $C --prefill data/spec/reason_math.txt --temp 0.7 --gen-long 1 /dev/null --spec 4 --sampler-test 2>&1 | grep -E "sampler|draft" | tee -a logs/spec/sampler_test.log
  ./mikasa_spec $M out $C --prefill data/spec/reason_essay.txt --temp 2.0 --gen-long 1 /dev/null --spec 4 --sampler-test 2>&1 | grep -E "sampler|draft" | tee -a logs/spec/sampler_test.log
fi
if has trials; then
  echo "== end-to-end sampling check: 6000 independent 3-token continuations at T=2.5, plain sampling vs verified n-gram drafts (oracle drafts, K=3)"
  for cfg in "plain --spec 0" "spec --spec 3 --spec-fixed --spec-oracle logs/spec/reason_essay.old.txt"; do set -- $cfg; n=$1; shift
    ./mikasa_spec $M out $C --prefill data/spec/reason_essay.txt --gen-long 4 /dev/null --temp 2.5 --seed 5 --spec-trials 6000 "$@" > logs/spec/trials_$n.log 2>&1; done
  $PY tools/cmp_trials.py logs/spec/trials_plain.log logs/spec/trials_spec.log | tee logs/spec/trials_cmp.txt
fi
if has copysweep; then tools/spec_copy_sweep.sh | tee logs/spec/copy_sweep.txt; fi
if has widesweep; then tools/spec_wide_sweep.sh | tee logs/spec/wide_sweep.txt; fi
if has msweep; then tools/spec_msweep.sh | tee logs/spec/msweep.txt; fi
if has sample; then
  for w in gen_py_logstats_direct edit_whole_py_direct shell_script_direct reason_essay; do
    for T in 0.7; do
      for k in 0 8; do run $w t${T}_s$k --spec $k --temp $T --seed 7; grep -hE "^spec: K" logs/spec/$w.t${T}_s$k.log | sed "s/^/$w T=$T: /"; done
    done
  done
fi
if has table; then $PY tools/spec_table.py logs/spec "$WL" | tee logs/spec/table.md; fi
