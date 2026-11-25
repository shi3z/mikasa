#!/bin/bash
# Full build / validation / timing chain of the tensor-core prefill path.   usage: tools/run_prefill_all.sh [build] [unit] [val] [needle] [timing] [slow]   (default: all but slow)
# Results land in logs/ ; a summary is printed at the end of each stage.  Baseline = the old B=16 GEMV prefill in the same binary.
cd "$(dirname "$0")/.." || exit 1
M=~/rrr/gguf/Inkling-2-Mini-UD-Q4_K_XL.gguf
NVCC=/usr/local/cuda/bin/nvcc
mkdir -p logs
stages="$*"; [ -z "$stages" ] && stages="build unit val needle timing"
has() { [[ " $stages " == *" $1 "* ]]; }

if has build; then
  $NVCC -O3 -std=c++17 -arch=sm_86 -I src -o mikasa_pf src/inkling_model.cu src/gguf.cpp 2>&1 | grep -E "error" ; echo "built mikasa_pf"
  $NVCC -O3 -std=c++17 -arch=sm_86 -I src -o pf_gemm_test src/pf_gemm_test.cu 2>&1 | grep -E "error"
  $NVCC -O3 -std=c++17 -arch=sm_86 -I src -o pf_attn_test src/pf_attn_test.cu 2>&1 | grep -E "error"
fi

if has unit; then
  echo "== unit tests"; ./pf_gemm_test | tee logs/unit_gemm.log | tail -8
  ./pf_attn_test 2>&1 | tee logs/unit_attn.log | tail -30
fi

# teacher-forced continuation: baseline (old path) free-runs 32 greedy tokens after the prompt and dumps the logits of every step;
# the fast path prefills the same prompt and is fed the SAME 32 tokens; argmax / logits of every step are compared.
val_prompt() {  # prompt kv pb...
  local f=$1 kv=$2; shift 2
  local tag=$(basename $f .txt)_$kv
  ./mikasa_pf $M out --lean --$kv --nslots 1 --prefill $f --gen-long 32 logs/vb_$tag.txt --gen-logits logs/vb_$tag.bin 2>&1 | grep -E "^prefill:" | sed "s|^|[$tag baseline] |"
  for pb in "$@"; do
    ./mikasa_pf $M out --lean --$kv --nslots 1 --pf-fast $pb --prefill $f --gen-long 32 logs/vf_${tag}_$pb.txt --force logs/vb_$tag.txt --gen-logits logs/vf_${tag}_$pb.bin 2>&1 | grep -E "^prefill:" | sed "s|^|[$tag pb=$pb] |"
    python3 tools/cmp_gen.py logs/vf_${tag}_$pb.bin logs/vb_$tag.bin
  done
}
if has val; then
  echo "== teacher-forced 32-step continuation vs baseline"
  for f in data/ctx4096.txt data/ctx3333.txt data/ctx1500.txt data/ctx512.txt data/ctx130.txt; do val_prompt $f kv16 128 256 512 1024; done 2>&1 | tee logs/val_forced.log
  val_prompt data/ctx4096.txt kv32 512 2>&1 | tee -a logs/val_forced.log
  echo "== noise floor: baseline FP16-KV continuation fed with the FP32-KV baseline tokens vs FP32-KV baseline (4096 prompt)"
  ./mikasa_pf $M out --lean --kv32 --nslots 1 --prefill data/ctx4096.txt --gen-long 32 logs/nf_b32.txt --gen-logits logs/nf_b32.bin 2>&1 | grep -E "^prefill:"
  ./mikasa_pf $M out --lean --kv16 --nslots 1 --prefill data/ctx4096.txt --gen-long 32 logs/nf_b16.txt --force logs/nf_b32.txt --gen-logits logs/nf_b16.bin 2>&1 | grep -E "^prefill:"
  python3 tools/cmp_gen.py logs/nf_b16.bin logs/nf_b32.bin | tee logs/val_noisefloor.log
  echo "== argmax agreement over every position (4096 prompt), baseline vs fast, and FP16-vs-FP32 baseline as reference noise"
  ./mikasa_pf $M out --lean --kv16 --nslots 1 --prefill data/ctx4096.txt --argmax-out logs/am_b16.txt | grep -E "^prefill:"
  ./mikasa_pf $M out --lean --kv32 --nslots 1 --prefill data/ctx4096.txt --argmax-out logs/am_b32.txt | grep -E "^prefill:"
  ./mikasa_pf $M out --lean --kv16 --nslots 1 --pf-fast 512 --pf-all --prefill data/ctx4096.txt --argmax-out logs/am_f16.txt | grep -E "^prefill:"
  ./mikasa_pf $M out --lean --kv32 --nslots 1 --pf-fast 512 --pf-all --prefill data/ctx4096.txt --argmax-out logs/am_f32.txt | grep -E "^prefill:"
  for p in "am_b16 am_f16" "am_b32 am_f32" "am_b16 am_b32"; do python3 tools/cmp_argmax.py logs/${p% *}.txt logs/${p#* }.txt 1024; done 2>&1 | tee logs/val_argmax.log
  echo "== two prefill calls (second continues the carried state) with non-multiple chunk sizes, 3333-token prompt, split at 1000 (PB=256) / 777 (PB=512)"
  for sp in "256 1000" "512 777" "128 2049"; do set -- $sp
    ./mikasa_pf $M out --lean --kv16 --nslots 1 --pf-fast $1 --pf-split $2 --prefill data/ctx3333.txt --gen-long 32 logs/vs_$1_$2.txt --force logs/vb_ctx3333_kv16.txt --gen-logits logs/vs_$1_$2.bin 2>&1 | grep -E "^prefill:"
    python3 tools/cmp_gen.py logs/vs_$1_$2.bin logs/vb_ctx3333_kv16.bin
  done 2>&1 | tee logs/val_split.log
  echo "== other KV slots (slot 2 / 3 of 4) free-running generation identical to baseline?"
  for sl in 2 3; do
    ./mikasa_pf $M out --lean --kv16 --nslots 4 --slot $sl --prefill data/ctx1500.txt --gen-long 32 logs/sl_b_$sl.txt 2>&1 | grep -E "^prefill:"
    ./mikasa_pf $M out --lean --kv16 --nslots 4 --slot $sl --pf-fast 512 --prefill data/ctx1500.txt --gen-long 32 logs/sl_f_$sl.txt 2>&1 | grep -E "^prefill:"
    echo "slot $sl: generated tokens differing: $(paste logs/sl_b_$sl.txt logs/sl_f_$sl.txt | awk '$1!=$2' | wc -l) of 32"
  done 2>&1 | tee logs/val_slots.log
  if has slow; then
    echo "== 16K prompt teacher-forced (slow baseline)"; val_prompt data/ctx16384.txt kv16 512 2>&1 | tee logs/val_forced_16k.log
  fi
fi

if has needle; then
  echo "== needle in a haystack (fast prefill PB=512, FP16 full-layer KV), expected answer 7391"
  for n in needle30k needle30k_mid needle60k needle60k_mid; do
    ./mikasa_pf $M out --lean --kv16 --nslots 1 --maxctx 65536 --pf-fast 512 --prefill data/$n.txt --gen-long 600 logs/gen_pf_$n.txt > logs/needle_pf_$n.log 2>&1
    ans=$(${PYTHON:-python3} tools/tok.py decode "$(tr '\n' ',' < logs/gen_pf_$n.txt)" 2>/dev/null | tee logs/needle_pf_$n.text | grep -o "content_text|>[^<]*" | tail -1)
    echo "$n: $(grep -E '^prefill:' logs/needle_pf_$n.log) -> $ans"
  done 2>&1 | tee logs/val_needle.log
  n=needle60k
  ./mikasa_pf $M out --lean --kv32 --nslots 1 --maxctx 65536 --pf-fast 512 --prefill data/$n.txt --gen-long 600 logs/gen_pf_${n}_kv32.txt > logs/needle_pf_${n}_kv32.log 2>&1
  echo "$n kv32: $(grep -E '^prefill:' logs/needle_pf_${n}_kv32.log) -> $(${PYTHON:-python3} tools/tok.py decode "$(tr '\n' ',' < logs/gen_pf_${n}_kv32.txt)" 2>/dev/null | grep -o 'content_text|>[^<]*' | tail -1)" | tee -a logs/val_needle.log
fi

if has timing; then
  echo "== whole-prompt prefill throughput (GPU event time, real tokens), FP16 KV, 4 slots x 64K allocated"
  : > logs/timing_prompt.log
  for pb in 256 512 1024; do
    for f in ctx256 ctx512 ctx4096 ctx16384 ctx32768 ctx61440; do
      r=$(./mikasa_pf $M out --lean --kv16 --nslots 4 --pf-fast $pb --prefill data/$f.txt 2>&1 | grep -E "^prefill:")
      echo "PB=$pb $f: $r" | tee -a logs/timing_prompt.log
    done
  done
  echo "== chunk throughput at depth (real tokens in cache)"
  for pb in 256 512 1024; do ./mikasa_pf $M out --lean --kv16 --nslots 4 --pf-fast $pb --prefill data/ctx61440.txt --time-depths 0,256,4096,16384,32768,60000 2>&1 | grep -A9 "chunk throughput" | tee -a logs/timing_depth.log; done
  echo "== baseline (old path) whole-prompt throughput"
  for f in ctx256 ctx512 ctx4096; do echo "baseline $f: $(./mikasa_pf $M out --lean --kv16 --nslots 4 --prefill data/$f.txt 2>&1 | grep -E '^prefill:')" | tee -a logs/timing_baseline.log; done
  if has slow; then for f in ctx16384 ctx32768 ctx61440; do echo "baseline $f: $(./mikasa_pf $M out --lean --kv16 --nslots 4 --prefill data/$f.txt 2>&1 | grep -E '^prefill:')" | tee -a logs/timing_baseline.log; done; fi
fi
