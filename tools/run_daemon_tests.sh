#!/bin/bash
# Daemon test chain.  usage: tools/run_daemon_tests.sh [build] [refs] [daemon] [http] [speed]   (default: all; ~15 min; needs a free GPU)
#   build  compile mikasa_daemon      refs   reference outputs of the old plain greedy loop (needs the GPU, run BEFORE the daemon)
#   daemon tests against the socket (exact cancel conc cache sample limit)      http   end-to-end through server/mikasa_server.py      speed  throughput via the server
# Logs in logs/daemon/.  Needs ~/rrr/venv-conv/bin/python (tokenizers/transformers).
cd "$(dirname "$0")/.." || exit 1
M=${MODEL:-$HOME/rrr/gguf/Inkling-2-Mini-UD-Q4_K_XL.gguf}
PY=${PYTHON:-$HOME/rrr/venv-conv/bin/python}
NVCC=/usr/local/cuda/bin/nvcc
L=logs/daemon; mkdir -p $L/ref
SOCK=/tmp/mikasa-test.sock; PORT=${PORT:-8791}
stages="$*"; [ -z "$stages" ] && stages="build refs daemon http speed"
has() { [[ " $stages " == *" $1 "* ]]; }
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
if [ "$used" -gt 2000 ] && { has refs || has daemon || has http || has speed; }; then echo "GPU is not free (${used} MiB used): refusing"; exit 1; fi
if has build; then $NVCC -O3 -std=c++17 -arch=sm_86 -I src -o mikasa_daemon src/inkling_model.cu src/gguf.cpp 2>&1 | grep -E "error"; echo built; fi
if has refs; then   # old plain loop (no --spec): the reference of the exactness tests
  for w in edit_whole_py_direct editcopy_jsonl_direct shell_script_direct gen_py_lru_direct reason_math; do
    ./mikasa_daemon $M /tmp/o_ref --lean --kv16 --nslots 1 --maxctx 8192 --pf-fast 512 --prefill spec/data/$w.txt --gen-long 500 $L/ref/$w.txt > $L/ref/$w.log 2>&1
  done; echo refs done
fi
start_daemon() { ./mikasa_daemon $M /tmp/o_d --serve --sock $SOCK --snaps ${SNAPS:-24} > $L/daemon.log 2>&1 & DPID=$!; until grep -q "ready:" $L/daemon.log; do sleep 2; kill -0 $DPID 2>/dev/null || { echo "daemon died"; tail $L/daemon.log; exit 1; }; done; }
if has daemon; then
  start_daemon
  $PY tools/daemon_tests.py $SOCK $L/ref exact cancel conc cache sample limit 2>&1 | grep -v "model of type" | tee $L/daemon_tests.log
  kill $DPID; wait $DPID 2>/dev/null
fi
if has http || has speed; then
  $PY server/mikasa_server.py --port $PORT --engine daemon --launch-daemon --model $M --daemon-sock $SOCK --daemon-args "--snaps ${SNAPS:-24}" > $L/server.log 2>&1 & SPID=$!
  until curl -s localhost:$PORT/health > /dev/null; do sleep 2; kill -0 $SPID 2>/dev/null || { echo "server died"; tail $L/server.log; exit 1; }; done
  has http && $PY tools/http_tests.py http://127.0.0.1:$PORT basic tools stream cancel reject 2>&1 | grep -v "model of type" | tee $L/http_tests.log
  has speed && $PY tools/http_tests.py http://127.0.0.1:$PORT speed 2>&1 | grep -v "model of type" | tee $L/speed.log
  kill $SPID; wait $SPID 2>/dev/null; sleep 2
  pgrep -f "mikasa_daemon.*--sock $SOCK" > /dev/null && echo "WARNING: daemon child still alive" || echo "daemon child stopped with the server"
fi
