#!/bin/bash
# usage: tools/nsys_prof.sh BIN TAG B   -> prof/TAG_B<B>.csv + table (per launch, 45 launches)
cd "$(dirname "$0")/.." || exit 1
M=${MODEL:-~/rrr/gguf/Inkling-2-Mini-UD-Q4_K_XL.gguf}
rm -f prof/$2_B$3.sqlite
TIMEB_DIVERSE=1 TIMEB_ONLY=$3 nsys profile -o prof/$2_B$3 -f true --cuda-graph-trace=node --trace=cuda $1 $M out --lean --kv16 --nslots 1 --maxctx 8192 --time-b 2>&1 | grep TIMEB | cut -c1-90
nsys stats --force-export=true -r cuda_gpu_kern_sum -f csv prof/$2_B$3.nsys-rep > prof/$2_B$3.csv 2>/dev/null
python3 tools/nsys_tab.py prof/$2_B$3.csv 45 ${4:-40}
