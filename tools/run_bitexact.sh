#!/bin/bash
# Bit-exactness check of the lean forward pass: logits dumps (md5) for B=1 decode (48 steps, --gen-logits) and for same-sequence
# batches of B = 2,3,5,8,9,13,16 (--pf-b B, last 32 prompt positions via --tail).  usage: tools/run_bitexact.sh BIN TAG
cd "$(dirname "$0")/.." || exit 1
BIN=$1; TAG=$2; M=${MODEL:-~/rrr/gguf/Inkling-2-Mini-UD-Q4_K_XL.gguf}
P=${PROMPT:?set PROMPT to a token-id file}
D=logs/bitexact/$TAG; mkdir -p $D; : > $D/md5.txt
C="--lean --kv16 --nslots 1 --maxctx 8192"
$BIN $M out $C --pf-fast 512 --prefill $P --gen-long 48 $D/gen.txt --gen-logits $D/gen_logits.bin > $D/gen.log 2>&1
echo "B1_decode $(md5sum < $D/gen_logits.bin | cut -c1-12) tokens $(md5sum < $D/gen.txt | cut -c1-12)" | tee -a $D/md5.txt
for B in 2 3 5 8 9 13 16; do
  $BIN $M out $C --prefill $P --pf-b $B --tail 32 $D/tail$B.bin > $D/tail$B.log 2>&1
  echo "B${B}_prefill_tail $(md5sum < $D/tail$B.bin | cut -c1-12)" | tee -a $D/md5.txt
done
