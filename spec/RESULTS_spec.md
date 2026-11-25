# Speculative decoding in the mikasa lean engine (greedy exact + sampling-compatible) -- results

Work copy `~/rrr/mikasa-spec` (from mikasa e12fd9f). RTX A6000, lean mode (`--lean --kv16 --nslots 1 --maxctx 8192 --pf-fast 512`), model Inkling-2-Mini-UD-Q4_K_XL.
All tok/s are wall-clock of the generation loop only (prompt prefill excluded), 1000-token cap (shorter when the model stops), one run per cell (run-to-run noise about 1%).
"plain" = the old `--gen-long` loop (forward_batch B=1 graph, one host round trip per token, 12.5-13 ms/token = 77-80 tok/s); the new loop with K=0 is within 1% of it.

## 1. What was built (src/spec_host.inc, small edits in batch_kernels.cuh / prefill_host.inc / long_host.inc / inkling_model.cu)
* `--spec K` (K=0: new loop without drafts, K>0 max drafted tokens per step, up to 15 with the batch kernels, up to 63 with `--spec-wide`). One graph per B (1..16), each graph =
  [H2D rollback index, kb_rollback of the previous step, H2D tokens/positions/slots, forward_batch(B), D2H argmax]; **one launch and one sync per step**. A rejected draft costs
  nothing extra: the rollback of step n runs at the start of step n+1 (`kb_rollback` skips `j<0`, i.e. "all accepted"). Verify = `forward_batch` over [last token + drafts],
  accepted = longest prefix whose argmax equals the draft, bonus token = argmax after the last accepted one, `batch_rollback(m+1)` for conv state (KV beyond is stale but harmless).
* Drafts: n-gram index over prompt + generated text, hash chain of the 128 latest occurrences of the last M tokens (`--spec-m`, default 1), candidate with the **longest backward match (<=64)**, ties latest
  (`--spec-first` = latest only). With `--spec-copy MIN` a copy proposer is tried first: 3-gram index, longest backward match of the 64 latest occurrences, plus a "sticky" continuation (keep copying from
  the source position of the previous pass while every emitted token equals the source token), used when the candidate has >= MIN (4) tokens; the n-gram drafter is the fallback.
* Adaptive width: at start the step cost of every B (launch+sync, real state) is measured; the acceptance probability of draft position i is learned online per match-length bucket (per-position counts shrunk
  towards the bucket's pooled rate); each step picks k maximising E[tokens]/cost(k+1) vs. 1/cost(1), so plain B=1 decoding is used whenever drafts are not worth it (`--spec-fixed` disables this: always full drafts).
  Measured cost (ms): B=1 12.3, 2 13.8, 3 16.6, 4 17.8, **5 23.1**, 8 28.4, **9 35.8**, 16 50.2 (jumps at B=5 and 9: kernel tiling); cost of a 24/32/48/64-row tensor-core pass 33/36/41/46 (see 4).
* `--temp T --seed S`: sampling (see 5). `--spec-oracle FILE`: drafts taken from a reference output (numerical-agreement test). `--spec-wide W`: verify passes with more than W rows use the tensor-core prefill path (see 4).

## 2. Exactness (greedy)
* Batch-kernel speculation (n-gram K<=8, +copy K<=15): **output identical token for token to the plain loop in all 22 workloads x 2 configs = 44/44** (`logs/spec/exact.txt`; the K=0 new loop is identical too).
  Earlier configurations (K=4, K=15, M=3) were also identical in 76/76 comparisons (56k tokens).
* Oracle test (reference output used as draft, fixed 8 rows, i.e. every verify pass uses the 8-row kernels): **100.0% of drafted tokens accepted in all 22 workloads (about 15k tokens)**: the B=8 and B=1 kernels
  never disagreed on an argmax in these runs. So no divergence was observed, which is evidence, not a proof (near-ties could still flip: the summation order differs between B).
* The tensor-core (wide, `--spec-wide`) verify path is **not** bit-compatible with B=1 decoding (FP16 activation rounding, other GEMM order): its output is "greedy decoding with a slightly different numeric", it
  diverged from the plain output after 90-744 tokens in 13 of 22 workloads (identical in the other 9, all 1000-token editcopy runs included). An oracle run with 16+ rows shows the first argmax flip after ~115 tokens (editcopy_file).
  Same kind of difference as FP16-KV vs FP32-KV (README: ~1.6% argmax flips on ordinary text), not a rollback bug (full-accept runs are fine, 9-row wide passes accepted 100%); but it means wide mode is **approximate greedy**.

## 3. Throughput per workload (plain old loop vs n-gram vs +copy vs wide)
Workloads (`tools/mkspec_prompts.py`, chat template via `server/chatfmt.render_prompt`): `*_direct` = assistant turn pre-started with `<|message_model|><|content_text|>` (or the tool name for tool calls), i.e. thinking switched off; without suffix the model thinks first (natural).
gen_* = write a script/library; edit_* = file in the prompt (2-2.2K tokens) + edit request (answer is a whole file / unified diff / write_file tool call); shell_* = bash tool call after a short agent history;
editcopy_* = answer reproduces most of the prompt (whole file with only 2 changes, 40-record JSONL edit, str_replace call with a big JSON-escaped old/new string); ctx24k = the edit_whole_py task after 24K tokens of unrelated text (decode at 25K context; the model first restates the file);
reason_* = thinking text (default effort). Shell calls are short (37-100 tokens), so their numbers are noisy.
"tokens/step" counts emitted tokens per verify pass; "accept. of drafted" = accepted/drafted tokens.

| workload | tokens | plain tok/s | config | tok/s | speedup | tokens/step | accept. of drafted | output identical to plain |
|---|---:|---:|---|---:|---:|---:|---:|---|
| gen_py_logstats_direct | 1000 | 79.7 | ngram K<=8 | 86.0 | 1.08x | 1.16 | 44% | IDENTICAL |
| gen_py_logstats_direct | 1000 | 79.7 | +copy K<=15 | 86.0 | 1.08x | 1.16 | 45% | IDENTICAL |
| gen_py_logstats_direct | 1000 | 79.7 | wide K<=63 | 86.0 | 1.08x | 1.17 | 43% | diverges @698 (of 1000) |
| gen_py_lru_direct | 1000 | 79.1 | ngram K<=8 | 94.4 | 1.19x | 1.35 | 49% | IDENTICAL |
| gen_py_lru_direct | 1000 | 79.1 | +copy K<=15 | 94.8 | 1.20x | 1.35 | 51% | IDENTICAL |
| gen_py_lru_direct | 1000 | 79.1 | wide K<=63 | 93.1 | 1.18x | 1.33 | 44% | diverges @358 (of 1000) |
| gen_c_vec_direct | 737 | 79.4 | ngram K<=8 | 83.0 | 1.04x | 1.09 | 35% | IDENTICAL |
| gen_c_vec_direct | 737 | 79.4 | +copy K<=15 | 82.6 | 1.04x | 1.09 | 33% | IDENTICAL |
| gen_c_vec_direct | 737 | 79.4 | wide K<=63 | 83.5 | 1.05x | 1.11 | 35% | diverges @522 (of 737) |
| edit_whole_py_direct | 1000 | 77.0 | ngram K<=8 | 103.7 | 1.35x | 1.62 | 56% | IDENTICAL |
| edit_whole_py_direct | 1000 | 77.0 | +copy K<=15 | 103.5 | 1.34x | 1.63 | 56% | IDENTICAL |
| edit_whole_py_direct | 1000 | 77.0 | wide K<=63 | 129.6 | 1.68x | 2.41 | 45% | diverges @295 (of 1000) |
| edit_diff_cpp_direct | 1000 | 77.2 | ngram K<=8 | 117.5 | 1.52x | 2.05 | 59% | IDENTICAL |
| edit_diff_cpp_direct | 1000 | 77.2 | +copy K<=15 | 117.6 | 1.52x | 2.13 | 58% | IDENTICAL |
| edit_diff_cpp_direct | 1000 | 77.2 | wide K<=63 | 124.6 | 1.61x | 2.30 | 48% | diverges @417 (of 1000) |
| edit_toolcall_py_direct | 1000 | 77.2 | ngram K<=8 | 110.8 | 1.44x | 1.93 | 50% | IDENTICAL |
| edit_toolcall_py_direct | 1000 | 77.2 | +copy K<=15 | 112.4 | 1.46x | 1.96 | 52% | IDENTICAL |
| edit_toolcall_py_direct | 1000 | 77.2 | wide K<=63 | 112.3 | 1.45x | 2.16 | 32% | diverges @574 (of 1000) |
| shell_script_direct | 85 | 77.3 | ngram K<=8 | 101.4 | 1.31x | 1.68 | 45% | IDENTICAL |
| shell_script_direct | 85 | 77.3 | +copy K<=15 | 105.4 | 1.36x | 1.79 | 48% | IDENTICAL |
| shell_script_direct | 85 | 77.3 | wide K<=63 | 101.3 | 1.31x | 1.83 | 31% | IDENTICAL |
| shell_find_direct | 37 | 80.4 | ngram K<=8 | 182.2 | 2.27x | 3.60 | 93% | IDENTICAL |
| shell_find_direct | 37 | 80.4 | +copy K<=15 | 153.5 | 1.91x | 3.60 | 64% | IDENTICAL |
| shell_find_direct | 37 | 80.4 | wide K<=63 | 189.4 | 2.35x | 4.50 | 48% | IDENTICAL |
| shell_pipeline_direct | 63 | 81.8 | ngram K<=8 | 95.0 | 1.16x | 1.38 | 37% | IDENTICAL |
| shell_pipeline_direct | 63 | 81.8 | +copy K<=15 | 94.8 | 1.16x | 1.38 | 37% | IDENTICAL |
| shell_pipeline_direct | 63 | 81.8 | wide K<=63 | 94.6 | 1.16x | 1.38 | 37% | IDENTICAL |
| gen_py_logstats | 1000 | 78.9 | ngram K<=8 | 89.4 | 1.13x | 1.23 | 50% | IDENTICAL |
| gen_py_logstats | 1000 | 78.9 | +copy K<=15 | 89.0 | 1.13x | 1.23 | 48% | IDENTICAL |
| gen_py_logstats | 1000 | 78.9 | wide K<=63 | 87.4 | 1.11x | 1.26 | 26% | diverges @744 (of 1000) |
| gen_c_vec | 1000 | 78.7 | ngram K<=8 | 87.0 | 1.10x | 1.26 | 30% | IDENTICAL |
| gen_c_vec | 1000 | 78.7 | +copy K<=15 | 87.6 | 1.11x | 1.28 | 31% | IDENTICAL |
| gen_c_vec | 1000 | 78.7 | wide K<=63 | 85.1 | 1.08x | 1.28 | 23% | diverges @90 (of 1000) |
| edit_whole_py | 1000 | 77.0 | ngram K<=8 | 109.0 | 1.42x | 1.73 | 61% | IDENTICAL |
| edit_whole_py | 1000 | 77.0 | +copy K<=15 | 108.9 | 1.42x | 1.73 | 60% | IDENTICAL |
| edit_whole_py | 1000 | 77.0 | wide K<=63 | 108.5 | 1.41x | 1.87 | 41% | diverges @404 (of 1000) |
| edit_toolcall_py | 1000 | 77.2 | ngram K<=8 | 118.9 | 1.54x | 1.99 | 64% | IDENTICAL |
| edit_toolcall_py | 1000 | 77.2 | +copy K<=15 | 116.0 | 1.50x | 1.87 | 64% | IDENTICAL |
| edit_toolcall_py | 1000 | 77.2 | wide K<=63 | 120.7 | 1.56x | 2.11 | 42% | diverges @172 (of 1000) |
| shell_script | 102 | 77.3 | ngram K<=8 | 92.9 | 1.20x | 1.36 | 48% | IDENTICAL |
| shell_script | 102 | 77.3 | +copy K<=15 | 93.2 | 1.21x | 1.38 | 47% | IDENTICAL |
| shell_script | 102 | 77.3 | wide K<=63 | 90.9 | 1.18x | 1.40 | 33% | IDENTICAL |
| shell_find | 85 | 79.4 | ngram K<=8 | 105.9 | 1.33x | 1.58 | 58% | IDENTICAL |
| shell_find | 85 | 79.4 | +copy K<=15 | 105.8 | 1.33x | 1.61 | 55% | IDENTICAL |
| shell_find | 85 | 79.4 | wide K<=63 | 110.6 | 1.39x | 1.71 | 51% | IDENTICAL |
| reason_puzzle | 1000 | 78.8 | ngram K<=8 | 88.3 | 1.12x | 1.21 | 40% | IDENTICAL |
| reason_puzzle | 1000 | 78.8 | +copy K<=15 | 88.6 | 1.12x | 1.22 | 42% | IDENTICAL |
| reason_puzzle | 1000 | 78.8 | wide K<=63 | 89.5 | 1.14x | 1.23 | 42% | diverges @715 (of 1000) |
| reason_essay | 1000 | 78.9 | ngram K<=8 | 86.1 | 1.09x | 1.25 | 27% | IDENTICAL |
| reason_essay | 1000 | 78.9 | +copy K<=15 | 86.0 | 1.09x | 1.25 | 27% | IDENTICAL |
| reason_essay | 1000 | 78.9 | wide K<=63 | 82.8 | 1.05x | 1.19 | 22% | diverges @461 (of 1000) |
| reason_math | 1000 | 78.7 | ngram K<=8 | 93.3 | 1.19x | 1.30 | 47% | IDENTICAL |
| reason_math | 1000 | 78.7 | +copy K<=15 | 92.8 | 1.18x | 1.29 | 47% | IDENTICAL |
| reason_math | 1000 | 78.7 | wide K<=63 | 92.6 | 1.18x | 1.29 | 47% | IDENTICAL |
| edit_whole_py_ctx24k_direct | 1000 | 70.7 | ngram K<=8 | 182.2 | 2.58x | 4.95 | 94% | IDENTICAL |
| edit_whole_py_ctx24k_direct | 1000 | 70.7 | +copy K<=15 | 193.6 | 2.74x | 6.40 | 90% | IDENTICAL |
| edit_whole_py_ctx24k_direct | 1000 | 70.7 | wide K<=63 | 384.9 | 5.44x | 9.99 | 88% | diverges @502 (of 1000) |
| editcopy_file_direct | 1000 | 76.9 | ngram K<=8 | 201.1 | 2.61x | 5.12 | 90% | IDENTICAL |
| editcopy_file_direct | 1000 | 76.9 | +copy K<=15 | 213.4 | 2.77x | 6.36 | 89% | IDENTICAL |
| editcopy_file_direct | 1000 | 76.9 | wide K<=63 | 397.1 | 5.16x | 11.10 | 78% | IDENTICAL |
| editcopy_jsonl_direct | 1000 | 77.2 | ngram K<=8 | 194.7 | 2.52x | 5.20 | 84% | IDENTICAL |
| editcopy_jsonl_direct | 1000 | 77.2 | +copy K<=15 | 209.7 | 2.72x | 6.32 | 86% | IDENTICAL |
| editcopy_jsonl_direct | 1000 | 77.2 | wide K<=63 | 347.3 | 4.50x | 14.91 | 53% | IDENTICAL |
| editcopy_replace_direct | 1000 | 77.3 | ngram K<=8 | 132.2 | 1.71x | 2.68 | 59% | IDENTICAL |
| editcopy_replace_direct | 1000 | 77.3 | +copy K<=15 | 131.9 | 1.71x | 2.67 | 59% | IDENTICAL |
| editcopy_replace_direct | 1000 | 77.3 | wide K<=63 | 124.4 | 1.61x | 3.05 | 32% | IDENTICAL |

Summary (n-gram K<=8 unless noted): fresh code generation 1.04-1.20x; reasoning text 1.09-1.19x (it repeats prompt phrases; the offline estimate of "no gain" was too pessimistic once M=1/longest match/cost-aware width are used, but the gain is small);
shell calls 1.2-2.3x; code/diff/tool-call editing 1.35-1.55x; copy-heavy output (whole file, JSONL, restated file) 2.5-2.8x with batch kernels, **4.5-5.4x with the wide path** (350-400 tok/s, 11-15 tokens per pass).
The copy proposer adds little over longest-match n-gram with M=1 (e.g. editcopy_file 201 -> 213, jsonl 195 -> 210 tok/s); it never hurt noticeably except shell_find (182 -> 154, 37-token sample).
The str_replace call (editcopy_replace) copies through JSON string escaping (`\n` instead of newlines, different tokenisation): 1.7x, wide path does not help (acceptance, not pass width, is the limit).
Sweeps: `logs/spec/msweep.txt` (n-gram order M=1,2,3,4,6; older acceptance estimator): M=1 and 2 best everywhere, M=6 loses ~half the gain, `--spec-fixed` (always full drafts) is slower than plain on 4 of 6 workloads (72-87 vs 79 tok/s on non-copy text).
`logs/spec/copy_sweep.txt`, `logs/spec/wide_sweep.txt` (K and switch point sweeps, current estimator): editcopy_file: K<=15 batch 208 tok/s, +wide(8) K15 282, K31 354, K63 400; editcopy_jsonl: 210 / 281 / 332 / 346-366.
Copy-min 2 vs 4 and K<=4/8/15 make <3% difference for the batch path once the width model is cost aware.

## 4. Wide verification (32/64 rows) via the tensor-core prefill path -- working but approximate
Per-pass latency of `forward_prefill` for one sequence at real state (depth 1000-3000, `--pf-fast PB --time-depths`, logits for the last token only): 16 rows 38 ms, 32 rows 46.6, 48 rows 52, 64 rows 57, 96 rows 63, 128 rows 70 ms
(with the per-row argmax head and graph-less launches the whole step measured 33 ms at 24 rows, 36 at 32, 46 at 64 in the spec loop; the batch kernels need 50 ms for 16 rows and 36 ms for 9 rows).
So a 64-row pass costs about 3.7x a plain step: the break-even is 3.7 accepted tokens per pass; copy-heavy output gives 11-15.
Implemented: per-application history buffers for the conv inputs (`init_pf_rollback`, +about 0.6 GB at PB=512) and a rollback table, so that the conv state can be rewound to "after j tokens" for any j after a tensor-core pass;
eager rollback of either path before the next step; greedy only (no sampling in wide mode); `--spec-wide W` uses it for B > W (W=4..8 best; switching at 8 rows is already cheaper than the 9-row batch graph).
Result: see table (w63 rows); editcopy_file 77 -> 397 tok/s, editcopy_jsonl 77 -> 347, ctx24k restate 71 -> 385. Caveat: approximate greedy as described in 2. Not validated: output quality impact of the flips beyond "they exist", position >30K, 64K contexts, nslots>1.

## 5. Sampling (temperature > 0)
Deterministic n-gram drafts are proposals with q = delta(d): accept d with probability p(d) (p = softmax(logits/T)); on rejection draw from p with d removed and renormalised; after a fully accepted draft draw the bonus token from p.
This reproduces ancestral sampling from softmax(logits/T) **exactly** (given exact logits). Implementation: `k_spec_sample` (one block per verify row, Gumbel-max with a counter-based RNG seed/step/row/vocab-index; no top-k/top-p/penalties).
Tests: (a) kernel-level chi-square: 100k draws per case on real logits, draft = top-1 / rank 3 / rare token / none: acceptance rate equals p(d) and the emitted token follows p (chi2 within dof, `logs/spec/sampler_test.log`).
This test found two real bugs, fixed: a 24-bit float u rounding to 1.0 (random-token outliers at ~0.5% per draw) and tail truncation of the Gumbel noise (visible at T=2); the final kernel keeps the tails in float via log1p of the exact distance to 1.
(b) end-to-end: 6000 independent 3-token continuations at T=2.5, plain sampling vs verified oracle drafts of width 4 (batch kernels): two-sample chi-square on token prefixes, `logs/spec/trials_cmp.txt` (final kernel):

```
first 1 generated token(s)         trials 6000 vs 6000, distinct 1229 / 1222, bins 3: chi2 = 0.1, dof = 2, p ~ 0.933
first 2 generated token(s)         trials 6000 vs 6000, distinct 2418 / 2382, bins 24: chi2 = 18.4, dof = 23, p ~ 0.735
first 3 generated token(s)         trials 6000 vs 6000, distinct 3095 / 3047, bins 13: chi2 = 9.5, dof = 12, p ~ 0.662
```
(p-values 0.66-0.93: no detectable difference.) Final-kernel sampler test at T=2.0: chi2 0.3 / 5.1 / 1.9 / 0.2 over 3 bins (dof 2) for draft = top-1 / rare token / other / none; at T=1.0 (older kernel before the tail fix) 4.1-5.9 over 6 bins.
Throughput at T=0.7 (older estimator, K<=8): logstats_direct 77.9 -> 85.5 tok/s, edit_whole_py_direct 76.2 -> 107.6, shell_script_direct 76.3 -> 93.1, reason_essay 78.1 -> 80.2; the sampler adds ~1.5% to the B=1 step.
Approximate parts: sampling uses the batch-kernel logits (same numeric caveat as greedy); the end-to-end test has limited power (distinct sequences are mostly unique); the sampler is not combined with the wide path.

## 6. Unfinished / honest list
* Wide verification is approximate (not token-identical to B=1 greedy in 13 of 22 workloads); an exact variant would need the batch kernels at B>16 (MB=16 is hard-wired) or a numerically matching prefill GEMM.
* The full chain `tools/run_spec_all.sh` was not re-run end to end after the last edits (stages were run separately; the oracle stage for the 32-row path only started: its result is "first flip at ~115 tokens", not an acceptance table). The copy/wide/M sweeps in logs/spec were produced by the sweep scripts with the same binary except for the msweep (older estimator).
* The oracle log `logs/spec/oracle.txt` is partial (3 workloads); the 100% claim above comes from the earlier complete run (`logs/spec/prev/`, 22/22 workloads at K=7 fixed, before the estimator change, which does not affect fixed-width oracle runs).
* One run per cell, 1000-token cap, short shell calls are noisy; no real agent logs were replayed (synthetic but realistic prompts built from this repo's files); no tree/multi-candidate drafts (forward_batch is a chain); n-gram search is host side (negligible vs 12 ms).
* Not tested: nslots>1 with speculation on slot>0 (swap_conv_state path is used but unverified), contexts > 25K for the wide path, `--kv32`.
* The wide path allocates ~0.6 GB extra and needs `--pf-fast >= 64`.

## 7. Reproduce
`tools/run_spec_all.sh [build prompts bench exact oracle sampler trials sample msweep copysweep widesweep table]` (about 1.5 h for everything; prompts need `~/rrr/venv-conv/bin/python`). Binary: `mikasa_spec`.
Key flags: `--lean --kv16 --nslots 1 --maxctx 8192 --pf-fast 512 --prefill data/spec/NAME.txt --gen-long 1000 OUT.txt --spec K [--spec-m 1] [--spec-copy 4] [--spec-wide 8] [--temp T --seed S]`.
