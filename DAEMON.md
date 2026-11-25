# mikasa daemon: resident engine for the OpenAI-compatible server

Work copy `~/rrr/mikasa-daemon` (clone of `~/rrr/mikasa` at 5b56928, nothing committed). RTX A6000 48 GB, model `Inkling-2-Mini-UD-Q4_K_XL.gguf`.
All numbers below come from `tools/run_daemon_tests.sh` (logs in `logs/daemon/`) and the console runs described in section 7; each cell is ONE run (no repetitions, so expect a few % noise,
more for sampled output at temperature 1.0, whose length and content differ from run to run).

## 1. What was built
| file | what |
|---|---|
| `src/daemon_host.inc` | the daemon (`mikasa MODEL OUTDIR --serve ...`): socket server, slot manager, prefix cache / snapshots, scheduler, speculative + multi-sequence decode |
| `src/inkling_model.cu`, `src/spec_host.inc` | `--serve` flags; `k_spec_sample` got a `prm_stride` argument (0 = old behaviour, 4 = per-row temperature/seed for 4 different sequences). Everything else of the engine is unchanged |
| `server/daemon_engine.py` | `DaemonEngine` (`make_engine('daemon')`): `generate(ids, max_new, sampling, cancel)` -> iterator of token ids with `.final` |
| `server/mikasa_server.py`, `chatfmt.py`, `engines.py` | `--launch-daemon / --daemon-sock / --daemon-bin / --daemon-args`, concurrent requests (no global lock for the daemon), 400 for unsupported sampling parameters, `/props` reports `n_ctx` and `total_slots`, usage carries `prompt_tokens_details.cached_tokens` and `timings`; tokenizer lock (see 6) |
| `~/rrr/harness-mikasa-test/webui.py` | COPY of the console with the preset `mikasa` (section 7). The running console on 8770 and its `harness/` were not touched |
| `tools/daemon_tests.py`, `http_tests.py`, `run_daemon_tests.sh`, `console_smoke.py`, `console_parallel.py` | tests |

Build: `nvcc -O3 -std=c++17 -arch=sm_86 -I src -o mikasa_daemon src/inkling_model.cu src/gguf.cpp` (about 10 s).
Run: `./mikasa_daemon MODEL /tmp/unused --serve --sock /tmp/mikasa.sock` (defaults: `--lean --kv16 --nslots 4 --maxctx 65536 --pf-fast 512 --spec 15 --spec-copy 4 --spec-m 1`, greedy-exact n-gram/copy speculation; `--spec-wide` is not used by the daemon).
Server: `python server/mikasa_server.py --port 8767 --engine daemon --launch-daemon --model MODEL --daemon-args "--snaps 24"` (the daemon is a child of the server: killed with it, `PR_SET_PDEATHSIG`).
The server only binds its port after the daemon is ready, so `/health` = ready (model load + 8 s graph capture; 15 s in total with a warm page cache, cold-disk load time was not measured).

Daemon flags: `--sock PATH` (unix socket, mode 0600), `--tcp PORT` (additionally 127.0.0.1), `--snaps N` (max snapshots, default 40; the console preset uses 24), `--snap-end` (also snapshot the exact end of every prompt, off), `--pf-chunk N` (prefill chunk, default 512), `--quiet`, plus all the existing engine flags.

## 2. Protocol (JSON lines, one object per line)
Client to daemon:
* `{"op":"gen","id":"r1","prompt_ids":[...],"max_new":N,"temperature":1.0,"seed":123,"stop_ids":[200006,199999],"top_k":0,"top_p":1.0,"spec_k":-1,"no_cache":false}`
  `temperature` 0 (default) = greedy; `seed` optional; `top_k > 0` or `top_p < 1` are answered with `finish:"error"` (never ignored); `spec_k` (0..15) caps the drafts of this request (0 = plain decode), `no_cache` forces a from-scratch prefill (tests).
* `{"op":"cancel","id":"r1"}`; closing the connection cancels all its requests; `{"op":"info"}`, `{"op":"stats"}` (n_ctx, nslots, counters, GPU memory).
Daemon to client:
* `{"op":"tokens","id":"r1","tokens":[...]}` (streamed after every step; the stop token is included as the last token)
* `{"op":"done","id":"r1","finish":"stop|length|cancelled|error","n_prompt":N,"n_cached":C,"n_gen":G,"timings":{queue_ms,prefill_ms,ttft_ms,decode_ms,decode_tps,cache:"none|snapshot|resident",steps,multi_steps,drafted,accepted},"error":"..."}`
`max_new` counts generated tokens including the stop token. Context limit: a prompt with `n_prompt + 2 > n_ctx` is rejected; generation stops with `finish:"length"` when the next position would not fit (65000-token prompt: 537 tokens, = 65536 - 65000 - 1). The server returns 400 `exceed_context_size_error` before even asking the daemon.

## 3. Design
* One GPU thread owns everything; one reader thread per connection only parses. 4 slots (the 4 KV caches of lean mode), more requests wait in a FIFO queue. Cancelled / disconnected requests are finished at the next step boundary (a decode step is 13-25 ms, a prefill chunk 100-250 ms).
* **Scheduler loop**: admit waiting requests; then ONE prefill chunk of one prefilling request (chunk = 512 tokens, 256 if somebody is generating; chunk ends are multiples of 256; this keeps other streams alive during a long prefill, they get one decode step per chunk); then ONE decode step of all generating requests:
  * exactly 1 generating request: speculative step (`forward_batch` over [last token + drafts], same adaptive n-gram / copy proposer and acceptance model as `--spec`, cost table measured once at start-up, rollback of the conv state of rejected drafts is lazy inside the graph, as in `run_spec`). Greedy: exact (see 5). Temperature > 0: the existing rejection-sampling verifier.
  * 2 or more: `forward_batch` multi mode, one plain token per sequence (no speculation), greedy or sampled per row (per-row temperature / seed / step).
* **Conv state residency**: slot 0's conv state lives in the engine arrays, slots 1-3 in `X.Sx*`; same-sequence kernels (prefill, speculative step) need the sequence's state in the engine arrays, so the daemon swaps it in with one kernel (`k_swap_tab`, no staging) when the active slot changes, flushing a pending lazy rollback first. Multi mode needs the natural layout.
* **Prefix cache** (per slot token history `hist` = tokens whose KV/conv/ring state is valid, always consistent: also after cancel and after speculative steps; speculation is capped so that no token beyond `max_new` is fed):
  1. `resident`: new prompt starts with the slot's whole history (at least one new token) -> prefill only the rest.
  2. `snapshot`: at the end of the chunk that ends at the last multiple of 512 of a prompt (so only prompts >= 512 tokens) the daemon copies the conv states (3.8 MB) and the last 1040 positions of all 40 SWA rings (340.8 MB; the ring is overwritten by later tokens, which is why a snapshot is needed) into a pool buffer. A request whose prompt shares >= 32 tokens with a slot's history restores the largest snapshot at p <= the common prefix of that slot (full-attention KV is linear and rows < p are still there, rows >= p are overwritten as usual) and prefills the rest. Snapshots are tied to their slot: when the slot's history is cut below a snapshot's position the snapshot is dropped. LRU eviction over all snapshots when the pool (`--snaps`) is full.
  3. else from scratch on the least recently used idle slot (conv states zeroed).
  The best slot among the IDLE ones is chosen; a request never waits for a busy slot that would have had a better prefix.
* Chunk alignment: the long/prefill results say splits at multiples of 128/256 give bit-identical results and unaligned splits only noise-level differences. Snapshots are taken at multiples of 512 and chunks end at multiples of 256, so a snapshot restore resumes at an aligned position (all 7 greedy runs that were restored from a snapshot gave exactly the from-scratch / reference tokens: 4 in the cache tests below, 3 in the exactness tests whose plain re-run hit a snapshot). The `resident` path resumes at an arbitrary position (unaligned): it is covered by one comparison that came out identical, but noise-level differences are possible by the earlier finding.
* Memory (A6000, GPU used incl. other 0.4 GB): weights 18.6 GB, +KV/buffers 29.3 GB, +prefill buffers 29.8 GB, after graph capture 30.3 GB (`logs/daemon/daemon.log`). Each snapshot = 344.6 MB; `--snaps 24` = 8.3 GB -> at most 38.6 GB; the default 40 would reach 44.1 GB (more than the 44 GB budget, so the console preset uses 24). Snapshot buffers are allocated lazily.
* The server / engine adapter: `DaemonEngine` keeps one connection, a reader thread demultiplexes by request id; closing the iterator (client left, parser saw the stop token) cancels; `/props` and `/stats` query the daemon; `/health` is 503 if the connection died (a CUDA error makes the daemon exit; nothing restarts it).

## 4. Results: correctness (all runs in `logs/daemon/daemon_tests.log`, `http_tests.log`)
Reference = the old plain greedy loop (`--gen-long 500` without `--spec`, `--lean --kv16 --nslots 1 --maxctx 8192 --pf-fast 512`), 5 prompts from `spec/data` (2-3K token edit/shell/code/reasoning prompts), 500 tokens (85 for the shell script).
* Daemon greedy with speculation (default): IDENTICAL to the reference token for token in 5/5. With `spec_k=0` (plain): 5/5 identical. Speedup of speculation as in RESULTS_spec (edit_whole 105, jsonl copy 204, shell 105, code gen 86, reasoning 90 tok/s vs 77-79 plain).
* 4 concurrent greedy requests (different prompts, staggered prefill, multi mode with the 4 slots; one request briefly in speculative mode): all 4 identical to their single-sequence reference (4/4; the earlier README caveat of "noise-level differences in multi mode" did not show up in these 4 runs / 1,585 tokens; this is evidence, not proof).
* Cancel: during decode the final message arrives 12.6 ms after the cancel (slot free, 0 active slots after 0.3 s); during the prefill of a 26,006-token prompt 115 ms after (one chunk finishes first); the next request works. Through the server: client disconnect after 30 streamed chunks -> 0 active slots after 0.05 s.
* Sampling: same seed twice gives the same 300 tokens; a different seed differs from token 1; 4 concurrent sampled requests run in multi mode (finish length, 200 tokens each). Sampling correctness itself is the sampler of RESULTS_spec (chi-square tests there); the daemon adds per-row parameters and the first token after prefill, which were NOT separately chi-square tested.
* `top_p < 1`, `top_k > 0`, `stop` strings, penalties, `min_p`, `n > 1`, `logit_bias`: HTTP 400 with a message (checked), not silently ignored.
* Context: prompts of 65535 and 70536 tokens rejected; 65000-token prompt generates 537 tokens and stops with `length`.

## 5. Results: prefix cache (3-turn conversation built with `chatfmt.render_prompt`: system + a 230-line CUDA source in the first user message, then the engine's own answers (content only) + follow-ups; greedy, 120 tokens)
| turn | prompt tokens | n_cached (path) | prefill with cache | prefill without (`no_cache`) | generated tokens |
|---|---:|---|---:|---:|---|
| 1 | 6861 | 0 | 2184 ms | 2190 ms | identical |
| 2 | 6880 | 6656 (snapshot) | 99 ms | 2193 ms | identical |
| 3 | 6900 | 6656 (snapshot) | 102 ms | 2198 ms | identical |
Time to first token = prefill time + 0.3-1 ms. Branching (same first turn, two different follow-ups, restored from the snapshot taken in turn 1): n_cached 6656, prefill 98 ms vs 2197-2198 ms, generated tokens identical (2/2). Resident path (prompt = previous prompt + its 60-token output): n_cached 2085 of 2086, prefill 25 ms vs 682 ms, tokens identical (1/1).
Only the 3 turns use a snapshot at 6656 (= 13 x 512): the rest of the prompt (up to 511 tokens) is always re-prefilled. In turn 2 and 3 the cached position stays 6656 because the new snapshot (taken again at 6656) is at the same position. The template does not reproduce the generated tokens of the previous turn exactly, so the `resident` path did not occur in the conversation; the snapshot path does the work.
In the console runs the second agent step reused the first 1024 tokens (system prompt + tools) of its own previous request (cached 1024, prefill 61 ms vs 300 ms).

## 6. Results: throughput (through the server, temperature 1.0 as the harness uses, one run per cell, `logs/daemon/speed.log`)
| case | tok/s |
|---|---:|
| chat, new code (39-token prompt, 600 tokens), single stream | 78 / 82 (two runs) |
| edit prompt (2143-token file + "rename function", 1500 tokens, copy-heavy), single stream | 195 / 196 (the first run of an earlier build gave 215 and 96: at temperature 1.0 the amount of copying varies from run to run) |
| 4 concurrent streams (2 chat + 2 edit, incl. prefill), aggregate | 163 (41-44 per stream) |
| 4 concurrent identical chat prompts, aggregate | 182 (46 per stream) |
Plain single-stream decoding is 77-80 tok/s, 4-stream multi mode in the earlier README: 198 aggregate at short context (synthetic KV); the daemon gets 174-182 with the host loop (sync + argmax + JSON per step).
Greedy tests: speculation 86-105 tok/s on shell / code / reasoning, 204 on a copy-heavy JSONL edit; 4 greedy concurrent short-prompt requests: 174 aggregate.
Prefill: 2,540 tok/s for a 65,000-token prompt, 3,150 tok/s at ~7K (2184 ms for 6861 tokens) with chunk 512.
Note: the single speed numbers at temperature 1.0 are one sample each of a random process.

## 7. Console (`~/rrr/harness-mikasa-test`, port 8771; the console on 8770 was never touched, no jobs were running when I checked)
Changes in the copy of `webui.py`: `PRESETS["mikasa"]` (start command = `venv-conv/bin/python mikasa-daemon/server/mikasa_server.py --port 8767 ... --launch-daemon ... --daemon-args "--snaps 24"`, port 8767 like the llama-server preset, 4 slots, ctx 65536/slot), `--engine mikasa|inkling` selects which preset the UI's (single) engine box and all jobs use (default stays llama-server; the original preset is unchanged), `healthy()` also checks `/props` `build_info` because both presets use port 8767 (so the console can tell which engine answers), `engine_ctx()` returns (65536, not shared) for mikasa, the engine log shown is `data/engine_<preset>.log`, the engine box shows the preset label. Readiness = `/health` 200 as before.
Test (`tools/console_smoke.py`, `tools/console_parallel.py`, copy started with `python3 webui.py --engine mikasa --host 127.0.0.1 --port 8771`):
* Start via the API: engine up after 15 s; agent task "create hello.py that prints hello from mikasa and run it": 2 steps, 6 s, tool call `bash` parsed from the stream, hello.py created, output `hello from mikasa`, final answer correct; second request cached 1024 tokens. `Stop` killed the server and the daemon (GPU back to 0.4 GB).
* 4 agent tasks in parallel (fib, primes, line counter, squares): all 4 finished in 14 s (`Done`; spot-checked outputs 832040, primes, squares are right). The FIRST attempt failed: 3 of 4 got HTTP 500 because the HF tokenizer / chat template are not thread-safe ("Already borrowed" class of error, the traceback was not captured then; the fix, a lock around tokenizer calls in `chatfmt.py`, removed the failures in the second attempt). The cause is inferred from the fix working, not from the traceback.
Not done: switching engines from the UI at run time (needs `--engine`), restarting a dead daemon, reasoning/long agent runs (only the tiny tasks above were run), the full MBPP eval mode.

## 8. Limits / unfinished (honest list)
* Speculation only while exactly one request is generating. With 2+ requests it is plain multi mode (decode of 4 streams = 41-46 tok/s each); speculation with several sequences is untested.
* While a request is being prefilled, the others advance one decode step per prefill chunk (256 tokens, ~85-100 ms): their rate drops to roughly 10 tok/s during a long prefill. Not measured separately.
* The daemon cannot share a prefix between slots (no cross-slot copy of the full-attention KV); a parallel task with the same system prompt prefills it again unless it lands on the slot that had it. Snapshots exist only for prompts >= 512 tokens and only at multiples of 512.
* A request waits (does not preempt) when all 4 slots are busy; no priorities, no limit on the queue length.
* Sampling: pure temperature only (top_k/top_p/min_p/penalties/stop strings rejected). The first token after prefill and the per-row multi-mode sampling were not chi-square tested.
* Only one exactness comparison family (5 prompts, 500 tokens, 4 concurrent) and 6 cache comparisons (3 turns, 2 branches, 1 resident); `--kv32`, `--spec-wide`, contexts above 26K with several slots active, long soak runs, cancel storms and memory growth over hours were not tested. The resident path was exercised twice.
* A CUDA error aborts the daemon process (the `CHECK` macro); the server reports 503 / connection lost, the console does not restart it.
* `/props` `total_slots` is 4 but the console's own limit of parallel tasks (`MAX_PARALLEL = 4`) is a separate constant that was only commented, not tied to the engine.
* Model load time with a cold page cache was not measured (15 s warm).
* The `mikasa_daemon` binary and `logs/` are untracked files in the work copy; nothing was committed.

## 9. Reproduce
`tools/run_daemon_tests.sh [build] [refs] [daemon] [http] [speed]` (needs a free GPU; about 15 min). Console: `cd ~/rrr/harness-mikasa-test && python3 webui.py --engine mikasa --host 127.0.0.1 --port 8771`, then `python3 ~/rrr/mikasa-daemon/tools/console_smoke.py http://127.0.0.1:8771 $(cat data/web_token)`.
