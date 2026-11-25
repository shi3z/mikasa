#!/usr/bin/env python3
"""Build the speculative-decoding workload prompts (token id files, one id per line) in data/spec/.   Run with ~/rrr/venv-conv/bin/python.
Workloads: gen_* (code generation), edit_* (code editing: whole file / diff / tool call), shell_* (shell tool calls after a short agent history), reason_* (reasoning text)."""
import sys, os, json
here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(here, "..", "server"))
import chatfmt as C
OUT = os.path.join(here, "..", "data", "spec"); os.makedirs(OUT, exist_ok=True)
R = lambda p: open(os.path.join(here, "..", p)).read()
def wr(name, msgs, tools=None, eff=None, direct=None):
    """natural prompt (the model decides whether to think first) and, if `direct` is given, a variant whose assistant turn is pre-started with that text
    (skips the thinking phase: plain answer / forced tool call), as when thinking is switched off in an application."""
    ids = C.render_prompt(msgs, tools=tools, reasoning_effort=eff)
    def w(n, ids): open(f"{OUT}/{n}.txt", "w").write("\n".join(map(str, ids)) + "\n"); print(n, len(ids), "tokens")
    w(name, ids)
    if direct: w(name + "_direct", ids + C.tokenizer().encode(direct, add_special_tokens=False).ids)
U = lambda t: {"role": "user", "content": t}
EFF = 0.0    # no / minimal thinking for the code and tool-call workloads (they are measured on the produced code); reasoning_* use the default effort

# ---- code generation
wr("gen_py_logstats", [U("Write a complete Python 3 script `logstats.py` that reads one or more web server access log files (common log format) given on the command line, and prints: total requests, requests per status code class (2xx/3xx/4xx/5xx), the 10 most requested paths, the 10 busiest minutes, and the average response size. Use argparse, dataclasses, type hints and docstrings; include a small unit test section using unittest. Output only the code.")], eff=EFF, direct="<|message_model|><|content_text|>")
wr("gen_py_lru", [U("Write a Python module implementing a thread-safe LRU cache with TTL support (class `TTLCache` with get, put, delete, clear, __len__, stats), a decorator `cached(ttl, maxsize)` for functions, and a pytest test file in the same answer (two code blocks). Add docstrings and type hints.")], eff=EFF, direct="<|message_model|><|content_text|>")
wr("gen_c_vec", [U("Write a C++17 header-only library `smallvec.hpp` providing a `SmallVector<T, N>` with inline storage for N elements, push_back, emplace_back, pop_back, insert, erase, iterators, copy/move constructors and assignment, and a `main.cpp` that tests it with a few asserts. Output the two files.")], eff=EFF, direct="<|message_model|><|content_text|>")

# ---- code editing
chat = R("server/chatfmt.py"); gg = R("src/gguf.cpp"); mk = R("tools/mkctx.py")
wr("edit_whole_py", [U("Here is a Python file:\n\n```python\n" + chat + "\n```\n\nRename the function `render_prompt` to `build_prompt` everywhere (keeping a backwards compatible alias `render_prompt = build_prompt` at the end), add an optional `system` parameter to it that, when given, prepends a system message, and add a short docstring to every function that lacks one. Output the complete edited file.")], eff=EFF, direct="<|message_model|><|content_text|>")
wr("edit_diff_cpp", [U("Here is a C++ file:\n\n```cpp\n" + gg + "\n```\n\nAdd bounds checking: every read from the memory-mapped file must verify that offset + size <= file size and throw std::runtime_error with a descriptive message otherwise. Output the change as a unified diff (diff -u style) against this file.")], eff=EFF, direct="<|message_model|><|content_text|>")
write_tool = [{"type": "function", "function": {"name": "write_file", "description": "Write (overwrite) a file with the given full content.",
               "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}}}]
wr("edit_toolcall_py", [U("You are editing files in a repository through tools. File tools/mkctx.py currently contains:\n\n```python\n" + mk + "\n```\n\nPlease create tools/mkctx2.py, a copy of this file in which the output files are also written in a JSON format ({\"ids\": [...]}) next to the .txt files, and with a --seed option instead of the fixed random seeds. Use the write_file tool.")], tools=write_tool, eff=EFF, direct='<|message_model|>write_file<|content_invoke_tool_json|>')

# ---- shell tool calls after a short agent history
bash_tool = [{"type": "function", "function": {"name": "bash", "description": "Run a shell command and return its output.",
              "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}}]
def tc(i, cmd): return {"role": "assistant", "content": "", "tool_calls": [{"id": f"call_{i}", "type": "function", "function": {"name": "bash", "arguments": json.dumps({"command": cmd})}}]}
def tr(i, out): return {"role": "tool", "tool_call_id": f"call_{i}", "content": out}
prefill_sh = R("tools/run_prefill_all.sh")
wr("shell_script", [U("Task: in /work/mikasa there is a validation script. Create a new script tools/run_spec_check.sh that does the same kind of thing (same helper functions, same structure, same variables) but for the speculative decoding experiments: it should build mikasa_spec, run three prompts with --spec 0, --spec 4 and --spec 8 and compare the outputs with cmp. Write it with a heredoc (cat > file <<'EOF') and make it executable, in one bash command."),
    tc(1, "ls /work/mikasa/tools && cat /work/mikasa/tools/run_prefill_all.sh"), tr(1, "run_all.sh run_needle.sh run_prefill_all.sh run_val.sh tok.py mkctx.py\n--- run_prefill_all.sh ---\n" + prefill_sh)], tools=bash_tool, eff=EFF, direct='<|message_model|>bash<|content_invoke_tool_json|>')
ls_out = "\n".join(f"-rw-r--r-- 1 user user {s} Nov 25 12:{i:02d} {n}" for i, (n, s) in enumerate([("batch_host.inc", 31022), ("batch_kernels.cuh", 38211), ("batch_kernels2.cuh", 22019), ("gguf.cpp", 4801), ("gguf.hpp", 1500), ("inkling_model.cu", 41022), ("kernels_common.cuh", 77319), ("long_host.inc", 24011), ("long_kernels.cuh", 20004), ("pf_attn.cuh", 14411), ("pf_gemm.cuh", 17002), ("pf_ops.cuh", 12012), ("prefill_host.inc", 19001)]))
wr("shell_find", [U("How many lines of CUDA/C++ code are there per file in /work/mikasa/src, which files mention forward_batch, and which function names defined with __global__ appear more than once across the files? Use the shell; give me the final tables in the end."),
    tc(1, "ls -l /work/mikasa/src"), tr(1, ls_out),
    tc(2, "cd /work/mikasa/src && wc -l *.cuh *.cu *.inc *.cpp | sort -n | tail -15"), tr(2, "   1500 gguf.hpp\n  4801 gguf.cpp\n 41022 inkling_model.cu\n  total 311k")], tools=bash_tool, eff=EFF, direct='<|message_model|>bash<|content_invoke_tool_json|>')
wr("shell_pipeline", [U("Write one bash command (pipeline with awk/sort/uniq/xargs, or a small loop) that, for each *.log file under /work/runs, extracts the lines with 'tok/s', computes mean and standard deviation of the tok/s value per file, and prints a table sorted by mean. Test it on the real data first if useful; use the shell tool.")], tools=bash_tool, eff=EFF, direct='<|message_model|>bash<|content_invoke_tool_json|>')

# ---- reasoning text (default effort = thinking)
wr("reason_puzzle", [U("Three friends, Alice, Bob and Carol, each own one pet (a cat, a dog, a parrot) and drive one car colour (red, blue, green). Alice does not own the dog. The person with the parrot drives the green car. Bob drives the red car. Carol does not own the cat. Who owns which pet and drives which car? Think carefully and explain your reasoning step by step.")])
wr("reason_essay", [U("Explain why mixture-of-experts language models are cheaper to run per token than dense models of the same total parameter count, but why batching many tokens at once reduces that advantage. Discuss memory bandwidth, expert routing, and speculative decoding.")])
wr("reason_math", [U("A tank is filled by two pipes. Pipe A alone fills it in 6 hours, pipe B alone in 9 hours. A drain empties a full tank in 18 hours. If all three are open and the tank starts empty, how long does it take to fill? Then, if the drain is closed after 2 hours, how long in total? Show your work.")])

# ---- long context: the same edit task after ~24K tokens of unrelated documentation in the same user message (decode at 25K context)
import glob, random
paras = []
for f in sorted(glob.glob(os.path.expanduser("~/rrr/llama.cpp-inkling/docs/**/*.md"), recursive=True)):
    paras += [p for p in open(f, errors="ignore").read().split("\n\n") if len(p.strip()) > 40]
random.Random(5).shuffle(paras)
body, tot = [], 0
for p in paras:
    body.append(p); tot += len(C.tokenizer().encode(p, add_special_tokens=False).ids) + 2
    if tot >= 24000: break
wr("edit_whole_py_ctx24k", [U("Background documentation (unrelated, for context only):\n\n" + "\n\n".join(body) + "\n\n=====\n\nHere is a Python file:\n\n```python\n" + chat + "\n```\n\nRename the function `render_prompt` to `build_prompt` everywhere (keeping a backwards compatible alias `render_prompt = build_prompt` at the end), add an optional `system` parameter to it that, when given, prepends a system message, and add a short docstring to every function that lacks one. Output the complete edited file.")], eff=EFF, direct="<|message_model|><|content_text|>")

# ---- copy-heavy edits: the answer reproduces most of the prompt (whole-file rewrite without commentary, JSONL edit, edit tool call with big old/new strings)
wr("editcopy_file", [U("Here is a Python file:\n\n```python\n" + chat + "\n```\n\nChange the default of the `reasoning_effort` parameter of render_prompt to 0.5, and make `tokenizer()` thread-safe with a lock. Reply with the complete edited file in a single code block and nothing else (no explanation).")], eff=EFF, direct="<|message_model|><|content_text|>```python\n")
import random
rnd = random.Random(3)
rows = []
for i in range(40):
    rows.append(json.dumps({"id": i, "name": "item-%03d" % rnd.randrange(1000), "price": round(rnd.uniform(1, 99), 2), "tags": rnd.sample(["red", "blue", "new", "sale", "eco", "pro"], 3), "stock": rnd.randrange(500), "meta": {"created": "2025-0%d-1%d" % (rnd.randrange(1, 9), rnd.randrange(10)), "active": rnd.random() < 0.7}}))
wr("editcopy_jsonl", [U("Here is a JSONL file (one record per line):\n\n```jsonl\n" + "\n".join(rows) + "\n```\n\nAdd a field \"currency\": \"EUR\" to every record right after \"price\", and set \"active\" to false for records whose stock is below 100. Output the complete resulting JSONL in one code block and nothing else.")], eff=EFF, direct="<|message_model|><|content_text|>```jsonl\n")
rep_tool = [{"type": "function", "function": {"name": "str_replace", "description": "Replace an exact string in a file. old_str must match the file content exactly; new_str is the replacement.",
             "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "old_str": {"type": "string"}, "new_str": {"type": "string"}}, "required": ["path", "old_str", "new_str"]}}}]
src = R("server/chatfmt.py"); a = src.index("class StreamParser"); blk = src[a:a + 4500]
wr("editcopy_replace", [U("File server/chatfmt.py contains this part:\n\n```python\n" + blk + "\n```\n\nRefactor the whole block shown above: add type hints and docstrings to every method of the classes in it. Do it with one str_replace call that replaces the entire block shown (old_str = the block exactly as shown, new_str = the refactored block).")], tools=rep_tool, eff=EFF, direct='<|message_model|>str_replace<|content_invoke_tool_json|>{"name":"str_replace","args":{"path":"server/chatfmt.py","old_str":"')
