"""Engine adapters. An engine turns a prompt (token ids) into a stream of generated token ids.
generate(prompt_ids, max_new, sampling, cancel_event) -> iterator of ints; it must stop promptly when `cancel` is set."""
import os, time, threading

class MockEngine:
    """Replays a fixed token sequence at a given speed: tests the HTTP/parse/stream path without a GPU."""
    def __init__(self, n_ctx=65536, tps=90.0, script=None):
        self.n_ctx = n_ctx; self.dt = 1.0 / tps; self.n = 0
        if script is None:
            import chatfmt as C
            txt = ('<|message_model|><|content_thinking|>The user wants a listing. I will run ls.<|end_message|>'
                   '<|message_model|>bash<|content_invoke_tool_json|>{"name":"bash","args":{"command":"ls -la"}}<|end_message|><|content_model_end_sampling|>')
            script = C.tokenizer().encode(txt, add_special_tokens=False).ids
        self.script = script
    def generate(self, ids, max_new, sampling, cancel):
        for i, t in enumerate(self.script[:max_new]):
            if cancel.is_set(): return
            time.sleep(self.dt); self.n += 1; yield t
    def stats(self): return dict(kind="mock", tokens=self.n)

def make_engine(kind, **kw):
    if kind == "mock": return MockEngine(n_ctx=kw.get("n_ctx", 65536), tps=kw.get("tps", 90.0))
    if kind == "daemon":
        from daemon_engine import DaemonEngine
        return DaemonEngine(model=kw["model"], n_ctx=kw.get("n_ctx", 65536))
    raise SystemExit("unknown engine " + kind)
