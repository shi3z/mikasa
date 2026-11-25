"""Prompt rendering and streaming output parsing for Inkling-2 (no engine needed).

Prompt: the official chat_template.jinja (via transformers) -> token ids. The template has no generation prompt: the model itself emits
`<|message_model|>` first. Output grammar (one or more messages, then end of sampling):
    <|message_model|> <|content_thinking|> reasoning <|end_message|>
    <|message_model|> <|content_text|> answer <|end_message|>
    <|message_model|> NAME <|content_invoke_tool_json|> {"name":"NAME","args":{...}} <|end_message|>
    <|content_model_end_sampling|>            (id 200006: stop)
"""
import json, os
from tokenizers import Tokenizer

TOKDIR = os.environ.get("INKLING_TOK_DIR", os.path.expanduser("~/rrr/inkling-tok"))
T_MSG_MODEL, T_THINK, T_TEXT, T_END_MSG, T_STOP, T_INVOKE_JSON, T_EOT = 200001, 200008, 200004, 200010, 200006, 200049, 199999
STOP_IDS = (T_STOP, T_EOT)

_tk = None
_at = None
def tokenizer():
    global _tk
    if _tk is None: _tk = Tokenizer.from_file(os.path.join(TOKDIR, "tokenizer.json"))
    return _tk
def _template():
    global _at
    if _at is None:
        import logging, warnings; warnings.filterwarnings("ignore"); logging.getLogger("transformers").setLevel(logging.ERROR)
        from transformers import AutoTokenizer
        _at = AutoTokenizer.from_pretrained(TOKDIR)
    return _at


def _fix_messages(messages):
    """OpenAI -> template: tool_call arguments must be parsed objects, content may be None."""
    out = []
    for m in messages:
        m = dict(m)
        if m.get("content") is None: m["content"] = ""
        if m.get("tool_calls"):
            tcs = []
            for tc in m["tool_calls"]:
                fn = dict(tc["function"]); a = fn.get("arguments", {})
                if isinstance(a, str):
                    try: a = json.loads(a) if a.strip() else {}
                    except Exception: a = {"_raw": a}
                fn["arguments"] = a; tcs.append(dict(tc, function=fn))
            m["tool_calls"] = tcs
        out.append(m)
    return out


def render_prompt(messages, tools=None, reasoning_effort=None):
    kw = {}
    if reasoning_effort is not None: kw["reasoning_effort"] = reasoning_effort
    text = _template().apply_chat_template(_fix_messages(messages), tools=tools or None, tokenize=False, add_generation_prompt=True, **kw)
    return tokenizer().encode(text, add_special_tokens=False).ids


class Detok:
    """Incremental detokenizer: emits only text that cannot change when more tokens arrive (no half UTF-8 characters)."""
    def __init__(self): self.ids = []; self.sent = 0
    def push(self, tid):
        self.ids.append(tid); s = tokenizer().decode(self.ids, skip_special_tokens=False)
        if s.endswith("�"): return ""
        out = s[self.sent:]; self.sent = len(s); return out
    def flush(self):
        s = tokenizer().decode(self.ids, skip_special_tokens=False); out = s[self.sent:]; self.sent = len(s); return out


class StreamParser:
    """Feed token ids; get events: ('reasoning', text) ('content', text) ('tool_start', index, name) ('tool_args', index, fragment) ('stop', reason).
    Tool call arguments are streamed as the JSON text of the `args` object (what an OpenAI client expects in function.arguments)."""
    def __init__(self):
        self.state = "idle"          # idle | header | think | text | tool
        self.dt = Detok(); self.header = ""; self.ntools = 0; self.buf = ""; self.args_started = False; self.stopped = False
    def feed(self, tid):
        ev = []
        if self.stopped: return ev
        if tid in STOP_IDS:
            ev += self._close(); self.stopped = True; ev.append(("stop", "tool_calls" if self.ntools else "stop")); return ev
        if tid == T_MSG_MODEL: ev += self._close(); self.state = "header"; self.dt = Detok(); self.header = ""; return ev
        if tid == T_END_MSG: ev += self._close(); self.state = "idle"; return ev
        if self.state == "header":
            if tid == T_THINK: self.state = "think"; self.dt = Detok(); return ev
            if tid == T_TEXT: self.state = "text"; self.dt = Detok(); return ev
            if tid == T_INVOKE_JSON:
                self.state = "tool"; self.dt = Detok(); self.buf = ""; self.args_started = False
                ev.append(("tool_start", self.ntools, self.header.strip())); return ev
            self.header += self.dt.push(tid); return ev
        t = self.dt.push(tid)
        if not t: return ev
        if self.state == "think": ev.append(("reasoning", t))
        elif self.state == "text": ev.append(("content", t))
        elif self.state == "tool": ev += self._tool_text(t)
        return ev
    def _tool_text(self, t):
        ev = []
        self.buf += t
        if not self.args_started:
            i = self.buf.find('"args":')
            if i < 0: return ev
            self.args_started = True; self.buf = self.buf[i + len('"args":'):]
        if len(self.buf) > 1:                                    # hold back one char: the outer object's closing brace
            ev.append(("tool_args", self.ntools, self.buf[:-1])); self.buf = self.buf[-1:]
        return ev
    def _close(self):
        ev = []
        if self.state in ("think", "text", "tool"):
            t = self.dt.flush()
            if t and self.state == "think": ev.append(("reasoning", t))
            elif t and self.state == "text": ev.append(("content", t))
            elif self.state == "tool":
                if t: ev += self._tool_text(t)
                rest = self.buf[:-1] if self.buf.endswith("}") else self.buf    # drop the outer closing brace
                if rest: ev.append(("tool_args", self.ntools, rest))
                self.buf = ""; self.ntools += 1
        return ev
