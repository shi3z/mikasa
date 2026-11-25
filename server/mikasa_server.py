#!/usr/bin/env python3
"""OpenAI-compatible HTTP front end for the mikasa engine (standard library + tokenizers/transformers for the tokenizer and chat template).

  python3 mikasa_server.py --port 8767 --engine mock                   # replays canned tokens (tests the protocol, no GPU)
  python3 mikasa_server.py --port 8767 --engine daemon --model X.gguf  # real engine (see engines.py)

Endpoints: GET /health, GET /props, GET /v1/models, POST /v1/chat/completions (stream or not; tools; chat_template_kwargs.reasoning_effort).
Streamed deltas use `reasoning_content`, `content` and `tool_calls` like llama-server, so the harness client works unchanged.
"""
import argparse, json, sys, threading, time, os, uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import chatfmt as C
from engines import make_engine

ENGINE = None
LOCK = threading.Lock()          # one sequence at a time until the engine can batch
STATS = dict(requests=0, prompt_tokens=0, completion_tokens=0)


def chunk(delta=None, finish=None, usage=None, cid="chatcmpl-x", model="inkling"):
    o = dict(id=cid, object="chat.completion.chunk", created=int(time.time()), model=model, choices=[] if usage else [dict(index=0, delta=delta or {}, finish_reason=finish)])
    if usage: o["usage"] = usage
    return "data: " + json.dumps(o, ensure_ascii=False) + "\n\n"


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    def log_message(self, *a): pass
    def _send(self, code, body, ctype="application/json"):
        b = body.encode() if isinstance(body, str) else body
        self.send_response(code); self.send_header("Content-Type", ctype); self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
    def do_GET(self):
        if self.path == "/health": return self._send(200, json.dumps({"status": "ok"}))
        if self.path == "/props": return self._send(200, json.dumps({"default_generation_settings": {"n_ctx": ENGINE.n_ctx}, "total_slots": 1, "build_info": "mikasa"}))
        if self.path in ("/v1/models", "/models"): return self._send(200, json.dumps({"object": "list", "data": [{"id": "inkling", "object": "model"}]}))
        if self.path == "/stats": return self._send(200, json.dumps(dict(STATS, engine=ENGINE.stats())))
        self._send(404, json.dumps({"error": {"message": "not found"}}))
    def do_POST(self):
        if self.path not in ("/v1/chat/completions", "/chat/completions"): return self._send(404, json.dumps({"error": {"message": "not found"}}))
        try:
            n = int(self.headers.get("Content-Length", "0")); req = json.loads(self.rfile.read(n))
        except Exception as e: return self._send(400, json.dumps({"error": {"message": "bad json: %s" % e}}))
        try: self.chat(req)
        except (BrokenPipeError, ConnectionResetError): pass
        except Exception as e:
            try: self._send(500, json.dumps({"error": {"message": repr(e)}}))
            except Exception: pass

    def chat(self, req):
        stream = bool(req.get("stream")); tools = req.get("tools") or None
        kw = req.get("chat_template_kwargs") or {}
        ids = C.render_prompt(req.get("messages") or [], tools=tools, reasoning_effort=kw.get("reasoning_effort"))
        max_new = int(req.get("max_tokens") or req.get("max_completion_tokens") or 4096)
        if len(ids) + 1 >= ENGINE.n_ctx:
            return self._send(400, json.dumps({"error": {"code": 400, "type": "exceed_context_size_error", "message": "the request exceeds the available context size (%d tokens, prompt %d)" % (ENGINE.n_ctx, len(ids)), "n_prompt_tokens": len(ids), "n_ctx": ENGINE.n_ctx}}))
        max_new = max(1, min(max_new, ENGINE.n_ctx - len(ids) - 1))
        sampling = dict(temperature=float(req.get("temperature", 1.0)), top_p=float(req.get("top_p", 1.0)), top_k=int(req.get("top_k", 0) or 0), seed=req.get("seed"))
        cancel = threading.Event(); cid = "chatcmpl-" + uuid.uuid4().hex[:12]
        parser = C.StreamParser(); content = reasoning = ""; calls = []; ntok = 0; finish = "length"; t0 = time.time()
        if stream:
            self.send_response(200); self.send_header("Content-Type", "text/event-stream"); self.send_header("Cache-Control", "no-store"); self.send_header("Transfer-Encoding", "chunked"); self.end_headers()
        def out(s):
            if not stream: return
            b = s.encode(); self.wfile.write(b"%x\r\n" % len(b) + b + b"\r\n"); self.wfile.flush()
        try:
            out(chunk(dict(role="assistant", content=None), cid=cid))
            with LOCK:
                for tid in ENGINE.generate(ids, max_new, sampling, cancel):
                    ntok += 1
                    for ev in parser.feed(tid):
                        k = ev[0]
                        if k == "reasoning": reasoning += ev[1]; out(chunk(dict(reasoning_content=ev[1]), cid=cid))
                        elif k == "content": content += ev[1]; out(chunk(dict(content=ev[1]), cid=cid))
                        elif k == "tool_start":
                            calls.append(dict(id="call_" + uuid.uuid4().hex[:10], type="function", function=dict(name=ev[2], arguments="")))
                            out(chunk(dict(tool_calls=[dict(index=ev[1], id=calls[-1]["id"], type="function", function=dict(name=ev[2], arguments=""))]), cid=cid))
                        elif k == "tool_args": calls[ev[1]]["function"]["arguments"] += ev[2]; out(chunk(dict(tool_calls=[dict(index=ev[1], function=dict(arguments=ev[2]))]), cid=cid))
                        elif k == "stop": finish = ev[1]
                    if parser.stopped: break
        except (BrokenPipeError, ConnectionResetError):
            cancel.set(); return                                  # client went away: the generator sees `cancel` and the engine stops
        finally: cancel.set()
        STATS["requests"] += 1; STATS["prompt_tokens"] += len(ids); STATS["completion_tokens"] += ntok
        usage = dict(prompt_tokens=len(ids), completion_tokens=ntok, total_tokens=len(ids) + ntok)
        if stream:
            out(chunk({}, finish=finish, cid=cid))
            if (req.get("stream_options") or {}).get("include_usage"): out(chunk(usage=usage, cid=cid))
            out("data: [DONE]\n\n"); self.wfile.write(b"0\r\n\r\n"); self.wfile.flush(); self.close_connection = True
        else:
            msg = dict(role="assistant", content=content or None); 
            if reasoning: msg["reasoning_content"] = reasoning
            if calls: msg["tool_calls"] = calls
            self._send(200, json.dumps(dict(id=cid, object="chat.completion", created=int(time.time()), model="inkling", choices=[dict(index=0, message=msg, finish_reason=finish)], usage=usage), ensure_ascii=False))


def main():
    global ENGINE
    ap = argparse.ArgumentParser(); ap.add_argument("--port", type=int, default=8767); ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--engine", default="mock"); ap.add_argument("--model", default=os.path.expanduser("~/rrr/gguf/Inkling-2-Mini-UD-Q4_K_XL.gguf"))
    ap.add_argument("--ctx", type=int, default=65536); ap.add_argument("--tps", type=float, default=90.0, help="mock engine speed")
    a = ap.parse_args(); ENGINE = make_engine(a.engine, model=a.model, n_ctx=a.ctx, tps=a.tps)
    ThreadingHTTPServer.daemon_threads = True
    srv = ThreadingHTTPServer((a.host, a.port), H)
    print("mikasa server on http://%s:%d  engine=%s n_ctx=%d" % (a.host, a.port, a.engine, ENGINE.n_ctx), flush=True)
    srv.serve_forever()

if __name__ == "__main__": main()
