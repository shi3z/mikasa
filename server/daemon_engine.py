"""DaemonEngine: adapter for the resident `mikasa --serve` daemon (JSON lines over a unix socket, see DAEMON.md).

generate(prompt_ids, max_new, sampling, cancel) -> stream object: iterate it to get generated token ids (the stop token included);
`.final` holds the daemon's final message afterwards ({finish, n_prompt, n_cached, n_gen, timings}). close() cancels a running request.
Many requests can be in flight at once (the daemon schedules up to `nslots` of them, the rest wait in its queue).
"""
import json, os, queue, socket, subprocess, threading, time, itertools, collections, ctypes, signal

STOP_IDS = (200006, 199999)


class DaemonError(RuntimeError): pass
class UnsupportedSampling(ValueError): pass


def _pdeathsig():
    try: ctypes.CDLL("libc.so.6").prctl(1, signal.SIGTERM)       # PR_SET_PDEATHSIG: the daemon dies with the server
    except Exception: pass


class _Stream:
    def __init__(self, eng, rid, q, cancel):
        self.eng, self.rid, self.q, self.cancel = eng, rid, q, cancel
        self.final = None; self.buf = collections.deque(); self.sent_cancel = False; self.closed = False
    def __iter__(self): return self
    def __next__(self):
        while True:
            if self.buf: return self.buf.popleft()
            if self.final is not None: raise StopIteration
            if self.cancel is not None and self.cancel.is_set() and not self.sent_cancel: self.sent_cancel = True; self.eng._send(dict(op="cancel", id=self.rid))
            try: m = self.q.get(timeout=0.05)
            except queue.Empty:
                if self.eng.dead: raise DaemonError("daemon connection lost" + (": " + self.eng.dead_reason if self.eng.dead_reason else ""))
                continue
            if m.get("op") == "tokens": self.buf.extend(m["tokens"])
            elif m.get("op") == "done":
                self.final = m; self.eng._forget(self.rid)
                if m.get("finish") == "error": raise DaemonError(m.get("error", "daemon error"))
            elif m.get("op") == "dead": raise DaemonError("daemon connection lost")
    def wait_final(self, timeout=3.0):
        """After the consumer saw the stop token: read the remaining messages up to the final one (normally immediate)."""
        t0 = time.time()
        while self.final is None and time.time() - t0 < timeout:
            try: m = self.q.get(timeout=0.05)
            except queue.Empty: continue
            if m.get("op") == "tokens": continue
            if m.get("op") == "done": self.final = m; self.eng._forget(self.rid)
            elif m.get("op") == "dead": return None
        return self.final
    def close(self):
        if self.closed: return
        self.closed = True
        if self.final is None:
            try: self.eng._send(dict(op="cancel", id=self.rid))
            except Exception: pass
            self.eng._forget(self.rid)
    def __del__(self):
        try: self.close()
        except Exception: pass


class DaemonEngine:
    concurrent = True                      # the server must not serialise requests
    def __init__(self, model=None, n_ctx=65536, sock=None, launch=False, daemon_bin=None, daemon_args=(), log_path=None, wait=900, out_dir=None):
        self.sock_path = sock or "/tmp/mikasa-%d.sock" % os.getpid(); self.proc = None
        self.dead = False; self.dead_reason = ""; self.lock = threading.Lock(); self.qs = {}; self.ids = itertools.count(1); self.waiters = []
        if launch:
            here = os.path.dirname(os.path.abspath(__file__))
            daemon_bin = daemon_bin or os.path.join(here, "..", "mikasa_daemon")
            out_dir = out_dir or "/tmp/mikasa-daemon-out"
            log = open(log_path, "w") if log_path else None
            self.proc = subprocess.Popen([daemon_bin, model, out_dir, "--serve", "--sock", self.sock_path] + list(daemon_args), stdout=log, stderr=subprocess.STDOUT if log else None,
                                         preexec_fn=_pdeathsig)
        self._connect(wait)
        info = self.info()
        self.n_ctx = int(info["n_ctx"]); self.nslots = int(info["nslots"]); self.daemon_info = info
        if n_ctx and n_ctx < self.n_ctx: self.n_ctx = n_ctx

    def _connect(self, wait):
        t0 = time.time()
        while True:
            if self.proc is not None and self.proc.poll() is not None: raise DaemonError("daemon exited with code %s while starting" % self.proc.returncode)
            try:
                s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM); s.connect(self.sock_path); break
            except OSError:
                s.close()
                if time.time() - t0 > wait: raise DaemonError("daemon socket %s did not appear in %d s" % (self.sock_path, wait))
                time.sleep(0.5)
        self.sock = s; self.wlock = threading.Lock()
        threading.Thread(target=self._reader, daemon=True).start()

    def _reader(self):
        buf = b""
        try:
            while True:
                d = self.sock.recv(1 << 20)
                if not d: break
                buf += d
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    if not line: continue
                    m = json.loads(line)
                    if m.get("op") in ("info", "stats"):
                        with self.lock: w = self.waiters.pop(0) if self.waiters else None
                        if w: w.put(m)
                        continue
                    with self.lock: q = self.qs.get(m.get("id"))
                    if q is not None: q.put(m)
        except Exception as e: self.dead_reason = repr(e)
        self.dead = True
        with self.lock:
            for q in self.qs.values(): q.put({"op": "dead"})
            for w in self.waiters: w.put({"op": "dead"})

    def _send(self, obj):
        b = (json.dumps(obj) + "\n").encode()
        with self.wlock:
            try: self.sock.sendall(b)
            except OSError as e: self.dead = True; raise DaemonError("send failed: %r" % e)

    def _forget(self, rid):
        with self.lock: self.qs.pop(rid, None)

    def _query(self, op):
        q = queue.Queue()
        with self.lock: self.waiters.append(q)
        self._send(dict(op=op))
        m = q.get(timeout=30)
        if m.get("op") == "dead": raise DaemonError("daemon connection lost")
        return m
    def info(self): return self._query("info")
    def stats(self): return self._query("stats")

    def validate(self, sampling):
        """Raise UnsupportedSampling for sampling parameters the daemon does not implement (never ignore them silently)."""
        if float(sampling.get("top_p", 1.0) or 1.0) < 1.0: raise UnsupportedSampling("top_p < 1 is not supported by the mikasa daemon (use top_p=1)")
        if int(sampling.get("top_k", 0) or 0) > 0: raise UnsupportedSampling("top_k > 0 is not supported by the mikasa daemon (use top_k=0)")
        if float(sampling.get("temperature", 1.0)) < 0: raise UnsupportedSampling("temperature < 0")

    def generate(self, ids, max_new, sampling, cancel=None, **opts):
        self.validate(sampling)
        rid = "r%d" % next(self.ids); q = queue.Queue()
        with self.lock: self.qs[rid] = q
        req = dict(op="gen", id=rid, prompt_ids=list(ids), max_new=int(max_new), temperature=float(sampling.get("temperature", 1.0)), stop_ids=list(STOP_IDS))
        if sampling.get("seed") is not None: req["seed"] = int(sampling["seed"])
        req.update(opts)                 # spec_k, no_cache (tests)
        try: self._send(req)
        except Exception: self._forget(rid); raise
        return _Stream(self, rid, q, cancel)

    def stats_dict(self):
        try: return self.stats()
        except Exception as e: return dict(error=repr(e))

    def close(self):
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            try: self.proc.wait(10)
            except Exception: self.proc.kill()
