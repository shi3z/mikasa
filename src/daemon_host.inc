// Resident daemon (`mikasa model out --serve`): loads the model once, serves generation requests over a local socket (JSON lines).
// Textually included after long_host.inc / prefill_host.inc / spec_host.inc. See DAEMON.md for the protocol and the design.
//
//  * 4 sequence slots (KV caches of lean mode). One GPU thread runs a scheduler loop; reader threads (one per connection) only parse requests.
//  * Per slot the token history whose KV / conv state is valid is remembered. A new request goes to the idle slot with the longest reusable prefix:
//      - "resident": the slot's current state == its whole history is a prefix of the prompt -> prefill only the rest,
//      - "snapshot": a snapshot (conv states + the last 1040 rows of every SWA ring + position) taken at a chunk boundary (multiple of 512) of an earlier
//        prompt is restored; the full-attention KV is linear, rows below the position are still in the slot, rows above are overwritten,
//      - else prefill from scratch on the least recently used idle slot.
//  * Scheduling: one prefill chunk (<= 512 tokens, ends at multiples of 256) of one request, then one decode step of all generating requests, repeat.
//    1 generating request: speculative step (n-gram / copy drafts, forward_batch rows of ONE sequence); >= 2: forward_batch multi mode, one token each.
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <deque>
#include <memory>
#include <csignal>
#include <cerrno>
#include <sys/socket.h>
#include <sys/un.h>
#include <netinet/in.h>
#include <unistd.h>
#include <arpa/inet.h>

struct DArgs {
    std::string sock = "/tmp/mikasa.sock";
    int tcp = 0;                 // >0: also listen on 127.0.0.1:tcp
    int snaps = 40;              // max snapshots (each ~174 MB)
    bool snap_end = false;       // also snapshot the exact end of every prompt (unaligned; default: only the last multiple of 512)
    int chunk = 512;             // prefill chunk when nobody else is generating (256 otherwise)
    int quiet = 0;
};
static DArgs g_dargs;
static std::string g_sock_path;

// ---------------------------------------------------------------------------------------------- tiny JSON
struct JV {
    enum T { NUL, NUM, STR, BOOL, ARR, OBJ } t = NUL;
    double n = 0; bool b = false; std::string s;
    std::vector<double> nums;                       // arrays that contain only numbers
    std::vector<JV> a;                              // other arrays
    std::vector<std::pair<std::string, JV>> o;
    const JV * get(const char * k) const { for (auto & kv : o) if (kv.first == k) return &kv.second; return nullptr; }
    double num(const char * k, double d) const { const JV * v = get(k); return v && v->t == NUM ? v->n : d; }
    bool has(const char * k) const { const JV * v = get(k); return v && v->t != NUL; }
    std::string str(const char * k, const std::string & d = "") const { const JV * v = get(k); return v && v->t == STR ? v->s : d; }
    bool bl(const char * k, bool d) const { const JV * v = get(k); return v && v->t == BOOL ? v->b : (v && v->t == NUM ? v->n != 0 : d); }
};
struct JP {
    const std::string & s; size_t i = 0; bool ok = true;
    explicit JP(const std::string & s_) : s(s_) {}
    void ws() { while (i < s.size() && (s[i] == ' ' || s[i] == '\t' || s[i] == '\r' || s[i] == '\n')) ++i; }
    bool lit(const char * w) { const size_t n = std::strlen(w); if (s.compare(i, n, w) == 0) { i += n; return true; } return false; }
    JV val() {
        JV v; ws();
        if (i >= s.size()) { ok = false; return v; }
        const char c = s[i];
        if (c == '{') {
            v.t = JV::OBJ; ++i; ws();
            if (i < s.size() && s[i] == '}') { ++i; return v; }
            while (ok) {
                ws(); JV k = val(); if (!ok || k.t != JV::STR) { ok = false; break; }
                ws(); if (i >= s.size() || s[i] != ':') { ok = false; break; } ++i;
                JV x = val(); v.o.emplace_back(k.s, std::move(x)); ws();
                if (i < s.size() && s[i] == ',') { ++i; continue; }
                if (i < s.size() && s[i] == '}') { ++i; break; }
                ok = false;
            }
        } else if (c == '[') {
            v.t = JV::ARR; ++i; ws();
            if (i < s.size() && s[i] == ']') { ++i; return v; }
            bool allnum = true;
            while (ok) {
                JV x = val(); if (!ok) break;
                if (x.t != JV::NUM) allnum = false;
                if (allnum) v.nums.push_back(x.n);
                else { if (!v.nums.empty() || v.a.empty()) { for (double d : v.nums) { JV y; y.t = JV::NUM; y.n = d; v.a.push_back(y); } v.nums.clear(); } v.a.push_back(std::move(x)); }
                ws();
                if (i < s.size() && s[i] == ',') { ++i; continue; }
                if (i < s.size() && s[i] == ']') { ++i; break; }
                ok = false;
            }
        } else if (c == '"') {
            v.t = JV::STR; ++i;
            while (i < s.size() && s[i] != '"') {
                if (s[i] == '\\' && i + 1 < s.size()) {
                    ++i; const char e = s[i];
                    if (e == 'n') v.s += '\n'; else if (e == 't') v.s += '\t'; else if (e == 'u') { v.s += '?'; i += 4; } else v.s += e;
                } else v.s += s[i];
                ++i;
            }
            if (i >= s.size()) ok = false; else ++i;
        } else if (lit("true")) { v.t = JV::BOOL; v.b = true; }
        else if (lit("false")) { v.t = JV::BOOL; v.b = false; }
        else if (lit("null")) { v.t = JV::NUL; }
        else {
            char * e = nullptr; const double d = std::strtod(s.c_str() + i, &e);
            if (e == s.c_str() + i) { ok = false; return v; }
            v.t = JV::NUM; v.n = d; i = e - s.c_str();
        }
        return v;
    }
};
static std::string jesc(const std::string & x) {
    std::string r;
    for (char c : x) { if (c == '"' || c == '\\') { r += '\\'; r += c; } else if (c == '\n') r += "\\n"; else if ((unsigned char) c < 32) r += ' '; else r += c; }
    return r;
}

// ---------------------------------------------------------------------------------------------- connections / requests
struct DConn {
    int fd = -1; std::mutex wm; std::atomic<bool> dead{false};
};
struct DReq {
    std::shared_ptr<DConn> conn; std::string id; std::vector<int> prompt;
    int max_new = 256; float temp = 1.0f; unsigned seed = 1; std::vector<int> stop; int spec_k = -1; bool no_cache = false;
    std::atomic<bool> cancel{false}; double t_arrive = 0;
};
static bool send_line(DConn & c, const std::string & s) {
    if (c.dead) return false;
    std::lock_guard<std::mutex> lk(c.wm);
    std::string b = s; b.push_back('\n');
    size_t off = 0;
    while (off < b.size()) {
        const ssize_t n = ::send(c.fd, b.data() + off, b.size() - off, MSG_NOSIGNAL);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) { c.dead = true; return false; }
        off += (size_t) n;
    }
    return true;
}

// ---------------------------------------------------------------------------------------------- GPU helpers
struct SwapEnt { float * a; float * b; int n; };
__global__ void k_swap_tab(const SwapEnt * __restrict__ tab) {
    const SwapEnt e = tab[blockIdx.y];
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= e.n) return;
    const float x = e.a[i]; e.a[i] = e.b[i]; e.b[i] = x;
}

constexpr int SNAP_WIN = NSWA + 16;                     // positions kept per SWA ring in a snapshot (1040)
constexpr int NSWA_APPS = NAPP - NAPP / 6;              // 40
struct DSnap { int p = 0; int slot = -1; float * ring = nullptr; float * conv = nullptr; double used = 0; };
struct DSeq {
    std::vector<int> out; NgramIndex NG, NGC; int sticky = -1, bk = 0, Kmax = 0; unsigned step = 0;
    double tried[10][64], accd[10][64];
    long steps = 0, drafted = 0, accepted = 0, copy_steps = 0, nodraft = 0, multi_steps = 0;
    DSeq() { for (int b = 0; b < 10; ++b) for (int i = 0; i < 64; ++i) { tried[b][i] = 0; accd[b][i] = 0; } }
};
struct DSlot {
    std::vector<int> hist;                  // tokens whose KV / conv state are in the slot (positions 0..size-1)
    std::vector<DSnap *> snaps;
    int phase = 0;                          // 0 idle, 1 prefill, 2 generate
    double last_used = 0;
    std::shared_ptr<DReq> req;
    int rb = -1;                            // pending lazy conv rollback (of the last speculative step on this slot)
    int pf_pos = 0, n_cached = 0, snap_q = 0, how = 0; double t_start = 0, t_first = 0, t_pf = 0;
    std::unique_ptr<DSeq> sq;
};

struct DRunner {
    Engine & g; BatchState & X; cudaStream_t s;
    cudaGraphExec_t gs[2][17] = {}, gm[2][5] = {};
    int *h_tok, *h_pos, *h_slot, *h_rb, *h_dr, *h_arg, *h_acc, *h_stok; float * h_prm;
    float * d_prm; int *d_dr, *d_acc, *d_stok;
    DRunner(Engine & g_, BatchState & X_, cudaStream_t s_) : g(g_), X(X_), s(s_) {
        CHECK(cudaMallocHost(&h_tok, 64)); CHECK(cudaMallocHost(&h_pos, 64)); CHECK(cudaMallocHost(&h_slot, 64)); CHECK(cudaMallocHost(&h_rb, 4));
        CHECK(cudaMallocHost(&h_dr, 64)); CHECK(cudaMallocHost(&h_arg, 64)); CHECK(cudaMallocHost(&h_acc, 64)); CHECK(cudaMallocHost(&h_stok, 64));
        CHECK(cudaMallocHost(&h_prm, 256));
        CHECK(cudaMalloc(&d_prm, 256)); CHECK(cudaMalloc(&d_dr, 64)); CHECK(cudaMalloc(&d_acc, 64)); CHECK(cudaMalloc(&d_stok, 64));
    }
    cudaGraphExec_t graph(int B, bool samp, bool multi) {
        cudaGraphExec_t & ex = multi ? gm[samp][B] : gs[samp][B];
        if (ex) return ex;
        cudaGraph_t gr;
        CHECK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
        if (!multi) {
            CHECK(cudaMemcpyAsync(X.rb_j, h_rb, 4, cudaMemcpyHostToDevice, s));
            kb_rollback<<<dim3((E + 255) / 256, NAPP * 4), 256, 0, s>>>(X.rb_tab, X.rb_j);
        }
        CHECK(cudaMemcpyAsync(X.btok, h_tok, B * 4, cudaMemcpyHostToDevice, s));
        CHECK(cudaMemcpyAsync(X.pos_arr, h_pos, B * 4, cudaMemcpyHostToDevice, s));
        CHECK(cudaMemcpyAsync(X.slot_arr, h_slot, B * 4, cudaMemcpyHostToDevice, s));
        if (samp) {
            CHECK(cudaMemcpyAsync(d_dr, h_dr, B * 4, cudaMemcpyHostToDevice, s));
            CHECK(cudaMemcpyAsync(d_prm, h_prm, B * 16, cudaMemcpyHostToDevice, s));
        }
        forward_batch(g, X, B, multi ? 1 : 0, s);
        CHECK(cudaMemcpyAsync(h_arg, X.barg, B * 4, cudaMemcpyDeviceToHost, s));
        if (samp) {
            k_spec_sample<<<B, 1024, 0, s>>>(X.logits, NVOCAB, NVOCAB_PAD, d_dr, d_prm, d_acc, d_stok, 4);
            CHECK(cudaMemcpyAsync(h_acc, d_acc, B * 4, cudaMemcpyDeviceToHost, s));
            CHECK(cudaMemcpyAsync(h_stok, d_stok, B * 4, cudaMemcpyDeviceToHost, s));
        }
        CHECK(cudaStreamEndCapture(s, &gr));
        CHECK(cudaGraphInstantiate(&ex, gr, nullptr, nullptr, 0));
        CHECK(cudaGraphDestroy(gr));
        return ex;
    }
    void run(int B, bool samp, bool multi) { CHECK(cudaGraphLaunch(graph(B, samp, multi), s)); CHECK(cudaStreamSynchronize(s)); }
    void set_prm(int row, float temp, unsigned seed, unsigned step) {
        h_prm[row * 4] = temp > 0 ? 1.0f / temp : 1.0f; std::memcpy(&h_prm[row * 4 + 1], &seed, 4); std::memcpy(&h_prm[row * 4 + 2], &step, 4);
    }
};

// ---------------------------------------------------------------------------------------------- the daemon
struct Daemon {
    Engine & g; BatchState & X; PFState & PF; cudaStream_t s; const LongArgs & A; DRunner dr;
    DSlot slots[4];
    int resident = 0;                       // slot whose conv state is in the engine's slot-0 arrays (0 = natural layout)
    SwapEnt * d_swap[4] = {nullptr, nullptr, nullptr, nullptr};
    double cost[17] = {0};
    std::vector<DSnap *> pool; std::vector<DSnap *> free_snaps; int nsnap_alloc = 0;
    int *pf_tok, *pf_pos, *pf_slot, *pf_arg, *w_rb;
    std::mutex qm; std::condition_variable qcv;
    std::deque<std::shared_ptr<DReq>> pending;
    std::map<std::pair<DConn *, std::string>, std::shared_ptr<DReq>> all;
    std::atomic<long> n_req{0}, n_prompt{0}, n_cached{0}, n_gen{0}, n_done{0};
    double t0 = 0; int rr = 0;
    long snap_taken = 0, snap_restored = 0, resident_hits = 0;
    std::atomic<int> st_active{0}, st_snaps{0};

    Daemon(Engine & g_, BatchState & X_, PFState & PF_, cudaStream_t s_, const LongArgs & A_) : g(g_), X(X_), PF(PF_), s(s_), A(A_), dr(g_, X_, s_) {
        CHECK(cudaMallocHost(&pf_tok, PF.PB * 4)); CHECK(cudaMallocHost(&pf_pos, PF.PB * 4)); CHECK(cudaMallocHost(&pf_slot, PF.PB * 4));
        CHECK(cudaMallocHost(&pf_arg, 64)); CHECK(cudaMallocHost(&w_rb, 4));
        for (int sl = 1; sl < 4; ++sl) {
            std::vector<SwapEnt> t;
            for (int il = 0; il < NAPP; ++il) {
                float * a[4] = {g.kstate[il], g.vstate[il], g.poststate[il], g.ffnstate[il]};
                float * b[4] = {X.Sxk[il], X.Sxv[il], X.Sxp[il], X.Sxf[il]};
                const int n[4] = {3 * NKV * HD, 3 * NKV * HD, 3 * E, 3 * E};
                for (int k = 0; k < 4; ++k) t.push_back({a[k], b[k] + (size_t) (sl - 1) * n[k], n[k]});
            }
            CHECK(cudaMalloc(&d_swap[sl], t.size() * sizeof(SwapEnt)));
            CHECK(cudaMemcpy(d_swap[sl], t.data(), t.size() * sizeof(SwapEnt), cudaMemcpyHostToDevice));
        }
        t0 = spec_now();
    }

    // ------------------------------------------------------------------ conv state residency
    void swap_slot(int sl) { if (sl > 0) k_swap_tab<<<dim3((3 * E + 255) / 256, NAPP * 4), 256, 0, s>>>(d_swap[sl]); }
    void eager_rollback(int j) {
        *w_rb = j;
        CHECK(cudaMemcpyAsync(X.rb_j, w_rb, 4, cudaMemcpyHostToDevice, s));
        kb_rollback<<<dim3((E + 255) / 256, NAPP * 4), 256, 0, s>>>(X.rb_tab, X.rb_j);
        CHECK(cudaStreamSynchronize(s));
    }
    void flush_rb() { if (slots[resident].rb >= 0) { eager_rollback(slots[resident].rb); slots[resident].rb = -1; } }
    void ensure_resident(int sl) {
        if (resident == sl) return;
        flush_rb();
        swap_slot(resident); resident = 0;          // back to the natural layout
        swap_slot(sl); resident = sl;
    }
    void zero_resident_conv() {
        for (int il = 0; il < NAPP; ++il) {
            CHECK(cudaMemsetAsync(g.kstate[il], 0, 3 * NKV * HD * 4, s)); CHECK(cudaMemsetAsync(g.vstate[il], 0, 3 * NKV * HD * 4, s));
            CHECK(cudaMemsetAsync(g.poststate[il], 0, 3 * E * 4, s)); CHECK(cudaMemsetAsync(g.ffnstate[il], 0, 3 * E * 4, s));
        }
    }

    // ------------------------------------------------------------------ snapshots
    DSnap * alloc_snap() {
        if (!free_snaps.empty()) { DSnap * x = free_snaps.back(); free_snaps.pop_back(); return x; }
        if (nsnap_alloc < g_dargs.snaps) {
            DSnap * x = new DSnap();
            CHECK(cudaMalloc(&x->ring, (size_t) NSWA_APPS * 2 * SNAP_WIN * NKV * HD * 4));
            CHECK(cudaMalloc(&x->conv, (size_t) NAPP * (6 * NKV * HD + 6 * E) * 4));
            pool.push_back(x); ++nsnap_alloc; return x;
        }
        DSnap * v = nullptr;                        // evict the least recently used one
        for (int sl = 0; sl < 4; ++sl) for (DSnap * x : slots[sl].snaps) if (!v || x->used < v->used) v = x;
        if (!v) return nullptr;
        drop_snap(v); DSnap * x = free_snaps.back(); free_snaps.pop_back(); return x;
    }
    void drop_snap(DSnap * x) {
        auto & v = slots[x->slot].snaps;
        v.erase(std::find(v.begin(), v.end(), x)); free_snaps.push_back(x); x->slot = -1;
    }
    void drop_snaps_above(int sl, int c) {
        std::vector<DSnap *> tmp = slots[sl].snaps;
        for (DSnap * x : tmp) if (x->p > c) drop_snap(x);
    }
    // copy the last SNAP_WIN positions before p of every SWA ring between the slot and a snapshot buffer
    void ring_xfer(bool save, int sl, int p, DSnap * sn) {
        const int n = std::min(p, SNAP_WIN), start = p - n, ring = X.ring;
        const int r0 = start % ring, l1 = std::min(n, ring - r0), l2 = n - l1;
        const size_t row = (size_t) NKV * HD;
        int idx = 0;
        for (int il = 0; il < NAPP; ++il) {
            if (!app_is_swa(il)) continue;
            for (int kv = 0; kv < 2; ++kv) {
                float * slot_buf = (float *) (kv ? X.kvs[il].v[sl] : X.kvs[il].k[sl]);
                float * sb = sn->ring + ((size_t) idx * 2 + kv) * SNAP_WIN * row;
                if (save) {
                    CHECK(cudaMemcpyAsync(sb, slot_buf + (size_t) r0 * row, (size_t) l1 * row * 4, cudaMemcpyDeviceToDevice, s));
                    if (l2) CHECK(cudaMemcpyAsync(sb + (size_t) l1 * row, slot_buf, (size_t) l2 * row * 4, cudaMemcpyDeviceToDevice, s));
                } else {
                    CHECK(cudaMemcpyAsync(slot_buf + (size_t) r0 * row, sb, (size_t) l1 * row * 4, cudaMemcpyDeviceToDevice, s));
                    if (l2) CHECK(cudaMemcpyAsync(slot_buf, sb + (size_t) l1 * row, (size_t) l2 * row * 4, cudaMemcpyDeviceToDevice, s));
                }
            }
            ++idx;
        }
    }
    void conv_xfer(bool save, DSnap * sn) {          // resident conv state <-> snapshot
        size_t off = 0;
        for (int il = 0; il < NAPP; ++il) {
            float * a[4] = {g.kstate[il], g.vstate[il], g.poststate[il], g.ffnstate[il]};
            const size_t n[4] = {(size_t) 3 * NKV * HD, (size_t) 3 * NKV * HD, (size_t) 3 * E, (size_t) 3 * E};
            for (int k = 0; k < 4; ++k) {
                if (save) CHECK(cudaMemcpyAsync(sn->conv + off, a[k], n[k] * 4, cudaMemcpyDeviceToDevice, s));
                else CHECK(cudaMemcpyAsync(a[k], sn->conv + off, n[k] * 4, cudaMemcpyDeviceToDevice, s));
                off += n[k];
            }
        }
    }
    void take_snapshot(int sl, int p) {                   // slot sl must be resident, state exactly after p tokens
        for (DSnap * x : slots[sl].snaps) if (x->p == p) { x->used = spec_now(); return; }
        DSnap * sn = alloc_snap();
        if (!sn) return;
        ring_xfer(true, sl, p, sn); conv_xfer(true, sn);
        sn->p = p; sn->slot = sl; sn->used = spec_now(); slots[sl].snaps.push_back(sn); ++snap_taken;
    }

    // ------------------------------------------------------------------ replies
    static std::string fnum(double v) { char b[64]; std::snprintf(b, sizeof b, "%.3f", v); return b; }
    void send_done(const std::shared_ptr<DReq> & r, const char * finish, int n_prompt_, int n_cached_, int n_gen_, const std::string & timings, const std::string & err = "") {
        std::string m = "{\"op\":\"done\",\"id\":\"" + jesc(r->id) + "\",\"finish\":\"" + finish + "\",\"n_prompt\":" + std::to_string(n_prompt_) +
                        ",\"n_cached\":" + std::to_string(n_cached_) + ",\"n_gen\":" + std::to_string(n_gen_);
        if (!timings.empty()) m += ",\"timings\":" + timings;
        if (!err.empty()) m += ",\"error\":\"" + jesc(err) + "\"";
        m += "}";
        send_line(*r->conn, m);
        std::lock_guard<std::mutex> lk(qm);
        all.erase({r->conn.get(), r->id});
    }
    void send_tokens(const DReq & r, const std::vector<int> & v) {
        if (v.empty()) return;
        std::string m = "{\"op\":\"tokens\",\"id\":\"" + jesc(r.id) + "\",\"tokens\":[";
        for (size_t i = 0; i < v.size(); ++i) { if (i) m += ','; m += std::to_string(v[i]); }
        m += "]}";
        send_line(*r.conn, m);
    }
    void finish(int sl, const char * why) {
        DSlot & S = slots[sl];
        const double now = spec_now();
        const int ngen = S.sq ? (int) S.sq->out.size() : 0;
        const double dec = S.t_first > 0 ? now - S.t_first : 0;
        std::string tm = "{\"queue_ms\":" + fnum((S.t_start - S.req->t_arrive) * 1000) + ",\"prefill_ms\":" + fnum(S.t_first > 0 ? (S.t_first - S.t_start) * 1000 : (now - S.t_start) * 1000) +
                         ",\"ttft_ms\":" + fnum(S.t_first > 0 ? (S.t_first - S.req->t_arrive) * 1000 : 0) + ",\"decode_ms\":" + fnum(dec * 1000) +
                         ",\"decode_tps\":" + fnum(dec > 0 && ngen > 1 ? (ngen - 1) / dec : 0) + ",\"cache\":\"" + (S.how == 2 ? "snapshot" : S.how == 1 ? "resident" : "none") + "\"";
        if (S.sq) tm += ",\"steps\":" + std::to_string(S.sq->steps) + ",\"multi_steps\":" + std::to_string(S.sq->multi_steps) + ",\"drafted\":" + std::to_string(S.sq->drafted) +
                        ",\"accepted\":" + std::to_string(S.sq->accepted);
        tm += "}";
        std::shared_ptr<DReq> r = S.req;
        n_gen += ngen; n_done++;
        if (!g_dargs.quiet)
            std::printf("[daemon] slot %d id=%s %s: prompt %zu cached %d (%s) gen %d, prefill %.0f ms, decode %.1f tok/s\n", sl, r->id.c_str(), why, r->prompt.size(), S.n_cached,
                        S.how == 2 ? "snapshot" : S.how == 1 ? "resident" : "none", ngen, S.t_first > 0 ? (S.t_first - S.t_start) * 1000 : 0.0, dec > 0 && ngen > 1 ? (ngen - 1) / dec : 0.0), std::fflush(stdout);
        send_done(r, why, (int) r->prompt.size(), S.n_cached, ngen, tm);
        S.phase = 0; S.req.reset(); S.sq.reset(); S.last_used = now;
    }

    // ------------------------------------------------------------------ admission
    static int lcp(const std::vector<int> & a, const std::vector<int> & b) {
        const size_t n = std::min(a.size(), b.size()); size_t i = 0;
        while (i < n && a[i] == b[i]) ++i;
        return (int) i;
    }
    bool admit(const std::shared_ptr<DReq> & r) {
        const int N = (int) r->prompt.size();
        int best_slot = -1, best_p = 0, best_how = 0; DSnap * best_snap = nullptr;
        int lru = -1;
        for (int sl = 0; sl < A.nslots; ++sl) {
            DSlot & S = slots[sl];
            if (S.phase != 0) continue;
            if (lru < 0 || S.last_used < slots[lru].last_used) lru = sl;
            if (r->no_cache) continue;
            const int c = std::min(lcp(r->prompt, S.hist), N - 1);
            if ((int) S.hist.size() == c && c > best_p) { best_p = c; best_slot = sl; best_how = 1; best_snap = nullptr; }
            for (DSnap * x : S.snaps) if (x->p <= c && x->p > best_p) { best_p = x->p; best_slot = sl; best_how = 2; best_snap = x; }
        }
        if (lru < 0) return false;
        if (best_p < 32) { best_slot = lru; best_p = 0; best_how = 0; best_snap = nullptr; }
        DSlot & S = slots[best_slot];
        flush_rb();
        ensure_resident(best_slot);
        if (best_how == 0) { S.hist.clear(); drop_snaps_above(best_slot, -1); zero_resident_conv(); }
        else if (best_how == 2) {
            ring_xfer(false, best_slot, best_snap->p, best_snap); conv_xfer(false, best_snap);
            S.hist.resize(best_snap->p); drop_snaps_above(best_slot, best_snap->p); best_snap->used = spec_now(); ++snap_restored;
        } else { ++resident_hits; drop_snaps_above(best_slot, (int) S.hist.size()); }
        S.rb = -1; S.req = r; S.phase = 1; S.pf_pos = best_p; S.n_cached = best_p; S.how = best_how; S.t_start = spec_now(); S.t_first = 0;
        S.sq.reset(new DSeq());
        S.snap_q = (N / 512) * 512;
        n_prompt += N; n_cached += best_p;
        return true;
    }

    // ------------------------------------------------------------------ prefill (one chunk)
    void prefill_chunk(int sl, bool others_generating) {
        DSlot & S = slots[sl]; DReq & R = *S.req;
        const int N = (int) R.prompt.size();
        flush_rb(); ensure_resident(sl);
        const int chunk = others_generating ? 256 : std::min(g_dargs.chunk, PF.PB);
        const int start = S.pf_pos;
        int end = std::min(N, (start / 256 + chunk / 256) * 256);
        if (start < S.snap_q && end > S.snap_q) end = S.snap_q;
        if (end <= start) end = std::min(N, start + 256);
        const int T = end - start, hk = (end == N) ? 1 : 0;
        for (int k = 0; k < T; ++k) { pf_tok[k] = R.prompt[start + k]; pf_pos[k] = start + k; pf_slot[k] = sl; }
        CHECK(cudaMemcpyAsync(PF.btok, pf_tok, T * 4, cudaMemcpyHostToDevice, s));
        CHECK(cudaMemcpyAsync(PF.pos_arr, pf_pos, T * 4, cudaMemcpyHostToDevice, s));
        CHECK(cudaMemcpyAsync(PF.slot_arr, pf_slot, T * 4, cudaMemcpyHostToDevice, s));
        forward_prefill(g, X, PF, T, hk, s);
        S.hist.insert(S.hist.end(), R.prompt.begin() + start, R.prompt.begin() + end);
        S.pf_pos = end;
        if (end == S.snap_q && end > 0) take_snapshot(sl, end);
        if (end == N && g_dargs.snap_end && N != S.snap_q) take_snapshot(sl, N);
        if (hk) {
            int first;
            if (R.temp > 0) {
                dr.set_prm(0, R.temp, R.seed, 0x7FFFFFF0u); dr.h_dr[0] = -1;
                CHECK(cudaMemcpyAsync(dr.d_dr, dr.h_dr, 4, cudaMemcpyHostToDevice, s)); CHECK(cudaMemcpyAsync(dr.d_prm, dr.h_prm, 16, cudaMemcpyHostToDevice, s));
                k_spec_sample<<<1, 1024, 0, s>>>(X.logits, NVOCAB, NVOCAB_PAD, dr.d_dr, dr.d_prm, dr.d_acc, dr.d_stok, 4);
                CHECK(cudaMemcpyAsync(dr.h_stok, dr.d_stok, 4, cudaMemcpyDeviceToHost, s));
                CHECK(cudaStreamSynchronize(s)); first = dr.h_stok[0];
            } else {
                CHECK(cudaMemcpyAsync(pf_arg, X.barg, 4, cudaMemcpyDeviceToHost, s)); CHECK(cudaStreamSynchronize(s)); first = pf_arg[0];
            }
            begin_gen(sl, first);
        } else CHECK(cudaStreamSynchronize(s));
    }
    void begin_gen(int sl, int first) {
        DSlot & S = slots[sl]; DReq & R = *S.req; DSeq & Q = *S.sq;
        S.t_first = spec_now(); S.phase = 2;
        Q.Kmax = std::max(0, std::min(R.spec_k >= 0 ? R.spec_k : A.spec.k, MB - 1));
        Q.NG.M = std::max(1, A.spec.m); Q.NG.chain = A.spec.chain; Q.NG.longest = A.spec.longest;
        Q.NGC.M = 3; Q.NGC.chain = 64; Q.NGC.longest = true;
        for (int t : R.prompt) { Q.NG.push(t); if (A.spec.copy_min > 0) Q.NGC.push(t); }
        Q.out.push_back(first); Q.NG.push(first); if (A.spec.copy_min > 0) Q.NGC.push(first);
        send_tokens(R, {first});
        bool stop = false;
        for (int t : R.stop) if (t == first) stop = true;
        if (stop) finish(sl, "stop");
        else if ((int) Q.out.size() >= R.max_new) finish(sl, "length");
        else if ((int) S.hist.size() + 1 >= X.maxctx) finish(sl, "length");
    }

    // ------------------------------------------------------------------ decoding
    // consume the outcome of one step: `fed` tokens went into the cache, `emit` are the new tokens (last one not fed yet). Returns true if the request finished.
    bool commit(int sl, const int * fed, int nfed, const std::vector<int> & emit) {
        DSlot & S = slots[sl]; DReq & R = *S.req; DSeq & Q = *S.sq;
        for (int i = 0; i < nfed; ++i) S.hist.push_back(fed[i]);
        std::vector<int> sent; const char * why = nullptr;
        for (int t : emit) {
            if (Q.sticky >= 0) { if (Q.sticky < (int) Q.NG.ctx.size() && Q.NG.ctx[Q.sticky] == t) ++Q.sticky; else Q.sticky = -1; }
            Q.out.push_back(t); Q.NG.push(t); if (A.spec.copy_min > 0) Q.NGC.push(t);
            sent.push_back(t);
            bool stop = false; for (int x : R.stop) if (x == t) stop = true;
            if (stop) { why = "stop"; break; }
            if ((int) Q.out.size() >= R.max_new) { why = "length"; break; }
        }
        send_tokens(R, sent);
        if (!why && (int) S.hist.size() + 1 >= X.maxctx) why = "length";
        if (why) { finish(sl, why); return true; }
        return false;
    }

    void gen_single(int sl) {
        DSlot & S = slots[sl]; DReq & R = *S.req; DSeq & Q = *S.sq;
        ensure_resident(sl);
        const int pos = (int) S.hist.size();
        const bool samp = R.temp > 0;
        int toks[16];
        int k = 0, L = 0, cont = 0;
        auto bucket = [](int l) { return l <= 1 ? 0 : l == 2 ? 1 : l == 3 ? 2 : l <= 5 ? 3 : l <= 9 ? 4 : l <= 15 ? 5 : l <= 31 ? 6 : 7; };
        auto qest = [&](int b, int i) {
            double ta = 0, aa = 0;
            for (int j = 0; j < 64; ++j) { ta += Q.tried[b][j]; aa += Q.accd[b][j]; }
            const double pr = (aa + 1.0) / (ta + 2.0);
            return (Q.accd[b][i] + 2.0 * pr) / (Q.tried[b][i] + 2.0);
        };
        const int Kmax = Q.Kmax;
        if (Kmax > 0 && pos + 1 < X.maxctx) {
            auto choose = [&](int b, int avail) {
                double prod = 1.0, Esum = 1.0, best = 1.0 / cost[1];
                int kk = 0;
                for (int i = 1; i <= avail; ++i) {
                    prod *= qest(b, i - 1); Esum += prod;
                    const double v = Esum / cost[i + 1];
                    if (v > best) { best = v; kk = i; }
                }
                return kk;
            };
            bool used = false;
            if (A.spec.copy_min > 0) {
                const int l = (int) Q.NG.ctx.size();
                int c2 = -1, Lc = 0;
                const bool st = Q.sticky >= 1 && Q.sticky < l && Q.NG.ctx[Q.sticky - 1] == Q.out.back();
                if (st) c2 = Q.sticky; else { Lc = Q.NGC.find(&c2); if (Lc == 0) c2 = -1; }
                if (c2 >= 0) {
                    const int avail = std::min(Kmax, l - c2);
                    if (avail >= A.spec.copy_min) {
                        const int b = st ? 8 : (Lc >= 32 ? 9 : bucket(Lc));
                        const int kk = choose(b, avail);
                        if (kk > 0) { k = kk; cont = c2; L = Lc ? Lc : 1; Q.bk = b; used = true; Q.copy_steps++; Q.sticky = c2; for (int i = 0; i < k; ++i) toks[1 + i] = Q.NG.ctx[cont + i]; }
                    }
                }
            }
            if (!used) {
                L = Q.NG.find(&cont);
                if (L > 0) {
                    const int avail = std::min(Kmax, (int) Q.NG.ctx.size() - cont);
                    Q.bk = bucket(L);
                    k = choose(Q.bk, avail);
                    for (int i = 0; i < k; ++i) toks[1 + i] = Q.NG.ctx[cont + i];
                }
            }
            if (pos + 1 + k >= X.maxctx) k = std::max(0, X.maxctx - pos - 1);
        }
        { const int rem = R.max_new - (int) Q.out.size(); if (k > rem - 1) k = std::max(0, rem - 1); }   // never feed tokens past max_new: the slot history then equals prompt + output - 1
        const int B = k + 1;
        toks[0] = Q.out.back();
        for (int i = 0; i < B; ++i) { dr.h_tok[i] = toks[i]; dr.h_pos[i] = pos + i; dr.h_slot[i] = sl; dr.h_dr[i] = (i + 1 < B) ? toks[i + 1] : -1; dr.set_prm(i, R.temp, R.seed, Q.step); }
        *dr.h_rb = S.rb;
        Q.step++;
        dr.run(B, samp, false);
        int m = 0; std::vector<int> emit;
        if (samp) { while (m < k && dr.h_acc[m]) { emit.push_back(toks[1 + m]); ++m; } emit.push_back(dr.h_stok[m]); }
        else { while (m < k && dr.h_arg[m] == toks[1 + m]) { emit.push_back(toks[1 + m]); ++m; } emit.push_back(dr.h_arg[m]); }
        if (k > 0 && L > 0) {
            const int b = Q.bk;
            for (int i = 0; i < m; ++i) { Q.tried[b][i] = Q.tried[b][i] * 0.995 + 1; Q.accd[b][i] = Q.accd[b][i] * 0.995 + 1; }
            if (m < k) { Q.tried[b][m] = Q.tried[b][m] * 0.995 + 1; Q.accd[b][m] *= 0.995; }
        }
        S.rb = (m == k) ? -1 : m + 1;
        Q.steps++; Q.drafted += k; Q.accepted += m; if (k == 0) Q.nodraft++;
        commit(sl, toks, m + 1, emit);
    }

    void gen_multi(const std::vector<int> & gl) {
        flush_rb(); ensure_resident(0);
        const int B = (int) gl.size();
        bool samp = false;
        for (int i = 0; i < B; ++i) {
            DSlot & S = slots[gl[i]]; DReq & R = *S.req; DSeq & Q = *S.sq;
            dr.h_tok[i] = Q.out.back(); dr.h_pos[i] = (int) S.hist.size(); dr.h_slot[i] = gl[i]; dr.h_dr[i] = -1; dr.set_prm(i, R.temp, R.seed, Q.step++);
            if (R.temp > 0) samp = true;
        }
        dr.run(B, samp, true);
        for (int i = 0; i < B; ++i) {
            DSlot & S = slots[gl[i]]; DReq & R = *S.req;
            const int tok = R.temp > 0 ? dr.h_stok[i] : dr.h_arg[i];
            int fed = dr.h_tok[i];
            S.sq->multi_steps++;
            commit(gl[i], &fed, 1, {tok});
        }
    }

    // ------------------------------------------------------------------ startup: capture graphs, measure the cost table
    void warmup() {
        std::printf("[daemon] capturing graphs and measuring step costs ...\n"); std::fflush(stdout);
        const double t = spec_now();
        const int Kmax = std::min(A.spec.k, MB - 1);
        const int nb = std::max(1, std::min(Kmax, MB - 1)) + 1;
        for (int B = 1; B <= nb; ++B) {
            for (int i = 0; i < B; ++i) { dr.h_tok[i] = 1000 + i; dr.h_pos[i] = 2000 + i; dr.h_slot[i] = 0; dr.h_dr[i] = -1; dr.set_prm(i, 1.0f, 1, 0); }
            *dr.h_rb = -1;
            dr.run(B, false, false); dr.run(B, true, false);
            const double t1 = spec_now(); const int Rr = 12;
            for (int r = 0; r < Rr; ++r) dr.run(B, false, false);
            cost[B] = (spec_now() - t1) / Rr * 1000.0;
        }
        for (int B = 2; B <= std::min(4, A.nslots); ++B) {
            for (int i = 0; i < B; ++i) { dr.h_tok[i] = 1000 + i; dr.h_pos[i] = 2000 + i; dr.h_slot[i] = i; dr.h_dr[i] = -1; dr.set_prm(i, 1.0f, 1, 0); }
            dr.run(B, false, true); dr.run(B, true, true);
        }
        std::printf("[daemon] step cost (ms, launch+sync) by B:");
        for (int B = 1; B <= nb; ++B) std::printf(" %d:%.2f", B, cost[B]);
        std::printf("\n[daemon] warmup %.1f s\n", spec_now() - t);
        reset_state(g); reset_extra_slots(g, X);
        CHECK(cudaStreamSynchronize(s));
        for (int sl = 0; sl < 4; ++sl) { slots[sl].hist.clear(); slots[sl].rb = -1; }
        resident = 0;
    }

    // ------------------------------------------------------------------ main scheduler loop
    void loop() {
        for (;;) {
            // admit
            {
                std::unique_lock<std::mutex> lk(qm);
                bool busy = false;
                for (int sl = 0; sl < A.nslots; ++sl) if (slots[sl].phase) busy = true;
                if (!busy && pending.empty()) qcv.wait_for(lk, std::chrono::milliseconds(100));
                while (!pending.empty()) {
                    std::shared_ptr<DReq> r = pending.front();
                    if (r->cancel || r->conn->dead) {
                        pending.pop_front(); lk.unlock();
                        send_done(r, "cancelled", (int) r->prompt.size(), 0, 0, ""); lk.lock(); continue;
                    }
                    bool idle = false;
                    for (int sl = 0; sl < A.nslots; ++sl) if (slots[sl].phase == 0) idle = true;
                    if (!idle) break;
                    pending.pop_front(); lk.unlock();
                    admit(r);
                    lk.lock();
                }
            }
            for (int sl = 0; sl < A.nslots; ++sl) {
                DSlot & S = slots[sl];
                if (S.phase && (S.req->cancel || S.req->conn->dead)) finish(sl, "cancelled");
            }
            std::vector<int> pf, gl; { int na = 0, ns = 0; for (int sl = 0; sl < 4; ++sl) { na += slots[sl].phase != 0; ns += (int) slots[sl].snaps.size(); } st_active = na; st_snaps = ns; }
            for (int sl = 0; sl < A.nslots; ++sl) { if (slots[sl].phase == 1) pf.push_back(sl); else if (slots[sl].phase == 2) gl.push_back(sl); }
            if (!pf.empty()) {
                int pick = pf[rr % pf.size()]; ++rr;
                prefill_chunk(pick, !gl.empty());
                gl.clear();
                for (int sl = 0; sl < A.nslots; ++sl) if (slots[sl].phase == 2) gl.push_back(sl);
            }
            if (gl.size() == 1) gen_single(gl[0]);
            else if (gl.size() >= 2) gen_multi(gl);
            { int na = 0, ns = 0; for (int sl = 0; sl < 4; ++sl) { na += slots[sl].phase != 0; ns += (int) slots[sl].snaps.size(); } st_active = na; st_snaps = ns; }
        }
    }

    // ------------------------------------------------------------------ network side
    std::string info_json(bool stats) {
        size_t fr = 0, tot = 0; cudaMemGetInfo(&fr, &tot);
        const int act = st_active, snaps = st_snaps;
        char b[1024];
        std::snprintf(b, sizeof b, "{\"op\":\"%s\",\"build\":\"mikasa-daemon\",\"n_ctx\":%d,\"nslots\":%d,\"spec_k\":%d,\"spec_copy\":%d,\"snaps_max\":%d,\"snaps_alloc\":%d,\"snaps_live\":%d,"
                      "\"active\":%d,\"requests\":%ld,\"done\":%ld,\"prompt_tokens\":%ld,\"cached_tokens\":%ld,\"gen_tokens\":%ld,\"snap_taken\":%ld,\"snap_restored\":%ld,\"resident_hits\":%ld,"
                      "\"gpu_used_mb\":%.0f,\"uptime_s\":%.1f}",
                      stats ? "stats" : "info", X.maxctx, A.nslots, A.spec.k, A.spec.copy_min, g_dargs.snaps, nsnap_alloc, snaps, act, (long) n_req, (long) n_done, (long) n_prompt, (long) n_cached,
                      (long) n_gen, snap_taken, snap_restored, resident_hits, (tot - fr) / 1048576.0, spec_now() - t0);
        return b;
    }
    void reader(std::shared_ptr<DConn> c) {
        std::string buf; char tmp[65536];
        std::vector<std::shared_ptr<DReq>> mine;
        for (;;) {
            const ssize_t n = ::recv(c->fd, tmp, sizeof tmp, 0);
            if (n < 0 && errno == EINTR) continue;
            if (n <= 0) break;
            buf.append(tmp, (size_t) n);
            size_t nl;
            while ((nl = buf.find('\n')) != std::string::npos) {
                std::string line = buf.substr(0, nl); buf.erase(0, nl + 1);
                if (line.empty()) continue;
                JP p(line); JV j = p.val();
                if (!p.ok || j.t != JV::OBJ) { send_line(*c, "{\"op\":\"error\",\"error\":\"bad json\"}"); continue; }
                const std::string op = j.str("op", "gen");
                if (op == "info") { send_line(*c, info_json(false)); continue; }
                if (op == "stats") { send_line(*c, info_json(true)); continue; }
                if (op == "cancel") {
                    std::lock_guard<std::mutex> lk(qm);
                    auto it = all.find({c.get(), j.str("id")});
                    if (it != all.end()) it->second->cancel = true;
                    qcv.notify_all(); continue;
                }
                if (op != "gen") { send_line(*c, "{\"op\":\"error\",\"error\":\"unknown op\"}"); continue; }
                auto r = std::make_shared<DReq>();
                r->conn = c; r->id = j.str("id"); r->t_arrive = spec_now(); n_req++;
                std::string err;
                const JV * pi = j.get("prompt_ids");
                if (!pi || pi->t != JV::ARR || pi->nums.empty()) err = "prompt_ids missing or empty";
                else { r->prompt.reserve(pi->nums.size()); for (double d : pi->nums) { if (d < 0 || d >= NVOCAB) { err = "token id out of range"; break; } r->prompt.push_back((int) d); } }
                r->max_new = (int) j.num("max_new", 256); if (r->max_new < 1) r->max_new = 1;
                r->temp = (float) j.num("temperature", 0.0);
                if (r->temp < 0) err = "temperature < 0";
                if (j.has("seed")) r->seed = (unsigned) (long long) j.num("seed", 1); else r->seed = (unsigned) (spec_now() * 1e6) ^ (unsigned) n_req;
                if (j.num("top_k", 0) > 0) err = "unsupported: top_k > 0";
                if (j.has("top_p") && j.num("top_p", 1.0) < 1.0) err = "unsupported: top_p < 1";
                const JV * st = j.get("stop_ids");
                if (st && st->t == JV::ARR) for (double d : st->nums) r->stop.push_back((int) d);
                r->spec_k = (int) j.num("spec_k", -1); r->no_cache = j.bl("no_cache", false);
                if (err.empty() && (int) r->prompt.size() + 2 > X.maxctx) err = "prompt too long for the context (" + std::to_string(r->prompt.size()) + " + 2 > " + std::to_string(X.maxctx) + ")";
                if (!err.empty()) { send_done(r, "error", (int) r->prompt.size(), 0, 0, "", err); continue; }
                { std::lock_guard<std::mutex> lk(qm); all[{c.get(), r->id}] = r; pending.push_back(r); mine.push_back(r); }
                qcv.notify_all();
            }
        }
        c->dead = true;
        { std::lock_guard<std::mutex> lk(qm); for (auto & kv : all) if (kv.first.first == c.get()) kv.second->cancel = true; }
        qcv.notify_all();
        ::close(c->fd);
    }
    void accept_loop(int lfd) {
        for (;;) {
            const int fd = ::accept(lfd, nullptr, nullptr);
            if (fd < 0) { if (errno == EINTR) continue; std::perror("accept"); return; }
            timeval tv; tv.tv_sec = 5; tv.tv_usec = 0; setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof tv);
            auto c = std::make_shared<DConn>(); c->fd = fd;
            std::thread(&Daemon::reader, this, c).detach();
        }
    }
};

static void daemon_sigterm(int) { if (!g_sock_path.empty()) unlink(g_sock_path.c_str()); _exit(0); }

static int run_serve(Engine & g, const LongArgs & A, cudaStream_t s) {
    print_mem("weights loaded");
    BatchState X; KVOpts ko;
    ko.kv16 = A.kv16; ko.maxctx = A.maxctx; ko.nslots = A.nslots; ko.ring = std::max(A.ring, NSWA + A.pf_fast); ko.alias0 = false; ko.ref = false;
    init_batch(g, X, ko);
    print_mem("after init_batch (KV + buffers)");
    PFState PF; init_pf(PF, A.pf_fast); g_moe_tb = 64;
    print_mem("after init_pf");
    Daemon D(g, X, PF, s, A);
    D.warmup();
    print_mem("after warmup");
    { // pre-allocate nothing else: snapshots are allocated lazily (each ~174 MB)
        std::printf("[daemon] snapshot = %.1f MB ring + %.1f MB conv; max %d snapshots = %.2f GB\n", (double) NSWA_APPS * 2 * SNAP_WIN * NKV * HD * 4 / 1e6,
                    (double) NAPP * (6 * NKV * HD + 6 * E) * 4 / 1e6, g_dargs.snaps,
                    (double) g_dargs.snaps * ((double) NSWA_APPS * 2 * SNAP_WIN * NKV * HD * 4 + (double) NAPP * (6 * NKV * HD + 6 * E) * 4) / 1e9);
    }
    std::signal(SIGPIPE, SIG_IGN); std::signal(SIGTERM, daemon_sigterm); std::signal(SIGINT, daemon_sigterm);
    int lfd = ::socket(AF_UNIX, SOCK_STREAM, 0);
    sockaddr_un ad; std::memset(&ad, 0, sizeof ad); ad.sun_family = AF_UNIX;
    std::strncpy(ad.sun_path, g_dargs.sock.c_str(), sizeof(ad.sun_path) - 1);
    unlink(g_dargs.sock.c_str()); g_sock_path = g_dargs.sock;
    if (::bind(lfd, (sockaddr *) &ad, sizeof ad) != 0 || ::listen(lfd, 16) != 0) { std::perror("bind/listen unix socket"); return 1; }
    chmod(g_dargs.sock.c_str(), 0600);
    std::thread(&Daemon::accept_loop, &D, lfd).detach();
    if (g_dargs.tcp > 0) {
        int tfd = ::socket(AF_INET, SOCK_STREAM, 0); int one = 1; setsockopt(tfd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
        sockaddr_in a4; std::memset(&a4, 0, sizeof a4); a4.sin_family = AF_INET; a4.sin_port = htons(g_dargs.tcp); a4.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        if (::bind(tfd, (sockaddr *) &a4, sizeof a4) != 0 || ::listen(tfd, 16) != 0) { std::perror("bind/listen tcp"); return 1; }
        std::thread(&Daemon::accept_loop, &D, tfd).detach();
    }
    std::printf("[daemon] ready: socket %s%s, n_ctx %d, slots %d, spec K=%d copy=%d\n", g_dargs.sock.c_str(), g_dargs.tcp ? " + tcp" : "", X.maxctx, A.nslots, A.spec.k, A.spec.copy_min);
    std::fflush(stdout);
    D.loop();
    return 0;
}
