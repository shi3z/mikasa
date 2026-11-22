// Inkling 2 Mini, full 48-application graph on the GPU (sm_86). Sequential decode, one token per step.
// usage: inkling_model <model.gguf> <outdir> [--tokens a,b,...] [--reps N] [--graph]
// Writes l_out-<il>.bin ([T][E]) and logits.bin ([T][200058]) to outdir.
#include "gguf.hpp"
#include "kernels_common.cuh"

#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <sstream>
#include <string>
#include <vector>
#include <sys/stat.h>

#define CHECK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    std::fprintf(stderr, "CUDA %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); std::exit(1); } } while (0)

constexpr int E = 2304, NH = 24, NKV = 8, HD = 128, NREL = 16, NSWA = 1024, NEXP = 128, NUSED = 8, FFN = 1152;
constexpr int NAPP = 48, NPHYS = 24, CORE_START = 6, CORE_END = 18, NPASS = 3, BLOCK = 6;
constexpr int MAXT = 1024;           // positions per KV cache (decode limit of the smem score buffer)
constexpr int NVOCAB = 200058, NVOCAB_PAD = 201024, MAXSTATES = 24;
constexpr float QK_CLIP = 22.627416610717773f, REL_CLIP = 8.0f, RMS_EPS = 1e-6f;
constexpr float SITU_BETA = 4.0f, SITU_LB = 25.0f, W_SCALE = 8.0f, SOFTCAP = 30.0f;
constexpr float LOG_ALPHA = 0.10000000149011612f, LOG_FLOOR = 8192.0f;

// ------------------------------------------------------------ host utils ----

struct Model {
    gguf::File f;
    const gguf::Tensor * find(const std::string & name) {
        for (const auto & t : f.tensors) if (t.name == name) return &t;
        std::fprintf(stderr, "missing tensor %s\n", name.c_str());
        std::exit(1);
    }
    bool has(const std::string & name) {
        for (const auto & t : f.tensors) if (t.name == name) return true;
        return false;
    }
    const uint8_t * ptr(const gguf::Tensor * t) { return f.base + f.data_start + t->offset; }
};

static void repack_q8_soa(const uint8_t * src, int K, int N, std::vector<int8_t> & qs, std::vector<uint16_t> & sc) {
    const int nb = K / 32;
    qs.resize((size_t) N * K);
    sc.resize((size_t) N * nb);
    for (int n = 0; n < N; ++n)
        for (int b = 0; b < nb; ++b) {
            const uint8_t * blk = src + ((size_t) n * nb + b) * 34;
            std::memcpy(&sc[(size_t) n * nb + b], blk, 2);
            std::memcpy(&qs[(size_t) n * K + b * 32], blk + 2, 32);
        }
}

static void repack_nv_soa(const uint8_t * src, int K, int N, uint8_t * qs, uint8_t * sc) {
    const int nb = K / 64;
    for (int n = 0; n < N; ++n)
        for (int b = 0; b < nb; ++b) {
            const uint8_t * blk = src + ((size_t) n * nb + b) * 36;
            std::memcpy(sc + (size_t) n * (K / 16) + b * 4, blk, 4);
            std::memcpy(qs + (size_t) n * (K / 2) + b * 32, blk + 4, 32);
        }
}

template <typename T>
static T * upload(const void * host, size_t count) {
    T * d = nullptr;
    CHECK(cudaMalloc(&d, count * sizeof(T)));
    CHECK(cudaMemcpy(d, host, count * sizeof(T), cudaMemcpyHostToDevice));
    return d;
}

static float * f32_or_null(Model & m, const std::string & name, size_t count) {
    if (!m.has(name)) return nullptr;
    const gguf::Tensor * t = m.find(name);
    return upload<float>(m.ptr(t), count);
}

static float * f32(Model & m, const std::string & name, size_t count) {
    float * p = f32_or_null(m, name, count);
    if (!p) { std::fprintf(stderr, "missing %s\n", name.c_str()); std::exit(1); }
    return p;
}

struct Q8W { int8_t * qs = nullptr; __half * sc = nullptr; int K = 0, N = 0; };

static Q8W load_q8(Model & m, const std::string & name, int K, int N) {
    const gguf::Tensor * t = m.find(name);
    std::vector<int8_t> qs;
    std::vector<uint16_t> sc;
    repack_q8_soa(m.ptr(t), K, N, qs, sc);
    Q8W w;
    w.qs = upload<int8_t>(qs.data(), qs.size());
    w.sc = upload<__half>(sc.data(), sc.size());
    w.K = K;
    w.N = N;
    return w;
}

static uint16_t * raw_bf16(Model & m, const std::string & name, size_t count) {
    const gguf::Tensor * t = m.find(name);
    return upload<uint16_t>(m.ptr(t), count);
}

// --------------------------------------------------------- physical layers ----

struct Phys {
    Q8W wq, wk, wv, wrel, wgate, wo;
    float *attn_norm = nullptr, *q_inp = nullptr, *k_inp = nullptr, *v_inp = nullptr;
    float *attn_res_query = nullptr;   // [nq][E], nq = 3 for full layers, 1 for SWA layers
    float *q_clip, *rel_clip, *rel_bias, *kconv, *vconv, *post_conv, *ffn_post_conv;
    uint16_t *kconv_proj = nullptr, *vconv_proj = nullptr, *sink_w;
    float *ffn_norm, *ffn_res_query, *router_w, *exp_bias, *gu_scale = nullptr, *dn_scale = nullptr;
    uint8_t *gu_qs = nullptr, *gu_sc = nullptr, *dn_qs = nullptr, *dn_sc = nullptr;   // NVFP4 experts
    int8_t *gu_q8 = nullptr, *dn_q8 = nullptr;                                         // Q8_0 experts
    __half *gu_q8s = nullptr, *dn_q8s = nullptr;
    bool exp_q8 = false;
    float *loop_down[4] = {nullptr, nullptr, nullptr, nullptr};
    float *loop_gain[4] = {nullptr, nullptr, nullptr, nullptr};
    float *loop_forget[4] = {nullptr, nullptr, nullptr, nullptr};
};

// application index -> physical layer (hparams.loop_layer)
static int loop_layer(int il) {
    const int n_core = CORE_END - CORE_START;
    if (il < CORE_START) return il;
    if (il < CORE_START + NPASS * n_core) return CORE_START + (il - CORE_START) % n_core;
    return il - (NPASS - 1) * n_core;
}

static bool app_is_swa(int il) { return (il % 6) != 5; }   // GGUF sliding_window_pattern

static void load_phys(Model & m, int ip, Phys & P) {
    const std::string p = "blk." + std::to_string(ip) + ".";
    P.wq = load_q8(m, p + "attn_q.weight", E, NH * HD);
    P.wk = load_q8(m, p + "attn_k.weight", E, NKV * HD);
    P.wv = load_q8(m, p + "attn_v.weight", E, NKV * HD);
    P.wrel = load_q8(m, p + "attn_rel.weight", E, NH * NREL);
    P.wgate = load_q8(m, p + "attn_gate.weight", E, NH * HD);
    P.wo = load_q8(m, p + "attn_output.weight", NH * HD, E);
    P.attn_norm = f32_or_null(m, p + "attn_norm.weight", E);
    P.q_inp = f32_or_null(m, p + "attn_q_inp_norm.weight", E);
    P.k_inp = f32_or_null(m, p + "attn_k_inp_norm.weight", E);
    P.v_inp = f32_or_null(m, p + "attn_v_inp_norm.weight", E);
    if (ip > 0) {
        const bool full = (ip % 6) == 5;
        P.attn_res_query = f32(m, p + "attn_res_query.weight", (full ? 3 : 1) * E);
    }
    P.q_clip = f32(m, p + "attn_q_clip.weight", NH * HD);
    P.rel_clip = f32(m, p + "attn_rel_clip.weight", NH * NREL);
    P.rel_bias = f32(m, p + "attn_rel_bias.weight", NSWA * NREL);
    P.kconv = f32(m, p + "attn_k_conv.weight", NKV * HD * 4);
    P.vconv = f32(m, p + "attn_v_conv.weight", NKV * HD * 4);
    P.post_conv = f32(m, p + "attn_post_conv.weight", E * 4);
    P.ffn_post_conv = f32(m, p + "ffn_post_conv.weight", E * 4);
    if ((ip % 6) == 5) {  // full attention: BF16 dynamic tap projections [E][32]
        P.kconv_proj = raw_bf16(m, p + "attn_k_conv_proj.weight", (size_t) E * 32);
        P.vconv_proj = raw_bf16(m, p + "attn_v_conv_proj.weight", (size_t) E * 32);
    }
    P.sink_w = raw_bf16(m, p + "attn_sink_proj.weight", (size_t) E * NH);
    P.ffn_norm = f32(m, p + "ffn_norm.weight", E);
    P.ffn_res_query = f32(m, p + "ffn_res_query.weight", E);
    P.router_w = f32(m, p + "ffn_gate_inp.weight", NEXP * E);
    P.exp_bias = f32(m, p + "exp_probs_b.bias", NEXP);
    P.gu_scale = f32_or_null(m, p + "ffn_gate_up_exps.scale", NEXP);
    P.dn_scale = f32_or_null(m, p + "ffn_down_exps.scale", NEXP);
    {
        // expert tensors are NVFP4 (type 40) or Q8_0 (type 8), depending on the layer
        const gguf::Tensor * tg = m.find(p + "ffn_gate_up_exps.weight");  // [E][2304][128]
        const gguf::Tensor * td = m.find(p + "ffn_down_exps.weight");     // [1152][2304][128]
        P.exp_q8 = (tg->type == 8);
        if (P.exp_q8) {
            std::vector<int8_t> qs((size_t) NEXP * 2304 * E);
            std::vector<uint16_t> sc((size_t) NEXP * 2304 * (E / 32));
            std::vector<int8_t> qd((size_t) NEXP * 2304 * FFN);
            std::vector<uint16_t> sd((size_t) NEXP * 2304 * (FFN / 32));
            const size_t pg = (size_t) 2304 * (E / 32) * 34, pd = (size_t) 2304 * (FFN / 32) * 34;
            for (int e = 0; e < NEXP; ++e) {
                std::vector<int8_t> q1; std::vector<uint16_t> s1;
                repack_q8_soa(m.ptr(tg) + e * pg, E, 2304, q1, s1);
                std::copy(q1.begin(), q1.end(), qs.begin() + (size_t) e * 2304 * E);
                std::copy(s1.begin(), s1.end(), sc.begin() + (size_t) e * 2304 * (E / 32));
                repack_q8_soa(m.ptr(td) + e * pd, FFN, 2304, q1, s1);
                std::copy(q1.begin(), q1.end(), qd.begin() + (size_t) e * 2304 * FFN);
                std::copy(s1.begin(), s1.end(), sd.begin() + (size_t) e * 2304 * (FFN / 32));
            }
            P.gu_q8 = upload<int8_t>(qs.data(), qs.size());
            P.gu_q8s = upload<__half>(sc.data(), sc.size());
            P.dn_q8 = upload<int8_t>(qd.data(), qd.size());
            P.dn_q8s = upload<__half>(sd.data(), sd.size());
        } else {
            const size_t per = (size_t) 2304 * (E / 64) * 36;
            std::vector<uint8_t> qs((size_t) NEXP * 2304 * (E / 2)), sc((size_t) NEXP * 2304 * (E / 16));
            for (int e = 0; e < NEXP; ++e)
                repack_nv_soa(m.ptr(tg) + e * per, E, 2304, &qs[(size_t) e * 2304 * (E / 2)], &sc[(size_t) e * 2304 * (E / 16)]);
            P.gu_qs = upload<uint8_t>(qs.data(), qs.size());
            P.gu_sc = upload<uint8_t>(sc.data(), sc.size());
            const size_t perd = (size_t) 2304 * (FFN / 64) * 36;
            std::vector<uint8_t> qd((size_t) NEXP * 2304 * (FFN / 2)), sd((size_t) NEXP * 2304 * (FFN / 16));
            for (int e = 0; e < NEXP; ++e)
                repack_nv_soa(m.ptr(td) + e * perd, FFN, 2304, &qd[(size_t) e * 2304 * (FFN / 2)], &sd[(size_t) e * 2304 * (FFN / 16)]);
            P.dn_qs = upload<uint8_t>(qd.data(), qd.size());
            P.dn_sc = upload<uint8_t>(sd.data(), sd.size());
        }
    }
    if (ip >= CORE_START && ip < CORE_END) {
        const char * names[4] = {"attn", "k", "v", "mlp"};
        for (int k = 0; k < 4; ++k) {
            P.loop_down[k] = f32(m, p + "loop_" + names[k] + "_down.weight", 4608 * 64);
            P.loop_gain[k] = f32(m, p + "loop_" + names[k] + "_gain.weight", 64 * E);
            P.loop_forget[k] = f32(m, p + "loop_" + names[k] + "_forget.weight", 64 * E);
        }
    }
}

// ------------------------------------------------------------- model state ----

struct Engine {
    Phys phys[NPHYS];
    Q8W tok_embd, head;
    float *tok_norm, *output_norm, *output_res_query, *loop_res_query, *loop_kv_res_query;
    // per application state
    float *kcache[NAPP], *vcache[NAPP], *kstate[NAPP], *vstate[NAPP], *poststate[NAPP], *ffnstate[NAPP];
    // scratch
    float *emb, *cur, *inp_sa, *x1, *x2, *tmp2, *hcat, *xw, *Qv, *K0, *V0, *R0, *G, *sinks, *Kconv, *Vconv, *Qc, *Rc,
        *dyn, *O, *gated, *attn_o, *attn_post, *ffn_inp, *mixed, *xf, *logits_router, *gu, *hbuf, *dn, *moe, *ffn_post,
        *lout, *head_in, *ctrl_g, *ctrl_f, *ctrl_out, *msgs, *bank_buf, *prelude_buf, *blocks_buf, *states_buf,
        *nst_buf, *scores_buf, *mix_out_buf, *logits, *inv_buf;
    int *ids;
    float *wts;
    // block bookkeeping (device buffers, host counters)
    int nb = 0, prelude_nb = 0, bank_nb = 0;
    float *inbuf, *mixed3, *mixA, *mixK, *r64, *dynk, *dynv, *ones, *head_norm, *xnorm;
    float *bank_last, *prelude_cur, *xs3;
    // msgs slots are ordered [attn-q(0), attn-q(3), k(1), v(2)] in memory so each 2-query mix writes contiguously:
    // msgs_ptr(0) = A0, msgs_ptr(3) = A1 (mixA rows), msgs_ptr(1) = K0, msgs_ptr(2) = K1 (mixK rows).
    float * msgs_ptr(int j) { static const int slot[4] = {0, 2, 3, 1}; return msgs + (size_t) slot[j] * E; }
    int *pos_dev = nullptr, *tok_dev = nullptr;
};

static void alloc_zero_dev(float ** p, size_t n);

static void init_scratch(Engine & g) {
    alloc_zero_dev(&g.inbuf, 3 * E);
    alloc_zero_dev(&g.mixed3, 3 * E);
    alloc_zero_dev(&g.mixA, 2 * E);
    alloc_zero_dev(&g.mixK, 2 * E);
    alloc_zero_dev(&g.r64, 3 * 64);
    alloc_zero_dev(&g.dynk, 32);
    alloc_zero_dev(&g.dynv, 32);
    alloc_zero_dev(&g.head_norm, E);
    alloc_zero_dev(&g.xnorm, E);
    CHECK(cudaMalloc(&g.pos_dev, sizeof(int)));
    CHECK(cudaMalloc(&g.tok_dev, sizeof(int)));
    std::vector<float> ones(NEXP, 1.0f);
    g.ones = upload<float>(ones.data(), NEXP);
}

static void alloc_zero_dev(float ** p, size_t n);
static void alloc_zero(float ** p, size_t n) {
    CHECK(cudaMalloc(p, n * sizeof(float)));
    CHECK(cudaMemset(*p, 0, n * sizeof(float)));
}

static void alloc_zero_dev(float ** p, size_t n) {
    CHECK(cudaMalloc(p, n * sizeof(float)));
    CHECK(cudaMemset(*p, 0, n * sizeof(float)));
}

static Engine * build_engine(Model & m) {
    Engine * en = new Engine();
    Engine & g = *en;
    {
        const gguf::Tensor * t = m.find("token_embd.weight");
        const int V = (int) t->ne[1];
        std::vector<int8_t> qs;
        std::vector<uint16_t> sc;
        repack_q8_soa(m.ptr(t), E, V, qs, sc);
        g.tok_embd.qs = upload<int8_t>(qs.data(), qs.size());
        g.tok_embd.sc = upload<__half>(sc.data(), sc.size());
        g.tok_embd.K = E;
        g.tok_embd.N = V;
    }
    g.tok_norm = f32(m, "token_embd_norm.weight", E);
    g.output_norm = f32(m, "output_norm.weight", E);
    g.output_res_query = f32(m, "output_res_query.weight", E);
    g.loop_res_query = f32(m, "loop_res_query.weight", E * 24);
    g.loop_kv_res_query = f32(m, "loop_kv_res_query.weight", E * 24);
    {
        const gguf::Tensor * t = m.find("output.weight");
        std::vector<int8_t> qs;
        std::vector<uint16_t> sc;
        repack_q8_soa(m.ptr(t), E, NVOCAB_PAD, qs, sc);
        g.head.qs = upload<int8_t>(qs.data(), qs.size());
        g.head.sc = upload<__half>(sc.data(), sc.size());
        g.head.K = E;
        g.head.N = NVOCAB_PAD;
    }
    for (int ip = 0; ip < NPHYS; ++ip) load_phys(m, ip, g.phys[ip]);

    for (int il = 0; il < NAPP; ++il) {
        alloc_zero(&g.kcache[il], (size_t) MAXT * NKV * HD);
        alloc_zero(&g.vcache[il], (size_t) MAXT * NKV * HD);
        alloc_zero(&g.kstate[il], 3 * NKV * HD);
        alloc_zero(&g.vstate[il], 3 * NKV * HD);
        alloc_zero(&g.poststate[il], 3 * E);
        alloc_zero(&g.ffnstate[il], 3 * E);
    }
    alloc_zero(&g.emb, E); alloc_zero(&g.cur, E); alloc_zero(&g.inp_sa, E);
    alloc_zero(&g.x1, E); alloc_zero(&g.x2, E); alloc_zero(&g.tmp2, 2 * E); alloc_zero(&g.hcat, 2 * E);
    alloc_zero(&g.xs3, 3 * E);
    g.xw = g.xs3; g.x1 = g.xs3 + E; g.x2 = g.xs3 + 2 * E;
    alloc_zero(&g.Qv, NH * HD); alloc_zero(&g.K0, NKV * HD); alloc_zero(&g.V0, NKV * HD);
    alloc_zero(&g.R0, NH * NREL); alloc_zero(&g.G, NH * HD); alloc_zero(&g.sinks, NH);
    alloc_zero(&g.Kconv, NKV * HD); alloc_zero(&g.Vconv, NKV * HD); alloc_zero(&g.Qc, NH * HD); alloc_zero(&g.Rc, NH * NREL);
    alloc_zero(&g.dyn, 32); alloc_zero(&g.O, NH * HD); alloc_zero(&g.gated, NH * HD); alloc_zero(&g.attn_o, E);
    alloc_zero(&g.attn_post, E); alloc_zero(&g.ffn_inp, E); alloc_zero(&g.mixed, E); alloc_zero(&g.xf, E);
    alloc_zero(&g.logits_router, NEXP); alloc_zero(&g.gu, NUSED * 2304); alloc_zero(&g.hbuf, NUSED * FFN);
    alloc_zero(&g.dn, NUSED * E); alloc_zero(&g.moe, E); alloc_zero(&g.ffn_post, E); alloc_zero(&g.lout, E);
    alloc_zero(&g.head_in, E); alloc_zero(&g.ctrl_g, E); alloc_zero(&g.ctrl_f, E); alloc_zero(&g.ctrl_out, E);
    alloc_zero(&g.msgs, 4 * E); alloc_zero(&g.bank_buf, MAXSTATES * E); alloc_zero(&g.prelude_buf, MAXSTATES * E);
    alloc_zero(&g.blocks_buf, MAXSTATES * E); alloc_zero(&g.states_buf, (MAXSTATES + 1) * E);
    alloc_zero(&g.nst_buf, (MAXSTATES + 1) * E); alloc_zero(&g.scores_buf, 4 * (MAXSTATES + 1)); alloc_zero(&g.inv_buf, MAXSTATES + 1);
    alloc_zero(&g.mix_out_buf, 3 * E); alloc_zero(&g.logits, NVOCAB_PAD); alloc_zero(&g.bank_last, E);
    alloc_zero(&g.prelude_cur, E);
    CHECK(cudaMalloc(&g.ids, NUSED * sizeof(int)));
    alloc_zero(&g.wts, NUSED);
    init_scratch(g);
    return en;
}

// ----------------------------------------------------------- mix over states ----

// states_buf holds the S states (contiguous [S][E]). out = [nq][E].
// Mix with states given as head rows [0,nhead) plus an optional tail row (no staging copy).
static void mix_states_src(Engine & g, int S, const float * head, int nhead, const float * tail,
                           const float * query, int nq, float * out, cudaStream_t s) {
    k_mix_scores2<<<S, 256, 0, s>>>(head, nhead, tail, query, nq, g.scores_buf, g.inv_buf, S, E, RMS_EPS);
    k_mix_out2<<<dim3((E + 255) / 256, nq), 256, 0, s>>>(head, nhead, tail, g.scores_buf, g.inv_buf, out, S, E);
}

// Mix over the staged states_buf (loop-message path).
static void mix_states(Engine & g, int S, const float * query, int nq, float * out, cudaStream_t s) {
    mix_states_src(g, S, g.states_buf, S, nullptr, query, nq, out, s);
}

// Mix over [blocks..., cur] without copying them into states_buf.
static int mix_cur(Engine & g, const float * query, int nq, float * out, cudaStream_t s) {
    const int S = g.nb + 1;
    mix_states_src(g, S, g.blocks_buf, g.nb, g.cur, query, nq, out, s);
    return S;
}

// states_buf = blocks[0..nb-1] followed by cur. Returns S.
static int stage_states(Engine & g, const float * cur, cudaStream_t s) {
    const int S = g.nb + 1;
    if (g.nb) CHECK(cudaMemcpyAsync(g.states_buf, g.blocks_buf, (size_t) g.nb * E * sizeof(float), cudaMemcpyDeviceToDevice, s));
    CHECK(cudaMemcpyAsync(g.states_buf + (size_t) g.nb * E, cur, E * sizeof(float), cudaMemcpyDeviceToDevice, s));
    return S;
}

// ------------------------------------------------------------- one token ----

// Host-side collection of intermediates (validation runs only).
struct Dumper {
    bool on = false;
    std::map<std::string, std::vector<float>> data;
    void grab(const std::string & name, const float * dev, size_t n, cudaStream_t s) {
        if (!on) return;
        std::vector<float> h(n);
        CHECK(cudaMemcpyAsync(h.data(), dev, n * sizeof(float), cudaMemcpyDeviceToHost, s));
        CHECK(cudaStreamSynchronize(s));
        auto & v = data[name];
        v.insert(v.end(), h.begin(), h.end());
    }
};

// control(cur, msg, module) into out: h = [rms(cur); rms(msg)], r = D h (64), g = G r, f = F r.
static void control(Engine & g, const Phys & P, const float * cur, const float * msg, int module, float * out, cudaStream_t s) {
    k_rms2_rows9<<<2, 256, 0, s>>>(cur, msg, g.hcat, E, RMS_EPS);   // h = [rms(cur); rms(msg)]
    gemv_f32_row8<<<64, 256, 0, s>>>(P.loop_down[module], g.hcat, g.r64, 2 * E);
    k_ctrl_gf_warp<<<(E * 32 + 255) / 256, 256, 0, s>>>(cur, msg, P.loop_gain[module], P.loop_forget[module], g.r64, out, E);
}

// Engine forward for one token at position pos. Applications il = 0..47 in the order of inkling2.cpp.
static void forward_token(Engine & g, Dumper & D, cudaStream_t s) {
    k_embed_q8_dev<<<(E + 255) / 256, 256, 0, s>>>(g.tok_embd.qs, g.tok_embd.sc, g.tok_dev, E, g.emb);
    k_rms_rows<<<1, 256, 0, s>>>(g.emb, g.tok_norm, g.cur, E, RMS_EPS);
    g.nb = 0;
    g.prelude_nb = 0;

    for (int il = 0; il < NAPP; ++il) {
        const int ip = loop_layer(il);
        const Phys & P = g.phys[ip];
        const bool swa = app_is_swa(il);
        const bool in_core_range = il >= CORE_START && il < CORE_START + NPASS * (CORE_END - CORE_START);
        const int pass = in_core_range ? (il - CORE_START) / (CORE_END - CORE_START) : 0;

        if (il == CORE_START) {                       // remember the prelude state
            g.prelude_nb = g.nb;
            CHECK(cudaMemcpyAsync(g.prelude_cur, g.cur, E * sizeof(float), cudaMemcpyDeviceToDevice, s));
        }
        bool have_msgs = false;
        if (in_core_range && pass > 0 && ip == CORE_START) {   // start of pass p>0: bank from previous pass
            g.bank_nb = g.nb - (g.prelude_nb + 1);
            if (g.bank_nb > 0)
                CHECK(cudaMemcpyAsync(g.bank_buf, g.blocks_buf + (size_t) (g.prelude_nb + 1) * E,
                                      (size_t) g.bank_nb * E * sizeof(float), cudaMemcpyDeviceToDevice, s));
            CHECK(cudaMemcpyAsync(g.bank_last, g.cur, E * sizeof(float), cudaMemcpyDeviceToDevice, s));
            g.nb = g.prelude_nb;
            CHECK(cudaMemcpyAsync(g.cur, g.prelude_cur, E * sizeof(float), cudaMemcpyDeviceToDevice, s));
        }
        if (in_core_range && pass > 0) {                // loop messages for this core layer
            have_msgs = true;
            const int row = 2 * (ip - CORE_START);
            const int Sb = g.bank_nb + 1;
            if (g.bank_nb > 0)
                CHECK(cudaMemcpyAsync(g.states_buf, g.bank_buf, (size_t) g.bank_nb * E * sizeof(float), cudaMemcpyDeviceToDevice, s));
            CHECK(cudaMemcpyAsync(g.states_buf + (size_t) g.bank_nb * E, g.bank_last, E * sizeof(float), cudaMemcpyDeviceToDevice, s));
            // msgs: 0 = attn-q, 1 = k, 2 = v, 3 = mlp (see inkling2.cpp); written directly by the mixes
            mix_states(g, Sb, g.loop_res_query + (size_t) row * E, 2, g.msgs_ptr(0), s);   // rows: msgs 0 and 3
            mix_states(g, Sb, g.loop_kv_res_query + (size_t) row * E, 2, g.msgs_ptr(1), s); // rows: msgs 1 and 2
        }

        // ----- attention input: attn_res mix (ip > 0), then loop control, then norms
        if (ip > 0) {
            const int nq = swa ? 1 : 3;
            mix_cur(g, P.attn_res_query, nq, g.inbuf, s);   // full: 3 rows; SWA: row 0, replicated below
            if (swa)
                for (int j = 1; j < 3; ++j)
                    CHECK(cudaMemcpyAsync(g.inbuf + (size_t) j * E, g.inbuf, E * sizeof(float), cudaMemcpyDeviceToDevice, s));
        } else {
            for (int j = 0; j < 3; ++j)
                CHECK(cudaMemcpyAsync(g.inbuf + (size_t) j * E, g.cur, E * sizeof(float), cudaMemcpyDeviceToDevice, s));
        }
        if (have_msgs) {
            for (int j = 0; j < 3; ++j) {
                control(g, P, g.inbuf + (size_t) j * E, g.msgs_ptr(j), j, g.inbuf + (size_t) j * E, s);   // in place
            }
        }
        if (swa) {
            k_rms_rows3_9<<<3, 256, 0, s>>>(g.inbuf, P.attn_norm, P.attn_norm, P.attn_norm, g.xs3, E, RMS_EPS);
        } else {
            k_rms_rows3_9<<<3, 256, 0, s>>>(g.inbuf, P.q_inp, P.k_inp, P.v_inp, g.xs3, E, RMS_EPS);
        }

        // ----- projections
        launch_gemv_q8(P.wq.qs, P.wq.sc, g.xw, g.Qv, E, NH * HD, s);
        launch_gemv_q8(P.wk.qs, P.wk.sc, g.x1, g.K0, E, NKV * HD, s);
        launch_gemv_q8(P.wv.qs, P.wv.sc, g.x2, g.V0, E, NKV * HD, s);
        launch_gemv_q8(P.wrel.qs, P.wrel.sc, g.xw, g.R0, E, NH * NREL, s);
        launch_gemv_q8(P.wgate.qs, P.wgate.sc, g.xw, g.G, E, NH * HD, s);
        gemv_bf16_row9<<<NH, 256, 0, s>>>(P.sink_w, g.xw, g.sinks, E, NH);

        // ----- convolutions (state per application); full layers add input-dependent taps
        if (swa) {
            k_conv4_pair<<<dim3((NKV * HD + 255) / 256, 2), 256, 0, s>>>(g.K0, g.kstate[il], P.kconv, g.Kconv,
                                                                       g.V0, g.vstate[il], P.vconv, g.Vconv, NKV * HD);
        } else {
            gemv_bf16_row9<<<32, 256, 0, s>>>(P.kconv_proj, g.x1, g.dynk, E, 32);
            gemv_bf16_row9<<<32, 256, 0, s>>>(P.vconv_proj, g.x2, g.dynv, E, 32);
            k_conv4_dyn<<<(NKV * HD + 255) / 256, 256, 0, s>>>(g.K0, g.kstate[il], P.kconv, g.dynk, g.Kconv, NKV * HD);
            k_conv4_dyn<<<(NKV * HD + 255) / 256, 256, 0, s>>>(g.V0, g.vstate[il], P.vconv, g.dynv, g.Vconv, NKV * HD);
        }

        // ----- clips, norms, cache write at pos
        k_clip_qr<<<2 * NH, 128, 0, s>>>(g.Qv, P.q_clip, g.Qc, HD, QK_CLIP, g.R0, P.rel_clip, g.Rc, NREL, REL_CLIP, NH);
        k_kclip_vrms_pos<<<2 * NKV, 256, 0, s>>>(g.Kconv, g.kcache[il], QK_CLIP, g.Vconv, g.vcache[il], g.pos_dev, NKV, HD, RMS_EPS);
        if (!swa) {
            k_scale_tau_pos<<<(NH * HD + 255) / 256, 256, 0, s>>>(g.Qc, NH * HD, g.pos_dev);
            k_scale_tau_pos<<<1, 256, 0, s>>>(g.Rc, NH * NREL, g.pos_dev);
        }

        // ----- attention over the window (SWA) or all keys (full)
        k_attn_decode_gen<<<NH, HD, 0, s>>>(g.Qc, g.kcache[il], g.vcache[il], g.Rc, P.rel_bias, g.sinks, g.pos_dev,
                                            swa ? NSWA : 0, g.O, g.G, g.gated);
        D.grab("kqv_out-" + std::to_string(il), g.O, NH * HD, s);
        launch_gemv_q8(P.wo.qs, P.wo.sc, g.gated, g.attn_o, NH * HD, E, s);


        // ----- block residual rules (ip % 6 == 0 starts a block)
        if (ip % BLOCK == 0) {
            CHECK(cudaMemcpyAsync(g.blocks_buf + (size_t) g.nb * E, g.cur, E * sizeof(float), cudaMemcpyDeviceToDevice, s));
            g.nb++;
            // cur (old) was copied to blocks above; the conv writes the new cur directly
            k_conv4_res<<<(E + 255) / 256, 256, 0, s>>>(g.attn_o, g.poststate[il], P.post_conv, g.cur, g.cur, E, 0);
        } else {
            k_conv4_res<<<(E + 255) / 256, 256, 0, s>>>(g.attn_o, g.poststate[il], P.post_conv, g.cur, g.cur, E, 1);  // cur += conv
        }
        CHECK(cudaMemcpyAsync(g.ffn_inp, g.cur, E * sizeof(float), cudaMemcpyDeviceToDevice, s));
        D.grab("ffn_inp-" + std::to_string(il), g.ffn_inp, E, s);

        // ----- FFN input mix and loop control (module 3)
        {
            mix_cur(g, P.ffn_res_query, 1, g.mixed, s);
        }
        if (have_msgs) {
            control(g, P, g.mixed, g.msgs_ptr(3), 3, g.mixed, s);   // in place
        }
        k_rms_rows9<<<1, 256, 0, s>>>(g.mixed, P.ffn_norm, g.xf, E, RMS_EPS);

        // ----- MoE
        gemv_f32_row9<<<NEXP, 256, 0, s>>>(P.router_w, g.xf, g.logits_router, E, NEXP);
        k_router_top8<<<1, NEXP, 0, s>>>(g.logits_router, P.exp_bias, g.ids, g.wts, W_SCALE);
        if (P.exp_q8) launch_q8_fused(P.gu_q8, P.gu_q8s, g.xf, 0, g.gu, E, 2304, g.ids, NUSED, s);
        else launch_nvfp4_fused(P.gu_qs, P.gu_sc, g.xf, 0, g.gu, E, 2304, g.ids, NUSED, s);
        if (D.on && il == 44) { D.grab("dbg_gu-44", g.gu, NUSED * 2304, s); D.grab("dbg_wts-44", g.wts, NUSED, s); }
        k_situ<<<dim3((FFN + 255) / 256, NUSED), 256, 0, s>>>(g.gu, P.gu_scale ? P.gu_scale : g.ones, g.ids, g.hbuf,
                                                             SITU_BETA, SITU_LB);
        if (D.on && il == 44) D.grab("dbg_hbuf-44", g.hbuf, NUSED * FFN, s);
        if (P.exp_q8) launch_q8_fused(P.dn_q8, P.dn_q8s, g.hbuf, FFN, g.dn, FFN, E, g.ids, NUSED, s);
        else launch_nvfp4_fused(P.dn_qs, P.dn_sc, g.hbuf, FFN, g.dn, FFN, E, g.ids, NUSED, s);
        if (D.on && il == 44) D.grab("dbg_dn-44", g.dn, NUSED * E, s);

        D.grab("ffn_moe_out-" + std::to_string(il), g.moe, E, s);
        k_moe_conv_res<<<(E + 255) / 256, 256, 0, s>>>(g.dn, P.dn_scale ? P.dn_scale : g.ones, g.ids, g.wts, g.moe,
                                                      g.ffnstate[il], P.ffn_post_conv, g.ffn_inp, g.cur, E);  // cur = ffn_inp + conv(moe)
        D.grab("l_out-" + std::to_string(il), g.cur, E, s);
    }

    // ----- output: mix over all blocks + cur, output norm, head, softcap
    mix_cur(g, g.output_res_query, 1, g.head_in, s);
    k_rms_rows9<<<1, 256, 0, s>>>(g.head_in, g.output_norm, g.xnorm, E, RMS_EPS);
    launch_gemv_q8(g.head.qs, g.head.sc, g.xnorm, g.logits, E, NVOCAB_PAD, s);
    k_softcap_cut<<<(NVOCAB + 255) / 256, 256, 0, s>>>(g.logits, NVOCAB, SOFTCAP);
}

static void set_pos_tok(Engine & g, int pos, int tok, cudaStream_t s) {
    CHECK(cudaMemcpyAsync(g.pos_dev, &pos, sizeof(int), cudaMemcpyHostToDevice, s));
    CHECK(cudaMemcpyAsync(g.tok_dev, &tok, sizeof(int), cudaMemcpyHostToDevice, s));
}

int main(int argc, char ** argv) {
    if (argc < 3) { std::fprintf(stderr, "usage: inkling_model model.gguf outdir [--tokens a,b] [--gen N] [--reps N] [--dump]\n"); return 2; }
    const std::string outdir = argv[2];
    std::vector<int> toks = {371, 151451, 149879, 151194, 149884};
    int gen = 0, reps = 0;
    bool dump = false;
    for (int i = 3; i < argc; ++i) {
        if (std::strcmp(argv[i], "--tokens") == 0 && i + 1 < argc) {
            toks.clear();
            std::stringstream ss(argv[++i]);
            std::string item;
            while (std::getline(ss, item, ',')) toks.push_back(std::atoi(item.c_str()));
        } else if (std::strcmp(argv[i], "--gen") == 0 && i + 1 < argc) gen = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--reps") == 0 && i + 1 < argc) reps = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--dump") == 0) dump = true;
    }
    const int P = (int) toks.size();
    mkdir(outdir.c_str(), 0755);
    if (P + gen > MAXT) { std::fprintf(stderr, "prompt + gen exceeds MAXT\n"); return 1; }

    Model m;
    m.f = gguf::open(argv[1]);
    Engine * en = build_engine(m);
    Engine & g = *en;
    std::printf("loaded model to GPU\n");
    cudaStream_t s;
    CHECK(cudaStreamCreate(&s));

    // one graph for every step: forward + argmax (position and token come from device memory)
    Dumper none;
    cudaGraph_t gr;
    CHECK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
    forward_token(g, none, s);
    k_argmax<<<1, 1024, 0, s>>>(g.logits, NVOCAB, g.tok_dev);
    CHECK(cudaStreamEndCapture(s, &gr));
    cudaGraphExec_t exec;
    CHECK(cudaGraphInstantiate(&exec, gr, nullptr, nullptr, 0));

    // prompt pass (teacher forcing) through the graph; optional eager pass with dumps
    std::vector<float> logits_all;
    Dumper D;
    D.on = dump;
    if (dump) {
        for (int t = 0; t < P; ++t) {
            set_pos_tok(g, t, toks[t], s);
            forward_token(g, D, s);
            k_argmax<<<1, 1024, 0, s>>>(g.logits, NVOCAB, g.tok_dev);
            std::vector<float> lg(NVOCAB);
            CHECK(cudaMemcpy(lg.data(), g.logits, NVOCAB * sizeof(float), cudaMemcpyDeviceToHost));
            logits_all.insert(logits_all.end(), lg.begin(), lg.end());
        }
        mkdir(outdir.c_str(), 0755);
        for (auto & kv : D.data) {
            FILE * f = std::fopen((outdir + "/" + kv.first + ".bin").c_str(), "wb");
            std::fwrite(kv.second.data(), sizeof(float), kv.second.size(), f);
            std::fclose(f);
        }
        FILE * f = std::fopen((outdir + "/logits.bin").c_str(), "wb");
        std::fwrite(logits_all.data(), sizeof(float), logits_all.size(), f);
        std::fclose(f);
    } else {
        for (int t = 0; t < P; ++t) {
            set_pos_tok(g, t, toks[t], s);
            CHECK(cudaGraphLaunch(exec, s));
        }
        CHECK(cudaStreamSynchronize(s));
    }
    int first_gen = 0;
    CHECK(cudaMemcpy(&first_gen, g.tok_dev, sizeof(int), cudaMemcpyDeviceToHost));
    std::printf("prompt done; next token: %d\n", first_gen);

    // continuous generation: same graph, pos incremented on the host, token fed back on device
    std::vector<int> out_tokens(gen > 0 ? gen : 1);
    int * host_tok = nullptr;
    CHECK(cudaMallocHost(&host_tok, sizeof(int) * (gen > 0 ? gen : 1)));
    int * pos_pin = nullptr;
    CHECK(cudaMallocHost(&pos_pin, sizeof(int) * (gen > 0 ? gen : 1)));
    for (int i = 0; i < gen; ++i) pos_pin[i] = P + i;
    if (gen > 0) {
        for (int i = 0; i < gen; ++i) {
            CHECK(cudaMemcpyAsync(g.pos_dev, pos_pin + i, sizeof(int), cudaMemcpyHostToDevice, s));
            CHECK(cudaGraphLaunch(exec, s));
            CHECK(cudaMemcpyAsync(host_tok + i, g.tok_dev, sizeof(int), cudaMemcpyDeviceToHost, s));
        }
        CHECK(cudaStreamSynchronize(s));
        for (int i = 0; i < gen; ++i) out_tokens[i] = host_tok[i];
        std::printf("generated %d tokens: ", gen);
        for (int i = 0; i < gen && i < 16; ++i) std::printf("%d ", out_tokens[i]);
        std::printf("...\n");
        FILE * f = std::fopen((outdir + "/gen_tokens.txt").c_str(), "w");
        for (int i = 0; i < gen; ++i) std::fprintf(f, "%d\n", out_tokens[i]);
        std::fclose(f);
    }

    if (reps > 0 && gen > 0) {
        cudaEvent_t e0, e1;
        CHECK(cudaEventCreate(&e0));
        CHECK(cudaEventCreate(&e1));
        CHECK(cudaEventRecord(e0, s));
        auto h0 = std::chrono::steady_clock::now();
        for (int r = 0; r < reps; ++r)
            for (int i = 0; i < gen; ++i) {
                CHECK(cudaMemcpyAsync(g.pos_dev, pos_pin + i, sizeof(int), cudaMemcpyHostToDevice, s));
                CHECK(cudaGraphLaunch(exec, s));
            }
        auto h1 = std::chrono::steady_clock::now();
        CHECK(cudaEventRecord(e1, s));
        CHECK(cudaEventSynchronize(e1));
        float ms = 0;
        CHECK(cudaEventElapsedTime(&ms, e0, e1));
        const double n = (double) reps * gen;
        const double host_us = std::chrono::duration<double, std::micro>(h1 - h0).count() / n;
        std::printf("generation over %d tokens x %d: GPU %.1f us/token, host %.1f us/token, tok/s %.1f\n",
                    gen, reps, ms * 1000.0 / n, host_us, 1e6 / (ms * 1000.0 / n));
    }
    gguf::close(m.f);
    return 0;
}
