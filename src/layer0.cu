// Inkling 2 Mini, layer 0 on the GPU (sequential decode: one token per step, KV/conv state kept).
// usage: layer0 <model.gguf> <outdir> [--tokens id,id,...] [--reps N]
// Writes per-token intermediates to <outdir>/<name>.bin (float32, [T][...]).
#include "gguf.hpp"
#include "kernels_common.cuh"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <sstream>
#include <vector>
#include <sys/stat.h>

#define CHECK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    std::fprintf(stderr, "CUDA %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); std::exit(1); } } while (0)

// model constants (blk.0 of inkling2)
constexpr int E = 2304, NH = 24, NKV = 8, HD = 128, NREL = 16, NSWA = 1024, NEXP = 128, NUSED = 8, FFN = 1152;
constexpr int MAXT = 2048;
constexpr float QK_CLIP = 22.627416610717773f, REL_CLIP = 8.0f, RMS_EPS = 1e-6f;
constexpr float SITU_BETA = 4.0f, SITU_LB = 25.0f, W_SCALE = 8.0f;

static std::vector<uint16_t> repack_q8_qs_dummy;

// ---------------------------------------------------------------- host ----

struct Model {
    gguf::File f;
    const gguf::Tensor * find(const std::string & name) {
        for (const auto & t : f.tensors) if (t.name == name) return &t;
        std::fprintf(stderr, "missing tensor %s\n", name.c_str());
        std::exit(1);
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

struct Q8W { int8_t * qs; __half * sc; int K, N; };

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

static float * load_f32(Model & m, const std::string & name, size_t count) {
    const gguf::Tensor * t = m.find(name);
    if (t->type != gguf::F32) { std::fprintf(stderr, "%s not F32\n", name.c_str()); std::exit(1); }
    return upload<float>(m.ptr(t), count);
}

// ------------------------------------------------------------- state ----

struct Layer0 {
    // weights
    Q8W tok_embd;
    float *tok_norm, *attn_norm, *ffn_norm, *ffn_res_query, *q_clip, *rel_clip, *rel_bias, *kconv_w, *vconv_w,
        *post_w, *ffn_post_w, *router_w, *exp_bias, *gu_scale, *dn_scale;
    Q8W wq, wk, wv, wrel, wgate, wo;
    uint16_t * sink_w;
    uint8_t *gu_qs, *gu_sc, *dn_qs, *dn_sc;
    // states
    float *kstate, *vstate, *poststate, *ffnstate;
    // KV cache
    float *kcache, *vcache;
    // scratch
    float *emb, *cur, *x1, *xw, *Qv, *K0, *V0, *R0, *G, *sinks, *Kconv, *Vconv, *Qc, *Rc, *O, *gated, *attn_o,
        *attn_post, *mixed, *x2, *xf, *logits, *gu, *hbuf, *dn, *moe, *ffn_post, *lout;
    int *ids;
    float *wts;
};

static void alloc_f(float ** p, size_t n) { CHECK(cudaMalloc(p, n * sizeof(float))); CHECK(cudaMemset(*p, 0, n * sizeof(float))); }

static Layer0 load_layer0(Model & m) {
    Layer0 L;
    // tok_embd: Q8 [E][V]
    {
        const gguf::Tensor * t = m.find("token_embd.weight");
        const int V = (int) t->ne[1];
        std::vector<int8_t> qs;
        std::vector<uint16_t> sc;
        repack_q8_soa(m.ptr(t), E, V, qs, sc);
        L.tok_embd.qs = upload<int8_t>(qs.data(), qs.size());
        L.tok_embd.sc = upload<__half>(sc.data(), sc.size());
        L.tok_embd.K = E;
        L.tok_embd.N = V;
    }
    L.tok_norm = load_f32(m, "token_embd_norm.weight", E);
    L.attn_norm = load_f32(m, "blk.0.attn_norm.weight", E);
    L.ffn_norm = load_f32(m, "blk.0.ffn_norm.weight", E);
    L.ffn_res_query = load_f32(m, "blk.0.ffn_res_query.weight", E);
    L.q_clip = load_f32(m, "blk.0.attn_q_clip.weight", NH * HD);
    L.rel_clip = load_f32(m, "blk.0.attn_rel_clip.weight", NH * NREL);
    L.rel_bias = load_f32(m, "blk.0.attn_rel_bias.weight", NSWA * NREL);
    L.kconv_w = load_f32(m, "blk.0.attn_k_conv.weight", NKV * HD * 4);
    L.vconv_w = load_f32(m, "blk.0.attn_v_conv.weight", NKV * HD * 4);
    L.post_w = load_f32(m, "blk.0.attn_post_conv.weight", E * 4);
    L.ffn_post_w = load_f32(m, "blk.0.ffn_post_conv.weight", E * 4);
    L.router_w = load_f32(m, "blk.0.ffn_gate_inp.weight", NEXP * E);
    L.exp_bias = load_f32(m, "blk.0.exp_probs_b.bias", NEXP);
    L.gu_scale = load_f32(m, "blk.0.ffn_gate_up_exps.scale", NEXP);
    L.dn_scale = load_f32(m, "blk.0.ffn_down_exps.scale", NEXP);
    // attention linears (Q8_0): K = input width, N = output width
    L.wq = load_q8(m, "blk.0.attn_q.weight", E, NH * HD);
    L.wk = load_q8(m, "blk.0.attn_k.weight", E, NKV * HD);
    L.wv = load_q8(m, "blk.0.attn_v.weight", E, NKV * HD);
    L.wrel = load_q8(m, "blk.0.attn_rel.weight", E, NH * NREL);
    L.wgate = load_q8(m, "blk.0.attn_gate.weight", E, NH * HD);
    L.wo = load_q8(m, "blk.0.attn_output.weight", NH * HD, E);
    // attn_sink_proj BF16 [E][NH] (ne0 = E)
    {
        const gguf::Tensor * t = m.find("blk.0.attn_sink_proj.weight");
        L.sink_w = upload<uint16_t>(m.ptr(t), (size_t) E * NH);
    }
    // NVFP4 experts, all 128 resident, SoA
    {
        const gguf::Tensor * tg = m.find("blk.0.ffn_gate_up_exps.weight");  // [E=2304 K][2304 N][128]
        const size_t per = (size_t) 2304 * (E / 64) * 36;
        std::vector<uint8_t> qs((size_t) NEXP * 2304 * (E / 2)), sc((size_t) NEXP * 2304 * (E / 16));
        for (int e = 0; e < NEXP; ++e)
            repack_nv_soa(m.ptr(tg) + e * per, E, 2304, &qs[(size_t) e * 2304 * (E / 2)], &sc[(size_t) e * 2304 * (E / 16)]);
        L.gu_qs = upload<uint8_t>(qs.data(), qs.size());
        L.gu_sc = upload<uint8_t>(sc.data(), sc.size());
    }
    {
        const gguf::Tensor * td = m.find("blk.0.ffn_down_exps.weight");  // [1152 K][2304 N][128]
        const size_t per = (size_t) 2304 * (FFN / 64) * 36;
        std::vector<uint8_t> qs((size_t) NEXP * 2304 * (FFN / 2)), sc((size_t) NEXP * 2304 * (FFN / 16));
        for (int e = 0; e < NEXP; ++e)
            repack_nv_soa(m.ptr(td) + e * per, FFN, 2304, &qs[(size_t) e * 2304 * (FFN / 2)], &sc[(size_t) e * 2304 * (FFN / 16)]);
        L.dn_qs = upload<uint8_t>(qs.data(), qs.size());
        L.dn_sc = upload<uint8_t>(sc.data(), sc.size());
    }
    // states and caches
    alloc_f(&L.kstate, 3 * NKV * HD);
    alloc_f(&L.vstate, 3 * NKV * HD);
    alloc_f(&L.poststate, 3 * E);
    alloc_f(&L.ffnstate, 3 * E);
    alloc_f(&L.kcache, (size_t) MAXT * NKV * HD);
    alloc_f(&L.vcache, (size_t) MAXT * NKV * HD);
    alloc_f(&L.emb, E); alloc_f(&L.cur, E); alloc_f(&L.x1, E); alloc_f(&L.xw, E);
    alloc_f(&L.Qv, NH * HD); alloc_f(&L.K0, NKV * HD); alloc_f(&L.V0, NKV * HD); alloc_f(&L.R0, NH * NREL);
    alloc_f(&L.G, NH * HD); alloc_f(&L.sinks, NH); alloc_f(&L.Kconv, NKV * HD); alloc_f(&L.Vconv, NKV * HD);
    alloc_f(&L.Qc, NH * HD); alloc_f(&L.Rc, NH * NREL); alloc_f(&L.O, NH * HD); alloc_f(&L.gated, NH * HD);
    alloc_f(&L.attn_o, E); alloc_f(&L.attn_post, E); alloc_f(&L.mixed, E); alloc_f(&L.x2, E); alloc_f(&L.xf, E);
    alloc_f(&L.logits, NEXP); alloc_f(&L.gu, NUSED * 2304); alloc_f(&L.hbuf, NUSED * FFN); alloc_f(&L.dn, NUSED * E);
    alloc_f(&L.moe, E); alloc_f(&L.ffn_post, E); alloc_f(&L.lout, E);
    CHECK(cudaMalloc(&L.ids, NUSED * sizeof(int)));
    alloc_f(&L.wts, NUSED);
    return L;
}

// ------------------------------------------------------------- forward ----

// Host-side collection of intermediates, one entry per token.
struct Dumps {
    bool on = false;
    std::map<std::string, std::vector<float>> data;
    void grab(const char * name, const float * dev, size_t n, cudaStream_t s) {
        if (!on) return;
        std::vector<float> h(n);
        CHECK(cudaMemcpyAsync(h.data(), dev, n * sizeof(float), cudaMemcpyDeviceToHost, s));
        CHECK(cudaStreamSynchronize(s));
        auto & v = data[name];
        v.insert(v.end(), h.begin(), h.end());
    }
};

static void forward_token(Layer0 & L, int tok, int pos, Dumps & D, cudaStream_t s) {
    // embedding and input norms
    k_embed_q8<<<(E + 255) / 256, 256, 0, s>>>(L.tok_embd.qs + 0, L.tok_embd.sc + 0, tok, E, L.emb);
    k_rms_rows<<<1, 256, 0, s>>>(L.emb, L.tok_norm, L.cur, E, RMS_EPS);
    k_rms_rows<<<1, 256, 0, s>>>(L.cur, nullptr, L.x1, E, RMS_EPS);
    D.grab("norm-0#1", L.x1, E, s);
    k_rms_rows<<<1, 256, 0, s>>>(L.cur, L.attn_norm, L.xw, E, RMS_EPS);

    // attention projections
    launch_gemv_q8(L.wq.qs, L.wq.sc, L.xw, L.Qv, E, NH * HD, s);
    launch_gemv_q8(L.wk.qs, L.wk.sc, L.xw, L.K0, E, NKV * HD, s);
    launch_gemv_q8(L.wv.qs, L.wv.sc, L.xw, L.V0, E, NKV * HD, s);
    launch_gemv_q8(L.wrel.qs, L.wrel.sc, L.xw, L.R0, E, NH * NREL, s);
    launch_gemv_q8(L.wgate.qs, L.wgate.sc, L.xw, L.G, E, NH * HD, s);
    gemv_bf16<<<(NH * 32 + 255) / 256, 256, 0, s>>>(L.sink_w, L.xw, L.sinks, E, NH);

    // short convolutions with state (K and V)
    k_conv4<<<(NKV * HD + 255) / 256, 256, 0, s>>>(L.K0, L.kstate, L.kconv_w, L.Kconv, NKV * HD);
    k_conv4<<<(NKV * HD + 255) / 256, 256, 0, s>>>(L.V0, L.vstate, L.vconv_w, L.Vconv, NKV * HD);

    // clips and norms; K and V go to the cache at this position
    k_clip_heads<<<NH, HD, 0, s>>>(L.Qv, L.q_clip, L.Qc, HD, QK_CLIP);
    k_clip_heads<<<NKV, HD, 0, s>>>(L.Kconv, nullptr, L.kcache + (size_t) pos * NKV * HD, HD, QK_CLIP);
    k_rms_rows<<<NKV, 256, 0, s>>>(L.Vconv, nullptr, L.vcache + (size_t) pos * NKV * HD, HD, RMS_EPS);
    k_clip_heads<<<NH, 128, 0, s>>>(L.R0, L.rel_clip, L.Rc, NREL, REL_CLIP);
    D.grab("Qcur-0", L.Qc, NH * HD, s);
    D.grab("Kcur-0", L.kcache + (size_t) pos * NKV * HD, NKV * HD, s);
    D.grab("Vcur-0", L.vcache + (size_t) pos * NKV * HD, NKV * HD, s);
    D.grab("Rcur-0", L.Rc, NH * NREL, s);

    // attention over the window, relative bias, sinks
    k_attn_decode<<<NH, HD, 0, s>>>(L.Qc, L.kcache, L.vcache, L.Rc, L.rel_bias, L.sinks, pos, L.O);
    D.grab("kqv_out-0", L.O, NH * HD, s);

    // gate, output projection, post convolution (block start: no residual)
    k_gate_mul<<<(NH * HD + 255) / 256, 256, 0, s>>>(L.O, L.G, L.gated, NH * HD);
    launch_gemv_q8(L.wo.qs, L.wo.sc, L.gated, L.attn_o, NH * HD, E, s);
    k_conv4<<<(E + 255) / 256, 256, 0, s>>>(L.attn_o, L.poststate, L.post_w, L.attn_post, E);
    D.grab("ffn_inp-0", L.attn_post, E, s);

    // mix(blocks=[cur], attn_post, ffn_res_query) and FFN input norm
    k_mix2<<<1, 256, 0, s>>>(L.cur, L.attn_post, L.ffn_res_query, L.mixed, E);
    k_rms_rows<<<1, 256, 0, s>>>(L.mixed, nullptr, L.x2, E, RMS_EPS);
    D.grab("norm-0#2", L.x2, E, s);
    k_rms_rows<<<1, 256, 0, s>>>(L.mixed, L.ffn_norm, L.xf, E, RMS_EPS);

    // MoE: router, top-8, fused gate_up, SITU, fused down, weighted sum
    gemv_f32<<<(NEXP * 32 + 255) / 256, 256, 0, s>>>(L.router_w, L.xf, L.logits, E, NEXP);
    D.grab("ffn_moe_logits-0", L.logits, NEXP, s);
    k_router_top8<<<1, NEXP, 0, s>>>(L.logits, L.exp_bias, L.ids, L.wts, W_SCALE);
    launch_nvfp4_fused(L.gu_qs, L.gu_sc, L.xf, 0, L.gu, E, 2304, L.ids, NUSED, s);
    k_situ<<<dim3((FFN + 255) / 256, NUSED), 256, 0, s>>>(L.gu, L.gu_scale, L.ids, L.hbuf, SITU_BETA, SITU_LB);
    launch_nvfp4_fused(L.dn_qs, L.dn_sc, L.hbuf, FFN, L.dn, FFN, E, L.ids, NUSED, s);
    k_moe_sum<<<(E + 255) / 256, 256, 0, s>>>(L.dn, L.dn_scale, L.ids, L.wts, L.moe, E);
    D.grab("ffn_moe_out-0", L.moe, E, s);

    // FFN post convolution and residual
    k_conv4<<<(E + 255) / 256, 256, 0, s>>>(L.moe, L.ffnstate, L.ffn_post_w, L.ffn_post, E);
    D.grab("ffn_out-0", L.ffn_post, E, s);
    k_add<<<(E + 255) / 256, 256, 0, s>>>(L.attn_post, L.ffn_post, L.lout, E);
    D.grab("l_out-0", L.lout, E, s);
}

// ---------------------------------------------------------------- main ----

int main(int argc, char ** argv) {
    if (argc < 3) { std::fprintf(stderr, "usage: layer0 model.gguf outdir [--tokens a,b,c] [--reps N]\n"); return 2; }
    std::string outdir = argv[2];
    std::vector<int> toks = {371, 151451, 149879, 151194, 149884};
    int reps = 0;
    for (int i = 3; i < argc; ++i) {
        if (std::strcmp(argv[i], "--tokens") == 0 && i + 1 < argc) {
            toks.clear();
            std::stringstream ss(argv[++i]);
            std::string item;
            while (std::getline(ss, item, ',')) toks.push_back(std::atoi(item.c_str()));
        } else if (std::strcmp(argv[i], "--reps") == 0 && i + 1 < argc) {
            reps = std::atoi(argv[++i]);
        }
    }
    if ((int) toks.size() > MAXT) { std::fprintf(stderr, "too many tokens\n"); return 1; }

    Model m;
    m.f = gguf::open(argv[1]);
    Layer0 L = load_layer0(m);
    std::printf("loaded layer 0 to GPU, %zu tokens\n", toks.size());

    cudaStream_t s;
    CHECK(cudaStreamCreate(&s));

    // one pass with dumps (validation), then timed passes without host copies
    Dumps D;
    D.on = true;
    for (size_t t = 0; t < toks.size(); ++t) forward_token(L, toks[t], (int) t, D, s);
    CHECK(cudaStreamSynchronize(s));

    mkdir(outdir.c_str(), 0755);
    for (auto & kv : D.data) {
        FILE * f = std::fopen((outdir + "/" + kv.first + ".bin").c_str(), "wb");
        std::fwrite(kv.second.data(), sizeof(float), kv.second.size(), f);
        std::fclose(f);
    }
    std::printf("wrote %zu intermediate tensors to %s\n", D.data.size(), outdir.c_str());

    if (reps > 0) {
        Dumps off;
        off.on = false;
        for (int w = 0; w < 2; ++w)
            for (size_t t = 0; t < toks.size(); ++t) forward_token(L, toks[t], (int) t, off, s);
        CHECK(cudaStreamSynchronize(s));
        cudaEvent_t e0, e1;
        CHECK(cudaEventCreate(&e0));
        CHECK(cudaEventCreate(&e1));
        CHECK(cudaEventRecord(e0, s));
        for (int r = 0; r < reps; ++r)
            for (size_t t = 0; t < toks.size(); ++t) forward_token(L, toks[t], (int) t, off, s);
        CHECK(cudaEventRecord(e1, s));
        CHECK(cudaEventSynchronize(e1));
        float ms = 0;
        CHECK(cudaEventElapsedTime(&ms, e0, e1));
        std::printf("layer 0 GPU time per token (stream, no host sync): %.1f us\n", ms * 1000.0 / (reps * toks.size()));
    }
    gguf::close(m.f);
    return 0;
}
