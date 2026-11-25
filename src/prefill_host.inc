// Tensor-core prefill path (textually included after long_host.inc). One chunk = up to PB consecutive tokens of ONE sequence.
//  - dense Q8_0 projections: pf_gemm_dense (weights = mma A operand), expert FFNs: grouped pf_gemm_moe (NVFP4 / Q8_0)
//  - attention: pf_attention (flash style, mma.sync) over the ring / linear KV caches written by kb_kv_write
//  - convs are parallel over time (pf_conv_*), the head is only evaluated for the last tail_k tokens
//  - the state afterwards (KV caches, conv states) is the same as after the same number of sequential steps.
#include "pf_ops.cuh"
#include "pf_attn.cuh"

static void swap_conv_state(Engine & g, BatchState & X, int sl, cudaStream_t s);
static int g_moe_tb = 0;            // tokens per MoE tile: 0 = auto (64 for chunks >= 512), else 16/32/64

struct PFState {
    int PB = 256;
    float *emb, *cur, *prelude_cur, *bank_last, *bank, *blocks, *scores, *inv, *inbuf, *xs3, *msgs, *hcat, *r64, *Qv, *R0, *G,
        *sinks, *dynk, *dynv, *Kconv, *Vconv, *Qc, *Rc, *ffn_inp, *mixed, *xf, *lrouter, *gu, *dn, *wts;
    __half *xs3h, *xfh, *gatedh, *hbuf;
    float *Hk, *Hv, *Hp, *Hf;                       // [3 + PB][C] shared by all applications
    int *ids, *perm, *invp, *rmap, *sexp, *tile_e, *tile_p, *tile_c, *ntiles, *btok, *pos_arr, *slot_arr;
    int * arg_all = nullptr;
    float * part = nullptr;     // split-K partials
};

static void init_pf(PFState & S, int PB) {
    S.PB = PB;
    const size_t T = PB;
    alloc_zero(&S.emb, T * E); alloc_zero(&S.cur, T * E); alloc_zero(&S.prelude_cur, T * E); alloc_zero(&S.bank_last, T * E);
    alloc_zero(&S.bank, T * MS_STATES * E); alloc_zero(&S.blocks, T * MS_STATES * E);
    alloc_zero(&S.scores, T * SC_STRIDE); alloc_zero(&S.inv, T * INV_STRIDE);
    alloc_zero(&S.inbuf, T * 3 * E); alloc_zero(&S.xs3, T * 3 * E); alloc_zero(&S.msgs, T * 4 * E); alloc_zero(&S.hcat, T * 2 * E);
    alloc_zero(&S.r64, T * 64); alloc_zero(&S.Qv, T * NH * HD); alloc_zero(&S.R0, T * NH * NREL); alloc_zero(&S.G, T * NH * HD);
    alloc_zero(&S.sinks, T * NH); alloc_zero(&S.dynk, T * 32); alloc_zero(&S.dynv, T * 32);
    alloc_zero(&S.Kconv, T * NKV * HD); alloc_zero(&S.Vconv, T * NKV * HD); alloc_zero(&S.Qc, T * NH * HD); alloc_zero(&S.Rc, T * NH * NREL);
    alloc_zero(&S.ffn_inp, T * E); alloc_zero(&S.mixed, T * E); alloc_zero(&S.xf, T * E); alloc_zero(&S.lrouter, T * NEXP);
    alloc_zero(&S.part, 16 * T * 128); alloc_zero(&S.gu, T * NUSED * 2304); alloc_zero(&S.dn, T * NUSED * E); alloc_zero(&S.wts, T * NUSED);
    CHECK(cudaMalloc(&S.xs3h, T * 3 * E * 2)); CHECK(cudaMalloc(&S.xfh, T * E * 2)); CHECK(cudaMalloc(&S.gatedh, T * NH * HD * 2));
    CHECK(cudaMalloc(&S.hbuf, T * NUSED * FFN * 2));
    alloc_zero(&S.Hk, (T + 3) * NKV * HD); alloc_zero(&S.Hv, (T + 3) * NKV * HD); alloc_zero(&S.Hp, (T + 3) * E); alloc_zero(&S.Hf, (T + 3) * E);
    CHECK(cudaMalloc(&S.ids, T * NUSED * 4)); CHECK(cudaMalloc(&S.perm, T * NUSED * 4)); CHECK(cudaMalloc(&S.invp, T * NUSED * 4));
    CHECK(cudaMalloc(&S.rmap, T * NUSED * 4)); CHECK(cudaMalloc(&S.sexp, T * NUSED * 4));
    const size_t ntl = T * NUSED / 16 + 256;
    CHECK(cudaMalloc(&S.tile_e, ntl * 4)); CHECK(cudaMalloc(&S.tile_p, ntl * 4)); CHECK(cudaMalloc(&S.tile_c, ntl * 4)); CHECK(cudaMalloc(&S.ntiles, 4));
    CHECK(cudaMalloc(&S.arg_all, T * 4)); CHECK(cudaMalloc(&S.btok, T * 4)); CHECK(cudaMalloc(&S.pos_arr, T * 4)); CHECK(cudaMalloc(&S.slot_arr, T * 4));
}

template <int FMT, int TB>
static void pf_launch_dense(const PFDense & pd, int nrb, cudaStream_t s) {
    dim3 g(nrb, (pd.T + TB - 1) / TB);
    pf_gemm_dense<FMT, TB><<<g, 128, (size_t) PF_NST * TB * PF_BK * 2, s>>>(pd);
}
static void pf_add_seg(PFDense & pd, int & nrb, const int8_t * qs, const __half * sc, int N, float * out, int ostride, int xoff) {
    PFSeg & sg = pd.seg[pd.nseg++];
    sg = {(const uint32_t *) tmap(qs), (const void *) tmap(sc), nrb, N, out, ostride, xoff};
    nrb += N / 128;
}

// Forward of one chunk. tokens are in S.btok / positions in S.pos_arr (pos0 + b) / slots in S.slot_arr (device), already uploaded.
// head_k > 0: logits of the last head_k tokens of the chunk -> X.logits [head_k][NVOCAB_PAD] (softcapped), X.barg.
static void forward_prefill(Engine & g, BatchState & X, PFState & S, int T, int head_k, cudaStream_t s, bool head_all = false) {
    constexpr int MS = MS_STATES;
    if (g_moe_tb == 0) g_moe_tb = (T >= 512) ? 64 : 32;
    const int B = T;
    kb_embed_q8<<<dim3((E + 255) / 256, B), 256, 0, s>>>(g.tok_embd.qs, g.tok_embd.sc, S.btok, E, S.emb);
    k_rms_rows<<<B, 256, 0, s>>>(S.emb, g.tok_norm, S.cur, E, RMS_EPS);
    int nb = 0, prelude_nb = 0, bank_nb = 0;
    const int eb = (E + 255) / 256;
    static const int slot[4] = {0, 2, 3, 1};
    auto msg = [&](int j) { return S.msgs + (size_t) slot[j] * E; };

    auto mix_cur = [&](const float * query, int nq, float * out, int out_bs, int rep) {
        const int Sn = nb + 1;
        kb_mix_scores2<<<dim3(Sn, B), 256, 0, s>>>(S.blocks, MS * E, nb, S.cur, E, query, nq, S.scores, S.inv, Sn, E, RMS_EPS);
        kb_mix_out2<<<dim3(eb, nq, B), 256, 0, s>>>(S.blocks, MS * E, nb, S.cur, E, S.scores, S.inv, out, out_bs, rep, Sn, E);
    };
    auto control_b = [&](const Phys & P, const float * cur, int cur_bs, const float * m, int m_bs, int module, float * out, int out_bs) {
        kb_rms2_rows9<<<dim3(2, B), 256, 0, s>>>(cur, cur_bs, m, m_bs, S.hcat, E, RMS_EPS);
        pf_sgemm_sk<float><<<dim3(1, (B + 63) / 64, 16), 256, 0, s>>>(P.loop_down[module], S.hcat, 2 * E, S.part, 2 * E, 64, B);
        pf_sred<<<(B * 64 + 255) / 256, 256, 0, s>>>(S.part, S.r64, B * 64, 16);
        pf_ctrl_gf<<<dim3(E / 32, (B + 31) / 32), 256, 0, s>>>(cur, cur_bs, m, m_bs, P.loop_gain[module], P.loop_forget[module], S.r64, 1,
                                                               out, out_bs, E, B);
    };

    for (int il = 0; il < NAPP; ++il) {
        const int ip = loop_layer(il);
        const Phys & P = g.phys[ip];
        const bool swa = app_is_swa(il);
        const bool in_core_range = il >= CORE_START && il < CORE_START + NPASS * (CORE_END - CORE_START);
        const int pass = in_core_range ? (il - CORE_START) / (CORE_END - CORE_START) : 0;

        if (il == CORE_START) {
            prelude_nb = nb;
            kb_copy<<<dim3(eb, B), 256, 0, s>>>(S.cur, E, S.prelude_cur, E, E);
        }
        bool have_msgs = false;
        if (in_core_range && pass > 0 && ip == CORE_START) {
            bank_nb = nb - (prelude_nb + 1);
            if (bank_nb > 0)
                kb_copy<<<dim3((bank_nb * E + 255) / 256, B), 256, 0, s>>>(S.blocks + (size_t) (prelude_nb + 1) * E, MS * E, S.bank, MS * E, bank_nb * E);
            kb_copy<<<dim3(eb, B), 256, 0, s>>>(S.cur, E, S.bank_last, E, E);
            nb = prelude_nb;
            kb_copy<<<dim3(eb, B), 256, 0, s>>>(S.prelude_cur, E, S.cur, E, E);
        }
        if (in_core_range && pass > 0) {
            have_msgs = true;
            const int row = 2 * (ip - CORE_START);
            const int Sb = bank_nb + 1;
            kb_mix_scores2<<<dim3(Sb, B), 256, 0, s>>>(S.bank, MS * E, bank_nb, S.bank_last, E, g.loop_res_query + (size_t) row * E, 2,
                                                      S.scores, S.inv, Sb, E, RMS_EPS);
            kb_mix_out2<<<dim3(eb, 2, B), 256, 0, s>>>(S.bank, MS * E, bank_nb, S.bank_last, E, S.scores, S.inv, msg(0), 4 * E, 1, Sb, E);
            kb_mix_scores2<<<dim3(Sb, B), 256, 0, s>>>(S.bank, MS * E, bank_nb, S.bank_last, E, g.loop_kv_res_query + (size_t) row * E, 2,
                                                      S.scores, S.inv, Sb, E, RMS_EPS);
            kb_mix_out2<<<dim3(eb, 2, B), 256, 0, s>>>(S.bank, MS * E, bank_nb, S.bank_last, E, S.scores, S.inv, msg(1), 4 * E, 1, Sb, E);
        }

        // ----- attention input
        if (ip > 0) {
            if (swa) mix_cur(P.attn_res_query, 1, S.inbuf, 3 * E, 3);
            else     mix_cur(P.attn_res_query, 3, S.inbuf, 3 * E, 1);
        } else {
            kb_rep3<<<dim3(eb, B), 256, 0, s>>>(S.cur, S.inbuf, E);
        }
        if (have_msgs)
            for (int j = 0; j < 3; ++j)
                control_b(P, S.inbuf + (size_t) j * E, 3 * E, msg(j), 4 * E, j, S.inbuf + (size_t) j * E, 3 * E);
        if (swa) pf_rms3_h<<<3 * B, 256, 0, s>>>(S.inbuf, P.attn_norm, P.attn_norm, P.attn_norm, S.xs3, S.xs3h, E, RMS_EPS);
        else     pf_rms3_h<<<3 * B, 256, 0, s>>>(S.inbuf, P.q_inp, P.k_inp, P.v_inp, S.xs3, S.xs3h, E, RMS_EPS);

        // ----- projections (K and V go straight into the history rows 3 + b)
        {
            PFDense pd; pd.nseg = 0; pd.x = S.xs3h; pd.xstride = 3 * E; pd.T = B; pd.K = E;
            int nrb = 0;
            pf_add_seg(pd, nrb, P.wq.qs, P.wq.sc, NH * HD, S.Qv, NH * HD, 0);
            pf_add_seg(pd, nrb, P.wk.qs, P.wk.sc, NKV * HD, S.Hk + (size_t) 3 * NKV * HD, NKV * HD, E);
            pf_add_seg(pd, nrb, P.wv.qs, P.wv.sc, NKV * HD, S.Hv + (size_t) 3 * NKV * HD, NKV * HD, 2 * E);
            pf_add_seg(pd, nrb, P.wgate.qs, P.wgate.sc, NH * HD, S.G, NH * HD, 0);
            pf_add_seg(pd, nrb, P.wrel.qs, P.wrel.sc, NH * NREL, S.R0, NH * NREL, 0);
            pf_launch_dense<PF_Q8, 64>(pd, nrb, s);
        }
        {
            const int gy = (B + 63) / 64;
            pf_sgemm_sk<uint16_t><<<dim3(1, gy, 16), 256, 0, s>>>(P.sink_w, S.xs3, 3 * E, S.part, E, NH, B);
            pf_sred<<<(B * NH + 255) / 256, 256, 0, s>>>(S.part, S.sinks, B * NH, 16);
            if (!swa) {
                pf_sgemm_sk<uint16_t><<<dim3(1, gy, 16), 256, 0, s>>>(P.kconv_proj, S.xs3 + E, 3 * E, S.part, E, 32, B);
                pf_sred<<<(B * 32 + 255) / 256, 256, 0, s>>>(S.part, S.dynk, B * 32, 16);
                pf_sgemm_sk<uint16_t><<<dim3(1, gy, 16), 256, 0, s>>>(P.vconv_proj, S.xs3 + 2 * E, 3 * E, S.part, E, 32, B);
                pf_sred<<<(B * 32 + 255) / 256, 256, 0, s>>>(S.part, S.dynv, B * 32, 16);
            }
        }

        // ----- convolutions over time
        pf_conv_prep<<<(NKV * HD + 255) / 256, 256, 0, s>>>(S.Hk, g.kstate[il], NKV * HD);
        pf_conv_prep<<<(NKV * HD + 255) / 256, 256, 0, s>>>(S.Hv, g.vstate[il], NKV * HD);
        pf_conv_kv<<<dim3((NKV * HD + 255) / 256, B, 2), 256, 0, s>>>(S.Hk, P.kconv, S.dynk, S.Kconv, S.Hv, P.vconv, S.dynv, S.Vconv, NKV * HD, swa ? 0 : 1);
        pf_conv_commit<<<(NKV * HD + 255) / 256, 256, 0, s>>>(S.Hk, g.kstate[il], NKV * HD, B);
        pf_conv_commit<<<(NKV * HD + 255) / 256, 256, 0, s>>>(S.Hv, g.vstate[il], NKV * HD, B);

        // ----- clips, cache write at pos + b
        kb_clip_qr<<<dim3(2 * NH, B), 128, 0, s>>>(S.Qv, P.q_clip, S.Qc, HD, QK_CLIP, S.R0, P.rel_clip, S.Rc, NREL, REL_CLIP, NH);
        if (swa || !X.kv16)
            pf_kv_write<float><<<(B * 16 + 7) / 8, 256, 0, s>>>(S.Kconv, S.Vconv, QK_CLIP, S.pos_arr, S.slot_arr, X.kvs[il], swa ? X.ring : 0, RMS_EPS, B);
        else
            pf_kv_write<__half><<<(B * 16 + 7) / 8, 256, 0, s>>>(S.Kconv, S.Vconv, QK_CLIP, S.pos_arr, S.slot_arr, X.kvs[il], 0, RMS_EPS, B);
        if (!swa) {
            kb_scale_tau<<<dim3((NH * HD + 255) / 256, B), 256, 0, s>>>(S.Qc, NH * HD, S.pos_arr);
            kb_scale_tau<<<dim3(2, B), 256, 0, s>>>(S.Rc, NH * NREL, S.pos_arr);
        }

        // ----- attention + output projection into the post-conv history rows
        if (swa)              pf_attention<float>(S.Qc, S.Rc, P.rel_bias, S.sinks, S.G, X.kvs[il], X.ring, NSWA, S.pos_arr, S.slot_arr, B, S.gatedh, s);
        else if (!X.kv16)     pf_attention<float>(S.Qc, S.Rc, P.rel_bias, S.sinks, S.G, X.kvs[il], 0, 0, S.pos_arr, S.slot_arr, B, S.gatedh, s);
        else                  pf_attention<__half>(S.Qc, S.Rc, P.rel_bias, S.sinks, S.G, X.kvs[il], 0, 0, S.pos_arr, S.slot_arr, B, S.gatedh, s);
        {
            PFDense pd; pd.nseg = 0; pd.x = S.gatedh; pd.xstride = NH * HD; pd.T = B; pd.K = NH * HD;
            int nrb = 0;
            pf_add_seg(pd, nrb, P.wo.qs, P.wo.sc, E, S.Hp + (size_t) 3 * E, E, 0);
            pf_launch_dense<PF_Q8, 64>(pd, nrb, s);
        }

        // ----- block residual rules
        pf_conv_prep<<<eb, 256, 0, s>>>(S.Hp, g.poststate[il], E);
        if (ip % BLOCK == 0) {
            kb_copy<<<dim3(eb, B), 256, 0, s>>>(S.cur, E, S.blocks + (size_t) nb * E, MS * E, E);
            nb++;
            pf_conv_res<<<dim3(eb, B), 256, 0, s>>>(S.Hp, P.post_conv, S.cur, S.cur, E, 0);
        } else {
            pf_conv_res<<<dim3(eb, B), 256, 0, s>>>(S.Hp, P.post_conv, S.cur, S.cur, E, 1);
        }
        pf_conv_commit<<<eb, 256, 0, s>>>(S.Hp, g.poststate[il], E, B);
        kb_copy<<<dim3(eb, B), 256, 0, s>>>(S.cur, E, S.ffn_inp, E, E);

        // ----- FFN input mix, loop control, norm
        mix_cur(P.ffn_res_query, 1, S.mixed, E, 1);
        if (have_msgs) control_b(P, S.mixed, E, msg(3), 4 * E, 3, S.mixed, E);
        pf_rms_h<<<B, 256, 0, s>>>(S.mixed, P.ffn_norm, S.xf, S.xfh, E, RMS_EPS);

        // ----- MoE
        pf_sgemm_sk<float><<<dim3(NEXP / 64, (B + 63) / 64, 16), 256, 0, s>>>(P.router_w, S.xf, E, S.part, E, NEXP, B);
        pf_router_top8<<<B, NEXP, 0, s>>>(S.part, 16, B, P.exp_bias, S.ids, S.wts, W_SCALE);
        auto moe_stage = [&](auto tbc) {
            constexpr int TB = decltype(tbc)::value;
            pf_group<TB><<<1, 1024, 0, s>>>(S.ids, B * NUSED, S.perm, S.invp, S.rmap, S.sexp, S.tile_e, S.tile_p, S.tile_c, S.ntiles);
            const dim3 gg(2304 / 128, B * NUSED / TB + NEXP);
            const size_t msm = (size_t) PF_NST * TB * PF_BK * 2;
            if (P.exp_q8) {
                pf_gemm_moe<PF_Q8, TB><<<gg, 128, msm, s>>>((const uint32_t *) tmap(P.gu_q8), (const void *) tmap(P.gu_q8s), 2304, E, S.xfh, E, S.rmap, S.gu,
                                                           S.tile_e, S.tile_p, S.tile_c, S.ntiles);
            } else {
                pf_gemm_moe<PF_NV4, TB><<<gg, 128, msm, s>>>((const uint32_t *) tmap(P.gu_qs), (const void *) tmap(P.gu_sc), 2304, E, S.xfh, E, S.rmap, S.gu,
                                                            S.tile_e, S.tile_p, S.tile_c, S.ntiles);
            }
            pf_situ<<<dim3((FFN + 255) / 256, B * NUSED), 256, 0, s>>>(S.gu, P.gu_scale ? P.gu_scale : g.ones, S.sexp, S.hbuf, SITU_BETA, SITU_LB);
            if (P.exp_q8) {
                pf_gemm_moe<PF_Q8, TB><<<gg, 128, msm, s>>>((const uint32_t *) tmap(P.dn_q8), (const void *) tmap(P.dn_q8s), E, FFN, S.hbuf, FFN, nullptr, S.dn,
                                                           S.tile_e, S.tile_p, S.tile_c, S.ntiles);
            } else {
                pf_gemm_moe<PF_NV4, TB><<<gg, 128, msm, s>>>((const uint32_t *) tmap(P.dn_qs), (const void *) tmap(P.dn_sc), E, FFN, S.hbuf, FFN, nullptr, S.dn,
                                                            S.tile_e, S.tile_p, S.tile_c, S.ntiles);
            }
        };
        if (g_moe_tb == 64) moe_stage(std::integral_constant<int, 64>());
        else if (g_moe_tb == 16) moe_stage(std::integral_constant<int, 16>());
        else moe_stage(std::integral_constant<int, 32>());
        pf_moe_sum<<<dim3((E + 255) / 256, B), 256, 0, s>>>(S.dn, P.dn_scale ? P.dn_scale : g.ones, S.ids, S.wts, S.invp, S.Hf + (size_t) 0, E);
        pf_conv_prep<<<eb, 256, 0, s>>>(S.Hf, g.ffnstate[il], E);
        pf_conv_res<<<dim3(eb, B), 256, 0, s>>>(S.Hf, P.ffn_post_conv, S.ffn_inp, S.cur, E, 1);
        pf_conv_commit<<<eb, 256, 0, s>>>(S.Hf, g.ffnstate[il], E, B);
        dump_cur(il, S.cur, B, s);
    }

    // ----- output: only the last head_k tokens (head_all: validation mode, argmax of every token into S.arg_all)
    auto head = [&](int K0, int hk) {
        const float * blocks0 = S.blocks + (size_t) K0 * MS * E;
        const int Sn = nb + 1;
        kb_mix_scores2<<<dim3(Sn, hk), 256, 0, s>>>(blocks0, MS * E, nb, S.cur + (size_t) K0 * E, E, g.output_res_query, 1, S.scores, S.inv, Sn, E, RMS_EPS);
        kb_mix_out2<<<dim3(eb, 1, hk), 256, 0, s>>>(blocks0, MS * E, nb, S.cur + (size_t) K0 * E, E, S.scores, S.inv, X.head_in, E, 1, Sn, E);
        kb_rms_rows9<<<hk, 256, 0, s>>>(X.head_in, g.output_norm, X.xnorm, E, RMS_EPS);
        launch_gemvb_q8(g.head.qs, g.head.sc, X.xnorm, E, X.logits, NVOCAB_PAD, E, NVOCAB_PAD, hk, s);
        kb_softcap_cut<<<dim3((NVOCAB + 255) / 256, hk), 256, 0, s>>>(X.logits, NVOCAB, NVOCAB_PAD, SOFTCAP);
        kb_argmax<<<hk, 1024, 0, s>>>(X.logits, NVOCAB, NVOCAB_PAD, X.barg);
    };
    if (head_all) {
        for (int K0 = 0; K0 < B; K0 += 16) {
            const int hk = std::min(16, B - K0);
            head(K0, hk);
            CHECK(cudaMemcpyAsync(S.arg_all + K0, X.barg, hk * 4, cudaMemcpyDeviceToDevice, s));
        }
    } else if (head_k > 0) head(B - head_k, head_k);
}

// Prefill n tokens of sequence `slot` at positions start.. in chunks of PB. Returns tokens/s (GPU events over the whole loop).
// The logits of the last tail_k (<= 16) tokens are returned in tail_logits ([tail_k][NVOCAB]); *last_arg = argmax of the last token.
static double prefill_fast(Engine & g, BatchState & X, PFState & S, int slot, const int * toks, int n, int start, int tail_k,
                           std::vector<float> * tail_logits, int * last_arg, cudaStream_t s, int * argmax_all = nullptr) {
    const int PB = S.PB;
    if (start + n > X.maxctx || slot >= X.nslots || X.ring < NSWA + PB - 1) { std::fprintf(stderr, "prefill_fast: out of range (ring %d needs >= %d)\n", X.ring, NSWA + PB - 1); std::exit(1); }
    tail_k = std::max(1, std::min(tail_k, 16));
    swap_conv_state(g, X, slot, s);
    std::vector<int> pos(PB), sl(PB, slot);
    cudaEvent_t e0, e1;
    CHECK(cudaEventCreate(&e0)); CHECK(cudaEventCreate(&e1));
    CHECK(cudaStreamSynchronize(s));
    CHECK(cudaEventRecord(e0, s));
    const int last_chunk_start = ((n - 1) / PB) * PB;
    for (int p = 0; p < n; p += PB) {
        const int T = std::min(PB, n - p);
        for (int k = 0; k < T; ++k) pos[k] = start + p + k;
        CHECK(cudaMemcpyAsync(S.btok, toks + p, T * sizeof(int), cudaMemcpyHostToDevice, s));
        CHECK(cudaMemcpyAsync(S.pos_arr, pos.data(), T * sizeof(int), cudaMemcpyHostToDevice, s));
        CHECK(cudaMemcpyAsync(S.slot_arr, sl.data(), T * sizeof(int), cudaMemcpyHostToDevice, s));
        CHECK(cudaStreamSynchronize(s));      // pinned-free host vectors: make sure the copies are done before they change
        const int hk = (p == last_chunk_start) ? std::min(tail_k, T) : 0;
        forward_prefill(g, X, S, T, hk, s, argmax_all != nullptr);
        if (argmax_all) CHECK(cudaMemcpyAsync(argmax_all + p, S.arg_all, T * 4, cudaMemcpyDeviceToHost, s));
        if (p == last_chunk_start) {
            if (tail_logits) {
                tail_logits->assign((size_t) hk * NVOCAB, 0.0f);
                CHECK(cudaMemcpy2DAsync(tail_logits->data(), NVOCAB * sizeof(float), X.logits, NVOCAB_PAD * sizeof(float), NVOCAB * sizeof(float), hk, cudaMemcpyDeviceToHost, s));
            }
            if (last_arg) CHECK(cudaMemcpyAsync(last_arg, X.barg + hk - 1, sizeof(int), cudaMemcpyDeviceToHost, s));
        }
    }
    CHECK(cudaEventRecord(e1, s));
    CHECK(cudaEventSynchronize(e1));
    float ms = 0;
    CHECK(cudaEventElapsedTime(&ms, e0, e1));
    swap_conv_state(g, X, slot, s);
    return n / (ms / 1000.0);
}
