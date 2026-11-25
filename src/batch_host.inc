// Host side of the multi-token forward pass (textually included into inkling_model.cu before main).
// forward_batch(B): tokens X.btok[0..B) (device) at positions *g.pos_dev + b. Produces X.logits [B][NVOCAB_PAD]
// (first NVOCAB valid, softcapped) and X.barg[B] (argmax per token). Engine state afterwards = state after B sequential
// forward_token steps; batch_rollback(j) rewinds the carried conv states to "after j tokens of the batch".

struct BatchState {
    float *emb, *cur, *prelude_cur, *bank_last, *bank, *blocks, *scores, *inv, *inbuf, *xs3, *msgs, *hcat, *r64, *Qv, *R0, *G,
        *sinks, *dynk, *dynv, *Kconv, *Vconv, *Qc, *Rc, *O, *gated, *ffn_inp, *mixed, *xf, *lrouter, *gu, *hbuf, *dn,
        *head_in, *xnorm, *logits, *wts;
    int *ids, *grp_e, *grp_n, *grp_t, *grp_nu, *btok, *barg, *rb_j, *pos_arr, *slot_arr;
    // extra sequence slots 1..3 (multi-sequence mode): KV caches [3][MAXT][NKV*HD] and conv states [3][3*C] per application
    float *kx[NAPP], *vx[NAPP], *Sxk[NAPP], *Sxv[NAPP], *Sxp[NAPP], *Sxf[NAPP];
    float *Hk[NAPP], *Hv[NAPP], *Hp[NAPP], *Hf[NAPP];     // [3 + MB][C] history/input buffers per application
    RbEntry * rb_tab;
};

static void init_batch(Engine & g, BatchState & X) {
    for (int ip = 0; ip < NPHYS; ++ip) {            // interleaved weight copies for the lane-per-row kernels
        const Phys & P = g.phys[ip];
        register_q8_il(P.wq.qs, P.wq.sc, NH * HD, E); register_q8_il(P.wk.qs, P.wk.sc, NKV * HD, E);
        register_q8_il(P.wv.qs, P.wv.sc, NKV * HD, E); register_q8_il(P.wrel.qs, P.wrel.sc, NH * NREL, E);
        register_q8_il(P.wgate.qs, P.wgate.sc, NH * HD, E); register_q8_il(P.wo.qs, P.wo.sc, E, NH * HD);
        if (P.exp_q8) { register_q8_il(P.gu_q8, P.gu_q8s, 2304, E, NEXP); register_q8_il(P.dn_q8, P.dn_q8s, E, FFN, NEXP); }
        else { register_nv_il(P.gu_qs, P.gu_sc, 2304, E, NEXP); register_nv_il(P.dn_qs, P.dn_sc, E, FFN, NEXP); }
    }
    register_q8_il(g.head.qs, g.head.sc, NVOCAB_PAD, E);
    CHECK(cudaDeviceSynchronize());
    alloc_zero(&X.emb, MB * E); alloc_zero(&X.cur, MB * E); alloc_zero(&X.prelude_cur, MB * E); alloc_zero(&X.bank_last, MB * E);
    alloc_zero(&X.bank, (size_t) MB * MS_STATES * E); alloc_zero(&X.blocks, (size_t) MB * MS_STATES * E);
    alloc_zero(&X.scores, MB * SC_STRIDE); alloc_zero(&X.inv, MB * INV_STRIDE);
    alloc_zero(&X.inbuf, MB * 3 * E); alloc_zero(&X.xs3, MB * 3 * E); alloc_zero(&X.msgs, MB * 4 * E); alloc_zero(&X.hcat, MB * 2 * E);
    alloc_zero(&X.r64, MB * 64); alloc_zero(&X.Qv, MB * NH * HD); alloc_zero(&X.R0, MB * NH * NREL); alloc_zero(&X.G, MB * NH * HD);
    alloc_zero(&X.sinks, MB * NH); alloc_zero(&X.dynk, MB * 32); alloc_zero(&X.dynv, MB * 32);
    alloc_zero(&X.Kconv, MB * NKV * HD); alloc_zero(&X.Vconv, MB * NKV * HD); alloc_zero(&X.Qc, MB * NH * HD);
    alloc_zero(&X.Rc, MB * NH * NREL); alloc_zero(&X.O, MB * NH * HD); alloc_zero(&X.gated, MB * NH * HD);
    alloc_zero(&X.ffn_inp, MB * E); alloc_zero(&X.mixed, MB * E); alloc_zero(&X.xf, MB * E); alloc_zero(&X.lrouter, MB * NEXP);
    alloc_zero(&X.gu, (size_t) MB * NUSED * 2304); alloc_zero(&X.hbuf, (size_t) MB * NUSED * FFN); alloc_zero(&X.dn, (size_t) MB * NUSED * E);
    alloc_zero(&X.head_in, MB * E); alloc_zero(&X.xnorm, MB * E); alloc_zero(&X.logits, (size_t) MB * NVOCAB_PAD);
    alloc_zero(&X.wts, MB * NUSED);
    CHECK(cudaMalloc(&X.ids, MB * NUSED * sizeof(int)));
    CHECK(cudaMalloc(&X.grp_e, NEXP * sizeof(int))); CHECK(cudaMalloc(&X.grp_n, NEXP * sizeof(int)));
    CHECK(cudaMalloc(&X.grp_t, NEXP * 16 * sizeof(int))); CHECK(cudaMalloc(&X.grp_nu, sizeof(int)));
    CHECK(cudaMalloc(&X.pos_arr, MB * sizeof(int))); CHECK(cudaMalloc(&X.slot_arr, MB * sizeof(int))); CHECK(cudaMemset(X.slot_arr, 0, MB * sizeof(int)));
    CHECK(cudaMalloc(&X.btok, MB * sizeof(int))); CHECK(cudaMalloc(&X.barg, MB * sizeof(int))); CHECK(cudaMalloc(&X.rb_j, sizeof(int)));
    std::vector<RbEntry> tab;
    for (int il = 0; il < NAPP; ++il) {
        alloc_zero(&X.Hk[il], (size_t) (3 + MB) * NKV * HD); alloc_zero(&X.Hv[il], (size_t) (3 + MB) * NKV * HD);
        alloc_zero(&X.Hp[il], (size_t) (3 + MB) * E); alloc_zero(&X.Hf[il], (size_t) (3 + MB) * E);
        alloc_zero(&X.kx[il], (size_t) 3 * MAXT * NKV * HD); alloc_zero(&X.vx[il], (size_t) 3 * MAXT * NKV * HD);
        alloc_zero(&X.Sxk[il], 3 * 3 * NKV * HD); alloc_zero(&X.Sxv[il], 3 * 3 * NKV * HD);
        alloc_zero(&X.Sxp[il], 3 * 3 * E); alloc_zero(&X.Sxf[il], 3 * 3 * E);
        tab.push_back({g.kstate[il], X.Hk[il], NKV * HD});
        tab.push_back({g.vstate[il], X.Hv[il], NKV * HD});
        tab.push_back({g.poststate[il], X.Hp[il], E});
        tab.push_back({g.ffnstate[il], X.Hf[il], E});
    }
    CHECK(cudaMalloc(&X.rb_tab, tab.size() * sizeof(RbEntry)));
    CHECK(cudaMemcpy(X.rb_tab, tab.data(), tab.size() * sizeof(RbEntry), cudaMemcpyHostToDevice));
}

// multi == 0: the B tokens are consecutive positions of ONE sequence (slot 0, positions X.pos_arr[b] = p + b).
// multi == 1: token b belongs to its own sequence slot X.slot_arr[b] (distinct slots, one token each) at position X.pos_arr[b].
static void forward_batch(Engine & g, BatchState & X, int B, int multi, cudaStream_t s) {
    constexpr int MS = MS_STATES;
    kb_embed_q8<<<dim3((E + 255) / 256, B), 256, 0, s>>>(g.tok_embd.qs, g.tok_embd.sc, X.btok, E, X.emb);
    k_rms_rows<<<B, 256, 0, s>>>(X.emb, g.tok_norm, X.cur, E, RMS_EPS);
    int nb = 0, prelude_nb = 0, bank_nb = 0;
    const int eb = (E + 255) / 256;
    static const int slot[4] = {0, 2, 3, 1};
    auto msg = [&](int j) { return X.msgs + (size_t) slot[j] * E; };

    auto mix_cur = [&](const float * query, int nq, float * out, int out_bs, int rep) {
        const int S = nb + 1;
        kb_mix_scores2<<<dim3(S, B), 256, 0, s>>>(X.blocks, MS * E, nb, X.cur, E, query, nq, X.scores, X.inv, S, E, RMS_EPS);
        kb_mix_out2<<<dim3(eb, nq, B), 256, 0, s>>>(X.blocks, MS * E, nb, X.cur, E, X.scores, X.inv, out, out_bs, rep, S, E);
    };
    auto control_b = [&](const Phys & P, const float * cur, int cur_bs, const float * m, int m_bs, int module, float * out, int out_bs) {
        kb_rms2_rows9<<<dim3(2, B), 256, 0, s>>>(cur, cur_bs, m, m_bs, X.hcat, E, RMS_EPS);
        gemvb_f32_row8<<<64, 256, 0, s>>>(P.loop_down[module], X.hcat, 2 * E, X.r64, 64, 2 * E, B);
        kb_ctrl_gf_warp<<<(E * 32 + 255) / 256, 256, 0, s>>>(cur, cur_bs, m, m_bs, P.loop_gain[module], P.loop_forget[module], X.r64,
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
            kb_copy<<<dim3(eb, B), 256, 0, s>>>(X.cur, E, X.prelude_cur, E, E);
        }
        bool have_msgs = false;
        if (in_core_range && pass > 0 && ip == CORE_START) {
            bank_nb = nb - (prelude_nb + 1);
            if (bank_nb > 0)
                kb_copy<<<dim3((bank_nb * E + 255) / 256, B), 256, 0, s>>>(X.blocks + (size_t) (prelude_nb + 1) * E, MS * E, X.bank, MS * E, bank_nb * E);
            kb_copy<<<dim3(eb, B), 256, 0, s>>>(X.cur, E, X.bank_last, E, E);
            nb = prelude_nb;
            kb_copy<<<dim3(eb, B), 256, 0, s>>>(X.prelude_cur, E, X.cur, E, E);
        }
        if (in_core_range && pass > 0) {
            have_msgs = true;
            const int row = 2 * (ip - CORE_START);
            const int Sb = bank_nb + 1;
            kb_mix_scores2<<<dim3(Sb, B), 256, 0, s>>>(X.bank, MS * E, bank_nb, X.bank_last, E, g.loop_res_query + (size_t) row * E, 2,
                                                      X.scores, X.inv, Sb, E, RMS_EPS);
            kb_mix_out2<<<dim3(eb, 2, B), 256, 0, s>>>(X.bank, MS * E, bank_nb, X.bank_last, E, X.scores, X.inv, msg(0), 4 * E, 1, Sb, E);
            kb_mix_scores2<<<dim3(Sb, B), 256, 0, s>>>(X.bank, MS * E, bank_nb, X.bank_last, E, g.loop_kv_res_query + (size_t) row * E, 2,
                                                      X.scores, X.inv, Sb, E, RMS_EPS);
            kb_mix_out2<<<dim3(eb, 2, B), 256, 0, s>>>(X.bank, MS * E, bank_nb, X.bank_last, E, X.scores, X.inv, msg(1), 4 * E, 1, Sb, E);
        }

        // ----- attention input
        if (ip > 0) {
            if (swa) mix_cur(P.attn_res_query, 1, X.inbuf, 3 * E, 3);
            else     mix_cur(P.attn_res_query, 3, X.inbuf, 3 * E, 1);
        } else {
            kb_rep3<<<dim3(eb, B), 256, 0, s>>>(X.cur, X.inbuf, E);
        }
        if (have_msgs)
            for (int j = 0; j < 3; ++j)
                control_b(P, X.inbuf + (size_t) j * E, 3 * E, msg(j), 4 * E, j, X.inbuf + (size_t) j * E, 3 * E);
        if (swa) kb_rms_rows3_9<<<3 * B, 256, 0, s>>>(X.inbuf, P.attn_norm, P.attn_norm, P.attn_norm, X.xs3, E, RMS_EPS);
        else     kb_rms_rows3_9<<<3 * B, 256, 0, s>>>(X.inbuf, P.q_inp, P.k_inp, P.v_inp, X.xs3, E, RMS_EPS);

        // ----- projections (K and V write straight into the history buffers, rows 3 + b)
        launch_gemvb_q8(P.wq.qs, P.wq.sc, X.xs3, 3 * E, X.Qv, NH * HD, E, NH * HD, B, s);
        launch_gemvb_q8(P.wk.qs, P.wk.sc, X.xs3 + E, 3 * E, X.Hk[il] + (size_t) 3 * NKV * HD, NKV * HD, E, NKV * HD, B, s);
        launch_gemvb_q8(P.wv.qs, P.wv.sc, X.xs3 + 2 * E, 3 * E, X.Hv[il] + (size_t) 3 * NKV * HD, NKV * HD, E, NKV * HD, B, s);
        launch_gemvb_q8(P.wrel.qs, P.wrel.sc, X.xs3, 3 * E, X.R0, NH * NREL, E, NH * NREL, B, s);
        launch_gemvb_q8(P.wgate.qs, P.wgate.sc, X.xs3, 3 * E, X.G, NH * HD, E, NH * HD, B, s);
        gemvb_bf16_row9<<<NH, 256, 0, s>>>(P.sink_w, X.xs3, 3 * E, X.sinks, NH, E, NH, B);

        // ----- convolutions
        if (!swa) {
            gemvb_bf16_row9<<<32, 256, 0, s>>>(P.kconv_proj, X.xs3 + E, 3 * E, X.dynk, 32, E, 32, B);
            gemvb_bf16_row9<<<32, 256, 0, s>>>(P.vconv_proj, X.xs3 + 2 * E, 3 * E, X.dynv, 32, E, 32, B);
        }
        kb_conv4_kv<<<dim3((NKV * HD + 255) / 256, 2), 256, 0, s>>>(X.Hk[il], g.kstate[il], P.kconv, X.dynk, X.Kconv,
                                                                  X.Hv[il], g.vstate[il], P.vconv, X.dynv, X.Vconv, NKV * HD, B, swa ? 0 : 1,
                                                                  multi, X.slot_arr, X.Sxk[il], X.Sxv[il]);

        // ----- clips, cache write at pos + b
        kb_clip_qr<<<dim3(2 * NH, B), 128, 0, s>>>(X.Qv, P.q_clip, X.Qc, HD, QK_CLIP, X.R0, P.rel_clip, X.Rc, NREL, REL_CLIP, NH);
        kb_kclip_vrms_pos<<<dim3(2 * NKV, B), 256, 0, s>>>(X.Kconv, g.kcache[il], QK_CLIP, X.Vconv, g.vcache[il], X.pos_arr, X.slot_arr, X.kx[il], X.vx[il], (size_t) MAXT * NKV * HD, NKV, HD, RMS_EPS);
        if (!swa) {
            kb_scale_tau<<<dim3((NH * HD + 255) / 256, B), 256, 0, s>>>(X.Qc, NH * HD, X.pos_arr);
            kb_scale_tau<<<dim3(2, B), 256, 0, s>>>(X.Rc, NH * NREL, X.pos_arr);
        }

        // ----- attention (query b sees positions <= pos + b), output projection into the post-conv history rows
        kb_attn2<<<dim3(NKV, B), 512, 0, s>>>(X.Qc, g.kcache[il], g.vcache[il], X.kx[il], X.vx[il], (size_t) MAXT * NKV * HD, X.Rc,
                                              P.rel_bias, X.sinks, X.pos_arr, X.slot_arr, swa ? NSWA : 0, X.O, X.G, X.gated);
        launch_gemvb_q8(P.wo.qs, P.wo.sc, X.gated, NH * HD, X.Hp[il] + (size_t) 3 * E, E, NH * HD, E, B, s);

        // ----- block residual rules
        if (ip % BLOCK == 0) {
            kb_copy<<<dim3(eb, B), 256, 0, s>>>(X.cur, E, X.blocks + (size_t) nb * E, MS * E, E);
            nb++;
            kb_conv4_res<<<eb, 256, 0, s>>>(X.Hp[il], g.poststate[il], P.post_conv, X.cur, X.cur, E, B, 0, multi, X.slot_arr, X.Sxp[il]);
        } else {
            kb_conv4_res<<<eb, 256, 0, s>>>(X.Hp[il], g.poststate[il], P.post_conv, X.cur, X.cur, E, B, 1, multi, X.slot_arr, X.Sxp[il]);
        }
        kb_copy<<<dim3(eb, B), 256, 0, s>>>(X.cur, E, X.ffn_inp, E, E);

        // ----- FFN input mix, loop control, norm
        mix_cur(P.ffn_res_query, 1, X.mixed, E, 1);
        if (have_msgs) control_b(P, X.mixed, E, msg(3), 4 * E, 3, X.mixed, E);
        kb_rms_rows9<<<B, 256, 0, s>>>(X.mixed, P.ffn_norm, X.xf, E, RMS_EPS);

        // ----- MoE: router, expert -> tokens grouping, grouped expert GEMVs
        gemvb_f32_row9<<<NEXP, 256, 0, s>>>(P.router_w, X.xf, E, X.lrouter, NEXP, E, NEXP, B);
        kb_router_top8<<<B, NEXP, 0, s>>>(X.lrouter, P.exp_bias, X.ids, X.wts, W_SCALE);
        kb_group<<<1, NEXP, 0, s>>>(X.ids, B, X.grp_e, X.grp_n, X.grp_t, X.grp_nu);
        const int nu_max = std::min(NEXP, NUSED * B);
        if (P.exp_q8) {
            q8_experts_rows2b<4><<<dim3(2304 / 128, nu_max), 128, 0, s>>>(tmap(P.gu_q8), tmap(P.gu_q8s), X.xf, E, 3, X.gu, E, 2304,
                                                                              X.grp_e, X.grp_n, X.grp_t, X.grp_nu);
        } else {
            launch_nvfp4_nch<2304>(P.gu_qs, P.gu_sc, X.xf, E, 3, X.gu, 2304, B, X.grp_e, X.grp_n, X.grp_t, X.grp_nu, s);
        }
        kb_situ<<<dim3((FFN + 255) / 256, NUSED * B), 256, 0, s>>>(X.gu, P.gu_scale ? P.gu_scale : g.ones, X.ids, X.hbuf, SITU_BETA, SITU_LB);
        if (P.exp_q8) {
            q8_experts_rows2b<4><<<dim3(E / 128, nu_max), 128, 0, s>>>(tmap(P.dn_q8), tmap(P.dn_q8s), X.hbuf, FFN, 0, X.dn, FFN, E,
                                                                           X.grp_e, X.grp_n, X.grp_t, X.grp_nu);
        } else {
            launch_nvfp4_nch<1152>(P.dn_qs, P.dn_sc, X.hbuf, FFN, 0, X.dn, E, B, X.grp_e, X.grp_n, X.grp_t, X.grp_nu, s);
        }
        kb_moe_conv_res<<<(E + 63) / 64, 64, 0, s>>>(X.dn, P.dn_scale ? P.dn_scale : g.ones, X.ids, X.wts, X.Hf[il], g.ffnstate[il],
                                                    P.ffn_post_conv, X.ffn_inp, X.cur, E, B, multi, X.slot_arr, X.Sxf[il]);
    }

    // ----- output
    mix_cur(g.output_res_query, 1, X.head_in, E, 1);
    kb_rms_rows9<<<B, 256, 0, s>>>(X.head_in, g.output_norm, X.xnorm, E, RMS_EPS);
    launch_gemvb_q8(g.head.qs, g.head.sc, X.xnorm, E, X.logits, NVOCAB_PAD, E, NVOCAB_PAD, B, s);
    kb_softcap_cut<<<dim3((NVOCAB + 255) / 256, B), 256, 0, s>>>(X.logits, NVOCAB, NVOCAB_PAD, SOFTCAP);
    kb_argmax<<<B, 1024, 0, s>>>(X.logits, NVOCAB, NVOCAB_PAD, X.barg);
}

// Rewind the carried conv states to "after j tokens of the last forward_batch" (0 <= j <= B). KV entries at positions
// >= pos + j are stale but get overwritten by later steps and are never read before (causal limit).
static void batch_rollback(Engine & g, BatchState & X, int j, cudaStream_t s) {
    (void) g;
    if (std::getenv("MIKASA_NEG_CONTROL")) return;   // negative control for the test: skip the rollback
    CHECK(cudaMemcpyAsync(X.rb_j, &j, sizeof(int), cudaMemcpyHostToDevice, s));
    kb_rollback<<<dim3((E + 255) / 256, NAPP * 4), 256, 0, s>>>(X.rb_tab, X.rb_j);
}

// ------------------------------------------------------------------ tests ----

static std::vector<int> read_tokens(const std::string & path) {
    std::vector<int> v;
    FILE * f = std::fopen(path.c_str(), "r");
    if (!f) { std::fprintf(stderr, "cannot open %s\n", path.c_str()); std::exit(1); }
    int x;
    while (std::fscanf(f, "%d", &x) == 1) { v.push_back(x); int c = std::fgetc(f); if (c == EOF) break; }
    std::fclose(f);
    return v;
}

static void reset_extra_slots(Engine & g, BatchState & X) {
    for (int il = 0; il < NAPP; ++il) {
        CHECK(cudaMemset(X.Sxk[il], 0, 3 * 3 * NKV * HD * sizeof(float))); CHECK(cudaMemset(X.Sxv[il], 0, 3 * 3 * NKV * HD * sizeof(float)));
        CHECK(cudaMemset(X.Sxp[il], 0, 3 * 3 * E * sizeof(float))); CHECK(cudaMemset(X.Sxf[il], 0, 3 * 3 * E * sizeof(float)));
    }
    (void) g;
}

static void reset_state(Engine & g) {
    for (int il = 0; il < NAPP; ++il) {
        CHECK(cudaMemset(g.kstate[il], 0, 3 * NKV * HD * sizeof(float)));
        CHECK(cudaMemset(g.vstate[il], 0, 3 * NKV * HD * sizeof(float)));
        CHECK(cudaMemset(g.poststate[il], 0, 3 * E * sizeof(float)));
        CHECK(cudaMemset(g.ffnstate[il], 0, 3 * E * sizeof(float)));
    }
}

struct SavedState { std::vector<float *> bufs; std::vector<size_t> n; };
static void snapshot_state(Engine & g, SavedState & st, bool save) {
    if (st.bufs.empty()) {
        for (int il = 0; il < NAPP; ++il) {
            float * src[4] = {g.kstate[il], g.vstate[il], g.poststate[il], g.ffnstate[il]};
            size_t cnt[4] = {3 * NKV * HD, 3 * NKV * HD, 3 * E, 3 * E};
            for (int k = 0; k < 4; ++k) { float * d; CHECK(cudaMalloc(&d, cnt[k] * 4)); st.bufs.push_back(d); st.n.push_back(cnt[k]); }
        }
    }
    for (int il = 0; il < NAPP; ++il) {
        float * src[4] = {g.kstate[il], g.vstate[il], g.poststate[il], g.ffnstate[il]};
        for (int k = 0; k < 4; ++k) {
            float * a = st.bufs[il * 4 + k];
            if (save) CHECK(cudaMemcpy(a, src[k], st.n[il * 4 + k] * 4, cudaMemcpyDeviceToDevice));
            else CHECK(cudaMemcpy(src[k], a, st.n[il * 4 + k] * 4, cudaMemcpyDeviceToDevice));
        }
    }
}

struct Runner {
    Engine & g;
    BatchState & X;
    cudaStream_t s;
    cudaGraphExec_t single = nullptr;
    std::map<int, cudaGraphExec_t> bexec;   // key = B + 100 * multi
    Runner(Engine & g_, BatchState & X_, cudaStream_t s_) : g(g_), X(X_), s(s_) {}
    cudaGraphExec_t capture_single() {
        if (single) return single;
        Dumper none;
        cudaGraph_t gr;
        CHECK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
        forward_token(g, none, s);
        k_argmax<<<1, 1024, 0, s>>>(g.logits, NVOCAB, g.tok_dev);
        CHECK(cudaStreamEndCapture(s, &gr));
        CHECK(cudaGraphInstantiate(&single, gr, nullptr, nullptr, 0));
        return single;
    }
    cudaGraphExec_t capture_batch(int B, int multi = 0) {
        auto it = bexec.find(B + 100 * multi);
        if (it != bexec.end()) return it->second;
        cudaGraph_t gr;
        cudaGraphExec_t ex;
        CHECK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
        forward_batch(g, X, B, multi, s);
        CHECK(cudaStreamEndCapture(s, &gr));
        CHECK(cudaGraphInstantiate(&ex, gr, nullptr, nullptr, 0));
        bexec[B + 100 * multi] = ex;
        return ex;
    }
    // one sequential step at position pos; logits (NVOCAB) copied to out if non-null
    void step(int pos, int tok, float * out) {
        cudaGraphExec_t ex = capture_single();
        set_pos_tok(g, pos, tok, s);
        CHECK(cudaGraphLaunch(ex, s));
        if (out) CHECK(cudaMemcpyAsync(out, g.logits, NVOCAB * sizeof(float), cudaMemcpyDeviceToHost, s));
        CHECK(cudaStreamSynchronize(s));
    }
    // multi-sequence step: token k of sequence slot slots[k] at position poss[k]
    void multi(const int * toks, const int * poss, const int * slots, int B, float * out) {
        cudaGraphExec_t ex = capture_batch(B, 1);
        CHECK(cudaMemcpyAsync(X.btok, toks, B * sizeof(int), cudaMemcpyHostToDevice, s));
        CHECK(cudaMemcpyAsync(X.pos_arr, poss, B * sizeof(int), cudaMemcpyHostToDevice, s));
        CHECK(cudaMemcpyAsync(X.slot_arr, slots, B * sizeof(int), cudaMemcpyHostToDevice, s));
        CHECK(cudaGraphLaunch(ex, s));
        if (out) CHECK(cudaMemcpy2DAsync(out, NVOCAB * sizeof(float), X.logits, NVOCAB_PAD * sizeof(float), NVOCAB * sizeof(float), B,
                                         cudaMemcpyDeviceToHost, s));
        CHECK(cudaStreamSynchronize(s));
    }
    void batch(int pos, const int * toks, int B, float * out /* [B][NVOCAB] or null */) {
        cudaGraphExec_t ex = capture_batch(B);
        CHECK(cudaMemcpyAsync(X.btok, toks, B * sizeof(int), cudaMemcpyHostToDevice, s));
        int pa[MB], sa[MB] = {0};
        for (int k = 0; k < B; ++k) pa[k] = pos + k;
        CHECK(cudaMemcpyAsync(X.pos_arr, pa, B * sizeof(int), cudaMemcpyHostToDevice, s));
        CHECK(cudaMemcpyAsync(X.slot_arr, sa, B * sizeof(int), cudaMemcpyHostToDevice, s));
        CHECK(cudaGraphLaunch(ex, s));
        if (out) CHECK(cudaMemcpy2DAsync(out, NVOCAB * sizeof(float), X.logits, NVOCAB_PAD * sizeof(float), NVOCAB * sizeof(float), B,
                                         cudaMemcpyDeviceToHost, s));
        CHECK(cudaStreamSynchronize(s));
    }
};

struct Cmp {
    double max_abs = 0, max_rel = 0;
    long n = 0, argmax_bad = 0;
    void add(const float * a, const float * ref) {
        double ma = 0, mr = 0, mx = 0;
        int ia = 0, ir = 0;
        for (int i = 0; i < NVOCAB; ++i) {
            ma = std::max(ma, (double) std::fabs(a[i] - ref[i]));
            mx = std::max(mx, (double) std::fabs(ref[i]));
            if (a[i] > a[ia]) ia = i;
            if (ref[i] > ref[ir]) ir = i;
        }
        mr = ma / mx;
        max_abs = std::max(max_abs, ma);
        max_rel = std::max(max_rel, mr);
        argmax_bad += (ia != ir);
        ++n;
    }
};

static int run_batch_test(Engine & g, const std::string & dir, cudaStream_t s) {
    BatchState X;
    init_batch(g, X);
    Runner R(g, X, s);
    const int Bs[5] = {1, 2, 4, 8, 16};
    bool all_ok = true;
    SavedState saved;
    std::vector<std::vector<int>> seqs(3);
    std::vector<std::vector<float>> refs(3);
    for (int pi = 1; pi <= 3; ++pi) {
        std::vector<int> seq = read_tokens(dir + "/seq" + std::to_string(pi) + ".txt");
        seqs[pi - 1] = seq;
        const int T = (int) seq.size();
        std::printf("== sequence %d: %d tokens\n", pi, T);
        std::vector<float> ref((size_t) T * NVOCAB);
        reset_state(g);
        auto t0 = std::chrono::steady_clock::now();
        for (int t = 0; t < T; ++t) R.step(t, seq[t], &ref[(size_t) t * NVOCAB]);
        std::printf("  sequential reference done (%.1f s)\n", std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
        std::vector<float> out((size_t) MB * NVOCAB);
        for (int B : Bs) {
            reset_state(g);
            Cmp c;
            for (int p = 0; p < T; p += B) {
                const int b = std::min(B, T - p);
                R.batch(p, &seq[p], b, out.data());
                for (int k = 0; k < b; ++k) c.add(&out[(size_t) k * NVOCAB], &ref[(size_t) (p + k) * NVOCAB]);
            }
            std::printf("  chunks B=%-2d: positions %ld  max abs diff %.3e  max rel diff %.3e  argmax mismatches %ld\n", B, c.n, c.max_abs, c.max_rel, c.argmax_bad);
            if (c.argmax_bad) all_ok = false;
        }
        // rollback test: B = 8 batch with a real prefix of j tokens and garbage after, rollback to j, continue singly.
        reset_state(g);
        const int ps[3] = {40, 120, 200};
        int pidx = 0;
        for (int t = 0; t <= T && pidx < 3; ++t) {
            if (t == ps[pidx]) {
                snapshot_state(g, saved, true);
                for (int j : {0, 3, 8}) {
                    int toks[8];
                    for (int k = 0; k < 8; ++k) toks[k] = (k < j) ? seq[t + k] : (int) ((seq[(t + k) % T] * 7u + 13u + k) % 200000u);
                    Cmp cb, cc;
                    R.batch(t, toks, 8, out.data());
                    for (int k = 0; k < j; ++k) cb.add(&out[(size_t) k * NVOCAB], &ref[(size_t) (t + k) * NVOCAB]);
                    batch_rollback(g, X, j, s);
                    CHECK(cudaStreamSynchronize(s));
                    std::vector<float> o1(NVOCAB);
                    for (int k = 0; k < 6; ++k) {      // continue with single steps
                        R.step(t + j + k, seq[t + j + k], o1.data());
                        cc.add(o1.data(), &ref[(size_t) (t + j + k) * NVOCAB]);
                    }
                    for (int k = 6; k < 10; k += 4) {   // then a batch of 4 (state must still be consistent)
                        R.batch(t + j + k, &seq[t + j + k], 4, out.data());
                        for (int q = 0; q < 4; ++q) cc.add(&out[(size_t) q * NVOCAB], &ref[(size_t) (t + j + k + q) * NVOCAB]);
                    }
                    std::printf("  rollback p=%-3d j=%d: batch prefix logits (%ld pos) abs %.3e rel %.3e | continuation (%ld pos) abs %.3e rel %.3e, argmax mismatches %ld\n",
                                t, j, cb.n, cb.max_abs, cb.max_rel, cc.n, cc.max_abs, cc.max_rel, cb.argmax_bad + cc.argmax_bad);
                    if (cb.argmax_bad + cc.argmax_bad) all_ok = false;
                    snapshot_state(g, saved, false);
                }
                ++pidx;
            }
            if (t < T) R.step(t, seq[t], nullptr);
        }
        refs[pi - 1] = std::move(ref);
    }
    // ---- multi-sequence mode: S independent sequences, one token each per step, staggered starts (different positions)
    {
        const int fidx[4] = {0, 1, 2, 0};          // slot -> sequence file (slot 3 repeats prompt 1 with a different offset)
        for (int S = 2; S <= 4; ++S) {
            const int start[4] = {0, 30, 70, 110};
            reset_state(g);
            reset_extra_slots(g, X);
            Cmp cs[4];
            std::vector<float> out((size_t) MB * NVOCAB);
            int step = 0, remaining = S;
            long steps_with_B[5] = {0};
            while (remaining > 0) {
                int toks[4], poss[4], slots[4], B = 0;
                remaining = 0;
                for (int i = 0; i < S; ++i) {
                    const int pos = step - start[i];
                    const int T = (int) seqs[fidx[i]].size();
                    if (pos >= 0 && pos < T) { toks[B] = seqs[fidx[i]][pos]; poss[B] = pos; slots[B] = i; ++B; }
                    if (pos < T - 1) ++remaining;
                }
                if (B > 0) {
                    R.multi(toks, poss, slots, B, out.data());
                    steps_with_B[B]++;
                    for (int k = 0; k < B; ++k) cs[slots[k]].add(&out[(size_t) k * NVOCAB], &refs[fidx[slots[k]]][(size_t) poss[k] * NVOCAB]);
                }
                ++step;
            }
            std::printf("== multi-sequence S=%d (steps with B=1..4 tokens: %ld %ld %ld %ld)\n", S, steps_with_B[1], steps_with_B[2], steps_with_B[3], steps_with_B[4]);
            for (int i = 0; i < S; ++i) {
                std::printf("  slot %d (prompt file %d, start step %d): positions %ld  max abs diff %.3e  max rel diff %.3e  argmax mismatches %ld\n",
                            i, fidx[i] + 1, start[i], cs[i].n, cs[i].max_abs, cs[i].max_rel, cs[i].argmax_bad);
                if (cs[i].argmax_bad) all_ok = false;
            }
        }
    }
    std::printf("%s\n", all_ok ? "ORACLE TEST: argmax matches at every position" : "ORACLE TEST: FAILED (argmax mismatches)");
    return all_ok ? 0 : 1;
}

static int run_batch_bench(Engine & g, const std::string & dir, cudaStream_t s, int only_b, int pos0) {
    BatchState X;
    init_batch(g, X);
    Runner R(g, X, s);
    std::vector<int> seq = read_tokens(dir + "/seq1.txt");
    reset_state(g);
    int p = 0;
    const int chunk = only_b ? only_b : 16;
    for (; p + chunk <= pos0; p += chunk) R.batch(p, &seq[p], chunk, nullptr);   // real context in the KV cache
    for (; p < pos0; ++p) R.step(p, seq[p], nullptr);
    std::printf("bench at position %d (context filled with real tokens)\n", pos0);
    cudaEvent_t e0, e1;
    CHECK(cudaEventCreate(&e0));
    CHECK(cudaEventCreate(&e1));
    const int reps = 100;
    float ms = 0;
    if (!only_b) {
        cudaGraphExec_t ex = R.capture_single();
        set_pos_tok(g, pos0, seq[pos0], s);
        for (int i = 0; i < 10; ++i) CHECK(cudaGraphLaunch(ex, s));
        CHECK(cudaEventRecord(e0, s));
        for (int i = 0; i < reps; ++i) CHECK(cudaGraphLaunch(ex, s));
        CHECK(cudaEventRecord(e1, s));
        CHECK(cudaEventSynchronize(e1));
        CHECK(cudaEventElapsedTime(&ms, e0, e1));
        std::printf("forward_token (graph, B=1 reference): %.3f ms/token\n", ms / reps);
    }
    std::printf("%4s %12s %12s %10s\n", "B", "ms/call", "ms/token", "speedup");
    double t1 = 0;
    for (int B : {1, 2, 4, 8, 16}) {
        if (only_b && B != only_b) continue;
        cudaGraphExec_t ex = R.capture_batch(B);
        CHECK(cudaMemcpy(X.btok, &seq[pos0], B * sizeof(int), cudaMemcpyHostToDevice));
        { int pa[MB], sa[MB] = {0}; for (int k = 0; k < B; ++k) pa[k] = pos0 + k;
          CHECK(cudaMemcpy(X.pos_arr, pa, sizeof(pa), cudaMemcpyHostToDevice)); CHECK(cudaMemcpy(X.slot_arr, sa, sizeof(sa), cudaMemcpyHostToDevice)); }
        for (int i = 0; i < 10; ++i) CHECK(cudaGraphLaunch(ex, s));
        CHECK(cudaEventRecord(e0, s));
        for (int i = 0; i < reps; ++i) CHECK(cudaGraphLaunch(ex, s));
        CHECK(cudaEventRecord(e1, s));
        CHECK(cudaEventSynchronize(e1));
        CHECK(cudaEventElapsedTime(&ms, e0, e1));
        const double call = ms / reps;
        if (B == 1) t1 = call;
        std::printf("%4d %12.3f %12.3f %9.2fx\n", B, call, call / B, t1 > 0 ? t1 * B / call : 0.0);
    }
    if (!only_b) {
        std::printf("multi-sequence decode (S sequences x 1 token per step, own positions/KV/state; context %d tokens each)\n", pos0);
        std::vector<int> sq[3] = {seq, read_tokens(dir + "/seq2.txt"), read_tokens(dir + "/seq3.txt")};
        std::printf("%4s %12s %14s %12s\n", "S", "ms/step", "agg tok/s", "vs 1-seq");
        double base = 0;
        for (int S = 1; S <= 4; ++S) {
            reset_state(g);
            reset_extra_slots(g, X);
            int toks[4], poss[4], slots[4];
            for (int t = 0; t < pos0; ++t) {
                for (int i = 0; i < S; ++i) { toks[i] = sq[i % 3][t]; poss[i] = t; slots[i] = i; }
                R.multi(toks, poss, slots, S, nullptr);
            }
            for (int i = 0; i < S; ++i) { toks[i] = sq[i % 3][pos0]; poss[i] = pos0; slots[i] = i; }
            cudaGraphExec_t ex = R.capture_batch(S, 1);
            CHECK(cudaMemcpy(X.btok, toks, sizeof(toks), cudaMemcpyHostToDevice));
            CHECK(cudaMemcpy(X.pos_arr, poss, sizeof(poss), cudaMemcpyHostToDevice));
            CHECK(cudaMemcpy(X.slot_arr, slots, sizeof(slots), cudaMemcpyHostToDevice));
            for (int i = 0; i < 10; ++i) CHECK(cudaGraphLaunch(ex, s));
            CHECK(cudaEventRecord(e0, s));
            for (int i = 0; i < reps; ++i) CHECK(cudaGraphLaunch(ex, s));
            CHECK(cudaEventRecord(e1, s));
            CHECK(cudaEventSynchronize(e1));
            CHECK(cudaEventElapsedTime(&ms, e0, e1));
            const double step = ms / reps;
            if (S == 1) base = 1000.0 / step;
            std::printf("%4d %12.3f %14.1f %11.2fx\n", S, step, S * 1000.0 / step, S * 1000.0 / step / base);
        }
    }
    return 0;
}
