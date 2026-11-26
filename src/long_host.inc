// Long-context drivers (textually included after batch_host.inc): prefill(), greedy generation, validation, timings.
// All of this works in lean mode (no forward_token): only forward_batch is used.

struct LongArgs {
    SpecArgs spec;
    bool multi_check = false, lean = false, kv16 = true, ref_all = false, time_decode = false, time_prefill = false, time_b = false;
    int maxctx = 65536, nslots = 4, ring = NSWA + MB, tail = 0, cmp_ref = 0, gen = 0, pf_b = 16, prefill_slot = 0, pf_fast = 0, layer_cmp = 0; bool pf_all = false; int pf_split = 0; std::string time_depths; std::string force_file, gen_logits;
    std::string prefill_file, tail_out, argmax_out, gen_out, ring_note;
};

static double now_s() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

static void print_mem(const char * tag) {
    size_t fr = 0, tot = 0;
    CHECK(cudaMemGetInfo(&fr, &tot));
    std::printf("[mem] %-34s used %.2f GB (free %.2f GB of %.2f GB)\n", tag, (tot - fr) / 1e9, fr / 1e9, tot / 1e9);
    std::fflush(stdout);
}

// Exchange the carried conv states (the "slot 0" state that forward_batch uses in same-sequence mode) with slot sl (1..3).
static void swap_conv_state(Engine & g, BatchState & X, int sl, cudaStream_t s) {
    if (sl == 0) return;
    float * tmp = nullptr;
    CHECK(cudaMalloc(&tmp, 3 * E * sizeof(float)));
    for (int il = 0; il < NAPP; ++il) {
        float * a[4] = {g.kstate[il], g.vstate[il], g.poststate[il], g.ffnstate[il]};
        float * b[4] = {X.Sxk[il], X.Sxv[il], X.Sxp[il], X.Sxf[il]};
        const size_t n[4] = {3 * NKV * HD, 3 * NKV * HD, 3 * E, 3 * E};
        for (int k = 0; k < 4; ++k) {
            float * bs = b[k] + (size_t) (sl - 1) * n[k];
            CHECK(cudaMemcpyAsync(tmp, a[k], n[k] * 4, cudaMemcpyDeviceToDevice, s));
            CHECK(cudaMemcpyAsync(a[k], bs, n[k] * 4, cudaMemcpyDeviceToDevice, s));
            CHECK(cudaMemcpyAsync(bs, tmp, n[k] * 4, cudaMemcpyDeviceToDevice, s));
        }
    }
    CHECK(cudaStreamSynchronize(s));
    CHECK(cudaFree(tmp));
}

// prefill(slot, tokens[], n): run the n tokens of sequence `slot` at positions start..start+n-1 through forward_batch in chunks of
// chunk (<= MB) tokens. argmax_out (optional, host, n ints): greedy next-token prediction after every token.
// tail_logits (optional): the logits of the last tail_k tokens are copied to host ([tail_k][NVOCAB]).
// Returns tokens per second (GPU time, events).
static double prefill(Engine & g, BatchState & X, Runner & R, int slot, const int * toks, int n, int start, int chunk,
                      int * argmax_out, std::vector<float> * tail_logits, int tail_k, cudaStream_t s) {
    if (start + n > X.maxctx || slot >= X.nslots || chunk > MB) { std::fprintf(stderr, "prefill: out of range\n"); std::exit(1); }
    swap_conv_state(g, X, slot, s);
    int *ptok, *ppos, *pslot, *parg;
    CHECK(cudaMallocHost(&ptok, n * sizeof(int))); CHECK(cudaMallocHost(&ppos, n * sizeof(int)));
    CHECK(cudaMallocHost(&pslot, n * sizeof(int))); CHECK(cudaMallocHost(&parg, n * sizeof(int)));
    for (int i = 0; i < n; ++i) { ptok[i] = toks[i]; ppos[i] = start + i; pslot[i] = slot; }
    // capture the graphs first (not timed)
    for (int p = 0; p < n; p += chunk) R.capture_batch(std::min(chunk, n - p), 0);
    if (tail_logits) tail_logits->assign((size_t) tail_k * NVOCAB, 0.0f);
    cudaEvent_t e0, e1;
    CHECK(cudaEventCreate(&e0)); CHECK(cudaEventCreate(&e1));
    CHECK(cudaStreamSynchronize(s));
    CHECK(cudaEventRecord(e0, s));
    for (int p = 0; p < n; p += chunk) {
        const int b = std::min(chunk, n - p);
        CHECK(cudaMemcpyAsync(X.btok, ptok + p, b * sizeof(int), cudaMemcpyHostToDevice, s));
        CHECK(cudaMemcpyAsync(X.pos_arr, ppos + p, b * sizeof(int), cudaMemcpyHostToDevice, s));
        CHECK(cudaMemcpyAsync(X.slot_arr, pslot + p, b * sizeof(int), cudaMemcpyHostToDevice, s));
        CHECK(cudaGraphLaunch(R.capture_batch(b, 0), s));
        CHECK(cudaMemcpyAsync(parg + p, X.barg, b * sizeof(int), cudaMemcpyDeviceToHost, s));
        if (tail_logits && p + b > n - tail_k) {
            for (int k = 0; k < b; ++k) {
                const int idx = p + k;
                if (idx >= n - tail_k)
                    CHECK(cudaMemcpyAsync(tail_logits->data() + (size_t) (idx - (n - tail_k)) * NVOCAB, X.logits + (size_t) k * NVOCAB_PAD,
                                          NVOCAB * sizeof(float), cudaMemcpyDeviceToHost, s));
            }
        }
    }
    CHECK(cudaEventRecord(e1, s));
    CHECK(cudaEventSynchronize(e1));
    float ms = 0;
    CHECK(cudaEventElapsedTime(&ms, e0, e1));
    swap_conv_state(g, X, slot, s);
    if (argmax_out) std::memcpy(argmax_out, parg, n * sizeof(int));
    CHECK(cudaFreeHost(ptok)); CHECK(cudaFreeHost(ppos)); CHECK(cudaFreeHost(pslot)); CHECK(cudaFreeHost(parg));
    return n / (ms / 1000.0);
}

static void compare_logits(const float * a, const float * b, int K, const char * tag) {
    double ma = 0, mr = 0, sum_rel = 0;
    int bad = 0;
    for (int k = 0; k < K; ++k) {
        const float * x = a + (size_t) k * NVOCAB, * y = b + (size_t) k * NVOCAB;
        double d = 0, mx = 0;
        int ix = 0, iy = 0;
        for (int i = 0; i < NVOCAB; ++i) {
            d = std::max(d, (double) std::fabs(x[i] - y[i]));
            mx = std::max(mx, (double) std::fabs(y[i]));
            if (x[i] > x[ix]) ix = i;
            if (y[i] > y[iy]) iy = i;
        }
        ma = std::max(ma, d); mr = std::max(mr, d / mx); sum_rel += d / mx;
        bad += (ix != iy);
    }
    std::printf("%s: %d positions, max abs diff %.4e, max rel diff %.4e (mean %.4e), argmax mismatches %d\n", tag, K, ma, mr, sum_rel / K, bad);
}

static int run_cmp_dump(const std::string & a, const std::string & b, int K) {
    std::vector<float> x((size_t) K * NVOCAB), y((size_t) K * NVOCAB);
    FILE * fa = std::fopen(a.c_str(), "rb"), * fb = std::fopen(b.c_str(), "rb");
    if (!fa || !fb) { std::fprintf(stderr, "cannot open dumps\n"); return 1; }
    if (std::fread(x.data(), 4, x.size(), fa) != x.size() || std::fread(y.data(), 4, y.size(), fb) != y.size()) { std::fprintf(stderr, "short dump\n"); return 1; }
    std::fclose(fa); std::fclose(fb);
    compare_logits(x.data(), y.data(), K, ("compare " + a + " vs " + b).c_str());
    return 0;
}

static int run_long(Engine & g, const LongArgs & A, cudaStream_t s) {
    print_mem("weights loaded");
    BatchState X;
    KVOpts ko;
    ko.kv16 = A.kv16; ko.maxctx = A.maxctx; ko.nslots = A.nslots; ko.ring = A.ring; ko.alias0 = !A.lean;
    ko.ref = A.ref_all || A.cmp_ref > 0;
    if (A.pf_fast) ko.ring = std::max(ko.ring, NSWA + A.pf_fast);
    if (!A.lean) { ko.alias0 = false; }   // long runs never share the engine KV
    init_batch(g, X, ko);
    X.ref = A.ref_all;
    print_mem("after init_batch (KV + buffers)");
    Runner R(g, X, s);
    PFState PF;
    if (A.pf_fast) { init_pf(PF, A.pf_fast); print_mem("after init_pf"); }
    std::printf("config: lean=%d kv=%s maxctx=%d nslots=%d ring=%d ref_all=%d\n", (int) A.lean, A.kv16 ? "fp16(full layers)" : "fp32", A.maxctx, A.nslots, A.ring, (int) A.ref_all);

    if (A.layer_cmp > 0 && !A.prefill_file.empty()) {
        // layer-by-layer comparison of the residual stream: baseline forward_batch (B = 16, eager) vs forward_prefill (one chunk)
        std::vector<int> toks = read_tokens(A.prefill_file);
        const int N = A.layer_cmp;
        std::vector<std::vector<float>> D1, D2;
        g_dump_cur = &D1; g_dump_n = N;
        for (int p = 0; p < N; p += 16) {
            const int b = std::min(16, N - p);
            int pa[MB], sa[MB] = {0};
            for (int k = 0; k < b; ++k) pa[k] = p + k;
            CHECK(cudaMemcpy(X.btok, &toks[p], b * 4, cudaMemcpyHostToDevice));
            CHECK(cudaMemcpy(X.pos_arr, pa, b * 4, cudaMemcpyHostToDevice));
            CHECK(cudaMemcpy(X.slot_arr, sa, b * 4, cudaMemcpyHostToDevice));
            g_dump_off = p;
            forward_batch(g, X, b, 0, s);
            CHECK(cudaStreamSynchronize(s));
        }
        reset_state(g);
        g_dump_cur = &D2; g_dump_off = 0;
        int la = -1;
        if (std::getenv("MIKASA_CMP_SELF")) {   // second run = baseline again with FP16-rounded K/V (kv32 only): perturbation calibration
            g_kv_rmask = 3;
            for (int p = 0; p < N; p += 16) {
                const int b = std::min(16, N - p);
                int pa[MB], sa[MB] = {0};
                for (int k = 0; k < b; ++k) pa[k] = p + k;
                CHECK(cudaMemcpy(X.btok, &toks[p], b * 4, cudaMemcpyHostToDevice));
                CHECK(cudaMemcpy(X.pos_arr, pa, b * 4, cudaMemcpyHostToDevice));
                CHECK(cudaMemcpy(X.slot_arr, sa, b * 4, cudaMemcpyHostToDevice));
                g_dump_off = p;
                forward_batch(g, X, b, 0, s);
                CHECK(cudaStreamSynchronize(s));
            }
        } else
        prefill_fast(g, X, PF, 0, toks.data(), N, 0, 1, nullptr, &la, s);
        g_dump_cur = nullptr;
        for (int il = 0; il < NAPP; ++il) {
            double num = 0, den = 0, worst = 0; int wt = 0;
            for (int t = 0; t < N; ++t) {
                double n2 = 0, d2 = 0;
                for (int c = 0; c < E; ++c) { const double a = D1[il][(size_t) t * E + c], b2 = D2[il][(size_t) t * E + c]; n2 += (a - b2) * (a - b2); d2 += a * a; }
                num += n2; den += d2;
                if (std::sqrt(n2 / d2) > worst) { worst = std::sqrt(n2 / d2); wt = t; }
            }
            std::printf("layer %2d (%s): relL2 over %d tokens %.3e, worst token %d: %.3e\n", il, app_is_swa(il) ? "swa " : "full", N, std::sqrt(num / den), wt, worst);
        }
        return 0;
    }
    if (!A.prefill_file.empty()) {
        std::vector<int> toks = read_tokens(A.prefill_file);
        const int N = (int) toks.size();
        std::printf("prefill file %s: %d tokens (slot %d)\n", A.prefill_file.c_str(), N, A.prefill_slot);
        std::vector<int> arg(N);
        std::vector<float> tail;
        const int tk = std::max(A.tail, 1);
        int Npre = N;
        if (A.cmp_ref > 0) Npre = N - A.cmp_ref;
        for (int sl = 0; sl < 4; ++sl) if (sl == A.prefill_slot) {}
        double tps;
        if (A.pf_fast) {
            int la = -1;
            std::fill(arg.begin(), arg.end(), -1);
            if (A.pf_split > 0 && A.pf_split < Npre) {   // two prefill calls: the second continues the carried state (start = split)
                const double t1 = prefill_fast(g, X, PF, A.prefill_slot, toks.data(), A.pf_split, 0, tk, nullptr, &la, s, A.pf_all ? arg.data() : nullptr);
                const double t2 = prefill_fast(g, X, PF, A.prefill_slot, toks.data() + A.pf_split, Npre - A.pf_split, A.pf_split, tk, (A.cmp_ref > 0 || A.tail_out.empty()) ? nullptr : &tail, &la, s, A.pf_all ? arg.data() + A.pf_split : nullptr);
                tps = Npre / (A.pf_split / t1 + (Npre - A.pf_split) / t2);
            } else
            tps = prefill_fast(g, X, PF, A.prefill_slot, toks.data(), Npre, 0, tk, (A.cmp_ref > 0 || A.tail_out.empty()) ? nullptr : &tail, &la, s, A.pf_all ? arg.data() : nullptr);
            if (!A.pf_all) arg[Npre - 1] = la; else la = arg[Npre - 1];
            std::printf("fast prefill: last-token argmax %d\n", la);
        } else tps = prefill(g, X, R, A.prefill_slot, toks.data(), Npre, 0, A.pf_b, arg.data(), (A.cmp_ref > 0 || A.tail_out.empty()) ? nullptr : &tail, tk, s);
        std::printf("prefill: %d tokens, %.1f tokens/s (chunk B=%d, GPU time)\n", Npre, tps, A.pf_fast ? A.pf_fast : A.pf_b);
        print_mem("after prefill");
        if (A.cmp_ref > 0) {
            // kernel-level validation: same KV prefix, the last cmp_ref tokens with optimised vs reference attention
            const int K = A.cmp_ref;
            if (A.prefill_slot != 0) { std::fprintf(stderr, "cmp-ref needs slot 0\n"); return 1; }
            std::vector<float> lnew((size_t) K * NVOCAB), lref((size_t) K * NVOCAB);
            int pa[MB], sa[MB] = {0};
            for (int k = 0; k < K; ++k) pa[k] = Npre + k;
            auto run_chunk = [&](bool ref, std::vector<float> & out) {
                X.ref = ref;
                cudaGraphExec_t ex = R.capture_batch(K, 0);
                CHECK(cudaMemcpy(X.btok, &toks[Npre], K * sizeof(int), cudaMemcpyHostToDevice));
                CHECK(cudaMemcpy(X.pos_arr, pa, K * sizeof(int), cudaMemcpyHostToDevice));
                CHECK(cudaMemcpy(X.slot_arr, sa, K * sizeof(int), cudaMemcpyHostToDevice));
                CHECK(cudaGraphLaunch(ex, s));
                CHECK(cudaMemcpy2DAsync(out.data(), NVOCAB * sizeof(float), X.logits, NVOCAB_PAD * sizeof(float), NVOCAB * sizeof(float), K, cudaMemcpyDeviceToHost, s));
                CHECK(cudaStreamSynchronize(s));
            };
            run_chunk(false, lnew);
            if (!A.tail_out.empty()) { FILE * f = std::fopen(A.tail_out.c_str(), "wb"); std::fwrite(lnew.data(), 4, lnew.size(), f); std::fclose(f); }
            batch_rollback(g, X, 0, s);
            CHECK(cudaStreamSynchronize(s));
            double t0 = now_s();
            run_chunk(true, lref);
            std::printf("reference chunk (B=%d at positions %d..%d) took %.2f s\n", K, Npre, Npre + K - 1, now_s() - t0);
            compare_logits(lnew.data(), lref.data(), K, "optimised vs naive reference attention (same KV prefix)");
        }
        if (!A.tail_out.empty() && A.cmp_ref == 0) {
            FILE * f = std::fopen(A.tail_out.c_str(), "wb");
            std::fwrite(tail.data(), 4, tail.size(), f);
            std::fclose(f);
        }
        if (!A.argmax_out.empty()) {
            FILE * f = std::fopen(A.argmax_out.c_str(), "w");
            for (int i = 0; i < Npre; ++i) std::fprintf(f, "%d\n", arg[i]);
            std::fclose(f);
        }
        if (A.gen > 0 && A.cmp_ref == 0 && A.spec.k >= 0) {
            X.ref = A.ref_all;
            std::vector<int> prompt(toks.begin(), toks.begin() + N);
            run_spec(g, X, A.spec, prompt, arg[N - 1], A.prefill_slot, A.gen, A.gen_out, s, A.pf_fast ? &PF : nullptr);
        } else
        if (A.gen > 0 && A.cmp_ref == 0) {
            // greedy decoding of the sequence of slot prefill_slot (B = 1, same-sequence graph)
            X.ref = A.ref_all;
            swap_conv_state(g, X, A.prefill_slot, s);
            cudaGraphExec_t ex = R.capture_batch(1, 0);
            int tok = arg[N - 1], pos = N;
            std::vector<int> out;
            std::vector<int> forced;
            if (!A.force_file.empty()) forced = read_tokens(A.force_file);
            FILE * flg = A.gen_logits.empty() ? nullptr : std::fopen(A.gen_logits.c_str(), "wb");
            std::vector<float> lg(NVOCAB);
            const int stop_ids[2] = {200006, 199999};
            const double t0 = now_s();
            for (int i = 0; i < A.gen; ++i) {
                if (!forced.empty()) { if (i >= (int) forced.size()) break; tok = forced[i]; }
                out.push_back(tok);
                if (forced.empty() && (tok == stop_ids[0] || tok == stop_ids[1])) break;
                if (pos >= X.maxctx) break;
                const int sl = A.prefill_slot;
                CHECK(cudaMemcpy(X.btok, &tok, 4, cudaMemcpyHostToDevice));
                CHECK(cudaMemcpy(X.pos_arr, &pos, 4, cudaMemcpyHostToDevice));
                CHECK(cudaMemcpy(X.slot_arr, &sl, 4, cudaMemcpyHostToDevice));
                CHECK(cudaGraphLaunch(ex, s));
                int pred = 0;
                CHECK(cudaMemcpyAsync(&pred, X.barg, 4, cudaMemcpyDeviceToHost, s));
                if (flg) { CHECK(cudaMemcpyAsync(lg.data(), X.logits, NVOCAB * 4, cudaMemcpyDeviceToHost, s)); }
                CHECK(cudaStreamSynchronize(s));
                if (flg) std::fwrite(lg.data(), 4, NVOCAB, flg);
                if (!forced.empty()) out.back() = pred;      // forced mode: record the model's argmax after the forced token
                tok = pred;
                ++pos;
            }
            if (flg) std::fclose(flg);
            const double dt = now_s() - t0;
            std::printf("generated %zu tokens in %.2f s (%.1f ms/token incl. host round trip)\n", out.size(), dt, 1000.0 * dt / out.size());
            if (!A.gen_out.empty()) {
                FILE * f = std::fopen(A.gen_out.c_str(), "w");
                for (int t : out) std::fprintf(f, "%d\n", t);
                std::fclose(f);
            }
        }
    }

    if (A.multi_check && !A.prefill_file.empty()) {
        // 4 sequences (different token slices, different lengths) prefilled into slots 0..3; then one multi-sequence step
        // (4 tokens, 4 slots, 4 positions) is compared with 4 independent single-sequence B=1 steps (same-sequence path).
        std::vector<int> toks = read_tokens(A.prefill_file);
        const int S = std::min(4, X.nslots);
        int len[4], nt[4];
        std::vector<int> arg(toks.size());
        for (int sl = 0; sl < S; ++sl) {
            len[sl] = (int) toks.size() - 400 - 53 * sl;
            prefill(g, X, R, sl, toks.data() + 37 * sl, len[sl], 0, 16, arg.data(), nullptr, 0, s);
            nt[sl] = arg[len[sl] - 1];
        }
        std::vector<float> ref((size_t) S * NVOCAB), mu((size_t) S * NVOCAB);
        X.ref = false;
        for (int sl = 0; sl < S; ++sl) {
            swap_conv_state(g, X, sl, s);
            R.batch_slot(sl, len[sl], &nt[sl], 1, &ref[(size_t) sl * NVOCAB]);
            batch_rollback(g, X, 0, s);
            CHECK(cudaStreamSynchronize(s));
            swap_conv_state(g, X, sl, s);
        }
        int poss[4], slots[4];
        for (int sl = 0; sl < S; ++sl) { poss[sl] = len[sl]; slots[sl] = sl; }
        R.multi(nt, poss, slots, S, mu.data());
        std::printf("multi-sequence check: S=%d, positions %d %d %d %d\n", S, len[0], len[1], S > 2 ? len[2] : 0, S > 3 ? len[3] : 0);
        for (int sl = 0; sl < S; ++sl) compare_logits(&mu[(size_t) sl * NVOCAB], &ref[(size_t) sl * NVOCAB], 1, ("  slot " + std::to_string(sl) + " multi-step vs single-sequence step").c_str());
    }
    if (A.time_b) {
        // verify cost per B (same-sequence batch graph, synthetic zero KV), ms per call: B = 1..16 at ctx 300 and 4000
        for (int il = 0; il < NAPP; ++il)
            for (int sl = 0; sl < 1; ++sl) {
                const bool swa = app_is_swa(il);
                const size_t bytes = (swa ? (size_t) X.ring : (size_t) X.maxctx) * NKV * HD * ((swa || !X.kv16) ? 4 : 2);
                CHECK(cudaMemset(X.kvs[il].k[sl], 0, bytes)); CHECK(cudaMemset(X.kvs[il].v[sl], 0, bytes));
            }
        reset_state(g);
        cudaEvent_t e0, e1;
        CHECK(cudaEventCreate(&e0)); CHECK(cudaEventCreate(&e1));
        for (int ctx : {300, 4000}) {
            if (std::getenv("TIMEB_ONLY") && ctx != 300) continue;
            std::printf("TIMEB ctx %d ms/call B=1..16:", ctx);
            for (int B = 1; B <= MB; ++B) {
                if (const char * only = std::getenv("TIMEB_ONLY")) { if (std::atoi(only) != B) continue; }
                cudaGraphExec_t ex = R.capture_batch(B, 0);
                std::vector<int> toks(B, 1234), pa(B);
                if (std::getenv("TIMEB_DIVERSE")) for (int k = 0; k < B; ++k) toks[k] = (1234 + 7919 * k + 13 * k * k) % 150000;
                for (int k = 0; k < B; ++k) pa[k] = ctx + k;
                int sa[MB] = {0};
                CHECK(cudaMemcpy(X.btok, toks.data(), B * 4, cudaMemcpyHostToDevice));
                CHECK(cudaMemcpy(X.pos_arr, pa.data(), B * 4, cudaMemcpyHostToDevice));
                CHECK(cudaMemcpy(X.slot_arr, sa, sizeof(sa), cudaMemcpyHostToDevice));
                for (int i = 0; i < 5; ++i) CHECK(cudaGraphLaunch(ex, s));
                CHECK(cudaEventRecord(e0, s));
                for (int i = 0; i < 40; ++i) CHECK(cudaGraphLaunch(ex, s));
                CHECK(cudaEventRecord(e1, s));
                CHECK(cudaEventSynchronize(e1));
                float ms = 0;
                CHECK(cudaEventElapsedTime(&ms, e0, e1));
                { int nu = 0; CHECK(cudaMemcpy(&nu, X.grp_nu, 4, cudaMemcpyDeviceToHost)); std::printf(" %d:%.2f(nu%d)", B, ms / 40, nu); }
            }
            std::printf("\n"); std::fflush(stdout);
        }
    }
    if (A.time_decode) {
        // decode step timing: synthetic (zero) KV contents; the cost does not depend on the data, only on the positions
        for (int il = 0; il < NAPP; ++il)
            for (int sl = 0; sl < X.nslots; ++sl) {
                const bool swa = app_is_swa(il);
                const size_t bytes = (swa ? (size_t) X.ring : (size_t) X.maxctx) * NKV * HD * ((swa || !X.kv16) ? 4 : 2);
                CHECK(cudaMemset(X.kvs[il].k[sl], 0, bytes)); CHECK(cudaMemset(X.kvs[il].v[sl], 0, bytes));
            }
        reset_state(g);
        reset_extra_slots(g, X);
        cudaEvent_t e0, e1;
        CHECK(cudaEventCreate(&e0)); CHECK(cudaEventCreate(&e1));
        std::printf("decode timing (ms per step, all sequences one token each, CUDA graph, 30 reps; positions = context length)\n");
        std::printf("%8s %10s %10s\n", "context", "S=1", "S=4");
        for (int ctx : {300, 1024, 8192, 32768, 65000}) {
            double t[2];
            int idx = 0;
            for (int S : {1, 4}) {
                if (S > X.nslots) { t[idx++] = -1; continue; }
                int toks[4], poss[4], slots[4];
                for (int i = 0; i < S; ++i) { toks[i] = 1000 + 17 * i; poss[i] = ctx - i * 3; slots[i] = i; }
                cudaGraphExec_t ex = R.capture_batch(S, 1);
                CHECK(cudaMemcpy(X.btok, toks, sizeof(toks), cudaMemcpyHostToDevice));
                CHECK(cudaMemcpy(X.pos_arr, poss, sizeof(poss), cudaMemcpyHostToDevice));
                CHECK(cudaMemcpy(X.slot_arr, slots, sizeof(slots), cudaMemcpyHostToDevice));
                for (int i = 0; i < 5; ++i) CHECK(cudaGraphLaunch(ex, s));
                CHECK(cudaEventRecord(e0, s));
                for (int i = 0; i < 30; ++i) CHECK(cudaGraphLaunch(ex, s));
                CHECK(cudaEventRecord(e1, s));
                CHECK(cudaEventSynchronize(e1));
                float ms = 0;
                CHECK(cudaEventElapsedTime(&ms, e0, e1));
                t[idx++] = ms / 30;
            }
            std::printf("%8d %10.2f %10.2f\n", ctx, t[0], t[1]);
            std::fflush(stdout);
        }
    }
    if (!A.time_depths.empty() && A.pf_fast && !A.prefill_file.empty()) {
        // steady-state chunk throughput at depth D: prefill the first D real tokens, then time one more chunk of PB real tokens
        std::vector<int> toks = read_tokens(A.prefill_file);
        std::printf("fast prefill chunk throughput at depth (real tokens in the cache, PB=%d)\n%8s %10s %10s\n", A.pf_fast, "depth", "ms/chunk", "tok/s");
        std::stringstream ss(A.time_depths);
        std::string item;
        while (std::getline(ss, item, ',')) {
            const int D = std::atoi(item.c_str());
            if (D + A.pf_fast > (int) toks.size() || D + A.pf_fast > X.maxctx) { std::printf("%8d (not enough tokens)\n", D); continue; }
            reset_state(g);
            int la;
            if (D > 0) prefill_fast(g, X, PF, 0, toks.data(), D, 0, 1, nullptr, &la, s);
            double best = 1e30;
            for (int r = 0; r < 3; ++r) {
                swap_conv_state(g, X, 0, s);
                const double tps = prefill_fast(g, X, PF, 0, toks.data() + D, A.pf_fast, D, 1, nullptr, &la, s);
                best = std::min(best, A.pf_fast / tps * 1000.0);
            }
            std::printf("%8d %10.2f %10.1f\n", D, best, A.pf_fast / best * 1000.0);
            std::fflush(stdout);
        }
    }
    if (A.time_prefill) {
        for (int il = 0; il < NAPP; ++il)
            for (int sl = 0; sl < 1; ++sl) {
                const bool swa = app_is_swa(il);
                const size_t bytes = (swa ? (size_t) X.ring : (size_t) X.maxctx) * NKV * HD * ((swa || !X.kv16) ? 4 : 2);
                CHECK(cudaMemset(X.kvs[il].k[sl], 0, bytes)); CHECK(cudaMemset(X.kvs[il].v[sl], 0, bytes));
            }
        reset_state(g);
        cudaEvent_t e0, e1;
        CHECK(cudaEventCreate(&e0)); CHECK(cudaEventCreate(&e1));
        std::printf("prefill chunk timing (same-sequence batch of B tokens at position ctx, synthetic KV)\n");
        std::printf("%8s %4s %10s %10s\n", "ctx", "B", "ms/chunk", "tok/s");
        for (int ctx : {256, 8192, 32768, 65000 - 32}) {
            for (int B : {16, MB}) {
                if (B == 16 && MB != 16 && false) continue;
                cudaGraphExec_t ex = R.capture_batch(B, 0);
                std::vector<int> toks(B, 1234), pa(B);
                for (int k = 0; k < B; ++k) pa[k] = ctx + k;
                int sa[MB] = {0};
                CHECK(cudaMemcpy(X.btok, toks.data(), B * 4, cudaMemcpyHostToDevice));
                CHECK(cudaMemcpy(X.pos_arr, pa.data(), B * 4, cudaMemcpyHostToDevice));
                CHECK(cudaMemcpy(X.slot_arr, sa, sizeof(sa), cudaMemcpyHostToDevice));
                for (int i = 0; i < 3; ++i) CHECK(cudaGraphLaunch(ex, s));
                CHECK(cudaEventRecord(e0, s));
                for (int i = 0; i < 10; ++i) CHECK(cudaGraphLaunch(ex, s));
                CHECK(cudaEventRecord(e1, s));
                CHECK(cudaEventSynchronize(e1));
                float ms = 0;
                CHECK(cudaEventElapsedTime(&ms, e0, e1));
                std::printf("%8d %4d %10.2f %10.1f\n", ctx, B, ms / 10, B * 10000.0 / ms);
                if (MB == 16) break;
            }
        }
    }
    return 0;
}
