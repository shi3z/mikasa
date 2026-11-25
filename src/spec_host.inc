#include <unordered_map>
#include <algorithm>
#include <cstdint>
// Speculative decoding (n-gram / prompt-lookup drafts) for the lean single-sequence path (textually included before long_host.inc).
//
// One decode step = ONE graph launch and ONE host sync:
//   [H2D rollback index] [kb_rollback of the previous step] [H2D tokens / positions / slots] [forward_batch(B)] [sampler] [D2H results]
// B = 1 + number of drafted tokens (1..16). Graphs are captured per B. The host decides B from the n-gram match and the measured cost table.
// Greedy mode: accept the longest draft prefix whose argmax agrees, then take the bonus token (exact up to the numerical differences between the
// B-token and the 1-token kernels, see RESULTS_spec.md). Sampling mode (temp > 0): draft tokens are deterministic proposals, so draft d is accepted with
// probability p(d), on rejection the token is drawn from p with d removed (renormalised): the emitted sequence has exactly the distribution of
// ancestral sampling from softmax(logits / temp).

struct SpecArgs {
    int k = -1;                 // max drafted tokens per step (<= 15); -1: speculation code not used (old greedy loop); 0: new loop without drafts
    int m = 1;                  // minimum n-gram match length
    int chain = 128;            // max candidates examined in the hash chain
    bool longest = true;        // pick the candidate with the longest match (else the most recent)
    bool fixed = false;         // no cost/acceptance model: always draft up to k tokens when a match exists
    float temp = 0.0f;          // 0 = greedy
    unsigned seed = 1;
    std::string oracle;         // file of reference tokens: drafts are taken from it (numerical-agreement test)
    std::string stats_out;
    bool sampler_test = false;
    int trials = 0;
    int wide = 0;               // >0: verify passes with more than `wide` rows (up to 64) use the tensor-core prefill path (greedy only, not bit-identical to B=1 decoding)
    int copy_min = 0;           // >0: copy proposer (3-gram index, longest backward match <= 64 of the 64 latest occurrences, sticky continuation) is tried first; min candidate tokens
};

// ---- sampler: one block per verify row. Row r has the draft token dr[r] (or -1). Outputs acc[r] (u < p(dr[r])) and tok[r]:
// a sample from p with dr[r] excluded (Gumbel-max). Counter-based RNG: (seed, step, row, index).
__device__ __forceinline__ unsigned spec_hash(unsigned a, unsigned b, unsigned c, unsigned d) {
    unsigned h = a * 0x9E3779B1u ^ (b + 0x7F4A7C15u) * 0x85EBCA6Bu;
    h ^= h >> 15; h *= 0x2C1B3C6Du; h ^= (c + 0x165667B1u) * 0xC2B2AE35u; h ^= h >> 12; h *= 0x297A2D39u;
    h ^= (d + 0x27D4EB2Fu) * 0x9E3779B1u; h ^= h >> 15; h *= 0x85EBCA6Bu; h ^= h >> 13; h *= 0xC2B2AE35u; h ^= h >> 16;
    return h;
}
// Gumbel noise g = -log(-log(u)), u = (h + 0.5) / 2^32, evaluated in float without losing the tails: for u > 1/2 the quantity -log(u) = -log1p(-d) is built
// from the exactly representable distance d = (2^32 - 0.5 - h) / 2^32 to 1 (a float u would round to 1.0 and truncate / coarsen the upper Gumbel tail, which
// decides the winner among ~150k small-probability tokens; an earlier version with 24-bit u failed the chi-square test at T=2).
__device__ __forceinline__ float spec_gumbel(unsigned h) {
    float e;
    if (h < 0x80000000u) e = -logf(((float) h + 0.5f) * 2.3283064365386963e-10f);
    else e = -log1pf(-(((float) (0xFFFFFFFFu - h)) + 0.5f) * 2.3283064365386963e-10f);
    return -logf(e);
}
__device__ __forceinline__ float spec_u01(unsigned h) { return ((h >> 9) + 0.5f) * (1.0f / 8388608.0f); }   // (0,1), max = 1 - 2^-24 (used for the accept test only)

__global__ void k_spec_sample(const float * __restrict__ logits_all, int n, int stride, const int * __restrict__ dr, const float * __restrict__ prm,
                              int * __restrict__ acc, int * __restrict__ tok) {
    __shared__ float red[32];
    __shared__ float rv[32];
    __shared__ int ri[32];
    const int row = blockIdx.x, lane = threadIdx.x & 31, wid = threadIdx.x >> 5, nw = blockDim.x >> 5;
    const float * lg = logits_all + (size_t) row * stride;
    const float invT = prm[0];
    const unsigned seed = __float_as_uint(prm[1]), step = __float_as_uint(prm[2]);
    const int d = dr[row];
    float mx = -INFINITY;
    for (int i = threadIdx.x; i < n; i += blockDim.x) mx = fmaxf(mx, lg[i]);
    for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
    if (lane == 0) red[wid] = mx;
    __syncthreads();
    mx = red[0];
    for (int w = 1; w < nw; ++w) mx = fmaxf(mx, red[w]);
    __syncthreads();
    float z = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) z += expf((lg[i] - mx) * invT);
    for (int o = 16; o > 0; o >>= 1) z += __shfl_xor_sync(0xffffffffu, z, o);
    if (lane == 0) red[wid] = z;
    __syncthreads();
    z = 0.0f;
    for (int w = 0; w < nw; ++w) z += red[w];
    float bv = -INFINITY; int bi = 0;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        if (i == d) continue;
        const float v = lg[i] * invT + spec_gumbel(spec_hash(seed, step, row, i));
        if (v > bv) { bv = v; bi = i; }
    }
    for (int o = 16; o > 0; o >>= 1) {
        const float ov = __shfl_xor_sync(0xffffffffu, bv, o); const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
        if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
    }
    if (lane == 0) { rv[wid] = bv; ri[wid] = bi; }
    __syncthreads();
    if (threadIdx.x == 0) {
        float v = rv[0]; int b = ri[0];
        for (int w = 1; w < nw; ++w) if (rv[w] > v || (rv[w] == v && ri[w] < b)) { v = rv[w]; b = ri[w]; }
        tok[row] = b;
        int a = 0;
        if (d >= 0) {
            const float pd = expf((lg[d] - mx) * invT) / z;
            a = spec_u01(spec_hash(seed ^ 0xA5A5A5A5u, step, row, 0xFFFFFFFFu)) < pd;
        }
        acc[row] = a;
    }
}

// ---- n-gram index over the whole context (prompt + generated)
struct NgramIndex {
    int M = 3, chain = 128; bool longest = true;
    std::vector<int> ctx, prev;
    std::unordered_map<uint64_t, int> head;
    uint64_t key(int end) const {
        uint64_t h = 1469598103934665603ull;
        for (int t = 0; t < M; ++t) { h ^= (uint64_t) (unsigned) ctx[end - t]; h *= 1099511628211ull; }
        return h;
    }
    void push(int t) {
        ctx.push_back(t); prev.push_back(-1);
        const int e = (int) ctx.size() - 1;
        if (e + 1 >= M) {
            const uint64_t k = key(e);
            auto it = head.find(k);
            if (it != head.end()) { prev[e] = it->second; it->second = e; } else head[k] = e;
        }
    }
    // best earlier occurrence of the suffix: returns match length (0 = none), *cont = index of the first continuation token
    int find(int * cont) const {
        const int n = (int) ctx.size();
        if (n < M + 1) return 0;
        int best = 0, bc = -1;
        int c = prev[n - 1];
        for (int it = 0; c >= 0 && it < chain; ++it, c = prev[c]) {
            int L = 0;
            while (L < 64 && c - L >= 0 && ctx[c - L] == ctx[n - 1 - L]) ++L;
            if (L >= M && L > best) { best = L; bc = c; if (!longest || L >= 64) break; }
        }
        if (best == 0) return 0;
        *cont = bc + 1;
        return best;
    }
};

struct SpecStats {
    long copy_steps = 0, steps = 0, tokens = 0, drafted = 0, accepted = 0, nodraft_steps = 0;
    long hist_B[65] = {0}, hist_acc[65] = {0}, wide_steps = 0;
};

// plain wall clock
static double spec_now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

struct SpecRunner {
    Engine & g; BatchState & X; cudaStream_t s; const SpecArgs & A;
    cudaGraphExec_t gx[17] = {nullptr};
    int *w_tok, *w_pos, *w_slot, *w_arg, *w_rb;
    unsigned seed = 1;
    int *h_tok, *h_pos, *h_slot, *h_rb, *h_dr, *h_arg, *h_acc, *h_stok;
    float * h_prm; float * d_prm; int *d_dr, *d_acc, *d_stok;
    SpecRunner(Engine & g_, BatchState & X_, cudaStream_t s_, const SpecArgs & A_) : g(g_), X(X_), s(s_), A(A_) {
        seed = A_.seed;
        CHECK(cudaMallocHost(&h_tok, 64)); CHECK(cudaMallocHost(&h_pos, 64)); CHECK(cudaMallocHost(&h_slot, 64)); CHECK(cudaMallocHost(&h_rb, 4));
        CHECK(cudaMallocHost(&h_dr, 64)); CHECK(cudaMallocHost(&h_arg, 256)); CHECK(cudaMallocHost(&h_acc, 64)); CHECK(cudaMallocHost(&h_stok, 64));
        CHECK(cudaMallocHost(&h_prm, 16));
        CHECK(cudaMallocHost(&w_tok, 256)); CHECK(cudaMallocHost(&w_pos, 256)); CHECK(cudaMallocHost(&w_slot, 256)); CHECK(cudaMallocHost(&w_arg, 256)); CHECK(cudaMallocHost(&w_rb, 4));
        CHECK(cudaMalloc(&d_prm, 16)); CHECK(cudaMalloc(&d_dr, 64)); CHECK(cudaMalloc(&d_acc, 64)); CHECK(cudaMalloc(&d_stok, 64));
    }
    cudaGraphExec_t graph(int B) {
        if (gx[B]) return gx[B];
        const bool samp = A.temp > 0;
        cudaGraph_t gr;
        CHECK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
        CHECK(cudaMemcpyAsync(X.rb_j, h_rb, 4, cudaMemcpyHostToDevice, s));
        kb_rollback<<<dim3((E + 255) / 256, NAPP * 4), 256, 0, s>>>(X.rb_tab, X.rb_j);
        CHECK(cudaMemcpyAsync(X.btok, h_tok, B * 4, cudaMemcpyHostToDevice, s));
        CHECK(cudaMemcpyAsync(X.pos_arr, h_pos, B * 4, cudaMemcpyHostToDevice, s));
        CHECK(cudaMemcpyAsync(X.slot_arr, h_slot, B * 4, cudaMemcpyHostToDevice, s));
        if (samp) {
            CHECK(cudaMemcpyAsync(d_dr, h_dr, B * 4, cudaMemcpyHostToDevice, s));
            CHECK(cudaMemcpyAsync(d_prm, h_prm, 16, cudaMemcpyHostToDevice, s));
        }
        forward_batch(g, X, B, 0, s);
        if (samp) {
            k_spec_sample<<<B, 1024, 0, s>>>(X.logits, NVOCAB, NVOCAB_PAD, d_dr, d_prm, d_acc, d_stok);
            CHECK(cudaMemcpyAsync(h_acc, d_acc, B * 4, cudaMemcpyDeviceToHost, s));
            CHECK(cudaMemcpyAsync(h_stok, d_stok, B * 4, cudaMemcpyDeviceToHost, s));
        } else
            CHECK(cudaMemcpyAsync(h_arg, X.barg, B * 4, cudaMemcpyDeviceToHost, s));
        CHECK(cudaStreamEndCapture(s, &gr));
        CHECK(cudaGraphInstantiate(&gx[B], gr, nullptr, nullptr, 0));
        return gx[B];
    }
    // stage the inputs of a step and run it (one launch, one sync)
    void run(int B, const int * toks, int pos, int slot, int rb, unsigned step) {
        for (int i = 0; i < B; ++i) { h_tok[i] = toks[i]; h_pos[i] = pos + i; h_slot[i] = slot; h_dr[i] = (i + 1 < B) ? toks[i + 1] : -1; }
        *h_rb = rb;
        h_prm[0] = A.temp > 0 ? 1.0f / A.temp : 1.0f;
        unsigned sd = seed; std::memcpy(&h_prm[1], &sd, 4); std::memcpy(&h_prm[2], &step, 4);
        CHECK(cudaGraphLaunch(graph(B), s));
        CHECK(cudaStreamSynchronize(s));
    }
};

// ---- wide (tensor-core prefill path) verification of B > 16 rows; rollback of either path is done eagerly before the next step
static void spec_eager_rollback(SpecRunner & SR, const RbEntry * tab, int * dj, int j, cudaStream_t s) {
    *SR.w_rb = j;
    CHECK(cudaMemcpyAsync(dj, SR.w_rb, 4, cudaMemcpyHostToDevice, s));
    kb_rollback<<<dim3((E + 255) / 256, NAPP * 4), 256, 0, s>>>(tab, dj);
}
static void spec_run_wide(Engine & g, BatchState & X, PFState & PF, SpecRunner & SR, int B, const int * toks, int pos, int slot, cudaStream_t s) {
    for (int i = 0; i < B; ++i) { SR.w_tok[i] = toks[i]; SR.w_pos[i] = pos + i; SR.w_slot[i] = slot; }
    CHECK(cudaMemcpyAsync(PF.btok, SR.w_tok, B * 4, cudaMemcpyHostToDevice, s));
    CHECK(cudaMemcpyAsync(PF.pos_arr, SR.w_pos, B * 4, cudaMemcpyHostToDevice, s));
    CHECK(cudaMemcpyAsync(PF.slot_arr, SR.w_slot, B * 4, cudaMemcpyHostToDevice, s));
    forward_prefill(g, X, PF, B, 0, s, true);
    CHECK(cudaMemcpyAsync(SR.w_arg, PF.arg_all, B * 4, cudaMemcpyDeviceToHost, s));
    CHECK(cudaStreamSynchronize(s));
}

static const char * SPEC_STOP_NOTE = "stop ids 200006 / 199999";

static void run_spec_sampler_test(Engine & g, BatchState & X, SpecRunner & SR, const std::vector<int> & prompt, int first, int slot, cudaStream_t s) {
    // distribution check of the rejection sampler: logits of one real position, draft d = (a) top-1 (b) rank 3 (c) a low-probability token (d) none;
    // 200k kernel launches with different seeds; the emitted token (draft if accepted else the resampled one) must follow softmax(logits / T).
    const int N = (int) prompt.size();
    swap_conv_state(g, X, slot, s);
    int tk = first;
    SR.run(1, &tk, N, slot, -1, 0);
    std::vector<float> lg(NVOCAB);
    CHECK(cudaMemcpy(lg.data(), X.logits, NVOCAB * 4, cudaMemcpyDeviceToHost));
    const float T = SR.A.temp > 0 ? SR.A.temp : 1.0f;
    std::vector<int> order(NVOCAB);
    for (int i = 0; i < NVOCAB; ++i) order[i] = i;
    std::partial_sort(order.begin(), order.begin() + 8, order.end(), [&](int a, int b) { return lg[a] > lg[b]; });
    double mx = lg[order[0]], Z = 0;
    for (int i = 0; i < NVOCAB; ++i) Z += std::exp((lg[i] - mx) / T);
    std::vector<double> p(NVOCAB);
    for (int i = 0; i < NVOCAB; ++i) p[i] = std::exp((lg[i] - mx) / T) / Z;
    std::printf("sampler test: T=%.2f, top-4 probabilities %.4f %.4f %.4f %.4f\n", T, p[order[0]], p[order[1]], p[order[2]], p[order[3]]);
    const int drafts[4] = {order[0], order[2], 7, -1};
    const int NT = 100000;
    for (int dcase = 0; dcase < 4; ++dcase) {
        const int d = drafts[dcase];
        std::map<int, int> hist;
        int nacc = 0;
        int *dd, *aa, *tt; float * pp;
        CHECK(cudaMalloc(&dd, 4)); CHECK(cudaMalloc(&aa, 4)); CHECK(cudaMalloc(&tt, 4)); CHECK(cudaMalloc(&pp, 12));
        CHECK(cudaMemcpy(dd, &d, 4, cudaMemcpyHostToDevice));
        for (int t = 0; t < NT; ++t) {
            float prm[3] = {1.0f / T, 0, 0}; unsigned sd = 12345u + dcase, st = t;
            std::memcpy(&prm[1], &sd, 4); std::memcpy(&prm[2], &st, 4);
            CHECK(cudaMemcpy(pp, prm, 12, cudaMemcpyHostToDevice));
            k_spec_sample<<<1, 1024, 0, s>>>(X.logits, NVOCAB, NVOCAB_PAD, dd, pp, aa, tt);
            int a, tk2;
            CHECK(cudaMemcpy(&a, aa, 4, cudaMemcpyDeviceToHost)); CHECK(cudaMemcpy(&tk2, tt, 4, cudaMemcpyDeviceToHost));
            nacc += a; hist[a ? d : tk2]++;
        }
        // chi-square over the tokens with expected count >= 20 plus an "other" bin
        double chi = 0, rest_e = NT, rest_o = NT; int bins = 0;
        for (auto & kv : hist) {}
        for (int i = 0; i < NVOCAB; ++i) {
            const double e = NT * p[i];
            if (e >= 20) { const double o = hist.count(i) ? hist[i] : 0; chi += (o - e) * (o - e) / e; rest_e -= e; rest_o -= o; ++bins; }
        }
        if (rest_e > 20) { chi += (rest_o - rest_e) * (rest_o - rest_e) / rest_e; ++bins; }
        const double pd = d >= 0 ? p[d] : 0;
        std::printf("  draft %-7d p(d)=%.5f: accept rate %.5f (expected %.5f); chi2 = %.1f over %d bins (dof ~ %d)\n", d, pd, (double) nacc / NT, pd, chi, bins, bins - 1);
        CHECK(cudaFree(dd)); CHECK(cudaFree(aa)); CHECK(cudaFree(tt)); CHECK(cudaFree(pp));
    }
}

static int run_spec(Engine & g, BatchState & X, const SpecArgs & A, const std::vector<int> & prompt, int first, int slot, int gen, const std::string & out_path,
                    cudaStream_t s, PFState * PFp = nullptr) {
    const int N = (int) prompt.size();
    const bool wide_on = A.wide > 0 && PFp && PFp->PB >= 64 && A.temp <= 0;
    if (wide_on) { if (!PFp->rb_tab) init_pf_rollback(g, *PFp); g_moe_tb = 32; }
    const int Kmax = std::max(0, std::min(A.k, wide_on ? 63 : MB - 1));
    SpecRunner SR(g, X, s, A);
    if (A.sampler_test) { run_spec_sampler_test(g, X, SR, prompt, first, slot, s); return 0; }
    swap_conv_state(g, X, slot, s);
    // ---- cost table: wall time of one step (launch + sync) per B, at the current position (state restored afterwards)
    double cost[65] = {0};
    {
        SavedState st; snapshot_state(g, st, true);
        int tk[16]; for (int i = 0; i < 16; ++i) tk[i] = first;
        for (int B = 1; B <= std::min(Kmax + 1, 16); ++B) {
            SR.run(B, tk, N, slot, -1, 0); SR.run(B, tk, N, slot, B, 0);
            const double t0 = spec_now();
            const int R = 12;
            for (int r = 0; r < R; ++r) SR.run(B, tk, N, slot, B, 0);
            cost[B] = (spec_now() - t0) / R * 1000.0;
        }
        if (wide_on && Kmax + 1 > A.wide) {
            int tkw[64]; for (int i = 0; i < 64; ++i) tkw[i] = first;
            int pts[8], np = 0; double cv[8];
            for (int B : {A.wide + 1, 16, 24, 32, 48, 64}) if (B <= Kmax + 1 && B > A.wide && (np == 0 || B > pts[np - 1])) {
                spec_run_wide(g, X, *PFp, SR, B, tkw, N, slot, s);
                const double t0 = spec_now();
                for (int r = 0; r < 6; ++r) spec_run_wide(g, X, *PFp, SR, B, tkw, N, slot, s);
                pts[np] = B; cv[np++] = (spec_now() - t0) / 6 * 1000.0;
            }
            for (int B = A.wide + 1; B <= Kmax + 1; ++B) {
                int j = 0; while (j + 1 < np && pts[j + 1] < B) ++j;
                cost[B] = (j + 1 >= np) ? cv[np - 1] : cv[j] + (cv[j + 1] - cv[j]) * (B - pts[j]) / (double) (pts[j + 1] - pts[j]);
            }
        }
        if (Kmax == 0) { SR.run(1, tk, N, slot, -1, 0); const double t0 = spec_now(); for (int r = 0; r < 12; ++r) SR.run(1, tk, N, slot, 1, 0); cost[1] = (spec_now() - t0) / 12 * 1000.0; }
        snapshot_state(g, st, false);
        CHECK(cudaStreamSynchronize(s));
        std::printf("spec: measured step cost (ms, launch+sync) by B:");
        for (int B = 1; B <= Kmax + 1; ++B) if (B <= 16 || B % 8 == 0 || B == Kmax + 1) std::printf(" %d:%.2f", B, cost[B]);
        std::printf("\n");
    }
    // ---- acceptance model (per match-length bucket, per draft position)
    auto bucket = [](int L) { return L <= 1 ? 0 : L == 2 ? 1 : L == 3 ? 2 : L <= 5 ? 3 : L <= 9 ? 4 : L <= 15 ? 5 : L <= 31 ? 6 : 7; };
    double tried[10][64], accd[10][64];
    for (int b = 0; b < 10; ++b) for (int i = 0; i < 64; ++i) { tried[b][i] = 0; accd[b][i] = 0; }
    // P(draft token i accepted | first i accepted) for match bucket b: per-position counts, shrunk towards the bucket's pooled rate (so that positions that
    // were rarely reached, i.e. deep into long drafts, inherit the observed rate instead of a fixed 0.5)
    auto qest = [&](int b, int i) {
        double ta = 0, aa = 0;
        for (int j = 0; j < 64; ++j) { ta += tried[b][j]; aa += accd[b][j]; }
        const double pr = (aa + 1.0) / (ta + 2.0);
        return (accd[b][i] + 2.0 * pr) / (tried[b][i] + 2.0);
    };

    NgramIndex NG0; NG0.M = std::max(1, A.m); NG0.chain = A.chain; NG0.longest = A.longest;
    for (int t : prompt) NG0.push(t);
    NgramIndex NGC0; NGC0.M = 3; NGC0.chain = 64; NGC0.longest = true;
    if (A.copy_min > 0) for (int t : prompt) NGC0.push(t);
    std::vector<int> oracle;
    if (!A.oracle.empty()) oracle = read_tokens(A.oracle);
    std::vector<int> out;
    SpecStats ST;
    auto generate = [&](int gen) {
    NgramIndex NG = NG0, NGC = NGC0;
    int sticky = -1, bk = 0;
    long copy_steps = 0;
    out.clear(); ST = SpecStats();
    const int stop_ids[2] = {200006, 199999};
    auto is_stop = [&](int t) { return t == stop_ids[0] || t == stop_ids[1]; };
    out.push_back(first); NG.push(first); if (A.copy_min > 0) NGC.push(first);
    int pos = N;            // position of out.back() (not yet fed)
    int rb = -1;            // rollback index for the next step
    unsigned step = 0;
    bool done = is_stop(first);
    int toks[64];
    bool prev_wide = false;
    while (!done && (int) out.size() < gen) {
        // ---- draft
        int k = 0, L = 0, cont = 0;
        if (Kmax > 0 && pos + 1 < X.maxctx) {
            if (!oracle.empty()) {
                const int n = (int) out.size();
                k = std::max(0, std::min(Kmax, (int) oracle.size() - n));
                for (int i = 0; i < k; ++i) toks[1 + i] = oracle[n + i];
            } else {
                auto choose = [&](int b, int avail) {
                    if (A.fixed) return avail;
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
                if (A.copy_min > 0) {
                    const int l = (int) NG.ctx.size();
                    int c2 = -1, Lc = 0;
                    const bool st = sticky >= 1 && sticky < l && NG.ctx[sticky - 1] == out.back();
                    if (st) c2 = sticky; else { Lc = NGC.find(&c2); if (Lc == 0) c2 = -1; }
                    if (c2 >= 0) {
                        const int avail = std::min(Kmax, l - c2);
                        if (avail >= A.copy_min) {
                            const int b = st ? 8 : (Lc >= 32 ? 9 : bucket(Lc));
                            const int kk = choose(b, avail);
                            if (kk > 0) { k = kk; cont = c2; L = Lc ? Lc : 1; bk = b; used = true; copy_steps++; sticky = c2;
                                          for (int i = 0; i < k; ++i) toks[1 + i] = NG.ctx[cont + i]; }
                        }
                    }
                }
                if (!used) {
                    L = NG.find(&cont);
                    if (L > 0) {
                        const int avail = std::min(Kmax, (int) NG.ctx.size() - cont);
                        bk = bucket(L);
                        k = choose(bk, avail);
                        for (int i = 0; i < k; ++i) toks[1 + i] = NG.ctx[cont + i];
                    }
                }
            }
            if (pos + 1 + k >= X.maxctx) k = std::max(0, X.maxctx - pos - 1);
        }
        const int B = k + 1;
        toks[0] = out.back();
        const bool wide = wide_on && B > A.wide;
        if (wide) {
            if (rb >= 0) spec_eager_rollback(SR, prev_wide ? PFp->rb_tab : X.rb_tab, prev_wide ? PFp->rb_j : X.rb_j, rb, s);
            spec_run_wide(g, X, *PFp, SR, B, toks, pos, slot, s);
            for (int i = 0; i < B; ++i) SR.h_arg[i] = SR.w_arg[i];
            ST.wide_steps++;
        } else {
            if (prev_wide && rb >= 0) { spec_eager_rollback(SR, PFp->rb_tab, PFp->rb_j, rb, s); rb = -1; }
            SR.run(B, toks, pos, slot, rb, step++);
        }
        prev_wide = wide;
        // ---- accept
        int m = 0;                                   // accepted draft tokens
        std::vector<int> emit;
        if (A.temp > 0) {
            while (m < k && SR.h_acc[m]) { emit.push_back(toks[1 + m]); ++m; }
            emit.push_back(SR.h_stok[m]);
        } else {
            while (m < k && SR.h_arg[m] == toks[1 + m]) { emit.push_back(toks[1 + m]); ++m; }
            emit.push_back(SR.h_arg[m]);
        }
        if (k > 0 && L > 0 && oracle.empty()) {
            const int b = bk;
            for (int i = 0; i < m; ++i) { tried[b][i] = tried[b][i] * 0.995 + 1; accd[b][i] = accd[b][i] * 0.995 + 1; }
            if (m < k) { tried[b][m] = tried[b][m] * 0.995 + 1; accd[b][m] *= 0.995; }
        }
        if (std::getenv("SPEC_DEBUG")) std::printf("step %u B=%d wide=%d m=%d first_mismatch_arg=%d draft=%d\n", step, B, (int) wide, m, m < k ? SR.h_arg[m] : -1, m < k ? toks[1 + m] : -1);
        rb = (m == k) ? -1 : m + 1;                  // state after the m+1 consumed tokens; if all were accepted nothing to undo
        pos += m + 1;
        ST.steps++; ST.hist_B[B]++; ST.drafted += k; ST.accepted += m; ST.hist_acc[m]++;
        if (k == 0) ST.nodraft_steps++;
        for (int t : emit) {
            if (sticky >= 0) { if (sticky < (int) NG.ctx.size() && NG.ctx[sticky] == t) ++sticky; else sticky = -1; }
            out.push_back(t); NG.push(t); if (A.copy_min > 0) NGC.push(t);
            if (is_stop(t)) { done = true; break; }
            if ((int) out.size() >= gen) break;
        }
        if (pos >= X.maxctx - 1) break;
    }
    ST.copy_steps = copy_steps;
    };
    SavedState st0; snapshot_state(g, st0, true);
    if (A.trials > 0) {
        // end-to-end distribution check: A.trials independent generations of `gen` tokens from the same prefilled state with different seeds
        std::map<std::vector<int>, int> hist;
        for (int t = 0; t < A.trials; ++t) {
            snapshot_state(g, st0, false);
            SR.seed = A.seed + 7919u * t;
            generate(gen);
            hist[std::vector<int>(out.begin() + 1, out.end())]++;
        }
        for (auto & kv : hist) { std::printf("TRIAL %d", kv.second); for (int x : kv.first) std::printf(" %d", x); std::printf("\n"); }
        return 0;
    }
    CHECK(cudaStreamSynchronize(s));
    const double t0 = spec_now();
    generate(gen);
    CHECK(cudaStreamSynchronize(s));
    const double dt = spec_now() - t0;
    ST.tokens = (long) out.size() - 1;
    std::printf("spec: K=%d M=%d %s%s temp=%.2f: generated %zu tokens in %.3f s = %.2f tok/s (%.3f ms/token); steps %ld, tokens/step %.3f\n", A.k, A.m,
                A.fixed ? "fixed " : "adaptive ", A.longest ? "longest-match" : "latest-match", A.temp, out.size(), dt, (out.size() - 1) / dt, 1000.0 * dt / (out.size() - 1),
                ST.steps, ST.steps ? (double) ST.tokens / ST.steps : 0.0);
    std::printf("spec: drafted %ld, accepted %ld (acceptance %.1f%%), mean accepted per step %.3f, mean accepted per drafting step %.3f, drafting steps %ld of %ld\n",
                ST.drafted, ST.accepted, ST.drafted ? 100.0 * ST.accepted / ST.drafted : 0.0, (double) ST.accepted / std::max(1L, ST.steps),
                (double) ST.accepted / std::max(1L, ST.steps - ST.nodraft_steps), ST.steps - ST.nodraft_steps, ST.steps);
    if (A.copy_min > 0) std::printf("spec: copy proposer used in %ld of %ld steps (min %d tokens)\n", ST.copy_steps, ST.steps, A.copy_min);
    if (wide_on) std::printf("spec: wide (tensor-core prefill path) verify passes: %ld of %ld\n", ST.wide_steps, ST.steps);
    std::printf("spec: B histogram:");
    for (int B = 1; B <= 64; ++B) if (ST.hist_B[B]) std::printf(" %d:%ld", B, ST.hist_B[B]);
    std::printf("\n");
    if (!A.stats_out.empty()) {
        FILE * f = std::fopen(A.stats_out.c_str(), "a");
        std::fprintf(f, "%d %d %d %.4f %zu %.5f %ld %ld %ld %ld\n", A.k, A.m, (int) A.fixed, A.temp, out.size() - 1, dt, ST.steps, ST.drafted, ST.accepted, ST.nodraft_steps);
        std::fclose(f);
    }
    if (!out_path.empty()) {
        FILE * f = std::fopen(out_path.c_str(), "w");
        for (int t : out) std::fprintf(f, "%d\n", t);
        std::fclose(f);
    }
    (void) SPEC_STOP_NOTE;
    return 0;
}
