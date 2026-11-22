// Q8_0 expert GEMV: current q8_experts_rows2 (2 rows/warp, one 32-elem block per lane) vs row-packed warps.
#include "kernels_common.cuh"
#include <cstdio>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>

// RP rows per warp; units = RP*NB (row, 32-element block), spread over lanes: u = lane + 32*t.
template <int RP, int NB, int WARPS>
__global__ void __launch_bounds__(WARPS * 32)
q8_pack(const int8_t * __restrict__ qs_all, const __half * __restrict__ sc_all,
        const float * __restrict__ x, int64_t x_slot_stride, float * __restrict__ y, int K, int N, const int * __restrict__ ids) {
    extern __shared__ float4 xs4[];   // x staged once per block (K floats)
    const int slot = blockIdx.y;
    {
        const float4 * gx = reinterpret_cast<const float4 *>(x + slot * x_slot_stride);
        for (int i = threadIdx.x; i < K / 4; i += WARPS * 32) xs4[i] = gx[i];
        __syncthreads();
    }
    const int64_t e = ids ? ids[slot] : 0;
    const int8_t * qs = qs_all + e * (int64_t) N * K;
    const __half * sc = sc_all + e * (int64_t) N * NB;
    const float4 * x4 = xs4;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int rowbase = (blockIdx.x * WARPS + warp) * RP;
    constexpr int UNITS = RP * NB;
    float acc[RP];
#pragma unroll
    for (int r = 0; r < RP; ++r) acc[r] = 0.0f;
#pragma unroll
    for (int u0 = 0; u0 < UNITS; u0 += 32) {
        const int u = u0 + lane;
        if (u < UNITS) {
            const int rs = u / NB, b = u - rs * NB;
            const int row = rowbase + rs;
            float tot = 0.0f;
            if (row < N) {
                float4 xv[8];
#pragma unroll
                for (int i = 0; i < 8; ++i) xv[i] = x4[b * 8 + i];
                const uint4 * q = reinterpret_cast<const uint4 *>(qs + (size_t) row * K);
                const float d = __half2float(sc[(size_t) row * NB + b]);
                const uint4 a0 = __ldg(q + b * 2), a1 = __ldg(q + b * 2 + 1);
                const uint32_t w[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
                float t = 0.0f;
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const float4 v = xv[i];
                    t += (float) (int8_t) w[i] * v.x + (float) (int8_t) (w[i] >> 8) * v.y
                       + (float) (int8_t) (w[i] >> 16) * v.z + (float) (int8_t) (w[i] >> 24) * v.w;
                }
                tot = d * t;
            }
#pragma unroll
            for (int r = 0; r < RP; ++r) acc[r] += (rs == r) ? tot : 0.0f;
        }
    }
#pragma unroll
    for (int r = 0; r < RP; ++r) {
        float a = acc[r];
        for (int off = 16; off > 0; off >>= 1) a += __shfl_xor_sync(0xffffffffu, a, off);
        if (lane == 0 && rowbase + r < N) y[(size_t) slot * N + rowbase + r] = a;
    }
}

template <typename F> static float graph_us(F f, cudaStream_t s, int R_ = 100) {
    cudaGraph_t g; cudaGraphExec_t ge;
    cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal);
    for (int i = 0; i < R_; ++i) f(s);
    cudaStreamEndCapture(s, &g); cudaGraphInstantiate(&ge, g, nullptr, nullptr, 0);
    cudaGraphLaunch(ge, s); cudaStreamSynchronize(s);
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    cudaEventRecord(a, s); for (int i = 0; i < 3; ++i) cudaGraphLaunch(ge, s); cudaEventRecord(b, s); cudaEventSynchronize(b);
    float ms; cudaEventElapsedTime(&ms, a, b); return ms * 1000.f / (3.f * R_);
}

#define LQ(RP_, NB_, W_) \
    static void lq_##RP_##_##NB_##_##W_(const int8_t * q, const __half * s_, const float * x, int64_t xs_, float * y, int K, int N, const int * ids, cudaStream_t st) { \
        dim3 grid(N / (W_ * RP_), 8); q8_pack<RP_, NB_, W_><<<grid, W_ * 32, K * sizeof(float), st>>>(q, s_, x, xs_, y, K, N, ids); }
LQ(2, 72, 4) LQ(4, 72, 4) LQ(2, 36, 4) LQ(4, 36, 4) LQ(8, 36, 4) LQ(4, 72, 8) LQ(4, 36, 8)

int main() {
    cudaStream_t s; cudaStreamCreate(&s);
    std::mt19937 rng(7); std::uniform_int_distribution<int> qd(-127, 127); std::uniform_real_distribution<float> ud(-1.f, 1.f);
    struct Cfg { const char * name; int K, N, xstride; } cfgs[] = {{"gate_up", 2304, 2304, 0}, {"down", 1152, 2304, 1152}};
    for (auto & c : cfgs) {
        const int NE = 128, slots = 8, NB = c.K / 32;
        const size_t qb = (size_t) NE * c.N * c.K, sb = (size_t) NE * c.N * NB;
        std::vector<int8_t> qs(qb); std::vector<__half> sc(sb);
        for (auto & v : qs) v = (int8_t) qd(rng);
        for (auto & v : sc) v = __float2half(0.01f * ud(rng));
        std::vector<float> x((size_t) slots * (c.xstride ? c.xstride : c.K));
        for (auto & v : x) v = ud(rng);
        std::vector<int> ids = {3, 17, 42, 60, 81, 99, 101, 127};
        int8_t * dq; __half * ds; float *dx, *dref, *dy; int * di;
        cudaMalloc(&dq, qb); cudaMalloc(&ds, sb * 2); cudaMalloc(&dx, x.size() * 4); cudaMalloc(&dref, 8 * c.N * 4); cudaMalloc(&dy, 8 * c.N * 4); cudaMalloc(&di, 32);
        cudaMemcpy(dq, qs.data(), qb, cudaMemcpyHostToDevice); cudaMemcpy(ds, sc.data(), sb * 2, cudaMemcpyHostToDevice);
        cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice); cudaMemcpy(di, ids.data(), 32, cudaMemcpyHostToDevice);
        launch_q8_fused(dq, ds, dx, c.xstride, dref, c.K, c.N, di, 8, s); cudaStreamSynchronize(s);
        std::vector<float> ref(8 * c.N); cudaMemcpy(ref.data(), dref, ref.size() * 4, cudaMemcpyDeviceToHost);
        double rmax = 0; for (float v : ref) rmax = std::max(rmax, (double) std::fabs(v));
        std::printf("== %s (K=%d N=%d)  ref max|y| %.3f\n", c.name, c.K, c.N, rmax);
        struct V { const char * n; void (*f)(const int8_t *, const __half *, const float *, int64_t, float *, int, int, const int *, cudaStream_t); int nb; };
        V vs[] = {{"current q8_experts_rows2 (2r/warp, 8w)", [](const int8_t * q, const __half * sc, const float * x, int64_t xs_, float * y, int K, int N, const int * ids, cudaStream_t st) { launch_q8_fused(q, sc, x, xs_, y, K, N, ids, 8, st); }, 0},
                  {"pack 2r/warp, 4w", lq_2_72_4, 72}, {"pack 4r/warp, 4w", lq_4_72_4, 72}, {"pack 4r/warp, 8w", lq_4_72_8, 72},
                  {"pack 2r/warp, 4w", lq_2_36_4, 36}, {"pack 4r/warp, 4w", lq_4_36_4, 36}, {"pack 8r/warp, 4w", lq_8_36_4, 36}, {"pack 4r/warp, 8w", lq_4_36_8, 36}};
        for (auto & v : vs) {
            if (v.nb && v.nb != NB) continue;
            cudaMemset(dy, 0, 8 * c.N * 4);
            v.f(dq, ds, dx, c.xstride, dy, c.K, c.N, di, s); cudaStreamSynchronize(s);
            std::vector<float> h(8 * c.N); cudaMemcpy(h.data(), dy, h.size() * 4, cudaMemcpyDeviceToHost);
            double md = 0; for (size_t i = 0; i < h.size(); ++i) md = std::max(md, (double) std::fabs(h[i] - ref[i]));
            float t = graph_us([&](cudaStream_t st) { v.f(dq, ds, dx, c.xstride, dy, c.K, c.N, di, st); }, s);
            std::printf("   %-40s %8.2f us  rel %.1e\n", v.n, t, md / rmax);
        }
        cudaFree(dq); cudaFree(ds); cudaFree(dx); cudaFree(dref); cudaFree(dy); cudaFree(di);
    }
    return 0;
}
