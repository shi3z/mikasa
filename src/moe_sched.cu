// MoE expert GEMV scheduling: production (one launch, grid = rows/16 x 8 slots) vs persistent CTAs
// that walk slot-major work items and stage x once per slot. Math per row is identical (R=4, LUT, 128-bit loads).
#include "kernels_common.cuh"
#include <cstdio>
#include <vector>
#include <random>
#include <cmath>

// work item = (slot, tile of 16 rows); items are slot-major, each CTA takes a contiguous range.
__global__ void __launch_bounds__(128, 4)
nv_persist(const uint8_t * __restrict__ qs_all, const uint8_t * __restrict__ sc_all, const float * __restrict__ x,
           int64_t x_slot_stride, float * __restrict__ y, int K, int N, const int * __restrict__ ids, int slots) {
    extern __shared__ float xs[];
    __shared__ float lut_ue[256];
    __shared__ float2 lut2[256];
    constexpr int R = 4, T = 128;
    const int nb = K / 64;
    const int tps = (N + 15) / 16;              // tiles per slot
    const int items = tps * slots;
    const int per = (items + gridDim.x - 1) / gridDim.x;
    const int i0 = blockIdx.x * per, i1 = min(items, i0 + per);
    for (int i = threadIdx.x; i < 256; i += T) { lut_ue[i] = ue4m3_dev((uint8_t) i); lut2[i] = make_float2(kE2M1d[i & 15], kE2M1d[i >> 4]); }
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int cur_slot = -1;
    for (int it = i0; it < i1; ++it) {
        const int slot = it / tps, tile = it % tps;
        if (slot != cur_slot) {
            __syncthreads();                     // previous item's xs readers are done (also covers LUT init on first)
            const float * xin = x + slot * x_slot_stride;
            for (int i = threadIdx.x; i < K; i += T) { const int b = i >> 6, r = i & 63; xs[b * 68 + r] = xin[i]; }
            __syncthreads();
            cur_slot = slot;
        }
        const int64_t e = ids ? ids[slot] : 0;
        const uint8_t * qs = qs_all + e * (int64_t) N * (K / 2);
        const uint8_t * sc = sc_all + e * (int64_t) N * (K / 16);
        const int row0 = tile * 16 + warp * R;
        if (row0 >= N) continue;
        float acc[R];
#pragma unroll
        for (int r = 0; r < R; ++r) acc[r] = 0.0f;
        for (int b = lane; b < nb; b += 32) {
            const float4 * xb = reinterpret_cast<const float4 *>(xs + b * 68);
            float4 xv[16];
#pragma unroll
            for (int i = 0; i < 16; ++i) xv[i] = xb[i];
#pragma unroll
            for (int r = 0; r < R; ++r) {
                const int row = row0 + r;
                if (row < N) {
                    const uint4 * q = reinterpret_cast<const uint4 *>(qs + (size_t) row * (K / 2));
                    const uint32_t sw = __ldg(reinterpret_cast<const uint32_t *>(sc + (size_t) row * (K / 16)) + b);
                    const uint4 a0 = __ldg(q + b * 2), a1 = __ldg(q + b * 2 + 1);
                    const uint32_t w[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
                    float tot = 0.0f;
#pragma unroll
                    for (int s2 = 0; s2 < 4; ++s2) {
                        const float d = lut_ue[(sw >> (8 * s2)) & 0xFF];
                        float sub = 0.0f;
#pragma unroll
                        for (int j = 0; j < 8; ++j) {
                            const int bi = s2 * 8 + j;
                            const uint32_t byte = (w[bi >> 2] >> (8 * (bi & 3))) & 0xFF;
                            const float xl = reinterpret_cast<const float *>(&xv[(s2 * 16 + j) >> 2])[(s2 * 16 + j) & 3];
                            const float xh = reinterpret_cast<const float *>(&xv[(s2 * 16 + j + 8) >> 2])[(s2 * 16 + j + 8) & 3];
                            const float2 v = lut2[byte];
                            sub += v.x * xl + v.y * xh;
                        }
                        tot += d * sub;
                    }
                    acc[r] += tot;
                }
            }
        }
#pragma unroll
        for (int r = 0; r < R; ++r) {
            float a = acc[r];
            for (int off = 16; off > 0; off >>= 1) a += __shfl_xor_sync(0xffffffffu, a, off);
            if (lane == 0 && row0 + r < N) y[(size_t) slot * N + row0 + r] = a;
        }
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

int main() {
    cudaStream_t s; cudaStreamCreate(&s);
    int sms = 0; cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
    std::mt19937 rng(3); std::uniform_int_distribution<int> qd(0, 255);
    struct Cfg { const char * name; int K, N; int xstride; } cfgs[] = {{"gate_up", 2304, 2304, 0}, {"down", 1152, 2304, 1152}};
    for (auto & c : cfgs) {
        const int NE = 128, slots = 8;
        const size_t qbytes = (size_t) NE * c.N * (c.K / 2), sbytes = (size_t) NE * c.N * (c.K / 16);
        std::vector<uint8_t> qs(qbytes), sc(sbytes);
        for (auto & v : qs) v = qd(rng);
        for (auto & v : sc) v = 0x38 + (qd(rng) & 7);
        std::vector<float> x((size_t) slots * (c.xstride ? c.xstride : c.K));
        for (auto & v : x) v = (float) (qd(rng) - 128) / 128.f;
        std::vector<int> ids = {3, 17, 42, 60, 81, 99, 101, 127};
        uint8_t *dq, *ds; float *dx, *dref, *dy; int *di;
        cudaMalloc(&dq, qbytes); cudaMalloc(&ds, sbytes); cudaMalloc(&dx, x.size() * 4); cudaMalloc(&dref, 8 * c.N * 4);
        cudaMalloc(&dy, 8 * c.N * 4); cudaMalloc(&di, 32);
        cudaMemcpy(dq, qs.data(), qbytes, cudaMemcpyHostToDevice); cudaMemcpy(ds, sc.data(), sbytes, cudaMemcpyHostToDevice);
        cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice); cudaMemcpy(di, ids.data(), 32, cudaMemcpyHostToDevice);
        const size_t smem = (size_t) (c.K / 64) * 68 * sizeof(float);
        // production reference
        launch_nvfp4_fused(dq, ds, dx, c.xstride, dref, c.K, c.N, di, 8, s); cudaStreamSynchronize(s);
        std::vector<float> ref(8 * c.N);
        cudaMemcpy(ref.data(), dref, ref.size() * 4, cudaMemcpyDeviceToHost);
        float t_prod = graph_us([&](cudaStream_t st) { launch_nvfp4_fused(dq, ds, dx, c.xstride, dy, c.K, c.N, di, 8, st); }, s);
        std::printf("== %s  production (grid %d x 8 slots, 128 thr): %7.2f us\n", c.name, (c.N + 15) / 16, t_prod);
        for (int ctas_per_sm : {1, 2, 4}) {
            const int G = sms * ctas_per_sm;
            float t = graph_us([&](cudaStream_t st) {
                nv_persist<<<G, 128, smem, st>>>(dq, ds, dx, c.xstride, dy, c.K, c.N, di, 8); }, s);
            nv_persist<<<G, 128, smem, s>>>(dq, ds, dx, c.xstride, dy, c.K, c.N, di, 8);
            cudaStreamSynchronize(s);
            std::vector<float> h(8 * c.N);
            cudaMemcpy(h.data(), dy, h.size() * 4, cudaMemcpyDeviceToHost);
            double md = 0; for (size_t i = 0; i < h.size(); ++i) md = std::max(md, (double) std::fabs(h[i] - ref[i]));
            std::printf("   persistent %d CTA/SM (grid %d): %7.2f us  max|diff| %.1e\n", ctas_per_sm, G, t, md);
        }
        cudaFree(dq); cudaFree(ds); cudaFree(dx); cudaFree(dref); cudaFree(dy); cudaFree(di);
    }
    return 0;
}
