// Standalone test of pf_attn.cuh against kb_attn_ref (no model needed).
#include "gguf.hpp"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#define CHECK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::fprintf(stderr, "CUDA %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); std::exit(1); } } while (0)
#include "long_kernels.cuh"
#include "pf_attn.cuh"

__device__ __forceinline__ float hrand(uint64_t i, uint32_t seed) {   // ~N(0,1) from 4 hashed uniforms
    float a = 0;
    for (int k = 0; k < 4; ++k) {
        uint64_t x = (i * 4 + k) * 0x9E3779B97F4A7C15ull + seed * 0xBF58476D1CE4E5B9ull;
        x ^= x >> 31; x *= 0x94D049BB133111EBull; x ^= x >> 29;
        a += (float) (x & 0xFFFFFF) / 16777216.0f;
    }
    return (a - 2.0f) * 1.7320508f;
}
__global__ void fill_gauss(float * p, size_t n, float sc, uint32_t seed) { size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x; if (i < n) p[i] = sc * hrand(i, seed); }
// KV rows: [rows][8][128]; K: scaled to norm <= 22.6 per head row (clip like real), V: rms-normalised
template <typename KT>
__global__ void fill_kv(KT * k, KT * v, size_t rows, uint32_t seed, float kscale) {
    size_t r = blockIdx.x * (size_t) 8 + blockIdx.y;   // (row, head)
    if (r >= rows * 8) return;
    __shared__ float red[2];
    float kk = kscale * hrand(r * 128 + threadIdx.x, seed), vv = hrand(r * 128 + threadIdx.x, seed + 77);
    float s1 = kk * kk, s2 = vv * vv;
    for (int o = 16; o; o >>= 1) { s1 += __shfl_xor_sync(~0u, s1, o); s2 += __shfl_xor_sync(~0u, s2, o); }
    __shared__ float a[4], b[4];
    if ((threadIdx.x & 31) == 0) { a[threadIdx.x >> 5] = s1; b[threadIdx.x >> 5] = s2; }
    __syncthreads();
    s1 = a[0] + a[1] + a[2] + a[3]; s2 = b[0] + b[1] + b[2] + b[3];
    float f = fminf(1.0f, 22.627f / (sqrtf(s1) + 1e-6f));
    kv_store(k + r * 128 + threadIdx.x, kk * f);
    kv_store(v + r * 128 + threadIdx.x, vv * rsqrtf(s2 / 128 + 1e-6f));
}
__global__ void set_pos(int * p, int pos0, int T) { int i = threadIdx.x + blockIdx.x * blockDim.x; if (i < T) p[i] = pos0 + i; }
__global__ void h2f(const __half * a, float * b, size_t n) { size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x; if (i < n) b[i] = __half2float(a[i]); }

int main(int argc, char ** argv) {
    const int TM = 256;
    float *Q, *R, *relW, *sink, *G, *Og, *Sref, *Oref, *gref, *gnew;
    __half * out;
    int * pos, * slot;
    CHECK(cudaMalloc(&Q, TM * 3072 * 4)); CHECK(cudaMalloc(&R, TM * 384 * 4)); CHECK(cudaMalloc(&relW, 1024 * 16 * 4));
    CHECK(cudaMalloc(&sink, TM * 24 * 4)); CHECK(cudaMalloc(&G, TM * 3072 * 4)); CHECK(cudaMalloc(&Og, TM * 3072 * 4));
    CHECK(cudaMalloc(&Oref, TM * 3072 * 4)); CHECK(cudaMalloc(&gref, TM * 3072 * 4)); CHECK(cudaMalloc(&gnew, TM * 3072 * 4));
    CHECK(cudaMalloc(&out, TM * 3072 * 2)); CHECK(cudaMalloc(&pos, TM * 4)); CHECK(cudaMalloc(&slot, TM * 4)); CHECK(cudaMemset(slot, 0, TM * 4));
    const size_t smax = 65536 + 16;
    const bool do_ref = !(argc > 1 && std::string(argv[1]) == "noref");
    size_t sref_n = do_ref ? (size_t) 128 * 24 * 33000 : 1;   // ref scratch sized for max 128 queries at <= 33000 positions
    CHECK(cudaMalloc(&Sref, sref_n * 4));
    fill_gauss<<<(TM * 3072 + 255) / 256, 256>>>(Q, TM * 3072, 0.9f, 1);   // head norm ~ 10
    fill_gauss<<<(TM * 384 + 255) / 256, 256>>>(R, TM * 384, 1.5f, 2);
    fill_gauss<<<(1024 * 16 + 255) / 256, 256>>>(relW, 1024 * 16, 0.3f, 3);
    fill_gauss<<<(TM * 24 + 255) / 256, 256>>>(sink, TM * 24, 1.0f, 4);
    fill_gauss<<<(TM * 3072 + 255) / 256, 256>>>(G, TM * 3072, 1.0f, 5);
    // caches
    const int ring = 1024 + 256;
    KVSlots kswa{}, kf16{}, kf32{};
    CHECK(cudaMalloc(&kswa.k[0], (size_t) ring * 1024 * 4)); CHECK(cudaMalloc(&kswa.v[0], (size_t) ring * 1024 * 4));
    CHECK(cudaMalloc(&kf16.k[0], smax * 1024 * 2)); CHECK(cudaMalloc(&kf16.v[0], smax * 1024 * 2));
    CHECK(cudaMalloc(&kf32.k[0], smax * 1024 * 4)); CHECK(cudaMalloc(&kf32.v[0], smax * 1024 * 4));
    fill_kv<float><<<dim3(ring, 8), 128>>>((float *) kswa.k[0], (float *) kswa.v[0], ring, 10, 1.2f);
    fill_kv<__half><<<dim3(smax, 8), 128>>>((__half *) kf16.k[0], (__half *) kf16.v[0], smax, 11, 1.2f);
    fill_kv<float><<<dim3(smax, 8), 128>>>((float *) kf32.k[0], (float *) kf32.v[0], smax, 11, 1.2f);
    CHECK(cudaDeviceSynchronize());
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    struct Case { const char * name; int kind; int pos0, T; };   // kind 0 swa, 1 full16, 2 full32
    std::vector<Case> cases;
    for (int kind = 0; kind < 3; ++kind)
        for (int p : {0, 100, 1000, 5000, 32000}) cases.push_back({kind == 0 ? "swa" : kind == 1 ? "full-fp16" : "full-fp32", kind, p, 256});
    for (int kind = 0; kind < 3; ++kind) cases.push_back({kind == 0 ? "swa" : kind == 1 ? "full-fp16" : "full-fp32", kind, 1000, 200});
    for (int kind = 0; kind < 3; ++kind) cases.push_back({kind == 0 ? "swa" : kind == 1 ? "full-fp16" : "full-fp32", kind, 63, 37});
    std::printf("%-10s %6s %4s | %10s %10s | %9s %9s\n", "kind", "pos0", "T", "max_abs", "max_rel", "ms", "TFLOP/s");
    for (auto & c : cases) {
        const int T = c.T;
        set_pos<<<1, 256>>>(pos, c.pos0, T);
        const KVSlots & kv = c.kind == 0 ? kswa : c.kind == 1 ? kf16 : kf32;
        const int r = c.kind == 0 ? ring : 0, win = c.kind == 0 ? 1024 : 0;
        auto run = [&]() {
            if (c.kind == 1) pf_attention<__half>(Q, R, relW, sink, G, kv, r, win, pos, slot, T, out, 0);
            else pf_attention<float>(Q, R, relW, sink, G, kv, r, win, pos, slot, T, out, 0);
        };
        run(); CHECK(cudaDeviceSynchronize());
        float best = 1e9;
        for (int rep = 0; rep < 5; ++rep) {
            cudaEventRecord(e0);
            for (int i = 0; i < 10; ++i) run();
            cudaEventRecord(e1); cudaEventSynchronize(e1);
            float ms; cudaEventElapsedTime(&ms, e0, e1); best = std::min(best, ms / 10);
        }
        double vis = 0;
        for (int t = 0; t < T; ++t) { int p = c.pos0 + t; vis += c.kind == 0 ? std::min(p + 1, 1024) : p + 1; }
        const double tf = 4.0 * 128 * 24 * vis / (best * 1e-3) / 1e12;
        double mabs = -1, mrel = -1;
        // reference (naive); chunks of <=128 queries so the scratch fits
        if (do_ref && c.pos0 + T <= 33000) {
            std::vector<float> a((size_t) T * 3072), b((size_t) T * 3072);
            h2f<<<(T * 3072 + 255) / 256, 256>>>(out, gnew, (size_t) T * 3072);
            CHECK(cudaMemcpy(a.data(), gnew, a.size() * 4, cudaMemcpyDeviceToHost));
            for (int t0 = 0; t0 < T; t0 += 128) {
                const int nb = std::min(128, T - t0);
                const size_t smaxq = c.pos0 + T + 1;
                if ((size_t) nb * 24 * smaxq > sref_n) { std::fprintf(stderr, "ref scratch too small\n"); return 1; }
                if (c.kind == 1)
                    kb_attn_ref<__half><<<dim3(24, nb), 256>>>(Q + (size_t) t0 * 3072, kv, r, R + (size_t) t0 * 384, relW, sink + t0 * 24, pos + t0, slot, win, Sref, smaxq, Oref, G + (size_t) t0 * 3072, gref);
                else
                    kb_attn_ref<float><<<dim3(24, nb), 256>>>(Q + (size_t) t0 * 3072, kv, r, R + (size_t) t0 * 384, relW, sink + t0 * 24, pos + t0, slot, win, Sref, smaxq, Oref, G + (size_t) t0 * 3072, gref);
                CHECK(cudaDeviceSynchronize());
                CHECK(cudaMemcpy(b.data() + (size_t) t0 * 3072, gref, (size_t) nb * 3072 * 4, cudaMemcpyDeviceToHost));
            }
            double mx = 0;
            for (size_t i = 0; i < a.size(); ++i) mx = std::max(mx, (double) std::fabs(b[i]));
            mabs = 0;
            for (size_t i = 0; i < a.size(); ++i) { double d = std::fabs(a[i] - b[i]); if (!(d <= 1e30)) d = 1e30; mabs = std::max(mabs, d); }
            mrel = mabs / mx;
        }
        std::printf("%-10s %6d %4d | %10.3e %10.3e | %9.4f %9.2f\n", c.name, c.pos0, T, mabs, mrel, best, tf);
        std::fflush(stdout);
    }
    // extra timings
    for (int p : {64000 - 256}) {
        set_pos<<<1, 256>>>(pos, p, 256);
        auto run = [&]() { pf_attention<__half>(Q, R, relW, sink, G, kf16, 0, 0, pos, slot, 256, out, 0); };
        run(); CHECK(cudaDeviceSynchronize());
        float best = 1e9;
        for (int rep = 0; rep < 5; ++rep) { cudaEventRecord(e0); for (int i = 0; i < 10; ++i) run(); cudaEventRecord(e1); cudaEventSynchronize(e1); float ms; cudaEventElapsedTime(&ms, e0, e1); best = std::min(best, ms / 10); }
        double vis = 0; for (int t = 0; t < 256; ++t) vis += p + t + 1;
        std::printf("full-fp16 pos0=%d T=256: %.4f ms, %.2f TFLOP/s\n", p, best, 4.0 * 128 * 24 * vis / (best * 1e-3) / 1e12);
    }
    return 0;
}
