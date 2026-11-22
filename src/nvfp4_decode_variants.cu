// NVFP4 fused expert GEMV: decode variants compared under identical structure (R=4, T=128, smem x, 8 experts).
// DEC 0: smem LUT (production), DEC 1: bit-op E2M1 decode, DEC 2: PRMT magnitude table + sign by integer ops.
#include "kernels_common.cuh"
#include <cstdio>
#include <vector>
#include <random>
#include <cmath>

// Doubled E2M1 value from a 4-bit code by bit operations.
__device__ __forceinline__ float e2m1x2_bits(uint32_t nib) {
    uint32_t e = (nib >> 1) & 3, m = nib & 1;
    float mag = (e == 0) ? (float) m : __int_as_float(((e + 127) << 23) | (m << 22));
    return (nib & 8) ? -mag : mag;
}

template <int DEC>
__device__ __forceinline__ void dec_pair(uint32_t byte, const float2 * lut2, float & lo, float & hi) {
    if (DEC == 0) {
        const float2 v = lut2[byte];
        lo = v.x; hi = v.y;
    } else if (DEC == 1) {
        lo = e2m1x2_bits(byte & 0xF);
        hi = e2m1x2_bits(byte >> 4);
    } else if (DEC == 3) {
        // warp-distributed LUT: lane l holds value kE2M1d[l & 15]; lookup = shuffle from lane `code`
        const float tv = kE2M1d[threadIdx.x & 15];
        lo = __shfl_sync(0xffffffffu, tv, byte & 0xF);
        hi = __shfl_sync(0xffffffffu, tv, byte >> 4);
    } else {
        // magnitude table {0,1,2,3,4,6,8,12} packed in two words; PRMT picks the byte by the 3-bit index.
        const uint32_t tlo = 0x03020100u, thi = 0x0C080604u;
        const uint32_t nl = byte & 0xF, nh = byte >> 4;
        const uint32_t ml = __byte_perm(tlo, thi, nl & 7) & 0xFF;
        const uint32_t mh = __byte_perm(tlo, thi, nh & 7) & 0xFF;
        const int32_t sl = ((int32_t) (nl << 28)) >> 31, sh = ((int32_t) (nh << 28)) >> 31;
        lo = (float) (int32_t) ((ml ^ sl) - sl);
        hi = (float) (int32_t) ((mh ^ sh) - sh);
    }
}

template <int DEC>
__global__ void nv_var(const uint8_t * __restrict__ qs_all, const uint8_t * __restrict__ sc_all, const float * __restrict__ x,
                       int64_t x_slot_stride, float * __restrict__ y, int K, int N, const int * __restrict__ ids) {
    extern __shared__ float xs[];
    __shared__ float lut_ue[256];
    __shared__ float2 lut2[256];
    constexpr int R = 4, T = 128;
    const int nb = K / 64, slot = blockIdx.y;
    const int64_t e = ids ? ids[slot] : 0;
    const uint8_t * qs = qs_all + e * (int64_t) N * (K / 2);
    const uint8_t * sc = sc_all + e * (int64_t) N * (K / 16);
    const float * xin = x + slot * x_slot_stride;
    for (int i = threadIdx.x; i < K; i += T) { const int b = i >> 6, r = i & 63; xs[b * 68 + r] = xin[i]; }
    for (int i = threadIdx.x; i < 256; i += T) { lut_ue[i] = ue4m3_dev((uint8_t) i); lut2[i] = make_float2(kE2M1d[i & 15], kE2M1d[i >> 4]); }
    __syncthreads();
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int row0 = (blockIdx.x * (T / 32) + warp) * R;
    if (row0 >= N) return;
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
                        float lo, hi;
                        dec_pair<DEC>(byte, lut2, lo, hi);
                        sub += lo * xl + hi * xh;
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

template <int DEC>
static void run(const char * name, int K, int N, uint8_t * qs, uint8_t * sc, float * x, int * ids, float * ref, cudaStream_t s) {
    const size_t smem = (size_t) (K / 64) * 68 * sizeof(float);
    float * y; cudaMalloc(&y, 8 * N * 4);
    dim3 grid((N + 15) / 16, 8);  // 4 warps x 4 rows per block
    auto f = [&](cudaStream_t st) { nv_var<DEC><<<grid, 128, smem, st>>>(qs, sc, x, 0, y, K, N, ids); };
    float us = graph_us(f, s);
    f(s); cudaStreamSynchronize(s);
    std::vector<float> h((size_t) 8 * N), r((size_t) 8 * N);
    cudaMemcpy(h.data(), y, h.size() * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(r.data(), ref, r.size() * 4, cudaMemcpyDeviceToHost);
    double md = 0; for (size_t i = 0; i < h.size(); ++i) md = std::max(md, (double) std::fabs(h[i] - r[i]));
    cudaFuncAttributes a; cudaFuncGetAttributes(&a, (const void *) nv_var<DEC>);
    double bytes = 8.0 * N * K * 36.0 / 64.0;
    std::printf("%-8s DEC=%d  %8.2f us  %7.1f GB/s  regs %3d  spill(local) %u B  max|diff| %.1e\n", name, DEC, us,
                bytes / (us * 1e-6) / 1e9, a.numRegs, (unsigned) a.localSizeBytes, md);
    cudaFree(y);
}

int main() {
    cudaStream_t s; cudaStreamCreate(&s);
    std::mt19937 rng(3); std::uniform_int_distribution<int> qd(0, 255);
    struct Cfg { const char * name; int K, N; } cfgs[] = {{"gate_up", 2304, 2304}, {"down", 1152, 2304}};
    for (auto & c : cfgs) {
        const int NE = 128;
        const size_t qbytes = (size_t) NE * c.N * (c.K / 2), sbytes = (size_t) NE * c.N * (c.K / 16);
        std::vector<uint8_t> qs(qbytes), sc(sbytes);
        for (auto & v : qs) v = qd(rng);
        for (auto & v : sc) v = 0x38 + (qd(rng) & 7);
        std::vector<float> x((size_t) c.K);
        for (auto & v : x) v = (float) (qd(rng) - 128) / 128.f;
        std::vector<int> ids = {3, 17, 42, 60, 81, 99, 101, 127};
        uint8_t *dq, *ds; float *dx, *dref; int *di;
        cudaMalloc(&dq, qbytes); cudaMalloc(&ds, sbytes); cudaMalloc(&dx, c.K * 4); cudaMalloc(&dref, 8 * c.N * 4); cudaMalloc(&di, 32);
        cudaMemcpy(dq, qs.data(), qbytes, cudaMemcpyHostToDevice); cudaMemcpy(ds, sc.data(), sbytes, cudaMemcpyHostToDevice);
        cudaMemcpy(dx, x.data(), c.K * 4, cudaMemcpyHostToDevice); cudaMemcpy(di, ids.data(), 32, cudaMemcpyHostToDevice);
        launch_nvfp4_fused(dq, ds, dx, 0, dref, c.K, c.N, di, 8, s); cudaStreamSynchronize(s);
        std::printf("== %s (K=%d N=%d, 8 experts)\n", c.name, c.K, c.N);
        run<0>(c.name, c.K, c.N, dq, ds, dx, di, dref, s);
        run<1>(c.name, c.K, c.N, dq, ds, dx, di, dref, s);
        run<2>(c.name, c.K, c.N, dq, ds, dx, di, dref, s);
        run<3>(c.name, c.K, c.N, dq, ds, dx, di, dref, s);
        cudaFree(dq); cudaFree(ds); cudaFree(dx); cudaFree(dref); cudaFree(di);
    }
    return 0;
}
