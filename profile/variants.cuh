// Variants of the row-packed NVFP4 expert GEMV. All compute y[slot][row] = sum_k w[row][k] * x[k] exactly like nvfp4_gemv_pack4 (same per-chunk math, same reduction order inside a row's chunks is NOT kept: lane partial sums are added in a different order only for split-K).
__device__ __forceinline__ uint4 ldg_stream(const uint4 * p) {      // 128-bit load, L1 no-allocate (weights are read once)
    uint4 r; asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p)); return r; }
__device__ __forceinline__ uint32_t ldg_stream32(const uint32_t * p) { uint32_t r; asm volatile("ld.global.nc.L1::no_allocate.u32 %0, [%1];" : "=r"(r) : "l"(p)); return r; }

// V1: identical to baseline but streaming loads (point 3 of the checklist).
template <int NCH, bool STREAM, int WPB>
__global__ void __launch_bounds__(WPB * 32)
nv_v1(const uint8_t * __restrict__ qs_all, const uint8_t * __restrict__ sc_all, const float * __restrict__ x, int64_t xss, float * __restrict__ y, int K, int N, const int * __restrict__ ids) {
    extern __shared__ float xs[]; __shared__ float kt[16];
    constexpr int T = WPB * 32, RP = 4, UNITS = RP * NCH;
    const int slot = blockIdx.y; const int64_t e = ids ? ids[slot] : 0;
    const uint8_t * qs = qs_all + e * (int64_t) N * (K / 2); const uint8_t * sc = sc_all + e * (int64_t) N * (K / 16); const float * xin = x + slot * xss;
    for (int i = threadIdx.x; i < K; i += T) { const int b = i >> 6, r = i & 63; xs[b * 68 + r] = xin[i]; }
    if (threadIdx.x < 16) kt[threadIdx.x] = kE2M1d[threadIdx.x];
    __syncthreads();
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31; const int rowbase = (blockIdx.x * WPB + warp) * RP;
    if (rowbase >= N) return;
    float acc[RP];
#pragma unroll
    for (int r = 0; r < RP; ++r) acc[r] = 0.0f;
#pragma unroll
    for (int u0 = 0; u0 < UNITS; u0 += 32) {
        const int u = u0 + lane;
        if (u < UNITS) {
            const int rs = u / NCH, c = u - rs * NCH; const int row = rowbase + rs; const int b = c >> 1, half = c & 1; float tot = 0.0f;
            if (row < N) {
                const float4 * xb = reinterpret_cast<const float4 *>(xs + b * 68 + half * 32); float4 xv[8];
#pragma unroll
                for (int i = 0; i < 8; ++i) xv[i] = xb[i];
                const float * xf = reinterpret_cast<const float *>(xv);
                uint4 a; uint32_t sw;
                if (STREAM) { a = ldg_stream(reinterpret_cast<const uint4 *>(qs + (size_t) row * (K / 2)) + c); sw = ldg_stream32(reinterpret_cast<const uint32_t *>(sc + (size_t) row * (K / 16)) + b); }
                else { a = __ldg(reinterpret_cast<const uint4 *>(qs + (size_t) row * (K / 2)) + c); sw = __ldg(reinterpret_cast<const uint32_t *>(sc + (size_t) row * (K / 16)) + b); }
                const uint32_t w[4] = {a.x, a.y, a.z, a.w};
#pragma unroll
                for (int h = 0; h < 2; ++h) { const float d = ue4m3_arith((sw >> (8 * (2 * half + h))) & 0xFF); float sub = 0.0f;
#pragma unroll
                    for (int j = 0; j < 8; ++j) { const int bi = h * 8 + j; const uint32_t byte = (w[bi >> 2] >> (8 * (bi & 3))) & 0xFF; sub += kt[byte & 15] * xf[h * 16 + j] + kt[byte >> 4] * xf[h * 16 + j + 8]; }
                    tot += d * sub; }
            }
#pragma unroll
            for (int r = 0; r < RP; ++r) acc[r] += (rs == r) ? tot : 0.0f;
        }
    }
#pragma unroll
    for (int r = 0; r < RP; ++r) { float a = acc[r]; for (int off = 16; off > 0; off >>= 1) a += __shfl_xor_sync(0xffffffffu, a, off); if (lane == 0 && rowbase + r < N) y[(size_t) slot * N + rowbase + r] = a; }
}

// V2: persistent, wave-aligned grid, one ROW per warp-task, static round-robin over rows (all tasks cost the same, so work stealing adds nothing but atomics).
// Bit-exact with nvfp4_gemv_pack4: row `rr` is the (rr & 3)-th row of its baseline 4-row group, so the lane that owned chunk c of that row in the baseline owns it here
// (unit u = rs*NCH + c, lane = u & 31, accumulated in ascending c) and the same butterfly reduction follows.
template <int NCH, int WPB>
__global__ void __launch_bounds__(WPB * 32)
nv_v2(const uint8_t * __restrict__ qs_all, const uint8_t * __restrict__ sc_all, const float * __restrict__ x, int64_t xss, float * __restrict__ y, int K, int N, const int * __restrict__ ids) {
    extern __shared__ float xs[]; __shared__ float kt[16];
    constexpr int T = WPB * 32;
    const int slot = blockIdx.y; const int64_t e = ids ? ids[slot] : 0;
    const uint8_t * qs = qs_all + e * (int64_t) N * (K / 2); const uint8_t * sc = sc_all + e * (int64_t) N * (K / 16); const float * xin = x + slot * xss;
    for (int i = threadIdx.x; i < K; i += T) { const int b = i >> 6, r = i & 63; xs[b * 68 + r] = xin[i]; }
    if (threadIdx.x < 16) kt[threadIdx.x] = kE2M1d[threadIdx.x];
    __syncthreads();
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    for (int row = blockIdx.x * WPB + warp; row < N; row += gridDim.x * WPB) {
        const int rs = row & 3;
        const int first = ((lane - rs * NCH) % 32 + 32) % 32;
        float acc = 0.0f;
#pragma unroll
        for (int c = first; c < NCH; c += 32) {
            const int b = c >> 1, half = c & 1; float tot = 0.0f;
            const float4 * xb = reinterpret_cast<const float4 *>(xs + b * 68 + half * 32); float4 xv[8];
#pragma unroll
            for (int i = 0; i < 8; ++i) xv[i] = xb[i];
            const float * xf = reinterpret_cast<const float *>(xv);
            const uint4 a = ldg_stream(reinterpret_cast<const uint4 *>(qs + (size_t) row * (K / 2)) + c);
            const uint32_t sw = ldg_stream32(reinterpret_cast<const uint32_t *>(sc + (size_t) row * (K / 16)) + b);
            const uint32_t w[4] = {a.x, a.y, a.z, a.w};
#pragma unroll
            for (int h = 0; h < 2; ++h) { const float d = ue4m3_arith((sw >> (8 * (2 * half + h))) & 0xFF); float sub = 0.0f;
#pragma unroll
                for (int j = 0; j < 8; ++j) { const int bi = h * 8 + j; const uint32_t byte = (w[bi >> 2] >> (8 * (bi & 3))) & 0xFF; sub += kt[byte & 15] * xf[h * 16 + j] + kt[byte >> 4] * xf[h * 16 + j + 8]; }
                tot += d * sub; }
            acc += tot;
        }
        for (int off = 16; off > 0; off >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, off);
        if (lane == 0) y[(size_t) slot * N + row] = acc;
    }
}

// V3: like V2 but a task is RPT consecutive rows whose loads are all issued before any math (memory-level parallelism like the baseline's 4-row group).
template <int NCH, int WPB, int RPT>
__global__ void __launch_bounds__(WPB * 32)
nv_v3(const uint8_t * __restrict__ qs_all, const uint8_t * __restrict__ sc_all, const float * __restrict__ x, int64_t xss, float * __restrict__ y, int K, int N, const int * __restrict__ ids) {
    extern __shared__ float xs[]; __shared__ float kt[16];
    constexpr int T = WPB * 32, CPL = (NCH + 31) / 32;
    const int slot = blockIdx.y; const int64_t e = ids ? ids[slot] : 0;
    const uint8_t * qs = qs_all + e * (int64_t) N * (K / 2); const uint8_t * sc = sc_all + e * (int64_t) N * (K / 16); const float * xin = x + slot * xss;
    for (int i = threadIdx.x; i < K; i += T) { const int b = i >> 6, r = i & 63; xs[b * 68 + r] = xin[i]; }
    if (threadIdx.x < 16) kt[threadIdx.x] = kE2M1d[threadIdx.x];
    __syncthreads();
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    for (int task = blockIdx.x * WPB + warp; task * RPT < N; task += gridDim.x * WPB) {
        uint4 a[RPT][CPL]; uint32_t sw[RPT][CPL];
#pragma unroll
        for (int j = 0; j < RPT; ++j) {
            const int row = task * RPT + j, rs = row & 3, first = ((lane - rs * NCH) % 32 + 32) % 32;
#pragma unroll
            for (int i = 0; i < CPL; ++i) { const int c = first + 32 * i;
                if (c < NCH) { a[j][i] = ldg_stream(reinterpret_cast<const uint4 *>(qs + (size_t) row * (K / 2)) + c); sw[j][i] = ldg_stream32(reinterpret_cast<const uint32_t *>(sc + (size_t) row * (K / 16)) + (c >> 1)); } }
        }
#pragma unroll
        for (int j = 0; j < RPT; ++j) {
            const int row = task * RPT + j, rs = row & 3, first = ((lane - rs * NCH) % 32 + 32) % 32;
            float acc = 0.0f;
#pragma unroll
            for (int i = 0; i < CPL; ++i) { const int c = first + 32 * i;
                if (c < NCH) {
                    const int b = c >> 1, half = c & 1; float tot = 0.0f;
                    const float4 * xb = reinterpret_cast<const float4 *>(xs + b * 68 + half * 32); float4 xv[8];
#pragma unroll
                    for (int q = 0; q < 8; ++q) xv[q] = xb[q];
                    const float * xf = reinterpret_cast<const float *>(xv); const uint32_t w[4] = {a[j][i].x, a[j][i].y, a[j][i].z, a[j][i].w};
#pragma unroll
                    for (int h = 0; h < 2; ++h) { const float d = ue4m3_arith((sw[j][i] >> (8 * (2 * half + h))) & 0xFF); float sub = 0.0f;
#pragma unroll
                        for (int jj = 0; jj < 8; ++jj) { const int bi = h * 8 + jj; const uint32_t byte = (w[bi >> 2] >> (8 * (bi & 3))) & 0xFF; sub += kt[byte & 15] * xf[h * 16 + jj] + kt[byte >> 4] * xf[h * 16 + jj + 8]; }
                        tot += d * sub; }
                    acc += tot;
                } }
            for (int off = 16; off > 0; off >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, off);
            if (lane == 0) y[(size_t) slot * N + row] = acc;
        }
    }
}

// V4: baseline math and mapping (bit-exact), but the first PF rounds of weight loads are issued BEFORE the x vector is staged to shared memory, so the
// staging latency and the first-load latency overlap.
template <int NCH, int PF>
__global__ void __launch_bounds__(128)
nv_v4(const uint8_t * __restrict__ qs_all, const uint8_t * __restrict__ sc_all, const float * __restrict__ x, int64_t xss, float * __restrict__ y, int K, int N, const int * __restrict__ ids) {
    extern __shared__ float xs[]; __shared__ float kt[16];
    constexpr int T = 128, RP = 4, UNITS = RP * NCH;
    const int slot = blockIdx.y; const int64_t e = ids ? ids[slot] : 0;
    const uint8_t * qs = qs_all + e * (int64_t) N * (K / 2); const uint8_t * sc = sc_all + e * (int64_t) N * (K / 16); const float * xin = x + slot * xss;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31; const int rowbase = (blockIdx.x * 4 + warp) * RP;
    uint4 pa[PF]; uint32_t ps[PF];
#pragma unroll
    for (int p = 0; p < PF; ++p) { const int u = p * 32 + lane;
        if (u < UNITS && rowbase < N) { const int rs = u / NCH, c = u - rs * NCH, row = rowbase + rs;
            pa[p] = ldg_stream(reinterpret_cast<const uint4 *>(qs + (size_t) row * (K / 2)) + c); ps[p] = ldg_stream32(reinterpret_cast<const uint32_t *>(sc + (size_t) row * (K / 16)) + (c >> 1)); } }
    for (int i = threadIdx.x; i < K; i += T) { const int b = i >> 6, r = i & 63; xs[b * 68 + r] = xin[i]; }
    if (threadIdx.x < 16) kt[threadIdx.x] = kE2M1d[threadIdx.x];
    __syncthreads();
    if (rowbase >= N) return;
    float acc[RP];
#pragma unroll
    for (int r = 0; r < RP; ++r) acc[r] = 0.0f;
#pragma unroll
    for (int u0 = 0; u0 < UNITS; u0 += 32) {
        const int u = u0 + lane;
        if (u < UNITS) {
            const int rs = u / NCH, c = u - rs * NCH; const int row = rowbase + rs; const int b = c >> 1, half = c & 1; float tot = 0.0f;
            const float4 * xb = reinterpret_cast<const float4 *>(xs + b * 68 + half * 32); float4 xv[8];
#pragma unroll
            for (int i = 0; i < 8; ++i) xv[i] = xb[i];
            const float * xf = reinterpret_cast<const float *>(xv);
            uint4 a; uint32_t sw; const int p = u0 / 32;
            if (p < PF) { a = pa[p < PF ? p : 0]; sw = ps[p < PF ? p : 0]; }
            else { a = ldg_stream(reinterpret_cast<const uint4 *>(qs + (size_t) row * (K / 2)) + c); sw = ldg_stream32(reinterpret_cast<const uint32_t *>(sc + (size_t) row * (K / 16)) + b); }
            const uint32_t w[4] = {a.x, a.y, a.z, a.w};
#pragma unroll
            for (int h = 0; h < 2; ++h) { const float d = ue4m3_arith((sw >> (8 * (2 * half + h))) & 0xFF); float sub = 0.0f;
#pragma unroll
                for (int j = 0; j < 8; ++j) { const int bi = h * 8 + j; const uint32_t byte = (w[bi >> 2] >> (8 * (bi & 3))) & 0xFF; sub += kt[byte & 15] * xf[h * 16 + j] + kt[byte >> 4] * xf[h * 16 + j + 8]; }
                tot += d * sub; }
#pragma unroll
            for (int r = 0; r < RP; ++r) acc[r] += (rs == r) ? tot : 0.0f;
        }
    }
#pragma unroll
    for (int r = 0; r < RP; ++r) { float a = acc[r]; for (int off = 16; off > 0; off >>= 1) a += __shfl_xor_sync(0xffffffffu, a, off); if (lane == 0 && rowbase + r < N) y[(size_t) slot * N + rowbase + r] = a; }
}
