// Fused router: 128 blocks compute one logit each; the LAST block to finish (atomic ticket) does the top-8 with ONE warp (no __syncthreads loop).
__device__ unsigned int g_ticket = 0;
__global__ void k_router_fused(const float * __restrict__ W, const float * __restrict__ x, const float * __restrict__ bias,
                               float * __restrict__ logits, int * __restrict__ ids, float * __restrict__ wts, float scale) {
    __shared__ float red[32]; __shared__ bool last;
    const int row = blockIdx.x; const float * wr = W + (size_t) row * E;
    float w[RMS_NV], acc = 0.0f;
#pragma unroll
    for (int j = 0; j < RMS_NV; ++j) { const int i = threadIdx.x + 256 * j; w[j] = (i < E) ? wr[i] : 0.0f; }
#pragma unroll
    for (int j = 0; j < RMS_NV; ++j) { const int i = threadIdx.x + 256 * j; if (i < E) acc += w[j] * x[i]; }
    const float t = block_sum(acc, red);
    if (threadIdx.x == 0) { logits[row] = t; __threadfence(); last = (atomicAdd(&g_ticket, 1u) == gridDim.x - 1); }
    __syncthreads();
    if (!last || threadIdx.x >= 32) return;
    __threadfence();
    const int lane = threadIdx.x; float pr[4], sel[4]; bool used[4] = {false, false, false, false}; int ch[8]; float cp[8];
#pragma unroll
    for (int j = 0; j < 4; ++j) { const int e = lane + 32 * j; const float l = __ldcg(&logits[e]); pr[j] = 1.0f / (1.0f + expf(-l)); sel[j] = pr[j] + bias[e]; }
    for (int k = 0; k < 8; ++k) {
        float v = -INFINITY; int idx = 0x7fffffff;
#pragma unroll
        for (int j = 0; j < 4; ++j) { const float c = used[j] ? -INFINITY : sel[j]; const int ci = lane + 32 * j; if (c > v || (c == v && ci < idx)) { v = c; idx = ci; } }
        float bv = v; int bi = idx;
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) { const float ov = __shfl_xor_sync(0xffffffffu, bv, off); const int oi = __shfl_xor_sync(0xffffffffu, bi, off); if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; } }
        ch[k] = bi;
#pragma unroll
        for (int j = 0; j < 4; ++j) if (lane + 32 * j == bi) { used[j] = true; cp[k] = pr[j]; }
    }
    // cp[k] only valid on the lane that owns the expert: gather by shuffle
    float sum = 0.0f, pv[8];
#pragma unroll
    for (int k = 0; k < 8; ++k) { float v = ((ch[k] & 31) == lane) ? cp[k] : 0.0f; for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffffu, v, off); pv[k] = v; sum += v; }
    sum += 1e-20f;
    if (lane < 8) { int ii = 0; float vv = 0; for (int k = 0; k < 8; ++k) if (k == lane) { ii = ch[k]; vv = pv[k]; } ids[lane] = ii; wts[lane] = vv / sum * scale; }
    if (lane == 0) g_ticket = 0;
}
