#include "attention/fused_attention.cuh"
#include "common/cuda_utils.cuh"

// Decode attention — v6, FlashDecoding split-K over the KV sequence.
//
// Why this exists (Phase 7). v3 launches one single-warp block per
// (batch, head) and streams the *entire* KV sequence in that one warp. At the
// reference workload (batch=8, n_heads=32) that is 256 blocks on a 128-SM 4090
// — ~2 blocks/SM, so most of the machine is idle and a lone warp cannot keep
// enough loads in flight to saturate HBM. Measured: v3 reaches 178 GB/s (18% of
// the 1008 GB/s peak) while PyTorch SDPA's flash_fwd_splitkv reaches 812 GB/s
// (81%). v3 was never bandwidth-bound; it was occupancy-bound. See
// docs/05-baseline-correction-journey.md.
//
// The fix is the technique v4 tried and mis-applied: split-K. Instead of one
// block per (batch, head), launch `n_splits` blocks per (batch, head), each
// owning a contiguous chunk of the KV sequence. The grid grows by n_splits×
// (e.g. 256 → 4096 blocks), which fills every SM and puts enough independent
// warps in flight to saturate HBM. Each block runs the exact v3 inner loop over
// its chunk and exports its *un-normalized* online-softmax partial (m, l,
// O_acc); a second combine kernel merges the partials per (batch, head) with
// the standard log-sum-exp rescale.
//
// The split-K idea was always right (it is what flash does). v4 failed because
// it layered split-K on top of the single-warp block geometry that had already
// surrendered two thirds of its occupancy — adding launch+combine cost without
// unlocking the parallelism the block geometry had thrown away. Here split-K
// *is* the parallelism.
//
// Layout & contract: identical to launch_decode_attention (fused_attention.cuh).
// head_dim must be 128 (VEC=4 halves per lane across a 32-lane warp).

namespace {

constexpr int VEC    = 4;   // halves per thread per vectorized load (head_dim/32)
constexpr int NWARPS = 4;   // warps per partial block (128 threads)

// Choose how many KV chunks to split each (batch, head) into. The lever is
// occupancy. Each partial block is NWARPS warps; on Ada an SM holds up to 48
// warps, so ~48/NWARPS blocks/SM. We want the grid (batch * n_heads * n_splits
// blocks) to fill the SMs a few waves deep, while keeping each chunk long enough
// (>= MIN_CHUNK) that each of the NWARPS warps still has real work after the
// stride, so the shared-memory combine doesn't dominate.
int choose_n_splits(int batch, int n_heads, int seqlen_kv) {
    const int NUM_SM        = 128;             // RTX 4090 (sm_89)
    const int TARGET_BLOCKS = 16 * NUM_SM;     // ~2 waves at 12 NWARPS-blocks/SM
    const int MIN_CHUNK     = 128;             // >= 32 KV per warp after stride
    const int MAX_SPLITS    = 128;

    const int bh = batch * n_heads;
    int max_useful = seqlen_kv / MIN_CHUNK;
    if (max_useful < 1) max_useful = 1;

    int desired = (TARGET_BLOCKS + bh - 1) / bh;   // ceil(TARGET / bh)
    int n = desired;
    if (n > max_useful)  n = max_useful;
    if (n > MAX_SPLITS)  n = MAX_SPLITS;
    if (n < 1)           n = 1;
    return n;
}

}  // namespace

int decode_attention_n_splits(int batch, int n_heads, int seqlen_kv) {
    return choose_n_splits(batch, n_heads, seqlen_kv);
}

// NWARPS warps per (batch, head, split). The warps cooperatively cover the KV
// chunk [j0, j1): warp w handles positions j0+w, j0+w+NWARPS, ... Each warp runs
// the v3 online-softmax inner loop over its stride and keeps its own (m, l,
// O_acc); the NWARPS partials are merged in shared memory, and the block writes
// one UN-normalized (m, l, O_acc) partial for its chunk. The cross-split merge
// happens in the combine kernel. More warps per block => higher SM occupancy,
// which is exactly what v3's single-warp block gave away.
__global__ void decode_attention_splitk_partial_kernel(
    const half* __restrict__ q, const half* __restrict__ k,
    const half* __restrict__ v,
    float* __restrict__ partial_o,   // [batch, n_heads, n_splits, head_dim]
    float* __restrict__ partial_m,   // [batch, n_heads, n_splits]
    float* __restrict__ partial_l,   // [batch, n_heads, n_splits]
    int batch, int n_heads, int n_kv_heads, int seqlen_kv, int head_dim,
    int n_splits, float softmax_scale) {

    const int batch_idx = blockIdx.x;
    const int head_idx  = blockIdx.y;
    const int split_idx = blockIdx.z;
    const int warp_id   = threadIdx.x >> 5;      // 0..NWARPS-1
    const int lane      = threadIdx.x & 31;      // 0..31; owns d-lanes lane*VEC..

    const int kv_head_idx = head_idx / (n_heads / n_kv_heads);
    const int slot = (batch_idx * n_heads + head_idx) * n_splits + split_idx;
    float* o_out = partial_o + static_cast<long>(slot) * head_dim + lane * VEC;

    const int chunk = ceil_div(seqlen_kv, n_splits);
    const int j0 = split_idx * chunk;
    const int j1 = min(j0 + chunk, seqlen_kv);

    // Whole-block empty split: null partial (m=-inf, l=0, O=0). O must be written
    // explicitly — the combine kernel reads it unconditionally, and 0 * NaN from
    // uninitialized scratch would poison the result even though alpha=0.
    if (j0 >= seqlen_kv) {
        if (warp_id == 0) {
            *reinterpret_cast<float4*>(o_out) = make_float4(0.f, 0.f, 0.f, 0.f);
            if (lane == 0) { partial_m[slot] = -INFINITY; partial_l[slot] = 0.f; }
        }
        return;
    }

    const half* q_ptr  = q + (batch_idx * n_heads    + head_idx)    * head_dim + lane * VEC;
    const half* k_base = k + (batch_idx * n_kv_heads + kv_head_idx) * seqlen_kv * head_dim;
    const half* v_base = v + (batch_idx * n_kv_heads + kv_head_idx) * seqlen_kv * head_dim;

    const float4 q_v = load_half4_as_float4(q_ptr);

    float4 o_v     = {0.0f, 0.0f, 0.0f, 0.0f};
    float  m_state = -INFINITY;
    float  l_state = 0.0f;

    // q·k across this lane's 4 d-lanes, warp-reduced and scaled -> s_j.
    #define SCORE(kv) (warp_reduce_sum(q_v.x*(kv).x + q_v.y*(kv).y            \
                                     + q_v.z*(kv).z + q_v.w*(kv).w) * softmax_scale)
    // One online-softmax step: fold score s and value vv into (m, l, o_v).
    #define UPDATE(s, vv) do {                                                \
        const float m_new = fmaxf(m_state, (s));                             \
        const float alpha = __expf(m_state - m_new);                         \
        const float p_j   = __expf((s)     - m_new);                         \
        o_v.x = o_v.x*alpha + p_j*(vv).x;  o_v.y = o_v.y*alpha + p_j*(vv).y;  \
        o_v.z = o_v.z*alpha + p_j*(vv).z;  o_v.w = o_v.w*alpha + p_j*(vv).w;  \
        l_state = l_state*alpha + p_j;  m_state = m_new;                      \
    } while (0)

    // This warp strides over the chunk starting at warp_id, unrolled 4-deep so
    // up to 8 K/V loads (4 positions × K+V) are in flight before any is consumed.
    // Online-softmax is a dependency chain; the loads are not, so overlapping
    // them is what lifts a memory-bound kernel from ~72% toward peak HBM.
    const int stride = NWARPS;
    int j = j0 + warp_id;
    for (; j + 3 * stride < j1; j += 4 * stride) {
        const float4 k0 = load_half4_as_float4(k_base + (j           ) * head_dim + lane*VEC);
        const float4 v0 = load_half4_as_float4(v_base + (j           ) * head_dim + lane*VEC);
        const float4 k1 = load_half4_as_float4(k_base + (j +   stride ) * head_dim + lane*VEC);
        const float4 v1 = load_half4_as_float4(v_base + (j +   stride ) * head_dim + lane*VEC);
        const float4 k2 = load_half4_as_float4(k_base + (j + 2*stride ) * head_dim + lane*VEC);
        const float4 v2 = load_half4_as_float4(v_base + (j + 2*stride ) * head_dim + lane*VEC);
        const float4 k3 = load_half4_as_float4(k_base + (j + 3*stride ) * head_dim + lane*VEC);
        const float4 v3 = load_half4_as_float4(v_base + (j + 3*stride ) * head_dim + lane*VEC);
        const float s0 = SCORE(k0), s1 = SCORE(k1), s2 = SCORE(k2), s3 = SCORE(k3);
        UPDATE(s0, v0); UPDATE(s1, v1); UPDATE(s2, v2); UPDATE(s3, v3);
    }
    for (; j < j1; j += stride) {
        const float4 k_v = load_half4_as_float4(k_base + j * head_dim + lane*VEC);
        const float4 v_v = load_half4_as_float4(v_base + j * head_dim + lane*VEC);
        UPDATE(SCORE(k_v), v_v);
    }
    #undef SCORE
    #undef UPDATE

    // --- merge the NWARPS warp-partials in shared memory ---
    // m_state/l_state are replicated across a warp's lanes; o_v is distributed
    // (lane owns d-lanes lane*VEC..). Stage each warp's o into s_o, its scalars
    // into s_m/s_l, then warp 0 does the log-sum-exp merge.
    __shared__ float s_o[NWARPS][128];   // head_dim == 128
    __shared__ float s_m[NWARPS];
    __shared__ float s_l[NWARPS];

    *reinterpret_cast<float4*>(&s_o[warp_id][lane * VEC]) = o_v;
    if (lane == 0) { s_m[warp_id] = m_state; s_l[warp_id] = l_state; }
    __syncthreads();

    if (warp_id == 0) {
        float gm = -INFINITY;
        #pragma unroll
        for (int w = 0; w < NWARPS; ++w) gm = fmaxf(gm, s_m[w]);

        float4 acc   = {0.0f, 0.0f, 0.0f, 0.0f};
        float  denom = 0.0f;
        #pragma unroll
        for (int w = 0; w < NWARPS; ++w) {
            const float alpha = __expf(s_m[w] - gm);   // 0 for -inf (idle warp)
            denom += s_l[w] * alpha;
            const float4 o_w = *reinterpret_cast<const float4*>(&s_o[w][lane * VEC]);
            acc.x += alpha * o_w.x;
            acc.y += alpha * o_w.y;
            acc.z += alpha * o_w.z;
            acc.w += alpha * o_w.w;
        }
        *reinterpret_cast<float4*>(o_out) = acc;         // un-normalized
        if (lane == 0) { partial_m[slot] = gm; partial_l[slot] = denom; }
    }
}

// One warp per (batch, head). Merges the n_splits partials with the standard
// online-softmax (log-sum-exp) rescale and writes the final fp16 output.
__global__ void decode_attention_splitk_combine_kernel(
    const float* __restrict__ partial_o,
    const float* __restrict__ partial_m,
    const float* __restrict__ partial_l,
    half* __restrict__ out,
    int batch, int n_heads, int head_dim, int n_splits) {

    const int batch_idx = blockIdx.x;
    const int head_idx  = blockIdx.y;
    const int tid       = threadIdx.x;
    const int base      = (batch_idx * n_heads + head_idx) * n_splits;

    // Global max across splits (empty splits carry -inf and drop out).
    float gm = -INFINITY;
    for (int s = 0; s < n_splits; ++s) gm = fmaxf(gm, partial_m[base + s]);

    float4 acc   = {0.0f, 0.0f, 0.0f, 0.0f};
    float  denom = 0.0f;
    for (int s = 0; s < n_splits; ++s) {
        const float alpha = __expf(partial_m[base + s] - gm);   // 0 for -inf
        denom += partial_l[base + s] * alpha;
        const float4 o_s = *reinterpret_cast<const float4*>(
            partial_o + static_cast<long>(base + s) * head_dim + tid * VEC);
        acc.x += alpha * o_s.x;
        acc.y += alpha * o_s.y;
        acc.z += alpha * o_s.z;
        acc.w += alpha * o_s.w;
    }

    const float inv = 1.0f / denom;
    acc.x *= inv; acc.y *= inv; acc.z *= inv; acc.w *= inv;
    store_float4_as_half4(
        out + (batch_idx * n_heads + head_idx) * head_dim + tid * VEC, acc);
}

void launch_decode_attention_splitk(
    const half* q, const half* k, const half* v, half* out,
    float* partial_o, float* partial_m, float* partial_l,
    int batch, int n_heads, int n_kv_heads, int seqlen_kv, int head_dim,
    int n_splits, float softmax_scale, cudaStream_t stream) {

    dim3 grid_partial(batch, n_heads, n_splits);
    decode_attention_splitk_partial_kernel<<<grid_partial, NWARPS * 32, 0, stream>>>(
        q, k, v, partial_o, partial_m, partial_l,
        batch, n_heads, n_kv_heads, seqlen_kv, head_dim, n_splits, softmax_scale);
    CUDA_CHECK_LAST();

    dim3 grid_combine(batch, n_heads);
    decode_attention_splitk_combine_kernel<<<grid_combine, 32, 0, stream>>>(
        partial_o, partial_m, partial_l, out,
        batch, n_heads, head_dim, n_splits);
    CUDA_CHECK_LAST();
}
