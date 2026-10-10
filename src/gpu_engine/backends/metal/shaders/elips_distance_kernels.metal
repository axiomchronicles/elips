// ELIPS Metal compute kernels: brute-force distance matrix + GPU top-k.
//
// Adapted from FAISS faiss/gpu_metal/MetalDistance.metal, MIT (c) Meta Platforms.
// Reimplemented in ELIPS style. ELIPS distance conventions (ordering-normalized,
// smaller == more similar) differ from FAISS and are matched here exactly:
//   euclidean   -> sqrt(sum (q-v)^2)   (TRUE L2, with the sqrt)
//   cosine      -> 1 - dot(q, v)       (inputs assumed L2-normalized on ingest)
//   dot_product -> -dot(q, v)
// The threadgroup top-k uses a smaller-index tiebreak so GPU results match the
// retired CPU std::partial_sort selection exactly.

#include <metal_stdlib>
using namespace metal;

// ============================================================
//  Distance matrix kernels (hand-tiled GEMM, 32x32 tile, no MPS)
//  params = [nq, nb, d, mode]; grid = ceil(nb/32) x ceil(nq/32),
//  16x16 threads/threadgroup, each thread owns a 2x2 micro-tile.
// ============================================================

kernel void elips_l2_matrix(
    device const float* queries   [[buffer(0)]],
    device const float* vectors   [[buffer(1)]],
    device       float* distances [[buffer(2)]],
    device const uint*  params    [[buffer(3)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint2 ltid [[thread_position_in_threadgroup]]
) {
    constexpr uint TILE_M = 32, TILE_N = 32, TILE_K = 16, TG_THREADS = 256;
    uint nq = params[0], nb = params[1], d = params[2];
    uint row0 = tgid.y * TILE_M, col0 = tgid.x * TILE_N;
    uint ty = ltid.y, tx = ltid.x, tid = ty * 16 + tx;

    float acc00 = 0.0f, acc01 = 0.0f, acc10 = 0.0f, acc11 = 0.0f;
    threadgroup float tgQ[TILE_M * TILE_K];
    threadgroup float tgV[TILE_N * TILE_K];

    for (uint dk = 0; dk < d; dk += TILE_K) {
        uint kLen = min(TILE_K, d - dk);
        for (uint i = tid; i < TILE_M * TILE_K; i += TG_THREADS) {
            uint mr = i / TILE_K, mk = i % TILE_K, gRow = row0 + mr;
            tgQ[i] = (gRow < nq && mk < kLen) ? queries[gRow * d + dk + mk] : 0.0f;
        }
        for (uint i = tid; i < TILE_N * TILE_K; i += TG_THREADS) {
            uint mr = i / TILE_K, mk = i % TILE_K, gCol = col0 + mr;
            tgV[i] = (gCol < nb && mk < kLen) ? vectors[gCol * d + dk + mk] : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint kk = 0; kk < TILE_K; kk++) {
            float q0 = tgQ[(ty * 2) * TILE_K + kk];
            float q1 = tgQ[(ty * 2 + 1) * TILE_K + kk];
            float v0 = tgV[(tx * 2) * TILE_K + kk];
            float v1 = tgV[(tx * 2 + 1) * TILE_K + kk];
            float d00 = q0 - v0; acc00 += d00 * d00;
            float d01 = q0 - v1; acc01 += d01 * d01;
            float d10 = q1 - v0; acc10 += d10 * d10;
            float d11 = q1 - v1; acc11 += d11 * d11;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    uint r0 = row0 + ty * 2, r1 = r0 + 1, c0 = col0 + tx * 2, c1 = c0 + 1;
    if (r0 < nq && c0 < nb) distances[r0 * nb + c0] = sqrt(acc00);
    if (r0 < nq && c1 < nb) distances[r0 * nb + c1] = sqrt(acc01);
    if (r1 < nq && c0 < nb) distances[r1 * nb + c0] = sqrt(acc10);
    if (r1 < nq && c1 < nb) distances[r1 * nb + c1] = sqrt(acc11);
}

// mode 0 -> cosine (1 - dot); mode 1 -> dot_product (-dot).
kernel void elips_ip_matrix(
    device const float* queries   [[buffer(0)]],
    device const float* vectors   [[buffer(1)]],
    device       float* distances [[buffer(2)]],
    device const uint*  params    [[buffer(3)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint2 ltid [[thread_position_in_threadgroup]]
) {
    constexpr uint TILE_M = 32, TILE_N = 32, TILE_K = 16, TG_THREADS = 256;
    uint nq = params[0], nb = params[1], d = params[2], mode = params[3];
    uint row0 = tgid.y * TILE_M, col0 = tgid.x * TILE_N;
    uint ty = ltid.y, tx = ltid.x, tid = ty * 16 + tx;

    float acc00 = 0.0f, acc01 = 0.0f, acc10 = 0.0f, acc11 = 0.0f;
    threadgroup float tgQ[TILE_M * TILE_K];
    threadgroup float tgV[TILE_N * TILE_K];

    for (uint dk = 0; dk < d; dk += TILE_K) {
        uint kLen = min(TILE_K, d - dk);
        for (uint i = tid; i < TILE_M * TILE_K; i += TG_THREADS) {
            uint mr = i / TILE_K, mk = i % TILE_K, gRow = row0 + mr;
            tgQ[i] = (gRow < nq && mk < kLen) ? queries[gRow * d + dk + mk] : 0.0f;
        }
        for (uint i = tid; i < TILE_N * TILE_K; i += TG_THREADS) {
            uint mr = i / TILE_K, mk = i % TILE_K, gCol = col0 + mr;
            tgV[i] = (gCol < nb && mk < kLen) ? vectors[gCol * d + dk + mk] : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint kk = 0; kk < TILE_K; kk++) {
            float q0 = tgQ[(ty * 2) * TILE_K + kk];
            float q1 = tgQ[(ty * 2 + 1) * TILE_K + kk];
            float v0 = tgV[(tx * 2) * TILE_K + kk];
            float v1 = tgV[(tx * 2 + 1) * TILE_K + kk];
            acc00 += q0 * v0; acc01 += q0 * v1;
            acc10 += q1 * v0; acc11 += q1 * v1;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // cosine: 1 - dot (bias 1); dot_product: -dot (bias 0). Both are bias - dot.
    float bias = (mode == 0u) ? 1.0f : 0.0f;
    uint r0 = row0 + ty * 2, r1 = r0 + 1, c0 = col0 + tx * 2, c1 = c0 + 1;
    if (r0 < nq && c0 < nb) distances[r0 * nb + c0] = bias - acc00;
    if (r0 < nq && c1 < nb) distances[r0 * nb + c1] = bias - acc01;
    if (r1 < nq && c0 < nb) distances[r1 * nb + c0] = bias - acc10;
    if (r1 < nq && c1 < nb) distances[r1 * nb + c1] = bias - acc11;
}

// ============================================================
//  GPU top-k: one threadgroup per query. Exact selection by chunked sort-merge.
//  The running best-P is kept sorted in mDist[0,P); each chunk of P distances is
//  loaded into [P,2P) and the whole 2P span is bitonic-sorted, so the smallest P
//  survive into [0,P). After all chunks, [0,k) is the exact global top-k. No
//  per-thread candidate arrays (keeps the runtime shader compiler happy) and
//  threadgroup memory is a fixed 2*MAXP slots regardless of P.
//  P = max(32, next_pow2(k)) is chosen host-side (>= k, so top-k always fits).
//  ELIPS distances are smaller==better for every metric, so the smallest are
//  kept. Ties break on smaller index for exact CPU-selection parity.
//  params = [nq, nb, k, P]. Supports k up to 1024 (MAXP); larger k is clamped.
// ============================================================
kernel void topk_select(
    device const float* distances    [[buffer(0)]],
    device       float* outDistances [[buffer(1)]],
    device       int*   outIndices   [[buffer(2)]],
    device const uint*  params       [[buffer(3)]],
    uint qi       [[threadgroup_position_in_grid]],
    uint tid      [[thread_position_in_threadgroup]],
    uint nthreads [[threads_per_threadgroup]]
) {
    constexpr uint MAXP = 1024;
    constexpr uint M = 2 * MAXP;
    threadgroup float mDist[M];
    threadgroup int   mIdx[M];
    uint nq = params[0], nb = params[1], k = params[2], P = params[3];
    if (qi >= nq || k == 0) return;
    const device float* row = distances + qi * nb;
    uint twoP = 2 * P;

    for (uint i = tid; i < P; i += nthreads) { mDist[i] = 1e38f; mIdx[i] = -1; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint nchunks = (nb + P - 1) / P;
    for (uint c = 0; c < nchunks; ++c) {
        uint base = c * P;
        for (uint i = tid; i < P; i += nthreads) {
            uint gi = base + i;
            if (gi < nb) { mDist[P + i] = row[gi]; mIdx[P + i] = (int)gi; }
            else         { mDist[P + i] = 1e38f;   mIdx[P + i] = -1; }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint k2 = 2; k2 <= twoP; k2 <<= 1) {
            for (uint j = k2 >> 1; j > 0; j >>= 1) {
                for (uint i = tid; i < twoP; i += nthreads) {
                    uint partner = i ^ j;
                    if (partner > i && partner < twoP) {
                        bool up = ((i & k2) == 0);
                        bool iGreater = (mDist[i] > mDist[partner]) ||
                            (mDist[i] == mDist[partner] && mIdx[i] > mIdx[partner]);
                        bool iLess = (mDist[i] < mDist[partner]) ||
                            (mDist[i] == mDist[partner] && mIdx[i] < mIdx[partner]);
                        bool swap = up ? iGreater : iLess;
                        if (swap) {
                            float td = mDist[i]; mDist[i] = mDist[partner]; mDist[partner] = td;
                            int ti = mIdx[i]; mIdx[i] = mIdx[partner]; mIdx[partner] = ti;
                        }
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        }
    }

    for (uint i = tid; i < k; i += nthreads) {
        if (i < P) { outDistances[qi * k + i] = mDist[i]; outIndices[qi * k + i] = mIdx[i]; }
        else       { outDistances[qi * k + i] = 1e38f;    outIndices[qi * k + i] = -1; }
    }
}

// ============================================================
//  IVF two-pass scan + merge (single query).
//  Adapted from FAISS faiss/gpu_metal (ivf_scan_list / ivf_merge_lists),
//  MIT (c) Meta Platforms. Pass 1 scans each probed list independently and
//  keeps that list's top-k; pass 2 merges the per-list top-k into the final
//  per-query top-k. fp32 math, ELIPS distance conventions, smaller-index
//  tiebreak -- so GPU IVF labels match the CPU IndexIVFFlat selection.
//  metric mode: 0 = euclidean (sqrt L2), 1 = cosine (1 - dot), 2 = dot (-dot).
//  Candidate layout is the host-gathered [list0 rows | list1 rows | ...]
//  contiguous array; `offsets[p]..offsets[p+1]` bounds list p, and the index
//  written out is the GLOBAL candidate row so the host maps it back to an id.
// ============================================================

inline float elips_row_distance(device const float* q, device const float* v,
                                uint d, uint metric) {
    float acc = 0.0f;
    if (metric == 0u) {                       // euclidean: sqrt(sum (q-v)^2)
        for (uint i = 0; i < d; ++i) { float t = q[i] - v[i]; acc += t * t; }
        return sqrt(acc);
    }
    for (uint i = 0; i < d; ++i) acc += q[i] * v[i];
    return (metric == 1u ? 1.0f : 0.0f) - acc;  // cosine 1-dot, dot_product -dot
}

// Compare-exchange a 2P-wide bitonic span in threadgroup memory, keeping the
// smallest P in [0,P). Ties break on smaller index -- identical ordering to
// topk_select above. (Shared by both IVF passes.)
#define ELIPS_IVF_BITONIC(mDist, mIdx, twoP)                                   \
    for (uint k2 = 2; k2 <= (twoP); k2 <<= 1) {                                \
        for (uint j = k2 >> 1; j > 0; j >>= 1) {                               \
            for (uint i = tid; i < (twoP); i += nthreads) {                    \
                uint partner = i ^ j;                                          \
                if (partner > i && partner < (twoP)) {                         \
                    bool up = ((i & k2) == 0);                                 \
                    bool iGreater = (mDist[i] > mDist[partner]) ||             \
                        (mDist[i] == mDist[partner] && mIdx[i] > mIdx[partner]);\
                    bool iLess = (mDist[i] < mDist[partner]) ||                \
                        (mDist[i] == mDist[partner] && mIdx[i] < mIdx[partner]);\
                    if (up ? iGreater : iLess) {                               \
                        float td = mDist[i]; mDist[i] = mDist[partner]; mDist[partner] = td; \
                        int ti = mIdx[i]; mIdx[i] = mIdx[partner]; mIdx[partner] = ti;        \
                    }                                                          \
                }                                                              \
            }                                                                  \
            threadgroup_barrier(mem_flags::mem_threadgroup);                   \
        }                                                                      \
    }

// Pass 1: one threadgroup per probed list. params = [nlists, dim, k, P, metric].
kernel void elips_ivf_scan_list(
    device const float* query    [[buffer(0)]],
    device const float* vectors  [[buffer(1)]],
    device const uint*  offsets  [[buffer(2)]],  // nlists + 1
    device       float* outDist  [[buffer(3)]],  // nlists * k
    device       int*   outIdx   [[buffer(4)]],  // nlists * k
    device const uint*  params   [[buffer(5)]],
    uint lp       [[threadgroup_position_in_grid]],
    uint tid      [[thread_position_in_threadgroup]],
    uint nthreads [[threads_per_threadgroup]]
) {
    constexpr uint MAXP = 1024;
    constexpr uint M = 2 * MAXP;
    threadgroup float mDist[M];
    threadgroup int   mIdx[M];
    uint nlists = params[0], dim = params[1], k = params[2], P = params[3], metric = params[4];
    if (lp >= nlists || k == 0) return;
    uint segBegin = offsets[lp], segEnd = offsets[lp + 1];
    uint len = segEnd - segBegin;
    uint twoP = 2 * P;

    for (uint i = tid; i < P; i += nthreads) { mDist[i] = 1e38f; mIdx[i] = -1; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint nchunks = (len + P - 1) / P;
    for (uint c = 0; c < nchunks; ++c) {
        uint base = c * P;
        for (uint i = tid; i < P; i += nthreads) {
            if (base + i < len) {
                uint gi = segBegin + base + i;
                mDist[P + i] = elips_row_distance(query, vectors + (ulong)gi * dim, dim, metric);
                mIdx[P + i] = (int)gi;
            } else { mDist[P + i] = 1e38f; mIdx[P + i] = -1; }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        ELIPS_IVF_BITONIC(mDist, mIdx, twoP)
    }

    for (uint i = tid; i < k; i += nthreads) {
        if (i < P) { outDist[lp * k + i] = mDist[i]; outIdx[lp * k + i] = mIdx[i]; }
        else       { outDist[lp * k + i] = 1e38f;   outIdx[lp * k + i] = -1; }
    }
}

// Pass 2: one threadgroup merges all nlists*k per-list candidates into the
// final top-k. params = [total (= nlists*k), k, P].
kernel void elips_ivf_merge_lists(
    device const float* inDist  [[buffer(0)]],  // nlists * k
    device const int*   inIdx   [[buffer(1)]],
    device       float* outDist [[buffer(2)]],  // k
    device       int*   outIdx  [[buffer(3)]],
    device const uint*  params  [[buffer(4)]],
    uint tid      [[thread_position_in_threadgroup]],
    uint nthreads [[threads_per_threadgroup]]
) {
    constexpr uint MAXP = 1024;
    constexpr uint M = 2 * MAXP;
    threadgroup float mDist[M];
    threadgroup int   mIdx[M];
    uint total = params[0], k = params[1], P = params[2];
    if (k == 0) return;
    uint twoP = 2 * P;

    for (uint i = tid; i < P; i += nthreads) { mDist[i] = 1e38f; mIdx[i] = -1; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint nchunks = (total + P - 1) / P;
    for (uint c = 0; c < nchunks; ++c) {
        uint base = c * P;
        for (uint i = tid; i < P; i += nthreads) {
            uint gi = base + i;
            if (gi < total) { mDist[P + i] = inDist[gi]; mIdx[P + i] = inIdx[gi]; }
            else            { mDist[P + i] = 1e38f;      mIdx[P + i] = -1; }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        ELIPS_IVF_BITONIC(mDist, mIdx, twoP)
    }

    for (uint i = tid; i < k; i += nthreads) {
        if (i < P) { outDist[i] = mDist[i]; outIdx[i] = mIdx[i]; }
        else       { outDist[i] = 1e38f;    outIdx[i] = -1; }
    }
}
