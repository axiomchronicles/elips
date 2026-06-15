#include <metal_stdlib>
using namespace metal;

kernel void cosine_distance_fp32(
    device const float* queries    [[buffer(0)]],
    device const float* database   [[buffer(1)]],
    device       float* distances  [[buffer(2)]],
    constant uint& nq  [[buffer(3)]],
    constant uint& nb  [[buffer(4)]],
    constant uint& dim [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= nq || gid.y >= nb) return;
    float dot = 0.0f;
    for (uint d = 0; d < dim; ++d) {
        float qv = queries[gid.x * dim + d];
        float dv = database[gid.y * dim + d];
        dot += qv * dv;
    }
    distances[gid.x * nb + gid.y] = 1.0f - dot;
}

kernel void euclidean_distance_fp32(
    device const float* queries    [[buffer(0)]],
    device const float* database   [[buffer(1)]],
    device       float* distances  [[buffer(2)]],
    constant uint& nq  [[buffer(3)]],
    constant uint& nb  [[buffer(4)]],
    constant uint& dim [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= nq || gid.y >= nb) return;
    float sum = 0.0f;
    for (uint d = 0; d < dim; ++d) {
        float diff = queries[gid.x * dim + d] - database[gid.y * dim + d];
        sum += diff * diff;
    }
    distances[gid.x * nb + gid.y] = sqrt(sum);
}

kernel void dot_product_fp32(
    device const float* queries    [[buffer(0)]],
    device const float* database   [[buffer(1)]],
    device       float* distances  [[buffer(2)]],
    constant uint& nq  [[buffer(3)]],
    constant uint& nb  [[buffer(4)]],
    constant uint& dim [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= nq || gid.y >= nb) return;
    float sum = 0.0f;
    for (uint d = 0; d < dim; ++d) {
        sum += queries[gid.x * dim + d] * database[gid.y * dim + d];
    }
    distances[gid.x * nb + gid.y] = -sum;
}

kernel void top_k_reduce(
    device const float*    distances  [[buffer(0)]],
    device const uint*     row_ids    [[buffer(1)]],
    device       float*    top_vals   [[buffer(2)]],
    device       uint*     top_idxs   [[buffer(3)]],
    constant     uint&     k          [[buffer(4)]],
    constant     uint&     nb         [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= nb) return;
    float dist = distances[gid];
    uint idx = gid;

    for (uint i = 0; i < k; ++i) {
        if (dist < top_vals[i] || i == 0) {
            for (uint j = k - 1; j > i; --j) {
                top_vals[j] = top_vals[j - 1];
                top_idxs[j] = top_idxs[j - 1];
            }
            top_vals[i] = dist;
            top_idxs[i] = idx;
            break;
        }
    }
}

kernel void pq_encode(
    device const float* vectors     [[buffer(0)]],
    device const float* codebook    [[buffer(1)]],
    device       uchar* codes       [[buffer(2)]],
    constant uint& n        [[buffer(3)]],
    constant uint& dim      [[buffer(4)]],
    constant uint& pq_dim   [[buffer(5)]],
    constant uint& n_centroids [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= n || gid.y >= pq_dim) return;
    uint sub_dim = dim / pq_dim;
    uint best_c = 0;
    float best_d = INFINITY;
    for (uint c = 0; c < n_centroids; ++c) {
        float dist = 0.0f;
        for (uint d = 0; d < sub_dim; ++d) {
            float diff = vectors[gid.x * dim + gid.y * sub_dim + d] -
                         codebook[gid.y * n_centroids * sub_dim + c * sub_dim + d];
            dist += diff * diff;
        }
        if (dist < best_d) {
            best_d = dist;
            best_c = c;
        }
    }
    codes[gid.x * pq_dim + gid.y] = static_cast<uchar>(best_c);
}
