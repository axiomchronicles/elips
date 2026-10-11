/* ELIPS minimal C API.
 *
 * A thin extern "C" shim over the C++ index lifecycle (index_factory / add /
 * search / free) for callers that cannot consume the C++ headers (FFI, other
 * languages). Built only with -DELIPS_ENABLE_C_API=ON.
 *
 * STATUS: intentionally minimal. It covers single-vector add and k-NN search
 * over the factory-built indexes. NOT covered (add if a real consumer needs
 * them): batched add/search, range search, train() for IVF indexes before add,
 * persistence/snapshots, metadata, and the Collection/Database surface. IVF
 * factory specs ("IVF...") build but will assert untrained until train() is
 * exposed here.
 *
 * Identity: callers supply an opaque int64 id per vector. It is encoded
 * big-endian into the first 8 bytes of the 16-byte RecordID and decoded back on
 * search, so no id<->RecordID map is kept. Ids must be unique per index.
 */
#ifndef ELIPS_C_API_H
#define ELIPS_C_API_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct elips_index elips_index;

/* Metric selector, mirrors elips::Metric. */
typedef enum {
    ELIPS_METRIC_COSINE = 0,
    ELIPS_METRIC_EUCLIDEAN = 1,
    ELIPS_METRIC_DOT_PRODUCT = 2
} elips_metric;

typedef enum {
    ELIPS_OK = 0,
    ELIPS_ERR_INVALID_ARG = 1,
    ELIPS_ERR_RUNTIME = 2
} elips_status;

/* Build an index from a FAISS-style factory spec ("Flat", "IVF256,Flat", ...).
 * Returns NULL on failure (bad spec/metric/dimension). Free with
 * elips_index_free. */
elips_index* elips_index_create(uint16_t dimension, elips_metric metric,
                                const char* spec);

/* Add one `dimension`-length vector under `id`. Returns ELIPS_OK or an error. */
elips_status elips_index_add(elips_index* index, int64_t id,
                             const float* vector, uint16_t dimension);

/* k-NN search. Writes up to `k` ids/distances into the caller's buffers and
 * stores the hit count in *out_count. out_ids/out_distances must hold `k`
 * elements each. Either buffer may be NULL to skip it. */
elips_status elips_index_search(const elips_index* index, const float* query,
                                uint16_t dimension, size_t k, int64_t* out_ids,
                                float* out_distances, size_t* out_count);

/* Number of stored records. */
size_t elips_index_ntotal(const elips_index* index);

/* Release an index. Safe on NULL. */
void elips_index_free(elips_index* index);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* ELIPS_C_API_H */
