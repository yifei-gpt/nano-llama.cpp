// kv_cache.h — per-layer F16 K/V buffers, laid out as contiguous per-stream slices
#pragma once

#include "ggml.h"
#include "ggml-backend.h"
#include <vector>

namespace nano {

struct Model;

// Slot s owns contiguous cells [s*n_ctx_pad, (s+1)*n_ctx_pad), viewed as a 4D per-stream tensor.
// Only attention layers get K/V buffers; recurrent layers leave k[il]/v[il] null (no KV).
struct KvCache {
    int n_layer    = 0;
    int n_embd_kv  = 0;
    int n_slots    = 0;
    int n_ctx_pad  = 0;   // per-stream cell capacity (n_ctx rounded up to a multiple of 256)
    int n_cells    = 0;

    std::vector<ggml_tensor *> k;
    std::vector<ggml_tensor *> v;

    ggml_context *          ctx = nullptr;
    ggml_backend_buffer_t   buf = nullptr;

    void init(const Model & model, int n_slots, int n_ctx, ggml_backend_buffer_type_t buft);
    void free();
    ~KvCache() { free(); }
};

} // namespace nano
