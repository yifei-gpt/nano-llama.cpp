// kv_cache.h — per-layer F16 K/V buffers, laid out as contiguous per-stream slices
#pragma once

#include "ggml.h"
#include "ggml-backend.h"
#include <vector>

namespace nano {

// Slot s owns contiguous cells [s*n_ctx_pad, (s+1)*n_ctx_pad), viewed as a 4D per-stream tensor.
struct KvCache {
    int n_layer    = 0;
    int n_embd_kv  = 0;
    int n_slots    = 0;
    int n_ctx_pad  = 0;   // per-stream cell capacity (n_ctx rounded up to a multiple of 256)
    int n_cells    = 0;   // n_slots * n_ctx_pad

    std::vector<ggml_tensor *> k;
    std::vector<ggml_tensor *> v;

    ggml_context *          ctx = nullptr;
    ggml_backend_buffer_t   buf = nullptr;

    void init(int n_layer, int n_embd_kv, int n_slots, int n_ctx, ggml_backend_buffer_type_t buft);
    void free();
    ~KvCache() { free(); }
};

} // namespace nano
