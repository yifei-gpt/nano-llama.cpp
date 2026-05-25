// recurrent_cache.h — per-layer conv + delta-net state for the recurrent (linear-attention) layers
#pragma once

#include "ggml.h"
#include "ggml-backend.h"
#include <vector>

namespace nano {

struct Model;

// Single-sequence recurrent state: for each recurrent layer, a conv ring buffer and a delta state,
// both F32, persistent across decode steps. The graph reads them and writes the updated state back.
struct RecurrentCache {
    int n_layer = 0;
    std::vector<ggml_tensor *> conv;   // [n_embd_r, 1] per recurrent layer, null for attention layers
    std::vector<ggml_tensor *> ssm;    // [n_embd_s, 1]

    ggml_context *        ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;

    bool active() const { return buf != nullptr; }
    void init(const Model & m, ggml_backend_buffer_type_t buft);
    void clear();   // zero all state — call at the start of a sequence
    void free();
    ~RecurrentCache() { free(); }
};

} // namespace nano
