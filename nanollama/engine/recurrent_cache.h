// recurrent_cache.h — per-slot conv + delta-net state for the recurrent (linear-attention) layers
#pragma once

#include "ggml.h"
#include "ggml-backend.h"
#include <vector>

namespace nano {

struct Model;

// Per recurrent layer: a conv ring buffer and a delta state, both F32, with one column per slot
// (sequence/stream). The graph reads a slot's previous state and writes back the updated state.
struct RecurrentCache {
    int n_layer = 0, n_slots = 0, n_embd_r = 0, n_embd_s = 0;
    std::vector<ggml_tensor *> conv;   // per recurrent layer; null for attention layers
    std::vector<ggml_tensor *> ssm;

    ggml_context *        ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;

    bool active() const { return buf != nullptr; }
    void init(const Model & m, int n_slots, ggml_backend_buffer_type_t buft);
    void clear();
    void clear_slot(int slot);
    void free();
    ~RecurrentCache() { free(); }
};

} // namespace nano
