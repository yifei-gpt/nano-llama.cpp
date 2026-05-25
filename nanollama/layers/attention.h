// attention.h — KV-cache store + per-stream multi-head attention
#pragma once
#include "ggml.h"
#include "nanollama/models/model.h"

namespace nano {

// Store K/V into the cache and compute per-stream GQA attention → [head_dim*n_head, n_tokens].
ggml_tensor * build_attention(
    ggml_context * ctx, ggml_cgraph * gf,
    ggml_tensor * q, ggml_tensor * k_cur, ggml_tensor * v_cur,
    ggml_tensor * k_cache, ggml_tensor * v_cache, ggml_tensor * kv_idxs,
    ggml_tensor * mask, int n_kv, const StreamLayout & sl,
    int head_dim, int n_head, int n_head_kv, float scale, bool flash);

} // namespace nano
