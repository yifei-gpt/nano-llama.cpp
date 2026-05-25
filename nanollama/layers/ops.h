// ops.h — small ggml-op helpers used by the forward graph
#pragma once
#include "ggml.h"

namespace nano {

// normalizes dim 0; for per-head Q/K norm pass x as [head_dim, n_head, n_tokens]
inline ggml_tensor * rms_norm(ggml_context * ctx, ggml_tensor * x, ggml_tensor * weight, float eps) {
    return ggml_mul(ctx, ggml_rms_norm(ctx, x, eps), weight);
}

inline ggml_tensor * linear(ggml_context * ctx, ggml_tensor * w, ggml_tensor * x) {
    return ggml_mul_mat(ctx, w, x);
}

inline ggml_tensor * rope_neox(ggml_context * ctx, ggml_tensor * x, ggml_tensor * pos,
                               int head_dim, float freq_base, int n_ctx_orig) {
    return ggml_rope_ext(ctx, x, pos, nullptr, head_dim, GGML_ROPE_TYPE_NEOX, n_ctx_orig,
                         freq_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
}

inline ggml_tensor * swiglu(ggml_context * ctx, ggml_tensor * gate, ggml_tensor * up) {
    return ggml_mul(ctx, ggml_silu(ctx, gate), up);
}

} // namespace nano
