// qwen3.h — Qwen3 weights and forward graph
#pragma once

#include "nanollama/models/model.h"

#include <string>
#include <vector>

namespace nano {

struct qwen3_layer {
    ggml_tensor * attn_norm   = nullptr;
    ggml_tensor * wq = nullptr, * wk = nullptr, * wv = nullptr, * wo = nullptr;
    ggml_tensor * attn_q_norm = nullptr;
    ggml_tensor * attn_k_norm = nullptr;
    ggml_tensor * ffn_norm    = nullptr;
    ggml_tensor * ffn_gate = nullptr, * ffn_up = nullptr, * ffn_down = nullptr;
};

struct qwen3_model : Model {
    ggml_tensor * tok_embd    = nullptr;
    ggml_tensor * output_norm = nullptr;
    ggml_tensor * output      = nullptr;   // may alias tok_embd (tied embeddings)
    std::vector<qwen3_layer> layers;

    ggml_tensor * build_graph(ggml_context * ctx, ggml_cgraph * gf, const graph_inputs & in,
                              KvCache & kv, RecurrentCache * rc, int n_kv,
                              const StreamLayout & sl, bool flash) const override;
};

bool qwen3_load(qwen3_model & model, const ModelParams & mp);

ggml_tensor * qwen3_build_graph(const qwen3_model & model, ggml_context * ctx, ggml_cgraph * gf,
                                const graph_inputs & in, KvCache & kv, int n_kv,
                                const StreamLayout & sl, bool flash);

} // namespace nano
