// qwen35.h — Qwen3.5 hybrid (gated-DeltaNet + periodic full-attention) weights and forward graph
#pragma once

#include "nanollama/models/model.h"

#include <vector>

namespace nano {

// one transformer block; a layer is either recurrent (gated DeltaNet) or full-attention — it fills
// only the tensors for its kind. attn_norm / post_attn_norm / FFN are shared.
struct qwen35_layer {
    ggml_tensor * attn_norm      = nullptr;
    ggml_tensor * post_attn_norm = nullptr;

    // full-attention layer: wq is double-width (query + output gate)
    ggml_tensor * wq = nullptr, * wk = nullptr, * wv = nullptr, * wo = nullptr;
    ggml_tensor * attn_q_norm = nullptr, * attn_k_norm = nullptr;

    // recurrent (gated DeltaNet) layer
    ggml_tensor * wqkv = nullptr, * wqkv_gate = nullptr;
    ggml_tensor * ssm_conv1d = nullptr, * ssm_dt = nullptr, * ssm_a = nullptr;
    ggml_tensor * ssm_alpha = nullptr, * ssm_beta = nullptr, * ssm_norm = nullptr, * ssm_out = nullptr;

    ggml_tensor * ffn_gate = nullptr, * ffn_up = nullptr, * ffn_down = nullptr;
};

struct qwen35_model : Model {
    // linear-attention (SSM) hparams
    int d_conv  = 0;    // conv kernel
    int d_state = 0;    // per-head K/V dim (head_k_dim == head_v_dim)
    int n_group = 0;    // K heads
    int dt_rank = 0;    // V heads
    int d_inner = 0;    // value channels (= head_v_dim * n_v_heads)
    int full_attn_interval = 4;
    int     rope_dim_count = 0;
    int32_t rope_sections[4] = { 0, 0, 0, 0 };

    ggml_tensor * tok_embd    = nullptr;
    ggml_tensor * output_norm = nullptr;
    ggml_tensor * output      = nullptr;   // tied to tok_embd
    std::vector<qwen35_layer> layers;

    bool is_recurrent(int il) const { return (il + 1) % full_attn_interval != 0; }
    int  conv_channels()      const { return d_inner + 2 * n_group * d_state; }
    int  n_embd_r()           const { return (d_conv - 1) * conv_channels(); }   // conv state size / seq
    int  n_embd_s()           const { return d_state * d_inner; }                // delta state size / seq

    int  n_pos_per_token()      const override { return 4; }   // M-RoPE
    bool is_recurrent_layer(int il) const override { return is_recurrent(il); }
    int  recurrent_conv_size()  const override { return n_embd_r(); }
    int  recurrent_state_size() const override { return n_embd_s(); }

    ggml_tensor * build_graph(ggml_context * ctx, ggml_cgraph * gf, const graph_inputs & in,
                              KvCache & kv, RecurrentCache * rc, int n_kv,
                              const StreamLayout & sl, bool flash) const override;
};

bool qwen35_load(qwen35_model & model, const ModelParams & mp);

} // namespace nano
