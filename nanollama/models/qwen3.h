// qwen3.h — hyper-parameters, weights, and forward graph
#pragma once

#include "ggml.h"
#include "ggml-backend.h"
#include "nanollama/config.h"

#include <string>
#include <vector>

namespace nano {

struct qwen3_hparams {
    int32_t n_vocab   = 0;
    int32_t n_embd    = 0;
    int32_t n_layer   = 0;
    int32_t n_head    = 0;
    int32_t n_head_kv = 0;
    int32_t head_dim  = 0;     // INDEPENDENT of n_embd/n_head (128 for Qwen3-4B)
    int32_t n_ff      = 0;
    int32_t n_ctx_train = 0;
    float   rms_eps     = 1e-6f;
    float   rope_theta  = 1e6f;

    int32_t n_embd_kv() const { return head_dim * n_head_kv; }
};

struct qwen3_layer {
    ggml_tensor * attn_norm   = nullptr;
    ggml_tensor * wq = nullptr, * wk = nullptr, * wv = nullptr, * wo = nullptr;
    ggml_tensor * attn_q_norm = nullptr;
    ggml_tensor * attn_k_norm = nullptr;
    ggml_tensor * ffn_norm    = nullptr;
    ggml_tensor * ffn_gate = nullptr, * ffn_up = nullptr, * ffn_down = nullptr;
};

struct qwen3_model {
    qwen3_hparams hparams;
    std::string   arch = "qwen3";
    std::string   name;

    ggml_tensor * tok_embd    = nullptr;
    ggml_tensor * output_norm = nullptr;
    ggml_tensor * output      = nullptr;   // may alias tok_embd (tied embeddings)

    std::vector<float> embd_f32;           // host embedding table for the lookup
    std::vector<qwen3_layer> layers;

    ggml_context *                     ctx_meta   = nullptr;
    ggml_context *                     ctx_repack = nullptr;   // weights in the CPU repack buffer
    std::vector<ggml_backend_buffer_t> bufs;
    int n_gpu_layers = 0;

    ~qwen3_model();
};

bool qwen3_load(qwen3_model & model, const ModelParams & mp);

// per-step graph inputs, created by the runner and filled after allocation
struct qwen3_inputs {
    ggml_tensor * embd    = nullptr;
    ggml_tensor * pos     = nullptr;
    ggml_tensor * kv_idxs = nullptr;
    ggml_tensor * mask    = nullptr;
    ggml_tensor * out_ids = nullptr;
};

// how a batch's tokens map onto KV streams (sequences live in attention dim 3)
struct StreamLayout {
    int n_stream  = 1;   // sequences attended in parallel
    int s0        = 0;   // first slot/stream
    int n_q       = 1;   // query tokens per stream (n_tokens = n_stream * n_q)
    int n_ctx_pad = 0;   // per-stream cell stride (KvCache::n_ctx_pad)
};

struct KvCache;

ggml_tensor * qwen3_build_graph(const qwen3_model & model, ggml_context * ctx, ggml_cgraph * gf,
                                const qwen3_inputs & in, KvCache & kv, int n_kv,
                                const StreamLayout & sl, bool flash);

} // namespace nano
