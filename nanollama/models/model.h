// model.h — arch-agnostic model base + per-step graph plumbing
#pragma once

#include "ggml.h"
#include "ggml-backend.h"
#include "nanollama/config.h"

#include <string>
#include <vector>

namespace nano {

// hyper-parameters common to every arch (arch-specific extras live in the subclass)
struct hparams_common {
    int32_t n_vocab   = 0;
    int32_t n_embd    = 0;
    int32_t n_layer   = 0;
    int32_t n_head    = 0;
    int32_t n_head_kv = 0;
    int32_t head_dim  = 0;     // INDEPENDENT of n_embd/n_head
    int32_t n_ff      = 0;
    int32_t n_ctx_train = 0;
    float   rms_eps     = 1e-6f;
    float   rope_theta  = 1e6f;

    int32_t n_embd_kv() const { return head_dim * n_head_kv; }
};

// per-step graph inputs, created by the runner and filled after allocation
struct graph_inputs {
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
struct RecurrentCache;

// Base for every model: holds the loaded weights' backing storage and builds the forward graph.
struct Model {
    hparams_common hparams;
    std::string    arch;
    std::string    name;

    // token embedding kept quantized; embed_tokens() dequantizes only the rows it needs (like llama, no
    // F32 table). embd_data → the model weight buffer on CPU, or embd_host (a host copy) on GPU.
    const void *      embd_data      = nullptr;
    ggml_type         embd_type      = GGML_TYPE_F32;
    size_t            embd_row_bytes = 0;
    std::vector<char> embd_host;

    ggml_context *     ctx_meta   = nullptr;
    ggml_context *     ctx_repack = nullptr;   // weights in the CPU repack buffer
    std::vector<ggml_backend_buffer_t> bufs;
    int n_gpu_layers = 0;

    void embed_tokens(const int32_t * ids, int n, float * dst) const;   // dequantize n token rows → F32 dst

    // arch traits (overridden by hybrid/multimodal models)
    virtual int  n_pos_per_token()     const { return 1; }       // 4 for M-RoPE models
    virtual bool is_recurrent_layer(int) const { return false; } // gated-DeltaNet layers
    virtual int  recurrent_conv_size() const { return 0; }       // n_embd_r (conv state / seq)
    virtual int  recurrent_state_size()const { return 0; }       // n_embd_s (delta state / seq)

    virtual ~Model();
    virtual ggml_tensor * build_graph(ggml_context * ctx, ggml_cgraph * gf, const graph_inputs & in,
                                      KvCache & kv, RecurrentCache * rc, int n_kv,
                                      const StreamLayout & sl, bool flash) const = 0;
};

// load the model named in the GGUF, dispatching on general.architecture
Model * load_model(const ModelParams & mp);

// shared loader helpers (used by each arch's <arch>_load)
struct GgufFile;
bool cuda_available();                          // a usable CUDA device is present
void load_embd(Model & m, ggml_tensor * tok_embd);   // set up the quantized token-embedding lookup for embed_tokens

} // namespace nano
