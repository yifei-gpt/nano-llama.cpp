// model_runner.h — single-backend forward runner (CPU or CUDA)
#pragma once

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "nanollama/config.h"
#include "nanollama/models/model.h"
#include "nanollama/engine/kv_cache.h"
#include "nanollama/engine/recurrent_cache.h"

#include <vector>

namespace nano {

struct ModelRunner {
    const Model * model = nullptr;
    ContextParams       cp;
    bool                on_gpu = false;

    ggml_backend_t        backend = nullptr;   // the whole model runs on this one backend
    ggml_gallocr_t        galloc  = nullptr;
    KvCache               kv;
    RecurrentCache        rc;                  // only allocated for recurrent (hybrid) models

    // graphs are rebuilt into these fixed buffers each step → stable addresses → CUDA-graph replay
    std::vector<char> dec_mem;
    std::vector<char> bat_mem;

    std::vector<float> logits_buf;

    // per token: its sequence position (pos, for the causal mask), destination KV cell (kv_dst),
    // and (optionally) a logit-output row. embd/mrope are set only for VLM image prefill.
    struct Batch {
        std::vector<int32_t> token;        // token ids (used when embd is empty)
        std::vector<float>   embd;         // precomputed input embeddings [n_embd*n_tokens]; overrides token lookup
        std::vector<int32_t> pos;          // sequence position per token
        std::vector<int32_t> mrope;        // explicit M-RoPE positions [4*n_tokens]; if empty, derived from pos
        std::vector<int32_t> kv_dst;
        std::vector<int32_t> logit_rows;
    };

    void init(const Model & model, const ContextParams & cp);
    void free();
    ~ModelRunner() { free(); }

    const float * decode(const int32_t * tokens, int n_tokens, int n_past);   // single sequence

    // multimodal single-sequence decode: precomputed input embeddings [n_embd*n_tokens] +
    // explicit M-RoPE positions pos4 [n_pos_per_token*n_tokens]; sequence-causal mask. Logits for the last token.
    const float * decode_embd(const float * embd, const int32_t * pos4, int n_tokens, int n_past);

    // per-stream batched forward: n_stream sequences of n_q tokens each → logits in logit_rows order
    const float * decode_batch(const Batch & b, int s0, int n_stream, int n_q, int n_kv);

    int  n_ctx_pad()    const { return kv.n_ctx_pad; }
    int  scratch_cell() const { return kv.n_slots * kv.n_ctx_pad; }   // dummy-write sink, never read
    bool is_recurrent() const { return rc.active(); }                 // qwen3.5 (gated-DeltaNet) model
    void reset_recurrent_slot(int slot) { rc.clear_slot(slot); }      // zero a slot's recurrent state

    int n_vocab() const { return model->hparams.n_vocab; }

private:
    void fill_embd(const int32_t * tokens, int n_tokens, float * dst) const;
};

} // namespace nano
