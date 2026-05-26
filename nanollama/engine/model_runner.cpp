// model_runner.cpp — builds and runs the ggml graph on a single backend
#include "nanollama/engine/model_runner.h"
#include "nanollama/common.h"

#include "ggml-cpu.h"
#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif

#include <algorithm>
#include <cmath>
#include <vector>

namespace nano {

// node budget scales with n_layer so larger models fit
static int graph_nodes(int n_layer) { return 64 * n_layer + 128; }

struct GraphBuild {
    ggml_context * ctx = nullptr;
    ggml_cgraph  * gf  = nullptr;
    graph_inputs   in;
    ggml_tensor *  logits = nullptr;
};

static size_t graph_mem_size(int n_layer) {
    const int n = graph_nodes(n_layer);
    return ggml_tensor_overhead() * n + ggml_graph_overhead_custom(n, false);
}

static GraphBuild build(const Model & model, KvCache & kv, RecurrentCache * rc, int n_tokens, int n_kv,
                        int n_out, const StreamLayout & sl, bool flash, void * membuf) {
    GraphBuild g;
    const int n_nodes = graph_nodes(model.hparams.n_layer);
    ggml_init_params ip = { graph_mem_size(model.hparams.n_layer), membuf, /*no_alloc=*/true };
    g.ctx = ggml_init(ip);
    g.gf  = ggml_new_graph_custom(g.ctx, n_nodes, false);

    g.in.embd    = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F32, model.hparams.n_embd, n_tokens);
                                                                             ggml_set_input(g.in.embd);
    g.in.pos     = ggml_new_tensor_1d(g.ctx, GGML_TYPE_I32, n_tokens * model.n_pos_per_token());
                                                                             ggml_set_input(g.in.pos);
    g.in.kv_idxs = ggml_new_tensor_1d(g.ctx, GGML_TYPE_I64, n_tokens);        ggml_set_input(g.in.kv_idxs);
    ggml_type mtype = flash ? GGML_TYPE_F16 : GGML_TYPE_F32;
    g.in.mask    = ggml_new_tensor_4d(g.ctx, mtype, n_kv, sl.n_q, 1, sl.n_stream); ggml_set_input(g.in.mask);
    g.in.out_ids = ggml_new_tensor_1d(g.ctx, GGML_TYPE_I32, n_out);           ggml_set_input(g.in.out_ids);

    g.logits = model.build_graph(g.ctx, g.gf, g.in, kv, rc, n_kv, sl, flash);
    ggml_set_output(g.logits);
    return g;
}

// M-RoPE position layout: [n_tokens*4] = 3 blocks of the token position + a zero block (text)
static void fill_positions(std::vector<int32_t> & pos, int n_pos_per_token, const int32_t * tok_pos, int n_tokens) {
    pos.resize((size_t) n_tokens * n_pos_per_token);
    if (n_pos_per_token == 4) {
        for (int i = 0; i < n_tokens; i++) {
            pos[i] = pos[n_tokens + i] = pos[2 * n_tokens + i] = tok_pos[i];
            pos[3 * n_tokens + i] = 0;
        }
    } else {
        for (int i = 0; i < n_tokens; i++) pos[i] = tok_pos[i];
    }
}

// causal mask: token i attends cells 0..pos[i] of its own stream (per-stream isolation is structural)
template <class T>
static void fill_causal(T * m, T zero, T neg, const int32_t * pos, int n_tokens, int n_kv) {
    for (int i = 0; i < n_tokens; i++) {
        const int p = pos[i];
        T * row = m + (size_t) i * n_kv;
        for (int j = 0; j < n_kv; j++) row[j] = (j <= p) ? zero : neg;
    }
}

static void set_mask(ggml_tensor * mask, bool flash, const int32_t * pos, int n_tokens, int n_kv) {
    const size_t melem = (size_t) n_kv * n_tokens;
    if (flash) {
        std::vector<ggml_fp16_t> m(melem);
        fill_causal(m.data(), ggml_fp32_to_fp16(0.0f), ggml_fp32_to_fp16(-INFINITY), pos, n_tokens, n_kv);
        ggml_backend_tensor_set(mask, m.data(), 0, melem * sizeof(ggml_fp16_t));
    } else {
        std::vector<float> m(melem);
        fill_causal(m.data(), 0.0f, -INFINITY, pos, n_tokens, n_kv);
        ggml_backend_tensor_set(mask, m.data(), 0, melem * sizeof(float));
    }
}

void ModelRunner::init(const Model & m, const ContextParams & cp_) {
    model = &m;
    cp = cp_;
    on_gpu = m.n_gpu_layers >= m.hparams.n_layer && m.n_gpu_layers > 0;

#ifdef GGML_USE_CUDA
    if (on_gpu) backend = ggml_backend_cuda_init(0);
#endif
    if (!backend) { backend = ggml_backend_cpu_init(); on_gpu = false;
                    ggml_backend_cpu_set_n_threads(backend, cp.n_threads); }

    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend);
    kv.init(m, cp.n_slots, cp.n_ctx, buft);
    rc.init(m, cp.n_slots, buft);
    galloc = ggml_gallocr_new(buft);

    dec_mem.resize(graph_mem_size(m.hparams.n_layer));
    bat_mem.resize(graph_mem_size(m.hparams.n_layer));
    StreamLayout sl{ /*n_stream=*/1, /*s0=*/0, /*n_q=*/1, kv.n_ctx_pad };
    GraphBuild d = build(m, kv, &rc, /*n_tokens=*/1, kv.n_ctx_pad, /*n_out=*/1, sl, cp.flash_attn, dec_mem.data());
    if (!ggml_gallocr_reserve(galloc, d.gf)) NANO_ABORT("failed to reserve graph buffer");
    ggml_free(d.ctx);

    NANO_LOG("runner: backend=%s flash_attn=%d n_ctx=%d n_slots=%d", on_gpu ? "CUDA" : "CPU",
             (int) cp.flash_attn, cp.n_ctx, cp.n_slots);
}

void ModelRunner::free() {
    if (galloc)  { ggml_gallocr_free(galloc); galloc = nullptr; }
    kv.free();
    rc.free();
    if (backend) { ggml_backend_free(backend); backend = nullptr; }
}

// token-id wrapper: look up embeddings + build sequential M-RoPE positions, then run decode_embd
const float * ModelRunner::decode(const int32_t * tokens, int n_tokens, int n_past) {
    std::vector<float> emb((size_t) model->hparams.n_embd * n_tokens);
    model->embed_tokens(tokens, n_tokens, emb.data());
    std::vector<int32_t> tok_pos(n_tokens), pos;
    for (int i = 0; i < n_tokens; i++) tok_pos[i] = n_past + i;
    fill_positions(pos, model->n_pos_per_token(), tok_pos.data(), n_tokens);
    return decode_embd(emb.data(), pos.data(), n_tokens, n_past);
}

const float * ModelRunner::decode_embd(const float * embd, const int32_t * pos4, int n_tokens, int n_past) {
    const int  n_vocab = model->hparams.n_vocab;
    const int  npp     = model->n_pos_per_token();
    const bool single  = (n_tokens == 1);
    const bool fa      = cp.flash_attn && single;

    int n_kv = GGML_PAD(n_past + n_tokens, 256);
    if (n_kv > kv.n_ctx_pad) n_kv = kv.n_ctx_pad;
    if (n_past == 0) rc.clear();
    std::vector<char> prefill_mem;
    void * membuf = single ? dec_mem.data() : (prefill_mem.resize(graph_mem_size(model->hparams.n_layer)), prefill_mem.data());

    StreamLayout sl{ /*n_stream=*/1, /*s0=*/0, /*n_q=*/n_tokens, kv.n_ctx_pad };
    GraphBuild g = build(*model, kv, &rc, n_tokens, n_kv, /*n_out=*/1, sl, fa, membuf);
    if (!ggml_gallocr_alloc_graph(galloc, g.gf)) NANO_ABORT("graph alloc failed");

    std::vector<int64_t> kvi(n_tokens);
    std::vector<int32_t> seqpos(n_tokens);
    for (int i = 0; i < n_tokens; i++) { kvi[i] = n_past + i; seqpos[i] = n_past + i; }
    const int32_t out_id = n_tokens - 1;

    ggml_backend_tensor_set(g.in.embd,    embd,        0, (size_t) model->hparams.n_embd * n_tokens * sizeof(float));
    ggml_backend_tensor_set(g.in.pos,     pos4,        0, (size_t) n_tokens * npp * sizeof(int32_t));
    ggml_backend_tensor_set(g.in.kv_idxs, kvi.data(),  0, (size_t) n_tokens * sizeof(int64_t));
    ggml_backend_tensor_set(g.in.out_ids, &out_id,     0, sizeof(int32_t));
    set_mask(g.in.mask, fa, seqpos.data(), n_tokens, n_kv);

    if (ggml_backend_graph_compute(backend, g.gf) != GGML_STATUS_SUCCESS) NANO_ABORT("graph compute failed");
    logits_buf.resize((size_t) n_vocab);
    ggml_backend_tensor_get(g.logits, logits_buf.data(), 0, logits_buf.size() * sizeof(float));
    ggml_free(g.ctx);
    return logits_buf.data();
}

const float * ModelRunner::decode_batch(const Batch & b, int s0, int n_stream, int n_q, int n_kv) {
    const int n_tokens = b.token.empty() ? (int) (b.embd.size() / model->hparams.n_embd) : (int) b.token.size();
    const int n_out    = (int) b.logit_rows.size();
    const int n_vocab  = model->hparams.n_vocab;
    const bool fa      = cp.flash_attn;

    n_kv = GGML_PAD(n_kv, 256); if (n_kv > kv.n_ctx_pad) n_kv = kv.n_ctx_pad;

    StreamLayout sl{ n_stream, s0, n_q, kv.n_ctx_pad };
    GraphBuild g = build(*model, kv, &rc, n_tokens, n_kv, n_out, sl, fa, bat_mem.data());
    if (!ggml_gallocr_alloc_graph(galloc, g.gf)) NANO_ABORT("graph alloc failed");

    std::vector<float> emb;
    const float * embd_src = b.embd.data();
    if (b.embd.empty()) { emb.resize((size_t) model->hparams.n_embd * n_tokens); model->embed_tokens(b.token.data(), n_tokens, emb.data()); embd_src = emb.data(); }
    std::vector<int64_t> kvi(n_tokens);
    for (int i = 0; i < n_tokens; i++) kvi[i] = b.kv_dst[i];
    std::vector<int32_t> pos;
    if (b.mrope.empty()) fill_positions(pos, model->n_pos_per_token(), b.pos.data(), n_tokens);
    else                 pos = b.mrope;

    ggml_backend_tensor_set(g.in.embd,    embd_src,            0, (size_t) model->hparams.n_embd * n_tokens * sizeof(float));
    ggml_backend_tensor_set(g.in.pos,     pos.data(),          0, pos.size() * sizeof(int32_t));
    ggml_backend_tensor_set(g.in.kv_idxs, kvi.data(),          0, (size_t) n_tokens * sizeof(int64_t));
    ggml_backend_tensor_set(g.in.out_ids, b.logit_rows.data(), 0, (size_t) n_out * sizeof(int32_t));
    set_mask(g.in.mask, fa, b.pos.data(), n_tokens, n_kv);

    if (ggml_backend_graph_compute(backend, g.gf) != GGML_STATUS_SUCCESS)
        NANO_ABORT("batch compute failed");

    logits_buf.resize((size_t) n_vocab * n_out);
    ggml_backend_tensor_get(g.logits, logits_buf.data(), 0, logits_buf.size() * sizeof(float));
    ggml_free(g.ctx);
    return logits_buf.data();
}

} // namespace nano
