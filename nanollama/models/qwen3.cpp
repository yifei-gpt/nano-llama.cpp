// qwen3.cpp — load weights from GGUF and build the forward graph
#include "nanollama/models/qwen3.h"
#include "nanollama/common.h"
#include "nanollama/utils/gguf.h"
#include "nanollama/engine/kv_cache.h"
#include "nanollama/layers/ops.h"
#include "nanollama/layers/attention.h"

#include "ggml-cpu.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif

#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>

namespace nano {

static ggml_backend_buffer_type_t cpu_repack_buffer_type() {
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (!dev) return nullptr;
    auto get_bufts = (ggml_backend_dev_get_extra_bufts_t)
        ggml_backend_reg_get_proc_address(ggml_backend_dev_backend_reg(dev), "ggml_backend_dev_get_extra_bufts");
    if (!get_bufts) return nullptr;
    for (ggml_backend_buffer_type_t * p = get_bufts(dev); p && *p; ++p)
        if (std::string(ggml_backend_buft_name(*p)) == "CPU_REPACK") return *p;
    return nullptr;
}

bool qwen3_load(qwen3_model & model, const ModelParams & mp) {
    GgufFile gf(mp.path);

    model.arch = gf.str(gkey::ARCH);
    model.name = gf.has(gkey::NAME) ? gf.str(gkey::NAME) : "qwen3";

    auto & hp = model.hparams;
    const std::string & a = model.arch;
    hp.n_layer     = gf.u32(arch_key(gkey::BLOCK_COUNT, a));
    hp.n_embd      = gf.u32(arch_key(gkey::EMBD_LEN,    a));
    hp.n_ff        = gf.u32(arch_key(gkey::FF_LEN,      a));
    hp.n_head      = gf.u32(arch_key(gkey::N_HEAD,      a));
    hp.n_head_kv   = gf.u32(arch_key(gkey::N_HEAD_KV,   a));
    hp.head_dim    = gf.u32(arch_key(gkey::KEY_LEN,     a));
    hp.n_ctx_train = gf.u32(arch_key(gkey::CTX_LEN,     a));
    hp.rms_eps     = gf.f32(arch_key(gkey::RMS_EPS,     a));
    hp.rope_theta  = gf.f32_or(arch_key(gkey::ROPE_FREQ_BASE, a), 1e6f);

    ggml_tensor * te = gf.tensor("token_embd.weight");
    if (!te) NANO_ABORT("missing token_embd.weight");
    hp.n_vocab = (int32_t) te->ne[1];

    if (hp.n_layer <= 0 || hp.n_embd <= 0 || hp.n_head <= 0 || hp.head_dim <= 0 || hp.n_ff <= 0 || hp.n_vocab <= 0)
        NANO_ABORT("model has a non-positive dimension in its hparams");
    if (hp.n_head_kv <= 0 || hp.n_head % hp.n_head_kv != 0)
        NANO_ABORT("n_head (%d) must be a multiple of n_head_kv (%d) for GQA", hp.n_head, hp.n_head_kv);

    NANO_LOG("model: %s | n_layer=%d n_embd=%d n_head=%d n_head_kv=%d head_dim=%d n_ff=%d n_vocab=%d",
             model.name.c_str(), hp.n_layer, hp.n_embd, hp.n_head, hp.n_head_kv, hp.head_dim, hp.n_ff, hp.n_vocab);
    NANO_LOG("model: n_ctx_train=%d rms_eps=%g rope_theta=%g", hp.n_ctx_train, hp.rms_eps, hp.rope_theta);

    const bool on_gpu = mp.n_gpu_layers > 0 && cuda_available();
    model.n_gpu_layers = on_gpu ? hp.n_layer : 0;

    const size_t n_tensors = 3 + (size_t) hp.n_layer * 11;
    ggml_init_params ip = { ggml_tensor_overhead() * (n_tensors + 8), nullptr, /*no_alloc=*/true };
    model.ctx_meta = ggml_init(ip);

    // repack only Q4_K matrices on AVX2 (rows%8==0), matching ggml's eligibility
    ggml_backend_buffer_type_t repack_buft = (!on_gpu && ggml_cpu_has_avx2()) ? cpu_repack_buffer_type() : nullptr;
    if (repack_buft) model.ctx_repack = ggml_init(ip);

    auto repackable = [&](const ggml_tensor * src) {
        return model.ctx_repack && src->type == GGML_TYPE_Q4_K && ggml_is_matrix(src) && src->ne[1] % 8 == 0;
    };
    auto dup = [&](const char * name) -> ggml_tensor * {
        ggml_tensor * src = gf.tensor(name);
        if (!src) return nullptr;
        ggml_tensor * t = ggml_dup_tensor(repackable(src) ? model.ctx_repack : model.ctx_meta, src);
        ggml_set_name(t, name);
        return t;
    };

    model.tok_embd    = dup("token_embd.weight");
    model.output_norm = dup("output_norm.weight");
    model.output      = dup("output.weight");
    bool tied = false;
    if (!model.output) { model.output = model.tok_embd; tied = true; }

    model.layers.resize(hp.n_layer);
    char nm[128];
    for (int i = 0; i < hp.n_layer; i++) {
        auto & L = model.layers[i];
        #define DUP(field, fmt) snprintf(nm, sizeof(nm), fmt, i); L.field = dup(nm); \
            if (!L.field) NANO_ABORT("missing tensor: %s", nm);
        DUP(attn_norm,   "blk.%d.attn_norm.weight");
        DUP(wq,          "blk.%d.attn_q.weight");
        DUP(wk,          "blk.%d.attn_k.weight");
        DUP(wv,          "blk.%d.attn_v.weight");
        DUP(wo,          "blk.%d.attn_output.weight");
        DUP(attn_q_norm, "blk.%d.attn_q_norm.weight");
        DUP(attn_k_norm, "blk.%d.attn_k_norm.weight");
        DUP(ffn_norm,    "blk.%d.ffn_norm.weight");
        DUP(ffn_gate,    "blk.%d.ffn_gate.weight");
        DUP(ffn_up,      "blk.%d.ffn_up.weight");
        DUP(ffn_down,    "blk.%d.ffn_down.weight");
        #undef DUP
    }

    // no repack-eligible tensors → drop the empty context (alloc rejects 0 tensors)
    if (model.ctx_repack && !ggml_get_first_tensor(model.ctx_repack)) {
        ggml_free(model.ctx_repack);
        model.ctx_repack = nullptr;
    }

    ggml_backend_buffer_type_t buft =
#ifdef GGML_USE_CUDA
        on_gpu ? ggml_backend_cuda_buffer_type(0) : ggml_backend_cpu_buffer_type();
#else
        ggml_backend_cpu_buffer_type();
#endif
    NANO_LOG("model: placing weights on %s%s", on_gpu ? "CUDA0" : "CPU", model.ctx_repack ? " (+CPU repack)" : "");

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(model.ctx_meta, buft);
    if (!buf) NANO_ABORT("failed to allocate weight buffer");
    model.bufs.push_back(buf);
    if (model.ctx_repack) {
        ggml_backend_buffer_t rbuf = ggml_backend_alloc_ctx_tensors_from_buft(model.ctx_repack, repack_buft);
        if (!rbuf) NANO_ABORT("failed to allocate repack weight buffer");
        model.bufs.push_back(rbuf);
    }

    std::ifstream fin(mp.path, std::ios::binary);
    if (!fin) NANO_ABORT("cannot reopen model file");
    std::vector<char> staging;
    auto load_one = [&](ggml_tensor * t) {
        if (!t) return;
        size_t off = gf.tensor_file_offset(t->name);
        size_t nb  = ggml_nbytes(t);
        staging.resize(nb);
        fin.seekg((std::streamoff) off);
        fin.read(staging.data(), (std::streamsize) nb);
        if (!fin) NANO_ABORT("short read for tensor %s", t->name);
        ggml_backend_tensor_set(t, staging.data(), 0, nb);
    };
    load_one(model.tok_embd);
    load_one(model.output_norm);
    if (!tied) load_one(model.output);
    for (auto & L : model.layers) {
        for (ggml_tensor * t : { L.attn_norm, L.wq, L.wk, L.wv, L.wo, L.attn_q_norm,
                                 L.attn_k_norm, L.ffn_norm, L.ffn_gate, L.ffn_up, L.ffn_down }) {
            load_one(t);
        }
    }

    load_embd(model, model.tok_embd);
    NANO_LOG("model: loaded %d layers%s", hp.n_layer, tied ? " (tied embeddings)" : "");
    return true;
}

ggml_tensor * qwen3_model::build_graph(ggml_context * ctx, ggml_cgraph * gf, const graph_inputs & in,
                                       KvCache & kv, RecurrentCache *, int n_kv,
                                       const StreamLayout & sl, bool flash) const {
    const qwen3_model & model = *this;
    const auto & hp = model.hparams;
    const int hd = hp.head_dim, n_head = hp.n_head, n_head_kv = hp.n_head_kv;
    const float kq_scale = 1.0f / sqrtf((float) hd);

    ggml_tensor * inpL = in.embd;

    for (int il = 0; il < hp.n_layer; il++) {
        const auto & L = model.layers[il];
        ggml_tensor * inpSA = inpL;

        ggml_tensor * cur = rms_norm(ctx, inpL, L.attn_norm, hp.rms_eps);

        ggml_tensor * Q = linear(ctx, L.wq, cur);
        ggml_tensor * K = linear(ctx, L.wk, cur);
        ggml_tensor * V = linear(ctx, L.wv, cur);

        const int64_t n_tokens = cur->ne[1];
        Q = ggml_reshape_3d(ctx, Q, hd, n_head,    n_tokens);
        K = ggml_reshape_3d(ctx, K, hd, n_head_kv, n_tokens);
        V = ggml_reshape_3d(ctx, V, hd, n_head_kv, n_tokens);

        Q = rms_norm(ctx, Q, L.attn_q_norm, hp.rms_eps);
        Q = rope_neox(ctx, Q, in.pos, hd, hp.rope_theta, hp.n_ctx_train);
        K = rms_norm(ctx, K, L.attn_k_norm, hp.rms_eps);
        K = rope_neox(ctx, K, in.pos, hd, hp.rope_theta, hp.n_ctx_train);

        cur = build_attention(ctx, gf, Q, K, V, kv.k[il], kv.v[il], in.kv_idxs,
                              in.mask, n_kv, sl, hd, n_head, n_head_kv, kq_scale, flash);
        cur = linear(ctx, L.wo, cur);

        // last layer: keep only the rows that need logits, so the FFN/norm/lm-head skip the rest
        if (il == hp.n_layer - 1 && in.out_ids) {
            cur   = ggml_get_rows(ctx, cur,   in.out_ids);
            inpSA = ggml_get_rows(ctx, inpSA, in.out_ids);
        }

        ggml_tensor * ffn_inp = ggml_add(ctx, cur, inpSA);

        cur = rms_norm(ctx, ffn_inp, L.ffn_norm, hp.rms_eps);
        ggml_tensor * gate = linear(ctx, L.ffn_gate, cur);
        ggml_tensor * up   = linear(ctx, L.ffn_up,   cur);
        cur = swiglu(ctx, gate, up);
        cur = linear(ctx, L.ffn_down, cur);

        inpL = ggml_add(ctx, cur, ffn_inp);
    }

    ggml_tensor * cur = inpL;
    cur = rms_norm(ctx, cur, model.output_norm, hp.rms_eps);
    cur = linear(ctx, model.output, cur);
    ggml_build_forward_expand(gf, cur);
    return cur;
}

} // namespace nano
