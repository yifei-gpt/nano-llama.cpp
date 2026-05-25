// qwen35.cpp — load Qwen3.5 weights from GGUF (forward graph in qwen35_graph.cpp)
#include "nanollama/models/qwen35.h"
#include "nanollama/common.h"
#include "nanollama/utils/gguf.h"

#include "ggml-backend.h"
#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif

#include <fstream>
#include <string>

namespace nano {

static bool cuda_available() {
#ifdef GGML_USE_CUDA
    return ggml_backend_cuda_get_device_count() > 0;
#else
    return false;
#endif
}

bool qwen35_load(qwen35_model & model, const ModelParams & mp) {
    GgufFile gf(mp.path);

    model.arch = gf.str(gkey::ARCH);
    model.name = gf.has(gkey::NAME) ? gf.str(gkey::NAME) : "qwen3.5";
    const std::string & a = model.arch;

    auto & hp = model.hparams;
    hp.n_layer     = gf.u32(arch_key(gkey::BLOCK_COUNT, a));
    hp.n_embd      = gf.u32(arch_key(gkey::EMBD_LEN,    a));
    hp.n_ff        = gf.u32(arch_key(gkey::FF_LEN,      a));
    hp.n_head      = gf.u32(arch_key(gkey::N_HEAD,      a));
    hp.n_head_kv   = gf.u32(arch_key(gkey::N_HEAD_KV,   a));
    hp.head_dim    = gf.u32(arch_key(gkey::KEY_LEN,     a));
    hp.n_ctx_train = gf.u32(arch_key(gkey::CTX_LEN,     a));
    hp.rms_eps     = gf.f32(arch_key(gkey::RMS_EPS,     a));
    hp.rope_theta  = gf.f32_or(arch_key(gkey::ROPE_FREQ_BASE, a), 1e7f);

    model.d_conv  = gf.u32(arch_key(gkey::SSM_CONV_KERNEL, a));
    model.d_state = gf.u32(arch_key(gkey::SSM_STATE_SIZE,  a));
    model.n_group = gf.u32(arch_key(gkey::SSM_GROUP_COUNT, a));
    model.dt_rank = gf.u32(arch_key(gkey::SSM_DT_RANK,     a));
    model.d_inner = gf.u32(arch_key(gkey::SSM_INNER_SIZE,  a));
    model.full_attn_interval = gf.u32(arch_key(gkey::FULL_ATTN_INTERVAL, a));
    model.rope_dim_count     = gf.u32(arch_key(gkey::ROPE_DIM_COUNT,     a));
    {
        auto secs = gf.arr_i32(arch_key(gkey::ROPE_SECTIONS, a));
        for (int i = 0; i < 4 && i < (int) secs.size(); i++) model.rope_sections[i] = secs[i];
    }

    ggml_tensor * te = gf.tensor("token_embd.weight");
    if (!te) NANO_ABORT("missing token_embd.weight");
    hp.n_vocab = (int32_t) te->ne[1];

    NANO_LOG("model: %s | n_layer=%d n_embd=%d n_head=%d n_head_kv=%d head_dim=%d n_ff=%d n_vocab=%d",
             model.name.c_str(), hp.n_layer, hp.n_embd, hp.n_head, hp.n_head_kv, hp.head_dim, hp.n_ff, hp.n_vocab);
    NANO_LOG("model: ssm d_conv=%d d_state=%d n_group=%d dt_rank=%d d_inner=%d | full_attn_interval=%d rope_sections=[%d,%d,%d,%d]",
             model.d_conv, model.d_state, model.n_group, model.dt_rank, model.d_inner, model.full_attn_interval,
             model.rope_sections[0], model.rope_sections[1], model.rope_sections[2], model.rope_sections[3]);

    const bool on_gpu = mp.n_gpu_layers > 0 && cuda_available();
    model.n_gpu_layers = on_gpu ? hp.n_layer : 0;

    const size_t n_tensors = 3 + (size_t) hp.n_layer * 15;
    ggml_init_params ip = { ggml_tensor_overhead() * (n_tensors + 8), nullptr, /*no_alloc=*/true };
    model.ctx_meta = ggml_init(ip);

    auto dup = [&](const char * name) -> ggml_tensor * {
        ggml_tensor * src = gf.tensor(name);
        if (!src) return nullptr;
        ggml_tensor * t = ggml_dup_tensor(model.ctx_meta, src);
        ggml_set_name(t, name);
        return t;
    };

    model.tok_embd    = dup("token_embd.weight");
    model.output_norm = dup("output_norm.weight");
    model.output      = model.tok_embd;   // tied embeddings

    model.layers.resize(hp.n_layer);
    char nm[128];
    #define DUP(field, fmt) snprintf(nm, sizeof(nm), fmt, i); L.field = dup(nm); \
        if (!L.field) NANO_ABORT("missing tensor: %s", nm);
    for (int i = 0; i < hp.n_layer; i++) {
        auto & L = model.layers[i];
        DUP(attn_norm,      "blk.%d.attn_norm.weight");
        DUP(post_attn_norm, "blk.%d.post_attention_norm.weight");
        DUP(ffn_gate,       "blk.%d.ffn_gate.weight");
        DUP(ffn_up,         "blk.%d.ffn_up.weight");
        DUP(ffn_down,       "blk.%d.ffn_down.weight");
        if (model.is_recurrent(i)) {
            DUP(wqkv,       "blk.%d.attn_qkv.weight");
            DUP(wqkv_gate,  "blk.%d.attn_gate.weight");
            DUP(ssm_conv1d, "blk.%d.ssm_conv1d.weight");
            DUP(ssm_dt,     "blk.%d.ssm_dt.bias");
            DUP(ssm_a,      "blk.%d.ssm_a");
            DUP(ssm_alpha,  "blk.%d.ssm_alpha.weight");
            DUP(ssm_beta,   "blk.%d.ssm_beta.weight");
            DUP(ssm_norm,   "blk.%d.ssm_norm.weight");
            DUP(ssm_out,    "blk.%d.ssm_out.weight");
        } else {
            DUP(wq,          "blk.%d.attn_q.weight");
            DUP(wk,          "blk.%d.attn_k.weight");
            DUP(wv,          "blk.%d.attn_v.weight");
            DUP(wo,          "blk.%d.attn_output.weight");
            DUP(attn_q_norm, "blk.%d.attn_q_norm.weight");
            DUP(attn_k_norm, "blk.%d.attn_k_norm.weight");
        }
    }
    #undef DUP

    ggml_backend_buffer_type_t buft =
#ifdef GGML_USE_CUDA
        on_gpu ? ggml_backend_cuda_buffer_type(0) : ggml_backend_cpu_buffer_type();
#else
        ggml_backend_cpu_buffer_type();
#endif
    NANO_LOG("model: placing weights on %s", on_gpu ? "CUDA0" : "CPU");

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(model.ctx_meta, buft);
    if (!buf) NANO_ABORT("failed to allocate weight buffer");
    model.bufs.push_back(buf);

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
    load_one(model.tok_embd);   // output is tied to it (no separate output.weight)
    load_one(model.output_norm);
    for (auto & L : model.layers) {
        for (ggml_tensor * t : { L.attn_norm, L.post_attn_norm, L.ffn_gate, L.ffn_up, L.ffn_down,
                                 L.wq, L.wk, L.wv, L.wo, L.attn_q_norm, L.attn_k_norm,
                                 L.wqkv, L.wqkv_gate, L.ssm_conv1d, L.ssm_dt, L.ssm_a,
                                 L.ssm_alpha, L.ssm_beta, L.ssm_norm, L.ssm_out })
            load_one(t);
    }

    {   // host F32 embedding table for the lookup (GPU get_rows can't read Q6_K)
        ggml_tensor * src = gf.tensor("token_embd.weight");
        auto to_float = ggml_get_type_traits(src->type)->to_float;
        if (!to_float) NANO_ABORT("token_embd type %s has no dequantizer", ggml_type_name(src->type));
        const int64_t n = (int64_t) hp.n_vocab * hp.n_embd;
        staging.resize(ggml_nbytes(src));
        fin.seekg((std::streamoff) gf.tensor_file_offset("token_embd.weight"));
        fin.read(staging.data(), (std::streamsize) ggml_nbytes(src));
        if (!fin) NANO_ABORT("short read for embedding dequant");
        model.embd_f32.resize(n);
        to_float(staging.data(), model.embd_f32.data(), n);
    }
    NANO_LOG("model: loaded %d layers (%d recurrent, %d attention)",
             hp.n_layer, hp.n_layer - hp.n_layer / model.full_attn_interval, hp.n_layer / model.full_attn_interval);
    return true;
}

} // namespace nano
