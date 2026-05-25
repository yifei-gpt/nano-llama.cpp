// qwen35.cpp — Qwen3.5 hybrid (gated-DeltaNet + periodic full-attention): load + forward graph
#include "nanollama/models/qwen35.h"
#include "nanollama/common.h"
#include "nanollama/utils/gguf.h"
#include "nanollama/engine/kv_cache.h"
#include "nanollama/engine/recurrent_cache.h"
#include "nanollama/layers/ops.h"
#include "nanollama/layers/attention.h"

#include "ggml-backend.h"
#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif

#include <cmath>
#include <fstream>
#include <string>

namespace nano {

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

    load_embd_table(model, gf, mp.path);
    NANO_LOG("model: loaded %d layers (%d recurrent, %d attention)",
             hp.n_layer, hp.n_layer - hp.n_layer / model.full_attn_interval, hp.n_layer / model.full_attn_interval);
    return true;
}

// recurrent (gated DeltaNet) layer — batched over sl.n_stream sequences of sl.n_q tokens each,
// each reading/writing its own slot in the conv/delta state cache (s0 + stream).
static ggml_tensor * build_recurrent(ggml_context * ctx, ggml_cgraph * gf, const qwen35_model & m,
                                     const qwen35_layer & L, ggml_tensor * cur,
                                     ggml_tensor * conv_cache, ggml_tensor * ssm_cache, const StreamLayout & sl) {
    const int    hk = m.d_state, nkh = m.n_group, nvh = m.dt_rank, hv = m.d_inner / nvh;
    const int    conv_ch = m.conv_channels(), dc = m.d_conv;
    const int    ns = sl.n_stream, nq = sl.n_q;   // n_tokens = ns * nq, token-major by stream
    const int    er = m.n_embd_r(), es = m.n_embd_s();
    const float  eps = m.hparams.rms_eps;
    const size_t cnb = conv_cache->nb[1], snb = ssm_cache->nb[1], coff = (size_t) sl.s0 * cnb, soff = (size_t) sl.s0 * snb;

    // projections + gates
    ggml_tensor * qkv = ggml_reshape_3d(ctx, linear(ctx, L.wqkv, cur), conv_ch, nq, ns);
    ggml_tensor * z   = linear(ctx, L.wqkv_gate, cur);
    ggml_tensor * beta = ggml_sigmoid(ctx, ggml_reshape_4d(ctx, linear(ctx, L.ssm_beta, cur), 1, nvh, nq, ns));
    ggml_tensor * alpha = ggml_reshape_3d(ctx, linear(ctx, L.ssm_alpha, cur), nvh, nq, ns);
    ggml_tensor * gate  = ggml_reshape_4d(ctx,
        ggml_mul(ctx, ggml_softplus(ctx, ggml_add(ctx, alpha, L.ssm_dt)), L.ssm_a), 1, nvh, nq, ns);

    // causal short conv over [prev per-stream conv state | qkv], write back the last (dc-1) frames
    ggml_tensor * conv_prev  = ggml_reshape_3d(ctx, ggml_view_2d(ctx, conv_cache, er, ns, cnb, coff), dc - 1, conv_ch, ns);
    ggml_tensor * conv_input = ggml_concat(ctx, conv_prev, ggml_transpose(ctx, qkv), 0);
    {
        ggml_tensor * last = ggml_view_3d(ctx, conv_input, dc - 1, conv_ch, ns,
                                          conv_input->nb[1], conv_input->nb[2], ggml_row_size(conv_input->type, nq));
        ggml_build_forward_expand(gf, ggml_cpy(ctx, last, ggml_view_2d(ctx, conv_cache, er, ns, cnb, coff)));
    }
    ggml_tensor * conv_out = ggml_silu(ctx, ggml_ssm_conv(ctx, conv_input, L.ssm_conv1d));

    // split conv output into q/k/v; L2-normalize q,k
    const size_t nb1 = ggml_row_size(conv_out->type, conv_ch);
    ggml_tensor * q = ggml_l2_norm(ctx, ggml_view_4d(ctx, conv_out, hk, nkh, nq, ns,
                          ggml_row_size(conv_out->type, hk), nb1, nb1 * nq, 0), eps);
    ggml_tensor * k = ggml_l2_norm(ctx, ggml_view_4d(ctx, conv_out, hk, nkh, nq, ns,
                          ggml_row_size(conv_out->type, hk), nb1, nb1 * nq, ggml_row_size(conv_out->type, hk * nkh)), eps);
    ggml_tensor * v = ggml_cont(ctx, ggml_view_4d(ctx, conv_out, hv, nvh, nq, ns,
                          ggml_row_size(conv_out->type, hv), nb1, nb1 * nq, ggml_row_size(conv_out->type, 2 * hk * nkh)));

    // fused gated delta net over the per-stream recurrent state; result packs [output | new_state]
    ggml_tensor * state = ggml_reshape_3d(ctx, ggml_view_2d(ctx, ssm_cache, es, ns, snb, soff), hv * hv * nvh, 1, ns);
    ggml_tensor * res = ggml_gated_delta_net(ctx, q, k, v, gate, beta, state);
    ggml_tensor * out = ggml_view_4d(ctx, res, hv, nvh, nq, ns,
                                     ggml_row_size(res->type, hv), ggml_row_size(res->type, hv * nvh),
                                     ggml_row_size(res->type, hv * nvh * nq), 0);
    ggml_tensor * new_state = ggml_view_4d(ctx, res, hv, hv, nvh, ns,
                                           ggml_row_size(res->type, hv), ggml_row_size(res->type, hv * hv),
                                           ggml_row_size(res->type, hv * hv * nvh),
                                           ggml_row_size(res->type, hv * nvh * nq * ns));
    ggml_build_forward_expand(gf, ggml_cpy(ctx, new_state, ggml_view_2d(ctx, ssm_cache, es, ns, snb, soff)));

    // gated RMSNorm (output·silu(z)) + output projection
    ggml_tensor * z2     = ggml_reshape_4d(ctx, z, hv, nvh, nq, ns);
    ggml_tensor * normed = gated_rms_norm(ctx, out, L.ssm_norm, z2, eps);
    ggml_tensor * o      = linear(ctx, L.ssm_out, ggml_reshape_3d(ctx, normed, hv * nvh, nq, ns));
    return ggml_reshape_2d(ctx, o, m.hparams.n_embd, nq * ns);
}

// full-attention layer — fused QKV+gate projection, per-head Q/K norm, M-RoPE, gated output
static ggml_tensor * build_attn_layer(ggml_context * ctx, ggml_cgraph * gf, const qwen35_model & m,
                                      const qwen35_layer & L, ggml_tensor * cur, ggml_tensor * pos,
                                      KvCache & kv, int il, ggml_tensor * kv_idxs, ggml_tensor * mask,
                                      int n_kv, const StreamLayout & sl, bool flash) {
    const int   hd  = m.hparams.head_dim, nh = m.hparams.n_head, nkv = m.hparams.n_head_kv;
    const float eps = m.hparams.rms_eps;
    const int64_t n_tokens = cur->ne[1];

    ggml_tensor * qfull = linear(ctx, L.wq, cur);   // query + gate, two halves per head
    const size_t  es = ggml_element_size(qfull);
    ggml_tensor * Q = ggml_view_3d(ctx, qfull, hd, nh, n_tokens, es * hd * 2, es * hd * 2 * nh, 0);
    Q = rms_norm(ctx, Q, L.attn_q_norm, eps);
    ggml_tensor * gate = ggml_view_3d(ctx, qfull, hd, nh, n_tokens, es * hd * 2, es * hd * 2 * nh, es * hd);
    gate = ggml_cont_2d(ctx, gate, hd * nh, n_tokens);

    ggml_tensor * K = rms_norm(ctx, ggml_reshape_3d(ctx, linear(ctx, L.wk, cur), hd, nkv, n_tokens), L.attn_k_norm, eps);
    ggml_tensor * V = ggml_reshape_3d(ctx, linear(ctx, L.wv, cur), hd, nkv, n_tokens);

    Q = rope_mrope(ctx, Q, pos, m.rope_dim_count, m.rope_sections, m.hparams.rope_theta, m.hparams.n_ctx_train);
    K = rope_mrope(ctx, K, pos, m.rope_dim_count, m.rope_sections, m.hparams.rope_theta, m.hparams.n_ctx_train);

    ggml_tensor * att = build_attention(ctx, gf, Q, K, V, kv.k[il], kv.v[il], kv_idxs, mask, n_kv, sl,
                                        hd, nh, nkv, 1.0f / sqrtf((float) hd), flash);
    att = ggml_mul(ctx, att, ggml_sigmoid(ctx, gate));   // gated attention
    return linear(ctx, L.wo, att);
}

ggml_tensor * qwen35_model::build_graph(ggml_context * ctx, ggml_cgraph * gf, const graph_inputs & in,
                                        KvCache & kv, RecurrentCache * rc, int n_kv,
                                        const StreamLayout & sl, bool flash) const {
    const float eps = hparams.rms_eps;
    ggml_tensor * inpL = in.embd;

    for (int il = 0; il < hparams.n_layer; il++) {
        const auto & L = layers[il];
        ggml_tensor * inpSA = inpL;

        ggml_tensor * cur = rms_norm(ctx, inpL, L.attn_norm, eps);
        if (is_recurrent(il))
            cur = build_recurrent(ctx, gf, *this, L, cur, rc->conv[il], rc->ssm[il], sl);
        else
            cur = build_attn_layer(ctx, gf, *this, L, cur, in.pos, kv, il, in.kv_idxs, in.mask, n_kv, sl, flash);

        if (il == hparams.n_layer - 1 && in.out_ids) {
            cur   = ggml_get_rows(ctx, cur,   in.out_ids);
            inpSA = ggml_get_rows(ctx, inpSA, in.out_ids);
        }
        ggml_tensor * ffn_inp = ggml_add(ctx, cur, inpSA);

        cur = rms_norm(ctx, ffn_inp, L.post_attn_norm, eps);
        cur = swiglu(ctx, linear(ctx, L.ffn_gate, cur), linear(ctx, L.ffn_up, cur));
        cur = linear(ctx, L.ffn_down, cur);
        inpL = ggml_add(ctx, cur, ffn_inp);
    }

    ggml_tensor * cur = rms_norm(ctx, inpL, output_norm, eps);
    cur = linear(ctx, output, cur);
    ggml_build_forward_expand(gf, cur);
    return cur;
}

} // namespace nano
