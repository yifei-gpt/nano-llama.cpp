// qwen35_graph.cpp — Qwen3.5 hybrid forward pass (gated-DeltaNet + periodic full-attention)
#include "nanollama/models/qwen35.h"
#include "nanollama/engine/kv_cache.h"
#include "nanollama/engine/recurrent_cache.h"
#include "nanollama/layers/ops.h"
#include "nanollama/layers/attention.h"
#include "nanollama/common.h"

#include <cmath>

namespace nano {

// recurrent (gated DeltaNet) layer — single sequence (n_seqs = 1)
static ggml_tensor * build_recurrent(ggml_context * ctx, ggml_cgraph * gf, const qwen35_model & m,
                                     const qwen35_layer & L, ggml_tensor * cur,
                                     ggml_tensor * conv_state, ggml_tensor * ssm_state) {
    const int     hk      = m.d_state;            // head_k_dim == head_v_dim
    const int     nkh     = m.n_group;            // K heads
    const int     nvh     = m.dt_rank;            // V heads
    const int     hv      = m.d_inner / nvh;       // head_v_dim
    const int     conv_ch = m.conv_channels();
    const int     dc      = m.d_conv;
    const float   eps     = m.hparams.rms_eps;
    const int64_t n_tokens = cur->ne[1];

    // projections + gates
    ggml_tensor * qkv = ggml_reshape_3d(ctx, linear(ctx, L.wqkv, cur), conv_ch, n_tokens, 1);
    ggml_tensor * z   = linear(ctx, L.wqkv_gate, cur);                       // [value_dim, n_tokens]

    ggml_tensor * beta = ggml_reshape_4d(ctx, linear(ctx, L.ssm_beta, cur), 1, nvh, n_tokens, 1);
    beta = ggml_sigmoid(ctx, beta);

    ggml_tensor * alpha = ggml_reshape_3d(ctx, linear(ctx, L.ssm_alpha, cur), nvh, n_tokens, 1);
    ggml_tensor * gate  = ggml_mul(ctx, ggml_softplus(ctx, ggml_add(ctx, alpha, L.ssm_dt)), L.ssm_a);
    gate = ggml_reshape_4d(ctx, gate, 1, nvh, n_tokens, 1);                  // -A_log.exp()·softplus

    // causal short conv over [prev conv state | qkv], then write back the last (dc-1) frames
    ggml_tensor * conv_states = ggml_reshape_3d(ctx, conv_state, dc - 1, conv_ch, 1);
    ggml_tensor * conv_input  = ggml_concat(ctx, conv_states, ggml_transpose(ctx, qkv), 0);
    {
        const int64_t s_idx     = conv_input->ne[0] - (dc - 1);
        ggml_tensor * last      = ggml_view_3d(ctx, conv_input, dc - 1, conv_ch, 1,
                                               conv_input->nb[1], conv_input->nb[2],
                                               ggml_row_size(conv_input->type, s_idx));
        ggml_tensor * dst       = ggml_view_2d(ctx, conv_state, m.n_embd_r(), 1, conv_state->nb[1], 0);
        ggml_build_forward_expand(gf, ggml_cpy(ctx, last, dst));
    }
    ggml_tensor * conv_out = ggml_silu(ctx, ggml_ssm_conv(ctx, conv_input, L.ssm_conv1d)); // [conv_ch, n_tokens, 1]

    // split conv output into q/k/v and L2-normalize q,k
    const size_t nb1 = ggml_row_size(conv_out->type, conv_ch);
    ggml_tensor * q = ggml_view_4d(ctx, conv_out, hk, nkh, n_tokens, 1,
                                   ggml_row_size(conv_out->type, hk), nb1, nb1 * n_tokens, 0);
    ggml_tensor * k = ggml_view_4d(ctx, conv_out, hk, nkh, n_tokens, 1,
                                   ggml_row_size(conv_out->type, hk), nb1, nb1 * n_tokens,
                                   ggml_row_size(conv_out->type, hk * nkh));
    ggml_tensor * v = ggml_view_4d(ctx, conv_out, hv, nvh, n_tokens, 1,
                                   ggml_row_size(conv_out->type, hv), nb1, nb1 * n_tokens,
                                   ggml_row_size(conv_out->type, 2 * hk * nkh));
    q = ggml_l2_norm(ctx, q, eps);
    k = ggml_l2_norm(ctx, k, eps);
    v = ggml_cont(ctx, v);

    // fused gated delta net: result packs [output | new_state]
    ggml_tensor * s3  = ggml_reshape_3d(ctx, ssm_state, hv * hv * nvh, 1, 1);
    ggml_tensor * res = ggml_gated_delta_net(ctx, q, k, v, gate, beta, s3);
    ggml_tensor * out = ggml_view_4d(ctx, res, hv, nvh, n_tokens, 1,
                                     ggml_row_size(res->type, hv), ggml_row_size(res->type, hv * nvh),
                                     ggml_row_size(res->type, hv * nvh * n_tokens), 0);
    ggml_tensor * new_state = ggml_view_4d(ctx, res, hv, hv, nvh, 1,
                                           ggml_row_size(res->type, hv), ggml_row_size(res->type, hv * hv),
                                           ggml_row_size(res->type, hv * hv * nvh),
                                           ggml_row_size(res->type, hv * nvh * n_tokens));
    ggml_build_forward_expand(gf, ggml_cpy(ctx, new_state,
        ggml_view_2d(ctx, ssm_state, m.n_embd_s(), 1, ssm_state->nb[1], 0)));

    // gated RMSNorm (output·silu(z)) + output projection
    ggml_tensor * z2     = ggml_reshape_4d(ctx, z, hv, nvh, n_tokens, 1);
    ggml_tensor * normed = gated_rms_norm(ctx, out, L.ssm_norm, z2, eps);
    ggml_tensor * o      = linear(ctx, L.ssm_out, ggml_reshape_3d(ctx, normed, hv * nvh, n_tokens, 1));
    return ggml_reshape_2d(ctx, o, m.hparams.n_embd, n_tokens);
}

// full-attention layer — fused QKV+gate projection, per-head Q/K norm, M-RoPE, gated output
static ggml_tensor * build_attn_layer(ggml_context * ctx, ggml_cgraph * gf, const qwen35_model & m,
                                      const qwen35_layer & L, ggml_tensor * cur, ggml_tensor * pos,
                                      KvCache & kv, int il, ggml_tensor * kv_idxs, ggml_tensor * mask,
                                      int n_kv, const StreamLayout & sl, bool flash) {
    const int   hd  = m.hparams.head_dim, nh = m.hparams.n_head, nkv = m.hparams.n_head_kv;
    const float eps = m.hparams.rms_eps;
    const int64_t n_tokens = cur->ne[1];

    ggml_tensor * qfull = linear(ctx, L.wq, cur);   // [hd*2*nh, n_tokens] — query + gate, 2 halves per head
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
            cur = build_recurrent(ctx, gf, *this, L, cur, rc->conv[il], rc->ssm[il]);
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
