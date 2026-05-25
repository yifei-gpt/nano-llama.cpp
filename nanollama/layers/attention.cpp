// attention.cpp — per-stream GQA attention (flash + soft_max paths) over the KV cache
#include "nanollama/layers/attention.h"

namespace nano {

ggml_tensor * build_attention(
    ggml_context * ctx, ggml_cgraph * gf,
    ggml_tensor * q, ggml_tensor * k_cur, ggml_tensor * v_cur,
    ggml_tensor * k_cache, ggml_tensor * v_cache, ggml_tensor * kv_idxs,
    ggml_tensor * mask, int n_kv, const StreamLayout & sl,
    int head_dim, int n_head, int n_head_kv, float scale, bool flash) {

    const int64_t n_tokens  = q->ne[2];
    const int64_t n_embd_kv = (int64_t) head_dim * n_head_kv;
    const int     n_stream  = sl.n_stream;
    const int     n_q       = sl.n_q;

    // store this step's K/V into the cache at their global cells (slot*n_ctx_pad + pos)
    ggml_tensor * k_cur_2d = ggml_reshape_2d(ctx, k_cur, n_embd_kv, n_tokens);
    ggml_tensor * v_cur_2d = ggml_reshape_2d(ctx, v_cur, n_embd_kv, n_tokens);
    ggml_build_forward_expand(gf, ggml_set_rows(ctx, k_cache, k_cur_2d, kv_idxs));
    ggml_build_forward_expand(gf, ggml_set_rows(ctx, v_cache, v_cur_2d, kv_idxs));

    // per-stream 4D view [head_dim, n_head_kv, n_kv, n_stream]; strides as in llama.cpp's get_k
    const size_t rk      = ggml_row_size(k_cache->type, head_dim);
    const size_t rkv     = ggml_row_size(k_cache->type, n_embd_kv);
    const size_t rstream = ggml_row_size(k_cache->type, n_embd_kv * sl.n_ctx_pad);
    ggml_tensor * k = ggml_view_4d(ctx, k_cache, head_dim, n_head_kv, n_kv, n_stream,
                                   rk, rkv, rstream, rstream * sl.s0);
    ggml_tensor * v = ggml_view_4d(ctx, v_cache, head_dim, n_head_kv, n_kv, n_stream,
                                   rk, rkv, rstream, rstream * sl.s0);

    // split the flat token axis into (n_q, n_stream): [head_dim, n_head, n_q, n_stream]
    q = ggml_view_4d(ctx, q, head_dim, n_head, n_q, n_stream,
                     q->nb[1], q->nb[2], q->nb[2] * n_q, 0);

    // permute to [head_dim, seq, head, stream] so the matmul broadcasts n_head_kv → n_head (GQA)
    ggml_tensor * qp = ggml_permute(ctx, q, 0, 2, 1, 3);
    ggml_tensor * kp = ggml_permute(ctx, k, 0, 2, 1, 3);
    ggml_tensor * vp = ggml_permute(ctx, v, 0, 2, 1, 3);

    ggml_tensor * cur;
    if (flash) {
        // V is stored non-transposed (same layout as K) so vp is passed directly; mask must be F16
        cur = ggml_flash_attn_ext(ctx, qp, kp, vp, mask, scale, 0.0f, 0.0f);
        ggml_flash_attn_ext_set_prec(cur, GGML_PREC_F32);
        cur = ggml_reshape_2d(ctx, cur, head_dim * n_head, n_tokens);
    } else {
        ggml_tensor * kq = ggml_mul_mat(ctx, kp, qp);
        ggml_mul_mat_set_prec(kq, GGML_PREC_F32);
        kq = ggml_soft_max_ext(ctx, kq, mask, scale, 0.0f);
        ggml_tensor * vt = ggml_cont(ctx, ggml_transpose(ctx, vp));
        ggml_tensor * kqv = ggml_mul_mat(ctx, vt, kq);
        cur = ggml_permute(ctx, kqv, 0, 2, 1, 3);
        cur = ggml_cont_2d(ctx, cur, head_dim * n_head, n_tokens);
    }
    ggml_build_forward_expand(gf, cur);
    return cur;
}

} // namespace nano
