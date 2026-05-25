// clip.cpp — load the Qwen3-VL mmproj ViT and encode an image to LLM-space embeddings
#include "nanollama/vision/clip.h"
#include "nanollama/utils/gguf.h"
#include "nanollama/common.h"

#include "ggml-backend.h"
#include "ggml-cpu.h"
#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif

#include <cmath>
#include <fstream>

namespace nano {

ClipModel::~ClipModel() {
    if (galloc) ggml_gallocr_free(galloc);
    for (auto b : bufs) if (b) ggml_backend_buffer_free(b);
    if (ctx)     ggml_free(ctx);
    if (backend) ggml_backend_free(backend);
}

static bool cuda_ok() {
#ifdef GGML_USE_CUDA
    return ggml_backend_cuda_get_device_count() > 0;
#else
    return false;
#endif
}

bool clip_load(ClipModel & m, const std::string & path, int n_gpu_layers) {
    GgufFile gf(path);
    if (gf.str(gkey::ARCH) != "clip") NANO_ABORT("mmproj is not a clip model");

    m.n_embd     = gf.u32("clip.vision.embedding_length");
    m.n_layer    = gf.u32("clip.vision.block_count");
    m.n_head     = gf.u32("clip.vision.attention.head_count");
    m.patch_size = gf.u32("clip.vision.patch_size");
    m.proj_dim   = gf.u32("clip.vision.projection_dim");
    m.n_merge    = gf.u32("clip.vision.spatial_merge_size");
    m.eps        = gf.f32("clip.vision.attention.layer_norm_epsilon");
    { auto a = gf.arr_f32("clip.vision.image_mean"); for (int i = 0; i < 3 && i < (int) a.size(); i++) m.mean[i] = a[i]; }
    { auto a = gf.arr_f32("clip.vision.image_std");  for (int i = 0; i < 3 && i < (int) a.size(); i++) m.std[i]  = a[i]; }
    if (m.n_merge != 2) NANO_ABORT("clip: only spatial_merge_size=2 supported, got %d", m.n_merge);

    NANO_LOG("clip: n_embd=%d n_layer=%d n_head=%d patch=%d merge=%d proj=%d",
             m.n_embd, m.n_layer, m.n_head, m.patch_size, m.n_merge, m.proj_dim);

    m.on_gpu = n_gpu_layers > 0 && cuda_ok();
#ifdef GGML_USE_CUDA
    m.backend = m.on_gpu ? ggml_backend_cuda_init(0) : ggml_backend_cpu_init();
#else
    m.backend = ggml_backend_cpu_init();
#endif

    const size_t n_tensors = 11 + (size_t) m.n_layer * 12;
    ggml_init_params ip = { ggml_tensor_overhead() * (n_tensors + 8), nullptr, /*no_alloc=*/true };
    m.ctx = ggml_init(ip);

    auto dup = [&](const char * name) -> ggml_tensor * {
        ggml_tensor * src = gf.tensor(name);
        if (!src) NANO_ABORT("clip: missing tensor %s", name);
        ggml_tensor * t = ggml_dup_tensor(m.ctx, src);
        ggml_set_name(t, name);
        return t;
    };
    m.patch_embd_0  = dup("v.patch_embd.weight");
    m.patch_embd_1  = dup("v.patch_embd.weight.1");
    m.patch_bias    = dup("v.patch_embd.bias");
    m.position_embd = dup("v.position_embd.weight");
    m.post_ln_w     = dup("v.post_ln.weight");
    m.post_ln_b     = dup("v.post_ln.bias");
    m.mm0_w = dup("mm.0.weight"); m.mm0_b = dup("mm.0.bias");
    m.mm1_w = dup("mm.2.weight"); m.mm1_b = dup("mm.2.bias");
    m.layers.resize(m.n_layer);
    char nm[96];
    #define VDUP(field, fmt) snprintf(nm, sizeof(nm), fmt, i); L.field = dup(nm);
    for (int i = 0; i < m.n_layer; i++) {
        auto & L = m.layers[i];
        VDUP(ln1_w,  "v.blk.%d.ln1.weight");    VDUP(ln1_b,  "v.blk.%d.ln1.bias");
        VDUP(ln2_w,  "v.blk.%d.ln2.weight");    VDUP(ln2_b,  "v.blk.%d.ln2.bias");
        VDUP(qkv_w,  "v.blk.%d.attn_qkv.weight"); VDUP(qkv_b, "v.blk.%d.attn_qkv.bias");
        VDUP(o_w,    "v.blk.%d.attn_out.weight"); VDUP(o_b,   "v.blk.%d.attn_out.bias");
        VDUP(up_w,   "v.blk.%d.ffn_up.weight");   VDUP(up_b,  "v.blk.%d.ffn_up.bias");
        VDUP(down_w, "v.blk.%d.ffn_down.weight"); VDUP(down_b,"v.blk.%d.ffn_down.bias");
    }
    #undef VDUP

    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(m.backend);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(m.ctx, buft);
    if (!buf) NANO_ABORT("clip: failed to allocate weights");
    m.bufs.push_back(buf);
    m.galloc = ggml_gallocr_new(buft);

    std::ifstream fin(path, std::ios::binary);
    std::vector<char> staging;
    for (ggml_tensor * t = ggml_get_first_tensor(m.ctx); t; t = ggml_get_next_tensor(m.ctx, t)) {
        size_t off = gf.tensor_file_offset(t->name), nb = ggml_nbytes(t);
        staging.resize(nb);
        fin.seekg((std::streamoff) off); fin.read(staging.data(), (std::streamsize) nb);
        if (!fin) NANO_ABORT("clip: short read for %s", t->name);
        ggml_backend_tensor_set(t, staging.data(), 0, nb);
    }
    NANO_LOG("clip: loaded %d ViT layers on %s", m.n_layer, m.on_gpu ? "CUDA0" : "CPU");
    return true;
}

// ---- ViT graph helpers ----
static ggml_tensor * vnorm(ggml_context * c, ggml_tensor * x, ggml_tensor * w, ggml_tensor * b, float eps) {
    x = ggml_norm(c, x, eps);
    x = ggml_mul(c, x, w);
    return ggml_add(c, x, b);
}

// reorder patches/positions into 2x2-merge-block raster order ([ne, npx, npy] -> [ne, npx*npy])
static ggml_tensor * reorder_2x2(ggml_context * c, ggml_tensor * t, int ne, int npx, int npy) {
    t = ggml_cont_4d(c, t, ne * 2, npx / 2, npy, 1);
    t = ggml_reshape_4d(c, t, ne * 2, npx / 2, 2, npy / 2);
    t = ggml_permute(c, t, 0, 2, 1, 3);
    return ggml_cont_3d(c, t, ne, npx * npy, 1);
}

std::vector<float> clip_encode(ClipModel & m, const ClipImage & img, int & n_tokens, int & gw, int & gh) {
    const int patch = m.patch_size, ne = m.n_embd, nh = m.n_head, dh = ne / nh;
    const int npx = img.nx / patch, npy = img.ny / patch, n_pos = npx * npy;
    gw = npx / m.n_merge; gh = npy / m.n_merge; n_tokens = gw * gh;
    const float kq_scale = 1.0f / sqrtf((float) dh);
    int sections[4] = { dh / 4, dh / 4, dh / 4, dh / 4 };

    const size_t mem = ggml_tensor_overhead() * 4096 + ggml_graph_overhead_custom(4096, false);
    std::vector<char> gbuf(mem);
    ggml_init_params ip = { mem, gbuf.data(), /*no_alloc=*/true };
    ggml_context * ctx = ggml_init(ip);
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 4096, false);

    ggml_tensor * inp_raw = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, img.nx, img.ny, 3); ggml_set_input(inp_raw);
    ggml_tensor * positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_pos * 4);       ggml_set_input(positions);

    // dual conv2d patch embedding + 2x2 reorder + patch bias
    ggml_tensor * inp = ggml_add(ctx,
        ggml_conv_2d(ctx, m.patch_embd_0, inp_raw, patch, patch, 0, 0, 1, 1),
        ggml_conv_2d(ctx, m.patch_embd_1, inp_raw, patch, patch, 0, 0, 1, 1));
    inp = ggml_permute(ctx, inp, 1, 2, 0, 3);
    inp = reorder_2x2(ctx, inp, ne, npx, npy);
    inp = ggml_add(ctx, inp, m.patch_bias);

    // interpolated learned position embedding (48x48 grid -> npx x npy), same reorder
    const int nside = (int) std::sqrt((double) m.position_embd->ne[1]);
    ggml_tensor * pe = ggml_reshape_3d(ctx, m.position_embd, ne, nside, nside);
    pe = ggml_permute(ctx, pe, 2, 0, 1, 3);
    pe = ggml_interpolate(ctx, pe, npx, npy, ne, 1, GGML_SCALE_MODE_BILINEAR);
    pe = ggml_permute(ctx, pe, 1, 2, 0, 3);
    pe = ggml_cont_2d(ctx, pe, ne, npx * npy);
    inp = ggml_add(ctx, inp, reorder_2x2(ctx, pe, ne, npx, npy));

    ggml_tensor * inpL = inp;
    for (int il = 0; il < m.n_layer; il++) {
        const auto & L = m.layers[il];
        ggml_tensor * cur = vnorm(ctx, inpL, L.ln1_w, L.ln1_b, m.eps);

        ggml_tensor * qkv = ggml_add(ctx, ggml_mul_mat(ctx, L.qkv_w, cur), L.qkv_b);
        ggml_tensor * Q = ggml_view_3d(ctx, qkv, dh, nh, n_pos, ggml_row_size(qkv->type, dh), qkv->nb[1], 0);
        ggml_tensor * K = ggml_view_3d(ctx, qkv, dh, nh, n_pos, ggml_row_size(qkv->type, dh), qkv->nb[1], ggml_row_size(qkv->type, ne));
        ggml_tensor * V = ggml_view_3d(ctx, qkv, dh, nh, n_pos, ggml_row_size(qkv->type, dh), qkv->nb[1], ggml_row_size(qkv->type, 2 * ne));
        Q = ggml_rope_multi(ctx, Q, positions, nullptr, dh / 2, sections, GGML_ROPE_TYPE_VISION, 32768, 10000.0f, 1, 0, 1, 32, 1);
        K = ggml_rope_multi(ctx, K, positions, nullptr, dh / 2, sections, GGML_ROPE_TYPE_VISION, 32768, 10000.0f, 1, 0, 1, 32, 1);

        ggml_tensor * q = ggml_permute(ctx, Q, 0, 2, 1, 3);
        ggml_tensor * k = ggml_permute(ctx, K, 0, 2, 1, 3);
        ggml_tensor * v = ggml_cont(ctx, ggml_permute(ctx, V, 1, 2, 0, 3));
        ggml_tensor * kq = ggml_soft_max_ext(ctx, ggml_mul_mat(ctx, k, q), nullptr, kq_scale, 0.0f);
        ggml_tensor * kqv = ggml_mul_mat(ctx, v, kq);
        kqv = ggml_cont_2d(ctx, ggml_permute(ctx, kqv, 0, 2, 1, 3), ne, n_pos);
        cur = ggml_add(ctx, ggml_mul_mat(ctx, L.o_w, kqv), L.o_b);

        inpL = ggml_add(ctx, cur, inpL);
        cur  = vnorm(ctx, inpL, L.ln2_w, L.ln2_b, m.eps);
        cur  = ggml_add(ctx, ggml_mul_mat(ctx, L.up_w, cur), L.up_b);
        cur  = ggml_gelu(ctx, cur);
        cur  = ggml_add(ctx, ggml_mul_mat(ctx, L.down_w, cur), L.down_b);
        inpL = ggml_add(ctx, inpL, cur);
    }

    ggml_tensor * emb = vnorm(ctx, inpL, m.post_ln_w, m.post_ln_b, m.eps);
    // merger: 2x2 patches -> one token, 2-layer GELU MLP -> LLM dim
    emb = ggml_reshape_3d(ctx, emb, ne * 4, n_pos / 4, 1);
    emb = ggml_add(ctx, ggml_mul_mat(ctx, m.mm0_w, emb), m.mm0_b);
    emb = ggml_gelu(ctx, emb);
    emb = ggml_add(ctx, ggml_mul_mat(ctx, m.mm1_w, emb), m.mm1_b);
    ggml_set_output(emb);
    ggml_build_forward_expand(gf, emb);

    if (!ggml_gallocr_alloc_graph(m.galloc, gf)) NANO_ABORT("clip: graph alloc failed");
    ggml_backend_tensor_set(inp_raw, img.data.data(), 0, img.data.size() * sizeof(float));
    {
        std::vector<int32_t> pos(n_pos * 4);
        int ptr = 0;
        for (int y = 0; y < npy; y += 2)
            for (int x = 0; x < npx; x += 2)
                for (int dy = 0; dy < 2; dy++)
                    for (int dx = 0; dx < 2; dx++) {
                        pos[ptr] = pos[2 * n_pos + ptr] = y + dy;
                        pos[n_pos + ptr] = pos[3 * n_pos + ptr] = x + dx;
                        ptr++;
                    }
        ggml_backend_tensor_set(positions, pos.data(), 0, pos.size() * sizeof(int32_t));
    }
    if (ggml_backend_graph_compute(m.backend, gf) != GGML_STATUS_SUCCESS) NANO_ABORT("clip compute failed");

    std::vector<float> out((size_t) m.proj_dim * n_tokens);
    ggml_backend_tensor_get(emb, out.data(), 0, out.size() * sizeof(float));
    ggml_free(ctx);
    return out;
}

} // namespace nano
