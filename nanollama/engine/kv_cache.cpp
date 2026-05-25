// kv_cache.cpp — allocate per-attention-layer F16 K/V buffers on a backend
#include "nanollama/engine/kv_cache.h"
#include "nanollama/models/model.h"
#include "nanollama/common.h"
#include "ggml-alloc.h"

namespace nano {

void KvCache::init(const Model & model, int n_slots_, int n_ctx, ggml_backend_buffer_type_t buft) {
    n_layer   = model.hparams.n_layer;
    n_embd_kv = model.hparams.n_embd_kv();
    n_slots   = n_slots_;
    n_ctx_pad = GGML_PAD(n_ctx, 256);    // multiple of 256 lets flash skip masked blocks
    n_cells   = n_slots * n_ctx_pad + 1; // +1 scratch cell: sink for dummy stream writes

    ggml_init_params ip = { ggml_tensor_overhead() * (2 * n_layer + 4), nullptr, /*no_alloc=*/true };
    ctx = ggml_init(ip);

    k.assign(n_layer, nullptr);
    v.assign(n_layer, nullptr);
    int n_attn = 0;
    for (int il = 0; il < n_layer; il++) {
        if (model.is_recurrent_layer(il)) continue;   // recurrent layers have no KV
        k[il] = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_embd_kv, n_cells);
        v[il] = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_embd_kv, n_cells);
        char nm[64];
        snprintf(nm, sizeof(nm), "cache_k_%d", il); ggml_set_name(k[il], nm);
        snprintf(nm, sizeof(nm), "cache_v_%d", il); ggml_set_name(v[il], nm);
        n_attn++;
    }

    buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
    if (!buf) NANO_ABORT("failed to allocate KV cache buffer");
    ggml_backend_buffer_clear(buf, 0);  // unwritten/padding cells must be finite, not NaN

    const double mb = ggml_backend_buffer_get_size(buf) / 1024.0 / 1024.0;
    NANO_LOG("kv-cache: %d/%d attn layers, %d slots x %d cells = %d, %.1f MiB", n_attn, n_layer, n_slots, n_ctx_pad, n_cells, mb);
}

void KvCache::free() {
    if (buf) { ggml_backend_buffer_free(buf); buf = nullptr; }
    if (ctx) { ggml_free(ctx); ctx = nullptr; }
    k.clear(); v.clear();
}

} // namespace nano
