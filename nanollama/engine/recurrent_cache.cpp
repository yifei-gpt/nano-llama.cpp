// recurrent_cache.cpp — allocate per-layer conv + delta state on a backend
#include "nanollama/engine/recurrent_cache.h"
#include "nanollama/models/model.h"
#include "nanollama/common.h"
#include "ggml-alloc.h"

namespace nano {

void RecurrentCache::init(const Model & m, ggml_backend_buffer_type_t buft) {
    n_layer = m.hparams.n_layer;
    const int n_embd_r = m.recurrent_conv_size();
    const int n_embd_s = m.recurrent_state_size();
    if (n_embd_r <= 0) return;   // not a recurrent model

    ggml_init_params ip = { ggml_tensor_overhead() * (2 * n_layer + 4), nullptr, /*no_alloc=*/true };
    ctx = ggml_init(ip);

    conv.assign(n_layer, nullptr);
    ssm.assign(n_layer, nullptr);
    char nm[64];
    for (int il = 0; il < n_layer; il++) {
        if (!m.is_recurrent_layer(il)) continue;
        conv[il] = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd_r, 1);
        ssm[il]  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd_s, 1);
        snprintf(nm, sizeof(nm), "conv_%d", il); ggml_set_name(conv[il], nm);
        snprintf(nm, sizeof(nm), "ssm_%d",  il); ggml_set_name(ssm[il],  nm);
    }

    buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
    if (!buf) NANO_ABORT("failed to allocate recurrent state buffer");
    ggml_backend_buffer_clear(buf, 0);

    const double mb = ggml_backend_buffer_get_size(buf) / 1024.0 / 1024.0;
    NANO_LOG("recurrent-cache: %.1f MiB (conv %d, state %d per recurrent layer)", mb, n_embd_r, n_embd_s);
}

void RecurrentCache::clear() {
    if (buf) ggml_backend_buffer_clear(buf, 0);
}

void RecurrentCache::free() {
    if (buf) { ggml_backend_buffer_free(buf); buf = nullptr; }
    if (ctx) { ggml_free(ctx); ctx = nullptr; }
    conv.clear(); ssm.clear();
}

} // namespace nano
