// recurrent_cache.cpp — allocate per-slot conv + delta state on a backend
#include "nanollama/engine/recurrent_cache.h"
#include "nanollama/models/model.h"
#include "nanollama/common.h"
#include "ggml-alloc.h"

namespace nano {

void RecurrentCache::init(const Model & m, int slots, ggml_backend_buffer_type_t buft) {
    n_layer = m.hparams.n_layer;
    n_slots = slots;
    n_embd_r = m.recurrent_conv_size();
    n_embd_s = m.recurrent_state_size();
    if (n_embd_r <= 0) return;   // not a recurrent model

    ggml_init_params ip = { ggml_tensor_overhead() * (2 * n_layer + 4), nullptr, /*no_alloc=*/true };
    ctx = ggml_init(ip);

    conv.assign(n_layer, nullptr);
    ssm.assign(n_layer, nullptr);
    char nm[64];
    for (int il = 0; il < n_layer; il++) {
        if (!m.is_recurrent_layer(il)) continue;
        conv[il] = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd_r, n_slots);
        ssm[il]  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd_s, n_slots);
        snprintf(nm, sizeof(nm), "conv_%d", il); ggml_set_name(conv[il], nm);
        snprintf(nm, sizeof(nm), "ssm_%d",  il); ggml_set_name(ssm[il],  nm);
    }

    buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
    if (!buf) NANO_ABORT("failed to allocate recurrent state buffer");
    ggml_backend_buffer_clear(buf, 0);

    const double mb = ggml_backend_buffer_get_size(buf) / 1024.0 / 1024.0;
    NANO_LOG("recurrent-cache: %.1f MiB (%d slots, conv %d, state %d per recurrent layer)",
             mb, n_slots, n_embd_r, n_embd_s);
}

void RecurrentCache::clear() {
    if (buf) ggml_backend_buffer_clear(buf, 0);
}

void RecurrentCache::clear_slot(int slot) {
    if (!buf) return;
    std::vector<float> zr(n_embd_r, 0.0f), zs(n_embd_s, 0.0f);
    for (int il = 0; il < n_layer; il++) {
        if (!conv[il]) continue;
        ggml_backend_tensor_set(conv[il], zr.data(), (size_t) slot * n_embd_r * sizeof(float), zr.size() * sizeof(float));
        ggml_backend_tensor_set(ssm[il],  zs.data(), (size_t) slot * n_embd_s * sizeof(float), zs.size() * sizeof(float));
    }
}

void RecurrentCache::free() {
    if (buf) { ggml_backend_buffer_free(buf); buf = nullptr; }
    if (ctx) { ggml_free(ctx); ctx = nullptr; }
    conv.clear(); ssm.clear();
}

} // namespace nano
