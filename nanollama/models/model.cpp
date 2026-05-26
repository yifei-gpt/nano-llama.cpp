// model.cpp — model base cleanup, shared loader helpers, architecture dispatch
#include "nanollama/models/model.h"
#include "nanollama/models/qwen3.h"
#include "nanollama/models/qwen35.h"
#include "nanollama/utils/gguf.h"
#include "nanollama/common.h"

#include "ggml-backend.h"
#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif

namespace nano {

Model::~Model() {
    for (auto b : bufs) if (b) ggml_backend_buffer_free(b);
    if (ctx_meta)   ggml_free(ctx_meta);
    if (ctx_repack) ggml_free(ctx_repack);
}

bool cuda_available() {
#ifdef GGML_USE_CUDA
    return ggml_backend_cuda_get_device_count() > 0;
#else
    return false;
#endif
}

// dequantize the n requested token-embedding rows into dst (F32)
void Model::embed_tokens(const int32_t * ids, int n, float * dst) const {
    const auto    to_float = ggml_get_type_traits(embd_type)->to_float;
    const char *  base     = (const char *) embd_data;
    const int64_t n_vocab  = hparams.n_vocab;
    for (int i = 0; i < n; i++) {
        int64_t id = ids[i];
        if (id < 0 || id >= n_vocab) id = 0;   // guard against out-of-range ids (avoid OOB read)
        to_float(base + (size_t) id * embd_row_bytes, dst + (size_t) i * hparams.n_embd, hparams.n_embd);
    }
}

// point embed_tokens at the quantized embedding: the model's host buffer (CPU) or a host copy (GPU)
void load_embd(Model & m, ggml_tensor * tok_embd) {
    if (!tok_embd) NANO_ABORT("missing token_embd.weight");
    if (!ggml_get_type_traits(tok_embd->type)->to_float)
        NANO_ABORT("token_embd type %s has no dequantizer", ggml_type_name(tok_embd->type));
    m.embd_type      = tok_embd->type;
    m.embd_row_bytes = ggml_row_size(tok_embd->type, m.hparams.n_embd);
    if (m.n_gpu_layers > 0) {
        m.embd_host.resize(ggml_nbytes(tok_embd));
        ggml_backend_tensor_get(tok_embd, m.embd_host.data(), 0, m.embd_host.size());
        m.embd_data = m.embd_host.data();
    } else {
        m.embd_data = tok_embd->data;
    }
}

Model * load_model(const ModelParams & mp) {
    std::string arch;
    { GgufFile gf(mp.path); arch = gf.str(gkey::ARCH); }   // peek the architecture

    if (arch == "qwen3") {
        auto * m = new qwen3_model();
        if (!qwen3_load(*m, mp)) { delete m; return nullptr; }
        return m;
    }
    if (arch == "qwen35") {
        auto * m = new qwen35_model();
        if (!qwen35_load(*m, mp)) { delete m; return nullptr; }
        return m;
    }
    NANO_ABORT("unsupported architecture '%s'", arch.c_str());
    return nullptr;
}

} // namespace nano
