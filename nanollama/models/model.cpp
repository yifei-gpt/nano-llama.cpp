// model.cpp — model base cleanup, shared loader helpers, architecture dispatch
#include "nanollama/models/model.h"
#include "nanollama/models/qwen3.h"
#include "nanollama/models/qwen35.h"
#include "nanollama/utils/gguf.h"
#include "nanollama/common.h"

#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif

#include <fstream>
#include <vector>

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

// dequantize token_embd.weight into m.embd_f32 (the GPU get_rows path can't read most quant types)
void load_embd_table(Model & m, GgufFile & gf, const std::string & path) {
    ggml_tensor * src = gf.tensor("token_embd.weight");
    auto to_float = ggml_get_type_traits(src->type)->to_float;
    if (!to_float) NANO_ABORT("token_embd type %s has no dequantizer", ggml_type_name(src->type));
    std::ifstream fin(path, std::ios::binary);
    if (!fin) NANO_ABORT("cannot reopen model file for embedding table");
    std::vector<char> staging(ggml_nbytes(src));
    fin.seekg((std::streamoff) gf.tensor_file_offset("token_embd.weight"));
    fin.read(staging.data(), (std::streamsize) ggml_nbytes(src));
    if (!fin) NANO_ABORT("short read for embedding dequant");
    m.embd_f32.resize((size_t) m.hparams.n_vocab * m.hparams.n_embd);
    to_float(staging.data(), m.embd_f32.data(), m.embd_f32.size());
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
