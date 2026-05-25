// model.cpp — model base cleanup + architecture dispatch
#include "nanollama/models/model.h"
#include "nanollama/models/qwen3.h"
#include "nanollama/utils/gguf.h"
#include "nanollama/common.h"

namespace nano {

Model::~Model() {
    for (auto b : bufs) if (b) ggml_backend_buffer_free(b);
    if (ctx_meta)   ggml_free(ctx_meta);
    if (ctx_repack) ggml_free(ctx_repack);
}

Model * load_model(const ModelParams & mp) {
    std::string arch;
    { GgufFile gf(mp.path); arch = gf.str(gkey::ARCH); }   // peek the architecture

    if (arch == "qwen3") {
        auto * m = new qwen3_model();
        if (!qwen3_load(*m, mp)) { delete m; return nullptr; }
        return m;
    }
    NANO_ABORT("unsupported architecture '%s'", arch.c_str());
    return nullptr;
}

} // namespace nano
