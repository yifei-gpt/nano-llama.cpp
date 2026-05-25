// config.h — parameter structs
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace nano {

struct ModelParams {
    std::string path;
    int  n_gpu_layers = 0;       // > 0 runs the whole model on GPU, 0 = CPU
};

struct ContextParams {
    int  n_ctx     = 4096;
    int  n_slots   = 1;
    int  n_threads = 16;
    bool flash_attn = true;
};

struct SamplingParams {        // temperature <= 0 ⇒ greedy
    float    temperature = 0.8f;
    int32_t  top_k       = 40;
    float    top_p       = 0.95f;
    float    min_p       = 0.05f;
    uint32_t seed        = 0xFFFFFFFF;
    int32_t  n_predict   = 256;
    bool     ignore_eos  = false;
};

} // namespace nano
