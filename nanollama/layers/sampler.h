// sampler.h — token sampling chain
#pragma once

#include "nanollama/config.h"
#include <cstdint>
#include <random>

namespace nano {

struct Sampler {
    SamplingParams sp;
    std::mt19937   rng;

    void init(const SamplingParams & p);

    // Pick a token from logits[n_vocab].  temperature <= 0 ⇒ greedy (argmax).
    int32_t sample(const float * logits, int n_vocab);
};

} // namespace nano
