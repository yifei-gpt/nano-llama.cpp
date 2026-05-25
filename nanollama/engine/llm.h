// llm.h — top-level facade
#pragma once

#include "nanollama/config.h"
#include "nanollama/models/qwen3.h"
#include "nanollama/engine/model_runner.h"
#include "nanollama/tokenizer/vocab.h"

#include <functional>
#include <string>
#include <vector>

namespace nano {

struct LLM {
    qwen3_model model;
    ModelRunner runner;
    Vocab       vocab;

    void load(const ModelParams & mp, const ContextParams & cp);

    const Vocab & get_vocab() const { return vocab; }

    // generate greedily/sampled from prompt_tokens, calling on_piece() per token; returns the text
    std::string generate(const std::vector<int32_t> & prompt_tokens,
                         const SamplingParams & sp,
                         const std::function<void(const std::string &)> & on_piece = nullptr);
};

} // namespace nano
