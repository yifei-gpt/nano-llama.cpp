// llm.cpp — single-sequence generate() facade
#include "nanollama/engine/llm.h"
#include "nanollama/layers/sampler.h"
#include "nanollama/common.h"

namespace nano {

void LLM::load(const ModelParams & mp, const ContextParams & cp) {
    if (!qwen3_load(model, mp)) NANO_ABORT("model load failed");
    vocab.load(GgufFile(mp.path));
    runner.init(model, cp);
}

std::string LLM::generate(const std::vector<int32_t> & prompt_tokens,
                          const SamplingParams & sp,
                          const std::function<void(const std::string &)> & on_piece) {
    Sampler sampler;
    sampler.init(sp);

    const int n_vocab = runner.n_vocab();
    const int n_ctx   = runner.cp.n_ctx;
    NANO_ASSERT(!prompt_tokens.empty());

    // prompt + generation must fit the cache (n_ctx cells); decode writes K/V at cell = n_past
    int n_prompt = (int) prompt_tokens.size();
    if (n_prompt > n_ctx) { NANO_LOG("prompt truncated to n_ctx=%d (was %d)", n_ctx, n_prompt); n_prompt = n_ctx; }

    const float * logits = runner.decode(prompt_tokens.data(), n_prompt, 0);
    int n_past = n_prompt;
    int32_t next = sampler.sample(logits, n_vocab);

    std::string out;
    for (int t = 0; t < sp.n_predict && n_past < n_ctx; t++) {
        if (!sp.ignore_eos && vocab.is_eog(next)) break;
        std::string piece = vocab.token_to_piece(next);
        out += piece;
        if (on_piece) on_piece(piece);

        const float * lg = runner.decode(&next, 1, n_past);
        n_past += 1;
        next = sampler.sample(lg, n_vocab);
    }
    return out;
}

} // namespace nano
