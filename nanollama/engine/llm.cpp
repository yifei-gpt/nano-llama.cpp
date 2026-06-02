// llm.cpp — single-sequence generate() facade
#include "nanollama/engine/llm.h"
#include "nanollama/layers/sampler.h"
#include "nanollama/common.h"

namespace nano {

void LLM::load(const ModelParams & mp, const ContextParams & cp) {
    model.reset(load_model(mp));
    if (!model) NANO_ABORT("model load failed");
    vocab.load(GgufFile(mp.path));
    runner.init(*model, cp);
}

std::string LLM::generate(const std::vector<int32_t> & prompt_tokens,
                          const SamplingParams & sp,
                          const std::function<void(const std::string &)> & on_piece) {
    Sampler sampler;
    sampler.init(sp);

    const int n_vocab = runner.n_vocab();
    const int n_ctx   = runner.cp.n_ctx;
    NANO_ASSERT(!prompt_tokens.empty());

    // prompt + generation must fit n_ctx cells; keep the most recent n_ctx tokens
    int n_prompt = (int) prompt_tokens.size();
    const int32_t * pdata = prompt_tokens.data();
    if (n_prompt > n_ctx) { NANO_LOG("prompt truncated to n_ctx=%d (was %d)", n_ctx, n_prompt); pdata += n_prompt - n_ctx; n_prompt = n_ctx; }

    const float * logits = runner.decode(pdata, n_prompt, 0);
    int n_past = n_prompt;
    int32_t next = sampler.sample(logits, n_vocab);

    std::string out;
    for (int t = 0; (sp.n_predict < 0 || t < sp.n_predict) && n_past < n_ctx; t++) {   // n_predict < 0 ⇒ unbounded
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
