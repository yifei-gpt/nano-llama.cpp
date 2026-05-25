// vlm.cpp — assemble a multimodal ChatML prompt, splice image embeddings with M-RoPE, generate
#include "nanollama/vision/vlm.h"
#include "nanollama/vision/clip.h"
#include "nanollama/engine/llm.h"
#include "nanollama/layers/sampler.h"

#include <algorithm>

namespace nano {

VlmInput build_vlm_input(const Model & model, const Vocab & vocab, ClipModel * clip,
                         const ClipImage * img, const std::string & user_text, bool think) {
    const int     n_embd  = model.hparams.n_embd;
    const float * tbl     = model.embd_f32.data();
    const bool    has_img = img && clip;

    std::vector<float> vemb; int n_img = 0, gw = 0, gh = 0;
    if (has_img) vemb = clip_encode(*clip, *img, n_img, gw, gh);

    // ChatML user turn, with a <|vision_start|>…<|vision_end|> span when an image is present
    std::string pre = has_img ? "<|im_start|>user\n<|vision_start|>" : "<|im_start|>user\n";
    std::string suf = (has_img ? std::string("<|vision_end|>") : std::string()) + user_text
                      + "<|im_end|>\n<|im_start|>assistant\n";
    if (!think) suf += "<think>\n\n</think>\n\n";
    auto pretok = vocab.tokenize(pre, false, true);
    auto suftok = vocab.tokenize(suf, false, true);

    // input embeddings (text via host lookup, image via clip) + M-RoPE positions
    VlmInput in;
    const int N = (int) pretok.size() + n_img + (int) suftok.size();
    in.n_tokens = N;
    in.embd.resize((size_t) N * n_embd);
    in.mrope.resize((size_t) 4 * N);
    std::vector<float>   & E = in.embd;
    std::vector<int32_t> & P = in.mrope;
    auto emit_text = [&](int32_t T, int j, int pos) {
        std::copy_n(tbl + (size_t) T * n_embd, n_embd, E.begin() + (size_t) j * n_embd);
        P[j] = P[N + j] = P[2 * N + j] = pos; P[3 * N + j] = 0;
    };
    int p = 0, j = 0;
    for (int32_t T : pretok) emit_text(T, j++, p++);
    if (has_img) {
        const int base = p;
        for (int i = 0; i < n_img; i++) {
            std::copy_n(vemb.begin() + (size_t) i * n_embd, n_embd, E.begin() + (size_t) j * n_embd);
            P[j] = base; P[N + j] = base + i / gw; P[2 * N + j] = base + i % gw; P[3 * N + j] = 0; j++;
        }
        p = base + std::max(gw, gh);
    }
    for (int32_t T : suftok) emit_text(T, j++, p++);
    in.mrope_next = p;
    return in;
}

std::string generate_vlm(LLM & llm, ClipModel * clip, const ClipImage * img,
                         const std::string & user_text, const SamplingParams & sp, bool think,
                         const std::function<void(const std::string &)> & on_piece,
                         const std::function<bool()> & cancelled) {
    const Vocab & vocab  = llm.get_vocab();
    const int     n_embd = llm.model->hparams.n_embd;
    const float * tbl    = llm.model->embd_f32.data();

    VlmInput in = build_vlm_input(*llm.model, vocab, clip, img, user_text, think);

    Sampler sampler; sampler.init(sp);
    const int n_vocab = llm.runner.n_vocab();
    int32_t next = sampler.sample(llm.runner.decode_embd(in.embd.data(), in.mrope.data(), in.n_tokens, 0), n_vocab);

    std::string out;
    int n_past = in.n_tokens, p = in.mrope_next;
    for (int t = 0; t < sp.n_predict && n_past < llm.runner.cp.n_ctx; t++) {   // stop before the cache fills
        if (cancelled && cancelled()) break;
        if (!sp.ignore_eos && vocab.is_eog(next)) break;
        std::string piece = vocab.token_to_piece(next);
        out += piece;
        if (on_piece) on_piece(piece);
        std::vector<float> e(tbl + (size_t) next * n_embd, tbl + (size_t) (next + 1) * n_embd);
        int32_t pos4[4] = { p, p, p, 0 };
        next = sampler.sample(llm.runner.decode_embd(e.data(), pos4, 1, n_past), n_vocab);
        n_past++; p++;
    }
    return out;
}

} // namespace nano
