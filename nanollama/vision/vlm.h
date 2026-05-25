// vlm.h — single-sequence vision-language generation (Qwen3.5): ChatML prompt + optional image
#pragma once

#include "nanollama/config.h"

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace nano {

struct LLM;
struct Model;
struct Vocab;
struct ClipModel;
struct ClipImage;

// Spliced prefill input for one ChatML user turn (text token embeddings + optional image embeddings),
// with per-token M-RoPE positions. mrope_next is the M-RoPE position of the first generated token.
struct VlmInput {
    std::vector<float>   embd;       // [n_embd * n_tokens]
    std::vector<int32_t> mrope;      // [4 * n_tokens]
    int                  n_tokens   = 0;
    int                  mrope_next = 0;
};

// Build the spliced embeddings + M-RoPE positions for a user turn (encodes the image via clip if present).
VlmInput build_vlm_input(const Model & model, const Vocab & vocab, ClipModel * clip,
                         const ClipImage * img, const std::string & user_text, bool think);

// Build a ChatML prompt from user_text (+ an optional preprocessed image encoded via clip), then
// generate greedily/sampled. Emits each token piece via on_piece; stops on EOS / n_predict / cancel.
// Returns the full generated text. img/clip may be null for text-only.
std::string generate_vlm(LLM & llm, ClipModel * clip, const ClipImage * img,
                         const std::string & user_text, const SamplingParams & sp, bool think,
                         const std::function<void(const std::string &)> & on_piece = nullptr,
                         const std::function<bool()> & cancelled = nullptr);

} // namespace nano
