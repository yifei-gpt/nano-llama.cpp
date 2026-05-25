// engine.h — continuous-batching inference engine: many sequences share one batched forward pass
#pragma once

#include "nanollama/config.h"
#include "nanollama/models/model.h"
#include "nanollama/engine/model_runner.h"
#include "nanollama/tokenizer/vocab.h"
#include "nanollama/layers/sampler.h"

#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace nano {

// an in-flight request occupying a KV slot (its sequence is one attention stream)
struct Slot {
    int  id     = -1;
    bool active = false;

    std::vector<int32_t> prompt;
    std::vector<int32_t> generated;
    int  n_past     = 0;       // tokens written to this slot's KV (= next write position)
    int  n_prompt   = 0;       // prompt length (text: prompt.size(); image: spliced embedding count)
    bool generating = false;   // prompt fully prefilled; now in the decode loop

    // VLM image prefill (Qwen3.5): precomputed spliced embeddings + M-RoPE positions; empty for text.
    // mrope_next is the M-RoPE position of the next generated token, which diverges from n_past once an
    // image is present (an image consumes max(grid_w,grid_h) positions but many sequence cells).
    std::vector<float>   prompt_embd;
    std::vector<int32_t> prompt_mrope;
    int                  mrope_next = 0;

    SamplingParams sp;
    Sampler        sampler;
    int32_t        last_token = -1;

    // callbacks run on the worker thread
    std::function<void(const std::string & piece)>         on_token;
    std::function<void(const std::string & finish_reason)> on_done;
    std::function<bool()>                                  is_cancelled;

    bool prefilling() const { return active && n_past < n_prompt; }
    void reset();
};

struct Engine {
    std::unique_ptr<Model> model;
    ModelRunner runner;
    Vocab       vocab;

    std::vector<Slot> slots;
    int n_ctx = 0, max_batch = 0, n_ctx_pad = 0;

    void free_slot(Slot & s);

    void load(const ModelParams & mp, const ContextParams & cp);
    const Vocab & get_vocab() const { return vocab; }

    // admit a request to a free slot, or nullptr if all are busy
    Slot * admit(std::vector<int32_t> prompt, const SamplingParams & sp,
                 std::function<void(const std::string &)> on_token,
                 std::function<void(const std::string &)> on_done,
                 std::function<bool()> is_cancelled = nullptr);

    // admit a VLM image request: prefill from spliced embeddings + explicit M-RoPE (Qwen3.5)
    Slot * admit_embd(const std::vector<float> & embd, const std::vector<int32_t> & mrope,
                      int n_tokens, int mrope_next, const SamplingParams & sp,
                      std::function<void(const std::string &)> on_token,
                      std::function<void(const std::string &)> on_done,
                      std::function<bool()> is_cancelled = nullptr);

    bool has_work() const;
    void step();

private:
    void sample_emit(Slot & s, const float * logits);
};

} // namespace nano
