// engine.cpp — continuous-batching scheduler over per-stream KV slots
#include "nanollama/engine/engine.h"
#include "nanollama/common.h"

#include <algorithm>
#include <vector>

namespace nano {

void Slot::reset() {
    active = false; generating = false; prompt.clear(); generated.clear();
    n_past = 0; n_prompt = 0; mrope_next = 0; last_token = -1;
    prompt_embd.clear(); prompt_mrope.clear();
    on_token = nullptr; on_done = nullptr; is_cancelled = nullptr;
}

void Engine::load(const ModelParams & mp, const ContextParams & cp) {
    model.reset(load_model(mp));
    if (!model) NANO_ABORT("model load failed");
    vocab.load(GgufFile(mp.path));
    runner.init(*model, cp);
    pool = std::make_unique<ThreadPool>(cp.n_threads);
    n_ctx = cp.n_ctx;
    max_batch = cp.n_ctx;
    n_ctx_pad = runner.n_ctx_pad();
    slots.resize(cp.n_slots);
    for (int i = 0; i < cp.n_slots; i++) slots[i].id = i;
    NANO_LOG("engine: %d slots, n_ctx=%d", cp.n_slots, n_ctx);
}

void Engine::free_slot(Slot & s) { s.reset(); }

Slot * Engine::admit(std::vector<int32_t> prompt, const SamplingParams & sp,
                     std::function<void(const std::string &)> on_token,
                     std::function<void(const std::string &)> on_done,
                     std::function<bool()> is_cancelled) {
    NANO_ASSERT(!prompt.empty());
    for (auto & s : slots) {
        if (s.active) continue;
        s.reset();
        s.active = true;
        s.prompt = std::move(prompt);
        if ((int) s.prompt.size() > n_ctx) s.prompt.resize(n_ctx);
        s.n_prompt = (int) s.prompt.size();
        s.sp = sp; s.on_token = std::move(on_token); s.on_done = std::move(on_done);
        s.is_cancelled = std::move(is_cancelled);
        s.sampler.init(sp);
        return &s;
    }
    return nullptr;
}

Slot * Engine::admit_embd(const std::vector<float> & embd, const std::vector<int32_t> & mrope,
                          int n_tokens, int mrope_next, const SamplingParams & sp,
                          std::function<void(const std::string &)> on_token,
                          std::function<void(const std::string &)> on_done,
                          std::function<bool()> is_cancelled) {
    NANO_ASSERT(n_tokens > 0 && n_tokens <= n_ctx);
    for (auto & s : slots) {
        if (s.active) continue;
        s.reset();
        s.active = true;
        s.prompt_embd = embd; s.prompt_mrope = mrope;
        s.n_prompt = n_tokens; s.mrope_next = mrope_next;
        s.sp = sp; s.on_token = std::move(on_token); s.on_done = std::move(on_done);
        s.is_cancelled = std::move(is_cancelled);
        s.sampler.init(sp);
        return &s;
    }
    return nullptr;
}

bool Engine::has_work() const {
    for (const auto & s : slots) if (s.active) return true;
    return false;
}

// emit an already-sampled token: append it, stream it, and finish on EOS / length
void Engine::emit_token(Slot & s, int32_t tok) {
    const bool eos   = !s.sp.ignore_eos && vocab.is_eog(tok);
    const bool limit = (int) s.generated.size() >= s.sp.n_predict;
    if (eos || limit) { if (s.on_done) s.on_done(limit ? "length" : "stop"); free_slot(s); return; }
    s.generated.push_back(tok);
    s.last_token = tok;
    if (s.on_token) s.on_token(vocab.token_to_piece(tok));
}

void Engine::sample_emit(Slot & s, const float * logits) {
    emit_token(s, s.sampler.sample(logits, runner.n_vocab()));
}

void Engine::step() {
    // drop disconnected clients before doing any work for them
    for (auto & s : slots) {
        if (s.active && s.is_cancelled && s.is_cancelled()) {
            if (s.on_done) s.on_done("cancel");
            free_slot(s);
        }
    }

    // DECODE: one token per generating slot, packed into streams [0, hi); gaps get a masked dummy
    int hi = 0;
    for (auto & s : slots) if (s.active && s.generating) hi = s.id + 1;
    if (hi > 0) {
        const int npp = model->n_pos_per_token();
        ModelRunner::Batch db;
        db.token.assign(hi, 0); db.pos.assign(hi, 0); db.kv_dst.assign(hi, runner.scratch_cell());
        if (npp == 4) db.mrope.assign((size_t) 4 * hi, 0);
        std::vector<Slot *> out;
        int n_kv = 1;
        for (int i = 0; i < hi; i++) {
            Slot & s = slots[i];
            if (!(s.active && s.generating)) continue;
            db.token[i]  = s.last_token;
            db.pos[i]    = s.n_past;
            db.kv_dst[i] = s.id * n_ctx_pad + s.n_past;
            if (npp == 4) { db.mrope[i] = db.mrope[hi + i] = db.mrope[2 * hi + i] = s.mrope_next; }
            db.logit_rows.push_back(i);
            out.push_back(&s);
            n_kv = std::max(n_kv, s.n_past + 1);
        }
        const float * logits = runner.decode_batch(db, /*s0=*/0, /*n_stream=*/hi, /*n_q=*/1, n_kv);
        const int n_vocab = runner.n_vocab();
        // sample slots in parallel (independent), then emit serially (touches shared state)
        const int n = (int) out.size();
        std::vector<int32_t> toks(n);
        pool->parallel_for(n, [&](int r) { toks[r] = out[r]->sampler.sample(logits + (size_t) r * n_vocab, n_vocab); });
        for (int r = 0; r < n; r++) { out[r]->n_past += 1; out[r]->mrope_next += 1; emit_token(*out[r], toks[r]); }
    }

    // PREFILL: per waiting slot — recurrent (qwen3.5) prefills the whole prompt at once and resets
    // its state first; dense (qwen3) chunks within a token budget. Samples the first token on completion.
    const bool recurrent = runner.is_recurrent();
    int budget = max_batch;
    for (auto & s : slots) {
        if (!s.active || !s.prefilling() || budget <= 0) continue;
        if (recurrent && s.n_past == 0) runner.reset_recurrent_slot(s.id);
        const int remaining = s.n_prompt - s.n_past;
        const int n = recurrent ? remaining : std::min(remaining, budget);
        const bool image = !s.prompt_embd.empty();
        ModelRunner::Batch pb;
        if (image) { pb.embd = s.prompt_embd; pb.mrope = s.prompt_mrope; }
        for (int i = 0; i < n; i++) {
            if (!image) pb.token.push_back(s.prompt[s.n_past + i]);
            pb.pos.push_back(s.n_past + i);
            pb.kv_dst.push_back(s.id * n_ctx_pad + s.n_past + i);
        }
        pb.logit_rows.push_back(n - 1);
        const float * logits = runner.decode_batch(pb, /*s0=*/s.id, /*n_stream=*/1, /*n_q=*/n, s.n_past + n);
        s.n_past += n;
        budget   -= n;
        if (!s.prefilling()) {
            s.generating = true;
            if (!image) s.mrope_next = s.n_past;
            sample_emit(s, logits);
        }
    }
}

} // namespace nano
