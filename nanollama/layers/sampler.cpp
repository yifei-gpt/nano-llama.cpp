// sampler.cpp — greedy / temp / top-k / top-p / min-p sampling
#include "nanollama/layers/sampler.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace nano {

void Sampler::init(const SamplingParams & p) {
    sp = p;
    rng.seed(p.seed == 0xFFFFFFFF ? std::random_device{}() : p.seed);
}

int32_t Sampler::sample(const float * logits, int n_vocab) {
    if (sp.temperature <= 0.0f) {
        int best = 0;
        for (int i = 1; i < n_vocab; i++) if (logits[i] > logits[best]) best = i;
        return best;
    }

    struct Cand { int32_t id; float v; };   // v holds the logit, then the probability
    std::vector<Cand> c(n_vocab);
    for (int i = 0; i < n_vocab; i++) c[i] = { i, logits[i] };

    int k = (sp.top_k > 0 && sp.top_k < n_vocab) ? sp.top_k : n_vocab;
    std::partial_sort(c.begin(), c.begin() + k, c.end(),
                      [](const Cand & a, const Cand & b) { return a.v > b.v; });
    c.resize(k);
    const int32_t top_id = c[0].id;   // argmax — fallback if the filters below empty the set

    float maxl = c[0].v;
    double sum = 0;
    for (auto & x : c) { x.v = std::exp((x.v - maxl) / sp.temperature); sum += x.v; }
    for (auto & x : c) x.v /= (float) sum;

    if (sp.top_p < 1.0f) {
        double cum = 0; size_t n = 0;
        for (; n < c.size(); ) { cum += c[n++].v; if (cum >= sp.top_p) break; }
        c.resize(n);
    }
    if (sp.min_p > 0.0f) {
        float thresh = sp.min_p * c[0].v;
        c.erase(std::remove_if(c.begin(), c.end(),
                [thresh](const Cand & x) { return x.v < thresh; }), c.end());
    }

    if (c.empty()) return top_id;   // every candidate filtered out (e.g. min_p > 1)

    std::vector<double> w(c.size());
    for (size_t i = 0; i < c.size(); i++) w[i] = c[i].v;
    std::discrete_distribution<size_t> dist(w.begin(), w.end());
    return c[dist(rng)].id;
}

} // namespace nano
