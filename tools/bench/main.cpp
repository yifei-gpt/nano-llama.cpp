// nano-bench — prompt-processing (pp) and token-generation (tg) throughput; -np>1 = batched throughput
#include "nanollama/nanollama.h"
#include "nanollama/engine/engine.h"
#include "ggml.h"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace nano;

static double secs_since(int64_t t0) { return (ggml_time_us() - t0) / 1e6; }

int main(int argc, char ** argv) {
    ModelParams mp; ContextParams cp;
    int n_pp = 512, n_tg = 128, n_rep = 3, n_parallel = 1;
    cp.n_threads = 16;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() -> const char * {
            if (i + 1 >= argc) { fprintf(stderr, "error: %s needs a value\n", a.c_str()); exit(1); }
            return argv[++i];
        };
        if      (a == "-m")    mp.path = next();
        else if (a == "-ngl")  mp.n_gpu_layers = atoi(next());
        else if (a == "-t")    cp.n_threads = atoi(next());
        else if (a == "-c")    cp.n_ctx = atoi(next());
        else if (a == "-pp")   n_pp = atoi(next());
        else if (a == "-tg")   n_tg = atoi(next());
        else if (a == "-np")   n_parallel = atoi(next());
        else if (a == "-r")    n_rep = atoi(next());
    }
    if (mp.path.empty()) { fprintf(stderr, "usage: %s -m MODEL.gguf [-ngl N] [-pp 512] [-tg 128] [-np 1]\n", argv[0]); return 1; }
    cp.n_ctx = std::max(cp.n_ctx, n_pp + n_tg + 8);

    ggml_backend_load_all();

    // ---- batched: n_parallel sequences sharing one continuous-batching engine ----
    if (n_parallel > 1) {
        cp.n_slots = n_parallel;
        cp.n_ctx   = n_pp + n_tg + 8;   // tight per-slot context (KV cost scales with this × n_slots)
        Engine eng; eng.load(mp, cp);
        const char * be = eng.runner.on_gpu ? "CUDA" : "CPU";
        std::vector<int32_t> prompt(n_pp, 100);
        SamplingParams sp; sp.temperature = 0.0f; sp.ignore_eos = true; sp.n_predict = n_tg;
        std::atomic<long> ntok{0};
        int64_t t0 = ggml_time_us();
        for (int i = 0; i < n_parallel; i++)
            eng.admit(prompt, sp, [&](const std::string &) { ntok++; }, nullptr);
        while (eng.has_work()) eng.step();
        double s = secs_since(t0);
        printf("| backend | ngl | test | t/s |\n|---|---|---|---|\n");
        printf("| %s | %d | np%d pp%d tg%d | %.1f |\n", be, mp.n_gpu_layers, n_parallel, n_pp, n_tg, s > 0 ? ntok / s : 0.0);
        return 0;
    }

    LLM llm; llm.load(mp, cp);
    std::vector<int32_t> prompt(n_pp, 100);   // arbitrary tokens

    printf("| backend | ngl | test | t/s |\n|---|---|---|---|\n");
    const char * be = llm.runner.on_gpu ? "CUDA" : "CPU";   // actual backend, not just the flag

    // ---- pp: process n_pp tokens in one forward ----
    for (int r = 0; r < n_rep; r++) {
        int64_t t0 = ggml_time_us();
        llm.runner.decode(prompt.data(), n_pp, 0);
        double s = secs_since(t0);
        if (r == n_rep - 1) printf("| %s | %d | pp%d | %.2f |\n", be, mp.n_gpu_layers, n_pp, s > 0 ? n_pp / s : 0.0);
    }

    // ---- tg: generate n_tg tokens one at a time ----
    for (int r = 0; r < n_rep; r++) {
        int32_t tok = 100; int n_past = 0;
        llm.runner.decode(&tok, 1, n_past++);   // warm one
        int64_t t0 = ggml_time_us();
        for (int i = 0; i < n_tg; i++) { llm.runner.decode(&tok, 1, n_past++); }
        double s = secs_since(t0);
        if (r == n_rep - 1) printf("| %s | %d | tg%d | %.2f |\n", be, mp.n_gpu_layers, n_tg, s > 0 ? n_tg / s : 0.0);
    }
    return 0;
}
