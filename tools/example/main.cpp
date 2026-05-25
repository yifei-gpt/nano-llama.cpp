// nano-cli — offline generation / chat
#include "nanollama/nanollama.h"
#include "nanollama/layers/sampler.h"
#include "nanollama/vision/clip.h"
#include "nanollama/vision/image.h"
#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>

using namespace nano;

// Qwen3-VL: encode the image, splice its embeddings into a ChatML prompt with M-RoPE positions,
// then generate. Text tokens advance position by 1; image tokens get (t=base, y=base+row, x=base+col)
// and the image consumes max(grid_w, grid_h) positions.
static void run_multimodal(LLM & llm, ClipModel & clip, const std::string & image_path,
                           const std::string & user_text, const SamplingParams & sp, bool think) {
    ClipImage img;
    if (!load_and_preprocess(image_path, clip.align(), clip.min_px(), clip.max_px(), clip.mean, clip.std, img)) {
        fprintf(stderr, "failed to load image %s\n", image_path.c_str()); return;
    }
    int n_img, gw, gh;
    std::vector<float> vemb = clip_encode(clip, img, n_img, gw, gh);
    fprintf(stderr, "image %dx%d -> %d vision tokens (%dx%d grid)\n", img.nx, img.ny, n_img, gw, gh);

    const Vocab & vocab = llm.get_vocab();
    const int     n_embd = llm.model->hparams.n_embd;
    const float * tbl    = llm.model->embd_f32.data();

    auto pretok = vocab.tokenize("<|im_start|>user\n<|vision_start|>", false, true);
    std::string suf = "<|vision_end|>" + user_text + "<|im_end|>\n<|im_start|>assistant\n";
    if (!think) suf += "<think>\n\n</think>\n\n";
    auto suftok = vocab.tokenize(suf, false, true);

    const int N = (int) pretok.size() + n_img + (int) suftok.size();
    std::vector<float>   E((size_t) N * n_embd);
    std::vector<int32_t> P((size_t) 4 * N);
    auto emit_text = [&](int32_t T, int j, int pos) {
        std::copy_n(tbl + (size_t) T * n_embd, n_embd, E.begin() + (size_t) j * n_embd);
        P[j] = P[N + j] = P[2 * N + j] = pos; P[3 * N + j] = 0;
    };
    int p = 0, j = 0;
    for (int32_t T : pretok) emit_text(T, j++, p++);
    const int base = p;
    for (int i = 0; i < n_img; i++) {
        std::copy_n(vemb.begin() + (size_t) i * n_embd, n_embd, E.begin() + (size_t) j * n_embd);
        P[j] = base; P[N + j] = base + i / gw; P[2 * N + j] = base + i % gw; P[3 * N + j] = 0; j++;
    }
    p = base + std::max(gw, gh);
    for (int32_t T : suftok) emit_text(T, j++, p++);

    Sampler sampler; sampler.init(sp);
    const int n_vocab = llm.runner.n_vocab();
    int32_t next = sampler.sample(llm.runner.decode_embd(E.data(), P.data(), N, 0), n_vocab);

    int n_past = N;
    for (int t = 0; t < sp.n_predict; t++) {
        if (!sp.ignore_eos && vocab.is_eog(next)) break;
        printf("%s", vocab.token_to_piece(next).c_str()); fflush(stdout);
        std::vector<float> e(tbl + (size_t) next * n_embd, tbl + (size_t) (next + 1) * n_embd);
        int32_t pos4[4] = { p, p, p, 0 };
        next = sampler.sample(llm.runner.decode_embd(e.data(), pos4, 1, n_past), n_vocab);
        n_past++; p++;
    }
    printf("\n");
}

static void usage(const char * argv0) {
    fprintf(stderr,
        "usage: %s -m MODEL.gguf [opts] [PROMPT]\n"
        "  -m PATH         model gguf (required)\n"
        "  -n N            max tokens to generate (default 128)\n"
        "  -ngl N          >0 = whole model on GPU, 0 = CPU (default 0)\n"
        "  -t N            CPU threads (default 16)\n"
        "  -c N            context size (default 4096)\n"
        "  --temp F        temperature (<=0 = greedy; default 0.8)\n"
        "  --top-k N       (default 40)   --top-p F (default 0.95)   --min-p F (default 0.05)\n"
        "  --seed N        RNG seed\n"
        "  --chat          wrap PROMPT in a ChatML user turn\n"
        "  --no-think      disable Qwen3 thinking (chat mode)\n"
        "  --no-flash      disable flash-attention path\n", argv0);
}

int main(int argc, char ** argv) {
    ModelParams mp; ContextParams cp; SamplingParams sp;
    int n_predict = 128; std::string prompt, mmproj, image_path; bool chat = false, think = true;
    cp.n_threads = 16;

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() -> const char * {
            if (i + 1 >= argc) { fprintf(stderr, "error: %s needs a value\n", a.c_str()); exit(1); }
            return argv[++i];
        };
        if      (a == "-m")          mp.path = next();
        else if (a == "-n")          n_predict = atoi(next());
        else if (a == "-ngl")        mp.n_gpu_layers = atoi(next());
        else if (a == "-t")          cp.n_threads = atoi(next());
        else if (a == "-c")          cp.n_ctx = atoi(next());
        else if (a == "--temp")      sp.temperature = atof(next());
        else if (a == "--top-k")     sp.top_k = atoi(next());
        else if (a == "--top-p")     sp.top_p = atof(next());
        else if (a == "--min-p")     sp.min_p = atof(next());
        else if (a == "--seed")      sp.seed = (uint32_t) strtoul(next(), nullptr, 10);
        else if (a == "--chat")      chat = true;
        else if (a == "--no-think")  think = false;
        else if (a == "--no-flash")  cp.flash_attn = false;
        else if (a == "--mmproj")    mmproj = next();
        else if (a == "--image")     image_path = next();
        else if (a == "-h" || a == "--help") { usage(argv[0]); return 0; }
        else                         prompt = a;
    }
    if (mp.path.empty()) { usage(argv[0]); return 1; }
    if (prompt.empty()) prompt = "Hello, my name is";
    sp.n_predict = n_predict;

    ggml_backend_load_all();   // register CPU/CUDA backends

    LLM llm;
    llm.load(mp, cp);

    if (!mmproj.empty() && !image_path.empty()) {   // multimodal: image + text -> generation
        ClipModel clip;
        clip_load(clip, mmproj, mp.n_gpu_layers);
        sp.n_predict = n_predict;
        run_multimodal(llm, clip, image_path, prompt, sp, think);
        return 0;
    }

    std::vector<int32_t> tokens;
    if (chat) {
        std::string templated = apply_chatml({{ "user", prompt }}, /*gen_prompt=*/true, think);
        tokens = llm.get_vocab().tokenize(templated, /*add_bos=*/false, /*parse_special=*/true);
    } else {
        tokens = llm.get_vocab().tokenize(prompt, /*add_bos=*/false, /*parse_special=*/false);
    }
    fprintf(stderr, "prompt: %zu tokens:", tokens.size());
    for (int32_t t : tokens) fprintf(stderr, " %d", t);
    fprintf(stderr, "\n");

    printf("%s", prompt.c_str());
    fflush(stdout);

    const int64_t t0 = ggml_time_us();
    int n_gen = 0;
    std::string text = llm.generate(tokens, sp, [&](const std::string & piece) {
        printf("%s", piece.c_str()); fflush(stdout); n_gen++;
    });
    const int64_t t1 = ggml_time_us();
    printf("\n");
    double secs = (t1 - t0) / 1e6;
    fprintf(stderr, "\n[%d tokens in %.2fs = %.2f tok/s]\n", n_gen, secs, secs > 0 ? n_gen / secs : 0.0);
    return 0;
}
