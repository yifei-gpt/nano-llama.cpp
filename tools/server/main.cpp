// nano-server — OpenAI-compatible HTTP server with continuous batching
#include "nanollama/engine/engine.h"
#include "nanollama/tokenizer/chat.h"
#include "ggml.h"

#include "httplib.h"
#include "nlohmann/json.hpp"

#include <atomic>
#include <condition_variable>
#include <cstdio>
#include <ctime>
#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <queue>
#include <string>
#include <thread>

using namespace nano;
using json = nlohmann::json;

template <class T> struct Chan {   // tiny blocking queue
    std::queue<T> q; std::mutex m; std::condition_variable cv;
    void push(T v) { { std::lock_guard<std::mutex> l(m); q.push(std::move(v)); } cv.notify_one(); }
    T pop() { std::unique_lock<std::mutex> l(m); cv.wait(l, [&]{ return !q.empty(); }); T v = std::move(q.front()); q.pop(); return v; }
    bool pop_for(T & out, int ms) {   // false on timeout
        std::unique_lock<std::mutex> l(m);
        if (!cv.wait_for(l, std::chrono::milliseconds(ms), [&]{ return !q.empty(); })) return false;
        out = std::move(q.front()); q.pop(); return true;
    }
};

// bridge between an HTTP handler thread and the worker thread
struct Req {
    std::vector<int32_t> prompt;
    SamplingParams       sp;
    int                  prompt_len = 0;
    Chan<std::optional<std::string>> deltas;   // token pieces; nullopt = finished
    std::string finish_reason;
    std::atomic<bool> cancelled{false};        // set when the client disconnects
};

// globals (single served model)
static Engine g_engine;
static std::string g_model_name = "nano-llama";
static bool        g_enable_thinking_default = true;

static Chan<std::shared_ptr<Req>> g_submit;
static std::atomic<bool> g_running{true};

// worker thread: the only thread that touches the engine
static void worker_loop() {
    std::deque<std::shared_ptr<Req>> pending;
    while (g_running) {
        while (true) {
            std::shared_ptr<Req> r;
            { std::unique_lock<std::mutex> l(g_submit.m);
              if (g_submit.q.empty()) break;
              r = std::move(g_submit.q.front()); g_submit.q.pop(); }
            pending.push_back(r);
        }
        while (!pending.empty()) {
            auto r = pending.front();
            Slot * s = g_engine.admit(r->prompt, r->sp,
                [r](const std::string & piece) { r->deltas.push(piece); },
                [r](const std::string & fr)    { r->finish_reason = fr; r->deltas.push(std::nullopt); },
                [r]()                          { return r->cancelled.load(); });
            if (!s) break;            // all slots busy; retry next loop
            pending.pop_front();
        }
        if (g_engine.has_work()) {
            g_engine.step();
        } else if (pending.empty()) {
            std::unique_lock<std::mutex> l(g_submit.m);
            g_submit.cv.wait_for(l, std::chrono::milliseconds(50), [&]{ return !g_submit.q.empty(); });
        }
    }
}

static SamplingParams parse_sampling(const json & j) {
    SamplingParams sp;
    if (j.contains("temperature")) sp.temperature = j["temperature"].get<float>();
    if (j.contains("top_k"))       sp.top_k       = j["top_k"].get<int>();
    if (j.contains("top_p"))       sp.top_p       = j["top_p"].get<float>();
    if (j.contains("min_p"))       sp.min_p       = j["min_p"].get<float>();
    if (j.contains("seed"))        sp.seed        = j["seed"].get<uint32_t>();
    if (j.contains("max_tokens"))  sp.n_predict   = j["max_tokens"].get<int>();
    if (j.contains("n_predict"))   sp.n_predict   = j["n_predict"].get<int>();
    if (j.contains("ignore_eos"))  sp.ignore_eos  = j["ignore_eos"].get<bool>();
    return sp;
}

static std::shared_ptr<Req> submit(std::vector<int32_t> prompt, const SamplingParams & sp) {
    auto r = std::make_shared<Req>();
    r->prompt = std::move(prompt);
    const int n = (int) r->prompt.size();
    r->prompt_len = n < g_engine.n_ctx ? n : g_engine.n_ctx;   // admit() truncates to n_ctx
    r->sp = sp;
    g_submit.push(r);
    return r;
}

// drain deltas into the final string; cancel the slot if the client disconnects
static std::string collect(const std::shared_ptr<Req> & r, int & n_completion,
                           const std::function<bool()> & client_gone) {
    std::string text;
    n_completion = 0;
    while (true) {
        std::optional<std::string> d;
        if (r->deltas.pop_for(d, 100)) {
            if (!d) break;
            text += *d; n_completion++;
        }
        if (!r->cancelled && client_gone()) r->cancelled = true;
    }
    return text;
}

// longest prefix of s ending on a UTF-8 boundary (holds back a split codepoint)
static size_t complete_utf8_len(const std::string & s) {
    size_t n = s.size(), i = n;
    while (i > 0 && ((unsigned char) s[i - 1] & 0xC0) == 0x80) i--;   // back over continuation bytes
    if (i == 0) return n;                                            // no lead byte → emit as-is
    unsigned char lead = (unsigned char) s[i - 1];
    size_t need = lead < 0x80 ? 1 : lead < 0xE0 ? 2 : lead < 0xF0 ? 3 : 4;
    return (n - (i - 1) >= need) ? n : i - 1;   // last sequence complete? emit all : hold it back
}

int main(int argc, char ** argv) {
    ModelParams mp; ContextParams cp; std::string host = "127.0.0.1"; int port = 8080;
    cp.n_threads = 16; cp.n_slots = 4; cp.n_ctx = 4096;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto nx = [&]() -> const char * {
            if (i + 1 >= argc) { fprintf(stderr, "error: %s needs a value\n", a.c_str()); exit(1); }
            return argv[++i];
        };
        if      (a == "-m")            mp.path = nx();
        else if (a == "-ngl")          mp.n_gpu_layers = atoi(nx());
        else if (a == "-t")            cp.n_threads = atoi(nx());
        else if (a == "-c")            cp.n_ctx = atoi(nx());
        else if (a == "-np")           cp.n_slots = atoi(nx());
        else if (a == "--host")        host = nx();
        else if (a == "--port")        port = atoi(nx());
        else if (a == "--reasoning")   g_enable_thinking_default = (std::string(nx()) != "off");
    }
    if (mp.path.empty()) { fprintf(stderr, "usage: %s -m MODEL.gguf [-ngl N -np SLOTS --port P]\n", argv[0]); return 1; }

    ggml_backend_load_all();
    g_engine.load(mp, cp);
    std::thread worker(worker_loop);

    httplib::Server svr;
    // uncaught handler exception → 400
    svr.set_exception_handler([](const httplib::Request &, httplib::Response & res, std::exception_ptr) {
        res.status = 400;
        res.set_content(R"({"error":"invalid request"})", "application/json");
    });

    svr.Get("/health", [](const httplib::Request &, httplib::Response & res) {
        res.set_content("{\"status\":\"ok\"}", "application/json");
    });
    svr.Get("/v1/models", [](const httplib::Request &, httplib::Response & res) {
        json j = {{"object","list"},{"data",{{{"id",g_model_name},{"object","model"},{"owned_by","nano-llama.cpp"}}}}};
        res.set_content(j.dump(), "application/json");
    });

    // tokenize -> submit -> respond (stream or whole)
    auto run = [&](const httplib::Request & req, std::vector<int32_t> prompt, const SamplingParams & sp,
                   bool stream, bool chat, httplib::Response & res) {
        if (prompt.empty()) {   // empty prompt would crash the worker
            res.status = 400;
            res.set_content(R"({"error":"empty prompt"})", "application/json");
            return;
        }
        auto r = submit(std::move(prompt), sp);
        static std::atomic<uint64_t> id_counter{0};
        const std::string id = "chatcmpl-" + std::to_string(id_counter++);
        const int64_t created = (int64_t) time(nullptr);

        if (!stream) {
            int n_completion = 0;
            std::string text = collect(r, n_completion, [&]{ return req.is_connection_closed(); });
            json msg, choice;
            if (chat) {
                choice = {{"index",0},{"message",{{"role","assistant"},{"content",text}}},
                          {"finish_reason", r->finish_reason}};
                msg = {{"id",id},{"object","chat.completion"},{"created",created},{"model",g_model_name},
                       {"choices",{choice}},
                       {"usage",{{"prompt_tokens",r->prompt_len},{"completion_tokens",n_completion},
                                 {"total_tokens",r->prompt_len + n_completion}}}};
            } else {
                msg = {{"content",text},{"stop_type",r->finish_reason},
                       {"tokens_evaluated",r->prompt_len},{"tokens_predicted",n_completion}};
            }
            res.set_content(msg.dump(), "application/json");
            return;
        }

        // streaming (SSE); pending buffers a UTF-8 char split across tokens
        auto pending = std::make_shared<std::string>();
        auto content_chunk = [id, created, chat](const std::string & c) {
            json j = chat
                ? json{{"id",id},{"object","chat.completion.chunk"},{"created",created},{"model",g_model_name},
                       {"choices",{{{"index",0},{"delta",{{"content",c}}},{"finish_reason",nullptr}}}}}
                : json{{"content",c},{"stop",false}};
            return "data: " + j.dump() + "\n\n";
        };
        res.set_chunked_content_provider("text/event-stream",
            [r, id, created, chat, pending, content_chunk](size_t, httplib::DataSink & sink) -> bool {
                auto d = r->deltas.pop();
                std::string out;
                if (d) {
                    *pending += *d;
                    size_t k = complete_utf8_len(*pending);   // emit only whole codepoints
                    if (k == 0) {                             // nothing complete yet — keep buffering
                        if (!sink.is_writable()) { r->cancelled = true; return false; }
                        return true;
                    }
                    out = content_chunk(pending->substr(0, k));
                    pending->erase(0, k);
                } else {
                    if (!pending->empty()) out = content_chunk(*pending);   // flush trailing bytes
                    json fin = chat
                        ? json{{"id",id},{"object","chat.completion.chunk"},{"created",created},{"model",g_model_name},
                               {"choices",{{{"index",0},{"delta",json::object()},{"finish_reason",r->finish_reason}}}}}
                        : json{{"content",""},{"stop",true},{"stop_type",r->finish_reason}};
                    out += "data: " + fin.dump() + "\n\ndata: [DONE]\n\n";
                }
                // probe the socket to catch a vanished client
                if (d && !sink.is_writable()) { r->cancelled = true; return false; }
                bool ok = sink.write(out.data(), out.size());
                if (!d) { sink.done(); return false; }   // stream finished normally
                if (!ok) r->cancelled = true;            // write failed → client gone
                return ok;
            });
    };

    svr.Post("/completion", [&](const httplib::Request & req, httplib::Response & res) {
        json j = json::parse(req.body, nullptr, false);
        if (j.is_discarded()) { res.status = 400; return; }
        auto prompt = g_engine.get_vocab().tokenize(j.value("prompt", std::string()), false, false);
        run(req, std::move(prompt), parse_sampling(j), j.value("stream", false), /*chat=*/false, res);
    });

    svr.Post("/v1/chat/completions", [&](const httplib::Request & req, httplib::Response & res) {
        json j = json::parse(req.body, nullptr, false);
        if (j.is_discarded()) { res.status = 400; return; }
        if (!j.contains("messages") || !j["messages"].is_array()) {
            res.status = 400;
            res.set_content(R"({"error":"'messages' must be an array"})", "application/json");
            return;
        }
        std::vector<ChatMessage> msgs;
        for (auto & m : j["messages"]) msgs.push_back({ m.value("role","user"), m.value("content","") });
        bool think = g_enable_thinking_default;
        if (j.contains("enable_thinking")) think = j["enable_thinking"].get<bool>();
        if (j.contains("chat_template_kwargs") && j["chat_template_kwargs"].contains("enable_thinking"))
            think = j["chat_template_kwargs"]["enable_thinking"].get<bool>();
        std::string text = apply_chatml(msgs, /*add_generation_prompt=*/true, think);
        auto prompt = g_engine.get_vocab().tokenize(text, /*add_bos=*/false, /*parse_special=*/true);
        run(req, std::move(prompt), parse_sampling(j), j.value("stream", false), /*chat=*/true, res);
    });

    fprintf(stderr, "nano-server listening on http://%s:%d\n", host.c_str(), port);
    svr.listen(host.c_str(), port);
    g_running = false; g_submit.cv.notify_all(); worker.join();
    return 0;
}
