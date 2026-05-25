# nano-llama.cpp

A small, readable LLM inference engine in C++/CUDA — GGUF, on CPU or a single GPU.

The kernels come from a trimmed copy of [ggml](https://github.com/ggml-org/ggml); the model,
tokenizer, KV cache, sampler, and server are written here in **~1,900 lines** of C++.

## Key Features

- 🚀 **Fast** — on a single GPU, the same tokens/sec as the reference ggml runtime.
- 🛠️ **Developer-friendly** — a small, readable codebase (~1,900 lines of C++); add your own models and operators following the `AGENTS.md` guide.
- 🔀 **Continuous-batching server** — OpenAI-compatible `/v1/chat/completions` + `/completion` with streaming; multiple in-flight requests batched into one forward pass.
- 🖥️ **Backend support** — the same code runs on CPU or CUDA; the GPU path uses flash-attention and CUDA-graph replay.
- 🪶 **No heavy dependencies** — a vendored ggml plus two small libraries (HTTP + JSON) for the server.

## Build

Requires CMake ≥ 3.18 and a C++17 compiler. For the GPU build, the CUDA toolkit.

```bash
# GPU (CUDA is on by default)
cmake -B build
cmake --build build -j

# CPU only
cmake -B build -DNANO_CUDA=OFF
cmake --build build -j
```

Binaries land in `build/`: `nano-example`, `nano-bench`, `nano-server`.

> The build uses the `nvcc` on your `PATH`. To force a specific toolkit (or target arch), pass
> `-DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.4/bin/nvcc -DCMAKE_CUDA_ARCHITECTURES=86`.

## Get a model

```bash
mkdir -p models
curl -L https://huggingface.co/Qwen/Qwen3-4B-GGUF/resolve/main/Qwen3-4B-Q4_K_M.gguf \
     -o models/Qwen3-4B-Q4_K_M.gguf
```

## Quick Start

### CLI

```bash
# GPU, greedy
./build/nano-example -m models/Qwen3-4B-Q4_K_M.gguf -ngl 99 --temp 0 -n 128 "The capital of France is"

# chat turn (ChatML), reasoning disabled
./build/nano-example -m models/Qwen3-4B-Q4_K_M.gguf -ngl 99 --chat --no-think "Explain RoPE in one sentence."

# CPU
./build/nano-example -m models/Qwen3-4B-Q4_K_M.gguf -ngl 0 -t 16 "Once upon a time,"
```

Flags: `-ngl` (>0 = whole model on GPU, 0 = CPU), `-t` threads, `-c` context, `-n` max tokens,
`--temp/--top-k/--top-p/--min-p/--seed`, `--chat`, `--no-think`.

### Benchmark

```bash
./build/nano-bench -m models/Qwen3-4B-Q4_K_M.gguf -ngl 99 -pp 512 -tg 128
```

### Server

```bash
./build/nano-server -m models/Qwen3-4B-Q4_K_M.gguf -ngl 99 -np 4 --port 8080
```

```bash
curl http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "messages": [{"role": "user", "content": "Name three primary colors."}],
  "max_tokens": 64
}'
```

Endpoints: `GET /health`, `GET /v1/models`, `POST /completion`, `POST /v1/chat/completions`
(supports `stream: true` and `chat_template_kwargs.enable_thinking`). `-np` sets the number of
concurrent slots; pass `--reasoning off` to disable Qwen3 thinking by default.

## Performance

Qwen3-4B-Q4_K_M. **GPU:** RTX 3090. **CPU:** AMD Ryzen Threadripper PRO 3955WX (16 cores, 16 threads).
tokens/sec, with peak memory (GPU = VRAM, CPU = process RSS):

| | prompt (pp512) | generate (tg128) | peak memory |
|---|---:|---:|---:|
| nano-llama.cpp (GPU) | 7005 | 187 | 3.3 GiB VRAM |
| llama.cpp (GPU)      | 6949 | 190 | 3.0 GiB VRAM |
| nano-llama.cpp (CPU) |  127 | 12   | 4.5 GiB RSS |
| llama.cpp (CPU)      |  134 | 12   | 4.2 GiB RSS |

Each sequence is its own attention stream (as in llama.cpp), so throughput rises with concurrency,
peaking at **~3,150 tok/s** aggregate generation on one 3090.

## License

MIT (see [LICENSE](LICENSE)). Vendored third-party code keeps its own license, all MIT:

- **ggml** — © The ggml authors ([ggml/LICENSE](ggml/LICENSE))
- **cpp-httplib** — © Yuji Hirose ([vendor/cpp-httplib/LICENSE](vendor/cpp-httplib/LICENSE))
- **nlohmann/json** — © Niels Lohmann ([vendor/nlohmann/LICENSE.MIT](vendor/nlohmann/LICENSE.MIT))
