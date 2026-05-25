# AGENTS.md

Guidance for agents **extending** this repo. It's a C++/CUDA inference engine small enough to read
and modify end to end: `ggml/` (vendored, trimmed to CPU+CUDA) supplies the tensor kernels, and
`nanollama/` is the part you'll actually change — model graph, tokenizer, KV cache, sampler,
batching, server, ~1,900 lines of project code. (`nanollama/tokenizer/unicode.cpp` and
`unicode-data.cpp` are ~8k lines of Unicode tables ported from llama.cpp — ignore them.)

## Pipeline (one request)

`qwen3_load` (GGUF → weights) → `Vocab` tokenize → `ModelRunner` builds a ggml graph and runs it on
one backend (`decode` single-seq / `decode_batch` batched) → `Sampler`. `LLM` drives a single
sequence; `Engine::step()` batches many sequences; `nano-server` wraps `Engine`.

## Layout

```
nanollama/
  models/qwen3.{h,cpp}      weights + forward graph
  engine/
    model_runner.{h,cpp}    backend, graph build/run, CUDA-graph replay
    kv_cache.{h,cpp}        per-layer K/V buffers
    llm.{h,cpp}             single-sequence generate()
    engine.{h,cpp}          continuous-batching scheduler
  layers/                   attention, sampler, small op helpers
  tokenizer/                vocab.* (byte-level BPE) + chat.* (ChatML)
  utils/gguf.{h,cpp}        GGUF reader
tools/{example,bench,server}/   nano-example, nano-bench, nano-server
ggml/                       vendored, trimmed to the CPU + CUDA backends
```

## Map (read in this order)

1. `models/qwen3.cpp` — `qwen3_load` and `qwen3_build_graph` (the forward pass). The core.
2. `layers/` — `ops.h` (rms_norm / linear / rope_neox / swiglu), `attention.*`, `sampler.*`.
3. `engine/model_runner.*` — graph build/run, CUDA-graph reuse.
4. `engine/kv_cache.*`, `engine/llm.*`, `engine/engine.*` (batching scheduler).
5. `tokenizer/`, `utils/gguf.*`.
6. `tools/{example,bench,server}/main.cpp`.

## Dev loop

```bash
cmake -B build                          # CUDA on by default; CPU only: -DNANO_CUDA=OFF
cmake --build build -j --target nano-example   # rebuild one target while iterating
```
(`NANO_CUDA` is the toggle — it force-enables ggml's CUDA backend, so setting `GGML_CUDA` directly does nothing. The build picks the `nvcc` on `PATH`; to force a toolkit/arch add `-DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.4/bin/nvcc -DCMAKE_CUDA_ARCHITECTURES=86`.)

New `.cpp` under `nanollama/` is compiled automatically (CMake globs it with `CONFIGURE_DEPENDS`) — no build-file edit. A new tool, though, needs its name added to the `foreach(tool ...)` list in `CMakeLists.txt`.

Verify **every** change:
- **Correctness** — greedy (`--temp 0`) output and tokenizer ids must stay identical to a reference run on the same prompt; CPU and GPU should agree.
- **Speed** — `nano-bench -ngl 99` (pp/tg) and a concurrent run against `nano-server`; a regression usually means a broken invariant below.

## Extending

### Add a model architecture

Add `nanollama/models/<arch>.{h,cpp}` mirroring `qwen3.*`:

1. Define an `<arch>_hparams` struct and a model struct of weight `ggml_tensor*`s (like `qwen3_model`).
2. `<arch>_load(model, mp)`: read hparams via `gkey`/`arch_key` from GGUF; `dup()` each tensor by its GGUF name into a model context; allocate with `ggml_backend_alloc_ctx_tensors_from_buft`; stream the data from the file; build the host F32 embedding table (the GPU path can't `get_rows` most quant types).
3. `<arch>_build_graph(...)`: write the forward pass with `layers/ops.h` (`rms_norm`, `linear`, `rope_neox`, `swiglu`) + `build_attention`, gathering the output rows at the last layer (`out_ids`) so the final norm → lm-head run only on those. Match the reference op order exactly — norm placement, RoPE type/theta, attention scale, MLP shape — or logits will silently drift.
4. Wire it in: `LLM::load`/`Engine::load` call `qwen3_load`, and `model_runner.cpp`'s `build()` calls `qwen3_build_graph` — today both hardcode Qwen3. Add a dispatch on `general.architecture` (read into `model.arch`) to pick the right load/build pair — and relax `qwen3_load`'s guard, which currently aborts unless the arch is `qwen3`.
5. `Vocab` is generic GGUF byte-level BPE; reuse it, but check the pretokenizer regex in `vocab.cpp` if the arch tokenizes differently.

### Add an operator / layer

**Level 1 — compose (almost always do this).** Most "ops" are a few `ggml_*` calls. Add a small
inline helper to `layers/ops.h` (like `rms_norm`/`swiglu`), or a block function like `build_attention`
for something bigger. ggml runs each op on whichever backend its inputs already live on, so a composed
op works on CPU and CUDA for free, with **zero** changes to vendored ggml. Follow the existing tensor
layouts (e.g. `[head_dim, n_head, n_tokens]`) and reshape/permute the way `build_attention` does.

**Level 2 — a genuinely new ggml op** (a kernel you can't express as a composition). Heavy and rarely
needed; it touches vendored ggml in several places. To add `GGML_OP_FOO`:

1. **`ggml/include/ggml.h`** — add `GGML_OP_FOO` to `enum ggml_op` (~line 479) and declare the public
   constructor `GGML_API struct ggml_tensor * ggml_foo(struct ggml_context *, struct ggml_tensor * a, …);`.
2. **`ggml/src/ggml.c`** — implement `ggml_foo()`: allocate the result tensor (its shape), set
   `result->op = GGML_OP_FOO` and `result->src[0..] = …`. Add the op's string to the `GGML_OP_NAME`
   and `GGML_OP_SYMBOL` tables (their length is `static_assert`-ed against `GGML_OP_COUNT`, so a
   missing entry won't compile — a useful guard).
3. **CPU backend** — implement `ggml_compute_forward_foo(...)` in `ggml/src/ggml-cpu/ops.cpp` (declare
   in `ops.h`), add a `case GGML_OP_FOO:` to the `ggml_compute_forward` dispatch switch in
   `ggml/src/ggml-cpu/ggml-cpu.c` (~line 1702), and give it a thread count in `ggml_get_n_tasks`
   (same file, ~line 2191).
4. **CUDA backend** — add `ggml/src/ggml-cuda/foo.cu` (+ `foo.cuh`) with the kernel and
   `ggml_cuda_op_foo(ctx, dst)`; add a `case GGML_OP_FOO:` to `ggml_cuda_compute_forward` in
   `ggml/src/ggml-cuda/ggml-cuda.cu` (~line 2780); **and** add the same case to
   `ggml_backend_cuda_device_supports_op` (~line 5034). Skipping that last one is the classic mistake —
   the op then silently isn't "supported" and you get a fallback/assert at runtime, not a compile error.

Validate any new op on a tiny input against a reference (compute on CPU, compare to a hand result or
PyTorch; CPU and CUDA outputs must match).

### Work on the engine

Two execution paths share one graph builder — find the layer your change belongs to.

- **`model_runner.cpp` — the ggml layer.** `build()` (static) creates the input tensors
  (`embd`, `pos`, `kv_idxs`, `mask`, `out_ids`) and the graph via `qwen3_build_graph`. Two entry points
  run it: `decode()` for one sequence (prompt prefill *or* a single token), and `decode_batch(b, s0,
  n_stream, n_q, n_kv)` — a **per-stream** batched forward of `n_stream` sequences (slots `s0..s0+n_stream-1`)
  with `n_q` tokens each. Both fill the inputs, call `ggml_backend_graph_compute`, then copy out logits.
  The speed trick: same-shape single-token decode (`dec_mem`) and batched (`bat_mem`) steps rebuild into
  fixed buffers with `n_kv` padded to 256, so ggml replays the captured CUDA graph — keep that topology
  stable. (Multi-token prompt prefill uses a transient buffer and isn't replayed.) The host embedding
  gather is `fill_embd()` (copies F32 rows into `embd`).
- **`kv_cache.*` — storage only.** Per-layer F16 K/V tensors of shape `[n_embd_kv, n_slots*n_ctx_pad + 1]`,
  allocated once on the backend. Slot `s` owns the *contiguous* cell range `[s*n_ctx_pad, (s+1)*n_ctx_pad)`
  (`n_ctx_pad` = `n_ctx` padded to 256), so the cache views as a 4D per-stream tensor `[head_dim,
  n_head_kv, n_kv, n_stream]` — exactly llama.cpp's `get_k` strides — and each query attends only its own
  slot's cells. The `+1` is a scratch cell: a write sink for gap-filler dummy tokens. *Which* cell a
  token's K/V lands in is the scatter index `kv_dst[i]` (`= slot*n_ctx_pad + pos`) in `ModelRunner::Batch`.
- **`engine.cpp` — the scheduler.** A `Slot` is one in-flight request (prompt, generated tokens, `n_past`,
  `generating`, per-slot `Sampler`, the `on_token`/`on_done`/`is_cancelled` callbacks). `step()` runs two
  passes: **decode** packs every generating slot into one per-stream `decode_batch` (streams `[0, hi)`,
  one token each; idle gaps get a masked dummy parked in the scratch cell); **prefill** then advances each
  waiting slot by a chunk as its own single-stream forward, sampling the first token when a prompt
  completes. Samples once per `logit_rows` entry; streams via `on_token`; frees finished/cancelled slots.
  - Change **scheduling** (priorities, fairness, prefill chunk size) → the two passes in `step()`.
  - Change **request lifecycle** (stop strings, cancellation, usage counts) → `Slot` + `sample_emit`.
  - Attention isolation is now **structural** (each stream views only its own slot), so the mask is plain
    causal (`j <= pos`). A KV-layout change must keep slots contiguous so the per-stream view stays valid.

## Invariants (don't break these)

- **Single backend.** The whole graph runs on one backend; this is what lets ggml capture & replay a CUDA graph. Don't reintroduce a multi-backend scheduler without measuring tg speed.
- **Host embedding lookup.** CUDA's `get_rows` can't read Q6_K, so the embedding is gathered on the host into an input tensor. Keep new graphs free of unsupported ops on the GPU path.
- **Stable graph topology.** Graphs are rebuilt into fixed buffers (`dec_mem`/`bat_mem`) so same-shape steps replay the captured CUDA graph. The shape is `n_kv` (padded to 256), plus — for batched decode — `n_stream` (active slots) and `n_q`: a steady set of running slots replays, while admitting/freeing re-captures. Changing shapes every step forfeits that speed.
