# AGENTS.md

Guidance for agents **extending** this repo. It's a C++/CUDA inference engine small enough to read
and modify end to end: `ggml/` (vendored, trimmed to CPU+CUDA) supplies the tensor kernels, and
`nanollama/` is the part you'll actually change — model graphs, tokenizer, KV/recurrent cache,
sampler, continuous batching, vision encoder, server.

It runs two architectures from one codebase: **Qwen3** (dense transformer) and **Qwen3.5** (a hybrid
that interleaves gated-DeltaNet recurrent/linear-attention layers with periodic full-attention, plus
a ViT vision encoder for image input).

### Line count (project code; run `wc -l`)

| area | lines | what |
|---|---|---|
| `nanollama/models/` | 738 | `model.*` (base + arch dispatch), `qwen3.*`, `qwen35.*` |
| `nanollama/engine/` | 836 | `model_runner`, `kv_cache`, `recurrent_cache`, `thread_pool`, `llm`, `engine` |
| `nanollama/vision/` | 488 | `image` (decode/resize), `clip` (ViT), `vlm` (splice + generate) |
| `nanollama/layers/` | 194 | `attention`, `ops.h`, `sampler` |
| `nanollama/tokenizer/` | 222 | `vocab` (byte-level BPE) + `chat` (ChatML) |
| `nanollama/utils/` | 141 | `gguf` reader |
| `nanollama/` top `.h` | 93 | `config`, `common`, umbrella header |
| **nanollama total** | **~2,700** | the engine you edit |
| `tools/` | ~520 | `nano-example`, `nano-bench`, `nano-server` |

Excluded: `nanollama/tokenizer/unicode*.{cpp,h}` (~8.6k lines of Unicode tables ported from
llama.cpp — ignore them) and `ggml/` (~155k lines, vendored).

## Pipeline (one request)

`load_model` (GGUF → weights, **dispatching on `general.architecture`**) → `Vocab` tokenize →
`ModelRunner` builds a ggml graph and runs it on one backend → `Sampler`. `LLM` drives a single
sequence; `Engine::step()` batches many; `nano-server` wraps `Engine`. For an image, the ViT
(`clip`) encodes it and `vlm` splices the patch embeddings into the prompt with M-RoPE positions.

## Layout

```
nanollama/
  models/
    model.{h,cpp}           Model base (arch traits) + load_model() arch dispatch
    qwen3.{h,cpp}           dense transformer: weights + forward graph
    qwen35.{h,cpp}          hybrid: gated-DeltaNet + periodic full-attention + M-RoPE
  engine/
    model_runner.{h,cpp}    backend, graph build/run, CUDA-graph replay
    kv_cache.{h,cpp}        per-attention-layer K/V buffers
    recurrent_cache.{h,cpp} per-slot conv + delta state for recurrent layers
    thread_pool.{h,cpp}     fixed worker pool (parallel per-slot sampling)
    llm.{h,cpp}             single-sequence generate()
    engine.{h,cpp}          continuous-batching scheduler
  vision/                   image preprocess, ViT (mmproj) encoder, VLM splicing
  layers/                   attention, sampler, small op helpers (ops.h)
  tokenizer/                vocab.* (byte-level BPE) + chat.* (ChatML)
  utils/gguf.{h,cpp}        GGUF reader
tools/{example,bench,server}/   nano-example, nano-bench, nano-server
ggml/                       vendored, trimmed to the CPU + CUDA backends
```

## Map (read in this order)

1. `models/model.{h,cpp}` — the `Model` base (arch traits: `n_pos_per_token`, `is_recurrent_layer`,
   `build_graph`) and `load_model()`, which peeks `general.architecture` and builds the right model.
2. `models/qwen3.cpp` — `qwen3_load` + the dense forward pass. The simplest core; read it first.
3. `layers/` — `ops.h` (rms_norm / linear / rope_neox / swiglu / rope_mrope / gated_rms_norm),
   `attention.*`, `sampler.*`.
4. `models/qwen35.cpp` — the hybrid: `build_recurrent` (gated DeltaNet) + `build_attn_layer`.
5. `engine/model_runner.*` (graph build/run, CUDA-graph reuse), `kv_cache.*`, `recurrent_cache.*`.
6. `engine/llm.*`, `engine/engine.*` (scheduler) + `engine/thread_pool.*`.
7. `vision/` (`image` → `clip` → `vlm`), `tokenizer/`, `utils/gguf.*`.
8. `tools/{example,bench,server}/main.cpp`.

## Dev loop

```bash
cmake -B build                          # CUDA on by default; CPU only: -DNANO_CUDA=OFF
cmake --build build -j --target nano-example   # rebuild one target while iterating
```
(`NANO_CUDA` is the toggle — it force-enables ggml's CUDA backend, so setting `GGML_CUDA` directly does nothing. The build picks the `nvcc` on `PATH`; to force a toolkit/arch add `-DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.4/bin/nvcc -DCMAKE_CUDA_ARCHITECTURES=86`.)

New `.cpp` under `nanollama/` is compiled automatically (CMake globs it with `CONFIGURE_DEPENDS`) — no build-file edit. A new tool, though, needs its name added to the `foreach(tool ...)` list in `CMakeLists.txt`.

Verify **every** change:
- **Correctness** — greedy (`--temp 0`) output and tokenizer ids must stay identical to a reference run on the same prompt; CPU and GPU should agree. For images, compare to `llama-mtmd-cli` (e.g. the headline must read "MEN WALK ON MOON").
- **Speed** — `nano-bench -ngl 99` (`-np N` for batched throughput) and a concurrent run against `nano-server`; a regression usually means a broken invariant below.

## Extending

### Add a model architecture

Add `nanollama/models/<arch>.{h,cpp}` mirroring `qwen3.*` (or `qwen35.*` for a hybrid):

1. Define a model struct subclassing `Model`, holding the weight `ggml_tensor*`s and any arch hparams.
   Override the traits you need: `n_pos_per_token()` (4 for M-RoPE), `is_recurrent_layer(il)`,
   `recurrent_conv_size()` / `recurrent_state_size()` (for SSM/linear-attention state).
2. `<arch>_load(model, mp)`: read hparams via `gkey`/`arch_key` from GGUF; `dup()` each tensor by its
   GGUF name into a model context; allocate with `ggml_backend_alloc_ctx_tensors_from_buft`; stream the
   data from the file; build the host F32 embedding table (the GPU path can't `get_rows` most quant types).
3. `<arch>_model::build_graph(...)`: write the forward pass with `layers/ops.h` + `build_attention`,
   gathering the output rows at the last layer (`out_ids`) so the final norm → lm-head run only on those.
   Match the reference op order exactly — norm placement, RoPE type/theta, attention scale, MLP shape —
   or logits will silently drift.
4. **Register it in `load_model()`** (`models/model.cpp`): it reads `general.architecture` and dispatches.
   Add a branch that constructs your model and calls `<arch>_load`. (`LLM`/`Engine` go through
   `load_model`, so nothing else hardcodes the arch.)
5. `Vocab` is generic GGUF byte-level BPE; reuse it, but check the pretokenizer regex in `vocab.cpp` if
   the arch tokenizes differently.

### Add vision / multimodal input

`vision/clip.*` is a self-contained ViT encoder loaded from a separate `mmproj` GGUF; `vision/image.*`
does smart-resize + normalize; `vision/vlm.cpp::build_vlm_input` encodes the image and splices the patch
embeddings into the ChatML prompt, assigning M-RoPE positions (text → `(p,p,p,0)`; image token `i` →
`(base, base+row, base+col, 0)`; an image consumes `max(grid_w,grid_h)` positions, so the post-image
text position diverges from the sequence length). Generation then runs on **embeddings** rather than
token ids (`decode_embd` single-seq, or `Engine::admit_embd` batched).

### Add an operator / layer

**Level 1 — compose (almost always do this).** Most "ops" are a few `ggml_*` calls. Add a small
inline helper to `layers/ops.h` (like `rms_norm`/`swiglu`/`gated_rms_norm`), or a block function like
`build_attention`/`build_recurrent` for something bigger. ggml runs each op on whichever backend its
inputs already live on, so a composed op works on CPU and CUDA for free, with **zero** changes to
vendored ggml. Follow the existing tensor layouts (e.g. `[head_dim, n_head, n_tokens]`) and
reshape/permute the way `build_attention` does. Qwen3.5's whole recurrent block is ~30 lines this way,
reusing fused ggml ops (`ggml_gated_delta_net`, `ggml_ssm_conv`, `ggml_rope_multi`).

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

Three forward entry points share one graph builder (`build()` in `model_runner.cpp`, which creates the
input tensors `embd`/`pos`/`kv_idxs`/`mask`/`out_ids` and calls `model->build_graph`):

- **`decode(tokens, n_tokens, n_past)`** — one sequence; looks up token embeddings, then delegates to
  `decode_embd`. Used by `LLM`/`nano-bench`.
- **`decode_embd(embd, pos4, n_tokens, n_past)`** — one sequence from precomputed embeddings + explicit
  M-RoPE; the multimodal/single-seq VLM path.
- **`decode_batch(b, s0, n_stream, n_q, n_kv)`** — a **per-stream** batched forward of `n_stream`
  sequences (slots `s0..s0+n_stream-1`) with `n_q` tokens each; `b` carries tokens *or* embeddings and
  optional explicit M-RoPE. The engine's hot path.

Caches and scheduler:

- **`kv_cache.*` — storage only (attention layers).** Per-attention-layer F16 K/V tensors of shape
  `[n_embd_kv, n_slots*n_ctx_pad + 1]`, allocated once. Recurrent layers get **no** KV (`k[il]`/`v[il]`
  are null). Slot `s` owns the *contiguous* cell range `[s*n_ctx_pad, (s+1)*n_ctx_pad)`, so the cache
  views as a 4D per-stream tensor `[head_dim, n_head_kv, n_kv, n_stream]` (llama.cpp's `get_k` strides),
  and each query attends only its own slot. The `+1` is a scratch sink for gap-filler dummy tokens.
  *Which* cell a token lands in is `kv_dst[i]` (`= slot*n_ctx_pad + pos`) in `ModelRunner::Batch`.
- **`recurrent_cache.* — per-slot SSM state.** Qwen3.5's recurrent layers carry a conv ring buffer +
  delta state per sequence (F32, one column per slot). This is **large** (~50 MB/slot for Qwen3.5), so it,
  not the KV cache, caps concurrency. State can't span steps, so a recurrent slot prefills its whole
  prompt at once and is cleared (`reset_recurrent_slot`) when a new sequence starts.
- **`engine.cpp` — the scheduler.** A `Slot` is one in-flight request (prompt/embeddings, generated
  tokens, `n_past`, `mrope_next`, per-slot `Sampler`, callbacks). `step()` runs **decode** (pack every
  generating slot into one per-stream `decode_batch`, one token each; idle gaps → masked dummy in the
  scratch cell) then **prefill** (advance each waiting slot; text chunks within a budget, recurrent/image
  slots prefill whole). `admit` takes token prompts; `admit_embd` takes spliced image embeddings + M-RoPE.
  - Per-slot **sampling is parallelized** across the `thread_pool` (each slot's argmax over a large vocab
    is independent); emitting/callbacks stay serial.
  - Change **scheduling** (priorities, prefill chunk size) → the two passes in `step()`.
  - Change **request lifecycle** (stop strings, cancellation) → `Slot` + `emit_token`.
  - Attention isolation is **structural** (each stream views only its own slot), so the mask is plain
    causal. A KV-layout change must keep slots contiguous so the per-stream view stays valid.

## Invariants (don't break these)

- **Single backend.** The whole graph runs on one backend; this is what lets ggml capture & replay a CUDA graph. Don't reintroduce a multi-backend scheduler without measuring tg speed.
- **Host embedding lookup.** CUDA's `get_rows` can't read Q6_K, so the embedding is gathered on the host into an input tensor. Keep new graphs free of unsupported ops on the GPU path.
- **Stable graph topology.** Graphs are rebuilt into fixed buffers (`dec_mem`/`bat_mem`) so same-shape steps replay the captured CUDA graph. The shape is `n_kv` (padded to 256), plus — for batched decode — `n_stream` (active slots) and `n_q`: a steady set of running slots replays, while admitting/freeing re-captures. Changing shapes every step forfeits that speed.
- **M-RoPE position ≠ sequence position.** For Qwen3.5 an image consumes `max(grid)` rope positions but many KV cells, so a slot tracks both `n_past` (KV cell) and `mrope_next` (rope position). The mask uses sequence positions; rope uses the 4-section M-RoPE positions.
- **Match the reference numerically where it's cheap.** Greedy text is token-for-token identical to llama.cpp (CPU+GPU). Vision output can differ on low-confidence tokens (independent ViT fp accumulation) but the algorithm — grid, token count, M-RoPE, flash-attn — must match.
