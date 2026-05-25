// clip.h — Qwen3-VL vision encoder (mmproj): load the ViT + run an image to LLM-space embeddings
#pragma once

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "nanollama/vision/image.h"

#include <string>
#include <vector>

namespace nano {

struct ClipLayer {
    ggml_tensor * ln1_w = nullptr, * ln1_b = nullptr, * ln2_w = nullptr, * ln2_b = nullptr;
    ggml_tensor * qkv_w = nullptr, * qkv_b = nullptr, * o_w = nullptr, * o_b = nullptr;
    ggml_tensor * up_w = nullptr, * up_b = nullptr, * down_w = nullptr, * down_b = nullptr;
};

struct ClipModel {
    int   n_embd = 0, n_layer = 0, n_head = 0, patch_size = 0, n_merge = 0, image_size = 0, proj_dim = 0;
    float eps = 1e-6f, mean[3] = {0.5f, 0.5f, 0.5f}, std[3] = {0.5f, 0.5f, 0.5f};

    ggml_tensor * patch_embd_0 = nullptr, * patch_embd_1 = nullptr, * patch_bias = nullptr;
    ggml_tensor * position_embd = nullptr, * post_ln_w = nullptr, * post_ln_b = nullptr;
    ggml_tensor * mm0_w = nullptr, * mm0_b = nullptr, * mm1_w = nullptr, * mm1_b = nullptr;
    std::vector<ClipLayer> layers;

    ggml_context *        ctx = nullptr;
    std::vector<ggml_backend_buffer_t> bufs;
    ggml_backend_t        backend = nullptr;
    ggml_gallocr_t        galloc  = nullptr;
    bool on_gpu = false;

    int align()   const { return patch_size * n_merge; }
    int min_px()  const { return 8    * patch_size * patch_size * n_merge * n_merge; }
    int max_px()  const { return 4096 * patch_size * patch_size * n_merge * n_merge; }
    ~ClipModel();
};

bool clip_load(ClipModel & m, const std::string & path, int n_gpu_layers);

// Run the encoder. Returns embeddings [proj_dim * n_tokens]; sets n_tokens and the merged grid (gw·gh = n_tokens).
std::vector<float> clip_encode(ClipModel & m, const ClipImage & img, int & n_tokens, int & gw, int & gh);

} // namespace nano
