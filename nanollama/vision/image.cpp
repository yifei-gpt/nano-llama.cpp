// image.cpp — image decode (stb_image) + Qwen-VL smart-resize/normalize
#include "nanollama/vision/image.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <algorithm>
#include <cmath>

namespace nano {

static inline float lerp(float a, float b, float t) { return a + (b - a) * t; }

// "smart resize": align W,H up to a multiple of `align`, then shrink/grow to fit [min_px, max_px]
static void smart_resize(int w, int h, int align, int min_px, int max_px, int & ow, int & oh) {
    auto round_f = [&](float x) { return std::max(align, (int) std::lround(x / align) * align); };
    auto floor_f = [&](float x) { return std::max(align, (int) std::floor(x / align) * align); };
    auto ceil_f  = [&](float x) { return std::max(align, (int) std::ceil (x / align) * align); };
    ow = round_f((float) w);
    oh = round_f((float) h);
    if ((int64_t) ow * oh > max_px) {
        const float beta = std::sqrt((float) w * h / max_px);
        ow = floor_f(w / beta);
        oh = floor_f(h / beta);
    } else if ((int64_t) ow * oh < min_px) {
        const float beta = std::sqrt((float) min_px / ((float) w * h));
        ow = ceil_f(w * beta);
        oh = ceil_f(h * beta);
    }
}

// bilinear-resize the loaded RGB image to the smart-resized size, then normalize into planar layout
static void preprocess_rgb(const unsigned char * src, int w, int h, int align, int min_px, int max_px,
                           const float mean[3], const float std_[3], ClipImage & out) {
    int nx, ny;
    smart_resize(w, h, align, min_px, max_px, nx, ny);
    out.nx = nx; out.ny = ny;
    out.data.resize((size_t) nx * ny * 3);
    const float xr = nx > 1 ? (float) (w - 1) / (nx - 1) : 0.0f;
    const float yr = ny > 1 ? (float) (h - 1) / (ny - 1) : 0.0f;
    for (int y = 0; y < ny; y++) {
        for (int x = 0; x < nx; x++) {
            const float px = x * xr, py = y * yr;
            const int x0 = std::min((int) px, w - 1), y0 = std::min((int) py, h - 1);
            const int x1 = std::min(x0 + 1, w - 1),    y1 = std::min(y0 + 1, h - 1);
            const float xf = px - x0, yf = py - y0;
            const size_t r0 = (size_t) y0 * w, r1 = (size_t) y1 * w;   // size_t: a huge image must not overflow int
            for (int ch = 0; ch < 3; ch++) {
                const float top = lerp((float) src[3 * (r0 + x0) + ch], (float) src[3 * (r0 + x1) + ch], xf);
                const float bot = lerp((float) src[3 * (r1 + x0) + ch], (float) src[3 * (r1 + x1) + ch], xf);
                const float u8  = std::round(lerp(top, bot, yf));   // llama rounds to u8 before normalizing
                out.data[(size_t) ch * nx * ny + (size_t) y * nx + x] = (u8 / 255.0f - mean[ch]) / std_[ch];
            }
        }
    }
}

bool load_and_preprocess(const std::string & path, int align, int min_px, int max_px,
                         const float mean[3], const float std_[3], ClipImage & out) {
    int w, h, c;
    unsigned char * src = stbi_load(path.c_str(), &w, &h, &c, 3);
    if (!src) return false;
    preprocess_rgb(src, w, h, align, min_px, max_px, mean, std_, out);
    stbi_image_free(src);
    return true;
}

bool load_and_preprocess(const unsigned char * bytes, int n_bytes, int align, int min_px, int max_px,
                         const float mean[3], const float std_[3], ClipImage & out) {
    int w, h, c;
    unsigned char * src = stbi_load_from_memory(bytes, n_bytes, &w, &h, &c, 3);
    if (!src) return false;
    preprocess_rgb(src, w, h, align, min_px, max_px, mean, std_, out);
    stbi_image_free(src);
    return true;
}

} // namespace nano
