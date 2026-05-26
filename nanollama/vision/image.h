// image.h — load + preprocess an image for the vision encoder (Qwen-VL "smart resize")
#pragma once

#include <string>
#include <vector>

namespace nano {

// preprocessed image: planar F32, normalized; data[c*nx*ny + y*nx + x]
struct ClipImage {
    int nx = 0, ny = 0;
    std::vector<float> data;
};

// Decode an image (file path or in-memory bytes), smart-resize to multiples of `align` within
// [min_px, max_px], normalize per channel. Returns false if it can't be decoded.
bool load_and_preprocess(const std::string & path, int align, int min_px, int max_px,
                         const float mean[3], const float std_[3], ClipImage & out);
bool load_and_preprocess(const unsigned char * bytes, int n_bytes, int align, int min_px, int max_px,
                         const float mean[3], const float std_[3], ClipImage & out);

} // namespace nano
