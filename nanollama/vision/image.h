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

// Load `path`, smart-resize so W,H are multiples of `align` with min_px ≤ W·H ≤ max_px (bilinear),
// then normalize per channel: (px/255 - mean)/std. Returns false if the file can't be decoded.
bool load_and_preprocess(const std::string & path, int align, int min_px, int max_px,
                         const float mean[3], const float std_[3], ClipImage & out);

} // namespace nano
