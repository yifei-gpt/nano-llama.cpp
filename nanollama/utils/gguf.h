// gguf.h — thin RAII wrapper over ggml's GGUF parser
#pragma once

#include <ggml.h>
#include <gguf.h>   // angle brackets: this file is itself named gguf.h — avoid self-include

#include <string>
#include <vector>
#include <cstdint>

namespace nano {

// metadata + tensor descriptors only (no_alloc); tensor data is streamed separately by the loader
struct GgufFile {
    gguf_context * gguf = nullptr;
    ggml_context * meta = nullptr;
    std::string    path;
    size_t         data_offset = 0;

    explicit GgufFile(const std::string & fname);
    ~GgufFile();
    GgufFile(const GgufFile &) = delete;
    GgufFile & operator=(const GgufFile &) = delete;

    // scalars (abort if a required key is missing)
    bool        has(const std::string & key) const;
    std::string str (const std::string & key) const;
    uint32_t    u32 (const std::string & key) const;
    int32_t     i32 (const std::string & key) const;
    float       f32 (const std::string & key) const;
    bool        boolean(const std::string & key, bool dflt) const;
    float       f32_or(const std::string & key, float dflt) const;

    // arrays
    std::vector<std::string> arr_strs(const std::string & key) const;
    std::vector<int32_t>     arr_i32 (const std::string & key) const;
    std::vector<float>       arr_f32 (const std::string & key) const;

    // tensors
    ggml_tensor * tensor(const std::string & name) const;            // nullptr if absent
    size_t        tensor_file_offset(const std::string & name) const;
};

} // namespace nano
