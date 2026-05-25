// gguf.cpp — minimal GGUF metadata/tensor reader over ggml's parser
#include "nanollama/utils/gguf.h"
#include "nanollama/common.h"

namespace nano {

GgufFile::GgufFile(const std::string & fname) : path(fname) {
    gguf_init_params params = { /*no_alloc=*/true, /*ctx=*/&meta };
    gguf = gguf_init_from_file(fname.c_str(), params);
    if (!gguf) NANO_ABORT("failed to open GGUF file: %s", fname.c_str());
    data_offset = gguf_get_data_offset(gguf);
}

GgufFile::~GgufFile() {
    if (gguf) gguf_free(gguf);
    if (meta) ggml_free(meta);
}

bool GgufFile::has(const std::string & key) const {
    return gguf_find_key(gguf, key.c_str()) >= 0;
}

static int64_t find_or_abort(const gguf_context * g, const std::string & key) {
    int64_t id = gguf_find_key(g, key.c_str());
    if (id < 0) NANO_ABORT("missing GGUF key: %s", key.c_str());
    return id;
}

std::string GgufFile::str(const std::string & key) const {
    return gguf_get_val_str(gguf, find_or_abort(gguf, key));
}
uint32_t GgufFile::u32(const std::string & key) const {
    return (uint32_t) i32(key);   // i32() handles any integer width
}
int32_t GgufFile::i32(const std::string & key) const {
    // token-id / count keys may be stored as any integer width — dispatch on actual type
    int64_t id = find_or_abort(gguf, key);
    switch (gguf_get_kv_type(gguf, id)) {
        case GGUF_TYPE_INT32:  return gguf_get_val_i32(gguf, id);
        case GGUF_TYPE_UINT32: return (int32_t) gguf_get_val_u32(gguf, id);
        case GGUF_TYPE_INT16:  return gguf_get_val_i16(gguf, id);
        case GGUF_TYPE_UINT16: return gguf_get_val_u16(gguf, id);
        case GGUF_TYPE_INT8:   return gguf_get_val_i8(gguf, id);
        case GGUF_TYPE_UINT8:  return gguf_get_val_u8(gguf, id);
        default: NANO_ABORT("key %s is not an integer type", key.c_str());
    }
}
static float read_f32(const gguf_context * g, int64_t id, const char * key) {
    switch (gguf_get_kv_type(g, id)) {     // float keys may be stored as F32 or F64
        case GGUF_TYPE_FLOAT32: return gguf_get_val_f32(g, id);
        case GGUF_TYPE_FLOAT64: return (float) gguf_get_val_f64(g, id);
        default: NANO_ABORT("key %s is not a float type", key);
    }
}
float GgufFile::f32(const std::string & key) const {
    return read_f32(gguf, find_or_abort(gguf, key), key.c_str());
}
bool GgufFile::boolean(const std::string & key, bool dflt) const {
    int64_t id = gguf_find_key(gguf, key.c_str());
    return id < 0 ? dflt : gguf_get_val_bool(gguf, id);
}
float GgufFile::f32_or(const std::string & key, float dflt) const {
    int64_t id = gguf_find_key(gguf, key.c_str());
    return id < 0 ? dflt : read_f32(gguf, id, key.c_str());
}

std::vector<std::string> GgufFile::arr_strs(const std::string & key) const {
    int64_t id = find_or_abort(gguf, key);
    size_t n = gguf_get_arr_n(gguf, id);
    std::vector<std::string> out;
    out.reserve(n);
    for (size_t i = 0; i < n; i++) out.emplace_back(gguf_get_arr_str(gguf, id, i));
    return out;
}
std::vector<int32_t> GgufFile::arr_i32(const std::string & key) const {
    int64_t id = find_or_abort(gguf, key);
    if (gguf_get_arr_type(gguf, id) != GGUF_TYPE_INT32) NANO_ABORT("gguf array %s is not int32", key.c_str());
    size_t n = gguf_get_arr_n(gguf, id);
    const int32_t * data = (const int32_t *) gguf_get_arr_data(gguf, id);
    return std::vector<int32_t>(data, data + n);
}
std::vector<float> GgufFile::arr_f32(const std::string & key) const {
    int64_t id = find_or_abort(gguf, key);
    if (gguf_get_arr_type(gguf, id) != GGUF_TYPE_FLOAT32) NANO_ABORT("gguf array %s is not float32", key.c_str());
    size_t n = gguf_get_arr_n(gguf, id);
    const float * data = (const float *) gguf_get_arr_data(gguf, id);
    return std::vector<float>(data, data + n);
}

ggml_tensor * GgufFile::tensor(const std::string & name) const {
    return ggml_get_tensor(meta, name.c_str());
}
size_t GgufFile::tensor_file_offset(const std::string & name) const {
    int64_t tid = gguf_find_tensor(gguf, name.c_str());
    if (tid < 0) NANO_ABORT("missing GGUF tensor: %s", name.c_str());
    return data_offset + gguf_get_tensor_offset(gguf, tid);
}

} // namespace nano
