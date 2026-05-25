// common.h — GGUF metadata key strings, logging, small helpers shared across the package.
#pragma once

#include <cstdio>
#include <cstdlib>
#include <string>

namespace nano {

// ---- logging ----
#define NANO_LOG(...)  do { fprintf(stderr, __VA_ARGS__); fputc('\n', stderr); } while (0)
#define NANO_ABORT(...) do { fprintf(stderr, "nano: fatal: "); fprintf(stderr, __VA_ARGS__); \
                             fputc('\n', stderr); abort(); } while (0)
#define NANO_ASSERT(x) do { if (!(x)) NANO_ABORT("assert failed: %s (%s:%d)", #x, __FILE__, __LINE__); } while (0)

// GGUF metadata keys; "%s" is filled with the arch name via arch_key()
namespace gkey {
    constexpr const char * ARCH            = "general.architecture";
    constexpr const char * NAME            = "general.name";
    constexpr const char * BLOCK_COUNT     = "%s.block_count";
    constexpr const char * CTX_LEN         = "%s.context_length";
    constexpr const char * EMBD_LEN        = "%s.embedding_length";
    constexpr const char * FF_LEN          = "%s.feed_forward_length";
    constexpr const char * N_HEAD          = "%s.attention.head_count";
    constexpr const char * N_HEAD_KV       = "%s.attention.head_count_kv";
    constexpr const char * KEY_LEN         = "%s.attention.key_length";    // head_dim
    constexpr const char * RMS_EPS         = "%s.attention.layer_norm_rms_epsilon";
    constexpr const char * ROPE_FREQ_BASE  = "%s.rope.freq_base";

    // qwen3.5 (hybrid gated-DeltaNet + attention)
    constexpr const char * ROPE_DIM_COUNT  = "%s.rope.dimension_count";
    constexpr const char * ROPE_SECTIONS   = "%s.rope.dimension_sections";
    constexpr const char * FULL_ATTN_INTERVAL = "%s.full_attention_interval";
    constexpr const char * SSM_CONV_KERNEL = "%s.ssm.conv_kernel";
    constexpr const char * SSM_STATE_SIZE  = "%s.ssm.state_size";
    constexpr const char * SSM_GROUP_COUNT = "%s.ssm.group_count";
    constexpr const char * SSM_DT_RANK     = "%s.ssm.time_step_rank";
    constexpr const char * SSM_INNER_SIZE  = "%s.ssm.inner_size";

    // tokenizer
    constexpr const char * TOK_LIST        = "tokenizer.ggml.tokens";
    constexpr const char * TOK_TYPES       = "tokenizer.ggml.token_type";
    constexpr const char * TOK_MERGES      = "tokenizer.ggml.merges";
    constexpr const char * TOK_BOS         = "tokenizer.ggml.bos_token_id";
    constexpr const char * TOK_EOS         = "tokenizer.ggml.eos_token_id";
    constexpr const char * TOK_ADD_BOS     = "tokenizer.ggml.add_bos_token";
}

inline std::string arch_key(const char * fmt, const std::string & arch) {
    char buf[256];
    snprintf(buf, sizeof(buf), fmt, arch.c_str());
    return buf;
}

} // namespace nano
