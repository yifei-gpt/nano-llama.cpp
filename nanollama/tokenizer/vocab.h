// vocab.h — byte-level BPE tokenizer
#pragma once

#include "nanollama/utils/gguf.h"
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace nano {

struct Vocab {
    std::vector<std::string> id_to_token;
    std::vector<int32_t>     token_type;       // 3=CONTROL, 4=USER_DEFINED treated as special
    std::unordered_map<std::string, int32_t> token_to_id;
    std::unordered_map<std::string, int32_t> merge_rank;  // "A B" -> rank (lower = merged first)

    std::vector<std::pair<std::string, int32_t>> specials;  // surface form -> id (for splitting)

    int32_t bos_id = -1, eos_id = -1;
    bool    add_bos = false;

    void load(const GgufFile & gf);

    // Encode text to token ids. parse_special splits out <|im_start|> etc.
    std::vector<int32_t> tokenize(const std::string & text, bool add_bos_token, bool parse_special) const;

    // decode one token to its text piece (byte-decoded for normal tokens)
    std::string token_to_piece(int32_t id) const;

    bool is_eog(int32_t id) const;

private:
    std::vector<int32_t> bpe_encode(const std::string & text) const;  // no special handling
};

} // namespace nano
