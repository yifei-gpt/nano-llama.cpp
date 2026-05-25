// vocab.cpp — byte-level BPE tokenizer (encode/decode) loaded from GGUF
#include "nanollama/tokenizer/vocab.h"
#include "nanollama/tokenizer/unicode.h"
#include "nanollama/common.h"

#include <algorithm>
#include <queue>

namespace nano {

// GPT-2-style pretokenizer regex
static const std::vector<std::string> QWEN2_REGEX = {
    "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
};

void Vocab::load(const GgufFile & gf) {
    id_to_token = gf.arr_strs(gkey::TOK_LIST);
    token_type  = gf.arr_i32(gkey::TOK_TYPES);
    token_to_id.reserve(id_to_token.size() * 2);
    for (int32_t i = 0; i < (int32_t) id_to_token.size(); i++) token_to_id[id_to_token[i]] = i;

    auto merges = gf.arr_strs(gkey::TOK_MERGES);
    for (int32_t i = 0; i < (int32_t) merges.size(); i++) merge_rank[merges[i]] = i;

    for (int32_t i = 0; i < (int32_t) id_to_token.size(); i++) {
        int t = i < (int32_t) token_type.size() ? token_type[i] : 1;
        if (t == 3 || t == 4) specials.push_back({ id_to_token[i], i });  // CONTROL / USER_DEFINED
    }
    // longest surface form first so e.g. "<|im_start|>" wins over "<"
    std::sort(specials.begin(), specials.end(),
              [](auto & a, auto & b) { return a.first.size() > b.first.size(); });

    bos_id  = gf.has(gkey::TOK_BOS) ? gf.i32(gkey::TOK_BOS) : -1;
    eos_id  = gf.has(gkey::TOK_EOS) ? gf.i32(gkey::TOK_EOS) : -1;
    add_bos = gf.boolean(gkey::TOK_ADD_BOS, false);
    NANO_LOG("vocab: %zu tokens, %zu merges, %zu special, bos=%d eos=%d",
             id_to_token.size(), merge_rank.size(), specials.size(), bos_id, eos_id);
}

bool Vocab::is_eog(int32_t id) const {
    if (id == eos_id) return true;
    // also treat <|im_end|> / <|endoftext|> as end-of-generation
    if (id >= 0 && id < (int32_t) id_to_token.size()) {
        const std::string & t = id_to_token[id];
        if (t == "<|im_end|>" || t == "<|endoftext|>") return true;
    }
    return false;
}

std::vector<int32_t> Vocab::bpe_encode(const std::string & text) const {
    std::vector<int32_t> out;
    for (const std::string & word : unicode_regex_split(text, QWEN2_REGEX, true)) {
        // ignore_merges: emit the whole pretoken if it is itself a vocab token
        auto whole = token_to_id.find(word);
        if (whole != token_to_id.end()) { out.push_back(whole->second); continue; }

        struct Sym { int prev, next; std::string text; };
        std::vector<Sym> syms;
        for (size_t i = 0; i < word.size(); ) {
            size_t len = unicode_len_utf8(word[i]);
            len = std::min(len, word.size() - i);
            syms.push_back({ (int) syms.size() - 1, (int) syms.size() + 1, word.substr(i, len) });
            i += len;
        }
        if (syms.empty()) continue;
        syms.back().next = -1;

        struct Bigram { int left, right, rank; size_t size; };
        auto cmp = [](const Bigram & a, const Bigram & b) {
            return a.rank > b.rank || (a.rank == b.rank && a.left > b.left);
        };
        std::priority_queue<Bigram, std::vector<Bigram>, decltype(cmp)> pq(cmp);

        auto try_add = [&](int l, int r) {
            if (l == -1 || r == -1) return;
            auto it = merge_rank.find(syms[l].text + " " + syms[r].text);
            if (it == merge_rank.end()) return;
            pq.push({ l, r, it->second, syms[l].text.size() + syms[r].text.size() });
        };
        for (int i = 0; i + 1 < (int) syms.size(); i++) try_add(i, i + 1);

        while (!pq.empty()) {
            Bigram b = pq.top(); pq.pop();
            Sym & l = syms[b.left];
            if (b.right == -1 || l.next != b.right) continue;       // stale
            Sym & r = syms[b.right];
            if (l.text.size() + r.text.size() != b.size) continue;  // stale
            l.text += r.text;
            l.next = r.next;
            if (r.next != -1) syms[r.next].prev = b.left;
            r.text.clear();
            try_add(l.prev, b.left);
            try_add(b.left, l.next);
        }

        for (int i = 0; i != -1; i = syms[i].next) {
            if (syms[i].text.empty()) continue;
            auto it = token_to_id.find(syms[i].text);
            if (it != token_to_id.end()) out.push_back(it->second);
        }
    }
    return out;
}

std::vector<int32_t> Vocab::tokenize(const std::string & text, bool add_bos_token, bool parse_special) const {
    std::vector<int32_t> out;
    if (add_bos_token && bos_id >= 0) out.push_back(bos_id);

    if (!parse_special || specials.empty()) {
        auto e = bpe_encode(text);
        out.insert(out.end(), e.begin(), e.end());
        return out;
    }

    // split on special-token surface forms (longest first), BPE the gaps
    size_t pos = 0;
    while (pos < text.size()) {
        size_t best = std::string::npos; int best_id = -1; size_t best_len = 0;
        for (auto & s : specials) {
            size_t f = text.find(s.first, pos);
            if (f != std::string::npos && (f < best || (f == best && s.first.size() > best_len))) {
                best = f; best_id = s.second; best_len = s.first.size();
            }
        }
        if (best == std::string::npos) {
            auto e = bpe_encode(text.substr(pos)); out.insert(out.end(), e.begin(), e.end()); break;
        }
        if (best > pos) { auto e = bpe_encode(text.substr(pos, best - pos)); out.insert(out.end(), e.begin(), e.end()); }
        out.push_back(best_id);
        pos = best + best_len;
    }
    return out;
}

std::string Vocab::token_to_piece(int32_t id) const {
    if (id < 0 || id >= (int32_t) id_to_token.size()) return "";
    int t = id < (int32_t) token_type.size() ? token_type[id] : 1;
    const std::string & tok = id_to_token[id];
    if (t == 3 || t == 4) return tok;     // control/user-defined: literal surface form
    std::string out;
    try {
        for (size_t i = 0; i < tok.size(); ) {
            size_t len = unicode_len_utf8(tok[i]);
            len = std::min(len, tok.size() - i);
            out += (char) unicode_utf8_to_byte(tok.substr(i, len));
            i += len;
        }
    } catch (...) { return tok; }   // not byte-encoded → fall back to the literal surface form
    return out;
}

} // namespace nano
