// chat.cpp — ChatML prompt formatting
#include "nanollama/tokenizer/chat.h"

namespace nano {

std::string apply_chatml(const std::vector<ChatMessage> & msgs,
                         bool add_generation_prompt, bool enable_thinking) {
    std::string s;
    for (const auto & m : msgs) {
        s += "<|im_start|>" + m.role + "\n" + m.content + "<|im_end|>\n";
    }
    if (add_generation_prompt) {
        s += "<|im_start|>assistant\n";
        if (!enable_thinking) s += "<think>\n\n</think>\n\n";  // Qwen3: suppress reasoning
    }
    return s;
}

} // namespace nano
