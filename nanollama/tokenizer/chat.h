// chat.h — ChatML prompt template
#pragma once
#include <string>
#include <vector>

namespace nano {

struct ChatMessage { std::string role; std::string content; };

// enable_thinking=false appends an empty <think></think> block to suppress reasoning
std::string apply_chatml(const std::vector<ChatMessage> & msgs,
                         bool add_generation_prompt, bool enable_thinking);

} // namespace nano
