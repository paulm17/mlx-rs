#include "llama.cpp/common/chat.h"
#include "llama.cpp/include/llama.h"

#include <cstdlib>
#include <cstring>
#include <map>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

using json = nlohmann::ordered_json;

struct mlx_rs_chat_template_result {
    char * prompt;
    char * generation_prompt;
    char * parser;
    int chat_format;
    char ** additional_stops;
    size_t additional_stops_count;
    bool parse_tool_calls;
};

struct mlx_rs_chat_parse_state {
    common_chat_parser_params params;
    std::string generated;
    common_chat_msg previous;
    bool has_previous = false;
};

static char * copy_string(const std::string & value) {
    auto * result = static_cast<char *>(std::malloc(value.size() + 1));
    if (result == nullptr) {
        return nullptr;
    }
    std::memcpy(result, value.data(), value.size());
    result[value.size()] = '\0';
    return result;
}

static void clear_result(mlx_rs_chat_template_result * result) {
    if (result == nullptr) {
        return;
    }
    std::free(result->prompt);
    std::free(result->generation_prompt);
    std::free(result->parser);
    for (size_t i = 0; i < result->additional_stops_count; ++i) {
        std::free(result->additional_stops[i]);
    }
    std::free(result->additional_stops);
    std::memset(result, 0, sizeof(*result));
}

static common_reasoning_format reasoning_format(const char * value) {
    return value == nullptr || std::strlen(value) == 0
        ? COMMON_REASONING_FORMAT_NONE
        : common_reasoning_format_from_name(value);
}

static void set_kwargs(const char * value, common_chat_templates_inputs & inputs) {
    if (value == nullptr || std::strlen(value) == 0) {
        return;
    }
    const auto parsed = json::parse(value);
    if (!parsed.is_object()) {
        return;
    }
    for (const auto & item : parsed.items()) {
        inputs.chat_template_kwargs[item.key()] = item.value().dump();
    }
}

static void set_result(
    const common_chat_params & params,
    bool parse_tool_calls,
    mlx_rs_chat_template_result * result) {
    result->prompt = copy_string(params.prompt);
    result->generation_prompt = copy_string(params.generation_prompt);
    result->parser = copy_string(params.parser);
    result->chat_format = static_cast<int>(params.format);
    result->parse_tool_calls = parse_tool_calls;
    result->additional_stops_count = params.additional_stops.size();
    if (result->additional_stops_count > 0) {
        result->additional_stops = static_cast<char **>(std::calloc(
            result->additional_stops_count, sizeof(char *)));
        for (size_t i = 0; i < result->additional_stops_count; ++i) {
            result->additional_stops[i] = copy_string(params.additional_stops[i]);
        }
    }
}

static common_chat_parser_params parser_from_result(
    int chat_format,
    bool parse_tool_calls,
    const char * generation_prompt,
    const char * parser_serialized,
    const char * reasoning) {
    common_chat_parser_params params;
    params.format = static_cast<common_chat_format>(chat_format);
    params.parse_tool_calls = parse_tool_calls;
    params.generation_prompt = generation_prompt == nullptr ? "" : generation_prompt;
    if (parser_serialized != nullptr && std::strlen(parser_serialized) > 0) {
        params.parser.load(parser_serialized);
    }
    params.reasoning_format = reasoning_format(reasoning);
    return params;
}

extern "C" int mlx_rs_apply_chat_template_oaicompat(
    const struct llama_model * model,
    const char * chat_template,
    const char * messages_json,
    const char * tools_json,
    const char * tool_choice,
    const char * reasoning,
    const char * chat_template_kwargs,
    bool add_generation_prompt,
    bool parallel_tool_calls,
    bool enable_thinking,
    bool add_bos,
    bool add_eos,
    bool parse_tool_calls,
    mlx_rs_chat_template_result * result) {
    if (model == nullptr || chat_template == nullptr || messages_json == nullptr || result == nullptr) {
        return -1;
    }
    try {
        auto templates = common_chat_templates_init(model, chat_template);
        if (!templates) {
            return -1;
        }
        common_chat_templates_inputs inputs;
        inputs.messages = common_chat_msgs_parse_oaicompat(json::parse(messages_json));
        if (tools_json != nullptr && std::strlen(tools_json) > 0) {
            inputs.tools = common_chat_tools_parse_oaicompat(json::parse(tools_json));
        }
        if (tool_choice != nullptr && std::strlen(tool_choice) > 0) {
            inputs.tool_choice = common_chat_tool_choice_parse_oaicompat(tool_choice);
        }
        inputs.reasoning_format = reasoning_format(reasoning);
        inputs.add_generation_prompt = add_generation_prompt;
        inputs.parallel_tool_calls = parallel_tool_calls;
        inputs.enable_thinking = enable_thinking;
        inputs.add_bos = add_bos;
        inputs.add_eos = add_eos;
        set_kwargs(chat_template_kwargs, inputs);
        const auto params = common_chat_templates_apply(templates.get(), inputs);
        clear_result(result);
        set_result(params, parse_tool_calls, result);
        return result->prompt == nullptr ? -2 : 0;
    } catch (...) {
        clear_result(result);
        return -3;
    }
}

extern "C" void mlx_rs_free_chat_template_result(mlx_rs_chat_template_result * result) {
    clear_result(result);
}

extern "C" int mlx_rs_parse_chat_response(
    const char * text,
    bool is_partial,
    int chat_format,
    bool parse_tool_calls,
    const char * generation_prompt,
    const char * parser_serialized,
    const char * reasoning,
    char ** output) {
    if (text == nullptr || output == nullptr) {
        return -1;
    }
    *output = nullptr;
    try {
        const auto params = parser_from_result(
            chat_format, parse_tool_calls, generation_prompt, parser_serialized, reasoning);
        const auto message = common_chat_parse(text, is_partial, params);
        *output = copy_string(message.to_json_oaicompat(true).dump());
        return *output == nullptr ? -2 : 0;
    } catch (...) {
        return -3;
    }
}

extern "C" int mlx_rs_init_chat_parse_state(
    int chat_format,
    bool parse_tool_calls,
    const char * generation_prompt,
    const char * parser_serialized,
    const char * reasoning,
    mlx_rs_chat_parse_state ** output) {
    if (output == nullptr) {
        return -1;
    }
    try {
        auto * state = new mlx_rs_chat_parse_state;
        state->params = parser_from_result(
            chat_format, parse_tool_calls, generation_prompt, parser_serialized, reasoning);
        *output = state;
        return 0;
    } catch (...) {
        return -3;
    }
}

extern "C" int mlx_rs_update_chat_parse_state(
    mlx_rs_chat_parse_state * state,
    const char * text_added,
    bool is_partial,
    char *** outputs,
    size_t * output_count) {
    if (state == nullptr || text_added == nullptr || outputs == nullptr || output_count == nullptr) {
        return -1;
    }
    *outputs = nullptr;
    *output_count = 0;
    try {
        state->generated += text_added;
        const auto current = common_chat_parse(state->generated, is_partial, state->params);
        const auto diffs = state->has_previous
            ? common_chat_msg_diff::compute_diffs(state->previous, current)
            : common_chat_msg_diff::compute_diffs(common_chat_msg(), current);
        std::vector<std::string> json_deltas;
        for (const auto & diff : diffs) {
            json delta = json::object();
            if (!diff.reasoning_content_delta.empty()) {
                delta["reasoning_content"] = diff.reasoning_content_delta;
            }
            if (!diff.content_delta.empty()) {
                delta["content"] = diff.content_delta;
            }
            if (diff.tool_call_index != std::string::npos) {
                json call = {
                    {"index", diff.tool_call_index},
                    {"id", diff.tool_call_delta.id},
                    {"type", "function"},
                    {"function", {
                        {"name", diff.tool_call_delta.name},
                        {"arguments", diff.tool_call_delta.arguments},
                    }},
                };
                delta["tool_calls"] = json::array({call});
            }
            if (!delta.empty()) {
                json_deltas.push_back(delta.dump());
            }
        }
        state->previous = current;
        state->has_previous = true;
        if (!json_deltas.empty()) {
            *outputs = static_cast<char **>(std::calloc(json_deltas.size(), sizeof(char *)));
            *output_count = json_deltas.size();
            for (size_t i = 0; i < json_deltas.size(); ++i) {
                (*outputs)[i] = copy_string(json_deltas[i]);
            }
        }
        return 0;
    } catch (...) {
        return -3;
    }
}

extern "C" void mlx_rs_free_chat_parse_outputs(char ** outputs, size_t count) {
    if (outputs == nullptr) {
        return;
    }
    for (size_t i = 0; i < count; ++i) {
        std::free(outputs[i]);
    }
    std::free(outputs);
}

extern "C" void mlx_rs_free_chat_parse_state(mlx_rs_chat_parse_state * state) {
    delete state;
}
