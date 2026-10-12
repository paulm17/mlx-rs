// Text-only adapter derived from upstream server-decision.cpp at 99b95488cac0f00ce3f05af113a8c1e287753f87.
// Upstream MIT license: vendor/llama-cpp-sys-2/llama.cpp/LICENSE.
#include "common.h"
#include "chat.h"
#include "src/llama-ext.h"
#include <algorithm>
#include <cmath>
#include <regex>
#include <stdexcept>
#include <cstdlib>
#include <cstring>
#include <memory>
using json = common_json;
namespace mlx_decision {
struct text_decision_task { llama_tokens tokens; struct { std::vector<llama_token> labels; std::vector<int32_t> markers; int32_t pointer=-1, column=0; } decision; };
enum server_decision_question_type {
    SERVER_DECISION_QUESTION_CHOICE,
    SERVER_DECISION_QUESTION_SCORE,
    SERVER_DECISION_QUESTION_NOUL,
};

struct server_decision_option {
    std::string key;
    json description; // null if not provided
};

struct server_decision_question {
    std::string id;
    server_decision_question_type type;
    json instructions;
    std::vector<server_decision_option> options; // in the order of the model outputs
};

struct text_decision_context {
    common_decision_type type = COMMON_DECISION_TYPE_NONE;

    // read the "<arch>.decision.*" metadata, type stays NONE if the model has none
    void init(const llama_model * model);
    // Private fixture configuration for parser/calibration tests without weights.
    void fixture(common_decision_type kind, const json & calibration) {
        type = kind; n_options_max = 255; noul_true_first = kind == COMMON_DECISION_TYPE_OPENJEV;
        for (const auto & [key, value] : calibration.items()) temperatures[key] = value.get<float>();
    }

    // true if the questions of a request start with the same tokens, and the model can continue from them
    bool can_share_prompt() const {
        switch (type) {
            case COMMON_DECISION_TYPE_OPENJEV:
            case COMMON_DECISION_TYPE_LEV:
            case COMMON_DECISION_TYPE_KEV:
            case COMMON_DECISION_TYPE_NIMBLE:
                return true;
            default:
                return false;
        }
    }

    // true if all the questions of a request go in one prompt, see fill_task_joint()
    bool is_joint() const {
        return type == COMMON_DECISION_TYPE_CLEF;
    }

    // true if the prompt of the model has a place for images
    // TODO: clef needs token and embedding entries in the same batch, see https://github.com/ggml-org/llama.cpp/pull/29622
    bool can_use_images() const {
        switch (type) {
            case COMMON_DECISION_TYPE_OPENJEV:
                return true;
            default:
                return false;
        }
    }

    // throw std::invalid_argument on bad input
    std::vector<server_decision_question> parse_questions(const json & body) const;

    // returns the state without its images, they are appended to files in order
    // images come from "images" and from the image_url parts of a state made of chat messages

    // number of prompts that are evaluated to answer this question, each one shows the options in a different order
    size_t n_variants(const server_decision_question & question) const;

    // set the prompt of one variant of this question, and where to read its result
    // mctx is only used if there are files
    void fill_task(
            const json & state,
            const std::vector<server_decision_question> & questions,
            const server_decision_question & question,
            size_t variant,
            text_decision_task & task) const;

    // set the prompt of all the questions, the result has the scores of all their options, in order

    // scores: the raw model outputs of each variant
    json format_answer(const server_decision_question & question, const std::vector<std::vector<float>> & scores) const;

private:
    const llama_vocab * vocab = nullptr;
    std::shared_ptr<const common_chat_template> tmpl; // the "systemone" template

    std::map<std::string, float> temperatures; // "<type>" or "<type>.<n_options bucket>"
    size_t n_options_max   = 0;
    bool   noul_true_first = false; // noul options are [true, false] instead of [false, true]
    bool   choice_sorted   = false; // choice options are in the order of their keys

    // OPENJEV, LEV, NIMBLE
    std::vector<llama_token> labels;
    std::vector<std::string> label_texts; // only if the label of an option is given to the template

    // LAYA, KEV
    llama_token token_marker      = LLAMA_TOKEN_NULL;
    llama_token token_sep         = LLAMA_TOKEN_NULL;
    std::string text_marker;
    size_t      max_head_tokens   = 0; // question + options
    size_t      max_option_tokens = 48;

    std::string render(
            const json & state,
            const std::vector<server_decision_question> & questions,
            const server_decision_question & question,
            size_t variant,
            size_t n_images) const;
    json render_options(const server_decision_question & question, size_t variant) const;
    size_t n_outputs(const server_decision_question & question) const;
    void fill_task_laya(llama_tokens & tokens, const server_decision_question & question, text_decision_task & task) const;

    float get_temperature(const server_decision_question & question) const;
};

static const char * decision_question_type_name(server_decision_question_type type) {
    switch (type) {
        case SERVER_DECISION_QUESTION_CHOICE: return "choice";
        case SERVER_DECISION_QUESTION_SCORE:  return "score";
        case SERVER_DECISION_QUESTION_NOUL:   return "noul";
    }
    return "";
}

// lev reads noul from a rating scale: 0 = certainly no, 8 = certainly yes
static const size_t DECISION_LEV_N_RATINGS = 9;

static std::string decision_meta_str(const llama_model * model, const std::string & key) {
    char buf[256];
    const int32_t n = llama_model_meta_val_str(model, key.c_str(), buf, sizeof(buf));
    return n < 0 ? "" : std::string(buf);
}

//
// model-specific setup
//

void text_decision_context::init(const llama_model * model) {
    *this = text_decision_context(); // the model can be reloaded

    const common_decision_type model_type = common_get_decision_type(model);
    if (model_type == COMMON_DECISION_TYPE_NONE) {
        return;
    }

    const std::string prefix    = decision_meta_str(model, "general.architecture") + ".decision.";
    const std::string type_name = decision_meta_str(model, prefix + "type");

    vocab = llama_model_get_vocab(model);

    const char * tmpl_src = llama_model_chat_template(model, "systemone");
    if (tmpl_src == nullptr) {
        throw std::runtime_error("decision model has no \"systemone\" template");
    }
    tmpl = std::make_shared<const common_chat_template>(tmpl_src, "", "");

    const std::string prefix_temp = prefix + "temperature.";
    for (int32_t i = 0; i < llama_model_meta_count(model); i++) {
        char key[256];
        char val[64];
        if (llama_model_meta_key_by_index(model, i, key, sizeof(key)) < 0 || !string_starts_with(key, prefix_temp)) {
            continue;
        }
        if (llama_model_meta_val_str_by_index(model, i, val, sizeof(val)) < 0) {
            continue;
        }
        const float temp = std::strtof(val, nullptr);
        if (temp <= 0.0f) {
            throw std::runtime_error(string_format("invalid decision temperature: %s = %s", key, val));
        }
        temperatures[key + prefix_temp.size()] = temp;
    }

    if (model_type == COMMON_DECISION_TYPE_OPENJEV) {
        // one letter per option, each must be a single token
        const std::string letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
        for (const char c : letters) {
            const auto toks = common_tokenize(vocab, std::string(1, c), false, false);
            if (toks.size() != 1) {
                throw std::runtime_error(string_format("decision label '%c' is not a single token", c));
            }
            labels.push_back(toks[0]);
        }
        n_options_max   = labels.size();
        noul_true_first = true;
    } else if (model_type == COMMON_DECISION_TYPE_LEV || model_type == COMMON_DECISION_TYPE_NIMBLE) {
        // label codes are A..Z then AA..ZZ, only the ones that are a single token are used
        std::vector<std::string> codes;
        for (char a = 'A'; a <= 'Z'; a++) {
            codes.push_back(std::string(1, a));
        }
        for (char a = 'A'; a <= 'Z'; a++) {
            for (char b = 'A'; b <= 'Z'; b++) {
                codes.push_back(std::string{a, b});
            }
        }
        for (const auto & code : codes) {
            const auto toks = common_tokenize(vocab, code, false, false);
            if (toks.size() == 1 && labels.size() < 255) {
                labels.push_back(toks[0]);
                label_texts.push_back(code);
            }
        }
        n_options_max = labels.size();
    } else if (model_type == COMMON_DECISION_TYPE_KEV) {
        // the hidden state of an option is read at the token that ends it
        const auto toks = common_tokenize(vocab, "<|box_end|>", false, true);
        if (toks.size() != 1) {
            throw std::runtime_error("decision model has no <|box_end|> token");
        }
        token_marker  = toks[0];
        n_options_max = 255;
    } else if (model_type == COMMON_DECISION_TYPE_LAYA) {
        token_marker = llama_vocab_mask(vocab);
        token_sep    = llama_vocab_sep(vocab);
        if (token_marker == LLAMA_TOKEN_NULL || token_sep == LLAMA_TOKEN_NULL) {
            throw std::runtime_error("decision model has no mask or sep token");
        }
        text_marker = common_token_to_piece(vocab, token_marker, true);

        const std::string val = decision_meta_str(model, prefix + "max_head_tokens");
        max_head_tokens = std::strtoul(val.c_str(), nullptr, 10);
        if (max_head_tokens == 0) {
            throw std::runtime_error("decision model has no valid max_head_tokens");
        }
        n_options_max = 255;
    } else if (model_type == COMMON_DECISION_TYPE_CLEF) {
        n_options_max   = 255;
        noul_true_first = true;
        choice_sorted   = true;
    } else {
        throw std::runtime_error("unsupported decision model type: " + type_name);
    }
    type = model_type;

}

//
// request parsing
//

std::vector<server_decision_question> text_decision_context::parse_questions(const json & body) const {
    if (!body.contains("state") || body.at("state").is_null()) {
        throw std::invalid_argument("\"state\" must be provided");
    }
    if (!body.contains("questions") || !body.at("questions").is_object() || body.at("questions").empty()) {
        throw std::invalid_argument("\"questions\" must be a non-empty object");
    }

    std::vector<server_decision_question> questions;
    for (const auto & [id, q] : body.at("questions").items()) {
        auto err = [&id = id](const std::string & msg) {
            return std::invalid_argument("questions." + id + ": " + msg);
        };
        if (!q.is_object()) {
            throw err("must be an object");
        }
        if (!q.contains("instructions") || q.at("instructions").is_null()) {
            throw err("\"instructions\" must be provided");
        }

        server_decision_question question;
        question.id           = id;
        question.instructions = q.at("instructions");

        const std::string type_name = q.value("type", std::string());
        const json        criteria  = q.contains("criteria") ? q.at("criteria") : json();

        if (type_name == "choice") {
            question.type = SERVER_DECISION_QUESTION_CHOICE;
            if (!criteria.is_object() || criteria.empty()) {
                throw err("\"criteria\" must be a non-empty object");
            }
            for (const auto & [key, description] : criteria.items()) {
                question.options.push_back({key, description});
            }
            if (choice_sorted) {
                std::sort(question.options.begin(), question.options.end(), [](const auto & a, const auto & b) {
                    return a.key < b.key;
                });
            }
        } else if (type_name == "score") {
            question.type = SERVER_DECISION_QUESTION_SCORE;
            if (!criteria.is_array() || criteria.size() < 2 || criteria.size() > 10) {
                throw err("\"criteria\" must be an array of 2 to 10 levels");
            }
            for (size_t i = 0; i < criteria.size(); i++) {
                question.options.push_back({std::to_string(i), criteria.at(i)});
            }
        } else if (type_name == "noul") {
            question.type = SERVER_DECISION_QUESTION_NOUL;
            if (!criteria.is_null() && !criteria.is_object()) {
                throw err("\"criteria\" must be an object");
            }
            for (const char * key : {"false", "true"}) {
                question.options.push_back({key, criteria.is_object() && criteria.contains(key) ? criteria.at(key) : json()});
            }
            if (noul_true_first) {
                std::swap(question.options[0], question.options[1]);
            }
        } else {
            throw err("\"type\" must be one of: choice, score, noul");
        }

        if (question.options.size() > n_options_max) {
            throw err(string_format("too many options (%zu), this model supports at most %zu", question.options.size(), n_options_max));
        }

        questions.push_back(std::move(question));
    }
    return questions;
}

//
// images
//

//
// prompt
//

// replace text in all strings of a JSON value
static json decision_replace_text(const json & val, const std::string & search, const std::string & replace) {
    if (val.is_string()) {
        std::string str = val.get<std::string>();
        string_replace_all(str, search, replace);
        return str;
    }
    if (val.is_array()) {
        json out = json::array();
        for (const auto & item : val) {
            out.push_back(decision_replace_text(item, search, replace));
        }
        return out;
    }
    if (val.is_object()) {
        json out = json::object();
        for (const auto & [key, item] : val.items()) {
            out[key] = decision_replace_text(item, search, replace);
        }
        return out;
    }
    return val;
}

// sort the keys of all objects of a JSON value
static json decision_sort_keys(const json & val) {
    if (val.is_array()) {
        json out = json::array();
        for (const auto & item : val) {
            out.push_back(decision_sort_keys(item));
        }
        return out;
    }
    if (val.is_object()) {
        std::map<std::string, json> sorted;
        for (const auto & [key, item] : val.items()) {
            sorted[key] = decision_sort_keys(item);
        }
        json out = json::object();
        for (const auto & [key, item] : sorted) {
            out[key] = item;
        }
        return out;
    }
    return val;
}

// kev flattens a JSON value into text, the keys of an object are kept as labels (kev/api.py: render)
static std::string decision_kev_render(const json & val, int indent = 0) {
    const std::string pad(2 * indent, ' ');
    if (val.is_null()) {
        return "";
    }
    if (val.is_string()) {
        return val.get<std::string>();
    }
    if (val.is_boolean()) {
        return val.get<bool>() ? "True" : "False";
    }
    if (val.is_array()) {
        std::string out;
        for (const auto & item : val) {
            const std::string text = decision_kev_render(item, indent + 1);
            out += (out.empty() ? "" : "\n") + pad + "- " + text.substr(std::min(text.size(), text.find_first_not_of(" \t\n\r")));
        }
        return out;
    }
    if (val.is_object()) {
        std::string out;
        for (const auto & [key, item] : val.items()) {
            const bool is_nested = item.is_object() || item.is_array();
            out += (out.empty() ? "" : "\n") + pad + key + (is_nested ? ":\n" : ": ") + decision_kev_render(item, is_nested ? indent + 1 : 0);
        }
        return out;
    }
    return val.dump();
}

// kev text input: special tokens written in the text must not be parsed as such
static std::string decision_kev_text(const json & val) {
    static const std::regex re_special("<\\|([A-Za-z0-9_]+)\\|>");
    return std::regex_replace(decision_kev_render(val), re_special, "<\xC2\xA6$1\xC2\xA6>");
}

size_t text_decision_context::n_variants(const server_decision_question & question) const {
    // lev shows the options of a choice in 2 orders, to cancel the preference for the first label
    if (type == COMMON_DECISION_TYPE_LEV && question.type == SERVER_DECISION_QUESTION_CHOICE && question.options.size() > 1) {
        return 2;
    }
    return 1;
}

size_t text_decision_context::n_outputs(const server_decision_question & question) const {
    if (type == COMMON_DECISION_TYPE_LEV && question.type == SERVER_DECISION_QUESTION_NOUL) {
        return DECISION_LEV_N_RATINGS;
    }
    return question.options.size();
}

json text_decision_context::render_options(const server_decision_question & question, size_t variant) const {
    const size_t n_options = question.options.size();

    // the second variant shows the options in the reverse order
    json options = json::array();
    for (size_t i = 0; i < n_options; i++) {
        const auto & opt = question.options[variant == 0 ? i : n_options - 1 - i];
        json option = json{
            {"key",         opt.key},
            {"description", opt.description},
        };
        if (type == COMMON_DECISION_TYPE_KEV) {
            option["key"] = decision_kev_text(opt.key);
            if (!opt.description.is_null()) {
                option["description"] = decision_kev_text(opt.description);
            }
        }
        if (!label_texts.empty()) {
            option["label"] = label_texts[i];
        }
        options.push_back(option);
    }
    return options;
}

std::string text_decision_context::render(
        const json & state,
        const std::vector<server_decision_question> & questions,
        const server_decision_question & question,
        size_t variant,
        size_t n_images) const {
    // the template is given raw JSON values, it serializes the ones that are not strings
    json inp = json{
        {"id",           question.id},
        {"type",         decision_question_type_name(question.type)},
        {"instructions", question.instructions},
        {"state",        state},
        {"options",      render_options(question, variant)},
    };

    // the nimble prompt lists all the questions of the request
    if (type == COMMON_DECISION_TYPE_NIMBLE) {
        inp["questions"] = json::array();
        for (const auto & q : questions) {
            inp["questions"].push_back(json{
                {"id",           q.id},
                {"type",         decision_question_type_name(q.type)},
                {"instructions", q.instructions},
                {"options",      render_options(q, 0)},
            });
        }
    }

    // lev was trained with sorted keys
    if (type == COMMON_DECISION_TYPE_LEV) {
        inp = decision_sort_keys(inp);
    }

    // the kev template only takes text
    if (type == COMMON_DECISION_TYPE_KEV) {
        inp["state"]        = decision_kev_text(state);
        inp["instructions"] = decision_kev_text(question.instructions);
    }

    // the input must not contain the marker of the options
    if (!text_marker.empty()) {
        inp = decision_replace_text(inp, text_marker, " ");
    }

    // the template puts one media marker per image
    json images = json::array();
    inp["images"] = images;

    jinja::context ctx(tmpl->source());
    jinja::global_from_json(ctx, inp, false);
    jinja::runtime runtime(ctx);
    const jinja::value results = runtime.execute(tmpl->prog);
    return jinja::runtime::gather_string_parts(results)->as_string().str();
}

void text_decision_context::fill_task(
        const json & state,
        const std::vector<server_decision_question> & questions,
        const server_decision_question & question,
        size_t variant,
        text_decision_task & task) const {
    const std::string prompt = render(state, questions, question, variant, 0);

    if (type == COMMON_DECISION_TYPE_OPENJEV || type == COMMON_DECISION_TYPE_LEV || type == COMMON_DECISION_TYPE_NIMBLE) {
        // lev reads the ratings of a noul question at its first labels, not at the digits
        task.decision.labels.assign(labels.begin(), labels.begin() + n_outputs(question));
    }

    llama_tokens tokens = common_tokenize(vocab, prompt, false, true);
    if (type == COMMON_DECISION_TYPE_LAYA) {
        fill_task_laya(tokens, question, task);
    }
    if (type == COMMON_DECISION_TYPE_KEV) {
        // an option is read at its end token, the question at the last token
        for (size_t i = 0; i < tokens.size(); i++) {
            if (tokens[i] == token_marker) {
                task.decision.markers.push_back(i);
            }
        }
        if (task.decision.markers.size() != question.options.size()) {
            throw std::runtime_error("unexpected layout of the decision prompt");
        }
        task.decision.pointer = tokens.size() - 1;
    }
    task.tokens = tokens;
}

// the prompt is: [cls] question [sep] ([marker] option)* [sep] state [sep]
// options and question are cut to fit max_head_tokens, the same way the model was trained
void text_decision_context::fill_task_laya(llama_tokens & tokens, const server_decision_question & question, text_decision_task & task) const {
    const size_t n_options = question.options.size();

    std::vector<size_t> markers;
    for (size_t i = 0; i < tokens.size(); i++) {
        if (tokens[i] == token_marker) {
            markers.push_back(i);
        }
    }
    const auto invalid = std::runtime_error("unexpected layout of the decision prompt");
    if (markers.size() != n_options || markers[0] < 2 || tokens[markers[0] - 1] != token_sep || tokens.back() != token_sep) {
        throw invalid;
    }
    const size_t head_end = markers[0] - 1;
    const size_t opts_end = std::find(tokens.begin() + markers.back(), tokens.end(), token_sep) - tokens.begin();
    if (opts_end + 1 >= tokens.size()) {
        throw invalid;
    }

    // marker + text of each option
    std::vector<llama_tokens> options;
    size_t n_options_tokens = 0;
    auto set_max = [&](size_t n_max) {
        n_options_tokens = 0;
        for (auto & opt : options) {
            opt.resize(std::min(opt.size(), n_max));
            n_options_tokens += opt.size();
        }
    };
    for (size_t i = 0; i < n_options; i++) {
        const size_t end = i + 1 < n_options ? markers[i + 1] : opts_end;
        options.emplace_back(tokens.begin() + markers[i], tokens.begin() + end);
    }
    set_max(max_option_tokens + 1);
    if (n_options_tokens + 16 > max_head_tokens) {
        // too many or too long options, shrink them evenly
        set_max(std::max((size_t) 4, (max_head_tokens - std::min(max_head_tokens, (size_t) 16)) / n_options));
    }
    const size_t n_question_max = std::max((size_t) 8, max_head_tokens - std::min(max_head_tokens, n_options_tokens));

    llama_tokens out;
    out.push_back(tokens[0]);
    out.insert(out.end(), tokens.begin() + 1, tokens.begin() + std::min(head_end, 1 + n_question_max));
    out.push_back(token_sep);
    for (const auto & opt : options) {
        task.decision.markers.push_back(out.size());
        out.insert(out.end(), opt.begin(), opt.end());
    }
    out.insert(out.end(), tokens.begin() + opts_end, tokens.end());
    tokens = std::move(out);

    // the output has one score per question type
    task.decision.column = question.type;
}

//
// joint prompt (clef)
//

// given to the template: text between the pieces of the prompt, and at the start of the span of a question or of an option
//
// answer
//

float text_decision_context::get_temperature(const server_decision_question & question) const {
    const size_t n = question.options.size();
    const std::string type_name = decision_question_type_name(question.type);

    // the temperature can depend on the number of options, the buckets are the ones used to fit it
    std::string bucket;
    if (type == COMMON_DECISION_TYPE_LEV) {
        bucket = n <= 8 ? "small" : n <= 26 ? "mid" : "large";
    } else {
        bucket = n <= 2 ? "2" : n <= 5 ? "3_5" : n <= 10 ? "6_10" : "11";
    }

    for (const auto & name : {type_name + "." + bucket, type_name}) {
        const auto it = temperatures.find(name);
        if (it != temperatures.end()) {
            return it->second;
        }
    }
    return 1.0f;
}

// confidence formulas are the ones published by TypeSafe

static double decision_confidence_choice(const std::vector<double> & probs) {
    if (probs.size() < 2) {
        return 1.0;
    }
    const double uniform = 1.0 / probs.size();
    const double p_max   = *std::max_element(probs.begin(), probs.end());
    return std::max(0.0, (p_max - uniform) / (1.0 - uniform));
}

static double decision_confidence_score(const std::vector<double> & probs) {
    if (probs.size() < 2) {
        return 1.0;
    }
    const size_t n    = probs.size();
    const size_t mode = std::max_element(probs.begin(), probs.end()) - probs.begin();

    // mean distance to the mode, relative to the one of a uniform distribution around its center
    double dist         = 0.0;
    double dist_uniform = 0.0;
    for (size_t i = 0; i < n; i++) {
        dist         += probs[i] * std::fabs((double) i - (double) mode);
        dist_uniform += std::fabs((double) i - (n - 1) / 2.0) / n;
    }
    return std::max(0.0, 1.0 - dist / dist_uniform);
}

json text_decision_context::format_answer(const server_decision_question & question, const std::vector<std::vector<float>> & scores) const {
    const size_t n = n_outputs(question);
    if (scores.size() != n_variants(question)) {
        throw std::runtime_error("decision result does not match the number of variants");
    }

    // softmax over the outputs of each variant, then the average of the variants
    const float temperature = get_temperature(question);
    std::vector<double> probs(n, 0.0);
    for (size_t v = 0; v < scores.size(); v++) {
        const auto & s = scores[v];
        if (s.size() != n) {
            throw std::runtime_error("decision result does not match the number of options");
        }
        // a joint head returns NaN if it could not use the decision order
        if (std::any_of(s.begin(), s.end(), [](float v) { return std::isnan(v); })) {
            throw std::runtime_error("the model could not evaluate the decision");
        }
        const float score_max = *std::max_element(s.begin(), s.end());
        std::vector<double> p(n);
        double sum = 0.0;
        for (size_t i = 0; i < n; i++) {
            p[i] = std::exp((double) (s[i] - score_max) / temperature);
            sum += p[i];
        }
        for (size_t i = 0; i < n; i++) {
            // the second variant is in the reverse order
            probs[v == 0 ? i : n - 1 - i] += p[i] / sum / scores.size();
        }
    }

    json answer = json{{"type", decision_question_type_name(question.type)}};

    if (question.type == SERVER_DECISION_QUESTION_NOUL) {
        if (type == COMMON_DECISION_TYPE_LEV) {
            double expected = 0.0;
            for (size_t i = 0; i < n; i++) {
                expected += probs[i] * i / (n - 1);
            }
            answer["noul"] = expected;
            return answer;
        }
        for (size_t i = 0; i < n; i++) {
            if (question.options[i].key == "true") {
                answer["noul"] = probs[i];
            }
        }
        return answer;
    }

    json probabilities = json::object();
    for (size_t i = 0; i < n; i++) {
        probabilities[question.options[i].key] = probs[i];
    }

    if (question.type == SERVER_DECISION_QUESTION_CHOICE) {
        const size_t best = std::max_element(probs.begin(), probs.end()) - probs.begin();
        answer["choice"]        = question.options[best].key;
        answer["probabilities"] = probabilities;
        answer["confidence"]    = decision_confidence_choice(probs);
    } else {
        double expected = 0.0;
        json legend = json::object();
        for (size_t i = 0; i < n; i++) {
            expected += i * probs[i];
            legend[question.options[i].key] = question.options[i].description;
        }
        answer["score"]         = expected;
        answer["legend"]        = legend;
        answer["probabilities"] = probabilities;
        answer["confidence"]    = decision_confidence_score(probs);
    }
    return answer;
}

//
// shared prompt prefix
//


extern "C" int mlx_rs_decision_kind(const llama_model * model) {
    return model ? static_cast<int>(common_get_decision_type(model)) : COMMON_DECISION_TYPE_NONE;
}
extern "C" const char * mlx_rs_decision_family(const llama_model * model) {
    switch (common_get_decision_type(model)) {
        case COMMON_DECISION_TYPE_OPENJEV: return "openjev";
        case COMMON_DECISION_TYPE_LEV: return "lev";
        case COMMON_DECISION_TYPE_KEV: return "kev";
        case COMMON_DECISION_TYPE_NIMBLE: return "nimble";
        case COMMON_DECISION_TYPE_LAYA: return "laya";
        case COMMON_DECISION_TYPE_CLEF: return "clef";
        case COMMON_DECISION_TYPE_UNKNOWN: return "unknown";
        default: return nullptr;
    }
}
extern "C" int mlx_rs_clef_decide(const llama_model *, llama_context *, const char *, char **, char **) noexcept;
static char * decision_copy(const std::string & value) {
    char * result = static_cast<char *>(std::malloc(value.size()+1));
    if (!result) throw std::bad_alloc();
    std::memcpy(result, value.c_str(), value.size()+1);
    return result;
}
extern "C" int mlx_rs_decide(const llama_model * model, llama_context * ctx, const char * request, char ** output, char ** error) noexcept {
    if (!output || !error) return 3;
    *output = nullptr; *error = nullptr;
    if (!model || !ctx || !request) return 3;
    const auto kind = common_get_decision_type(model);
    if (kind == COMMON_DECISION_TYPE_CLEF) return mlx_rs_clef_decide(model, ctx, request, output, error);
    if (kind == COMMON_DECISION_TYPE_NONE || kind == COMMON_DECISION_TYPE_UNKNOWN) {
        try { *error = decision_copy("unsupported decision model family"); } catch (...) {}
        return 2;
    }
    try {
        text_decision_context adapter;
        adapter.init(model);
        const auto body = json::parse(request);
        const auto questions = adapter.parse_questions(body);
        // Build and validate every prompt before native inference starts.
        std::vector<std::vector<text_decision_task>> tasks(questions.size());
        size_t input_tokens = 0;
        for (size_t q = 0; q < questions.size(); ++q) {
            for (size_t v = 0; v < adapter.n_variants(questions[q]); ++v) {
                text_decision_task task;
                adapter.fill_task(body.at("state"), questions, questions[q], v, task);
                const auto n = task.tokens.size();
                if (!n || n > llama_n_ctx(ctx) || n > llama_n_batch(ctx) || n > llama_n_ubatch(ctx))
                    throw std::invalid_argument("decision prompt must fit context, n_batch and n_ubatch (" + std::to_string(n) + " tokens)");
                input_tokens += n;
                tasks[q].push_back(std::move(task));
            }
        }
        const auto memory = llama_get_memory(ctx);
        struct cleanup { llama_memory_t memory; ~cleanup() { if (memory) llama_memory_clear(memory, true); } } clear{memory};
        json answers = json::object();
        for (size_t q = 0; q < questions.size(); ++q) {
            std::vector<std::vector<float>> variants;
            for (const auto & task : tasks[q]) {
                if (memory) llama_memory_clear(memory, true);
                std::unique_ptr<llama_batch_ext, decltype(&llama_batch_ext_free)> batch(llama_batch_ext_init(ctx), llama_batch_ext_free);
                if (!batch) throw std::bad_alloc();
                const bool read_logits = !task.decision.labels.empty();
                for (size_t i = 0; i < task.tokens.size(); ++i) {
                    const int32_t idx = llama_batch_ext_add_token(batch.get(), 0, task.tokens[i]);
                    const llama_pos pos = static_cast<llama_pos>(i);
                    if (idx < 0 || !llama_batch_ext_set_pos(batch.get(), idx, &pos) ||
                        !llama_batch_ext_set_output_embd(batch.get(), idx, !read_logits) ||
                        !llama_batch_ext_set_output_logits(batch.get(), idx, read_logits && i+1 == task.tokens.size()))
                        throw std::runtime_error("failed to construct decision batch");
                }
                const int status = llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get());
                if (status != 0) throw std::runtime_error("decision inference failed: " + std::to_string(status));
                std::vector<float> scores;
                if (read_logits) {
                    const float * logits = llama_get_logits_ith(ctx, task.tokens.size()-1);
                    if (!logits) throw std::runtime_error("missing decision logits");
                    for (const auto label : task.decision.labels) scores.push_back(logits[label]);
                } else {
                    const int32_t dim = llama_model_n_embd_out(model);
                    const int32_t half = dim / 2;
                    const float * query = task.decision.pointer >= 0 ? llama_get_embeddings_ith(ctx, task.decision.pointer) : nullptr;
                    if (task.decision.pointer >= 0 && (!query || half == 0 || dim % 2)) throw std::runtime_error("invalid decision query embedding");
                    for (const auto marker : task.decision.markers) {
                        const float * option = llama_get_embeddings_ith(ctx, marker);
                        if (!option) throw std::runtime_error("missing decision option embedding");
                        if (query) {
                            float dot = 0.0f;
                            for (int32_t i = 0; i < half; ++i) dot += query[i] * option[half+i];
                            scores.push_back(dot / sqrtf(static_cast<float>(half)));
                        } else {
                            if (task.decision.column < 0 || task.decision.column >= dim) throw std::runtime_error("invalid decision output column");
                            scores.push_back(option[task.decision.column]);
                        }
                    }
                }
                if (std::any_of(scores.begin(), scores.end(), [](float v) { return !std::isfinite(v); })) throw std::runtime_error("non-finite decision scores");
                variants.push_back(std::move(scores));
            }
            answers[questions[q].id] = adapter.format_answer(questions[q], variants);
        }
        *output = decision_copy(json{{"answers", answers}, {"usage", json{{"input_tokens", input_tokens}, {"output_tokens", 0}}}}.dump());
        return 0;
    } catch (const std::exception & e) {
        try { *error = decision_copy(e.what()); } catch (...) {}
        return dynamic_cast<const std::invalid_argument *>(&e) || dynamic_cast<const common_json_error *>(&e) ? 1 : 3;
    } catch (...) { return 3; }
}

extern "C" int mlx_rs_decision_format_fixture(int kind, const char * request, const char * scores, const char * temperatures, char ** output) noexcept {
    *output = nullptr;
    try {
        text_decision_context adapter;
        adapter.fixture(static_cast<common_decision_type>(kind), json::parse(temperatures));
        const auto body = json::parse(request);
        const auto questions = adapter.parse_questions(body);
        const auto raw = json::parse(scores);
        json answers = json::object();
        for (const auto & q : questions) {
            std::vector<std::vector<float>> variants;
            for (const auto & row : raw.at(q.id)) {
                std::vector<float> values;
                for (const auto & value : row) values.push_back(value.get<float>());
                variants.push_back(std::move(values));
            }
            answers[q.id] = adapter.format_answer(q, variants);
        }
        *output = decision_copy(json{{"answers", answers}, {"kev_state", decision_kev_text(body.at("state"))}}.dump());
        return 0;
    } catch (...) { return 1; }
}

} // namespace mlx_decision
