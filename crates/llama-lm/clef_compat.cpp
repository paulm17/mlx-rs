// Text-only adapter derived from llama.cpp 99b95488cac0f00ce3f05af113a8c1e287753f87.
// See vendor/llama-cpp-sys-2/llama.cpp/LICENSE (MIT).
#include "common.h"
#include "chat.h"
#include "src/llama-ext.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
using json = common_json;

enum server_decision_question_type { SERVER_DECISION_QUESTION_CHOICE, SERVER_DECISION_QUESTION_SCORE, SERVER_DECISION_QUESTION_NOUL };
struct server_decision_option { std::string key; json description; };
struct server_decision_question { std::string id; server_decision_question_type type; json instructions; std::vector<server_decision_option> options; };
struct clef_task { std::vector<llama_token> tokens; struct { std::vector<int32_t> order; int n_scores = 0; } decision; };
struct server_decision_context {
    common_decision_type type = COMMON_DECISION_TYPE_CLEF;
    const llama_vocab * vocab;
    std::shared_ptr<const common_chat_template> tmpl;
    std::map<std::string, float> temperatures;
    size_t n_options_max = 255;
    bool noul_true_first = true, choice_sorted = true;
    server_decision_context() = default;
    explicit server_decision_context(const llama_model * model);
    std::vector<server_decision_question> parse_questions(const json & body) const;
    void fill_task_joint(const json & state, const std::vector<server_decision_question> & questions, clef_task & task) const;
    float get_temperature(const server_decision_question & question) const;
    size_t n_variants(const server_decision_question &) const { return 1; }
    size_t n_outputs(const server_decision_question & q) const { return q.options.size(); }
    json format_answer(const server_decision_question & question, const std::vector<std::vector<float>> & scores) const;
};
static const char * decision_question_type_name(server_decision_question_type type) {
    switch (type) {
        case SERVER_DECISION_QUESTION_CHOICE: return "choice";
        case SERVER_DECISION_QUESTION_SCORE:  return "score";
        case SERVER_DECISION_QUESTION_NOUL:   return "noul";
    }
    return "";
}

static std::string decision_meta_str(const llama_model * model, const std::string & key) {
    char buf[256];
    const int32_t n = llama_model_meta_val_str(model, key.c_str(), buf, sizeof(buf));
    return n < 0 ? "" : std::string(buf);
}

server_decision_context::server_decision_context(const llama_model * model) {
    if (common_get_decision_type(model) != COMMON_DECISION_TYPE_CLEF) throw std::invalid_argument("model does not support Clef decisions");
    vocab = llama_model_get_vocab(model);
    const char * source = llama_model_chat_template(model, "systemone");
    if (!source) throw std::runtime_error("decision model has no systemone template");
    tmpl = std::make_shared<const common_chat_template>(source, "", "");
    const std::string prefix = decision_meta_str(model, "general.architecture") + ".decision.temperature.";
    for (int32_t i = 0; i < llama_model_meta_count(model); ++i) {
        char key[256], value[64];
        if (llama_model_meta_key_by_index(model, i, key, sizeof(key)) < 0 || !string_starts_with(key, prefix)) continue;
        if (llama_model_meta_val_str_by_index(model, i, value, sizeof(value)) < 0) continue;
        const float temp = std::strtof(value, nullptr);
        if (!std::isfinite(temp) || temp <= 0) throw std::runtime_error("invalid decision temperature");
        temperatures[key + prefix.size()] = temp;
    }
}
std::vector<server_decision_question> server_decision_context::parse_questions(const json & body) const {
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
static const std::string CLEF_MARKER        = "<<clef:";
static const std::string CLEF_SEP           = "<<clef:sep>>";
static const std::string CLEF_MARK_QUESTION = "<<clef:question>>";
static const std::string CLEF_MARK_OPTION   = "<<clef:option>>";

void server_decision_context::fill_task_joint(const json & state, const std::vector<server_decision_question> & questions, clef_task & task) const {
    json inp_questions = json::array();
    for (const auto & question : questions) {
        json options = json::array();
        for (const auto & opt : question.options) {
            options.push_back(json{
                {"key",         opt.key},
                {"description", opt.description},
            });
        }
        inp_questions.push_back(json{
            {"id",           question.id},
            {"type",         decision_question_type_name(question.type)},
            {"instructions", question.instructions},
            {"options",      options},
        });
    }

    // the template is given raw JSON values with sorted keys, and no marker in the input
    json inp = json{
        {"state",     state},
        {"questions", inp_questions},
    };
    inp = decision_replace_text(decision_sort_keys(inp), CLEF_MARKER, "<<clef ");
    inp["sep"]           = CLEF_SEP;
    inp["mark_question"] = CLEF_MARK_QUESTION;
    inp["mark_option"]   = CLEF_MARK_OPTION;

    jinja::context ctx(tmpl->source());
    jinja::global_from_json(ctx, inp, false);
    jinja::runtime runtime(ctx);
    const jinja::value results = runtime.execute(tmpl->prog);
    const std::string prompt   = jinja::runtime::gather_string_parts(results)->as_string().str();

    // the model was trained with the pieces tokenized one by one
    llama_tokens tokens;
    size_t i_question = 0;
    for (std::string piece : string_split(prompt, CLEF_SEP)) {
        int32_t order = LLAMA_DECISION_ORDER_NONE;
        if (string_starts_with(piece, CLEF_MARK_QUESTION)) {
            piece = piece.substr(CLEF_MARK_QUESTION.size());
            if (i_question >= questions.size()) {
                throw std::runtime_error("unexpected layout of the decision prompt");
            }
            switch (questions[i_question++].type) {
                case SERVER_DECISION_QUESTION_NOUL:   order = LLAMA_DECISION_ORDER_QUESTION_NOUL;   break;
                case SERVER_DECISION_QUESTION_CHOICE: order = LLAMA_DECISION_ORDER_QUESTION_CHOICE; break;
                case SERVER_DECISION_QUESTION_SCORE:  order = LLAMA_DECISION_ORDER_QUESTION_SCORE;  break;
            }
        } else if (string_starts_with(piece, CLEF_MARK_OPTION)) {
            piece = piece.substr(CLEF_MARK_OPTION.size());
            order = LLAMA_DECISION_ORDER_OPTION;
            task.decision.n_scores++;
        }

        const llama_tokens piece_tokens = common_tokenize(vocab, piece, false, true);
        if (order != LLAMA_DECISION_ORDER_NONE && piece_tokens.empty()) {
            throw std::invalid_argument("the instructions and the options of a question must not be empty");
        }
        tokens.insert(tokens.end(), piece_tokens.begin(), piece_tokens.end());
        task.decision.order.resize(tokens.size(), order);
    }

    size_t n_options = 0;
    for (const auto & question : questions) {
        n_options += question.options.size();
    }
    if (i_question != questions.size() || (size_t) task.decision.n_scores != n_options) {
        throw std::runtime_error("unexpected layout of the decision prompt");
    }

    task.tokens = std::move(tokens);
}

float server_decision_context::get_temperature(const server_decision_question & question) const {
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

json server_decision_context::format_answer(const server_decision_question & question, const std::vector<std::vector<float>> & scores) const {
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


static char * clef_copy(const std::string & s) {
    char * p = static_cast<char *>(std::malloc(s.size()+1));
    if (!p) throw std::bad_alloc();
    std::memcpy(p, s.c_str(), s.size()+1); return p;
}
extern "C" void mlx_rs_decision_free(char * p) { std::free(p); }
extern "C" bool mlx_rs_supports_clef(const llama_model * model) { return model && common_get_decision_type(model) == COMMON_DECISION_TYPE_CLEF; }
// 1 invalid request, 2 unsupported, 3 runtime error. Both outputs owned by caller.
extern "C" int mlx_rs_clef_decide(const llama_model * model, llama_context * ctx, const char * request, char ** output, char ** error) noexcept {
    if (!output || !error) return 3;
    *output = nullptr; *error = nullptr;
    try {
        if (!model || !ctx || !request) throw std::invalid_argument("missing decision input");
        if (!mlx_rs_supports_clef(model)) return 2;
        server_decision_context decision(model);
        const auto body = json::parse(request);
        const auto questions = decision.parse_questions(body);
        clef_task task; decision.fill_task_joint(body.at("state"), questions, task);
        const size_t n = task.tokens.size();
        if (!n || n > llama_n_ctx(ctx) || n > llama_n_batch(ctx) || n > llama_n_ubatch(ctx))
            throw std::invalid_argument("joint decision prompt must fit context, n_batch and n_ubatch (" + std::to_string(n) + " tokens)");
        if (task.decision.n_scores > static_cast<int>(n)) throw std::runtime_error("invalid decision output count");
        const auto memory = llama_get_memory(ctx);
        if (memory) llama_memory_clear(memory, true);
        // Clear context on success and failure; requests never share a prefix.
        struct cleanup { llama_memory_t memory; ~cleanup() { if (memory) llama_memory_clear(memory, true); } } clear{memory};
        std::unique_ptr<llama_batch_ext, decltype(&llama_batch_ext_free)> batch(llama_batch_ext_init(ctx), llama_batch_ext_free);
        if (!batch) throw std::bad_alloc();
        for (size_t i=0; i<n; ++i) {
            int32_t idx = llama_batch_ext_add_token(batch.get(), 0, task.tokens[i]);
            llama_pos pos = static_cast<llama_pos>(i);
            if (idx < 0 || !llama_batch_ext_set_pos(batch.get(), idx, &pos) ||
                !llama_batch_ext_set_output_embd(batch.get(), idx, true) ||
                !llama_batch_ext_set_decision_order(batch.get(), idx, static_cast<llama_decision_order>(task.decision.order[i])))
                throw std::runtime_error("failed to construct joint decision batch");
        }
        const int status = llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get());
        if (status != 0) throw std::runtime_error("decision inference failed: " + std::to_string(status));
        std::vector<float> scores;
        for (int32_t i=0; i<task.decision.n_scores; ++i) {
            const float * row = llama_get_embeddings_ith(ctx, i);
            if (!row || !std::isfinite(row[0])) throw std::runtime_error("missing or non-finite decision output");
            scores.push_back(row[0]);
        }
        json answers = json::object(); size_t offset = 0;
        for (const auto & q : questions) {
            const size_t count = q.options.size();
            const std::vector<float> slice(scores.begin()+offset, scores.begin()+offset+count);
            answers[q.id] = decision.format_answer(q, {slice}); offset += count;
        }
        json result = {{"answers", answers}, {"usage", json{{"input_tokens", n}, {"output_tokens", 0}}}};
        *output = clef_copy(result.dump()); return 0;
    } catch (const std::exception & e) {
        try { *error = clef_copy(e.what()); } catch (...) {}
        return dynamic_cast<const std::invalid_argument *>(&e) || dynamic_cast<const common_json_error *>(&e) ? 1 : 3;
    } catch (...) { return 3; }
}

// Private native fixture entrypoint, exercised from Rust unit tests without weights.
extern "C" int mlx_rs_clef_format_fixture(const char * request, const char * scores_json, const char * temperatures_json, char ** output) noexcept {
    *output = nullptr;
    try {
        server_decision_context decision;
        const auto temperatures = json::parse(temperatures_json);
        for (const auto & [key, value] : temperatures.items()) decision.temperatures[key] = value.get<float>();
        const auto questions = decision.parse_questions(json::parse(request));
        const auto scores = json::parse(scores_json);
        json answers = json::object();
        for (const auto & q : questions) {
            std::vector<float> values;
            for (const auto & value : scores.at(q.id)) values.push_back(value.get<float>());
            answers[q.id] = decision.format_answer(q, {values});
        }
        *output = clef_copy(answers.dump()); return 0;
    } catch (...) { return 1; }
}
