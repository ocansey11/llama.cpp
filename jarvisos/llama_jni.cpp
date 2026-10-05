// JarvisOS JNI bridge to llama.cpp, for com.android.server.jarvis.inference.LlamaEngine.
//
// Mirrors what vendor/cactus/jarvisos/jarvis_jni.cpp gives CactusWrapper:
// init / destroy, embed, complete, last error. complete() returns the same
// JSON envelope Cactus does ({"success", "error", "response",
// "function_calls":[{"name","arguments"}], ...}) so the Java side reads both
// engines with one parser.
//
// Runs inside system_server: nothing here opens a network connection or
// starts a process, and every entry point catches C++ exceptions, because an
// exception crossing JNI would take system_server down.

#include <jni.h>

#include <android/log.h>

#include <chrono>
#include <mutex>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "chat.h"
#include "common.h"
#include "json.h"
#include "llama.h"

#define TAG "LlamaJni"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN, TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

using json = nlohmann::ordered_json;

namespace {

struct Engine {
    llama_model* model = nullptr;
    llama_context* ctx = nullptr;
    common_chat_templates_ptr templates;
    bool embedding = false;
    // One call at a time per model: a llama_context is not thread-safe.
    std::mutex lock;
};

std::mutex g_error_lock;
std::string g_last_error;

void set_error(const std::string& message) {
    LOGE("%s", message.c_str());
    std::lock_guard<std::mutex> guard(g_error_lock);
    g_last_error = message;
}

std::string to_string(JNIEnv* env, jstring s) {
    if (s == nullptr) return std::string();
    const char* chars = env->GetStringUTFChars(s, nullptr);
    std::string out(chars ? chars : "");
    if (chars) env->ReleaseStringUTFChars(s, chars);
    return out;
}

// NewStringUTF wants modified UTF-8 and aborts on anything else; model output
// is ordinary UTF-8 and may be cut mid-character. Go through a byte array.
jstring to_jstring(JNIEnv* env, const std::string& s) {
    jbyteArray bytes = env->NewByteArray(static_cast<jsize>(s.size()));
    if (bytes == nullptr) return nullptr;
    env->SetByteArrayRegion(bytes, 0, static_cast<jsize>(s.size()),
            reinterpret_cast<const jbyte*>(s.data()));
    jclass string_class = env->FindClass("java/lang/String");
    jmethodID ctor = env->GetMethodID(string_class, "<init>", "([BLjava/lang/String;)V");
    jstring charset = env->NewStringUTF("UTF-8");
    jstring out = static_cast<jstring>(env->NewObject(string_class, ctor, bytes, charset));
    env->DeleteLocalRef(bytes);
    env->DeleteLocalRef(charset);
    env->DeleteLocalRef(string_class);
    return out;
}

void log_callback(enum ggml_log_level level, const char* text, void*) {
    // llama.cpp logs every tensor at load; only pass on what matters.
    if (level == GGML_LOG_LEVEL_ERROR) {
        __android_log_print(ANDROID_LOG_ERROR, TAG, "%s", text);
    } else if (level == GGML_LOG_LEVEL_WARN) {
        __android_log_print(ANDROID_LOG_WARN, TAG, "%s", text);
    }
}

std::string error_envelope(const std::string& message) {
    json out;
    out["success"] = false;
    out["error"] = message;
    out["response"] = "";
    out["function_calls"] = json::array();
    return out.dump(-1, ' ', false, json::error_handler_t::replace);
}

// Jarvis sends tools as [{name, description, parameters}]; OpenAI clients wrap
// each one as {type:"function", function:{...}}. Accept both.
std::vector<common_chat_tool> parse_tools(const std::string& tools_json) {
    std::vector<common_chat_tool> tools;
    if (tools_json.empty()) return tools;
    json parsed = json::parse(tools_json);
    if (!parsed.is_array()) return tools;
    for (const auto& item : parsed) {
        const json& fn = item.contains("function") ? item.at("function") : item;
        common_chat_tool tool;
        tool.name = fn.value("name", "");
        tool.description = fn.value("description", "");
        tool.parameters = fn.contains("parameters")
                ? fn.at("parameters").dump()
                : std::string("{\"type\":\"object\",\"properties\":{}}");
        if (!tool.name.empty()) tools.push_back(std::move(tool));
    }
    return tools;
}

// Feeds tokens to the model in batch-sized pieces. Returns false on failure.
bool decode_all(Engine* engine, std::vector<llama_token>& tokens) {
    const int n_batch = static_cast<int>(llama_n_batch(engine->ctx));
    for (size_t i = 0; i < tokens.size(); i += n_batch) {
        const int n = static_cast<int>(std::min(tokens.size() - i, static_cast<size_t>(n_batch)));
        if (llama_decode(engine->ctx, llama_batch_get_one(tokens.data() + i, n)) != 0) {
            return false;
        }
    }
    return true;
}

double ms_since(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - start).count();
}

std::string complete(Engine* engine, const std::string& messages_json,
        const std::string& options_json, const std::string& tools_json) {
    if (engine->embedding) return error_envelope("model was loaded for embeddings only");

    int max_tokens = 512;
    float temperature = 0.0f;
    if (!options_json.empty()) {
        json options = json::parse(options_json);
        max_tokens = options.value("max_tokens", max_tokens);
        temperature = options.value("temperature", temperature);
    }

    common_chat_templates_inputs inputs;
    inputs.messages = common_chat_msgs_parse_oaicompat(common_json::parse(messages_json));
    inputs.tools = parse_tools(tools_json);
    inputs.add_generation_prompt = true;
    inputs.use_jinja = true;
    // Answers, not visible reasoning: small models spend their whole budget
    // thinking otherwise. Anything that still comes out is split off below.
    inputs.enable_thinking = false;
    inputs.reasoning_format = COMMON_REASONING_FORMAT_DEEPSEEK;
    const common_chat_params chat = common_chat_templates_apply(engine->templates.get(), inputs);

    const llama_vocab* vocab = llama_model_get_vocab(engine->model);
    std::vector<llama_token> prompt = common_tokenize(vocab, chat.prompt, true, true);
    const int n_ctx = static_cast<int>(llama_n_ctx(engine->ctx));
    if (static_cast<int>(prompt.size()) + 16 >= n_ctx) {
        return error_envelope("prompt is " + std::to_string(prompt.size())
                + " tokens, context is " + std::to_string(n_ctx));
    }
    max_tokens = std::min(max_tokens, n_ctx - static_cast<int>(prompt.size()) - 1);

    // Each request stands alone; the service keeps the conversation.
    llama_memory_clear(llama_get_memory(engine->ctx), true);

    const auto start = std::chrono::steady_clock::now();
    if (!decode_all(engine, prompt)) return error_envelope("llama_decode failed on the prompt");
    const double prefill_ms = ms_since(start);

    llama_sampler* sampler = llama_sampler_chain_init(llama_sampler_chain_default_params());
    if (temperature <= 0.0f) {
        llama_sampler_chain_add(sampler, llama_sampler_init_greedy());
    } else {
        llama_sampler_chain_add(sampler, llama_sampler_init_top_k(40));
        llama_sampler_chain_add(sampler, llama_sampler_init_top_p(0.95f, 1));
        llama_sampler_chain_add(sampler, llama_sampler_init_temp(temperature));
        llama_sampler_chain_add(sampler, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));
    }

    std::string text;
    int generated = 0;
    bool failed = false;
    const auto decode_start = std::chrono::steady_clock::now();
    while (generated < max_tokens) {
        llama_token token = llama_sampler_sample(sampler, engine->ctx, -1);
        if (llama_vocab_is_eog(vocab, token)) break;
        text += common_token_to_piece(vocab, token, true);
        generated++;

        bool stopped = false;
        for (const std::string& stop : chat.additional_stops) {
            const size_t at = stop.empty() ? std::string::npos : text.find(stop);
            if (at != std::string::npos) {
                text.erase(at);
                stopped = true;
                break;
            }
        }
        if (stopped) break;

        if (llama_decode(engine->ctx, llama_batch_get_one(&token, 1)) != 0) {
            failed = true;
            break;
        }
    }
    const double decode_ms = ms_since(decode_start);
    llama_sampler_free(sampler);
    if (failed) return error_envelope("llama_decode failed while generating");

    // Let llama.cpp's own parser for this model's template pull the tool
    // calls and any reasoning out of the raw text.
    common_chat_msg message;
    try {
        common_chat_parser_params parser_params(chat);
        parser_params.reasoning_format = COMMON_REASONING_FORMAT_DEEPSEEK;
        if (!chat.parser.empty()) parser_params.parser.load(chat.parser);
        message = common_chat_parse(text, false, parser_params);
    } catch (const std::exception& e) {
        LOGW("could not parse model output (%s); returning it as text", e.what());
        message = common_chat_msg();
        message.content = text;
    }

    json calls = json::array();
    for (const common_chat_tool_call& call : message.tool_calls) {
        json entry;
        entry["name"] = call.name;
        json arguments = json::parse(call.arguments, nullptr, false);
        entry["arguments"] = arguments.is_object() ? arguments : json::object();
        calls.push_back(entry);
    }

    json out;
    out["success"] = true;
    out["error"] = nullptr;
    out["cloud_handoff"] = false;
    out["response"] = message.content;
    out["function_calls"] = calls;
    out["raw"] = text;
    out["time_to_first_token_ms"] = prefill_ms;
    out["total_time_ms"] = ms_since(start);
    out["prefill_tokens"] = prompt.size();
    out["decode_tokens"] = generated;
    out["prefill_tps"] = prefill_ms > 0 ? prompt.size() * 1000.0 / prefill_ms : 0.0;
    out["decode_tps"] = decode_ms > 0 ? generated * 1000.0 / decode_ms : 0.0;
    return out.dump(-1, ' ', false, json::error_handler_t::replace);
}

}  // namespace

extern "C" {

JNIEXPORT jlong JNICALL
Java_com_android_server_jarvis_inference_LlamaEngine_nativeInit(
        JNIEnv* env, jclass, jstring modelPath, jboolean embedding, jint nCtx, jint nThreads) {
    static std::once_flag once;
    std::call_once(once, [] {
        llama_log_set(log_callback, nullptr);
        llama_backend_init();
    });

    const std::string path = to_string(env, modelPath);
    try {
        auto engine = std::make_unique<Engine>();
        engine->embedding = embedding == JNI_TRUE;

        llama_model_params model_params = llama_model_default_params();
        model_params.n_gpu_layers = 0;
        engine->model = llama_model_load_from_file(path.c_str(), model_params);
        if (engine->model == nullptr) {
            set_error("could not load model: " + path);
            return 0;
        }

        llama_context_params ctx_params = llama_context_default_params();
        ctx_params.n_ctx = nCtx > 0 ? nCtx : 4096;
        ctx_params.n_batch = 512;
        ctx_params.n_threads = nThreads > 0 ? nThreads : 4;
        ctx_params.n_threads_batch = ctx_params.n_threads;
        ctx_params.no_perf = true;
        if (engine->embedding) {
            ctx_params.embeddings = true;
            // A text longer than this is cut, not rejected; see nativeEmbed.
            ctx_params.n_ubatch = ctx_params.n_batch;
        }
        engine->ctx = llama_init_from_model(engine->model, ctx_params);
        if (engine->ctx == nullptr) {
            llama_model_free(engine->model);
            set_error("could not create a context for: " + path);
            return 0;
        }

        if (!engine->embedding) {
            engine->templates = common_chat_templates_init(engine->model, "");
        }
        LOGI("loaded %s (%s, ctx %u, %d threads)", path.c_str(),
                engine->embedding ? "embeddings" : "chat",
                llama_n_ctx(engine->ctx), ctx_params.n_threads);
        return reinterpret_cast<jlong>(engine.release());
    } catch (const std::exception& e) {
        set_error(std::string("init failed: ") + e.what());
        return 0;
    }
}

JNIEXPORT void JNICALL
Java_com_android_server_jarvis_inference_LlamaEngine_nativeDestroy(JNIEnv*, jclass, jlong handle) {
    Engine* engine = reinterpret_cast<Engine*>(handle);
    if (engine == nullptr) return;
    {
        std::lock_guard<std::mutex> guard(engine->lock);
        engine->templates.reset();
        if (engine->ctx) llama_free(engine->ctx);
        if (engine->model) llama_model_free(engine->model);
    }
    delete engine;
}

JNIEXPORT jfloatArray JNICALL
Java_com_android_server_jarvis_inference_LlamaEngine_nativeEmbed(
        JNIEnv* env, jclass, jlong handle, jstring text, jboolean normalize) {
    Engine* engine = reinterpret_cast<Engine*>(handle);
    if (engine == nullptr || !engine->embedding) {
        set_error("embed: model was not loaded for embeddings");
        return nullptr;
    }
    const std::string input = to_string(env, text);
    try {
        std::lock_guard<std::mutex> guard(engine->lock);
        const llama_vocab* vocab = llama_model_get_vocab(engine->model);
        std::vector<llama_token> tokens = common_tokenize(vocab, input, true, true);
        if (tokens.empty()) {
            set_error("embed: text produced no tokens");
            return nullptr;
        }
        // The whole text must fit one batch to be pooled into one vector.
        const size_t limit = std::min<size_t>(llama_n_batch(engine->ctx), llama_n_ctx(engine->ctx));
        if (tokens.size() > limit) tokens.resize(limit);

        llama_memory_clear(llama_get_memory(engine->ctx), true);
        llama_batch batch = llama_batch_get_one(tokens.data(), static_cast<int32_t>(tokens.size()));
        const int rc = llama_model_has_encoder(engine->model)
                ? llama_encode(engine->ctx, batch) : llama_decode(engine->ctx, batch);
        if (rc != 0) {
            set_error("embed: model run failed");
            return nullptr;
        }

        const float* raw = llama_pooling_type(engine->ctx) == LLAMA_POOLING_TYPE_NONE
                ? llama_get_embeddings_ith(engine->ctx, -1)
                : llama_get_embeddings_seq(engine->ctx, 0);
        if (raw == nullptr) {
            set_error("embed: model returned no embedding");
            return nullptr;
        }
        const int dim = llama_model_n_embd_out(engine->model);
        std::vector<float> out(dim);
        if (normalize == JNI_TRUE) {
            common_embd_normalize(raw, out.data(), dim, 2);
        } else {
            std::copy(raw, raw + dim, out.begin());
        }
        jfloatArray result = env->NewFloatArray(dim);
        if (result != nullptr) env->SetFloatArrayRegion(result, 0, dim, out.data());
        return result;
    } catch (const std::exception& e) {
        set_error(std::string("embed failed: ") + e.what());
        return nullptr;
    }
}

JNIEXPORT jstring JNICALL
Java_com_android_server_jarvis_inference_LlamaEngine_nativeComplete(
        JNIEnv* env, jclass, jlong handle,
        jstring messagesJson, jstring optionsJson, jstring toolsJson) {
    Engine* engine = reinterpret_cast<Engine*>(handle);
    if (engine == nullptr) return to_jstring(env, error_envelope("model not initialized"));
    const std::string messages = to_string(env, messagesJson);
    const std::string options = to_string(env, optionsJson);
    const std::string tools = to_string(env, toolsJson);
    try {
        std::lock_guard<std::mutex> guard(engine->lock);
        return to_jstring(env, complete(engine, messages, options, tools));
    } catch (const std::exception& e) {
        set_error(std::string("complete failed: ") + e.what());
        return to_jstring(env, error_envelope(e.what()));
    }
}

JNIEXPORT jstring JNICALL
Java_com_android_server_jarvis_inference_LlamaEngine_nativeGetLastError(JNIEnv* env, jclass) {
    std::lock_guard<std::mutex> guard(g_error_lock);
    return to_jstring(env, g_last_error);
}

}  // extern "C"
