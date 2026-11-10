// The functions of stubs/llama.h, with a tiny deterministic "model": logits[row][v] = ((row * 7 + v * 3) % 11) / 10.
// The environment steers the failures a test needs: STUB_VOCAB, STUB_NAN_ROW, STUB_INF_ROW, STUB_DECODE_FAIL,
// STUB_NO_LOGITS, STUB_NO_MASK, STUB_MASK_ID, STUB_LOAD_FAIL.
#include "llama.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static std::vector<float> g_logits;
static int g_vocab() {
    const char * e = std::getenv("STUB_VOCAB");
    return e ? std::atoi(e) : 8;
}

void llama_backend_init(void) {}
llama_model_params llama_model_default_params(void) { return {0}; }
llama_model * llama_model_load_from_file(const char *, llama_model_params) {
    return std::getenv("STUB_LOAD_FAIL") ? nullptr : (llama_model *) 1;
}
llama_context_params llama_context_default_params(void) { return {0, 0, 0, 0, 0}; }
llama_context * llama_init_from_model(llama_model *, llama_context_params) { return (llama_context *) 1; }
void llama_set_causal_attn(llama_context *, bool) {}
const llama_vocab * llama_model_get_vocab(const llama_model *) { return (const llama_vocab *) 1; }
int32_t llama_vocab_n_tokens(const llama_vocab *) { return g_vocab(); }

llama_batch llama_batch_init(int32_t n, int32_t, int32_t) {
    llama_batch b;
    b.n_tokens = n;
    b.token    = new int32_t[n];
    b.pos      = new llama_pos[n];
    b.n_seq_id = new int32_t[n];
    b.seq_id   = new int32_t *[n];
    for (int i = 0; i < n; i++) {
        b.seq_id[i] = new int32_t[1];
    }
    b.logits = new int8_t[n];
    return b;
}
void llama_batch_free(llama_batch b) {
    for (int i = 0; i < b.n_tokens; i++) {
        delete[] b.seq_id[i];
    }
    delete[] b.token;
    delete[] b.pos;
    delete[] b.n_seq_id;
    delete[] b.seq_id;
    delete[] b.logits;
}

int32_t llama_decode(llama_context *, llama_batch b) {
    if (std::getenv("STUB_DECODE_FAIL")) {
        return 7;
    }
    int vocab = g_vocab();
    g_logits.assign((size_t) b.n_tokens * vocab, 0.0f);
    for (int r = 0; r < b.n_tokens; r++) {
        for (int v = 0; v < vocab; v++) {
            g_logits[(size_t) r * vocab + v] = (float) (((r * 7 + v * 3) % 11) / 10.0);
        }
    }
    const char * nan_row = std::getenv("STUB_NAN_ROW");
    if (nan_row && std::atoi(nan_row) < b.n_tokens) {
        g_logits[(size_t) std::atoi(nan_row) * vocab + 1] = std::strtof("nan", nullptr);
    }
    const char * inf_row = std::getenv("STUB_INF_ROW");
    if (inf_row && std::atoi(inf_row) < b.n_tokens) {
        g_logits[(size_t) std::atoi(inf_row) * vocab] = std::strtof("inf", nullptr);
    }
    return 0;
}
float * llama_get_logits(llama_context *) { return std::getenv("STUB_NO_LOGITS") ? nullptr : g_logits.data(); }

int32_t llama_model_meta_val_str(const llama_model *, const char * key, char * buf, size_t n) {
    if (std::getenv("STUB_NO_MASK") || std::strcmp(key, "tokenizer.ggml.mask_token_id") != 0) {
        return -1;
    }
    const char * id = std::getenv("STUB_MASK_ID");
    return std::snprintf(buf, n, "%s", id ? id : "7");
}
