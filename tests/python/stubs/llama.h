// A stand-in for llama.cpp's llama.h with only what tools/ref/llada_ref.cpp calls, so the tool's own checks can be
// built and run with the host compiler and no model. See tests/python/test_llada_ref.py.
#pragma once
#include <cstddef>
#include <cstdint>

typedef int32_t llama_pos;
struct llama_model;
struct llama_context;
struct llama_vocab;
struct llama_model_params {
    int unused;
};
struct llama_context_params {
    int n_ctx, n_batch, n_ubatch, n_threads, n_threads_batch;
};
struct llama_batch {
    int32_t       n_tokens;
    int32_t *     token;
    llama_pos *   pos;
    int32_t *     n_seq_id;
    int32_t **    seq_id;
    int8_t *      logits;
};

void                  llama_backend_init(void);
llama_model_params    llama_model_default_params(void);
llama_model *         llama_model_load_from_file(const char * path, llama_model_params params);
llama_context_params  llama_context_default_params(void);
llama_context *       llama_init_from_model(llama_model * model, llama_context_params params);
void                  llama_set_causal_attn(llama_context * ctx, bool causal);
const llama_vocab *   llama_model_get_vocab(const llama_model * model);
int32_t               llama_vocab_n_tokens(const llama_vocab * vocab);
llama_batch           llama_batch_init(int32_t n_tokens, int32_t embd, int32_t n_seq_max);
void                  llama_batch_free(llama_batch batch);
int32_t               llama_decode(llama_context * ctx, llama_batch batch);
float *               llama_get_logits(llama_context * ctx);
int32_t               llama_model_meta_val_str(const llama_model * model, const char * key, char * buf, size_t buf_size);
