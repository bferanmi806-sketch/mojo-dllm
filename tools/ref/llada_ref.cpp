// Reference oracle for mojo-dllm, built on llama.cpp's libllama.
//
//   llada_ref logits   MODEL TOKENS_CSV ROWS_CSV OUT.f32 THREADS
//       Non-causal forward over TOKENS; writes f32 logits for ROWS as [rows, vocab].
//
//   llada_ref generate-dream MODEL PROMPT_TOKENS_CSV GEN_LEN STEPS THREADS
//       Dream's diffusion_generate (Dream-org/Dream generation_utils.py),
//       greedy, alg="entropy", eps=1e-3, with the logit shift applied here:
//       llama.cpp returns raw rows and row p-1 scores position p.
//
//   llada_ref generate MODEL PROMPT_TOKENS_CSV GEN_LEN BLOCK_LEN STEPS THREADS
//       LLaDA's reference sampler (ML-GSAI/LLaDA generate.py, temperature 0,
//       remasking="low_confidence", cfg 0) driven by llama.cpp logits. Prints the
//       final canvas token ids as CSV on stdout.
//
// Every input is checked and every failure prints "llada_ref: ..." on stderr and exits non-zero (1: bad
// arguments, 2: the model or a file, 3: llama.cpp or a non-finite logit), because this output is the ground
// truth the parity bound is measured against: a number that is not a number, a row outside the canvas, an
// unwritable output or a NaN must never become a file that looks like a result.
//
// llama.cpp's own diffusion example is not used for token parity: on the pinned
// commit it never applies cfg/alg_temp, adds Gumbel noise to one row only, and
// samples from the distribution even at temperature 0 (see docs/correctness.md).

#include "llama.h"

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

[[noreturn]] static void die(int code, const char * fmt, ...) {
    std::fprintf(stderr, "llada_ref: ");
    va_list ap;
    va_start(ap, fmt);
    std::vfprintf(stderr, fmt, ap);
    va_end(ap);
    std::fprintf(stderr, "\n");
    std::exit(code);
}

// A whole-string integer in [lo, hi]. atoi turns "abc" into 0 and "12x" into 12: both would run as if valid.
static long parse_int(const char * s, const char * what, long lo, long hi) {
    char * end = nullptr;
    errno      = 0;
    long v     = std::strtol(s, &end, 10);
    // strtol skips leading blanks, which no caller of this tool writes: refuse them as well.
    if (s[0] == '\0' || std::isspace((unsigned char) s[0]) || end == s || *end != '\0' || errno == ERANGE) {
        die(1, "%s must be an integer, got '%s'", what, s);
    }
    if (v < lo || v > hi) {
        die(1, "%s must be between %ld and %ld, got %ld", what, lo, hi, v);
    }
    return v;
}

// Comma-separated integers; an empty item ("1,,2", a trailing comma) is a mistake, not a skipped value.
static std::vector<int> parse_csv(const char * s, const char * what) {
    std::vector<int> out;
    std::stringstream ss(s);
    std::string item;
    bool any = false;
    while (std::getline(ss, item, ',')) {
        any = true;
        out.push_back((int) parse_int(item.c_str(), what, INT_MIN, INT_MAX));
    }
    if (!any || (*s != '\0' && s[std::strlen(s) - 1] == ',')) {
        die(1, "%s must be a non-empty comma-separated list of integers, got '%s'", what, s);
    }
    return out;
}

// Every logit of one row must be a number; a NaN or an infinity would give an arbitrary argmax and a poisoned file.
static void check_finite(const float * row, int n_vocab, const char * what, int index) {
    for (int v = 0; v < n_vocab; v++) {
        if (!std::isfinite(row[v])) {
            die(3, "refusing the output: %s %d has a non-finite logit at vocab id %d", what, index, v);
        }
    }
}

static void check_token_ids(const std::vector<int> & tokens, int n_vocab, const char * what) {
    for (size_t i = 0; i < tokens.size(); i++) {
        if (tokens[i] < 0 || tokens[i] >= n_vocab) {
            die(1, "%s[%zu] = %d is outside the vocabulary 0..%d", what, i, tokens[i], n_vocab - 1);
        }
    }
}

struct Ctx {
    llama_model *   model = nullptr;
    llama_context * ctx   = nullptr;
    int             n_vocab = 0;
};

static Ctx open(const char * path, int n_tokens, int threads) {
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    Ctx c;
    c.model = llama_model_load_from_file(path, mp);
    if (!c.model) {
        die(2, "failed to load %s", path);
    }
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx           = n_tokens;
    cp.n_batch         = n_tokens;
    cp.n_ubatch        = n_tokens;
    cp.n_threads       = threads;
    cp.n_threads_batch = threads;
    c.ctx              = llama_init_from_model(c.model, cp);
    if (!c.ctx) {
        die(2, "failed to create context");
    }
    llama_set_causal_attn(c.ctx, false);
    c.n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(c.model));
    return c;
}

// Logits for every position of `tokens`, row-major [n, vocab].
static const float * forward(Ctx & c, const std::vector<int> & tokens) {
    llama_batch b = llama_batch_init((int) tokens.size(), 0, 1);
    for (size_t i = 0; i < tokens.size(); i++) {
        b.token[i]     = tokens[i];
        b.pos[i]       = (llama_pos) i;
        b.n_seq_id[i]  = 1;
        b.seq_id[i][0] = 0;
        b.logits[i]    = 1;
    }
    b.n_tokens = (int) tokens.size();
    int rc     = llama_decode(c.ctx, b);
    llama_batch_free(b);
    if (rc != 0) {
        die(3, "llama_decode failed: %d", rc);
    }
    const float * lg = llama_get_logits(c.ctx);
    if (lg == nullptr) {
        die(3, "llama.cpp returned no logits");
    }
    return lg;
}

static int cmd_logits(int argc, char ** argv) {
    if (argc != 7) {
        std::fprintf(stderr, "usage: llada_ref logits MODEL TOKENS ROWS OUT THREADS\n");
        return 1;
    }
    auto tokens  = parse_csv(argv[3], "TOKENS");
    auto rows    = parse_csv(argv[4], "ROWS");
    int  threads = (int) parse_int(argv[6], "THREADS", 1, 4096);
    for (int r : rows) {
        if (r < 0 || r >= (int) tokens.size()) {
            die(1, "row %d is outside the %zu tokens (0..%zu)", r, tokens.size(), tokens.size() - 1);
        }
    }
    Ctx c = open(argv[2], (int) tokens.size(), threads);
    check_token_ids(tokens, c.n_vocab, "TOKENS");
    const float * lg = forward(c, tokens);
    // Written beside the target and moved into place only when every row is finite and the file closed, so a
    // failed run never leaves a short or poisoned file under the name the parity tool reads.
    std::string part = std::string(argv[5]) + ".part";
    FILE * f = std::fopen(part.c_str(), "wb");
    if (!f) {
        die(2, "cannot open %s for writing: %s", part.c_str(), std::strerror(errno));
    }
    for (int r : rows) {
        const float * row = lg + (size_t) r * c.n_vocab;
        for (int v = 0; v < c.n_vocab; v++) {
            if (!std::isfinite(row[v])) {
                std::fclose(f);
                std::remove(part.c_str());
                die(3, "refusing the output: row %d has a non-finite logit at vocab id %d", r, v);
            }
        }
        if (std::fwrite(row, sizeof(float), c.n_vocab, f) != (size_t) c.n_vocab) {
            std::fclose(f);
            std::remove(part.c_str());
            die(2, "short write to %s: %s", part.c_str(), std::strerror(errno));
        }
    }
    if (std::fclose(f) != 0) {
        std::remove(part.c_str());
        die(2, "could not close %s: %s", part.c_str(), std::strerror(errno));
    }
    if (std::rename(part.c_str(), argv[5]) != 0) {
        std::remove(part.c_str());
        die(2, "cannot move %s to %s: %s", part.c_str(), argv[5], std::strerror(errno));
    }
    std::fprintf(stderr, "wrote %zu rows x %d vocab\n", rows.size(), c.n_vocab);
    return 0;
}

static int cmd_generate(int argc, char ** argv) {
    if (argc != 8) {
        std::fprintf(stderr, "usage: llada_ref generate MODEL PROMPT GEN_LEN BLOCK_LEN STEPS THREADS\n");
        return 1;
    }
    auto prompt    = parse_csv(argv[3], "PROMPT");
    int  gen_len   = (int) parse_int(argv[4], "GEN_LEN", 1, 1 << 20);
    int  block_len = (int) parse_int(argv[5], "BLOCK_LEN", 1, 1 << 20);
    int  steps     = (int) parse_int(argv[6], "STEPS", 1, 1 << 20);
    int  threads   = (int) parse_int(argv[7], "THREADS", 1, 4096);
    if (gen_len % block_len != 0) {
        std::fprintf(stderr, "llada_ref: gen_len must be a multiple of block_len\n");
        return 1;
    }
    int num_blocks = gen_len / block_len;
    if (steps % num_blocks != 0) {
        std::fprintf(stderr, "llada_ref: steps must be a multiple of the block count\n");
        return 1;
    }
    int steps_per_block = steps / num_blocks;
    int n_prompt        = (int) prompt.size();
    std::vector<int> x(prompt);
    Ctx  c = open(argv[2], n_prompt + gen_len, threads);
    check_token_ids(prompt, c.n_vocab, "PROMPT");
    char buf[32];
    if (llama_model_meta_val_str(c.model, "tokenizer.ggml.mask_token_id", buf, sizeof(buf)) < 0) {
        std::fprintf(stderr, "llada_ref: model has no tokenizer.ggml.mask_token_id\n");
        return 1;
    }
    const int MASK = (int) parse_int(buf, "tokenizer.ggml.mask_token_id", 0, c.n_vocab - 1);
    x.resize(n_prompt + gen_len, MASK);

    for (int nb = 0; nb < num_blocks; nb++) {
        int b0 = n_prompt + nb * block_len;
        int b1 = b0 + block_len;
        int m  = 0;
        for (int i = b0; i < b1; i++) {
            m += x[i] == MASK;
        }
        std::vector<int> transfer(steps_per_block, m / steps_per_block);
        for (int i = 0; i < m % steps_per_block; i++) {
            transfer[i] += 1;
        }
        for (int s = 0; s < steps_per_block; s++) {
            const float * lg = forward(c, x);
            std::vector<std::pair<double, int>> cand;
            std::vector<int> x0(x.size(), -1);
            for (int i = b0; i < b1; i++) {
                if (x[i] != MASK) {
                    continue;
                }
                const float * row = lg + (size_t) i * c.n_vocab;
                check_finite(row, c.n_vocab, "canvas row", i);
                int best = 0;
                for (int v = 1; v < c.n_vocab; v++) {
                    if (row[v] > row[best]) {
                        best = v;
                    }
                }
                double mx = row[best], z = 0.0;
                for (int v = 0; v < c.n_vocab; v++) {
                    z += std::exp((double) row[v] - mx);
                }
                x0[i] = best;
                cand.push_back({ 1.0 / z, i });
            }
            int k = std::min(transfer[s], (int) cand.size());
            std::stable_sort(cand.begin(), cand.end(),
                             [](const auto & a, const auto & b) { return a.first > b.first; });
            for (int j = 0; j < k; j++) {
                x[cand[j].second] = x0[cand[j].second];
            }
        }
    }
    for (size_t i = 0; i < x.size(); i++) {
        std::printf(i ? ",%d" : "%d", x[i]);
    }
    std::printf("\n");
    return 0;
}

static int cmd_generate_dream(int argc, char ** argv) {
    if (argc != 7) {
        std::fprintf(stderr, "usage: llada_ref generate-dream MODEL PROMPT GEN_LEN STEPS THREADS\n");
        return 1;
    }
    auto   prompt   = parse_csv(argv[3], "PROMPT");
    // parse_csv refuses an empty list, so row p-1 of the first canvas position is never row -1.
    int    gen_len  = (int) parse_int(argv[4], "GEN_LEN", 1, 1 << 20);
    int    steps    = (int) parse_int(argv[5], "STEPS", 1, 1 << 20);
    int    threads  = (int) parse_int(argv[6], "THREADS", 1, 4096);
    int    n_prompt = (int) prompt.size();
    Ctx    c        = open(argv[2], n_prompt + gen_len, threads);
    check_token_ids(prompt, c.n_vocab, "PROMPT");
    char   buf[32];
    if (llama_model_meta_val_str(c.model, "tokenizer.ggml.mask_token_id", buf, sizeof(buf)) < 0) {
        std::fprintf(stderr, "llada_ref: model has no tokenizer.ggml.mask_token_id\n");
        return 1;
    }
    const int    MASK = (int) parse_int(buf, "tokenizer.ggml.mask_token_id", 0, c.n_vocab - 1);
    const double eps  = 1e-3;
    std::vector<int> x(prompt);
    x.resize(n_prompt + gen_len, MASK);
    for (int i = 0; i < steps; i++) {
        std::vector<int> cand;
        for (int p = n_prompt; p < n_prompt + gen_len; p++) {
            if (x[p] == MASK) {
                cand.push_back(p);
            }
        }
        if (cand.empty()) {
            break;
        }
        const float * lg = forward(c, x);
        std::vector<std::pair<double, int>> conf;
        std::vector<int> x0(cand.size());
        for (size_t j = 0; j < cand.size(); j++) {
            const float * row = lg + (size_t) (cand[j] - 1) * c.n_vocab;
            check_finite(row, c.n_vocab, "canvas row", cand[j] - 1);
            int best = 0;
            for (int v = 1; v < c.n_vocab; v++) {
                if (row[v] > row[best]) {
                    best = v;
                }
            }
            double mx = row[best], z = 0.0;
            for (int v = 0; v < c.n_vocab; v++) {
                z += std::exp((double) row[v] - mx);
            }
            double logz = std::log(z), neg_h = 0.0;
            for (int v = 0; v < c.n_vocab; v++) {
                double lp = (double) row[v] - mx - logz;
                neg_h += std::exp(lp) * lp;
            }
            x0[j] = best;
            conf.push_back({ neg_h, (int) j });
        }
        int n = (int) cand.size();
        if (i < steps - 1) {
            double t = 1.0 - (double) i * (1.0 - eps) / steps;
            double s = 1.0 - (double) (i + 1) * (1.0 - eps) / steps;
            n        = (int) ((double) cand.size() * (1.0 - s / t));
        }
        std::stable_sort(conf.begin(), conf.end(), [](const auto & a, const auto & b) { return a.first > b.first; });
        for (int j = 0; j < n; j++) {
            x[cand[conf[j].second]] = x0[conf[j].second];
        }
    }
    for (size_t i = 0; i < x.size(); i++) {
        std::printf(i ? ",%d" : "%d", x[i]);
    }
    std::printf("\n");
    return 0;
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: llada_ref {logits|generate} ...\n");
        return 1;
    }
    if (std::strcmp(argv[1], "logits") == 0) {
        return cmd_logits(argc, argv);
    }
    if (std::strcmp(argv[1], "generate") == 0) {
        return cmd_generate(argc, argv);
    }
    if (std::strcmp(argv[1], "generate-dream") == 0) {
        return cmd_generate_dream(argc, argv);
    }
    std::fprintf(stderr, "unknown command %s\n", argv[1]);
    return 1;
}
