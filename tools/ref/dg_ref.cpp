// Reference oracle for mojo-dllm's DiffusionGemma, built on libllama from the
// ggml-org/llama.cpp#24423 branch (scripts/build_dg_ref.sh builds both).
//
//   dg_ref info MODEL
//       Reads the vocabulary and hyperparameters only (no weights) and prints
//       key=value lines: n_vocab, canvas_length, bos, eos, final_logit_softcapping.
//
//   dg_ref logits MODEL PROMPT_CSV CANVAS_CSV OUT.f32 THREADS
//                 [--cached] [--prev PREV.f32 --prev-temp-inv F]
//       One denoising step over a fixed prompt and canvas. Writes the canvas
//       rows' logits as f32 [canvas_length, n_vocab], row-major. The prompt rows
//       are not written: only the canvas is sampled from.
//
//         default   the unified pass: [prompt | canvas] in one forward, prompt
//                   rows causal, canvas rows bidirectional over everything.
//         --cached  the prompt K/V path: a prefill of the prompt alone writes
//                   its K/V, then one decode step runs the canvas against it.
//                   The branch states that this agrees with the unified pass.
//         --prev    self-conditioning from PREV.f32, the previous step's canvas
//                   logits [canvas_length, n_vocab] as this tool wrote them, with
//                   the gate open (use_sc = 1) and temp_inv = 1 / the temperature
//                   of the step that produced PREV. Without it the zero
//                   self-conditioning path runs (what a block's first step is).
//
// The sampler is not reimplemented here: unlike LLaDA's, DiffusionGemma's loop
// (random canvas, entropy-bounded acceptance, re-noising) is the #65 work, and
// this tool exists to compare one forward pass at a time.
//
// Attention runs without flash attention, as in the branch's own
// llama-diffusion-gemma-eval: its eager fp32 softmax is the reference the
// branch's exactness checks use, and the cached path keeps its K/V store in f32.

#include "llama.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

static std::vector<int> parse_csv(const char * s) {
    std::vector<int> out;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (!item.empty()) {
            out.push_back(std::atoi(item.c_str()));
        }
    }
    return out;
}

static std::vector<float> read_f32(const char * path, size_t expect) {
    FILE * f = std::fopen(path, "rb");
    if (!f) {
        std::fprintf(stderr, "dg_ref: cannot open %s\n", path);
        std::exit(2);
    }
    std::vector<float> v(expect);
    const size_t got = std::fread(v.data(), sizeof(float), expect, f);
    // One more byte must not exist: a buffer for a different canvas length is a mistake, not a prefix.
    const bool extra = std::fgetc(f) != EOF;
    std::fclose(f);
    if (got != expect || extra) {
        std::fprintf(stderr, "dg_ref: %s does not hold exactly %zu floats\n", path, expect);
        std::exit(2);
    }
    return v;
}

static llama_model * load(const char * path, bool vocab_only) {
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.vocab_only         = vocab_only;
    llama_model * model   = llama_model_load_from_file(path, mp);
    if (!model) {
        std::fprintf(stderr, "dg_ref: failed to load %s\n", path);
        std::exit(2);
    }
    return model;
}

static bool meta(const llama_model * m, const char * key, std::string & out) {
    char buf[64];
    if (llama_model_meta_val_str(m, key, buf, sizeof(buf)) < 0) {
        return false;
    }
    out = buf;
    return true;
}

static int cmd_info(int argc, char ** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: dg_ref info MODEL\n");
        return 1;
    }
    llama_model *       model = load(argv[2], true);
    const llama_vocab * vocab = llama_model_get_vocab(model);
    std::printf("n_vocab=%d\n", llama_vocab_n_tokens(vocab));
    std::printf("bos=%d\n", (int) llama_vocab_bos(vocab));
    std::printf("eos=%d\n", (int) llama_vocab_eos(vocab));
    std::string canvas;
    if (!meta(model, "diffusion.canvas_length", canvas)) {
        std::fprintf(stderr, "dg_ref: the model has no diffusion.canvas_length\n");
        return 2;
    }
    std::printf("canvas_length=%s\n", canvas.c_str());
    // Optional in the branch's loader as well: a model without the key applies no soft-cap.
    std::string softcap;
    std::printf("final_logit_softcapping=%s\n",
                meta(model, "diffusion-gemma.final_logit_softcapping", softcap) ? softcap.c_str() : "none");
    llama_model_free(model);
    return 0;
}

struct Options {
    bool        cached         = false;
    const char * prev          = nullptr;
    bool        has_temp_inv   = false;
    float       prev_temp_inv  = 1.0f;
};

static llama_batch make_batch(const std::vector<int> & ids, int pos0, const std::vector<char> & want_logits) {
    llama_batch b = llama_batch_init((int) ids.size(), 0, 1);
    b.n_tokens    = (int) ids.size();
    for (size_t i = 0; i < ids.size(); i++) {
        b.token[i]     = ids[i];
        b.pos[i]       = (llama_pos) (pos0 + (int) i);
        b.n_seq_id[i]  = 1;
        b.seq_id[i][0] = 0;
        b.logits[i]    = want_logits[i];
    }
    return b;
}

static int cmd_logits(int argc, char ** argv) {
    const char * usage =
        "usage: dg_ref logits MODEL PROMPT_CSV CANVAS_CSV OUT.f32 THREADS [--cached] [--prev PREV.f32 --prev-temp-inv F]\n";
    if (argc < 7) {
        std::fputs(usage, stderr);
        return 1;
    }
    Options opt;
    for (int i = 7; i < argc; i++) {
        if (!std::strcmp(argv[i], "--cached")) {
            opt.cached = true;
        } else if (!std::strcmp(argv[i], "--prev") && i + 1 < argc) {
            opt.prev = argv[++i];
        } else if (!std::strcmp(argv[i], "--prev-temp-inv") && i + 1 < argc) {
            opt.prev_temp_inv = (float) std::atof(argv[++i]);
            opt.has_temp_inv  = true;
        } else {
            std::fprintf(stderr, "dg_ref: unknown argument %s\n%s", argv[i], usage);
            return 1;
        }
    }
    if ((opt.prev != nullptr) != opt.has_temp_inv) {
        std::fprintf(stderr, "dg_ref: --prev and --prev-temp-inv go together; the temperature is never guessed\n");
        return 1;
    }
    if (opt.has_temp_inv && !(opt.prev_temp_inv > 0.0f && std::isfinite(opt.prev_temp_inv))) {
        std::fprintf(stderr, "dg_ref: --prev-temp-inv must be a positive number\n");
        return 1;
    }

    const std::vector<int> prompt = parse_csv(argv[3]);
    const std::vector<int> canvas = parse_csv(argv[4]);
    const char *           out_path = argv[5];
    const int              threads  = std::atoi(argv[6]);
    const int              P        = (int) prompt.size();
    const int              C        = (int) canvas.size();
    if (P <= 0 || C <= 0 || threads <= 0) {
        std::fprintf(stderr, "dg_ref: the prompt, the canvas and the thread count must all be non-empty / positive\n");
        return 1;
    }

    llama_model *       model   = load(argv[2], false);
    const llama_vocab * vocab   = llama_model_get_vocab(model);
    const int           n_vocab = llama_vocab_n_tokens(vocab);

    // The graph splits [prompt | canvas] at n_tokens - canvas_length, taken from the GGUF, not from this
    // call: a canvas of another length would silently move the split.
    std::string canvas_meta;
    if (!meta(model, "diffusion.canvas_length", canvas_meta)) {
        std::fprintf(stderr, "dg_ref: the model has no diffusion.canvas_length\n");
        return 2;
    }
    if (std::atoi(canvas_meta.c_str()) != C) {
        std::fprintf(stderr, "dg_ref: the canvas has %d tokens, the model's diffusion.canvas_length is %s\n", C,
                     canvas_meta.c_str());
        return 2;
    }
    for (const auto * ids : { &prompt, &canvas }) {
        for (int t : *ids) {
            if (t < 0 || t >= n_vocab) {
                std::fprintf(stderr, "dg_ref: token id %d is outside the vocabulary of %d\n", t, n_vocab);
                return 2;
            }
        }
    }

    std::vector<float> prev;
    if (opt.prev) {
        prev = read_f32(opt.prev, (size_t) C * n_vocab);
    }

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx                = P + C;
    cp.n_batch              = P + C;
    cp.n_ubatch             = P + C;  // a non-causal pass needs the whole sequence in one ubatch
    cp.n_threads            = threads;
    cp.n_threads_batch      = threads;
    cp.no_perf              = true;
    cp.flash_attn_type      = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    llama_context * ctx     = llama_init_from_model(model, cp);
    if (!ctx) {
        std::fprintf(stderr, "dg_ref: failed to create the context\n");
        return 2;
    }
    llama_set_causal_attn(ctx, false);

    // Self-conditioning state for the canvas pass. PREV is borrowed until llama_decode returns.
    auto set_sc = [&]() {
        if (opt.prev) {
            llama_diffusion_set_sc(model, prev.data(), 1.0f, opt.prev_temp_inv, true);
        } else {
            llama_diffusion_set_sc(model, nullptr, 0.0f, 1.0f, false);
        }
    };

    // Batch index of the first canvas row in the output: the unified batch carries the prompt too.
    int canvas_row0 = 0;
    if (!opt.cached) {
        std::vector<int> ids(prompt);
        ids.insert(ids.end(), canvas.begin(), canvas.end());
        std::vector<char> want(P + C, 0);
        for (int i = P; i < P + C; i++) {
            want[i] = 1;
        }
        llama_batch b = make_batch(ids, 0, want);
        set_sc();
        const int rc = llama_decode(ctx, b);
        llama_batch_free(b);
        if (rc != 0) {
            std::fprintf(stderr, "dg_ref: llama_decode failed: %d\n", rc);
            return 3;
        }
        canvas_row0 = P;
    } else {
        // PREFILL: the prompt alone, no self-conditioning, writes each layer's K/V. Its logits are unused,
        // but one row has to be requested so the batch has an output.
        std::vector<char> want(P, 0);
        want[P - 1] = 1;
        llama_diffusion_set_sc(model, nullptr, 0.0f, 1.0f, false);
        llama_diffusion_set_phase(model, /*PKV_PREFILL=*/1, P, /*off=*/0);
        llama_batch pre = make_batch(prompt, 0, want);
        int rc = llama_decode(ctx, pre);
        llama_batch_free(pre);
        if (rc != 0) {
            std::fprintf(stderr, "dg_ref: the prefill failed: %d\n", rc);
            return 3;
        }
        // DECODE: the canvas at positions P.., against the stored prompt K/V.
        llama_diffusion_set_phase(model, /*PKV_DECODE=*/2, P, 0);
        set_sc();
        llama_batch dec = make_batch(canvas, P, std::vector<char>(C, 1));
        rc              = llama_decode(ctx, dec);
        llama_batch_free(dec);
        llama_diffusion_set_phase(model, /*PKV_UNIFIED=*/0, 0, 0);
        if (rc != 0) {
            std::fprintf(stderr, "dg_ref: the decode step failed: %d\n", rc);
            return 3;
        }
    }
    llama_diffusion_set_sc(model, nullptr, 0.0f, 1.0f, false);

    // Write to a side file and rename, so an interrupted run never leaves a file that reads as a result.
    const std::string part = std::string(out_path) + ".part";
    FILE *            f    = std::fopen(part.c_str(), "wb");
    if (!f) {
        std::fprintf(stderr, "dg_ref: cannot open %s for writing\n", part.c_str());
        return 2;
    }
    size_t non_finite = 0;
    double max_abs    = 0.0;
    for (int j = 0; j < C; j++) {
        const float * row = llama_get_logits_ith(ctx, canvas_row0 + j);
        if (!row) {
            std::fprintf(stderr, "dg_ref: the context returned no logits for canvas row %d\n", j);
            std::fclose(f);
            std::remove(part.c_str());
            return 3;
        }
        for (int v = 0; v < n_vocab; v++) {
            if (!std::isfinite(row[v])) {
                non_finite++;
            } else {
                max_abs = std::fmax(max_abs, std::fabs((double) row[v]));
            }
        }
        if (std::fwrite(row, sizeof(float), n_vocab, f) != (size_t) n_vocab) {
            std::fprintf(stderr, "dg_ref: short write to %s\n", part.c_str());
            std::fclose(f);
            std::remove(part.c_str());
            return 2;
        }
    }
    if (std::fclose(f) != 0 || non_finite != 0) {
        std::fprintf(stderr, "dg_ref: refusing the output: %zu non-finite logits (or the file failed to close)\n",
                     non_finite);
        std::remove(part.c_str());
        return 3;
    }
    if (std::rename(part.c_str(), out_path) != 0) {
        std::fprintf(stderr, "dg_ref: cannot move %s to %s\n", part.c_str(), out_path);
        return 2;
    }
    std::fprintf(stderr, "wrote %d rows x %d vocab (%s, self-conditioning %s) max|logit| %.4f\n", C, n_vocab,
                 opt.cached ? "prefill + decode" : "unified", opt.prev ? "on" : "off", max_abs);
    llama_free(ctx);
    llama_model_free(model);
    return 0;
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: dg_ref {info|logits} ...\n");
        return 1;
    }
    if (std::strcmp(argv[1], "info") == 0) {
        return cmd_info(argc, argv);
    }
    if (std::strcmp(argv[1], "logits") == 0) {
        return cmd_logits(argc, argv);
    }
    std::fprintf(stderr, "dg_ref: unknown command %s\n", argv[1]);
    return 1;
}
