// Reference dumper: runs llama.cpp (CPU) on a prompt and writes every graph node
// whose name ends with the given suffix (e.g. "-0" for layer 0) as raw float data.
// Output: <outdir>/<name>.f32 (or raw bytes for other types) and <outdir>/index.tsv
#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <clocale>
#include <sstream>
#include <map>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <sys/stat.h>

struct dump_state {
    std::string suffix;   // node name suffix to keep, e.g. "-0"
    std::string outdir;
    FILE * index = nullptr;
    std::map<std::string, int> seen;
};

static bool name_matches(const char * name, const std::string & suffix) {
    size_t n = std::strlen(name), m = suffix.size();
    if (m == 0) return true;
    if (n < m) return false;
    return std::strcmp(name + n - m, suffix.c_str()) == 0;
}

static bool dump_cb(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * st = static_cast<dump_state *>(user_data);
    if (!name_matches(t->name, st->suffix)) {
        return true;
    }
    if (ask) {
        return true;
    }
    size_t nbytes = ggml_nbytes(t);
    std::vector<uint8_t> buf(nbytes);
    ggml_backend_tensor_get(t, buf.data(), 0, nbytes);

    // same name can appear more than once (e.g. norm-0): keep every occurrence
    int occ = st->seen[t->name]++;
    std::string base = t->name;
    if (occ > 0) base += "#" + std::to_string(occ + 1);
    std::string path = st->outdir + "/" + base + ".bin";
    FILE * f = std::fopen(path.c_str(), "wb");
    if (!f) return false;
    std::fwrite(buf.data(), 1, nbytes, f);
    std::fclose(f);

    std::fprintf(st->index, "%s\t%s\t%lld\t%lld\t%lld\t%lld\t%zu\n", base.c_str(), ggml_type_name(t->type),
        (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2], (long long) t->ne[3], nbytes);
    return true;  // ask=false: true continues the graph
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");
    common_params params;
    common_init();

    // extra flags, removed before llama.cpp parses the rest
    std::string suffix = "-0", outdir = "dump", ids_arg;
    std::vector<char *> args;
    for (int i = 0; i < argc; ++i) {
        if (std::strcmp(argv[i], "--ids") == 0 && i + 1 < argc) { ids_arg = argv[++i]; continue; }
        if (std::strcmp(argv[i], "--suffix") == 0 && i + 1 < argc) { suffix = argv[++i]; continue; }
        if (std::strcmp(argv[i], "--outdir") == 0 && i + 1 < argc) { outdir = argv[++i]; continue; }
        args.push_back(argv[i]);
    }
    int ac = (int) args.size();
    if (!common_params_parse(ac, args.data(), params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    mkdir(outdir.c_str(), 0755);

    dump_state st;
    st.suffix = suffix;
    st.outdir = outdir;
    st.index = std::fopen((outdir + "/index.tsv").c_str(), "w");

    llama_backend_init();
    params.cb_eval = dump_cb;
    params.cb_eval_user_data = &st;
    params.warmup = false;

    auto llama_init = common_init_from_params(params);
    llama_model * model = llama_init->model();
    llama_context * ctx = llama_init->context();
    if (!model || !ctx) {
        std::fprintf(stderr, "failed to init\n");
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    std::vector<llama_token> tokens;
    if (!ids_arg.empty()) {
        std::stringstream ss(ids_arg);
        std::string item;
        while (std::getline(ss, item, ',')) tokens.push_back((llama_token) std::atoi(item.c_str()));
    } else {
        tokens = common_tokenize(ctx, params.prompt, llama_vocab_get_add_bos(vocab), true);
    }
    std::printf("tokens (%zu):", tokens.size());
    for (auto t : tokens) std::printf(" %d", t);
    std::printf("\n");

    common_batch batch = common_batch_get_one(ctx, tokens);
    if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get())) {
        std::fprintf(stderr, "decode failed\n");
        return 1;
    }

    std::fclose(st.index);
    llama_backend_free();
    return 0;
}
