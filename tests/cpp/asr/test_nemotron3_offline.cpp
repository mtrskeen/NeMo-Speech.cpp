// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "nemotron3_model.h"

namespace {

struct RefArray {
    std::vector<int64_t> shape;
    std::vector<float> f;
    int64_t numel() const {
        int64_t n = 1;
        for (auto d : shape) n *= d;
        return n;
    }
};

RefArray load_ref(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("missing: " + path);
    char magic[4];
    f.read(magic, 4);
    if (std::memcmp(magic, "NERB", 4) != 0) throw std::runtime_error("bad magic: " + path);
    uint32_t code;
    int64_t n_dims;
    f.read(reinterpret_cast<char*>(&code), 4);
    f.read(reinterpret_cast<char*>(&n_dims), 8);
    RefArray a;
    a.shape.resize(n_dims);
    f.read(reinterpret_cast<char*>(a.shape.data()), 8 * n_dims);
    a.f.resize(a.numel());
    f.read(reinterpret_cast<char*>(a.f.data()), 4 * a.numel());
    return a;
}

float max_abs_diff(const float* a, const float* b, size_t n) {
    float m = 0.f;
    for (size_t i = 0; i < n; i++) {
        float d = std::fabs(a[i] - b[i]);
        if (d > m) m = d;
    }
    return m;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s <model.gguf> <ref-dir> [tolerance]\n", argv[0]);
        return 2;
    }
    const std::string gguf_path = argv[1];
    const std::string ref_dir = argv[2];
    const float tol = (argc > 3) ? std::stof(argv[3]) : 1e-4f;

    ggml_runtime::Params backend_params;
    backend_params.use_gpu = false;
    ggml_runtime::BackendManager bm(backend_params);
    nemo_speech::asr::Nemotron3Model model(bm, gguf_path);

    auto mel = load_ref(ref_dir + "/mel_transposed.bin");
    auto gold_probs = load_ref(ref_dir + "/probs.bin");
    auto gold_pre = load_ref(ref_dir + "/pre_encode.bin");

    const int t_mel = 1000;
    std::printf("[offline test] running chunk with %d mel frames...\n", t_mel);

    auto out = model.run_chunk(mel.f.data(), t_mel, nullptr, 0, nullptr, 0);

    std::printf("[offline test] out.preds.size = %zu, gold_probs.size = %zu\n",
                out.preds.size(), gold_probs.f.size());
    std::printf("[offline test] out.chunk_embs.size = %zu, gold_pre.size = %zu\n",
                out.chunk_embs.size(), gold_pre.f.size());

    float diff_pre = max_abs_diff(out.chunk_embs.data(), gold_pre.f.data(), gold_pre.f.size());
    std::printf("pre_encode max_abs_diff: %.6e\n", diff_pre);

    auto gold_en = load_ref(ref_dir + "/embed_norm.bin");
    std::printf("embed_norm max_abs_diff: %.6e\n", max_abs_diff(out.dbg_embed_norm.data(), gold_en.f.data(), gold_en.f.size()));

    auto gold_l0 = load_ref(ref_dir + "/layer0.bin");
    std::printf("layer0 max_abs_diff: %.6e\n", max_abs_diff(out.dbg_layer0.data(), gold_l0.f.data(), gold_l0.f.size()));

    auto gold_fn = load_ref(ref_dir + "/final_norm.bin");
    std::printf("final_norm max_abs_diff: %.6e\n", max_abs_diff(out.dbg_final_norm.data(), gold_fn.f.data(), gold_fn.f.size()));

    auto gold_pj = load_ref(ref_dir + "/encoder_proj.bin");
    std::printf("encoder_proj max_abs_diff: %.6e\n", max_abs_diff(out.dbg_proj.data(), gold_pj.f.data(), gold_pj.f.size()));

    auto gold_sb = load_ref(ref_dir + "/subpixel.bin");
    std::printf("subpixel max_abs_diff: %.6e (out size %zu, gold size %zu)\n",
        max_abs_diff(out.dbg_sub.data(), gold_sb.f.data(), gold_sb.f.size()), out.dbg_sub.size(), gold_sb.f.size());

    float diff_probs = max_abs_diff(out.preds.data(), gold_probs.f.data(), gold_probs.f.size());
    std::printf("probs max_abs_diff: %.6e\n", diff_probs);

    if (diff_probs <= tol) {
        std::printf("SUCCESS: offline parity verified within tolerance (<= %.1e)!\n", tol);
        return 0;
    } else {
        std::printf("FAILURE: diff_probs = %.6e exceeds tolerance %.1e\n", diff_probs, tol);
        return 1;
    }
}
