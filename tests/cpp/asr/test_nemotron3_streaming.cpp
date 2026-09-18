// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <numeric>
#include <string>
#include <vector>

#include "diar_pipeline.h"
#include "fe.h"

namespace {

struct RefArray {
    std::vector<int64_t> shape;
    std::vector<float> data;
    int64_t numel() const {
        int64_t n = 1;
        for (auto d : shape) n *= d;
        return n;
    }
};

RefArray load_npy_f32(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("failed to open: " + path);
    char magic[6];
    f.read(magic, 6);
    if (std::memcmp(magic, "\x93NUMPY", 6) != 0) {
        throw std::runtime_error("bad NPY magic: " + path);
    }
    uint8_t major, minor;
    f.read(reinterpret_cast<char*>(&major), 1);
    f.read(reinterpret_cast<char*>(&minor), 1);
    uint32_t header_len = 0;
    if (major == 1) {
        uint16_t hlen16 = 0;
        f.read(reinterpret_cast<char*>(&hlen16), 2);
        header_len = hlen16;
    } else {
        f.read(reinterpret_cast<char*>(&header_len), 4);
    }
    std::string header(header_len, ' ');
    f.read(&header[0], header_len);

    RefArray arr;
    auto pos = header.find("'shape':");
    if (pos != std::string::npos) {
        auto open_paren = header.find('(', pos);
        auto close_paren = header.find(')', open_paren);
        if (open_paren != std::string::npos && close_paren != std::string::npos) {
            std::string s = header.substr(open_paren + 1, close_paren - open_paren - 1);
            size_t start = 0;
            while (start < s.size()) {
                size_t comma = s.find(',', start);
                std::string token = (comma == std::string::npos) ? s.substr(start) : s.substr(start, comma - start);
                size_t p1 = token.find_first_not_of(" \t\r\n");
                if (p1 != std::string::npos) {
                    size_t p2 = token.find_last_not_of(" \t\r\n");
                    std::string val = token.substr(p1, p2 - p1 + 1);
                    if (!val.empty()) {
                        arr.shape.push_back(std::stoll(val));
                    }
                }
                if (comma == std::string::npos) break;
                start = comma + 1;
            }
        }
    }
    int64_t n = arr.numel();
    arr.data.resize(n);
    f.read(reinterpret_cast<char*>(arr.data.data()), n * sizeof(float));
    return arr;
}

RefArray load_ref_probs(const std::string& path) {
    if (path.size() >= 4 && path.substr(path.size() - 4) == ".npy") {
        return load_npy_f32(path);
    }
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("missing: " + path);
    char magic[4];
    f.read(magic, 4);
    if (std::memcmp(magic, "NERB", 4) == 0) {
        uint32_t code;
        int64_t n_dims;
        f.read(reinterpret_cast<char*>(&code), 4);
        f.read(reinterpret_cast<char*>(&n_dims), 8);
        RefArray a;
        a.shape.resize(n_dims);
        f.read(reinterpret_cast<char*>(a.shape.data()), 8 * n_dims);
        a.data.resize(a.numel());
        f.read(reinterpret_cast<char*>(a.data.data()), 4 * a.numel());
        return a;
    }
    f.seekg(0, std::ios::end);
    size_t sz = f.tellg();
    f.seekg(0, std::ios::beg);
    RefArray a;
    a.shape = {1, static_cast<int64_t>(sz / (8 * sizeof(float))), 8};
    a.data.resize(sz / sizeof(float));
    f.read(reinterpret_cast<char*>(a.data.data()), sz);
    return a;
}

double compute_correlation(const float* x, const float* y, size_t n) {
    if (n < 2) return 1.0;
    double sum_x = 0, sum_y = 0;
    for (size_t i = 0; i < n; i++) {
        sum_x += x[i];
        sum_y += y[i];
    }
    double mean_x = sum_x / n;
    double mean_y = sum_y / n;
    double num = 0, den_x = 0, den_y = 0;
    for (size_t i = 0; i < n; i++) {
        double dx = x[i] - mean_x;
        double dy = y[i] - mean_y;
        num += dx * dy;
        den_x += dx * dx;
        den_y += dy * dy;
    }
    if (den_x <= 1e-12 || den_y <= 1e-12) return 1.0;
    return num / std::sqrt(den_x * den_y);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr,
            "usage: %s <model.gguf> <ref-dir>\n"
            "       %s <model.gguf> <audio.wav> <ref_probs.npy> [--preset P] [--gpu N]\n",
            argv[0], argv[0]);
        return 2;
    }

    const std::string gguf_path = argv[1];
    std::string wav_path;
    std::string ref_probs_path;
    std::string preset = "low";
    int gpu = -1;

    if (argc == 3) {
        const std::string ref_dir = argv[2];
        wav_path = ref_dir + "/ref30.wav";
        ref_probs_path = ref_dir + "/probs.npy";
    } else {
        wav_path = argv[2];
        ref_probs_path = argv[3];
        for (int i = 4; i < argc; i++) {
            std::string a = argv[i];
            if (a == "--preset" && i + 1 < argc) {
                preset = argv[++i];
            } else if (a == "--gpu" && i + 1 < argc) {
                gpu = std::stoi(argv[++i]);
            }
        }
    }

    std::vector<float> audio;
    int sr = 0;
    if (!read_wav_mono_16k(wav_path, audio, sr) || sr != 16000) {
        std::fprintf(stderr, "failed to read 16 kHz mono wav: %s\n", wav_path.c_str());
        return 1;
    }

    ggml_runtime::Params backend_params;
    backend_params.use_gpu = (gpu >= 0);
    ggml_runtime::BackendManager bm(backend_params);

    nemo_speech::asr::BatchingConfig batching;
    nemo_speech::asr::DiarModel model(bm, gguf_path, batching);
    nemo_speech::asr::DiarGeometry geo = nemo_speech::asr::DiarGeometry::preset(preset);
    nemo_speech::asr::DiarStream stream(model, geo);

    const size_t push = 160 * 16;  // 160 ms chunks
    for (size_t off = 0; off < audio.size(); off += push) {
        stream.feed_audio(audio.data() + off, std::min(push, audio.size() - off));
    }
    stream.finish();

    auto probs = stream.frame_probs();
    int64_t n_frames = stream.n_frames();
    int n_spk = model.cfg().num_speakers;

    std::printf("[streaming test] produced %ld frames (%d speakers, %.1f ms/frame)\n",
                (long)n_frames, n_spk, stream.seconds_per_frame() * 1000.0);

    auto ref = load_ref_probs(ref_probs_path);
    int64_t ref_frames = (ref.shape.size() >= 2) ? ref.shape[ref.shape.size() - 2] : ref.numel() / n_spk;
    std::printf("[streaming test] ref has %ld frames (%zu elements)\n", (long)ref_frames, ref.data.size());

    if (n_frames != ref_frames) {
        std::fprintf(stderr, "FAILURE: frame count mismatch (ours %ld vs ref %ld)\n",
                     (long)n_frames, (long)ref_frames);
        return 1;
    }

    size_t total_elements = static_cast<size_t>(n_frames) * n_spk;
    if (probs.size() < total_elements || ref.data.size() < total_elements) {
        std::fprintf(stderr, "FAILURE: buffer size mismatch (ours %zu vs ref %zu, want %zu)\n",
                     probs.size(), ref.data.size(), total_elements);
        return 1;
    }

    float max_diff = 0.0f;
    double sum_diff = 0.0;
    size_t count_gt_05 = 0;
    size_t count_gt_01 = 0;
    for (size_t i = 0; i < total_elements; i++) {
        float d = std::fabs(probs[i] - ref.data[i]);
        if (d > max_diff) max_diff = d;
        sum_diff += d;
        if (d > 0.05f) count_gt_05++;
        if (d > 0.01f) count_gt_01++;
    }
    double mean_diff = sum_diff / total_elements;

    std::printf("max_abs_diff: %.6e\n", max_diff);
    std::printf("mean_abs_diff: %.6e\n", mean_diff);
    std::printf("diff > 0.01: %zu / %zu (%.2f%%)\n", count_gt_01, total_elements,
                100.0 * count_gt_01 / total_elements);
    std::printf("diff > 0.05: %zu / %zu (%.2f%%)\n", count_gt_05, total_elements,
                100.0 * count_gt_05 / total_elements);

    // Compute speaker correlations across timeline
    std::vector<float> ours_spk(n_frames);
    std::vector<float> ref_spk(n_frames);
    bool corr_ok = true;
    for (int s = 0; s < std::min(n_spk, 2); s++) {
        for (int64_t f = 0; f < n_frames; f++) {
            ours_spk[f] = probs[f * n_spk + s];
            ref_spk[f] = ref.data[f * n_spk + s];
        }
        double r = compute_correlation(ours_spk.data(), ref_spk.data(), n_frames);
        std::printf("speaker %d correlation: %.6f\n", s, r);
        if (r < 0.99) {
            corr_ok = false;
        }
    }

    // Segments check
    auto segs = stream.segments();
    std::printf("[streaming test] produced %zu segments:\n", segs.size());
    for (const auto& s : segs) {
        std::printf("  [%6.2fs - %6.2fs] speaker %d\n", s.t0, s.t1, s.speaker);
    }

    bool pass = (mean_diff < 0.01) && (count_gt_05 < total_elements * 0.03) && corr_ok && (segs.size() >= 4);

    if (pass) {
        std::printf("SUCCESS: Nemotron-3 streaming parity verified within tolerance!\n");
        return 0;
    } else {
        std::fprintf(stderr, "FAILURE: streaming parity did not meet acceptance criteria\n");
        return 1;
    }
}
