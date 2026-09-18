// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "batching.h"
#include "nn.h"
#include "runtime.h"
#include "sortformer_model.h"

namespace nemo_speech::asr {

struct Nemotron3TransformerConfig {
    int n_layers = 31;
    int d_model = 512;
    int n_heads = 8;
    int head_dim = 64;
    int ffn_size = 2048;
    int pos_emb_max_len = 5000;
    float rope_base = 10000.0f;
};

struct Nemotron3ModelConfig {
    Nemotron3TransformerConfig transformer;
    int num_speakers = 8;
    DiarScoringConfig scoring;

    int tf_d_model = 192;
    int fc_d_model = 512;
    int subpixel_kernel = 3;
    int subsampling_factor = 8;
    bool high_resolution = true;
    bool use_learnable_sil_emb = true;

    // FE
    int sample_rate = 16000;
    float window_size = 0.025f;
    float window_stride = 0.01f;
    int n_fft = 512;
    int n_mels = 128;
    float preemph = 0.97f;
    float log_zero_guard = 5.9604645e-8f;
};

// One pre-LN RoPE Transformer block (31 in total):
//   x -> norm1 -> w_qkv -> RoPE -> MHA -> out_proj -> +res1
//     -> norm2 -> ffn_net_0 -> gelu_erf -> ffn_net_3 -> +res2
class Nemotron3Block : public ggml_runtime::Module {
   public:
    Nemotron3Block(const std::string& name, const Nemotron3TransformerConfig& cfg);
    ~Nemotron3Block();

    void define_tensors(ggml_runtime::Session* session) override;
    ggml_runtime::TensorBag build_graph(
        ggml_runtime::Session* session, ggml_runtime::TensorBag input_tensors,
        ggml_runtime::TensorContainer* tc) override;
    void set_data(ggml_runtime::Session* session) override;

   private:
    std::string name_;
    Nemotron3TransformerConfig cfg_;
    ggml_runtime::LayerNorm* norm1_;
    ggml_runtime::Linear* w_qkv_;
    ggml_runtime::Linear* out_proj_;
    ggml_runtime::LayerNorm* norm2_;
    ggml_runtime::Linear* ffn_net_0_;
    ggml_runtime::Linear* ffn_net_3_;
};

// Root Module for Nemotron-3 Diarization graph.
// Per-call inputs:
//   input.mel            (n_mels, T_mel, 1, B)
//   input.state          (512, L_state, B) - optional compact state (spkcache + fifo)
//   input.attention_mask (total_frames, 1, 1, B) - optional
// Output bag:
//   [0] preds (num_speakers, total_frames * 8, B) - 10ms high-resolution
//   [1] chunk_embs (512, T_chunk, B) - 80ms encoder embeddings of the new chunk
class Nemotron3Graph : public ggml_runtime::Module {
   public:
    explicit Nemotron3Graph(const Nemotron3ModelConfig& cfg);
    ~Nemotron3Graph();

    void define_tensors(ggml_runtime::Session* session) override;
    ggml_runtime::TensorBag build_graph(
        ggml_runtime::Session* session, ggml_runtime::TensorBag input_tensors,
        ggml_runtime::TensorContainer* tc) override;
    void set_data(ggml_runtime::Session* session) override;

   private:
    Nemotron3ModelConfig cfg_;
    ggml_runtime::Linear* pre_proj_;
    ggml_runtime::LayerNorm* embed_norm_;
    std::vector<Nemotron3Block*> layers_;
    ggml_runtime::LayerNorm* final_norm_;
    ggml_runtime::Linear* encoder_proj_;
    ggml_runtime::Conv1D* subpixel_conv_;
    ggml_runtime::Linear* head_hidden_;
    ggml_runtime::Linear* head_spks_;
};

class Nemotron3Model {
   public:
    Nemotron3Model(
        ggml_runtime::BackendManager& bm, const std::string& gguf_path,
        const BatchingConfig& batching = {});
    ~Nemotron3Model();

    const Nemotron3ModelConfig& cfg() const { return cfg_; }
    const std::vector<float>& mel_basis() const { return mel_basis_; }
    const std::vector<float>& learnable_sil_emb() const { return learnable_sil_emb_; }
    ggml_runtime::Session* session() const { return session_.get(); }

    int subsampled_len(int t_mel) const {
        return t_mel / cfg_.subsampling_factor;
    }

    struct ChunkOutput {
        std::vector<float> preds;             // (total_frames * 8) x n_spk, frame-major (10ms)
        std::vector<float> preds_downsampled; // total_frames x n_spk, frame-major (80ms avg)
        int total_frames = 0;                 // encoder frames (each = 8 pred frames)
        std::vector<float> chunk_embs;        // chunk_frames x 512, frame-major
        int chunk_frames = 0;                 // encoder frames of new audio
        std::vector<float> dbg_embed_norm;
        std::vector<float> dbg_layer0;
        std::vector<float> dbg_final_norm;
        std::vector<float> dbg_proj;
        std::vector<float> dbg_sub;
    };

    ChunkOutput run_chunk(
        const float* mel, int t_mel, const float* spkcache, int spkcache_frames, const float* fifo,
        int fifo_frames);
    BatchMetrics batch_metrics() const;

   private:
    class Nemotron3Batcher;

    Nemotron3ModelConfig cfg_;
    std::vector<float> mel_basis_;
    std::vector<float> learnable_sil_emb_;
    std::unique_ptr<ggml_runtime::GGUFLoader> loader_;
    std::unique_ptr<Nemotron3Graph> graph_;
    std::unique_ptr<ggml_runtime::Session> session_;
    std::unique_ptr<Nemotron3Batcher> batcher_;
};

}  // namespace nemo_speech::asr
