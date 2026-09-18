// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "nemotron3_model.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

using namespace nemo_speech::asr;

static Nemotron3ModelConfig
parse_config(const ggml_runtime::GGUFLoader& loader) {
    Nemotron3ModelConfig cfg;

    Nemotron3TransformerConfig& t = cfg.transformer;
    t.n_layers = loader.get_u32("nemotron3_diar.encoder.n_layers", 31);
    t.d_model = loader.get_u32("nemotron3_diar.encoder.d_model", 512);
    t.n_heads = loader.get_u32("nemotron3_diar.encoder.n_heads", 8);
    t.head_dim = t.d_model / t.n_heads;
    t.ffn_size = loader.get_u32("nemotron3_diar.encoder.ffn_size", 2048);
    t.pos_emb_max_len = loader.get_u32("nemotron3_diar.encoder.pos_emb_max_len", 5000);
    t.rope_base = 10000.0f;

    cfg.num_speakers = loader.get_u32("nemotron3_diar.num_speakers", 8);
    cfg.tf_d_model = loader.get_u32("nemotron3_diar.head.tf_d_model", 192);
    cfg.fc_d_model = loader.get_u32("nemotron3_diar.head.fc_d_model", 512);
    cfg.subsampling_factor = loader.get_u32("nemotron3_diar.encoder.subsampling_factor", 8);
    cfg.high_resolution = loader.get_bool("nemotron3_diar.high_resolution", true);
    cfg.use_learnable_sil_emb = loader.get_bool("nemotron3_diar.scoring.use_learnable_sil_emb", true);

    DiarScoringConfig& s = cfg.scoring;
    s.sil_frames_per_spk = loader.get_u32("nemotron3_diar.scoring.spkcache_sil_frames_per_spk", 1);
    s.pred_score_threshold = loader.get_f32("nemotron3_diar.scoring.pred_score_threshold", 0.25f);
    s.scores_boost_latest = loader.get_f32("nemotron3_diar.scoring.scores_boost_latest", 0.05f);
    s.sil_threshold = loader.get_f32("nemotron3_diar.scoring.sil_threshold", 0.2f);
    s.strong_boost_rate = loader.get_f32("nemotron3_diar.scoring.strong_boost_rate", 0.75f);
    s.weak_boost_rate = loader.get_f32("nemotron3_diar.scoring.weak_boost_rate", 1.5f);
    s.min_pos_scores_rate = loader.get_f32("nemotron3_diar.scoring.min_pos_scores_rate", 0.5f);

    cfg.sample_rate = loader.get_u32("nemotron3_diar.preprocessor.sample_rate", 16000);
    cfg.window_size = loader.get_f32("nemotron3_diar.preprocessor.window_size", 0.025f);
    cfg.window_stride = loader.get_f32("nemotron3_diar.preprocessor.window_stride", 0.01f);
    cfg.n_fft = loader.get_u32("nemotron3_diar.preprocessor.n_fft", 512);
    cfg.n_mels = loader.get_u32("nemotron3_diar.preprocessor.features", 128);
    cfg.preemph = loader.get_f32("nemotron3_diar.preprocessor.preemph", 0.97f);
    cfg.log_zero_guard = loader.get_f32("nemotron3_diar.preprocessor.log_zero_guard", cfg.log_zero_guard);

    return cfg;
}

Nemotron3Block::Nemotron3Block(const std::string& name, const Nemotron3TransformerConfig& cfg)
    : name_(name), cfg_(cfg) {
    const int64_t ln_shape[4] = {cfg.d_model, 1, 1, 1};
    norm1_ = new ggml_runtime::LayerNorm(name + ".norm1", ln_shape);
    w_qkv_ = new ggml_runtime::Linear(name + ".attn.w_qkv", cfg.d_model, cfg.d_model * 3, /*use_bias=*/false);
    out_proj_ = new ggml_runtime::Linear(name + ".attn.out_proj", cfg.d_model, cfg.d_model, /*use_bias=*/true);
    norm2_ = new ggml_runtime::LayerNorm(name + ".norm2", ln_shape);
    ffn_net_0_ = new ggml_runtime::Linear(name + ".ffn.net.0", cfg.d_model, cfg.ffn_size, /*use_bias=*/true);
    ffn_net_3_ = new ggml_runtime::Linear(name + ".ffn.net.3", cfg.ffn_size, cfg.d_model, /*use_bias=*/true);
}

Nemotron3Block::~Nemotron3Block() {
    delete norm1_;
    delete w_qkv_;
    delete out_proj_;
    delete norm2_;
    delete ffn_net_0_;
    delete ffn_net_3_;
}

void
Nemotron3Block::define_tensors(ggml_runtime::Session* session) {
    norm1_->define_tensors(session);
    w_qkv_->define_tensors(session);
    out_proj_->define_tensors(session);
    norm2_->define_tensors(session);
    ffn_net_0_->define_tensors(session);
    ffn_net_3_->define_tensors(session);
}

void
Nemotron3Block::set_data(ggml_runtime::Session* session) {
    norm1_->set_data(session);
    w_qkv_->set_data(session);
    out_proj_->set_data(session);
    norm2_->set_data(session);
    ffn_net_0_->set_data(session);
    ffn_net_3_->set_data(session);
}

ggml_runtime::TensorBag
Nemotron3Block::build_graph(
    ggml_runtime::Session* session, ggml_runtime::TensorBag input_tensors,
    ggml_runtime::TensorContainer* tc) {
    auto x = input_tensors.get_tensor(0);
    auto bf_ctx = tc->get_ctx_of_buffer_type(x.buft);
    ggml_context* ctx = bf_ctx.ctx;

    const int n_head = cfg_.n_heads;
    const int d_k = cfg_.d_model / n_head;
    const int64_t T = x.tensor->ne[1];
    const int64_t B = x.tensor->ne[2];

    // Pre-LN 1
    ggml_runtime::TensorBag ln1_in;
    ln1_in.add_tensor(x);
    auto x_n1 = norm1_->build_graph(session, ln1_in, tc).get_tensor(0);

    // QKV projection (1536, T, B)
    ggml_runtime::TensorBag qkv_in;
    qkv_in.add_tensor(x_n1);
    auto qkv = w_qkv_->build_graph(session, qkv_in, tc).get_tensor(0);

    // Slice Q, K, V
    auto q_view = ggml_view_3d(ctx, qkv.tensor, cfg_.d_model, T, B, qkv.tensor->nb[1], qkv.tensor->nb[2], 0);
    auto k_view = ggml_view_3d(ctx, qkv.tensor, cfg_.d_model, T, B, qkv.tensor->nb[1], qkv.tensor->nb[2], cfg_.d_model * sizeof(float));
    auto v_view = ggml_view_3d(ctx, qkv.tensor, cfg_.d_model, T, B, qkv.tensor->nb[1], qkv.tensor->nb[2], 2 * cfg_.d_model * sizeof(float));

    auto q_cont = ggml_cont(ctx, q_view);
    auto k_cont = ggml_cont(ctx, k_view);
    auto v_cont = ggml_cont(ctx, v_view);

    auto q_mh = ggml_reshape_4d(ctx, q_cont, d_k, n_head, T, B);
    auto k_mh = ggml_reshape_4d(ctx, k_cont, d_k, n_head, T, B);
    auto v_mh = ggml_reshape_4d(ctx, v_cont, d_k, n_head, T, B);

    // RoPE (NeoX, rotate half [-x2, x1])
    ggml_tensor* pos = ggml_arange(ctx, 0.0f, static_cast<float>(T), 1.0f);
    pos = ggml_cast(ctx, pos, GGML_TYPE_I32);
    auto q_rot = ggml_rope_ext(
        ctx, q_mh, pos, nullptr, d_k, GGML_ROPE_TYPE_NEOX, static_cast<int>(T),
        cfg_.rope_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    auto k_rot = ggml_rope_ext(
        ctx, k_mh, pos, nullptr, d_k, GGML_ROPE_TYPE_NEOX, static_cast<int>(T),
        cfg_.rope_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

    auto q_p = ggml_cont(ctx, ggml_permute(ctx, q_rot, 0, 2, 1, 3));
    auto k_p = ggml_cont(ctx, ggml_permute(ctx, k_rot, 0, 2, 1, 3));

    auto scores = ggml_mul_mat(ctx, k_p, q_p);
    scores = ggml_scale_inplace(ctx, scores, 1.0f / std::sqrt(static_cast<float>(d_k)));
    if (input_tensors.tensor_count() >= 2) {
        scores = ggml_add(ctx, scores, input_tensors.get_tensor(1).tensor);
    }
    auto probs = ggml_soft_max_inplace(ctx, scores);

    auto v_t = ggml_cont(ctx, ggml_permute(ctx, ggml_permute(ctx, v_mh, 2, 1, 0, 3), 0, 2, 1, 3));
    auto attn = ggml_permute(ctx, ggml_mul_mat(ctx, v_t, probs), 0, 2, 1, 3);
    auto merged = ggml_reshape_3d(ctx, ggml_cont(ctx, attn), n_head * d_k, T, B);

    ggml_runtime::TensorBag proj_in;
    proj_in.add_tensor(ggml_runtime::ggml_bf_tensor(merged, x.buft));
    auto attn_out = out_proj_->build_graph(session, proj_in, tc).get_tensor(0);

    // Residual 1
    auto h1 = ggml_add(ctx, x.tensor, attn_out.tensor);

    // Pre-LN 2
    ggml_runtime::TensorBag ln2_in;
    ln2_in.add_tensor(ggml_runtime::ggml_bf_tensor(h1, x.buft));
    auto x_n2 = norm2_->build_graph(session, ln2_in, tc).get_tensor(0);

    // FFN
    ggml_runtime::TensorBag ffn_in;
    ffn_in.add_tensor(x_n2);
    auto ffn_0 = ffn_net_0_->build_graph(session, ffn_in, tc).get_tensor(0);
    auto gelu = ggml_gelu_erf(ctx, ffn_0.tensor);
    ggml_runtime::TensorBag gelu_bag;
    gelu_bag.add_tensor(ggml_runtime::ggml_bf_tensor(gelu, ffn_0.buft));
    auto ffn_3 = ffn_net_3_->build_graph(session, gelu_bag, tc).get_tensor(0);

    // Residual 2
    auto out = ggml_add(ctx, h1, ffn_3.tensor);

    ggml_runtime::TensorBag ret;
    ret.add_tensor(ggml_runtime::ggml_bf_tensor(out, x.buft));
    return ret;
}

Nemotron3Graph::Nemotron3Graph(const Nemotron3ModelConfig& cfg) : cfg_(cfg) {
    pre_proj_ = new ggml_runtime::Linear(
        "encoder.pre_encode.proj", cfg.n_mels * cfg.subsampling_factor, cfg.transformer.d_model, /*use_bias=*/false);
    const int64_t ln_shape[4] = {cfg.transformer.d_model, 1, 1, 1};
    embed_norm_ = new ggml_runtime::LayerNorm("encoder.embed_norm", ln_shape);

    layers_.reserve(cfg.transformer.n_layers);
    for (int i = 0; i < cfg.transformer.n_layers; ++i) {
        layers_.push_back(new Nemotron3Block("encoder.layers." + std::to_string(i), cfg.transformer));
    }

    final_norm_ = new ggml_runtime::LayerNorm("encoder.final_norm", ln_shape);
    encoder_proj_ = new ggml_runtime::Linear("encoder_proj", cfg.transformer.d_model, cfg.tf_d_model, /*use_bias=*/true);

    subpixel_conv_ = new ggml_runtime::Conv1D(
        "head.subpixel_upsample", cfg.tf_d_model, cfg.tf_d_model * cfg.subsampling_factor,
        cfg.subpixel_kernel, /*stride=*/1, /*padding=*/1, /*dilation=*/1, /*use_bias=*/true, /*is_dw=*/false);

    head_hidden_ = new ggml_runtime::Linear("head.first_hidden_to_hidden", cfg.tf_d_model, cfg.tf_d_model, /*use_bias=*/true);
    head_spks_ = new ggml_runtime::Linear("head.single_hidden_to_spks", cfg.tf_d_model, cfg.num_speakers, /*use_bias=*/true);
}

Nemotron3Graph::~Nemotron3Graph() {
    delete pre_proj_;
    delete embed_norm_;
    for (auto* l : layers_) delete l;
    delete final_norm_;
    delete encoder_proj_;
    delete subpixel_conv_;
    delete head_hidden_;
    delete head_spks_;
}

void
Nemotron3Graph::define_tensors(ggml_runtime::Session* session) {
    pre_proj_->define_tensors(session);
    embed_norm_->define_tensors(session);
    for (auto* l : layers_) l->define_tensors(session);
    final_norm_->define_tensors(session);
    encoder_proj_->define_tensors(session);
    subpixel_conv_->define_tensors(session);
    head_hidden_->define_tensors(session);
    head_spks_->define_tensors(session);
}

void
Nemotron3Graph::set_data(ggml_runtime::Session* session) {
    pre_proj_->set_data(session);
    embed_norm_->set_data(session);
    for (auto* l : layers_) l->set_data(session);
    final_norm_->set_data(session);
    encoder_proj_->set_data(session);
    subpixel_conv_->set_data(session);
    head_hidden_->set_data(session);
    head_spks_->set_data(session);
}

ggml_runtime::TensorBag
Nemotron3Graph::build_graph(
    ggml_runtime::Session* session, ggml_runtime::TensorBag input_tensors,
    ggml_runtime::TensorContainer* tc) {
    (void)input_tensors;
    auto mel = tc->get_tensor_by_name("input.mel");
    auto bf_ctx = tc->get_ctx_of_buffer_type(mel.buft);
    ggml_context* ctx = bf_ctx.ctx;

    const int64_t T_mel = mel.tensor->ne[1];
    const int64_t B = mel.tensor->ne[3];
    const int64_t T_chunk = T_mel / cfg_.subsampling_factor;
    const int64_t T_mel_valid = T_chunk * cfg_.subsampling_factor;

    ggml_tensor* mel_in = mel.tensor;
    if (T_mel != T_mel_valid) {
        mel_in = ggml_view_3d(
            ctx, mel.tensor, cfg_.n_mels, T_mel_valid, B,
            mel.tensor->nb[1], mel.tensor->nb[3], 0);
    }

    // 1. Frontend: 8-frame stacking (128 -> 1024) + Linear(1024, 512)
    auto mel_stacked = ggml_reshape_3d(ctx, mel_in, cfg_.n_mels * cfg_.subsampling_factor, T_chunk, B);
    ggml_runtime::TensorBag pre_in;
    pre_in.add_tensor(ggml_runtime::ggml_bf_tensor(mel_stacked, mel.buft));
    auto pre = pre_proj_->build_graph(session, pre_in, tc);
    auto chunk_embs = pre.get_tensor(0);

    // 2. Concat compact state (spkcache + fifo) and current chunk
    ggml_tensor* x = chunk_embs.tensor;
    if (tc->has_tensor_by_name("input.state")) {
        x = ggml_concat(ctx, tc->get_tensor_by_name("input.state").tensor, x, 1);
    }
    const int64_t total_frames = x->ne[1];

    // 3. embed_norm on the concatenated sequence
    ggml_runtime::TensorBag emb_norm_in;
    emb_norm_in.add_tensor(ggml_runtime::ggml_bf_tensor(x, chunk_embs.buft));
    auto h = embed_norm_->build_graph(session, emb_norm_in, tc).get_tensor(0);
    auto h_embed_norm = h;
    ggml_runtime::ggml_bf_tensor h_layer0(nullptr, nullptr);

    // 4. 31 Transformer blocks
    for (size_t i = 0; i < layers_.size(); ++i) {
        ggml_runtime::TensorBag layer_in;
        layer_in.add_tensor(h);
        if (tc->has_tensor_by_name("input.attention_mask")) {
            layer_in.add_tensor(tc->get_tensor_by_name("input.attention_mask"));
        }
        h = layers_[i]->build_graph(session, layer_in, tc).get_tensor(0);
        if (i == 0) h_layer0 = h;
    }

    // 5. final_norm
    ggml_runtime::TensorBag fn_in;
    fn_in.add_tensor(h);
    auto fn_out = final_norm_->build_graph(session, fn_in, tc).get_tensor(0);

    // 6. encoder_proj (512 -> 192)
    ggml_runtime::TensorBag ep_in;
    ep_in.add_tensor(fn_out);
    auto proj = encoder_proj_->build_graph(session, ep_in, tc).get_tensor(0);

    // 7. subpixel Conv1D: transpose (192, total_frames, B) -> (total_frames, 192, B)
    auto proj_t = ggml_cont(ctx, ggml_permute(ctx, proj.tensor, 1, 0, 2, 3));
    ggml_runtime::TensorBag conv_in;
    conv_in.add_tensor(ggml_runtime::ggml_bf_tensor(proj_t, proj.buft));
    auto sub = subpixel_conv_->build_graph(session, conv_in, tc).get_tensor(0);

    // 8. Pixel shuffle upsampling: (total_frames, 1536, B) -> (192, total_frames * 8, B)
    auto sub_4d = ggml_reshape_4d(ctx, sub.tensor, total_frames, cfg_.tf_d_model, cfg_.subsampling_factor, B);
    auto perm = ggml_permute(ctx, sub_4d, 2, 0, 1, 3);
    auto cont = ggml_cont(ctx, perm);
    auto upsampled = ggml_reshape_3d(ctx, cont, cfg_.tf_d_model, total_frames * cfg_.subsampling_factor, B);

    // 9. relu -> first_hidden_to_hidden -> relu -> single_hidden_to_spks -> sigmoid
    auto act1 = ggml_relu(ctx, upsampled);
    ggml_runtime::TensorBag h_in1;
    h_in1.add_tensor(ggml_runtime::ggml_bf_tensor(act1, sub.buft));
    auto h2 = head_hidden_->build_graph(session, h_in1, tc).get_tensor(0);

    auto act2 = ggml_relu(ctx, h2.tensor);
    ggml_runtime::TensorBag h_in2;
    h_in2.add_tensor(ggml_runtime::ggml_bf_tensor(act2, h2.buft));
    auto logits = head_spks_->build_graph(session, h_in2, tc).get_tensor(0);
    auto preds = ggml_sigmoid(ctx, logits.tensor);

    ggml_runtime::TensorBag out;
    out.add_tensor(ggml_runtime::ggml_bf_tensor(preds, logits.buft));
    out.add_tensor(chunk_embs);
    out.add_tensor(h_embed_norm);
    out.add_tensor(h_layer0);
    out.add_tensor(fn_out);
    out.add_tensor(proj);
    out.add_tensor(sub);
    return out;
}

class Nemotron3Model::Nemotron3Batcher {
   public:
    struct Key {
        int t_mel = 0;
        bool operator==(const Key& other) const { return t_mel == other.t_mel; }
    };

    struct Request {
        std::vector<float> mel;
        std::vector<float> spkcache;
        std::vector<float> fifo;
        int spkcache_frames = 0;
        int fifo_frames = 0;
    };

    Nemotron3Batcher(Nemotron3Model* model, const BatchingConfig& batching)
        : model_(model), queue_(batching, [this](const Key& key, std::vector<Request>&& requests) {
              return execute(key, std::move(requests));
          }) {}

    ChunkOutput run(
        const float* mel, int t_mel, const float* spkcache, int spkcache_frames, const float* fifo,
        int fifo_frames) {
        const int sub_factor = model_->cfg_.subsampling_factor;
        t_mel = (t_mel / sub_factor) * sub_factor;
        const int d = model_->cfg_.transformer.d_model;
        Request request;
        request.mel.assign(
            mel, mel + static_cast<size_t>(model_->cfg_.n_mels) * static_cast<size_t>(t_mel));
        if (spkcache_frames > 0) {
            request.spkcache.assign(spkcache, spkcache + static_cast<size_t>(d) * spkcache_frames);
        }
        if (fifo_frames > 0)
            request.fifo.assign(fifo, fifo + static_cast<size_t>(d) * fifo_frames);
        request.spkcache_frames = spkcache_frames;
        request.fifo_frames = fifo_frames;
        return queue_.run({t_mel}, std::move(request));
    }

    BatchMetrics metrics() const { return queue_.metrics(); }

   private:
    std::vector<ChunkOutput> execute(const Key& key, std::vector<Request>&& requests) {
        const int B = static_cast<int>(requests.size());
        const int d = model_->cfg_.transformer.d_model;
        const int n_mels = model_->cfg_.n_mels;
        const int n_spk = model_->cfg_.num_speakers;
        const int sub_factor = model_->cfg_.subsampling_factor;
        const int t_chunk = key.t_mel / sub_factor;

        int max_state_frames = 0;
        for (const auto& request : requests) {
            max_state_frames =
                std::max(max_state_frames, request.spkcache_frames + request.fifo_frames);
        }
        const int total_enc_frames = max_state_frames + t_chunk;
        const int total_pred_frames = total_enc_frames * sub_factor;

        const size_t mel_item = static_cast<size_t>(n_mels) * key.t_mel;
        const size_t state_item = static_cast<size_t>(d) * max_state_frames;

        std::vector<float> mel(mel_item * B);
        std::vector<float> state(state_item * B, 0.0f);
        bool needs_padding_mask = false;
        for (int b = 0; b < B; ++b) {
            const auto& request = requests[static_cast<size_t>(b)];
            std::memcpy(
                mel.data() + static_cast<size_t>(b) * mel_item, request.mel.data(),
                mel_item * sizeof(float));
            const int state_frames = request.spkcache_frames + request.fifo_frames;
            if (state_frames < max_state_frames)
                needs_padding_mask = true;
            if (state_frames > 0) {
                const int state_offset = max_state_frames - state_frames;
                float* dst = state.data() + static_cast<size_t>(b) * state_item +
                             static_cast<size_t>(state_offset) * d;
                if (request.spkcache_frames > 0) {
                    std::memcpy(
                        dst, request.spkcache.data(),
                        static_cast<size_t>(request.spkcache_frames) * d * sizeof(float));
                    dst += static_cast<size_t>(request.spkcache_frames) * d;
                }
                if (request.fifo_frames > 0) {
                    std::memcpy(
                        dst, request.fifo.data(),
                        static_cast<size_t>(request.fifo_frames) * d * sizeof(float));
                }
            }
        }

        std::vector<ggml_runtime::Session::Input> inputs;
        inputs.push_back({"input.mel", GGML_TYPE_F32, mel.data(), {n_mels, key.t_mel, 1, B}});
        if (max_state_frames > 0) {
            inputs.push_back(
                {"input.state", GGML_TYPE_F32, state.data(), {d, max_state_frames, B}});
        }
        std::vector<float> attention_mask;
        if (needs_padding_mask) {
            attention_mask.assign(static_cast<size_t>(total_enc_frames) * B, 0.0f);
            for (int b = 0; b < B; ++b) {
                const auto& request = requests[static_cast<size_t>(b)];
                const int state_frames = request.spkcache_frames + request.fifo_frames;
                const int state_offset = max_state_frames - state_frames;
                auto mask_base = static_cast<size_t>(b) * total_enc_frames;
                std::fill_n(attention_mask.begin() + mask_base, state_offset, -1e9f);
            }
            inputs.push_back(
                {"input.attention_mask", GGML_TYPE_F32, attention_mask.data(), {total_enc_frames, 1, 1, B}});
        }

        const size_t preds_item = static_cast<size_t>(total_pred_frames) * n_spk;
        const size_t embs_item = static_cast<size_t>(t_chunk) * d;
        std::vector<float> preds(preds_item * B);
        std::vector<float> embs(embs_item * B);
        std::vector<float> d_en(total_enc_frames * d);
        std::vector<float> d_l0(total_enc_frames * d);
        std::vector<float> d_fn(total_enc_frames * d);
        std::vector<float> d_pj(total_enc_frames * 192);
        std::vector<float> d_sb(total_enc_frames * 1536);

        std::vector<ggml_runtime::Session::Output> outputs(7);
        outputs[0].index = 0;
        outputs[0].host_buffer = preds.data();
        outputs[0].nbytes = preds.size() * sizeof(float);
        outputs[1].index = 1;
        outputs[1].host_buffer = embs.data();
        outputs[1].nbytes = embs.size() * sizeof(float);
        outputs[2].index = 2;
        outputs[2].host_buffer = d_en.data();
        outputs[2].nbytes = d_en.size() * sizeof(float);
        outputs[3].index = 3;
        outputs[3].host_buffer = d_l0.data();
        outputs[3].nbytes = d_l0.size() * sizeof(float);
        outputs[4].index = 4;
        outputs[4].host_buffer = d_fn.data();
        outputs[4].nbytes = d_fn.size() * sizeof(float);
        outputs[5].index = 5;
        outputs[5].host_buffer = d_pj.data();
        outputs[5].nbytes = d_pj.size() * sizeof(float);
        outputs[6].index = 6;
        outputs[6].host_buffer = d_sb.data();
        outputs[6].nbytes = d_sb.size() * sizeof(float);
        model_->session_->run(inputs, outputs);

        std::vector<ChunkOutput> results(static_cast<size_t>(B));
        for (int b = 0; b < B; ++b) {
            auto& result = results[static_cast<size_t>(b)];
            const auto& request = requests[static_cast<size_t>(b)];
            const int state_frames = request.spkcache_frames + request.fifo_frames;
            const int state_offset = max_state_frames - state_frames;
            result.total_frames = state_frames + t_chunk;
            result.chunk_frames = t_chunk;

            // High-resolution predictions slice (10ms)
            const int pred_offset = state_offset * sub_factor;
            const int valid_pred_frames = result.total_frames * sub_factor;
            const auto pred_begin = preds.begin() + static_cast<size_t>(b) * preds_item +
                                    static_cast<size_t>(pred_offset) * n_spk;
            result.preds.assign(
                pred_begin, pred_begin + static_cast<size_t>(valid_pred_frames) * n_spk);

            // Compute downsampled predictions for state compression (80ms avg)
            result.preds_downsampled.assign(static_cast<size_t>(result.total_frames) * n_spk, 0.0f);
            for (int f = 0; f < result.total_frames; ++f) {
                for (int s = 0; s < sub_factor; ++s) {
                    for (int spk = 0; spk < n_spk; ++spk) {
                        result.preds_downsampled[f * n_spk + spk] +=
                            result.preds[(f * sub_factor + s) * n_spk + spk] / static_cast<float>(sub_factor);
                    }
                }
            }

            const auto emb_begin = embs.begin() + static_cast<size_t>(b) * embs_item;
            result.chunk_embs.assign(emb_begin, emb_begin + embs_item);
            result.dbg_embed_norm = d_en;
            result.dbg_layer0 = d_l0;
            result.dbg_final_norm = d_fn;
            result.dbg_proj = d_pj;
            result.dbg_sub = d_sb;
        }
        return results;
    }

    Nemotron3Model* model_;
    MicroBatcher<Key, Request, ChunkOutput> queue_; 
};

Nemotron3Model::Nemotron3Model(
    ggml_runtime::BackendManager& bm, const std::string& gguf_path,
    const BatchingConfig& batching) {
    loader_ = std::make_unique<ggml_runtime::GGUFLoader>(gguf_path);
    cfg_ = parse_config(*loader_);

    const int n_freq = cfg_.n_fft / 2 + 1;
    mel_basis_.resize(static_cast<size_t>(cfg_.n_mels) * n_freq);
    const char* fb = loader_->get_tensor_file_data("preprocessor.fb", mel_basis_.size() * sizeof(float));
    std::memcpy(mel_basis_.data(), fb, mel_basis_.size() * sizeof(float));

    if (loader_->has_tensor("learnable_sil_emb")) {
        learnable_sil_emb_.resize(cfg_.transformer.d_model);
        const char* se = loader_->get_tensor_file_data("learnable_sil_emb", learnable_sil_emb_.size() * sizeof(float));
        std::memcpy(learnable_sil_emb_.data(), se, learnable_sil_emb_.size() * sizeof(float));
    }

    graph_ = std::make_unique<Nemotron3Graph>(cfg_);
    session_ = std::make_unique<ggml_runtime::Session>(bm, graph_.get(), loader_.get());
    session_->set_run_cache_capacity(48);
    session_->setup();
    batcher_ = std::make_unique<Nemotron3Batcher>(this, batching);
}

Nemotron3Model::~Nemotron3Model() = default;

Nemotron3Model::ChunkOutput
Nemotron3Model::run_chunk(
    const float* mel, int t_mel, const float* spkcache, int spkcache_frames, const float* fifo,
    int fifo_frames) {
    return batcher_->run(mel, t_mel, spkcache, spkcache_frames, fifo, fifo_frames);
}

BatchMetrics
Nemotron3Model::batch_metrics() const {
    return batcher_->metrics();
}
