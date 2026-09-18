# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Nemotron-3-Diarization-preview -> GGUF (F32 first).

Architecture differs from Sortformer v2: feature_stacking+Linear frontend,
31-layer pre-LN RoPE Transformer (d512, ffn 2048), 192-dim head with
subpixel Conv1D upsampling to 10 ms, 8 speakers, learnable silence embedding.
"""
from __future__ import annotations

import tarfile
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import yaml

ARCH = "nemotron3_diar"

SKIP_EXACT = {
    "preprocessor.featurizer.window",
    "sortformer_modules.hidden_to_spks.weight",
    "sortformer_modules.hidden_to_spks.bias",
}
SKIP_PREFIX = ("spec_augmentation.", "loss.")


def remap(name: str) -> Optional[str]:
    if name in SKIP_EXACT or name.startswith(SKIP_PREFIX) or name.endswith(".num_batches_tracked"):
        return None
    if name == "preprocessor.featurizer.fb":
        return "preprocessor.fb"
    if name.startswith("sortformer_modules.encoder_proj."):
        return name.replace("sortformer_modules.encoder_proj.", "encoder_proj.", 1)
    if name.startswith("sortformer_modules.learnable_sil_emb"):
        return "learnable_sil_emb"
    if name.startswith("sortformer_modules."):
        return name.replace("sortformer_modules.", "head.", 1)
    if name.startswith("encoder."):
        return name
    print(f"[convert] WARNING: unrecognized tensor skipped: {name}")
    return None


def _load(nemo_path: Path):
    t = tarfile.open(nemo_path)
    cfg = yaml.safe_load(t.extractfile("model_config.yaml").read().decode("utf-8"))
    sd = torch.load(t.extractfile("model_weights.ckpt"), map_location="cpu", weights_only=False)
    return cfg, sd


def _metadata(w, cfg):
    from gguf import GGUFWriter  # noqa: F401

    def kv(k, v):
        if isinstance(v, bool):
            w.add_bool(k, v)
        elif isinstance(v, int):
            w.add_uint32(k, v)
        elif isinstance(v, float):
            w.add_float32(k, v)
        else:
            w.add_string(k, str(v))

    w.add_name("nemotron-3-diarization-preview")
    kv(f"{ARCH}.num_speakers", int(cfg["max_num_of_spks"]))
    kv(f"{ARCH}.high_resolution", bool(cfg["high_resolution"]))
    kv(f"{ARCH}.output_subsampling_factor", int(cfg["output_subsampling_factor"]))
    kv(f"{ARCH}.streaming_mode", bool(cfg["streaming_mode"]))
    pp = cfg["preprocessor"]
    for k in ("sample_rate", "n_fft", "features"):
        kv(f"{ARCH}.preprocessor.{k}", int(pp[k]))
    for k in ("window_size", "window_stride", "dither"):
        kv(f"{ARCH}.preprocessor.{k}", float(pp.get(k, 0.0)))
    kv(f"{ARCH}.preprocessor.normalize", str(pp.get("normalize", "NA")))
    enc = cfg["encoder"]
    kv(f"{ARCH}.encoder.feat_in", int(enc["feat_in"]))
    kv(f"{ARCH}.encoder.n_layers", int(enc["n_layers"]))
    kv(f"{ARCH}.encoder.d_model", int(enc["d_model"]))
    kv(f"{ARCH}.encoder.n_heads", int(enc["n_heads"]))
    kv(f"{ARCH}.encoder.ffn_size", int(enc["d_model"] * float(enc["ff_expansion"])))
    kv(f"{ARCH}.encoder.subsampling", str(enc["subsampling"]))
    kv(f"{ARCH}.encoder.subsampling_factor", int(enc["subsampling_factor"]))
    kv(f"{ARCH}.encoder.self_attention_model", str(enc["self_attention_model"]))
    kv(f"{ARCH}.encoder.pre_block_norm", bool(enc["pre_block_norm"]))
    kv(f"{ARCH}.encoder.qkv_bias", bool(enc["qkv_bias"]))
    kv(f"{ARCH}.encoder.xscaling", bool(enc["xscaling"]))
    kv(f"{ARCH}.encoder.pos_emb_max_len", int(enc["pos_emb_max_len"]))
    sf = cfg["sortformer_modules"]
    kv(f"{ARCH}.head.tf_d_model", int(sf["tf_d_model"]))
    kv(f"{ARCH}.head.fc_d_model", int(sf["fc_d_model"]))
    kv(f"{ARCH}.head.subpixel_kernel", 3)
    for k in ("spkcache_sil_frames_per_spk", "pred_score_threshold", "scores_boost_latest",
              "sil_threshold", "strong_boost_rate", "weak_boost_rate", "min_pos_scores_rate"):
        if k in sf:
            kv(f"{ARCH}.scoring.{k}", sf[k])
    kv(f"{ARCH}.scoring.use_learnable_sil_emb", bool(sf.get("use_learnable_sil_emb", False)))
    kv(f"{ARCH}.head.chunk_len", int(sf.get("chunk_len", 0)))
    kv(f"{ARCH}.head.spkcache_len", int(sf.get("spkcache_len", 0)))


def _pick_dtype(key: str, weight_type: str):
    from gguf import GGMLQuantizationType

    if weight_type == "f32":
        return GGMLQuantizationType.F32

    # In F16 mode:
    # 1. preprocessor.fb and learnable_sil_emb are read as float32 in C++ loader
    if key in ("preprocessor.fb", "learnable_sil_emb"):
        return GGMLQuantizationType.F32
    # 2. LayerNorm weights and biases must be F32 (C++ LayerNorm assertion)
    if any(norm in key for norm in (".norm1.", ".norm2.", ".embed_norm.", ".final_norm.")):
        return GGMLQuantizationType.F32
    # 3. All biases stay F32
    if key.endswith(".bias"):
        return GGMLQuantizationType.F32
    # 4. Linear and Conv weights -> F16
    return GGMLQuantizationType.F16


def convert(nemo_path: Path, out_path: Path, weight_type: str = "f32") -> None:
    if weight_type not in ("f32", "f16"):
        raise ValueError("nemotron3_diar supports --outtype f32|f16")
    from gguf import GGMLQuantizationType, GGUFWriter

    cfg, sd = _load(nemo_path)
    w = GGUFWriter(str(out_path), ARCH)
    _metadata(w, cfg)
    mapped = 0
    dtype_counts = {}
    for name, tensor in sd.items():
        key = remap(name)
        if key is None:
            continue
        arr = tensor.detach().to(torch.float32).cpu().numpy()
        qtype = _pick_dtype(key, weight_type)
        if qtype == GGMLQuantizationType.F32:
            arr = arr.astype(np.float32)
        elif qtype == GGMLQuantizationType.F16:
            arr = arr.astype(np.float16)
        w.add_tensor(key, arr, raw_dtype=qtype)
        dtype_counts[qtype.name] = dtype_counts.get(qtype.name, 0) + 1
        mapped += 1
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"[convert] nemotron3_diar: wrote {mapped} tensors ({dtype_counts}) -> {out_path}")
