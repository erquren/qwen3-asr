"""Static, batch-one Qwen3-ASR audio encoder using standard ONNX operators.

The module reuses the official Apache-2.0 qwen-asr layers and weights without
modifying them. Fixed chunk geometry avoids dynamic Unfold/Slice shape logic in
RKNN. The caller supplies the additive attention mask explicitly.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def output_length(mel_frames: int) -> int:
    if mel_frames <= 0:
        raise ValueError("mel_frames must be positive")
    complete, tail = divmod(mel_frames, 100)
    return complete * 13 + (tail + 7) // 8


def attention_mask(frames: int, valid_frames: int | None = None, windowed: bool = True) -> np.ndarray:
    valid_frames = frames if valid_frames is None else valid_frames
    if not 100 <= valid_frames <= frames or frames % 100:
        raise ValueError("Use full 100-frame buckets and at least 100 valid frames")
    length, valid = output_length(frames), output_length(valid_frames)
    allowed = np.ones((length, length), dtype=bool)
    if windowed:
        ids = np.arange(length) // 104
        allowed &= ids[:, None] == ids[None, :]
    allowed[:, valid:] = False
    # Unused output rows must remain finite, even when a whole window is padding.
    for row in range(valid, length):
        if not allowed[row].any():
            allowed[row, row] = True
    return np.where(allowed, 0.0, -10000.0).astype(np.float32)[None, None]


def load_audio_tower(model_dir: str | Path):
    from safetensors import safe_open
    from qwen_asr.core.transformers_backend.configuration_qwen3_asr import Qwen3ASRAudioEncoderConfig
    from qwen_asr.core.transformers_backend.modeling_qwen3_asr import Qwen3ASRAudioEncoder

    root = Path(model_dir)
    raw = json.loads((root / "config.json").read_text())["thinker_config"]["audio_config"]
    expected = {"num_mel_bins": 128, "n_window": 50, "n_window_infer": 800,
                "d_model": 896, "encoder_attention_heads": 14, "output_dim": 1024}
    for key, value in expected.items():
        if raw.get(key) != value:
            raise ValueError(f"Unsupported 0.6B encoder configuration: {key}={raw.get(key)!r}, expected {value}")
    config = Qwen3ASRAudioEncoderConfig(**raw)
    config._attn_implementation = "eager"
    tower = Qwen3ASRAudioEncoder(config).float().eval()
    prefix = "thinker.audio_tower."
    with safe_open(root / "model.safetensors", framework="pt", device="cpu") as source, torch.no_grad():
        target = tower.state_dict()
        actual = {key[len(prefix):] for key in source.keys() if key.startswith(prefix)}
        if actual != set(target):
            raise ValueError(f"Encoder keys differ: missing={set(target)-actual}, extra={actual-set(target)}")
        for name, tensor in target.items():
            tensor.copy_(source.get_tensor(prefix + name))
    return tower


class StaticEncoder(nn.Module):
    def __init__(self, tower, frames: int):
        super().__init__()
        if frames not in (200, 400, 1500):
            raise ValueError("Supported mel-frame buckets are 200, 400 and 1500")
        self.tower = tower
        self.frames = frames
        self.chunks = frames // 100
        self.width = tower.config.d_model
        self.heads = tower.config.encoder_attention_heads
        if self.width % self.heads:
            raise ValueError("Encoder width must be divisible by attention heads")
        self.head_dim = self.width // self.heads
        self.tokens = self.chunks * 13

    def forward(self, input_features, attention_mask):
        a = self.tower
        # [1,128,N*100] -> [N,1,128,100], preserving one-second boundaries.
        hidden = input_features.reshape(1, 128, self.chunks, 100).permute(0, 2, 1, 3)
        hidden = hidden.reshape(self.chunks, 1, 128, 100)
        for conv in (a.conv2d1, a.conv2d2, a.conv2d3):
            hidden = F.gelu(conv(hidden))
        hidden = hidden.permute(0, 3, 1, 2).reshape(self.chunks, 13, -1)
        hidden = a.conv_out(hidden)
        hidden = hidden + a.positional_embedding.positional_embedding[:13].unsqueeze(0)
        hidden = hidden.reshape(1, self.tokens, self.width)
        for layer in a.layers:
            norm = layer.self_attn_layer_norm(hidden)
            attn = layer.self_attn
            q, k, v = [projection(norm).reshape(1, self.tokens, self.heads, self.head_dim).transpose(1, 2)
                       for projection in (attn.q_proj, attn.k_proj, attn.v_proj)]
            weights = torch.softmax(q @ k.transpose(-1, -2) * attn.scaling + attention_mask, dim=-1)
            values = (weights @ v).transpose(1, 2).reshape(1, self.tokens, self.width)
            hidden = hidden + attn.out_proj(values)
            hidden = hidden + layer.fc2(layer.activation_fn(layer.fc1(layer.final_layer_norm(hidden))))
        return a.proj2(a.act(a.proj1(a.ln_post(hidden))))


def metrics(reference: np.ndarray, actual: np.ndarray) -> dict:
    if reference.shape != actual.shape:
        raise ValueError(f"Output shape mismatch: {reference.shape} != {actual.shape}")
    ref, got = reference.astype(np.float64).ravel(), actual.astype(np.float64).ravel()
    if ref.shape != got.shape or not np.isfinite(got).all():
        raise ValueError("Output shape mismatch or non-finite output")
    delta = got - ref
    denominator = max(float(np.linalg.norm(ref)), 1e-30)
    return {"cosine": float(np.dot(ref, got) / max(np.linalg.norm(ref) * np.linalg.norm(got), 1e-30)),
            "max_abs": float(np.max(np.abs(delta))), "relative_l2": float(np.linalg.norm(delta) / denominator)}
