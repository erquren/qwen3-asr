#!/usr/bin/env python3
"""Extract official Qwen3-ASR safetensors as a standalone Qwen3 decoder.

No model code or pickle is loaded. Weight bytes are copied unchanged; the
separate token embedding table is converted to FP32 in bounded-size chunks.
The ordinary-RoPE adaptation is valid only for equal T/H/W position indices,
as used by the official ASR get_rope_index implementation. Run parity before
feeding the result to RKLLM. This script does not claim board compatibility.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import struct
from pathlib import Path

import numpy as np


def read_header(path: Path) -> tuple[dict, int]:
    with path.open("rb") as f:
        raw = f.read(8)
        if len(raw) != 8:
            raise ValueError(f"Truncated safetensors file: {path}")
        size = struct.unpack("<Q", raw)[0]
        if size > 100_000_000:
            raise ValueError("Safetensors header exceeds supported bound")
        header = json.loads(f.read(size))
    data_start = size + 8
    for name, info in header.items():
        if name == "__metadata__":
            continue
        begin, end = info["data_offsets"]
        if not (0 <= begin <= end <= path.stat().st_size - data_start):
            raise ValueError(f"Invalid tensor range: {name}")
    return header, data_start


def expected_shapes(c: dict) -> dict[str, list[int]]:
    h, d = c["hidden_size"], c["head_dim"]
    q, kv, mid = c["num_attention_heads"] * d, c["num_key_value_heads"] * d, c["intermediate_size"]
    shapes = {"model.embed_tokens.weight": [c["vocab_size"], h],
              "model.norm.weight": [h], "lm_head.weight": [c["vocab_size"], h]}
    for i in range(c["num_hidden_layers"]):
        prefix = f"model.layers.{i}."
        layer = {"self_attn.q_proj.weight": [q, h], "self_attn.k_proj.weight": [kv, h],
                 "self_attn.v_proj.weight": [kv, h], "self_attn.o_proj.weight": [h, q],
                 "self_attn.q_norm.weight": [d], "self_attn.k_norm.weight": [d],
                 "mlp.gate_proj.weight": [mid, h], "mlp.up_proj.weight": [mid, h],
                 "mlp.down_proj.weight": [h, mid], "input_layernorm.weight": [h],
                 "post_attention_layernorm.weight": [h]}
        if c.get("attention_bias", False):
            layer.update({f"self_attn.{p}_proj.bias": [n] for p, n in [("q", q), ("k", kv), ("v", kv), ("o", h)]})
        shapes.update({prefix + key: value for key, value in layer.items()})
    return shapes


def extract(source: Path, output: Path, revision: str | None) -> dict:
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Refusing to overwrite nonempty output directory: {output}")
    full_config = json.loads((source / "config.json").read_text())
    if full_config.get("model_type") != "qwen3_asr":
        raise ValueError("Expected official qwen3_asr checkpoint")
    text_config = full_config["thinker_config"]["text_config"]
    rope = text_config.get("rope_scaling") or {}
    if rope.get("rope_type", rope.get("type", "default")) != "default":
        raise ValueError("Only default RoPE with equal position planes is supported")
    files = sorted(source.glob("*.safetensors"))
    if not files:
        raise ValueError("No safetensors found")
    entries = {}
    for path in files:
        header, offset = read_header(path)
        for name, info in header.items():
            if name.startswith("thinker.model.") or name == "thinker.lm_head.weight":
                renamed = name.removeprefix("thinker.")
                if renamed in entries:
                    raise ValueError(f"Duplicate tensor: {renamed}")
                entries[renamed] = (path, offset, info)
    expected = expected_shapes(text_config)
    if entries.keys() != expected.keys():
        raise ValueError(f"Decoder keys differ. Missing={sorted(expected.keys()-entries.keys())}; extra={sorted(entries.keys()-expected.keys())}")
    for name, (_, _, info) in entries.items():
        if info["shape"] != expected[name] or info["dtype"] not in ("BF16", "F16", "F32"):
            raise ValueError(f"Unsupported shape or dtype: {name}: {info}")
        elem = {"BF16": 2, "F16": 2, "F32": 4}[info["dtype"]]
        if int(np.prod(info["shape"])) * elem != info["data_offsets"][1]-info["data_offsets"][0]:
            raise ValueError(f"Tensor byte size mismatch: {name}")

    output.mkdir(parents=True, exist_ok=True)
    result_header = {"__metadata__": {"format": "pt"}}
    offset = 0
    for name, (_, _, info) in sorted(entries.items()):
        size = info["data_offsets"][1] - info["data_offsets"][0]
        result_header[name] = {"dtype": info["dtype"], "shape": info["shape"], "data_offsets": [offset, offset+size]}
        offset += size
    serialized = json.dumps(result_header, separators=(",", ":")).encode()
    serialized += b" " * ((-len(serialized)) % 8)
    hashes = {}
    with (output / "model.safetensors").open("wb") as dst:
        dst.write(struct.pack("<Q", len(serialized)))
        dst.write(serialized)
        for name, (path, data_start, info) in sorted(entries.items()):
            begin, end = info["data_offsets"]
            remaining = end-begin
            digest = hashlib.sha256()
            with path.open("rb") as src:
                src.seek(data_start + begin)
                while remaining:
                    data = src.read(min(8 * 1024 * 1024, remaining))
                    if not data:
                        raise ValueError(f"Unexpected EOF: {name}")
                    digest.update(data)
                    dst.write(data)
                    remaining -= len(data)
            hashes[name] = digest.hexdigest()

    # Preserve both stored tensors independently, even when source config says tied.
    # This avoids silently replacing a learned LM head during from_pretrained().
    keys = ("vocab_size", "hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads",
            "num_key_value_heads", "head_dim", "hidden_act", "max_position_embeddings", "initializer_range",
            "rms_norm_eps", "rope_theta", "attention_bias", "attention_dropout", "use_cache")
    config = {key: text_config[key] for key in keys if key in text_config}
    config.update(model_type="qwen3", architectures=["Qwen3ForCausalLM"], tie_word_embeddings=False,
                  rope_scaling=None, use_sliding_window=False, sliding_window=None,
                  torch_dtype={"BF16": "bfloat16", "F16": "float16", "F32": "float32"}[entries["lm_head.weight"][2]["dtype"]])
    generation = json.loads((source / "generation_config.json").read_text()) if (source / "generation_config.json").exists() else {}
    for key in ("bos_token_id", "eos_token_id", "pad_token_id"):
        value = generation.get(key, text_config.get(key))
        if value is not None:
            config[key] = value
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    copied = []
    for name in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "special_tokens_map.json",
                 "added_tokens.json", "generation_config.json", "chat_template.jinja", "chat_template.json", "LICENSE"):
        if (source / name).is_file():
            shutil.copyfile(source / name, output / name)
            copied.append(name)
    if not (output / "chat_template.jinja").exists() and (output / "chat_template.json").exists():
        template = json.loads((output / "chat_template.json").read_text()).get("chat_template")
        if isinstance(template, str):
            (output / "chat_template.jinja").write_text(template)
    if not (output / "tokenizer_config.json").exists() or not ((output / "tokenizer.json").exists() or
         ((output / "vocab.json").exists() and (output / "merges.txt").exists())):
        raise ValueError("Source lacks a complete tokenizer; download tokenizer files before extraction")

    path, data_start, info = entries["model.embed_tokens.weight"]
    dtype = {"BF16": "<u2", "F16": "<f2", "F32": "<f4"}[info["dtype"]]
    raw = np.memmap(path, mode="r", dtype=dtype, offset=data_start+info["data_offsets"][0], shape=tuple(info["shape"]))
    embed = np.lib.format.open_memmap(output / "embed_tokens.npy", mode="w+", dtype=np.float32, shape=tuple(info["shape"]))
    for row in range(0, raw.shape[0], 1024):
        chunk = raw[row:row+1024]
        embed[row:row+1024] = (chunk.astype(np.uint32) << 16).view(np.float32) if info["dtype"] == "BF16" else chunk.astype(np.float32)
    embed.flush()
    del embed, raw
    report = {"source": str(source.resolve()), "source_revision": revision, "tensor_count": len(entries),
              "parameter_count": sum(int(np.prod(v[2]["shape"])) for v in entries.values()),
              "weight_bytes": offset, "tensor_sha256": hashes, "copied_tokenizer_files": copied,
              "embedding_shape": info["shape"], "embedding_dtype": "float32",
              "stored_head_equals_embedding": hashes["lm_head.weight"] == hashes["model.embed_tokens.weight"],
              "tie_word_embeddings": False, "rope_adaptation": "default RoPE, valid for identical T/H/W position IDs",
              "numerical_parity": "not_run", "rkllm_compile": "not_run", "board_inference": "not_run"}
    (output / "extraction_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-revision")
    args = parser.parse_args()
    report = extract(args.source, args.output, args.source_revision)
    print(json.dumps({k: v for k, v in report.items() if k != "tensor_sha256"}, indent=2))


if __name__ == "__main__":
    main()
