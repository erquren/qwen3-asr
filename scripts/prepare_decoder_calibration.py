#!/usr/bin/env python3
"""Assemble ASR prompt+audio embeddings for RKLLM acoustic-aware calibration.

Supply reference encoder outputs from representative recordings as .npy or
.npz files. This saves reviewable .npy arrays AND the nested-list JSON required
by RKLLM 1.2.3; that SDK does not accept a .npy path as input_embed. A small set
of test clips is only smoke calibration and does not validate ASR accuracy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def load_audio_embedding(path: Path, key: str, hidden_size: int) -> np.ndarray:
    loaded = np.load(path, allow_pickle=False)
    if isinstance(loaded, np.lib.npyio.NpzFile):
        with loaded:
            value = np.asarray(loaded[key], dtype=np.float32)
    else:
        value = np.asarray(loaded, dtype=np.float32)
    if value.ndim == 2:
        value = value[None]
    if value.ndim != 3 or value.shape[0] != 1 or value.shape[2] != hidden_size or value.shape[1] < 1:
        raise ValueError(f"Expected [1, audio_tokens, {hidden_size}] embedding: {path}: {value.shape}")
    if not np.isfinite(value).all():
        raise ValueError(f"Non-finite encoder output: {path}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Original model directory, for special-token IDs")
    parser.add_argument("--decoder", type=Path, required=True, help="Extracted decoder directory with FP32 embedding table")
    parser.add_argument("--audio-embeddings", type=Path, action="append", required=True, help="Repeat per calibration recording")
    parser.add_argument("--key", default="reference", help="NPZ encoder-output key")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--language", help="Optional forced output language, e.g. English")
    parser.add_argument("--target", action="append", help="Optional verified raw ASR target per recording, including desired format/EOS")
    parser.add_argument("--max-context", type=int, default=4096)
    args = parser.parse_args()
    from transformers import AutoTokenizer
    targets = args.target or [""] * len(args.audio_embeddings)
    if len(targets) != len(args.audio_embeddings):
        raise ValueError("Supply one --target for every --audio-embeddings, or omit all targets")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError(f"Refusing to overwrite nonempty output: {args.output_dir}")
    config = json.loads((args.source / "config.json").read_text())["thinker_config"]
    hidden = config["text_config"]["hidden_size"]
    table = np.load(args.decoder / "embed_tokens.npy", mmap_mode="r", allow_pickle=False)
    if table.shape != (config["text_config"]["vocab_size"], hidden) or table.dtype != np.float32:
        raise ValueError("Unexpected extracted embedding table shape or dtype")
    tokenizer = AutoTokenizer.from_pretrained(args.decoder, local_files_only=True, trust_remote_code=False)
    messages = [{"role": "system", "content": ""}, {"role": "user", "content": [{"type": "audio", "audio": ""}]}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if args.language:
        prompt += f"language {args.language}<asr_text>"
    audio_id = config["audio_token_id"]
    audio_token = tokenizer.convert_ids_to_tokens(audio_id)
    if prompt.count(audio_token) != 1:
        raise ValueError("Expected exactly one audio placeholder in the official ASR prompt")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset, manifest = [], []
    for index, (path, target) in enumerate(zip(args.audio_embeddings, targets)):
        audio = load_audio_embedding(path, args.key, hidden)
        expanded = prompt.replace(audio_token, audio_token * audio.shape[1], 1)
        ids = np.array(tokenizer(expanded, add_special_tokens=False).input_ids, dtype=np.int64)
        mask = ids == audio_id
        if int(mask.sum()) != audio.shape[1]:
            raise ValueError("Tokenizer did not preserve one token for each encoder frame")
        if ids.size + len(tokenizer(target, add_special_tokens=False).input_ids) > args.max_context:
            raise ValueError("Calibration input/target exceeds selected context")
        inputs = np.asarray(table[ids], dtype=np.float32)[None].copy(order="C")
        inputs[0, mask] = audio[0]
        output = args.output_dir / f"inputs_embeds_{index:03d}.npy"
        np.save(output, inputs, allow_pickle=False)
        dataset.append({"input_embed": inputs.tolist(), "target": target})
        manifest.append({"source_encoder_output": str(path.resolve()), "source_key": args.key,
                         "output": output.name, "shape": list(inputs.shape), "dtype": str(inputs.dtype),
                         "audio_tokens": int(audio.shape[1]), "prompt_token_ids": ids.tolist(),
                         "has_verified_target": bool(target), "sha256": hashlib.sha256(output.read_bytes()).hexdigest()})
    (args.output_dir / "data_quant.json").write_text(json.dumps(dataset, separators=(",", ":")) + "\n")
    report = {"sample_count": len(dataset), "format": "RKLLM 1.2.3 input_embed nested [batch,sequence,hidden] numeric lists",
              "calibration_scope": "caller-supplied encoder outputs; sample coverage must be evaluated separately",
              "language": args.language, "raw_prompt": prompt, "max_context": args.max_context, "samples": manifest}
    (args.output_dir / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
