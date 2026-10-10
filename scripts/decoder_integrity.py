"""Stable SHA-256 identity for the exact decoder assets used by a test/build."""
from __future__ import annotations

import hashlib
from pathlib import Path


def embedding_schema_valid(array, config: dict) -> bool:
    """Require the complete, exact FP32 vocabulary table, including reserved rows."""
    return tuple(array.shape) == (config["vocab_size"], config["hidden_size"]) and str(array.dtype) == "float32"


def fingerprint_decoder(directory: Path) -> dict:
    required = ("config.json", "model.safetensors", "embed_tokens.npy", "tokenizer_config.json")
    for name in required:
        if not (directory / name).is_file():
            raise ValueError(f"Missing decoder artifact: {name}")
    if not (directory / "tokenizer.json").is_file() and not all((directory / p).is_file() for p in ("vocab.json", "merges.txt")):
        raise ValueError("Missing complete decoder tokenizer")
    optional = ("tokenizer.json", "vocab.json", "merges.txt", "added_tokens.json", "special_tokens_map.json",
                "generation_config.json", "chat_template.json", "chat_template.jinja")
    names = sorted(set(required) | {name for name in optional if (directory / name).is_file()})
    result = {}
    for name in names:
        path = directory / name
        digest = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        result[name] = {"bytes": path.stat().st_size, "sha256": digest.hexdigest()}
    return result
