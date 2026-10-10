"""Download the pinned official safetensors checkpoint without remote code."""
import argparse
import hashlib
import json
from pathlib import Path

MODEL_ID = "Qwen/Qwen3-ASR-0.6B"
REVISION = "5eb144179a02acc5e5ba31e748d22b0cf3e303b0"
FILES = ["model.safetensors", "config.json", "generation_config.json", "preprocessor_config.json",
         "tokenizer_config.json", "chat_template.json", "merges.txt", "vocab.json", "README.md"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    from huggingface_hub import snapshot_download
    snapshot_download(MODEL_ID, revision=REVISION, local_dir=args.output, allow_patterns=FILES)
    report = {"model_id": MODEL_ID, "revision": REVISION, "license": "Apache-2.0", "files": {}}
    for name in FILES:
        path = args.output / name
        digest = hashlib.sha256()
        with path.open("rb") as file:
            for block in iter(lambda: file.read(8*1024*1024), b""):
                digest.update(block)
        report["files"][name] = {"bytes": path.stat().st_size, "sha256": digest.hexdigest()}
    (args.output / "source_manifest.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
