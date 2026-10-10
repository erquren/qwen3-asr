"""Export a fixed Qwen3-ASR encoder bucket from official safetensors."""
import argparse
import json
from pathlib import Path

import torch
from encoder import StaticEncoder, attention_mask, load_audio_tower


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seconds", type=int, choices=[2, 4, 15], required=True)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    frames = args.seconds * 100
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    tower = load_audio_tower(args.model)
    model = StaticEncoder(tower, frames).eval()
    features = torch.zeros(1, 128, frames)
    mask = torch.from_numpy(attention_mask(frames))
    with torch.inference_mode():
        torch.onnx.export(model, (features, mask), str(target), opset_version=18,
                          input_names=["input_features", "attention_mask"], output_names=["audio_embeds"],
                          dynamic_axes=None, do_constant_folding=True, dynamo=False)
    print(json.dumps({"onnx": str(target), "frames": frames, "tokens": model.tokens,
                      "output_dim": tower.config.output_dim, "opset": 18, "batch": 1}))


if __name__ == "__main__":
    main()
