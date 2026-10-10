"""Create real/synthetic encoder goldens from untouched official layers.

The installed qwen-asr 0.0.6 eager backend ignores cu_seqlens. Windowed goldens
therefore run each official layer independently on its official cu_seqlens
segments, rather than using our exported graph or its dense attention mask.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch import nn

from encoder import StaticEncoder, attention_mask, load_audio_tower, metrics, output_length


class IndependentlyWindowedLayer(nn.Module):
    def __init__(self, original):
        super().__init__()
        self.original = original

    def forward(self, hidden_states, cu_seqlens, **kwargs):
        bounds = cu_seqlens.tolist()
        outputs = []
        for start, stop in zip(bounds, bounds[1:]):
            local = torch.tensor([0, stop-start], dtype=cu_seqlens.dtype, device=cu_seqlens.device)
            outputs.append(self.original(hidden_states[start:stop], local, **kwargs)[0])
        return (torch.cat(outputs, dim=0),)


def official_reference(tower, features, valid, windowed):
    original_layers = tower.layers
    try:
        if windowed:
            tower.layers = nn.ModuleList([IndependentlyWindowedLayer(layer) for layer in original_layers])
        return tower(torch.from_numpy(features[0, :, :valid]),
                     feature_lens=torch.tensor([valid])).last_hidden_state.numpy()[None]
    finally:
        tower.layers = original_layers


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--audio", required=True)
    parser.add_argument("--seconds", type=int, choices=[2, 4, 15], required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    from transformers import WhisperFeatureExtractor
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    frames = args.seconds * 100
    extractor = WhisperFeatureExtractor.from_pretrained(args.model, local_files_only=True)
    waveform, rate = sf.read(args.audio, dtype="float32")
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)
    if rate != 16000:
        import librosa
        waveform = librosa.resample(waveform, orig_sr=rate, target_sr=16000)
    samples = args.seconds * 16000
    waveform = np.pad(waveform[:samples], (0, max(0, samples-len(waveform))))
    real = extractor(waveform, sampling_rate=16000, padding=False, return_tensors="np")["input_features"]
    if real.shape != (1, 128, frames):
        raise ValueError(f"Unexpected mel shape {real.shape}")
    np.save(root / "mel_filters.npy", extractor.mel_filters.astype(np.float32))
    rng = np.random.default_rng(20261010)
    synthetic = rng.normal(0, 0.25, (1, 128, frames)).astype(np.float32)
    partial = {2: 137, 4: 317, 15: 1437}[args.seconds]
    padded = real.copy()
    padded[:, :, partial:] = 0
    cases = [("speech", real, frames, True), ("synthetic", synthetic, frames, True),
             ("padded_speech", padded, partial, True)]
    if args.seconds == 15:
        short = real.copy()
        short[:, :, 100:] = 0
        cases.extend([("one_second_in_15s", short, 100, True), ("speech_global", real, frames, False)])
    tower = load_audio_tower(args.model)
    export_model = StaticEncoder(tower, frames).eval()
    report = {"seconds": args.seconds, "reference": "official layers split at official cu_seqlens for windowed attention",
              "cases": [], "thresholds": {"relative_l2": 1e-5, "max_abs": 0.001}}
    outputs = {}
    with torch.inference_mode():
        for name, features, valid, windowed in cases:
            mask = attention_mask(frames, valid, windowed)
            reference = official_reference(tower, features, valid, windowed)
            actual = export_model(torch.from_numpy(features), torch.from_numpy(mask)).numpy()
            if not np.isfinite(actual).all():
                raise ValueError(f"Non-finite full PyTorch output: {name}")
            length = output_length(valid)
            result = metrics(reference, actual[:, :length])
            passed = result["relative_l2"] < 1e-5 and result["max_abs"] < 0.001
            np.savez(root / f"{name}.npz", input_features=features, attention_mask=mask,
                     reference=reference, valid_tokens=np.array(length, dtype=np.int64))
            outputs[name] = reference
            report["cases"].append({"name": name, "valid_frames": valid, "valid_tokens": length,
                                     "windowed": windowed, "pytorch_parity": result, "passed": passed})
            print(json.dumps(report["cases"][-1]), flush=True)
        if "speech_global" in outputs:
            report["global_vs_windowed"] = metrics(outputs["speech"], outputs["speech_global"])
    report["passed"] = all(case["passed"] for case in report["cases"])
    (root / "pytorch_parity.json").write_text(json.dumps(report, indent=2) + "\n")
    if not report["passed"]:
        raise SystemExit("Official-to-export-wrapper parity failed")


if __name__ == "__main__":
    main()
