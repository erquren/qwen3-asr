"""Prepare identical official processor inputs for a labeled ASR smoke corpus."""
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import qwen_asr  # Registers the official processor, without remote code.
    from qwen_asr.inference.utils import normalize_audio_input
    from transformers import AutoProcessor
    from encoder import attention_mask, output_length
    processor = AutoProcessor.from_pretrained(args.source, local_files_only=True)
    manifest = json.loads(args.manifest.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    samples = []
    for index, source in enumerate(manifest["samples"]):
        audio = Path(source["audio_path"])
        if not audio.is_absolute():
            audio = args.manifest.parent / audio
        waveform = normalize_audio_input(str(audio))
        language = source.get("language", "Chinese")
        prompt = processor.apply_chat_template([
            {"role": "system", "content": ""},
            {"role": "user", "content": [{"type": "audio", "audio": ""}]},
        ], tokenize=False, add_generation_prompt=True) + f"language {language}<asr_text>"
        inputs = processor(text=[prompt], audio=[waveform], return_tensors="np", padding=True)
        values = {key: np.asarray(value) for key, value in inputs.items()}
        valid = int(values["feature_attention_mask"].sum())
        if not 100 <= valid <= 800:
            raise ValueError(f"This smoke corpus requires 100..800 full mel frames: {source['id']}: {valid}")
        bucket = next(seconds for seconds in (2, 4, 15) if seconds*100 >= valid)
        padded = np.zeros((1, 128, bucket*100), dtype=np.float32)
        padded[:, :, :valid] = values["input_features"][:, :, :valid]
        values.update(padded_features=padded, encoder_attention_mask=attention_mask(bucket*100, valid),
                      valid_frames=np.array(valid), valid_tokens=np.array(output_length(valid)))
        name = f"sample_{index:03d}"
        np.savez(args.output / f"{name}.npz", **values)
        samples.append({**source, "input_file": f"{name}.npz", "key": name,
                        "valid_frames": valid, "audio_tokens": output_length(valid), "bucket_seconds": bucket})
    report = {"dataset": manifest.get("dataset"), "manifest": str(args.manifest.resolve()),
              "sample_count": len(samples), "language_forced": True, "samples": samples,
              "attention_note": "All full utterances have <=800 mel frames; original eager and 104-token windows agree."}
    (args.output / "inputs.json").write_text(json.dumps(report, indent=2, ensure_ascii=False)+"\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
