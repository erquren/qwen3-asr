"""Greedy transcription: full official ASR or converted encoder + HF decoder.

This script intentionally never calls the RKLLM runtime. A decoder-side host
comparison cannot establish quantized RKLLM recognition accuracy.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["official", "decoder"], required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, help="Directory of encoder outputs for decoder mode")
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    from transformers import AutoTokenizer, Qwen3ForCausalLM
    source = json.loads((args.inputs / "inputs.json").read_text())
    if args.mode == "official":
        from qwen_asr import Qwen3ASRModel
        wrapper = Qwen3ASRModel.from_pretrained(str(args.model), device_map="cpu", dtype=torch.float32,
                                               attn_implementation="eager", max_new_tokens=args.max_new_tokens)
        model, tokenizer = wrapper.model, wrapper.processor.tokenizer
    else:
        if args.embeddings is None:
            raise ValueError("Decoder mode requires encoder embeddings")
        model = Qwen3ForCausalLM.from_pretrained(args.model, local_files_only=True, torch_dtype=torch.float32,
                                                attn_implementation="eager", low_cpu_mem_usage=True).eval()
        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {"backend": args.label, "mode": args.mode, "decoder_dtype": "float32", "device": "cpu",
              "decoding": {"do_sample": False, "num_beams": 1, "max_new_tokens": args.max_new_tokens},
              "rkllm_inference": "not_run", "samples": []}
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False)+"\n")
    with torch.inference_mode():
        for sample in source["samples"]:
            item = np.load(args.inputs / sample["input_file"])
            started = time.monotonic()
            if args.mode == "official":
                kwargs = {key: torch.from_numpy(item[key]) for key in
                          ("input_ids", "attention_mask", "input_features", "feature_attention_mask")}
                outputs = model.generate(**kwargs, max_new_tokens=args.max_new_tokens, do_sample=False,
                                         num_beams=1)
                tokens = outputs.sequences[0, kwargs["input_ids"].shape[1]:]
            else:
                ids = torch.from_numpy(item["input_ids"])
                embeddings = model.model.embed_tokens(ids)
                audio_id = tokenizer.convert_tokens_to_ids(tokenizer.audio_token)
                audio_mask = ids == audio_id
                audio = np.load(args.embeddings / f"{sample['key']}.npy", allow_pickle=False)
                if audio.shape != (1, sample["audio_tokens"], model.config.hidden_size):
                    raise ValueError(f"Wrong audio embedding shape: {audio.shape}")
                if int(audio_mask.sum()) != audio.shape[1]:
                    raise ValueError("Audio placeholder count does not match encoder output")
                embeddings[audio_mask] = torch.from_numpy(audio).reshape(-1, model.config.hidden_size)
                outputs = model.generate(inputs_embeds=embeddings, attention_mask=torch.from_numpy(item["attention_mask"]),
                                         max_new_tokens=args.max_new_tokens, do_sample=False, num_beams=1,
                                         return_dict_in_generate=True)
                tokens = outputs.sequences[0]
            prediction = tokenizer.decode(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            truncated = len(tokens) >= args.max_new_tokens and int(tokens[-1]) not in (151643, 151645)
            row = {"id": sample["id"], "reference": sample["reference"], "hypothesis": prediction,
                   "token_ids": tokens.tolist(), "elapsed_seconds": time.monotonic()-started,
                   "generation_limit_reached": truncated, "status": "truncated" if truncated else "ok"}
            report["samples"].append(row)
            args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False)+"\n")
            print(json.dumps(row, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
