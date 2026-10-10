#!/usr/bin/env python3
"""Compare official ASR decoder with the extracted ordinary Qwen3 on CPU.

Loads official installed qwen-asr and transformers package classes, never
remote model code. Sequential loading keeps peak memory below two FP32 models.
Tests text IDs, input embeddings, padding, and cached autoregressive decoding.
This is pre-quantization numerical validation, not an RK3588 inference test.
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from decoder_integrity import embedding_schema_valid, fingerprint_decoder


def tensor_map(path: Path, original: bool) -> dict:
    weights = {}
    for file in sorted(path.glob("*.safetensors")):
        with safe_open(file, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if original:
                    if key.startswith("thinker.model.") or key == "thinker.lm_head.weight":
                        weights[key.removeprefix("thinker.")] = handle.get_tensor(key)
                elif key.startswith("model.") or key == "lm_head.weight":
                    weights[key] = handle.get_tensor(key)
    return weights


def load_decoder(path: Path, original: bool):
    from transformers import Qwen3Config, Qwen3ForCausalLM
    from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding

    if original:
        from qwen_asr.core.transformers_backend.configuration_qwen3_asr import Qwen3ASRTextConfig
        from qwen_asr.core.transformers_backend.modeling_qwen3_asr import (
            Qwen3ASRThinkerTextModel, Qwen3ASRThinkerTextRotaryEmbedding,
        )
        raw = json.loads((path / "config.json").read_text())["thinker_config"]["text_config"]
        config = Qwen3ASRTextConfig(**raw)
        config._attn_implementation = "eager"
        with torch.device("meta"):
            core = Qwen3ASRThinkerTextModel(config)
            head = torch.nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        core.rotary_emb = Qwen3ASRThinkerTextRotaryEmbedding(config, device="cpu")
        state = tensor_map(path, True)
        core.load_state_dict({k.removeprefix("model."): v for k, v in state.items() if k.startswith("model.")}, strict=True, assign=True)
        head.load_state_dict({"weight": state["lm_head.weight"]}, strict=True, assign=True)
    else:
        config = Qwen3Config.from_pretrained(path, local_files_only=True)
        config._attn_implementation = "eager"
        with torch.device("meta"):
            model = Qwen3ForCausalLM(config)
        model.model.rotary_emb = Qwen3RotaryEmbedding(config, device="cpu")
        state = tensor_map(path, False)
        model.load_state_dict(state, strict=True, assign=True)
        core, head = model.model, model.lm_head
    del state
    core = core.to(dtype=torch.float32).eval()
    head = head.to(dtype=torch.float32).eval()
    return core, head, config


def make_cases(vocab: int, hidden: int) -> list[dict]:
    generator = torch.Generator().manual_seed(20261010)
    cases = []
    for length in (1, 17, 64):
        cases.append({"name": f"token_ids_{length}", "input_ids": torch.randint(0, min(vocab, 150000), (1, length), generator=generator)})
    cases.append({"name": "mixed_inputs_embeds_23", "input_ids": torch.randint(0, min(vocab, 150000), (1, 23), generator=generator),
                  "replacement": torch.randn(1, 9, hidden, generator=generator) * 0.1})
    cases.append({"name": "left_padding_batch2", "input_ids": torch.randint(0, min(vocab, 150000), (2, 11), generator=generator),
                  "attention_mask": torch.tensor([[0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1], [1] * 11])})
    return cases


@torch.inference_mode()
def evaluate(path: Path, original: bool) -> dict[str, torch.Tensor]:
    core, head, config = load_decoder(path, original)
    results = {}
    for case in make_cases(config.vocab_size, config.hidden_size):
        name = case["name"]
        ids = case["input_ids"]
        mask = case.get("attention_mask", torch.ones_like(ids))
        positions = mask.cumsum(-1) - 1
        positions.masked_fill_(mask == 0, 1)
        kwargs = {"attention_mask": mask, "position_ids": positions, "use_cache": False}
        if "replacement" in case:
            embeddings = core.embed_tokens(ids).clone()
            embeddings[:, 5:14] = case["replacement"]
            kwargs["inputs_embeds"] = embeddings
        else:
            kwargs["input_ids"] = ids
        hidden = core(**kwargs).last_hidden_state
        results[name + "/hidden"] = hidden.clone()
        results[name + "/logits_last2"] = head(hidden[:, -2:]).clone()
        print(f"{'official' if original else 'qwen3'}: {name}", flush=True)

    prompt = torch.tensor([[151644, 8948, 198, 151645, 198, 151644, 872, 198]])
    prefix = core(input_ids=prompt, attention_mask=torch.ones_like(prompt), use_cache=True)
    cache = prefix.past_key_values
    results["cached/prefill_logits"] = head(prefix.last_hidden_state[:, -1:]).clone()
    next_id = results["cached/prefill_logits"].argmax(-1)
    for i in range(3):
        position = prompt.shape[1] + i
        step = core(input_ids=next_id, attention_mask=torch.ones(1, position+1, dtype=torch.long),
                    position_ids=torch.tensor([[position]]), past_key_values=cache,
                    cache_position=torch.tensor([position]), use_cache=True)
        cache = step.past_key_values
        results[f"cached/step_{i}_logits"] = head(step.last_hidden_state).clone()
        next_id = results[f"cached/step_{i}_logits"].argmax(-1)
    positions = torch.tensor([[0, 1, 123, 1023, 4095, 65535]])
    cos, sin = core.rotary_emb(torch.zeros(1, 6, config.hidden_size), positions)
    results["rotary/cos"] = cos.clone()
    results["rotary/sin"] = sin.clone()
    del core, head, cache, prefix, step
    gc.collect()
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--decoder", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--rtol", type=float, default=1e-4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    start = time.monotonic()
    before = fingerprint_decoder(args.decoder)
    original = evaluate(args.source, True)
    converted = evaluate(args.decoder, False)
    report = {"passed": True, "dtype": "float32", "device": "cpu", "attention": "eager",
              "source": str(args.source.resolve()), "decoder": str(args.decoder.resolve()),
              "decoder_artifacts": before,
              "atol": args.atol, "rtol": args.rtol, "checks": {}, "board_inference": "not_run"}
    for name, ref in original.items():
        actual = converted[name]
        diff = (actual-ref).abs()
        passed = bool(torch.allclose(actual, ref, atol=args.atol, rtol=args.rtol))
        check = {"shape": list(ref.shape), "max_abs_error": float(diff.max()), "mean_abs_error": float(diff.mean()), "passed": passed}
        if "logits" in name:
            check["argmax_equal"] = bool(torch.equal(ref.argmax(-1), actual.argmax(-1)))
            passed = passed and check["argmax_equal"]
            check["passed"] = passed
        report["checks"][name] = check
        report["passed"] &= passed
    # Verify the FP32 embedding artifact over every row without an extra model.
    table = np.load(args.decoder / "embed_tokens.npy", mmap_mode="r")
    config = json.loads((args.decoder / "config.json").read_text())
    expected_shape = (config["vocab_size"], config["hidden_size"])
    table_valid = embedding_schema_valid(table, config)
    with safe_open(args.decoder / "model.safetensors", framework="pt", device="cpu") as f:
        tensor = f.get_tensor("model.embed_tokens.weight")
        equal = table_valid and tuple(tensor.shape) == expected_shape and all(
            np.array_equal(table[i:i+1024], tensor[i:i+1024].float().numpy()) for i in range(0, expected_shape[0], 1024))
    report["embedding_shape"] = list(table.shape)
    report["embedding_dtype"] = str(table.dtype)
    report["embedding_shape_dtype_valid"] = bool(table_valid)
    report["embedding_fp32_exact"] = bool(equal)
    report["passed"] &= bool(equal)
    report["decoder_artifacts_unchanged_during_test"] = before == fingerprint_decoder(args.decoder)
    report["passed"] &= report["decoder_artifacts_unchanged_during_test"]
    report["elapsed_seconds"] = time.monotonic()-start
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
