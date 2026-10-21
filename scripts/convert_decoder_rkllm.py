#!/usr/bin/env python3
"""Compile a parity-checked standalone decoder for RK3588 using RKLLM 1.2.3.

Host compilation only. The existence of an RKLLM file does not validate NPU
inference, acoustic-token embedding input, transcription accuracy, or latency.
Use a separate vendor SDK environment; its Transformers pin differs from ASR.
"""
from __future__ import annotations

import argparse
import ctypes
import gc
import hashlib
import importlib.metadata
import json
import os
import platform
import time
import traceback
from pathlib import Path
from decoder_integrity import fingerprint_decoder


def rss_snapshot() -> dict:
    result = {}
    path = Path("/proc/self/status")
    if path.exists():
        for line in path.read_text().splitlines():
            if line.startswith(("VmRSS:", "VmHWM:")):
                name, value, unit = line.split()
                result[name.rstrip(":") + "_kib"] = int(value)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decoder", type=Path, required=True)
    parser.add_argument("--parity-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--quant", choices=("fp16", "w8a8", "w8a8_g128", "w4a16", "w4a16_g128"), default="fp16")
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--max-context", type=int, default=4096)
    parser.add_argument("--load-dtype", choices=("float32", "float16", "bfloat16"), default="float32")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--reclaim-memory-before-export", action="store_true",
                        help="Collect Python garbage and request glibc to release unused allocator pages before export")
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError(f"Refusing to overwrite output: {args.output}")
    parity = json.loads(args.parity_report.read_text())
    if parity.get("passed") is not True:
        raise ValueError("An explicitly passing decoder parity report is required")
    config = json.loads((args.decoder / "config.json").read_text())
    if config.get("model_type") != "qwen3" or config.get("architectures") != ["Qwen3ForCausalLM"]:
        raise ValueError("Expected independently extracted Qwen3ForCausalLM decoder")
    if not 32 <= args.max_context <= min(16384, config["max_position_embeddings"]) or args.max_context % 32:
        raise ValueError("Max context must be a multiple of 32 in [32, 16384]")
    if args.quant != "fp16" and (args.dataset is None or not args.dataset.is_file()):
        raise ValueError("Quantization requires an explicit calibration dataset")
    if not isinstance(parity.get("decoder_artifacts"), dict) or not parity["decoder_artifacts"]:
        raise ValueError("Parity report is not bound to decoder artifact fingerprints; rerun the parity checker")
    fingerprints = fingerprint_decoder(args.decoder)
    if parity["decoder_artifacts"] != fingerprints:
        raise ValueError("Decoder artifacts changed since numerical parity; rerun the parity checker")
    version = importlib.metadata.version("rkllm-toolkit")
    if version != "1.3.1":
        raise RuntimeError(f"Expected RKLLM Toolkit 1.3.1, found {version}")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ.setdefault("OMP_NUM_THREADS", str(args.threads))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_HOME", str(args.output.parent.resolve() / ".hf-cache"))
    os.environ.setdefault("HF_DATASETS_CACHE", str(Path(os.environ["HF_HOME"]) / "datasets"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    report = {"passed": False, "target_platform": "rk3588", "num_npu_core": 3,
              "host": platform.platform(), "python": platform.python_version(),
              "load_dtype": args.load_dtype, "device": "cpu", "quant": args.quant,
              "max_context": args.max_context, "decoder": str(args.decoder.resolve()),
              "parity_report": str(args.parity_report.resolve()), "steps": {},
              "decoder_artifacts": fingerprints,
              "board_inference": "not_run", "asr_accuracy": "not_run"}
    def checkpoint(stage: str) -> None:
        report["stage"] = stage
        report.setdefault("stage_rss", {})[stage] = rss_snapshot()
        report["elapsed_seconds"] = time.monotonic()-start
        args.report.write_text(json.dumps(report, indent=2, default=str) + "\n")

    llm = None
    try:
        import torch
        from rkllm.api import RKLLM
        torch.set_num_threads(args.threads)
        report["versions"] = {p: importlib.metadata.version(p) for p in ("rkllm-toolkit", "torch", "transformers", "numpy")}
        llm = RKLLM()
        checkpoint("loading")
        ret = llm.load_huggingface(model=str(args.decoder.resolve()), device="cpu", dtype=args.load_dtype)
        report["steps"]["load_huggingface"] = ret
        if ret != 0:
            raise RuntimeError(f"RKLLM load_huggingface returned {ret}")
        loaded_model = getattr(getattr(llm, "base", None), "model", None)
        report["loaded_model_dtype"] = str(getattr(loaded_model, "dtype", "unavailable"))
        checkpoint("loaded")
        kwargs = {"do_quantization": args.quant != "fp16", "optimization_level": 1,
                  "target_platform": "rk3588", "num_npu_core": 3, "max_context": args.max_context}
        if args.quant != "fp16":
            kwargs.update(quantized_dtype=args.quant, quantized_algorithm="normal" if args.quant.startswith("w8a8") else "grq",
                          dataset=str(args.dataset.resolve()))
        report["build_kwargs"] = kwargs
        checkpoint("building")
        ret = llm.build(**kwargs)
        report["steps"]["build"] = ret
        if ret != 0:
            raise RuntimeError(f"RKLLM build returned {ret}")
        checkpoint("built")
        if args.reclaim_memory_before_export:
            report["allocator_cleanup"] = {"gc_collected": gc.collect()}
            try:
                trim = ctypes.CDLL(None).malloc_trim
                trim.argtypes = [ctypes.c_size_t]
                trim.restype = ctypes.c_int
                report["allocator_cleanup"]["malloc_trim_result"] = trim(0)
            except (AttributeError, OSError) as exc:
                report["allocator_cleanup"]["malloc_trim_unavailable"] = str(exc)
        checkpoint("exporting")
        ret = llm.export_rkllm(str(args.output.resolve()))
        report["steps"]["export_rkllm"] = ret
        if ret != 0 or not args.output.is_file() or not args.output.stat().st_size:
            raise RuntimeError(f"RKLLM export failed or missing artifact: return={ret}")
        digest = hashlib.sha256()
        with args.output.open("rb") as f:
            for chunk in iter(lambda: f.read(8*1024*1024), b""):
                digest.update(chunk)
        report.update(passed=True, output=str(args.output.resolve()), output_bytes=args.output.stat().st_size, sha256=digest.hexdigest())
        checkpoint("complete")
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
    finally:
        report["elapsed_seconds"] = time.monotonic()-start
        args.report.write_text(json.dumps(report, indent=2, default=str) + "\n")
        print(json.dumps(report, indent=2, default=str), flush=True)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
