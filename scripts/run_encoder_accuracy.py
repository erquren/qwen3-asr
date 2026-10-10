"""Save converted encoder outputs for labeled ASR recognition evaluation."""
import argparse
import gc
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["onnx_fp32", "onnx_fp16", "rknn_simulator"], required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads((args.inputs / "inputs.json").read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    for seconds in sorted({row["bucket_seconds"] for row in source["samples"]}):
        rknn = None
        session = None
        try:
            if args.backend == "rknn_simulator":
                from rknn.api import RKNN
                rknn = RKNN(verbose=False)
                def checked(name, result):
                    if result != 0:
                        raise RuntimeError(f"RKNN {name}: {result}")
                checked("config", rknn.config(target_platform="rk3588", optimization_level=3))
                checked("load_onnx", rknn.load_onnx(model=str(args.models / f"encoder-{seconds}s.fp16.onnx")))
                checked("build", rknn.build(do_quantization=False))
                checked("init_simulator", rknn.init_runtime())
            else:
                import onnxruntime as ort
                options = ort.SessionOptions()
                options.intra_op_num_threads, options.inter_op_num_threads = 2, 1
                precision = args.backend.removeprefix("onnx_")
                session = ort.InferenceSession(str(args.models / f"encoder-{seconds}s.{precision}.onnx"),
                                               sess_options=options, providers=["CPUExecutionProvider"])
            for sample in source["samples"]:
                if sample["bucket_seconds"] != seconds:
                    continue
                item = np.load(args.inputs / sample["input_file"])
                features, mask = item["padded_features"], item["encoder_attention_mask"]
                output = (rknn.inference(inputs=[features, mask], data_format=["nchw", "nchw"])[0]
                          if rknn is not None else session.run(None, {"input_features": features, "attention_mask": mask})[0])
                if output.shape != (1, seconds*13, 1024) or not np.isfinite(output).all():
                    raise ValueError(f"Invalid full encoder output: {sample['id']}")
                output = output[:, :sample["audio_tokens"]].astype(np.float32)
                np.save(args.output / f"{sample['key']}.npy", output, allow_pickle=False)
                records.append({"id": sample["id"], "shape": list(output.shape), "bucket_seconds": seconds})
        finally:
            if rknn is not None:
                rknn.release()
            del session, rknn
            gc.collect()
    (args.output / "manifest.json").write_text(json.dumps({"backend": args.backend, "board_inference": "not_run", "samples": records}, indent=2)+"\n")


if __name__ == "__main__":
    main()
