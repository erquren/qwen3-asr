"""Compile a fixed two-input encoder ONNX for RK3588 with RKNN Toolkit2."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import time
import traceback

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seconds", type=int, choices=[2, 4, 15], required=True)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--optimization-level", type=int, choices=[0, 1, 2, 3], default=3)
    parser.add_argument("--cases", help="Optional directory of independent NPZ goldens for host simulator")
    parser.add_argument("--report", help="JSON stage/simulator report")
    args = parser.parse_args()
    from rknn.api import RKNN
    rknn = RKNN(verbose=args.verbose)
    started = time.monotonic()
    report = {"passed": False, "target": "rk3588", "quantized": False, "output": args.output,
              "seconds": args.seconds, "toolkit": importlib.metadata.version("rknn-toolkit2"), "steps": {}, "board_inference": "not_run",
              "simulator": "not_run", "optimization_level": args.optimization_level,
              "cases": [], "limits": {"relative_l2": 0.03, "max_abs": 0.5}}
    def checked(name, result):
        report["steps"][name] = result
        if result != 0:
            raise RuntimeError(f"{name} returned {result}")
    try:
        checked("config", rknn.config(target_platform="rk3588", optimization_level=args.optimization_level))
        frames, tokens = args.seconds * 100, args.seconds * 13
        checked("load_onnx", rknn.load_onnx(model=args.onnx, inputs=["input_features", "attention_mask"],
                              input_size_list=[[1, 128, frames], [1, 1, tokens, tokens]]))
        checked("build", rknn.build(do_quantization=False))
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        checked("export_rknn", rknn.export_rknn(args.output))
        report["output_bytes"] = Path(args.output).stat().st_size
        if args.cases:
            checked("init_runtime_simulator", rknn.init_runtime())
            from check_encoder_onnx import metrics
            for path in sorted(Path(args.cases).glob("*.npz")):
                case = np.load(path)
                outputs = rknn.inference(inputs=[case["input_features"], case["attention_mask"]],
                                         data_format=["nchw", "nchw"])
                if not outputs or outputs[0].shape != (1, tokens, 1024):
                    raise ValueError(f"Unexpected simulator output shape for {path.name}")
                if not np.isfinite(outputs[0]).all():
                    raise ValueError(f"Non-finite values in full simulator output: {path.name}")
                result = metrics(case["reference"], outputs[0][:, :int(case["valid_tokens"])])
                result.update(name=path.stem, passed=result["relative_l2"] < 0.03 and result["max_abs"] < 0.5)
                report["cases"].append(result)
                print(json.dumps(result), flush=True)
            if not report["cases"] or not all(row["passed"] for row in report["cases"]):
                raise ValueError("Simulator numerical parity failed")
            report["simulator"] = "passed"
        report["passed"] = True
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        traceback.print_exc()
    finally:
        rknn.release()
        report["elapsed_seconds"] = time.monotonic()-started
        if args.report:
            Path(args.report).parent.mkdir(parents=True, exist_ok=True)
            Path(args.report).write_text(json.dumps(report, indent=2)+"\n")
        print(json.dumps(report), flush=True)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
