"""Run ONNX Runtime CPU on saved independent official encoder goldens."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import onnx
import onnxruntime as ort


def metrics(reference, actual):
    if reference.shape != actual.shape:
        raise ValueError(f"Output shape mismatch: {reference.shape} != {actual.shape}")
    reference, actual = reference.astype(np.float64).ravel(), actual.astype(np.float64).ravel()
    if actual.shape != reference.shape or not np.isfinite(actual).all():
        raise ValueError("Invalid output shape or non-finite values")
    delta = actual-reference
    return {"cosine": float(np.dot(reference, actual) / (np.linalg.norm(reference)*np.linalg.norm(actual))),
            "max_abs": float(np.max(np.abs(delta))),
            "relative_l2": float(np.linalg.norm(delta)/np.linalg.norm(reference))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--cases", required=True)
    parser.add_argument("--precision", choices=["fp32", "fp16"], required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    onnx.checker.check_model(args.onnx)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(args.onnx, sess_options=options, providers=["CPUExecutionProvider"])
    limits = {"fp32": (1e-4, 0.003), "fp16": (0.01, 0.2)}[args.precision]
    report = {"precision": args.precision, "runtime": ort.__version__, "onnx_checker": "passed",
              "inputs": [{"name": v.name, "shape": v.shape, "type": v.type} for v in session.get_inputs()],
              "outputs": [{"name": v.name, "shape": v.shape, "type": v.type} for v in session.get_outputs()],
              "limits": {"relative_l2": limits[0], "max_abs": limits[1]}, "cases": []}
    for file in sorted(Path(args.cases).glob("*.npz")):
        case = np.load(file)
        start = time.perf_counter()
        output = session.run(None, {key: case[key] for key in ("input_features", "attention_mask")})[0]
        elapsed = time.perf_counter()-start
        if not np.isfinite(output).all():
            raise ValueError(f"Non-finite values in full output, including padding: {file.name}")
        result = metrics(case["reference"], output[:, :int(case["valid_tokens"])])
        passed = result["relative_l2"] < limits[0] and result["max_abs"] < limits[1]
        row = {"name": file.stem, **result, "passed": passed, "cpu_seconds": elapsed}
        report["cases"].append(row)
        print(json.dumps(row), flush=True)
    if not report["cases"]:
        raise ValueError("No goldens found")
    report["passed"] = all(case["passed"] for case in report["cases"])
    Path(args.report).write_text(json.dumps(report, indent=2)+"\n")
    if not report["passed"]:
        raise SystemExit("ONNX parity failed")


if __name__ == "__main__":
    main()
