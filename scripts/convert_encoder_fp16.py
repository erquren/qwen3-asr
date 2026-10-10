"""Convert weights/compatible operators to FP16, retaining FP32 external I/O."""
import argparse
import onnx
from onnxconverter_common import float16


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    model = onnx.load(args.input)
    model = float16.convert_float_to_float16(model, keep_io_types=True)
    onnx.checker.check_model(model)
    onnx.save(model, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
