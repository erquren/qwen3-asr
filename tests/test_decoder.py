"""Lightweight decoder extraction/validation tests; no model downloads needed."""
import importlib.util
import json
import struct
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


extractor = module("extract_qwen3_asr_decoder")
converter = module("convert_decoder_rkllm")
integrity = module("decoder_integrity")
calibration = module("prepare_decoder_calibration")


class DecoderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source"
        self.source.mkdir()
        self.output = self.root / "output"
        self.text = {"vocab_size": 8, "hidden_size": 4, "intermediate_size": 8,
                     "num_hidden_layers": 1, "num_attention_heads": 2, "num_key_value_heads": 1,
                     "head_dim": 2, "max_position_embeddings": 128, "hidden_act": "silu",
                     "rope_theta": 1000000, "rms_norm_eps": 1e-6, "attention_bias": False,
                     "tie_word_embeddings": True, "rope_scaling": {"rope_type": "default", "mrope_section": [1, 0, 0]}}
        self.write_config()
        (self.source / "tokenizer_config.json").write_text("{}")
        (self.source / "vocab.json").write_text("{}")
        (self.source / "merges.txt").write_text("#version: 0.2\n")
        (self.source / "generation_config.json").write_text(json.dumps({"eos_token_id": [2, 3], "pad_token_id": 0}))
        (self.source / "chat_template.json").write_text(json.dumps({"chat_template": "{{ messages }}"}))
        self.shapes = extractor.expected_shapes(self.text)
        self.write_weights()

    def write_config(self):
        (self.source / "config.json").write_text(json.dumps({"model_type": "qwen3_asr", "thinker_config": {"text_config": self.text}}))

    def write_weights(self, omit=None, dtype="BF16"):
        header = {"__metadata__": {"format": "pt"}}
        data = bytearray()
        for index, (key, shape) in enumerate(sorted(self.shapes.items())):
            if key == omit:
                continue
            values = np.full(shape, index+1, dtype="<f4")
            raw = (values.view("<u4") >> 16).astype("<u2").tobytes() if dtype == "BF16" else values.tobytes()
            header["thinker."+key] = {"shape": shape, "dtype": dtype, "data_offsets": [len(data), len(data)+len(raw)]}
            data.extend(raw)
        encoded = json.dumps(header).encode()
        encoded += b" " * ((-len(encoded)) % 8)
        (self.source / "model.safetensors").write_bytes(struct.pack("<Q", len(encoded)) + encoded + data)

    def test_bf16_extraction_preserves_tensors_and_fp32_embedding(self):
        report = extractor.extract(self.source, self.output, "test-revision")
        config = json.loads((self.output / "config.json").read_text())
        self.assertEqual(report["tensor_count"], 14)
        self.assertFalse(report["stored_head_equals_embedding"])
        self.assertFalse(config["tie_word_embeddings"])
        self.assertIsNone(config["rope_scaling"])
        self.assertEqual(config["architectures"], ["Qwen3ForCausalLM"])
        self.assertEqual(config["eos_token_id"], [2, 3])
        table = np.load(self.output / "embed_tokens.npy")
        self.assertEqual(table.dtype, np.float32)
        self.assertEqual(table.shape, (8, 4))
        self.assertTrue(np.all(table == 2))
        self.assertEqual((self.output / "chat_template.jinja").read_text(), "{{ messages }}")
        source_header, source_start = extractor.read_header(self.source / "model.safetensors")
        output_header, output_start = extractor.read_header(self.output / "model.safetensors")
        a = (self.source / "model.safetensors").read_bytes()
        b = (self.output / "model.safetensors").read_bytes()
        for key in self.shapes:
            s0, s1 = source_header["thinker."+key]["data_offsets"]
            t0, t1 = output_header[key]["data_offsets"]
            self.assertEqual(a[source_start+s0:source_start+s1], b[output_start+t0:output_start+t1])

    def test_fp32_embedding_extraction(self):
        self.write_weights(dtype="F32")
        extractor.extract(self.source, self.output, None)
        self.assertTrue(np.all(np.load(self.output / "embed_tokens.npy") == 2))

    def test_missing_weight_rejected(self):
        self.write_weights(omit="lm_head.weight")
        with self.assertRaisesRegex(ValueError, "Decoder keys differ"):
            extractor.extract(self.source, self.output, None)

    def test_unsupported_rope_rejected(self):
        self.text["rope_scaling"]["rope_type"] = "yarn"
        self.write_config()
        with self.assertRaisesRegex(ValueError, "Only default RoPE"):
            extractor.extract(self.source, self.output, None)

    def test_overwrite_rejected(self):
        self.output.mkdir()
        (self.output / "keep.txt").write_text("keep")
        with self.assertRaisesRegex(ValueError, "Refusing to overwrite"):
            extractor.extract(self.source, self.output, None)

    def test_truncated_tensor_rejected(self):
        p = self.source / "model.safetensors"
        p.write_bytes(p.read_bytes()[:-1])
        with self.assertRaisesRegex(ValueError, "Invalid tensor range"):
            extractor.extract(self.source, self.output, None)

    def convert_args(self, passed=True, **extra):
        self.output.mkdir(exist_ok=True)
        (self.output / "config.json").write_text(json.dumps({"model_type": "qwen3", "architectures": ["Qwen3ForCausalLM"], "max_position_embeddings": 65536}))
        parity = self.root / "parity.json"
        parity.write_text(json.dumps({"passed": passed}))
        args = ["convert_decoder_rkllm.py", "--decoder", str(self.output), "--parity-report", str(parity),
                "--output", str(self.root / "out.rkllm"), "--report", str(self.root / "compile.json")]
        for key, value in extra.items():
            args.extend(["--"+key.replace("_", "-"), str(value)])
        return args

    def test_failed_parity_rejected_before_sdk_import(self):
        with mock.patch.object(sys, "argv", self.convert_args(passed=False)):
            with self.assertRaisesRegex(ValueError, "passing decoder parity"):
                converter.main()

    def test_context_alignment_and_limit_validated(self):
        for value in (0, 33, 16416):
            with self.subTest(value=value), mock.patch.object(sys, "argv", self.convert_args(max_context=value)):
                with self.assertRaisesRegex(ValueError, "multiple of 32"):
                    converter.main()

    def test_quantization_requires_calibration(self):
        with mock.patch.object(sys, "argv", self.convert_args(quant="w8a8")):
            with self.assertRaisesRegex(ValueError, "calibration dataset"):
                converter.main()

    def test_unbound_passing_report_rejected(self):
        with mock.patch.object(sys, "argv", self.convert_args()):
            with self.assertRaisesRegex(ValueError, "not bound to decoder artifact fingerprints"):
                converter.main()

    def test_fingerprints_detect_config_change(self):
        extractor.extract(self.source, self.output, None)
        old = converter.fingerprint_decoder(self.output)
        p = self.output / "config.json"
        p.write_text(p.read_text() + "\n")
        self.assertNotEqual(old, converter.fingerprint_decoder(self.output))

    def test_empty_truncated_and_wrong_dtype_embedding_rejected(self):
        c = {"vocab_size": 8, "hidden_size": 4}
        for value in (np.zeros((0, 4), np.float32), np.zeros((7, 4), np.float32),
                      np.zeros((8, 3), np.float32), np.zeros((8, 4), np.float16)):
            self.assertFalse(integrity.embedding_schema_valid(value, c))
        self.assertTrue(integrity.embedding_schema_valid(np.zeros((8, 4), np.float32), c))

    def test_stale_fingerprint_rejected_before_sdk_import(self):
        extractor.extract(self.source, self.output, None)
        old = converter.fingerprint_decoder(self.output)
        args = self.convert_args()
        (self.root / "parity.json").write_text(json.dumps({"passed": True, "decoder_artifacts": old}))
        with mock.patch.object(sys, "argv", args):
            with self.assertRaisesRegex(ValueError, "changed since numerical parity"):
                converter.main()

    def test_audio_calibration_shape_and_finite_values(self):
        p = self.root / "audio.npy"
        np.save(p, np.ones((3, 4), np.float32))
        self.assertEqual(calibration.load_audio_embedding(p, "reference", 4).shape, (1, 3, 4))
        for bad in (np.ones((2, 3, 4), np.float32), np.ones((1, 0, 4), np.float32),
                    np.ones((1, 3, 5), np.float32), np.full((1, 3, 4), np.nan)):
            np.save(p, bad)
            with self.assertRaises(ValueError):
                calibration.load_audio_embedding(p, "reference", 4)

    def test_compile_wrapper_records_verified_artifact_and_stages(self):
        extractor.extract(self.source, self.output, None)
        args = self.convert_args()
        (self.root / "parity.json").write_text(json.dumps({"passed": True, "decoder_artifacts": converter.fingerprint_decoder(self.output)}))
        calls = {}
        class FakeSDK:
            def load_huggingface(self, **kwargs):
                calls["load"] = kwargs
                return 0
            def build(self, **kwargs):
                calls["build"] = kwargs
                return 0
            def export_rkllm(self, path):
                Path(path).write_bytes(b"unit-test mock artifact")
                return 0
        fake_torch = types.SimpleNamespace(set_num_threads=lambda _: None)
        fake_api = types.SimpleNamespace(RKLLM=FakeSDK)
        with mock.patch.object(sys, "argv", args), mock.patch.dict(sys.modules, {"torch": fake_torch, "rkllm.api": fake_api}), \
             mock.patch.object(converter.importlib.metadata, "version", return_value="unit-test"), mock.patch("builtins.print"), \
             mock.patch.dict(converter.os.environ):
            converter.main()
        result = json.loads((self.root / "compile.json").read_text())
        self.assertTrue(result["passed"])
        self.assertEqual(result["stage"], "complete")
        self.assertEqual(result["steps"], {"load_huggingface": 0, "build": 0, "export_rkllm": 0})
        self.assertEqual(calls["load"]["device"], "cpu")
        self.assertEqual(calls["build"]["target_platform"], "rk3588")
        self.assertEqual(calls["build"]["num_npu_core"], 3)
        self.assertFalse(calls["build"]["do_quantization"])
        self.assertIn("exporting", result["stage_rss"])
        self.assertEqual(len(result["sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
