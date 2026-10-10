"""ASR generation protocol regressions with tiny mocked models, no weights."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_asr_accuracy


class AccuracyProtocol(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.inputs = self.root / "inputs"
        self.inputs.mkdir()
        self.audio = self.root / "audio"
        self.audio.mkdir()
        self.ids = np.array([[2, 5, 3]], dtype=np.int64)
        np.savez(self.inputs / "sample.npz", input_ids=self.ids,
                 attention_mask=np.ones_like(self.ids),
                 input_features=np.zeros((1, 128, 100), dtype=np.float32),
                 feature_attention_mask=np.ones((1, 100), dtype=np.int64))
        self.manifest = {"samples": [{"id": "one", "reference": "你好", "key": "sample",
                                      "input_file": "sample.npz", "audio_tokens": 1}]}
        (self.inputs / "inputs.json").write_text(json.dumps(self.manifest))
        np.save(self.audio / "sample.npy", np.array([[[11, 12, 13, 14]]], dtype=np.float32))
        self.generated = [7, 151645]
        self.decoded = []
        self.calls = []
        self.tokenizer = SimpleNamespace(audio_token="<|audio_pad|>",
                                         convert_tokens_to_ids=lambda _: 5,
                                         decode=self.decode)

    def decode(self, tokens, **kwargs):
        self.decoded.append((tokens.tolist(), kwargs))
        return "你好"

    def run_mode(self, mode, generated=None, limit=256):
        if generated is not None:
            self.generated = generated
        parent = self

        class FakeThinker:
            def generate(self, input_ids, **kwargs):
                parent.calls.append(kwargs)
                continuation = torch.tensor([parent.generated], dtype=torch.long)
                return SimpleNamespace(sequences=torch.cat((input_ids, continuation), dim=1))

        class FakeOfficial:
            # Reproduce qwen-asr 0.0.6's wrapper contract. Passing the option to
            # this outer call must fail because the wrapper supplies it itself.
            def generate(self, input_ids=None, **kwargs):
                return FakeThinker().generate(input_ids=input_ids, return_dict_in_generate=True, **kwargs)

        class FakeDecoder:
            def __init__(self):
                self.config = SimpleNamespace(hidden_size=4)
                self.model = SimpleNamespace(embed_tokens=lambda ids: ids.float().unsqueeze(-1).expand(-1, -1, 4).clone())

            def eval(self):
                return self

            def generate(self, **kwargs):
                parent.calls.append(kwargs)
                return SimpleNamespace(sequences=torch.tensor([parent.generated], dtype=torch.long))

        official_factory = mock.Mock(return_value=SimpleNamespace(
            model=FakeOfficial(), processor=SimpleNamespace(tokenizer=self.tokenizer)))
        fake_qwen_asr = SimpleNamespace(Qwen3ASRModel=SimpleNamespace(from_pretrained=official_factory))
        fake_transformers = SimpleNamespace(
            Qwen3ForCausalLM=SimpleNamespace(from_pretrained=mock.Mock(return_value=FakeDecoder())),
            AutoTokenizer=SimpleNamespace(from_pretrained=mock.Mock(return_value=self.tokenizer)))
        output = self.root / f"{mode}.json"
        argv = ["run_asr_accuracy.py", "--mode", mode, "--model", str(self.root / "no_weights"),
                "--inputs", str(self.inputs), "--embeddings", str(self.audio), "--label", mode,
                "--max-new-tokens", str(limit), "--output", str(output)]
        with mock.patch.object(sys, "argv", argv), mock.patch.dict(sys.modules, {
            "qwen_asr": fake_qwen_asr, "transformers": fake_transformers,
        }), mock.patch("builtins.print"), mock.patch.object(torch, "set_num_threads"):
            run_asr_accuracy.main()
        return json.loads(output.read_text()), official_factory

    def test_official_wrapper_owns_return_dict_and_crops_prompt(self):
        report, factory = self.run_mode("official")
        self.assertEqual(report["samples"][0]["token_ids"], self.generated)
        self.assertEqual(self.decoded[0][0], self.generated)
        self.assertEqual(report["samples"][0]["status"], "ok")
        self.assertIs(factory.call_args.kwargs["dtype"], torch.float32)
        self.assertEqual(factory.call_args.kwargs["attn_implementation"], "eager")
        self.assertFalse(self.calls[0]["do_sample"])
        self.assertEqual(self.calls[0]["num_beams"], 1)
        self.assertTrue(self.calls[0]["return_dict_in_generate"])

    def test_decoder_replaces_only_audio_and_keeps_generated_prefix(self):
        report, _ = self.run_mode("decoder")
        embedded = self.calls[0]["inputs_embeds"].numpy()
        np.testing.assert_array_equal(embedded, [[[2, 2, 2, 2], [11, 12, 13, 14], [3, 3, 3, 3]]])
        self.assertNotIn("input_ids", self.calls[0])
        self.assertFalse(self.calls[0]["do_sample"])
        self.assertEqual(self.calls[0]["num_beams"], 1)
        self.assertEqual(report["samples"][0]["token_ids"], self.generated)
        self.assertEqual(self.decoded[0][0], self.generated)
        self.assertEqual(report["rkllm_inference"], "not_run")

    def test_limit_without_eos_is_retained_and_flagged(self):
        report, _ = self.run_mode("official", generated=[7, 8], limit=2)
        row = report["samples"][0]
        self.assertEqual(row["token_ids"], [7, 8])
        self.assertEqual(row["hypothesis"], "你好")
        self.assertTrue(row["generation_limit_reached"])
        self.assertEqual(row["status"], "truncated")

    def test_eos_at_generation_limit_is_success(self):
        for eos in (151643, 151645):
            with self.subTest(eos=eos):
                report, _ = self.run_mode("official", generated=[7, eos], limit=2)
                self.assertFalse(report["samples"][0]["generation_limit_reached"])
                self.assertEqual(report["samples"][0]["status"], "ok")

    def test_audio_placeholder_mismatch_is_rejected(self):
        np.save(self.audio / "sample.npy", np.zeros((1, 2, 4), dtype=np.float32))
        self.manifest["samples"][0]["audio_tokens"] = 2
        (self.inputs / "inputs.json").write_text(json.dumps(self.manifest))
        with self.assertRaisesRegex(ValueError, "placeholder count"):
            self.run_mode("decoder")


if __name__ == "__main__":
    unittest.main()
