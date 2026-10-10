"""Metric contract tests; standard library only, no model or torch imports."""
import itertools
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from asr_metrics import edit_counts, normalize_text, score_manifest, score_records, score_text


class ASRMetrics(unittest.TestCase):
    def test_normalization(self):
        self.assertEqual(normalize_text("　你好， 世界！ＡＢＣ１２３\n"), "你好世界abc123")
        self.assertEqual(normalize_text("Straße…\tCafé"), "strassecafé")
        self.assertEqual(normalize_text("繁體，简体 123 一二三"), "繁體简体123一二三")
        self.assertEqual(normalize_text(" a  b\n c!", remove_whitespace=False), "a b c")
        self.assertEqual(normalize_text("价格￥5 + 2"), "价格¥5+2")

    def test_identical_and_full_deletion(self):
        self.assertEqual(score_text("你好，世界！", "你好世界")["error_rate"], 0)
        self.assertEqual(score_text("你好世界", "")["deletions"], 4)
        self.assertEqual(score_text("你好世界", "")["error_rate"], 1)

    def test_edit_counts(self):
        counts = edit_counts("kitten", "sitting")
        self.assertEqual((counts["substitutions"], counts["deletions"], counts["insertions"]), (2, 0, 1))
        self.assertEqual(edit_counts("abc", "ac")["deletions"], 1)
        self.assertEqual(edit_counts("", "abc")["insertions"], 3)
        self.assertEqual(edit_counts("", "")["errors"], 0)

    def test_edit_counts_exhaustively(self):
        strings = ["".join(p) for n in range(4) for p in itertools.product("ab", repeat=n)]
        for reference in strings:
            for hypothesis in strings:
                scored = edit_counts(reference, hypothesis)
                reverse = edit_counts(hypothesis, reference)
                self.assertEqual(scored["errors"], reverse["errors"])
                self.assertEqual(len(reference) - scored["deletions"] + scored["insertions"], len(hypothesis))
                self.assertEqual(scored["errors"], sum(scored[k] for k in ("substitutions", "deletions", "insertions")))

    def test_weighted_corpus_not_utterance_average(self):
        result = score_records([
            {"id": "short", "reference": "a", "hypothesis": "x"},
            {"id": "long", "reference": "bcdefghij", "hypothesis": "bcdefghij"},
        ])
        self.assertAlmostEqual(result["error_rate"], 0.1)
        self.assertEqual(result["reference_units"], 10)

    def test_failure_inclusion(self):
        manifest = {"samples": [{"id": i, "reference": "你好"} for i in range(4)]}
        result = score_manifest(manifest, [
            {"id": 0, "hypothesis": "你好", "status": "ok"},
            {"id": 1, "hypothesis": "你好", "status": "truncated"},
            {"id": 2, "hypothesis": "你好", "error": "runtime failure"},
        ])
        self.assertEqual(result["sample_count"], 4)
        self.assertEqual(result["failure_count"], 3)
        self.assertEqual(result["deletions"], 6)
        self.assertEqual(result["reference_units"], 8)
        self.assertEqual(result["error_rate"], 0.75)
        self.assertEqual(result["samples"][1]["hypothesis"], "你好")
        self.assertEqual(result["samples"][1]["hypothesis_scored"], "")

    def test_empty_success_not_runtime_failure(self):
        result = score_records([{"id": "a", "reference": "你好", "hypothesis": ""}])
        self.assertEqual(result["error_rate"], 1)
        self.assertEqual(result["failure_count"], 0)

    def test_insertion_rate_can_exceed_one(self):
        self.assertEqual(score_text("a", "abcde")["error_rate"], 4)

    def test_word_metric(self):
        result = score_text("Ask NOT what your country can do.", "ask what your country can do", metric="wer")
        self.assertEqual(result["deletions"], 1)
        self.assertEqual(result["reference_units"], 7)
        self.assertEqual(result["error_rate"], 1 / 7)

    def test_bad_inputs(self):
        with self.assertRaises(ValueError):
            score_text("， ！", "anything")
        with self.assertRaises(ValueError):
            score_records([])
        with self.assertRaises(TypeError):
            normalize_text(None)
        with self.assertRaises(ValueError):
            score_text("abc", "abc", metric="accuracy")
        with self.assertRaises(TypeError):
            score_records([{"id": 1, "reference": "a", "hypothesis": 17}])
        with self.assertRaises(ValueError):
            score_records([{"id": 1, "reference": "a"}, {"id": 1, "reference": "b"}])
        manifest = {"samples": [{"id": 1, "reference": "a"}]}
        with self.assertRaises(ValueError):
            score_manifest(manifest, [{"id": 2, "hypothesis": "a"}])
        with self.assertRaises(ValueError):
            score_manifest(manifest, [{"id": 1, "hypothesis": "a"}, {"id": 1, "hypothesis": "a"}])

    def test_cli_includes_missing_predictions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "manifest.json").write_text(json.dumps({"samples": [{"id": "a", "reference": "你好"}]}), encoding="utf-8")
            (root / "predictions.json").write_text("[]", encoding="utf-8")
            output = subprocess.check_output([
                sys.executable, str(Path(__file__).resolve().parents[1] / "scripts/asr_metrics.py"),
                "--manifest", str(root / "manifest.json"), "--predictions", str(root / "predictions.json"),
            ], text=True)
            result = json.loads(output)
            self.assertEqual(result["failure_count"], 1)
            self.assertEqual(result["error_rate"], 1)


if __name__ == "__main__":
    unittest.main()
