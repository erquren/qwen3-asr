#!/usr/bin/env python3
"""Dependency-free, failure-inclusive ASR scoring for fixed labeled manifests.

CER normalizes Unicode with NFKC and casefold, removes all Unicode punctuation
(categories P*) and whitespace, then scores Unicode code points. It does not
rewrite numbers, translate text, or convert traditional/simplified Chinese.
WER uses the same normalization but preserves whitespace word boundaries.

Corpus rates are sum(S + D + I) / sum(reference units), not an average of
utterance rates. Missing/failed/truncated predictions are scored as empty text
(all reference units deleted) and remain in the denominator. Raw partial
outputs are preserved for diagnosis. Empty normalized references are rejected.
This is a transparent smoke-test metric, not an official FLEURS scoring recipe.
"""
from __future__ import annotations

import argparse
import json
import unicodedata
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any


NORMALIZATION = {
    "unicode": "NFKC",
    "case": "casefold",
    "punctuation": "remove Unicode general categories P*",
    "cer_whitespace": "remove characters where str.isspace() is true",
    "cer_units": "Unicode code points",
    "wer_units": "whitespace-delimited tokens after punctuation removal",
    "number_or_script_conversion": False,
    "aggregation": "sum(S+D+I) / sum(reference units)",
    "failure_policy": "missing, non-ok, or errored predictions become empty text; never drop references",
    "empty_reference_policy": "raise ValueError",
    "edit_tie_break": "substitution, deletion, insertion",
}


def normalize_text(text: str, *, remove_whitespace: bool = True) -> str:
    """Normalize without modifying the caller's original reference text."""
    if not isinstance(text, str):
        raise TypeError("ASR text must be a string")
    text = unicodedata.normalize("NFKC", text).casefold()
    text = "".join(ch for ch in text if not unicodedata.category(ch).startswith("P"))
    if remove_whitespace:
        return "".join(ch for ch in text if not ch.isspace())
    return " ".join(text.split())


def edit_counts(reference: Sequence[Any], hypothesis: Sequence[Any]) -> dict[str, int]:
    """Levenshtein distance and deterministic S/D/I breakdown; O(n*m) memory."""
    n, m = len(reference), len(hypothesis)
    costs = [list(range(m + 1))]
    for i in range(1, n + 1):
        row = [i] + [0] * m
        for j in range(1, m + 1):
            row[j] = min(
                costs[i - 1][j - 1] + (reference[i - 1] != hypothesis[j - 1]),
                costs[i - 1][j] + 1,
                row[j - 1] + 1,
            )
        costs.append(row)
    i, j = n, m
    substitutions = deletions = insertions = 0
    while i or j:
        if i and j and reference[i - 1] == hypothesis[j - 1] and costs[i][j] == costs[i - 1][j - 1]:
            i, j = i - 1, j - 1
        elif i and j and costs[i][j] == costs[i - 1][j - 1] + 1:
            substitutions += 1
            i, j = i - 1, j - 1
        elif i and costs[i][j] == costs[i - 1][j] + 1:
            deletions += 1
            i -= 1
        else:
            insertions += 1
            j -= 1
    return {
        "substitutions": substitutions,
        "deletions": deletions,
        "insertions": insertions,
        "errors": substitutions + deletions + insertions,
        "reference_units": n,
        "hypothesis_units": m,
    }


def score_text(reference: str, hypothesis: str, *, metric: str = "cer") -> dict[str, Any]:
    """Score one successful hypothesis; error_rate is a fraction, not percent."""
    if metric not in {"cer", "wer"}:
        raise ValueError("metric must be cer or wer")
    ref = normalize_text(reference, remove_whitespace=metric == "cer")
    hyp = normalize_text(hypothesis, remove_whitespace=metric == "cer")
    ref_units = ref if metric == "cer" else ref.split()
    hyp_units = hyp if metric == "cer" else hyp.split()
    if not ref_units:
        raise ValueError("normalized reference must not be empty")
    counts = edit_counts(ref_units, hyp_units)
    return {
        **counts,
        "error_rate": counts["errors"] / counts["reference_units"],
        "reference_normalized": ref,
        "hypothesis_normalized": hyp,
    }


def score_records(records: Iterable[dict[str, Any]], *, metric: str = "cer") -> dict[str, Any]:
    """Score records with id, reference, hypothesis, and optional status/error.

    A successful status is 'ok' or 'success'. Absent status defaults to 'ok'
    only when a string hypothesis is supplied. None is a missing prediction.
    """
    samples = []
    seen = set()
    totals = dict.fromkeys(("substitutions", "deletions", "insertions", "errors", "reference_units", "hypothesis_units"), 0)
    failures = 0
    for record in records:
        sample_id = record["id"]
        if sample_id in seen:
            raise ValueError(f"duplicate sample ID: {sample_id}")
        seen.add(sample_id)
        original = record.get("hypothesis")
        if original is not None and not isinstance(original, str):
            raise TypeError(f"hypothesis for {sample_id} must be a string or None")
        status = record.get("status", "ok" if original is not None else "missing")
        error = record.get("error")
        failed = status not in {"ok", "success"} or original is None or bool(error)
        scored_hypothesis = "" if failed else original
        scored = score_text(record["reference"], scored_hypothesis, metric=metric)
        samples.append({
            "id": sample_id,
            "language": record.get("language"),
            "reference": record["reference"],
            "hypothesis": original,
            "hypothesis_scored": scored_hypothesis,
            "status": status,
            "failed": failed,
            "error": error,
            **scored,
        })
        failures += int(failed)
        for key in totals:
            totals[key] += scored[key]
    if not samples:
        raise ValueError("cannot score an empty corpus")
    return {
        "metric": metric,
        "normalization": NORMALIZATION,
        "sample_count": len(samples),
        "success_count": len(samples) - failures,
        "failure_count": failures,
        **totals,
        "error_rate": totals["errors"] / totals["reference_units"],
        "error_rate_percent": 100 * totals["errors"] / totals["reference_units"],
        "samples": samples,
    }


def score_manifest(manifest: dict[str, Any], predictions: Sequence[dict[str, Any]], *, metric: str = "cer") -> dict[str, Any]:
    """Join predictions onto the entire manifest so missing rows cannot vanish."""
    by_id = {}
    for prediction in predictions:
        sample_id = prediction["id"]
        if sample_id in by_id:
            raise ValueError(f"duplicate prediction ID: {sample_id}")
        by_id[sample_id] = prediction
    expected = {sample["id"] for sample in manifest["samples"]}
    unexpected = by_id.keys() - expected
    if unexpected:
        raise ValueError(f"predictions contain unknown sample IDs: {sorted(unexpected, key=str)}")
    records = []
    for sample in manifest["samples"]:
        prediction = by_id.get(sample["id"], {"status": "missing", "hypothesis": None})
        records.append({
            "id": sample["id"],
            "reference": sample["reference"],
            "language": sample.get("language"),
            "hypothesis": prediction.get("hypothesis"),
            "status": prediction.get("status", "ok"),
            "error": prediction.get("error"),
        })
    return score_records(records, metric=metric)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True, help="JSON list of {id,hypothesis,status?,error?}, or object with predictions/samples list")
    parser.add_argument("--metric", choices=("cer", "wer"), default="cer")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    predictions = json.loads(args.predictions.read_text(encoding="utf-8"))
    if isinstance(predictions, dict):
        predictions = predictions.get("predictions", predictions.get("samples"))
    if not isinstance(predictions, list):
        parser.error("predictions must be a JSON list or an object with a predictions/samples list")
    report = score_manifest(manifest, predictions, metric=args.metric)
    result = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(result, encoding="utf-8")
    else:
        print(result, end="")


if __name__ == "__main__":
    main()
