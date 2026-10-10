"""Reproduce the five fixed FLEURS rows from a hash-verified public Parquet."""
import argparse
import hashlib
import json
from pathlib import Path

REVISION = "168de341b3db6859a9bac1c50a2ef5e3b47647e0"
SOURCE_SHA256 = "87c0aebbe183f3a36ac87b5c3421b6ab57036824744ff695029a3f858e7622fd"
SOURCE_BYTES = 695674033
ROWS = (9, 10, 11, 26, 34)
SOURCE_URL = f"https://huggingface.co/datasets/google/fleurs/resolve/{REVISION}/cmn_hans_cn/test/0000.parquet"


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b""):
            result.update(block)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.parquet.stat().st_size != SOURCE_BYTES or sha256(args.parquet) != SOURCE_SHA256:
        raise ValueError("Source parquet size/SHA-256 differs from pinned corpus")
    import pyarrow.parquet as pq
    table = pq.read_table(args.parquet, columns=["id", "audio", "raw_transcription", "num_samples"])
    (args.output / "audio").mkdir(parents=True, exist_ok=True)
    samples = []
    for index in ROWS:
        audio = table["audio"][index].as_py()
        name = Path(audio["path"]).name
        if not name.endswith(".wav") or name in {".wav", "..wav"}:
            raise ValueError("Unexpected source audio filename")
        data = audio["bytes"]
        (args.output / "audio" / name).write_bytes(data)
        samples.append({
            "id": f"fleurs_cmn_hans_cn_test_{Path(name).stem}",
            "fleurs_sentence_id": table["id"][index].as_py(),
            "source_row_index": index, "source_filename": name,
            "audio_path": f"audio/{name}", "language": "Chinese",
            "reference": table["raw_transcription"][index].as_py(),
            "duration_seconds": table["num_samples"][index].as_py() / 16000,
            "audio_sha256": hashlib.sha256(data).hexdigest(),
            "audio_size_bytes": len(data),
        })
    manifest = {"dataset": "google/fleurs", "config": "cmn_hans_cn", "split": "test",
                "revision": REVISION, "source_sha256": SOURCE_SHA256,
                "source_size_bytes": SOURCE_BYTES, "source_url": SOURCE_URL,
                "license": "CC-BY-4.0", "license_url": "https://creativecommons.org/licenses/by/4.0/",
                "attribution": "Conneau et al. (2022), FLEURS; https://arxiv.org/abs/2205.12446",
                "selection": "Fixed before inference: first 3 female and 2 male recordings with 1..8 seconds, no ASCII letters/digits, distinct sentence IDs.",
                "source_modifications": "none; exact embedded WAVE bytes and raw_transcription",
                "samples": samples}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False)+"\n")
    print(f"Verified and extracted {len(samples)} complete utterances")


if __name__ == "__main__":
    main()
