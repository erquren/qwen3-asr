# RKLLM 1.2.3 toolchain and decoder audit

## Scope

This project extracts the decoder from **Qwen/Qwen3-ASR-0.6B**, revision
`5eb144179a02acc5e5ba31e748d22b0cf3e303b0`, and tests whether Rockchip's host
toolchain can compile it for **RK3588, three NPU cores**. Host compilation and
PyTorch parity are separate from actual board inference. No board latency,
recognition quality, or audio-input runtime compatibility is implied by a
successful export.

## Official SDK sources

- [Rockchip release 1.2.3](https://github.com/airockchip/rknn-llm/tree/release-v1.2.3)
- [Official requirements](https://github.com/airockchip/rknn-llm/blob/release-v1.2.3/rkllm-toolkit/packages/requirements.txt)
- [Official conversion example](https://github.com/airockchip/rknn-llm/blob/release-v1.2.3/examples/rkllm_api_demo/export/export_rkllm.py)
- [Official ASR implementation](https://github.com/QwenLM/Qwen3-ASR/blob/main/qwen_asr/core/transformers_backend/modeling_qwen3_asr.py)

The release provides Linux x86_64 wheels for CPython 3.9, 3.10, 3.11 and 3.12.
The tested wheel is `rkllm_toolkit-1.2.3-cp312-cp312-linux_x86_64.whl`, 7,001,169
bytes, SHA-256:

```text
5437f04f7c455aae10f0098bf2d9a053fb67cfe8bc8b89a4bbc3772e93cb059e
```

The SDK explicitly supports `load_huggingface(device="cpu")`. CUDA is optional
for this conversion path. Its API constrains `max_context` to a multiple of 32
from 32 through 16,384; RK3588 supports one, two, or three NPU cores.

## Dependency isolation

The ASR reference environment uses `torch==2.6.0+cpu`,
`transformers==4.57.6`, `accelerate==1.12.0`, and `qwen-asr==0.0.6`.
RKLLM 1.2.3 instead pins `transformers==4.55.2`,
`accelerate>=1.0.1,<=1.5.2`, `safetensors==0.5.3`, and NumPy <=1.26.4.
**Do not install both full dependency sets into the same environment.**

A clean SDK environment can be prepared as follows. The extra CPU constraint
prevents pip from silently fetching NVIDIA/CUDA packages during dependency
resolution. Install the CPU torch and torchvision wheels before the SDK.

```bash
python3.12 -m venv .venv-rkllm
source .venv-rkllm/bin/activate
python -m pip install --upgrade pip setuptools wheel
printf 'torch==2.6.0+cpu\ntorchvision==0.21.0+cpu\n' > constraints-cpu.txt
python -m pip install --index-url https://download.pytorch.org/whl/cpu \
  -c constraints-cpu.txt torch==2.6.0+cpu torchvision==0.21.0+cpu
curl -fL \
  https://raw.githubusercontent.com/airockchip/rknn-llm/release-v1.2.3/rkllm-toolkit/packages/rkllm_toolkit-1.2.3-cp312-cp312-linux_x86_64.whl \
  -o rkllm_toolkit-1.2.3-cp312-cp312-linux_x86_64.whl
echo '5437f04f7c455aae10f0098bf2d9a053fb67cfe8bc8b89a4bbc3772e93cb059e  rkllm_toolkit-1.2.3-cp312-cp312-linux_x86_64.whl' | sha256sum -c -
export BUILD_CUDA_EXT=0
python -m pip install --no-build-isolation -c constraints-cpu.txt \
  -r requirements-rkllm.txt \
  ./rkllm_toolkit-1.2.3-cp312-cp312-linux_x86_64.whl
python -c 'from rkllm.api import RKLLM; RKLLM()'
```

The constrained container test used an SDK overlay virtual environment that
read the existing CPU-only base packages, then installed SDK-pinned versions
locally. It did not overwrite the ASR environment. The SDK overlay consumed
about 520 MiB beyond the shared CPU base. Import and `RKLLM()` construction
passed with `torch==2.6.0+cpu`, `torchvision==0.21.0+cpu`,
`transformers==4.55.2`, and CUDA unavailable. The original ASR environment
remained on Transformers 4.57.6 and Accelerate 1.12.0. The clean-environment
commands above are the recommended independent reproduction, rather than
depending on an absolute-path overlay.

## Extraction semantics

`extract_qwen3_asr_decoder.py` is independently written and does not execute
downloaded model code. It parses safetensors metadata, checks every expected
decoder tensor name and shape, and copies these weights byte-for-byte:

- `thinker.model.*` → `model.*`
- `thinker.lm_head.weight` → `lm_head.weight`

The extracted checkpoint has 311 stored tensors, 751,632,384 stored parameters
(including both embedding/head copies), and 1,503,264,768 tensor-data bytes.
Counting the identical head/embedding matrix only once gives 596,049,920
unique decoder parameters, consistent with the model's 0.6B designation.
It keeps BF16 weights and additionally writes `embed_tokens.npy` as FP32 with
shape `[151936, 1024]`. Converting this embedding table is chunked and does not
need a full PyTorch model in memory. Safetensors headers and NumPy headers add
small overhead to the above raw byte counts.

The official checkpoint stores separate head and embedding tensors while its
text config says `tie_word_embeddings=true`. Their extracted SHA-256 hashes
are identical. The standalone config nevertheless sets
`tie_word_embeddings=false` so that both stored tensors are preserved exactly
and future checkpoint differences cannot be silently overwritten by automatic
weight tying.

The configuration is standard `Qwen3ForCausalLM` with default ordinary RoPE.
This adaptation is justified for the official ASR position scheme: its
`get_rope_index` repeats the same position indices over all three T/H/W planes.
Selecting/interleaving equal planes is equivalent to ordinary RoPE. The
adaptation must not be generalized to inputs with unequal multidimensional
position indices. No tensor permutation is needed for equal-plane positions.

Tokenizer, vocabulary, merges, generation configuration, and chat-template
files are copied. Original EOS token IDs `[151643, 151645]` and padding ID
`151643` are retained. Model vocabulary capacity is 151,936, while the copied
tokenizer currently exposes 151,705 tokens; reserved embedding rows are kept.

## Test stages

1. Lightweight unit tests cover BF16 and FP32 extraction, exact tensor-byte
   preservation, FP32 embedding values, configuration changes, chat templates,
   malformed/truncated inputs, missing tensors, non-default RoPE rejection,
   overwrite rejection, parity gating, context limits and calibration gating.
2. `check_decoder_parity.py` loads official package classes and standard Qwen3
   in separate sequential CPU FP32 passes. It tests token lengths 1/17/64,
   mixed token/random acoustic-like embeddings, left padding, prefill plus
   three cached decoding steps, rotary positions through 65,535, and every
   row of the FP32 embedding artifact. Random embedding input is a decoder
   interface test, not transcription accuracy evaluation.
3. `convert_decoder_rkllm.py` requires a passing parity report and emits a
   machine-readable report for SDK loading, build, artifact export, size,
   SHA-256, and exceptions. FP16 does not require calibration. W8A8/W4A16
   require an explicitly supplied dataset; text-only calibration should not
   be presented as validated ASR calibration.
4. Real RK3588 runtime checks remain separate: matching runtime/driver,
   external audio/token embeddings, cache handling, EOS handling, audio
   transcription accuracy and performance on representative recordings.

Actual run results belong in the repository's test report; generated model
weights, SDK wheels and private recordings are intentionally not committed.

### Artifact identity and calibration format

The parity checker fingerprints decoder weights, config, FP32 embedding table,
tokenizer/vocabulary, generation settings and chat templates before testing,
and verifies they remain unchanged afterward. Conversion recomputes these
hashes and refuses an unbound or stale parity report. The embedding test also
requires exactly `[vocab_size, hidden_size]` and FP32 before comparing all rows;
an empty or shortened array cannot pass vacuously.

`prepare_decoder_calibration.py` uses the original ASR chat prompt and token
IDs, expands the audio placeholder once per encoder output frame, looks up
the surrounding text embeddings, and inserts the supplied real encoder
outputs. It saves both `.npy` arrays and vendor-ready JSON. Verified reference
transcriptions can optionally be supplied with `--target`; no target is
invented when that argument is omitted.

**RKLLM 1.2.3 format gotcha:** `input_embed` must contain nested numeric lists
of shape `[1, sequence, hidden]`. It does not accept a path string to an `.npy`
file, despite the abbreviated SDK API doc showing an empty string as an
example. This was tested with the installed vendor `get_calib_dataset`:
nested 3D lists succeeded, `.npy` path strings and 2D lists failed. The official
[multimodal calibration example](https://github.com/airockchip/rknn-llm/blob/release-v1.2.3/examples/multimodal_model_demo/data/make_input_embeds_for_quantize.py)
also serializes the loaded arrays as lists. The conversion wrapper directs
Hugging Face dataset caches into a writable output-side directory by default.

The actual smoke calibration combines official FP32 encoder outputs from the
same speech fixture cropped/padded into 2 s, 4 s and 15 s test cases. Audio
frame counts are 26, 52 and 195; assembled forced-English prompt lengths are
44, 70 and 213 tokens. No reference transcription is invented or supplied.
The vendor loader successfully reads all three inputs and pads/batches them
to `[3, 213, 1024]`. This checks that real acoustic features reach the
quantizer. Three lengths of one English fixture are not a diverse,
representative calibration corpus, and do not establish recognition accuracy
or a quantized-vs-FP16 WER/CER result.

### Reproducing acoustic calibration

Run this preparation with the ASR/export environment, after reference encoder
outputs have been produced. Each NPZ file must contain the `reference` output
with hidden width 1024. These commands use no invented transcription target:

```bash
python scripts/prepare_decoder_calibration.py \
  --source work/models/Qwen3-ASR-0.6B \
  --decoder work/models/decoder_hf \
  --audio-embeddings work/cases/2s/speech.npz \
  --audio-embeddings work/cases/4s/speech.npz \
  --audio-embeddings work/cases/15s/speech.npz \
  --language English --output-dir work/calibration/speech
```

Then use the separate SDK environment for compilation, pointing at the
passing report produced by `check_decoder_parity.py`:

```bash
.venv-rkllm/bin/python scripts/convert_decoder_rkllm.py \
  --decoder work/models/decoder_hf \
  --parity-report work/cases/decoder-parity.json \
  --quant w8a8 --load-dtype float16 --max-context 4096 \
  --reclaim-memory-before-export \
  --dataset work/calibration/speech/data_quant.json \
  --output work/artifacts/decoder.w8a8.rk3588.rkllm \
  --report work/cases/decoder-w8a8-compile.json
```

The optional memory-reclaim flag performs Python garbage collection and
requests glibc to return unused allocator pages before serialization. It does
not patch Rockchip code, remove model tensors, or alter calibration inputs.
The wrapper saves stage-by-stage status and RSS measurements before export,
so an externally killed process leaves evidence of its last completed stage.

### Host resource failures are not successful exports

On the test host, the first W8A8 attempt (FP32 loading) and second attempt
(FP16 loading) completed all 28 optimization layers and 311 conversion
tensors, but were terminated by SIGKILL during serialization and left empty
files. The second attempt's direct-child high-water RSS was 5,603,572 KiB,
and the process ran 127.75 seconds. Exit 137 / signal 9 is established;
an OOM cause is inferred from resource pressure, not proven by a kernel log
or cgroup OOM counter. Empty output files are not usable artifacts and must
not count as successful builds.

Approximately 1.3 GiB of inactive temporary environment data occupied the
RAM-backed `/tmp`. It was moved non-destructively to disk with its original
path preserved by a symlink, increasing available RAM from about 6.5 GiB to
7.7 GiB. A subsequent attempt retained the **same three acoustic calibration
inputs**, FP16 loading, RK3588 target, three cores and 4096-token context,
and enabled the optional allocator cleanup. Its final result is recorded
in the test report; the earlier failures remain part of the evidence.

## Final verified host results

The hardened reference-vs-extracted decoder check passed in 14.63 seconds:
all 16 hidden/logit/rotary comparisons were **bit-identical** (maximum absolute
error 0), all compared greedy token IDs matched, and the full FP32 embedding
table matched exactly. All nine decoder-asset fingerprints were unchanged
during testing. The 15 lightweight decoder tests also passed.

| Artifact | SDK load / build / export | Size, bytes | SDK elapsed |
| --- | --- | ---: | ---: |
| FP16, CPU FP32 load | 0 / 0 / 0 | 1,529,743,212 | 16.68 s |
| W8A8, CPU FP16 load, same three acoustic samples | 0 / 0 / 0 | 936,054,076 | 126.71 s |

Both successful artifacts use RK3588, three cores, optimization level 1 and
4096-token context. FP16 was rebuilt after fingerprint gating was added and
had the same SHA-256 as its earlier successful build:

```text
FP16  b186eae6e5e22be630363d1c1ae6509beb8a90896458ec8fd53da9ad1536af55
W8A8  0e9942a3c9723a760dd8283081fcc7a032c28d66c3f1e41df2506746984f0d15
```

The successful W8A8 process high-water RSS was 5,646,988 KiB (about 5.39 GiB).
One-second process-tree sampling observed only about 4.04 GiB and missed the
short serialization peak; the high-water metric is therefore reported
separately. Allocator cleanup reduced live RSS from 3,723,380 KiB after build
to 3,485,288 KiB immediately before export. The wrapper's wall time including
startup and one-second sampling was 130.20 seconds. This resource evidence
supports a memory-pressure explanation for the previous failures, but still
does not establish a kernel-confirmed OOM event.

**Not established by these tests:** execution of either `.rkllm` on a real
RK3588, quantized ASR quality, multilingual WER/CER, real-time streaming,
board memory use, or board latency. The passing host artifacts and unquantized
decoder parity do not substitute for those checks.
