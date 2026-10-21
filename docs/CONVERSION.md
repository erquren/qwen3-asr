# 从官方权重转换与验证

目标：Qwen3-ASR-0.6B 的 **RKNN 兼容 encoder ONNX**。decoder 使用独立 HF → RKLLM 路径。
所有命令从仓库根目录执行。下列三个 Python 环境必须分开；CPU 可完成此次转换，不需要 CUDA。

## 1. ONNX / 官方参考环境

测试版本：Linux x86_64、Python 3.12、CPU PyTorch 2.6.0、Transformers 4.57.6、qwen-asr 0.0.6。
建议至少 10 GB RAM，并为权重、多个 ONNX/RKNN/RKLLM 产物及依赖留出约 20 GB 空间。串行执行大模型步骤。

```bash
python3.12 -m venv .venv-export
.venv-export/bin/pip install torch==2.6.0+cpu --index-url https://download.pytorch.org/whl/cpu
.venv-export/bin/pip install -r requirements-export.txt
.venv-export/bin/pip install --no-deps qwen-asr==0.0.6
.venv-export/bin/python scripts/download_model.py --output models/Qwen3-ASR-0.6B
```

最后一步固定官方 revision `5eb144179a02acc5e5ba31e748d22b0cf3e303b0`，仅下载 safetensors 与配置/tokenizer；不会执行远程模型代码。
`source_manifest.json` 记录每个文件的 SHA-256。qwen-asr 的 UI、服务端、vLLM 附加依赖不参与这里的导出。

## 2. 导出固定输入 ONNX

```bash
for s in 2 4 15; do
  .venv-export/bin/python scripts/export_encoder.py \
    --model models/Qwen3-ASR-0.6B --seconds "$s" \
    --output "artifacts/encoder-${s}s.fp32.onnx"
  .venv-export/bin/python scripts/convert_encoder_fp16.py \
    --input "artifacts/encoder-${s}s.fp32.onnx" \
    --output "artifacts/encoder-${s}s.fp16.onnx"
done
```

opset 18；没有 TensorRT 自定义算子；外部输入和输出保持 FP32。FP16 文件指内部权重/兼容算子转为 FP16，不能把输入直接改成 FP16。

| 桶 | input_features | attention_mask | audio_embeds |
| --- | --- | --- | --- |
| 2 s | `[1,128,200]` | `[1,1,26,26]` | `[1,26,1024]` |
| 4 s | `[1,128,400]` | `[1,1,52,52]` | `[1,52,1024]` |
| 15 s | `[1,128,1500]` | `[1,1,195,195]` | `[1,195,1024]` |

内部 encoder 宽度 896，最终输出投影为 1024。导出器会检查固定 0.6B 配置，不把形状不匹配的模型默认为兼容。

### 部署输入约定

- batch 固定为 1；采样率 16 kHz，128 mel bins，FFT 400，hop 160，使用官方 `WhisperFeatureExtractor`。
- 每 100 个 mel 帧独立经过三层卷积，产生 13 个 token；位置编码每个卷积块重新从 0 开始。
- `attention_mask` 是加到 attention logits 上的浮点 mask，允许处为 0，禁止处为 -10000。
- 默认双向 attention 窗口为 104 token，即 8 个卷积块。15 秒分成 `[104,91]`，不是全序列 attention。
- 有效 mel 长度至少 100；小于 100 帧的原生短输入不在本次兼容性范围内。不要假定先补到 100 帧与原生短输入完全等价。
- 向静态桶填充时，先计算真实音频的 mel，再把有效长度后的 **mel 数值填 0**，屏蔽无效 key。给波形补静音会得到不同的 log-mel，不能混用两种参考。
- 输出只保留前 `13*(valid_frames//100) + ceil((valid_frames%100)/8)` 个 token；padding 输出不能送入 decoder。
- `scripts/encoder.py` 的 `attention_mask()` 和 `output_length()` 是上述规则的参考实现。全 padding 窗口仍保留有限输出，避免 NaN。

## 3. 独立数值参考与 ONNX Runtime 测试

测试音频单独下载，不提交录音。固定 OpenAI Whisper fixture 及其 SHA-256：

```bash
mkdir -p fixtures
curl -fL https://raw.githubusercontent.com/openai/whisper/6e3be77e1a105e59086e3e21ff5f609fd6fa89a5/tests/jfk.flac -o fixtures/jfk.flac
echo '63a4b1e4c1dc655ac70961ffbf518acd249df237e5a0152faae9a4a836949715  fixtures/jfk.flac' | sha256sum -c -
for s in 2 4 15; do
  .venv-export/bin/python scripts/prepare_encoder_cases.py \
    --model models/Qwen3-ASR-0.6B --audio fixtures/jfk.flac \
    --seconds "$s" --output-dir "artifacts/cases/${s}s"
  for precision in fp32 fp16; do
    .venv-export/bin/python scripts/check_encoder_onnx.py \
      --onnx "artifacts/encoder-${s}s.${precision}.onnx" \
      --cases "artifacts/cases/${s}s" --precision "$precision" \
      --report "artifacts/cases/${s}s/onnx_${precision}.json"
  done
done
.venv-export/bin/python -m unittest discover -s tests -v
```

覆盖真实语音、随机输入、非整秒有效长度和 15 秒桶内仅 1 秒有效的边界情况。测试还生成官方 mel filter 的 FP32 `mel_filters.npy`，形状 `[201,128]`。

重要参考差异：qwen-asr 0.0.6 的 eager/SDPA 路径计算了 `cu_seqlens`，却没有把分窗 mask 传入 layer，所以 15 秒原生 eager 是全局 attention。窗口参考通过 **按官方 cu_seqlens 切开每个原生 layer 调用**构造，未使用导出图自己的 attention 实现。全局模式也单独与原生 eager 对比。

FP16 验证是“转换后的 ONNX 相对官方 FP32 参考”的误差测试；不是原生官方 FP16 路径的逐算子等价声明。FP32 导出时没有官方 FP16 分支专用的 residual clamp。

## 4. RKNN Toolkit2 2.5.0

独立创建 `.venv-rknn`；新版仍要求 PyTorch 不高于 2.4.0。

```bash
python3.12 -m venv .venv-rknn
.venv-rknn/bin/pip install torch==2.4.0+cpu --index-url https://download.pytorch.org/whl/cpu
.venv-rknn/bin/pip install -r requirements-rknn.txt
```

转换器检查版本为 2.5.0，报告记录 ONNX/RKNN SHA-256。本轮路径和结果见 [SDK_UPGRADE.md](SDK_UPGRADE.md)。

```bash
for s in 2 4 15; do
  OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv-rknn/bin/python scripts/convert_encoder_rknn.py \
    --onnx "artifacts/encoder-${s}s.fp16.onnx" --seconds "$s" \
    --output "artifacts/encoder-${s}s.fp16.rk3588.rknn" \
    --cases "artifacts/cases/${s}s" --report "artifacts/cases/${s}s/rknn.json" \
    > "artifacts/rknn-${s}s.log" 2>&1
done
```

`build(do_quantization=False)`，默认 optimization level 3。`--cases` 在主机 simulator 上对比独立 golden，**不是 RK3588 实机测试**。
检查 JSON 返回码的同时保留并阅读编译日志；非零级别的底层编译信息不能因 export 返回 0 而忽略。

## 5. 独立 decoder → RKLLM

```bash
.venv-export/bin/python scripts/extract_qwen3_asr_decoder.py \
  --source models/Qwen3-ASR-0.6B --output models/decoder_hf \
  --source-revision 5eb144179a02acc5e5ba31e748d22b0cf3e303b0
.venv-export/bin/python scripts/check_decoder_parity.py \
  --source models/Qwen3-ASR-0.6B --decoder models/decoder_hf \
  --report artifacts/decoder-parity.json
```

保留 decoder 的 BF16 safetensors，额外导出 FP32 `embed_tokens.npy`。等值三平面位置索引使 ASR 的 MRoPE 可改为普通 Qwen3 RoPE；不能将此结论推广到不同 T/H/W 位置的多模态模型。

先按 [RKLLM 工具链说明](RKLLM_TOOLCHAIN_AUDIT.md) 创建独立 `.venv-rkllm`；历史文档记录旧版本；本轮使用 `requirements-rkllm.txt` 的 RKLLM 1.3.1 / Transformers 5.8.0，必须与导出环境分开。

```bash
.venv-rkllm/bin/python scripts/convert_decoder_rkllm.py \
  --decoder models/decoder_hf --parity-report artifacts/decoder-parity.json \
  --quant fp16 --output artifacts/decoder.fp16.rk3588.rkllm \
  --report artifacts/decoder-fp16.json

.venv-export/bin/python scripts/prepare_decoder_calibration.py \
  --source models/Qwen3-ASR-0.6B --decoder models/decoder_hf --language English \
  --audio-embeddings artifacts/cases/2s/speech.npz \
  --audio-embeddings artifacts/cases/4s/speech.npz \
  --audio-embeddings artifacts/cases/15s/speech.npz \
  --output-dir artifacts/calibration
.venv-rkllm/bin/python scripts/convert_decoder_rkllm.py \
  --decoder models/decoder_hf --parity-report artifacts/decoder-parity.json \
  --quant w8a8 --dataset artifacts/calibration/data_quant.json \
  --load-dtype float16 --reclaim-memory-before-export \
  --output artifacts/decoder.w8a8.rk3588.rkllm --report artifacts/decoder-w8a8.json
```

转换门禁会验证 parity 报告与当前 decoder 权重、配置、tokenizer、embedding 的 SHA-256 一致，拒绝使用过期报告。
1.3.1 的校准 JSON 索引 pickle forward kwargs（`sample` / `token_nums`），不再使用旧版嵌套 `input_embed`。
W8A8 用真实 encoder 输出加官方文本 prompt 的 embedding 做 smoke calibration；同一个英文样本的三种长度不是代表性校准集，不能证明量化后 WER/CER。

## 6. 尚需板端验证

需要真实 RK3588、匹配的 RKNN runtime/driver 和 RKLLM runtime 1.3.1，验证模型加载、音频/token embedding 输入、KV cache、EOS、识别准确率及延迟。主机导出、simulator 数值通过均不能替代这些步骤。
本仓库不包含权重、录音、SDK wheel 或凭据。实际主机测试结果与已知限制见 [VALIDATION.md](VALIDATION.md)。
