# 中文识别精度回归

2026-10-10 在 CPU 主机实际执行；不是只检查 tensor cosine。

## 结果

固定 FLEURS Mandarin 测试集 5 条完整录音，共 **33.84 秒、129 个归一化参考字符**。

| 路径 | 错误字符数 / 参考字符数 | CER | 相对原模型 CER 变化 |
| --- | ---: | ---: | ---: |
| 官方模型 FP32 greedy | 1 / 129 | 0.7752% | — |
| ONNX FP32 encoder + HF FP32 decoder | 1 / 129 | 0.7752% | 0 个百分点 |
| ONNX FP16 encoder + HF FP32 decoder | 1 / 129 | 0.7752% | 0 个百分点 |
| RKNN host simulator encoder + HF FP32 decoder | 1 / 129 | 0.7752% | 0 个百分点 |

四条路径的 **5/5 条原始识别文本和生成 token IDs 都相同**。唯一计入 CER 的错误是第 3 条多输出一个“转”。无失败、无截断、没有剔除错误样本。

**这些结果只说明本次 5 条短句回归未观察到 encoder 转换导致的识别变化，不能推断完整测试集精度，也不能推断 RKLLM 量化后或 RK3588 实机精度。**

## 逐条参考与输出

以下输出在四条路径上相同。下方复现命令会在本地产生逐条文本、token IDs 和 S/D/I 编辑距离；原始测试 JSON 不提交到代码仓库。

1. 参考：这里几乎都是沙滩，游泳很安全，大部分地方都有新西兰圣诞树的树荫。
   输出：这里几乎都是沙滩，游泳很安全。大部分地方都有新西兰圣诞树的树荫。
   CER：0 / 29。标点差异不计入本次 CER。
2. 参考 / 输出：当所有可用的资源在一个组织的职能部门之间得到有效利用时，创造力和独创性就会发生。
   CER：0 / 38。
3. 参考：他随后被转送到剑桥的阿登布鲁克医院。
   输出：他随后被转转送到剑桥的阿登布鲁克医院。
   CER：1 / 17，1 次插入。
4. 参考 / 输出：在中东温暖的气候中，房子并不是多重要。
   CER：0 / 17。
5. 参考 / 输出：他没有设定削减数额，只是表示将根据中国的经济产出来决定数额。
   CER：0 / 28。

## 测试协议

- 数据：Google FLEURS，`cmn_hans_cn/test`，固定 revision `168de341b3db6859a9bac1c50a2ef5e3b47647e0`。
- 按原始行序选择首次满足 1–8 秒、无 ASCII 字母/数字、句子 ID 不重复的 3 条女性及 2 条男性录音；行号 9、10、11、26、34。选择发生在推理前。
- 原始音频字节、完整 `raw_transcription` 不变；下载和抽取均核对 SHA-256。没有为匹配输出而修改参考。
- 与 W8A8 校准用的英文 JFK 音频分开。这里的“校准外测试”不表示已知 Qwen 预训练未使用 FLEURS。
- 每条 5.04–7.74 秒，全部使用 **15 秒静态桶**。这组识别精度测试不覆盖 2/4 秒桶；2/4 秒已有独立 tensor 数值测试。
- 统一官方 processor、forced Chinese prompt、FP32 decoder、greedy decoding、`do_sample=False`、`num_beams=1`、最多 256 新 token、同一 EOS 集。
- 基线是官方 FP32 `model.generate` 输出，未使用 `transcribe()` 的重复文本清理。所测输出没有触发该清理的病态长重复。
- 所有原生 mel 长度 ≤800，最多 104 个音频 token，避免官方 eager 与长音频分窗语义差异干扰比较。
- CER：Unicode NFKC + casefold，移除 Unicode 标点和空白；不改写数字、不做繁简转换。按 Unicode 字符计算 Levenshtein S/D/I，**累计错误数 / 累计参考字符数**，不平均单句百分比。
- 缺失、失败、截断样本按空文本计为全删除并保留在分母。空参考报错。

## 复现

先完成 [CONVERSION.md](CONVERSION.md) 中的模型、ONNX、SDK 和 decoder 准备。以下抽取读取 Parquet 数据，不执行远程 dataset Python 代码。需要 `pyarrow==21.0.0`；RKLLM 环境已有该包。

```bash
mkdir -p artifacts/accuracy
curl -fL https://huggingface.co/datasets/google/fleurs/resolve/168de341b3db6859a9bac1c50a2ef5e3b47647e0/cmn_hans_cn/test/0000.parquet \
  -o artifacts/accuracy/test.parquet
.venv-rkllm/bin/python scripts/extract_accuracy_fixtures.py \
  --parquet artifacts/accuracy/test.parquet \
  --output artifacts/accuracy
.venv-export/bin/python scripts/prepare_asr_inputs.py \
  --source models/Qwen3-ASR-0.6B --manifest artifacts/accuracy/manifest.json \
  --output artifacts/accuracy/inputs
.venv-export/bin/python scripts/run_asr_accuracy.py --mode official \
  --model models/Qwen3-ASR-0.6B --inputs artifacts/accuracy/inputs \
  --label official_fp32 --output artifacts/accuracy/official_fp32_predictions.json

for backend in onnx_fp32 onnx_fp16 rknn_simulator; do
  py=.venv-export/bin/python
  [ "$backend" != rknn_simulator ] || py=.venv-rknn/bin/python
  "$py" scripts/run_encoder_accuracy.py --backend "$backend" --models artifacts \
    --inputs artifacts/accuracy/inputs --output "artifacts/accuracy/embeddings/$backend"
  .venv-export/bin/python scripts/run_asr_accuracy.py --mode decoder \
    --model models/decoder_hf --inputs artifacts/accuracy/inputs \
    --embeddings "artifacts/accuracy/embeddings/$backend" --label "$backend" \
    --output "artifacts/accuracy/${backend}_predictions.json"
done
for backend in official_fp32 onnx_fp32 onnx_fp16 rknn_simulator; do
  .venv-export/bin/python scripts/asr_metrics.py --manifest artifacts/accuracy/manifest.json \
    --predictions "artifacts/accuracy/${backend}_predictions.json" \
    --output "artifacts/accuracy/${backend}_cer.json"
done
```

Parquet 约 696 MB；读出音频字典有额外 RAM 开销，应与大模型推理串行。

## 数据归属与许可

[Google FLEURS](https://huggingface.co/datasets/google/fleurs)，Conneau et al., *FLEURS: Few-shot Learning Evaluation of Universal Representations of Speech* (2022)，[论文](https://arxiv.org/abs/2205.12446)。数据及上述参考文本依 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) 使用，非仓库代码的 Apache-2.0 许可。原始文本未修改；CER 归一化方法已声明。音频不提交 Git 或模型仓库。
