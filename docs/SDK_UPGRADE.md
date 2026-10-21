# RKNN 2.5.0 / RKLLM 1.3.1 适配

本轮使用 Qwen/Qwen3-ASR-0.6B，固定 revision
`5eb144179a02acc5e5ba31e748d22b0cf3e303b0`。权重预期 SHA-256：
`79d6cbd4c98c7bbffe9db2edac07f56cd6637d0d5944b27f6c2b8353840323ea`。

工作目录：`/workspace/qwen3-asr-0.6b-rk3588-test`。
本轮机器为 x86_64 CPU 主机；RK3588 测试等待用户提供连接方式。

## 环境隔离

| 用途 | Python 环境 | 关键依赖 |
| --- | --- | --- |
| 官方参考、ONNX、HF decoder | `qwen3-asr/.venv-export` | torch 2.6.0+cpu、transformers 4.57.6、qwen-asr 0.0.6 |
| RKNN 编译、host simulator | `qwen3-asr/.venv-rknn` | rknn-toolkit2 2.5.0、torch 2.4.0+cpu |
| RKLLM 编译 | `qwen3-asr-0.6b-rk3588-test/.venv` | rkllm-toolkit 1.3.1、torch 2.6.0+cpu、transformers 5.8.0 |

RKNN 的 PyTorch 上限仍为 2.4.0；RKLLM 1.3.1 依赖 Transformers 5.8.0。
不可把三个环境合并。完整安装快照保存在工作目录 `logs/requirements-*.freeze.txt`；
RKLLM 环境另有 `pyproject.toml` 和 `uv.lock`。

## 复现

先下载并校验官方模型和测试录音，再运行：

```bash
python /workspace/qwen3-asr-0.6b-rk3588-test/run_conversion.py
python /workspace/qwen3-asr-0.6b-rk3588-test/package_models.py
```

工作流串行处理大模型步骤，保留逐阶段日志和完成标记；中断后可续跑。
转换器拒绝覆盖已有输出。若某一步转换失败且已写入模型，需要先检查日志、
移走失败产物，再重新执行。已有完成标记只用于本轮续跑，变更输入后应使用新目录。

## RKLLM 校准格式变化

1.3.1 使用 `[{"sample":"sample_000.pkl","token_nums":N}, ...]` 索引文件，
样本保存 standalone decoder 的 `inputs_embeds`、`attention_mask`、`position_ids`、
`use_cache` forward kwargs；结构依照官方 `make_input_embeds_for_quantize.py`。
旧版 `input_embed` 嵌套数组 JSON 会被新 SDK 拒绝。校准 embedding 使用与该 checkpoint forward 相匹配的 BF16；可检查的 NPY 仍保留 FP32。
脚本记录每份样本的 dtype 和哈希。
序列化输入由本项目生成。

## 检查范围

- 三个静态桶：2/4/15 秒；ONNX FP32、ONNX FP16、RKNN FP16。
- 真实语音、随机输入、部分有效长度、15 秒内短音频及全局 attention 对照。
- decoder 抽取权重和 FP32 embedding 完整性、独立 decoder 数值对照。
- FP16 和 W8A8 decoder，RK3588、3 NPU 核、4096 context。
- W8A8 使用独立 JFK 英文音频的 2/4/15 秒 acoustic embedding 做 smoke calibration。
  三个截取长度不能视为代表性校准集。
- 中文识别回归：固定 FLEURS 5 条、129 个归一化参考字符；官方 FP32 与
  ONNX/RKNN encoder 接同一 HF FP32 decoder 比较，逐条保留 token IDs 和 CER。
- `audio_devel` 的 RKLLM callback/参数 ABI 适配 1.3.1，官方新头文件 aarch64 编译；
  4 种调度模式 API 测试和 35 项 Python 测试。

RKNN host simulator 通过构建中的模型执行；不是在 x86_64 上加载导出的 RKNN 二进制。
报告记录同一构建导出的二进制 SHA-256。识别回归使用 HF decoder，
不能据此声称 RKLLM W8A8 或 RK3588 板端识别精度已经通过。

## 官方来源

- [RKNN Toolkit2](https://github.com/airockchip/rknn-toolkit2)，本轮头文件取自 commit
  `2c723177a94fac65cbc036d722907ddbc91980cf`。
- [RKLLM](https://github.com/airockchip/rknn-llm)，本机 wheel 1.3.1，运行库及头文件必须配套。
- [Qwen3-ASR](https://huggingface.co/Qwen/Qwen3-ASR-0.6B)。
- [Google FLEURS](https://huggingface.co/datasets/google/fleurs)，CC BY 4.0；测试音频独立于校准音频。

## 本轮结果（2026-10-20）

- 三个 RKNN 编码器：11 个独立 simulator 数值用例全部通过，最低 cosine 0.999988796，最大 relative L2 0.004734。
- decoder：16 项 FP32 对照完全一致；RKLLM 1.3.1 FP16（1460.2 MiB）和 W8A8（894.0 MiB）均构建成功。
- 官方、ONNX FP32、ONNX FP16、RKNN simulator 四条识别路径：5/5 文本和 token IDs 一致，CER 均为 1/129 = 0.7752%。
- 35 项 Python 测试、4 项 C++ 调度/API 测试通过；新 SDK 头文件 aarch64 backend 编译通过。
- 15 秒 RKNN 底层仍打印 `Unkown op target: 0`，build/export 返回 0，所有 simulator 对照通过；optimization level 2 复测日志仍在。
  该诊断不能被成功返回码掩盖，需要随板端测试继续核查。

完整逐样本报告：`/workspace/qwen3-asr-0.6b-rk3588-test/RESULTS.md` 和 `RESULTS.json`。
部署模型：`/workspace/qwen3-asr-0.6b-rk3588-test/deploy/models`，7 个文件均通过 SHA-256 校验。
