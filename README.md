# qwen3-asr

将官方 **Qwen3-ASR** 模型导出为 **支持 RKNN 转换的 ONNX 模型**，面向 RK3588 部署。

- 音频编码器：官方权重 → 两输入、固定形状 ONNX → RKNN
- 文本解码器：官方权重 → 独立 Hugging Face Qwen3 解码器 → RKLLM

主要测试模型为 `Qwen/Qwen3-ASR-0.6B`。编码器使用 2 秒、4 秒、15 秒静态输入桶；解码器单独转换，不能把整个 ASR 模型的 ONNX 直接当成 RKLLM 输入。

## 本轮 SDK 适配

RKNN 2.5.0 / RKLLM 1.3.1 的转换与验证见 [SDK_UPGRADE.md](docs/SDK_UPGRADE.md)。以下结果是旧 SDK 的历史记录。

## 历史实测结果

- 2/4/15 秒 ONNX FP32、FP16：导出、ONNX 检查及数值对比通过
- 2/4/15 秒 RKNN：RKNN Toolkit2 2.3.2 编译、主机 simulator 数值对比通过
- 独立 decoder：16 项 FP32 对比逐位一致；RKLLM 1.2.3 的 FP16、W8A8 编译通过
- 中文识别：5 条 FLEURS 测试录音、129 个参考字符。官方 FP32、ONNX FP32/FP16、RKNN 模拟 encoder 接同一 HF decoder 的输出完全一致，CER 均 **0.7752%（1/129）**

这是小样本主机回归，**尚未测试 RK3588 板端推理或 RKLLM 量化后的识别精度**。15 秒 RKNN 编译仍有一次 `Unkown op target: 0` 日志；图中 GELU 的 Erf 有 CPU 回退，不是纯 NPU 执行。

## 使用与报告

- [安装、转换及输入约定](docs/CONVERSION.md)
- [中文精度测试、逐条参考文本与复现](docs/ASR_ACCURACY.md)
- [完整测试范围、数值结果与已知限制](docs/VALIDATION.md)
- [RKLLM 工具链、真实音频 embedding 校准](docs/RKLLM_TOOLCHAIN_AUDIT.md)

模型权重、SDK wheel、录音和原始测试日志不提交到本代码仓库。依赖分为导出、RKNN、RKLLM 三个环境，避免 Transformers 版本冲突。

本仓库原创代码使用 Apache License 2.0。模型权重及第三方软件遵循各自许可证。
