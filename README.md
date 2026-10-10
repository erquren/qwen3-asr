# qwen3-asr

将官方 **Qwen3-ASR** 模型导出为 **支持 RKNN 转换的 ONNX 模型**，面向 RK3588 部署。

- 音频编码器：官方权重 → 两输入、固定形状 ONNX → RKNN
- 文本解码器：官方权重 → 独立 Hugging Face Qwen3 解码器 → RKLLM

主要测试模型为 `Qwen/Qwen3-ASR-0.6B`。编码器使用 2 秒、4 秒、15 秒静态输入桶；解码器单独转换，不能把整个 ASR 模型的 ONNX 直接当成 RKLLM 输入。

转换脚本和测试正在验证中。成功导出、数值一致性、RKNN/RKLLM 编译和实际 RK3588 板端推理会分别记录；未执行的阶段不会标记为通过。

本仓库原创代码使用 Apache License 2.0。模型权重及第三方软件遵循各自许可证。
