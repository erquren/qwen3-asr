# 转换测试报告

测试日期：2026-10-10。Linux x86_64，约 9.7 GiB RAM，无 CUDA、无 RK3588 开发板。重型步骤串行执行。

## 完成范围

| 阶段 | 结果 |
| --- | --- |
| 官方固定 safetensors 下载与逐文件 SHA-256 | 通过 |
| 2/4/15 秒 encoder FP32 ONNX | 导出、checker、CPU ORT 对比通过 |
| 2/4/15 秒 encoder FP16 ONNX | 转换、checker、CPU ORT 对比通过 |
| 2/4/15 秒 RK3588 RKNN | build/export 返回 0，host simulator 对比通过 |
| 官方 decoder → 标准 Qwen3 HF | 311 tensor 完整提取，FP32 embedding 每行检查通过 |
| 原始与独立 decoder FP32 | 16 项 hidden/logits/cache/RoPE 对比最大误差均为 0 |
| RK3588 RKLLM FP16 | SDK 1.2.3 load/build/export 通过，1,529,743,212 bytes |
| RK3588 RKLLM W8A8 | 同一 3 条音频 embedding 校准，最终 load/build/export 通过，936,054,076 bytes |
| 中文识别精度 | 4 路径各 5 条；CER 均 1/129 = 0.7752%，输出一致 |
| RK3588 实机加载/推理/延迟 | 未执行 |
| RKLLM FP16 / W8A8 识别准确率 | 未执行 |

详细识别结论、参考文本和 CER 定义见 [ASR_ACCURACY.md](ASR_ACCURACY.md)。

## Encoder 数值结果

下面是每组所有测试案例中的最差值；FP32、FP16 和 RKNN 都对比独立官方层 FP32 参考。2/4 秒各 3 组，15 秒 5 组。

| 桶 / 后端 | 最小 cosine | 最大 relative L2 | 最大绝对误差 |
| --- | ---: | ---: | ---: |
| 2 s ONNX FP32 | 0.999999999971 | 0.000007719 | 0.000002194 |
| 2 s ONNX FP16 | 0.999999708014 | 0.000825535 | 0.000119478 |
| 2 s RKNN simulator | 0.999996116698 | 0.002802365 | 0.000457452 |
| 4 s ONNX FP32 | 0.999999999945 | 0.000010527 | 0.000004508 |
| 4 s ONNX FP16 | 0.999999596150 | 0.000903905 | 0.000338761 |
| 4 s RKNN simulator | 0.999985445396 | 0.005397447 | 0.001721904 |
| 15 s ONNX FP32 | 0.999999999950 | 0.000010042 | 0.000004627 |
| 15 s ONNX FP16 | 0.999998819228 | 0.001539182 | 0.001068419 |
| 15 s RKNN simulator | 0.999986373149 | 0.005221194 | 0.002614174 |

验收阈值在脚本中提前给出：ONNX FP32 relative L2 <1e-4 / max abs <0.003；ONNX FP16 <0.01 / <0.2；RKNN simulator <0.03 / <0.5。不是单凭 cosine 判定。完整输出含 padding 均检查有限值，比较有效前缀时保留正确 token 长度。

覆盖：公开英文语音、固定随机输入、137/317/1437 帧尾部补齐、15 秒桶仅 100 有效帧、15 秒全局与分窗 mask。主机 ORT/模拟器耗时不是 NPU 性能数据。

## 必须保留的限制

1. **15 秒编译日志**：RKNN 2.3.2 出现一次 `E RKNN: Unkown op target: 0`。build/export/init-simulator 均返回 0，5 组模拟数值通过，但不能因此认定该诊断无害或板端一定可运行。详细模式再次复现。厂商仓库也有[同类报告](https://github.com/airockchip/rknn-toolkit2/issues/408)，本次没有获得可确认根因的板端证据。
2. **CPU 回退**：详细图中 GELU 展开的 22 个 Erf 算子位于 CPU。生成的是 RKNN 可编译的混合执行图，不能宣称全图纯 NPU 或给出未经测量的 NPU 加速比。
3. **官方长音频参考差异**：qwen-asr 0.0.6 eager 未应用它计算的分窗边界。15 秒全局与分窗输出 cosine 约 0.9720、relative L2 约 0.2355，不能混当一个参考。分窗 golden 用原生 layer 按原生 cu_seqlens 分段执行；未声称实测 CUDA FlashAttention。中文识别输入全部 ≤800 mel 帧，规避了这一差异。
4. **FP16 语义**：验证的是 ONNX FP16 相对官方 FP32 的转换误差。FP32 tracing 未包含官方原生 FP16 专用 residual clamp，因此未声称原生官方 FP16 逐算子等价。
5. **W8A8 资源重试**：前两次在完成 28 层优化后，export 阶段被 SIGKILL，留下 0-byte 文件，均记录为失败。第二次峰值子进程 RSS 约 5.34 GiB；因无 OOM 计数器，内存压力是推断。释放闲置 RAM 盘占用后，使用相同 3 样本、CPU FP16 加载、标准 GC/malloc_trim 回收，第三次成功，峰值约 5.39 GiB。没有修改 SDK 私有实现。
6. **校准覆盖有限**：W8A8 输入是同一英文语音的 2/4/15 秒 encoder 输出加官方 prompt，分别为 44/70/213 token，保留真实 acoustic embedding；没有编造参考 transcript。它只是校准链 smoke test，不足以证明量化后的识别质量。
7. **识别样本有限**：5 条中文共 129 字，只覆盖 15 秒桶的短语音 padding。不能将观察到的 0 CER 增量推广为全任务无精度损失。

## 复现

官方固定 revision 为 `5eb144179a02acc5e5ba31e748d22b0cf3e303b0`。下载器逐文件生成 SHA-256；转换、parity 和 CER 脚本会在指定输出目录产生本地机器可读结果。原始测试日志和 JSON 不提交到代码仓库。

库版本分开固定：导出 Transformers 4.57.6 / Torch 2.6.0 CPU；RKNN Toolkit2 2.3.2；RKLLM Toolkit 1.2.3 / Transformers 4.55.2。仓库内轻量测试还覆盖长度/mask、窗口隔离、tensor 名称和字节、非法/过期门禁、缺失与截断样本计分、官方 generate 包装器参数约束。
