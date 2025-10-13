## vLLM 后端

vLLM 的“attention 后端”就是用来计算注意力核心算子`（Q·Kᵀ→softmax→·V）`的实现插件。
不同后端用不同的 GPU 内核/编译方式来做同一件事，目标是兼顾速度、显存占用、兼容性和功能（各种 mask / 变体）。
你可以把它理解成“算子引擎”，vLLM 把调度、KV 缓存、并行等框架层工作和底层 attention 计算解耦了，因此可以换不同后端。

### 1 常见的后端
下面是常见后端及特点（推理场景）：

>FlashAttention（flash-attn）
- `IO-aware` 的高性能实现，长序列显存与速度都很强
- 但 `需要较新的` GPU（通常 `SM≥8，Ampere` 起），功能上对非常花哨的注意力变体支持没那么灵活

>Triton（triton_attn）
- 基于 `OpenAI Triton` 的内核生成，通用、兼容性好，对老显卡也友好，性能通常不错但未必达到 `FA` 的极限；适合“能稳跑就好”的大多数部署。

>FlexAttention（flex_attention）
- 可编程式 `attention`：通过可组合的 `mask/score` 修改实现各种变体（滑窗、稀疏、`prefix/paged` 等），灵活性高
- 极限性能可能略逊于手写内核
- 适合需要非标准注意力的模型/研究

>xFormers（xformers）
- `Meta` 的高效注意力集合
- 很多卡/环境下也能跑，性能介于 `Triton` 与 `FA` 之间（取决于场景），安装门槛相对低

>Torch 原生（torch）
- 最通用的兜底实现，兼容性最好、性能最弱
- 只有当其他后端不可用时才建议用

### 2 后端到底“负责”什么？

- 计算注意力核心步骤（含掩码、数值稳定 `softmax` 等）
- 支持不同精度（`fp16/bf16`）与不同 head 形状（`MHA/MQA/GQA`）
- 与 `vLLM` 的 `KV 缓存（paged attention）、可变序列长度 batch、分块预填充` 协同

在不同硬件上选择合适的 `tile/block`、减少显存与 `HBM` 访问

### 3 该怎么选？

- 新卡（A100/4090/H100 等）+ 标准因果注意力 → 先选 FlashAttention
- 老卡（如 RTX 2080Ti, SM7.5）或想要稳兼容 → 选 Triton 或 xFormers
- 需要自定义/稀疏/滑窗/更灵活的注意力 → 试 FlexAttention

### 4 在 vLLM 里怎么切换？

两种等价方式（择一）：
```sh
# 方式1：命令行
... --attention-backend TRITON_ATTN_VLLM_V1      # 或 flash_attn / flex_attention / xformers / torch

# 方式2：环境变量
export VLLM_ATTENTION_BACKEND=TRITON_ATTN_VLLM_V1
```

实战提示：像你用的 `RTX 2080 Ti（SM 7.5）` 不支持 `FlashAttention-2`，直接用 `--attention-backend triton` 最省心。

### 5 所有后端

vLLM 支持的所有 Attention 后端（Backend）列表，包括 CUDA、ROCm、Intel、实验性与特殊用途的实现。

#### 🧩 主流 CUDA 路线（NVIDIA GPU）

| 后端名                             | 说明                              | 适用硬件                    | 优势                     | 备注                     |
| ------------------------------- | ------------------------------- | ----------------------- | ---------------------- | ---------------------- |
| **FLASH_ATTN**                  | 原始 FlashAttention kernel（v1/v2） | Ampere+（A100/3090/4090） | 极高性能，显存高效              | 不支持 SM < 8.0（如 2080Ti） |
| **FLASH_ATTN_VLLM_V1**          | vLLM 自带优化版 FlashAttention       | Ampere+                 | 稳定、兼容 vLLM 内部 KV cache | 通常默认启用                 |
| **TRITON_ATTN_VLLM_V1**         | 基于 Triton 的自研 Attention         | 所有 CUDA GPU（含 2080Ti）   | 通用、兼容性好、安装简单           | 稍慢于 FlashAttention     |
| **XFORMERS / XFORMERS_VLLM_V1** | Meta 的 fused attention kernel   | 所有 CUDA GPU             | 简单、稳定、速度适中             | 需额外安装 `xformers`       |

👉 对你（2080Ti）：用 TRITON_ATTN_VLLM_V1 或 XFORMERS_VLLM_V1 最合适

#### 🧠 PyTorch 原生 / 通用接口

| 后端名                    | 说明                                             | 优势                     | 缺点                      |
| ---------------------- | ---------------------------------------------- | ---------------------- | ----------------------- |
| **TORCH_SDPA**         | PyTorch 2.x 的 `scaled_dot_product_attention()` | 官方支持，兼容性最高             | 性能一般                    |
| **TORCH_SDPA_VLLM_V1** | vLLM 调优版 SDPA                                  | 通用性强                   | 仍偏慢，用于兜底                |
| **FLEX_ATTENTION**     | PyTorch FlexAttention (2024 新特性)               | 可编程、灵活，可组合各种 mask/bias | 仍在实验期，性能略逊 Flash/Triton |
| **NO_ATTENTION**       | 禁用注意力（仅测试）                                     | 调试用                    | 实际无用                    |
| **TREE_ATTN**          | 树形注意力（实验）                                      | 用于层级结构模型               | 目前极少使用                  |

#### 🧮 MLA 系列（Multi-Head Latent Attention）

| 后端名                                                  | 说明                        | 场景                           |
| ---------------------------------------------------- | ------------------------- | ---------------------------- |
| **TRITON_MLA / TRITON_MLA_VLLM_V1**                  | Triton 实现的 MLA（混合局部注意力）   | 多模态、图像/视频 LLM                |
| **FLASHMLA / FLASHMLA_VLLM_V1**                      | FlashAttention + MLA 组合   | 高端 GPU 的多模态模型                |
| **CUTLASS_MLA**                                      | NVIDIA CUTLASS 内核实现       | 精度高，兼容 CUDA Toolkit          |
| **FLASH_ATTN_MLA**                                   | FlashAttention + MLA 混合实现 | 性能优，实验性                      |
| **FLASHINFER / FLASHINFER_VLLM_V1 / FLASHINFER_MLA** | 来自 FlashInfer 库（新兴高效推理库）  | 面向下一代高吞吐推理（比如 FlashDecoding） |

👉 简单理解：这些是为多模态 / 视频 / MLLM / 长上下文模型准备的变体，不适合普通文本 LLM

#### 🔥 ROCm 路线（AMD GPU）

| 后端名                                                         | 说明                        | 平台   | 备注            |
| ----------------------------------------------------------- | ------------------------- | ---- | ------------- |
| **ROCM_FLASH**                                              | AMD GPU 上的 FlashAttention | ROCm | 对应 CUDA 版的 FA |
| **ROCM_AITER_FA / ROCM_AITER_MLA / ROCM_AITER_MLA_VLLM_V1** | AMD 官方/社区提供的 Attention 实现 | ROCm | 与 CUDA 版功能对齐  |

👉 只有在使用 AMD 显卡 + ROCm 环境 时才会触发。

#### 🧩 Intel / 特殊架构支持
| 后端名                         | 说明                                      | 平台                   | 备注            |
| --------------------------- | --------------------------------------- | -------------------- | ------------- |
| **IPEX**                    | Intel Extension for PyTorch (CPU / GPU) | Intel GPU / Xeon CPU | 针对 Intel 硬件优化 |
| **PALLAS / PALLAS_VLLM_V1** | XLA / JAX 编译器后端                         | TPU / XLA            | 实验性           |
| **DUAL_CHUNK_FLASH_ATTN**   | 双块 FlashAttention（chunked decoding）     | 高性能 streaming decode |               |
| **DIFFERENTIAL_FLASH_ATTN** | 可微分 FlashAttention（用于训练或对比实验）           | 研究用途                 |               |



#### 🧭 总结：选哪个后端？

| 设备类型                         | 推荐后端                                      | 说明            |
| ---------------------------- | ----------------------------------------- | ------------- |
| RTX 4090 / A100 / H100       | `FLASH_ATTN_VLLM_V1`                      | 最快            |
| RTX 2080Ti / 3080（不支持 SM8.0） | `TRITON_ATTN_VLLM_V1`                     | 稳定兼容          |
| AMD GPU                      | `ROCM_FLASH` / `ROCM_AITER_MLA_VLLM_V1`   | ROCm 环境       |
| Intel GPU / CPU              | `IPEX`                                    | Intel 平台      |
| 多模态（Qwen-VL、MiniCPM-V）       | `TRITON_MLA_VLLM_V1` / `FLASHMLA_VLLM_V1` | 专用            |
| 想尝鲜新特性                       | `FLEX_ATTENTION`                          | 可编程 attention |
| 调试 / 最通用                     | `TORCH_SDPA_VLLM_V1`                      | 最安全兜底方案       |
