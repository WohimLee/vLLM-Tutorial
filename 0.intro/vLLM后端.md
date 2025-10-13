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
... --attention-backend triton      # 或 flash_attn / flex_attention / xformers / torch

# 方式2：环境变量
export VLLM_ATTENTION_BACKEND=triton
```

实战提示：像你用的 `RTX 2080 Ti（SM 7.5）` 不支持 `FlashAttention-2`，直接用 `--attention-backend triton` 最省心。