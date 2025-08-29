## vLLM 入门与实践
- 官方 User Guide: https://docs.vllm.ai/en/stable/usage/index.html

>📦 安装 vLLM（需要 CUDA 环境）
```
pip install vllm
```
---

### 1 什么是 vLLM？
vLLM 是一个高性能推理引擎，专门为 大语言模型 (LLM) 优化，目标是解决传统推理框架在 吞吐量、延迟、显存效率 上的不足。它的特点包括：

- PagedAttention：提出了分页注意力机制，可以灵活管理 KV Cache，支持高并发请求并减少显存碎片
- 高吞吐量并发：支持批处理和动态调度，大幅提升多用户请求场景下的吞吐量
- 多后端支持：既能基于 PyTorch，也能利用 Triton kernel 来加速低层算子

### 2 核心组件
##### PagedAttention

- 核心创新点，区别于传统 Attention
- 传统实现中，KV Cache 会连续分配，导致显存浪费和碎片化
- vLLM 引入类似 `虚拟内存分页` 的思想：
    - 将 KV Cache 分为固定大小的 page
    - 每个请求的缓存按需分配 page，并能复用和回收

>好处
- 显存利用率更高
- 动态批处理时，可以高效插入/移除请求

##### Triton Kernel

- vLLM 在底层算子上大量使用 Triton 编写的高性能自定义 kernel
- Triton 是一个用于 GPU 编程的语言（由 OpenAI 开发），能在 Python 中写高效 kernel
- 在 vLLM 中，Triton kernel 被用来实现：
    - 高效的 Attention 算法（包括 PagedAttention）
    - LayerNorm、softmax 等关键算子
- 相比仅用 PyTorch, Triton 内核更贴近 CUDA 性能, 同时开发效率更高

##### 调度器 (Scheduler)

- 负责批处理请求和分配 GPU 资源
- 特点：
    - 动态批处理 (dynamic batching)：能在同一时间批量处理不同长度/进度的请求
    - 公平性与吞吐量平衡：避免长序列请求“拖慢”短请求
    - 与 PagedAttention 配合，支持高效并发

##### 内存管理

- 除了 `PagedAttention` 管理 `KV Cache`，`vLLM` 还在权重加载和中间张量存储上做了优化
- 支持 `张量并行 (tensor parallelism) ` 和 `流水线并行 (pipeline parallelism)`
- 可以高效利用多 GPU

##### 执行引擎 (Execution Engine)

- 类似于推理的“核心调度循环”：
    - 接收用户请求
    - 调用调度器，组织 batch
    - 调用 Triton 内核完成计算
    - 将输出返回用户
- 与 Hugging Face Transformers API 兼容，可以快速接入




### 3 工作流程

##### 1 请求进入

- 用户通过 API 发送生成请求（例如多用户同时输入不同 prompt）

##### 2 调度器处理

- 调度器将请求分配到批次
- 如果有请求正在进行，会插入到动态批次中

##### 3 PagedAttention 分配 KV Cache

- 为新请求分配 page
- 管理正在运行的请求缓存，避免浪费

##### 4 执行内核

- 通过 Triton 内核计算注意力和其它必要算子
- 高效并行执行 batch

##### 5 生成输出

- 将本步生成的 token 返回
- 如果请求未完成，保持缓存等待下次调度

##### 6 结束与回收

- 请求完成后，释放对应的 page
- 资源进入空闲池，供后续请求使用
