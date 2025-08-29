## Work Flow

```sh
vllm serve /home/azen/model/qwen2.5-7B-instruct \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.95 \
  --dtype float16
```

### 一、启动阶段（Server bring-up）

##### 1 解析参数 & 初始化运行时

- `--dtype float16`：按 `FP16` 加载/推理（权重、激活多为 `FP16`；少量归一化/累加仍可用 `FP32`）
- `--max-model-len 8192`：把可服务的上下文长度上限设为 8192（不等于模型原生上限，但调度器会以此为界做校验/切分）
- `--gpu-memory-utilization 0.95`：规划 `KV Cache` 的“分页池”大小（`PagedAttention` 的 `page 池`），目标是把可用显存的 `~95%` 留给权重之外的缓存与中间张量（剩余 `~5%` 给 `CUDA 运行时`、`通信缓冲`等）

##### 2 加载权重 & Tokenizer

- 从 `/home/azen/model/qwen2.5-7B-instruct` 读取模型权重与 tokenizer（通常兼容 HF 目录结构）
- 如果是多卡环境，还会根据拓扑选择 `张量并行/流水并行`（这里假设单卡）

##### 3 构建 PagedAttention 的内存池

- 依据 gpu_memory_utilization 和 max-model-len 估算最大并发 × 序列长度能占用的 KV 大小，预先分页化（按固定大小 page）建立空闲页链表与 page table
- 这一步决定了后续能否在高并发下稳定分配/回收 KV（减少碎片）

##### 4 编译/加载高性能内核
- 准备 Triton/CUDA 内核（例如 PagedAttention、RMSNorm/LayerNorm、Softmax、采样头等）。

- 可能进行 warmup（例如用空 batch 走一遍图），提高后续稳定吞吐。

##### 5 启动服务端

- 起一个 OpenAI 兼容的 `HTTP API（FastAPI/Uvicorn）`，常见端点：`/v1/completions`、`/v1/chat/completions`、`/v1/models` 等
- 准备 `请求队列`、`调度器（scheduler）`、`采样器（sampler）` 与 `生成循环（generation loop）`

### 二、单次请求的生命周期（Single request）

以一次 chat.completions 为例（Python 客户端示意）：
```py
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")

resp = client.chat.completions.create(
    model="qwen2.5-7B-instruct",
    messages=[{"role": "user", "content": "你好，解释一下分页注意力。"}],
    temperature=0.7, top_p=0.9, max_tokens=512, stream=True
)
for chunk in resp:
    print(chunk.choices[0].delta.content or "", end="")
```

##### 1 接收与校验

- `Server` 收到请求 → 校验 max_tokens + prompt_len ≤ max_model_len(=8192)，并解析解码参数（`temperature`、`top_p`、`stop` 等）

- 记录需要 logprobs / stream / 并行采样 的附加开销设置

##### 2 入队 & 调度（prefill/decoding 两阶段）

- 调度器把请求放入等待队列，按时序与策略做 `动态批处理（dynamic batching）`
- **Prefill 阶段**：模型读入完整 `prompt`，构建 `KV Cache`
- **Decoding 阶段**：每步只追加 `1` 个新 `token`（或 speculative/多 token），复用已缓存 KV 进行增量计算

##### 3 PagedAttention 的 KV 分页分配

- 为该请求从空闲池按页拿 KV 空间（page 分配），把请求的 token 段映射到 `page table`
- 这样即使多个请求的长度/进度不同，也能高效拼成一个批次而不挤占连续大块显存

##### 4 内核执行（Triton 核心加速）

在每个迭代步：

- 执行 `注意力算子（PagedAttention Triton kernel）`：根据查询 token 的位置索引对应的 KV pages，避免大规模拷贝/拼接
- 线性层、Norm、激活等核心算子（多为 Triton/CUDA 内核）
- 采样头（logits → 采样/贪心/温控、top-p/top-k）

如果 `stream=True`，边算边把新 `token` 增量推流给客户端

##### 5 停止条件与回收

- 命中 `stop_words / 生成到 max_tokens / EOS token` → 请求完成
- 调度器释放该请求占用的 `KV pages`（回收到空闲池），更新统计/日志

### 三、多并发下的调度细节（Many requests）

##### 1 动态批处理

- 同步/异步抵达的请求被放进同一批（或多个批）里跑
- Prefill/Decoding 分时交替：短请求不会被长请求“绑架”，调度器会把不同阶段的请求合理混批

##### 2 页式 KV 复用

- 某个请求结束后，它的 pages 立刻回到空闲池，新请求可即时复用，显著降低碎片与等待
- “分页”使得不同长度、不同进度的序列能紧密装箱进同一批的计算里

##### 3 吞吐与延迟的平衡

- 大批次提高 `吞吐`，但会拉长 `首 token 延迟`
- vLLM 的策略通常在这两者间折中（可通过并行度、并发上限、分片策略等服务端参数调优）

### 四、与参数的关系与影响

- `--max-model-len 8192`
    - 上限闸门：任何请求的 prompt_len + max_tokens 都不得超 8192
    - 调度器据此估算一次请求所需 KV 页数，防止过量分配
    - 如果模型原生位置编码允许更长（需 Rope scaling 等），仍以该值作为服务约束

- `--gpu-memory-utilization 0.95`

    - 直接影响 KV 池规模，越大 → 支持更长的上下文或更多并发
    - 过高会逼近 OOM 边缘（CUDA 运行、通信、临时 buffer 都要显存）
    - 一般建议在 0.90~0.95 之间迭代观察吞吐/OOM 率

- `--dtype float16`

    - 降低权重与激活占用，释放更多显存给 KV
    - 与 bfloat16 相比，FP16 在数值稳定性上略逊，但显存略省（具体看硬件/内核实现）
    - 若显存足够、关心稳定性，可尝试 --dtype bfloat16

### 五、端到端“时序图”（文字版）

- **启动**：参数解析 → 加载权重/Tokenizer → 预建 KV 分页池 → 预热内核 → 起 HTTP 服务
- **请求到达**：校验长度与采样参数 → 入队
批处理：调度器把同窗期的请求组批
- **Prefill**：分配 KV pages → Triton PagedAttention 计算 → KV 写入 page table
- **Decoding（循环）**：读 KV pages → Triton 计算 → 采样 →（可流式返回）
- **完成**：命中停止条件 → 回收 pages → 统计/日志

### 六、常见实战提示
- 流式返回：前端体验更好，也减轻服务端积压
- 长提示词：若以长 prompt 为主，可考虑关闭 chunked prefill（或调小 chunk）来减少切块带来的对齐开销
- 并发压测：逐步升高并发，观察 首 token 延迟、吞吐 (tokens/s)、OOM 三项指标，微调 gpu_memory_utilization 与批处理策略
- 停用词/截断：合理设置 stop/max_tokens，避免无谓的长尾生成占用 KV