## vLLM 入门与实践
- 官方 User Guide: https://docs.vllm.ai/en/stable/usage/index.html

>📦 安装 vLLM（需要 CUDA 环境）
```
pip install vllm
```


---
### 🧩 第一阶段：基础认知 — 搭建与运行

>目标：理解 vLLM 是什么、能干嘛、怎么跑起来。

##### 1 vLLM 基本概念
- vLLM 的定位：高效的 LLM 推理框架（重点是 PagedAttention + 高吞吐 KV cache 管理）
- 与 HuggingFace Transformers、DeepSpeed、Triton 的关系
- 核心模块：Engine, Worker, Scheduler, PagedAttention

##### 2 安装与运行
- 环境要求：CUDA、PyTorch 版本匹配、驱动版本
- 安装方式：`pip install vllm` / 从源码构建
- 基本命令：
    ```sh
    vllm serve model_name
    vllm generate --prompt "Hello world"
    ```
- 认识常用参数：--tensor-parallel-size, --gpu-memory-utilization, --max-model-len, --attention-backend

#####  3 基本调试与部署

- 查看 `GPU` 占用与吞吐
- 理解日志信息（如你上次贴的那种启动日志）
- `Web API` 调用（OpenAI 兼容接口）

---
### ⚙️ 第二阶段：核心机制与架构

>目标：理解 `vLLM` 为什么快、怎么实现高吞吐、与普通推理框架的不同。

##### 1 内核机制：PagedAttention

- `KV Cache` 的作用
- 传统实现的瓶颈：`O(N²) memory` / 复制
- `vLLM` 的分页思想：`Page Manager` / `Block Table` / `Virtualized KV Cache`
- `Memory Pool` 与 `Page Table` 的映射机制
- 动态批处理（Continuous batching）

##### 2 Scheduler 与任务并发

- 请求队列与 `Batch` 合并策略
- `Prefill` / `Decode` 分阶段调度
- 请求抢占与动态批次重组
- `Streaming` 输出机制

##### 3 Attention Backend（后端实现）

- 各后端原理与比较

    - FlashAttention（高性能）
    - Triton（通用）
    - FlexAttention（可扩展）
    - xFormers / Torch（兼容兜底）
- 后端选择策略、环境变量配置

---

### 🚀 第三阶段：性能优化与部署技巧

>目标：学会让 vLLM 跑得更快、更省显存、更稳定。

##### 1 性能调优

- `--gpu-memory-utilization` 影响
- 批量大小与上下文长度的平衡
- 张量并行（Tensor Parallel）
- 多 `GPU` 启动模式
- 混合精度（`FP16 / BF16 / INT8`）

##### 2 模型兼容性与量化

- 支持的模型架构：`Llama, Mistral, Qwen, Phi, Gemma` 等
- 量化支持：`AWQ, GPTQ, FP8` 等（`vllm.quantization`）
- `vLLM + LoRA`（Adapter 支持）

##### 3 服务化与集群

- `RESTful / OpenAI API` 部署
- `vLLM + FastAPI / Gradio / LangChain`
- 分布式部署（`Ray / Kubernetes / Triton Server`）
- 高并发 / 多租户策略

---
### 🧠 第四阶段：深入理解与扩展

目标：能够修改、扩展、甚至贡献 vLLM。

##### 1 代码架构阅读

- `vllm/core/` 核心代码结构
- `vllm/engine/`：Engine + Scheduler
- `vllm/attention/`：PagedAttention backend
- `vllm/model_executor/`：模型包装与执行
- `vllm/worker/`：多进程与 GPU 调度

##### 2 插件与自定义

- 自定义 `logits processor / sampler`
- 自定义模型加载（非 `HuggingFace` 模型）
- 添加新的 `Attention backend`（如 `FlexAttention` 实验）
- 修改 `KV` 管理策略（`block` 大小、分页策略）

##### 3 深入阅读材料

- 📘 官方文档：https://docs.vllm.ai
- 📄 论文：
    - “vLLM: Easy, Fast, and Cheap LLM Serving with PagedAttention” (arXiv 2023)
    - “FlashAttention” / “FlexAttention” papers

- 📂 源码学习路线：从 engine.py → scheduler.py → attention.py
- 🧩 GitHub 项目：https://github.com/vllm-project/vllm
- 常看 Issues、PR、Discussions 了解演进方向