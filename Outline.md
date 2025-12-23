## Outline

### 一、课程目标与受众定位

#### 🎯 教学目标

学完后，学员能够：

1. 理解 **LLM 推理与部署的核心瓶颈**
2. 理解 **vLLM 的设计思想与核心机制**
3. 使用 vLLM 完成：

   * 本地 / 单机 GPU 部署
   * OpenAI-compatible API 服务
   * 多卡并行（TP / DP）
   * 离线批量推理
4. 掌握 **生产级部署** 的关键点（性能、稳定性、监控、扩展）

#### 👥 目标受众

* 算法工程师 / 平台工程师
* LLM 推理优化 & 部署方向
* 已会 PyTorch / Transformers，但不熟推理系统

---

### 二、整体课程结构（推荐 12～15 讲）


#### 第 0 章：导论（Why vLLM）

##### 0.1 LLM 部署的真实痛点

* 为什么 **训练≠部署**
* 推理阶段的三大瓶颈

  * 显存
  * 吞吐
  * 延迟
* 为什么 naive Transformers serving 会崩

##### 0.2 vLLM 解决了什么问题

* vLLM vs HuggingFace pipeline
* vLLM vs TensorRT-LLM / Triton
* 适用 & 不适用场景



#### 第 1 章：LLM 推理基础（必讲原理）

##### 1.1 从一次 forward 看推理过程

* Prefill vs Decode
* KV Cache 的本质
* 为什么 Decode 是瓶颈

##### 1.2 Batch 为什么在 LLM 中很难

* 动态序列长度
* 不同请求的 decode 不同步

##### 1.3 推理性能指标

* QPS / Throughput
* TTFT（Time to First Token）
* Latency 分布（P50 / P99）



#### 第 2 章：vLLM 架构总览

##### 2.1 vLLM 核心设计

* **PagedAttention**
* Continuous Batching
* Block-based KV Cache

##### 2.2 vLLM 整体架构图

* Scheduler
* Worker
* Engine
* Tokenizer

##### 2.3 vLLM 与 Transformers 的关系

* 模型权重如何加载
* 支持的模型类型



#### 第 3 章：vLLM 环境与快速上手

##### 3.1 安装与依赖

* CUDA / 驱动 / PyTorch
* vLLM 安装方式（pip / source）

##### 3.2 第一个 vLLM 推理示例

* Python API
* 单 prompt
* 多 prompt

##### 3.3 常见踩坑

* 显存不够
* tokenizer 不匹配
* FlashAttention 相关问题



#### 第 4 章：vLLM 推理工作流（重点）

##### 4.1 vLLM 的完整请求生命周期

* 请求进入
* Scheduler 排队
* Prefill
* Decode
* 回收 KV Block

##### 4.2 Continuous Batching 详解

* 与 static batching 对比
* 为什么能提升吞吐
* 对延迟的影响



#### 第 5 章：Offline Batched Inference（离线推理）

##### 5.1 适用场景

* 数据生成
* 批量评测
* RAG 文档 embedding / 生成

##### 5.2 离线推理代码实战

* Dataset → prompts
* 批量生成
* 控制 batch / max tokens

##### 5.3 性能调优

* batch size
* max_model_len
* GPU 利用率分析



#### 第 6 章：OpenAI-Compatible Server（核心实战）

##### 6.1 为什么要 OpenAI 接口

* 生态兼容
* 前后端解耦

##### 6.2 启动 vLLM Server

* 启动参数详解
* 模型加载流程

##### 6.3 API 使用

* /v1/chat/completions
* /v1/completions
* 流式输出（streaming）



#### 第 7 章：vLLM 配置与参数调优（非常关键）

##### 7.1 核心参数详解

* max_model_len
* gpu_memory_utilization
* tensor_parallel_size
* max_num_seqs

##### 7.2 不同场景的推荐配置

* Chatbot
* 高并发 API
* 长文本生成

##### 7.3 显存估算方法

* 权重
* KV Cache
* 激活



#### 第 8 章：多 GPU 并行（进阶）

##### 8.1 并行方式概念扫盲

* DP / TP / PP / SP
* vLLM 支持哪些？

##### 8.2 Tensor Parallel 实战

* 多卡启动方式
* NCCL 通信
* 常见错误排查

##### 8.3 性能对比分析

* 单卡 vs 多卡
* 通信开销



### 第 9 章：vLLM + Triton / CUDA 内核

##### 9.1 Triton 在 vLLM 中的作用

* Attention kernel
* 为什么比原生 PyTorch 快

##### 9.2 内核选择与兼容性

* 不同 GPU 架构
* FlashAttention / Triton fallback


#### 第 10 章：生产级部署架构设计

##### 10.1 单机部署架构

* Nginx / FastAPI / vLLM
* 多进程 vs 单进程

##### 10.2 多机部署

* 多实例负载均衡
* 模型副本策略

##### 10.3 与 RAG / Agent 系统集成

* LangChain
* LlamaIndex


#### 第 11 章：监控、稳定性与限流

##### 11.1 性能监控

* GPU 利用率
* QPS / 延迟
* 队列长度

##### 11.2 稳定性问题

* OOM
* 长请求拖慢系统
* 冷启动

##### 11.3 限流与保护

* max_tokens 限制
* 请求超时
* 用户级限流



#### 第 12 章：vLLM 与其他推理框架对比



#### 第 13 章：真实案例与最佳实践

### 三、配套建议

#### 📂 目录结构建议（与你现有一致）

```
0.intro
1.inference_basics
2.vllm_arch
3.quickstart
4.workflow
5.offline_infer
6.openai_server
7.config_tuning
8.multi_gpu
9.triton
10.production
11.monitoring
12.comparison
13.case_study
```

#### 🧪 强烈建议配套

* 每章一个 **可运行 demo**
* 显存 / 性能对比实验
* 参数调优表格

