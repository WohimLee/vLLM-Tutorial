## vLLM 入门与实践
- 官方 User Guide: https://docs.vllm.ai/en/stable/usage/index.html

### 1 什么是 vLLM？
vLLM 是一个高性能的 LLM 推理引擎

>核心特点
- `PagedAttention` 内存管理
- 连续批处理机制
- 支持流式输出与 OpenAI 兼容接口
- 支持多模型部署与 GPU 优化

### 2 环境配置与安装
>📦 安装 vLLM（需要 CUDA 环境）
```
pip install vllm
```

### 3 离线批量推理

- 示例代码：使用 LLM 类与 SamplingParams 进行批量文本生成
- 采样参数：设置温度（temperature）、核心采样概率（top_p）等
- 模型加载：支持 OPT、LLaMA、Qwen 等模型
```py

```


### 4 在线推理服务部署

- 启动命令：vllm serve 启动服务
- 参数配置：
    - 设置最大模型长度（--max-model-len）
    - 启用/禁用异步输出处理（--disable-async-output-proc）
- API 接入：兼容 OpenAI API，可通过 Python 客户端进行调用