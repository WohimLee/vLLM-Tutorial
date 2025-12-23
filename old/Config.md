## Config 配置信息

### 1 常见配置信息

>所有涉及的配置信息
```yaml
# vLLM 配置文件示例（带注释）
# 可用于集中管理模型、推理、编译等相关参数

# 模型与分词器配置
model: "/home/azen/model/qwen2.5-7B-instruct"   # 模型路径（本地或 HuggingFace Hub）
tokenizer: "/home/azen/model/qwen2.5-7B-instruct" # tokenizer 路径，默认与 model 同路径
skip_tokenizer_init: false    # 是否跳过 tokenizer 初始化（一般设为 false）
tokenizer_mode: auto          # tokenizer 模式（auto 表示自动检测）
tokenizer_revision: null      # tokenizer 版本控制（用于 HuggingFace repo）
revision: null                # 模型版本控制（用于 HuggingFace repo）
trust_remote_code: false      # 是否信任远程代码（谨慎开启）

# speculative decoding（推测解码）相关
speculative_config: null      # speculative decoding 配置（实验性，可设为 null 关闭）

# 精度 & KV 缓存配置
dtype: torch.float16          # 推理精度（float16 或 bfloat16）
kv_cache_dtype: auto          # KV Cache 精度（auto 表示自动选择）

# 序列长度 & 内存管理
max_seq_len: 32768            # 模型最大上下文长度
gpu_memory_utilization: 0.95  # GPU 显存利用率（决定 KV Cache 池大小）

# 模型加载相关
download_dir: null            # 模型下载目录（HuggingFace 自动缓存时使用）
load_format: auto             # 加载权重的格式（auto: 自动选择）
tensor_parallel_size: 1       # 张量并行大小（多 GPU 推理时调整）
pipeline_parallel_size: 1     # 流水线并行大小（多节点时调整）
disable_custom_all_reduce: false # 是否禁用自定义 all-reduce（分布式通信）

# 量化相关
quantization: null            # 模型量化配置（null 表示不启用）

# 执行与调度
enforce_eager: false          # 是否强制使用 eager 模式（禁用编译图）
device_config: cuda           # 设备配置（cuda / cpu）
seed: 0                       # 随机种子（影响采样）
served_model_name: "/home/azen/model/qwen2.5-7B-instruct"  # 服务端展示的模型名
enable_prefix_caching: null   # 是否启用 prefix caching（节省 KV Cache）

# Prefill & 输出处理
chunked_prefill_enabled: false # 是否启用分块预填充（长 prompt 用，关闭更稳定）
use_async_output_proc: true   # 是否启用异步输出处理（提升吞吐）
use_cached_outputs: true      # 是否缓存输出（避免重复计算）

# 解码配置
decoding_config:
  backend: auto               # 解码后端（auto 表示自动选择）
  disable_fallback: false     # 是否禁用 fallback 解码
  disable_any_whitespace: false # 是否禁用空白符处理
  disable_additional_properties: false # 是否禁用额外属性
  reasoning_backend: ""       # reasoning 解码后端（为空表示默认）

# 可观测性配置
observability_config:
  show_hidden_metrics_for_version: null # 显示隐藏指标的版本（调试用）
  otlp_traces_endpoint: null    # OTLP Traces 端点（分布式追踪）
  collect_detailed_traces: null # 是否收集详细追踪数据

# 编译配置
compilation_config:
  level: 0                     # 编译优化等级（0 表示关闭/最基础）
  debug_dump_path: ""           # 调试信息输出路径
  cache_dir: ""                 # 编译缓存目录
  backend: ""                   # 编译后端（可选 inductor 等）
  custom_ops: []                # 自定义算子
  splitting_ops: null           # 分割算子配置
  use_inductor: true            # 是否使用 TorchInductor
  compile_sizes: []             # 编译时输入大小
  inductor_compile_config:
    enable_auto_functionalized_v2: false
  inductor_passes: {}
  cudagraph_mode: 0             # CUDA 图模式（0 表示关闭）
  use_cudagraph: true           # 是否启用 CUDA Graphs
  cudagraph_num_of_warmups: 0   # CUDA Graph 预热次数
  cudagraph_capture_sizes:      # 支持捕获的 batch size 列表
    [256,248,240,232,224,216,208,200,192,184,176,168,
     160,152,144,136,128,120,112,104,96,88,80,72,
     64,56,48,40,32,24,16,8,4,2,1]
  cudagraph_copy_inputs: false  # 是否复制输入到 CUDA Graph
  full_cuda_graph: false        # 是否启用完整 CUDA Graph
  pass_config:
    enable_fusion: false        # 是否启用算子融合
    enable_noop: false          # 是否启用 No-op pass
  max_capture_size: 256         # 最大捕获 batch size
  local_cache_dir: null         # 本地缓存目录
```

### 2 使用 yaml 文件启动
>`basic.yaml`
```yaml
# config.yaml — vLLM 服务配置示例

# 指定模型路径（与命令行中 POSITIONAL model_tag 替代）
model: "/home/azen/model/qwen2.5-7B-instruct"
host: 0.0.0.0
port: 33333

# 基础设置
dtype: float16
max_model_len: 32768          # 最大上下文长度
gpu_memory_utilization: 0.95  # gpu 显存利用率
tensor_parallel_size: 1
pipeline_parallel_size: 1
```

>`launch.sh`
- 你可以保留多个 YAML（比如 `dev_config.yaml`、`prod_config.yaml`），在不同场景切换。
- 如果只想临时覆盖某些参数，可以 命令行参数覆盖 YAML。例如：
    - 这样会以 `8192` 覆盖 YAML 文件里的 `32768`
```sh
vllm serve --config config.yaml --max-model-len 8192
```
