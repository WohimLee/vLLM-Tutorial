## OpenAI-Compatible Server

- 启动命令：vllm serve 启动服务
- 参数配置：
    - 设置最大模型长度（--max-model-len）
    - 启用/禁用异步输出处理（--disable-async-output-proc）
- API 接入：兼容 OpenAI API，可通过 Python 客户端进行调用


### 1 Server 端
#### 方案 A（最简单、最稳）：把上下文长度降下来

对 2080 Ti，我建议先从 8K 起步，再慢慢加。
```sh
vllm serve /home/azen/model/qwen2.5-7B-instruct \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.95 \
  --dtype float16
```

- --max-model-len 8192：把目标降到 KV cache 能承受的范围；
- --gpu-memory-utilization 0.95：在安全范围内多挤一点显存给 KV（你日志里是 0.90）；
- 你的卡不支持 bfloat16，日志已自动退回 fp16，显式写上 --dtype float16 更直观。

如果 8192 仍报同样错误，按 6144 → 4096 逐级往下试一下。

#### 方案 B：降低并发带来的 KV 压力（可与 A 同用）

限制同时跟踪的序列数，减少调度峰值 KV 需求：
```sh
vllm serve /home/azen/model/qwen2.5-7B-instruct \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.95 \
  --max-num-seqs 1 \
  --dtype float16
```
- `--max-num-seqs 1` 会牺牲吞吐但更稳，不易 OOM/报 KV 不足。

#### 方案 C：用权重量化以腾出 KV 空间（最省显存）

权重量化后，权重占用显存大幅下降，KV 可分到更多空间。对单 2080 Ti 很实用。

##### （1）bitsandbytes 4-bit（最方便）
```sh
vllm serve /home/azen/model/qwen2.5-7B-instruct \
  --quantization bitsandbytes \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.95 \
  --dtype float16
```

- 若 16K 仍不稳，退到 12K / 8K。
- 首次用 bitsandbytes 会稍慢（需要编译/加载）。

##### （2）已有 AWQ/GPTQ 量化权重
如果你下载的是 AWQ/GPTQ 量化版：
```sh
# AWQ
vllm serve /home/azen/model/qwen2.5-7B-instruct-awq \
  --quantization awq \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.95 \
  --dtype float16
# 或 GPTQ
vllm serve /home/azen/model/qwen2.5-7B-instruct-gptq \
  --quantization gptq \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.95 \
  --dtype float16
```

#### 方案 D：多卡切分（如果你有≥2 张 GPU）

把模型切到两张卡上，KV 也随之扩展：
```sh
CUDA_VISIBLE_DEVICES=0,1 vllm serve /home/azen/model/qwen2.5-7B-instruct \
  --tensor-parallel-size 2 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.95 \
  --dtype float16
```

#### 方案 E：启用 CPU 交换空间兜底（会变慢，但能跑）

给 KV 开一点 CPU 交换空间，防止卡死（延迟会上升）：
```sh
vllm serve /home/azen/model/qwen2.5-7B-instruct \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.95 \
  --swap-space 8 \
  --dtype float16
```

### 2 Client 端
当你用 vllm serve 成功跑起来后，它会默认启动一个 OpenAI API 兼容的服务。

默认访问方式
- 默认监听地址：http://localhost:8000/v1
- 接口风格：与 OpenAI 的 Chat Completions API 一致。
#### 1 curl
```sh
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/home/azen/model/qwen2.5-7B-instruct",
    "messages": [
      {"role": "user", "content": "你好，请介绍一下你自己"}
    ]
  }'
```

#### 2 openai 库
⚠️ 注意：这里 api_key 可以随便填（比如 "EMPTY"），因为本地服务默认不做鉴权。
```py
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")

resp = client.chat.completions.create(
    model="/home/azen/model/qwen2.5-7B-instruct",
    messages=[
        {"role": "user", "content": "写一首七言绝句"}
    ]
)

print(resp.choices[0].message.content)
```

#### 3 requests 库

```py
import requests

url = "http://localhost:8000/v1/chat/completions"

headers = {
    "Content-Type": "application/json",
}

data = {
    "model": "/home/azen/model/qwen2.5-7B-instruct",
    "messages": [
        {"role": "user", "content": "你好，请介绍一下你自己"}
    ]
}

resp = requests.post(url, headers=headers, json=data)

print(resp.status_code)
print(resp.json())
```
- url 指向 http://localhost:8000/v1/chat/completions（vLLM serve 默认启动在 8000 端口）。
- model 字段要和你 vllm serve 的 --model 参数一致（你的情况就是 "/home/azen/model/qwen2.5-7B-instruct"）。
- api_key 不需要，vLLM 默认不检查，可以忽略。