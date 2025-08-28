## OpenAI-Compatible Server
### 1 Server 
#### 方案 A（最简单、最稳）：把上下文长度降下来

对 2080 Ti，我建议先从 8K 起步，再慢慢加。
```sh
vllm serve /home/buding/model/qwen2.5-7B-instruct \
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
vllm serve /home/buding/model/qwen2.5-7B-instruct \
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
vllm serve /home/buding/model/qwen2.5-7B-instruct \
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
vllm serve /home/buding/model/qwen2.5-7B-instruct-awq \
  --quantization awq \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.95 \
  --dtype float16
# 或 GPTQ
vllm serve /home/buding/model/qwen2.5-7B-instruct-gptq \
  --quantization gptq \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.95 \
  --dtype float16
```

#### 方案 D：多卡切分（如果你有≥2 张 GPU）

把模型切到两张卡上，KV 也随之扩展：
```sh
CUDA_VISIBLE_DEVICES=0,1 vllm serve /home/buding/model/qwen2.5-7B-instruct \
  --tensor-parallel-size 2 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.95 \
  --dtype float16
```

#### 方案 E：启用 CPU 交换空间兜底（会变慢，但能跑）

给 KV 开一点 CPU 交换空间，防止卡死（延迟会上升）：
```sh
vllm serve /home/buding/model/qwen2.5-7B-instruct \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.95 \
  --swap-space 8 \
  --dtype float16
```