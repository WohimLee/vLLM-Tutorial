vllm serve /home/azen/model/qwen2.5-7B-instruct \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.95 \
  --dtype float16