python -m vllm.entrypoints.openai.api_server \
  --model models/Qwen3-8B \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.85 \
  --max-num-seqs 64 \
  --tensor-parallel-size 1 \
  --port 8000


# python -m vllm.entrypoints.openai.api_server \
#   --model LLaMA-Factory/saves/llama3-8b/lora/sft_r_16/checkpoint-411 \
#   --max-model-len 8192 \
#   --gpu-memory-utilization 0.85 \
#   --max-num-seqs 64 \
#   --tensor-parallel-size 1 \
#   --port 8000