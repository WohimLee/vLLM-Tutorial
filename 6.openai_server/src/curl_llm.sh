

# curl -v http://localhost:6666/v1/chat/completions \
#   -H "Content-Type: application/json" \
#   -d '{
#     "model": "models/Qwen3-8B",
#     "messages": [{"role":"user","content":"给我一句话总结今天的工作重点"}]
#   }'


curl -v https://cst-reMm3CI1.llamafactory.com.cn:6666/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "models/Qwen3-8B",
    "messages": [{"role":"user","content":"给我一句话总结今天的工作重点"}]
  }'