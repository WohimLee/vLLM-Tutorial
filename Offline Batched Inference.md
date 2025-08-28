## Offline Batched Inference
>Example
- Model: /home/buding/model/qwen2.5-7B-instruct
- 环境: 2 x 2080 Ti
```py
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm import LLM, SamplingParams

# Sample prompts.
prompts = [
    "你是谁？",
    "介绍一下北京",
    "今天天气怎么样？",
]
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)


def main():
    # Create an LLM.
    llm = LLM(
        model="/home/buding/model/qwen2.5-7B-instruct",# 放自己的目录
        enable_chunked_prefill=False,   # key fix
        gpu_memory_utilization=0.98 # <— more VRAM for KV
    ) 
    # Generate texts from the prompts.
    # The output is a list of RequestOutput objects
    # that contain the prompt, generated text, and other information.
    outputs = llm.generate(prompts, sampling_params)
    # Print the outputs.
    print("\nGenerated Outputs:\n" + "-" * 60)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt:    {prompt!r}")
        print(f"Output:    {generated_text!r}")
        print("-" * 60)


if __name__ == "__main__":
    main()
```