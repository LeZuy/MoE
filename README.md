## 🧠 Open-Source MoE Models

| Model | Params (Total / Active) | Experts (k/N) | MMLU (%) | HuggingFace | Tuning (QLoRA)  | GitHub |
|:------|:-----------------------:|:-------------:|:--------:|:-----------:|:-----:|:----:|
|[Switch Transformer (2021) ](https://arxiv.org/abs/2101.03961)| 1.6T / ~7B | 1 / 2048 | N/A | N/A | |  |
| [OpenMoE (2024)](https://arxiv.org/abs/2402.01739)| 8B–34B / ~2B | 4 / 32 | ~52.0 | [Model](https://huggingface.co/OrionZheng/openmoe-8b) |  |  |
|  [DeepSeekMoE (2024)](https://arxiv.org/abs/2401.06066)| 16B / 2.8B | 6+2 / 64 | 79.0 | [Model](https://huggingface.co/deepseek-ai/deepseek-moe-16b-base) |  |  |
| [Mixtral 8x7B (2024)](https://arxiv.org/abs/2401.04088) | 47B / 13B | 2 / 8 | 70.6 | [Model](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1) |  | [MoE-PEFT](https://github.com/TUDB-Labs/MoE-PEFT) |
| [Phi-3.5-MoE (2024)](https://arxiv.org/abs/2404.14219) | 41.9B / 6.6B | 2 / 16 | 78.9 | [Model](https://huggingface.co/microsoft/Phi-3.5-MoE-instruct) |  |  |
| [OLMoE (2024)](https://arxiv.org/abs/2409.02060) | __7B / 1B__ | __8 / 64__ | 64.3 | [Model](https://huggingface.co/allenai/OLMoE-1B-7B-0924) | 12 GB VRAM / 32 GB RAM| [MoE-PEFT](https://github.com/TUDB-Labs/MoE-PEFT) |
| [Llama-4 Scout (2025)](https://ai.meta.com/blog/llama-4-multimodal-intelligence/) | 109B / 17B | 1 / 16 | 74.3 (*) | [Model](https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E-Instruct) | |  |
|[Qwen3-30B-A3B (2025)](https://arxiv.org/abs/2505.09388)| __30B / 3B__ | __8 / 128__ | __79.6__ | [Model](https://huggingface.co/Qwen/Qwen3-30B-A3B) | 48 GB VRAM / 128 GB RAM| [qwen3-8b from scratch](https://github.com/Chen-Oliver/qwen3-8b-base) |