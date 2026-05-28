## Toward Decentralized MoE Models
#### Open source MoE
| Model | Params (Total / Active) | Experts (k/N) | MMLU (%) | Weight | Tuning (QLoRA)  | Model Source |
|:----|:----:|:----:|:----:|:----:|:----:|:----:|
|[Switch Transformer (2021) ](https://arxiv.org/abs/2101.03961)| 1.6T / ~7B | 1 / 2048 |  | [Model](https://huggingface.co/google/switch-c-2048) |  |  |
| [DeepSeekMoE (2024)](https://arxiv.org/abs/2401.06066)| 16B / 2.8B | 6+2 / 64 | 79.0 | [Model](https://huggingface.co/deepseek-ai/deepseek-moe-16b-base) |  |  |
| [Mixtral 8x7B (2024)](https://arxiv.org/abs/2401.04088) | 47B / 13B | 2 / 8 | 70.6 | [Model](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1) | 48 GB VRAM / 128 GB RAM | [MoE-PEFT](https://github.com/TUDB-Labs/MoE-PEFT) |
| [OLMoE (2024)](https://arxiv.org/abs/2409.02060) | 7B / 1B | 8 / 64 | 64.3 | [Model](https://huggingface.co/allenai/OLMoE-1B-7B-0924) | 12 GB VRAM / 32 GB RAM| [MoE-PEFT](https://github.com/TUDB-Labs/MoE-PEFT) |
| [Qwen3-30B-A3B (2025)](https://arxiv.org/abs/2505.09388)| 30B / 3B | 8 / 128 | 79.6 | [Model](https://huggingface.co/Qwen/Qwen3-30B-A3B) | 48 GB VRAM / 128 GB RAM| [qwen3-8b from scratch](https://github.com/Chen-Oliver/qwen3-8b-base) |
| [Qwen3.5-35B-A3B](https://arxiv.org/abs/2505.09388)| 35B / 3B | 8 / 256| |[Model](https://huggingface.co/HauhauCS/Qwen3.5-35B-A3B-Uncensored-HauhauCS-Aggressive)| |[HFT Github](https://github.com/huggingface/transformers/tree/main/src/transformers/models/qwen3_5_moe) |
| [Qwen3-Next-80B-A3B](https://arxiv.org/abs/2505.09388) |80B / 3B| **10 / 512__** | |[Model](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct)|  | [HFT Github](https://github.com/huggingface/transformers/tree/main/src/transformers/models/qwen3_next) |

#### Papers to read
| Paper | What's it about? | 
|:----|:----|
|[Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](https://arxiv.org/abs/1701.06538)| Mixture of Experts |
|[A Comprehensive Survey of Mixture-of-Experts: Algorithms, Theory, and Applications](https://arxiv.org/abs/2503.07137)|Survey|
|[A Survey on Mixture of Experts in Large Language Models](https://arxiv.org/abs/2407.06204)|Survey|
|[Unified Scaling Laws for Routed Language Models](https://arxiv.org/abs/2202.01169)|Studied the scaling law of MoE language models|
|[Mixture of A Million Experts](https://arxiv.org/abs/2407.04153)|Experimental proof that scaling experts improve inference ([Implementation](https://github.com/huyphan168/PEER))|
|[Hierarchical tree of experts](https://www.cs.toronto.edu/~hinton/absps/hme.pdf)|  |
|[DHT-MoE](https://arxiv.org/pdf/2002.04013)|MoE + DHT to work with millions of experts, experts are organized into a DHT for quick retrieval|
|[TA-MoE](https://arxiv.org/pdf/2302.09915)| Graph-topology aware MoE routing to take communication cost in MoE training loss function|

#### Datasets
+ [MMLU](https://huggingface.co/datasets/cais/mmlu) Multiple-choice questions
+ [GSM8K](https://huggingface.co/datasets/openai/gsm8k) Math exact questions
+ [MATH](https://github.com/hendrycks/math)
+ [WikiText](https://huggingface.co/datasets/Salesforce/wikitext/viewer/wikitext-103-raw-v1/train)  Text generation task

#### Fine tuning a LLM
[Fine tuning a LLM](https://github.com/geronimi73/qlora-minimal/tree/main)
[Unsloth](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide)