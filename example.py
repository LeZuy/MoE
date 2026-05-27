import os
import torch

from dotenv import load_dotenv
from transformers import AutoTokenizer
from models.olmoe.modeling_olmoe import OlmoeForCausalLM

EXMPL_PROMPT = "Bitcoin is"
MAX_LENGTH = 64

if __name__ == "__main__":

    load_dotenv()
    hf_token = os.environ["HF_TOKEN"]
    pretrained_path = os.environ["PRETRAINED_MODEL_PATH"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = OlmoeForCausalLM.from_pretrained(
        model = pretrained_path,
        token = hf_token
        ).to(device)
    
    print(f"Model: {model}")

    routing_info = []
    hooks = []
    
    def router_hook(module, input, output):
        router_logits, router_scores, router_indices = output
        routing_info.append({
            "indices": router_indices.detach().cpu(),
            "scores": router_scores.detach().cpu(),
            "logits": router_logits.detach().cpu(),
        })
        
    for layer_idx, layer in enumerate(model.model.layers):
        hook = layer.mlp.gate.register_forward_hook(router_hook)
        hooks.append(hook)

    tokenizer = AutoTokenizer.from_pretrained(pretrained_path)
    print(f"Tokenizer: {tokenizer}")

    out = model.generate(**tokenizer(EXMPL_PROMPT, return_tensors="pt"), max_length=MAX_LENGTH)
    print(f"Output: {out}")
    print(f"Detokenized: {tokenizer.decode(out[0])}")

    for hook in hooks:
        hook.remove()

    for layer_idx, info in enumerate(routing_info):
        print(f"\nLayer {layer_idx}:")
        print(f"\t + Expert indices: {info['indices']}")
        print(f"\t + Expert scores:  {info['scores']}")
        print(f"\t + Logit scores:  {info['logits']}")