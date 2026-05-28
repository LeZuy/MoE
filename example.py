from models.olmoe.modeling_olmoe import OlmoeForCausalLM
from transformers import AutoTokenizer

PRETRAINED_PATH = "/home/duy.le004/.cache/huggingface/hub/models--allenai--OLMoE-1B-7B-0924/snapshots/6d84c48581ece794365f2b8e9cfb043c68ade9c5"

if __name__ == "__main__":

    model = OlmoeForCausalLM.from_pretrained(PRETRAINED_PATH)
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

    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_PATH)
    print(f"Tokenizer: {tokenizer}")

    out = model.generate(**tokenizer("Bitcoin is", return_tensors="pt"), max_length=64)
    print(f"Output: {out}")
    print(f"Detokenized: {tokenizer.decode(out[0])}")

    for hook in hooks:
        hook.remove()

    for layer_idx, info in enumerate(routing_info):
        print(f"\nLayer {layer_idx}:")
        print(f"\t + Expert indices: {info['indices']}")
        print(f"\t + Expert scores:  {info['scores']}")
        print(f"\t + Logit scores:  {info['logits']}")