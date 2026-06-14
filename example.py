import os
import torch
import time
from dotenv import load_dotenv
from transformers import AutoTokenizer
from models.olmoe.modeling_olmoe import OlmoeForCausalLM
from models.olmoe.decentralized.attngate import AttnGate
from network.router_agent import Router

EXMPL_PROMPT = "What is Bitcoin?"
MAX_LENGTH = 8
CONFIG_PATH = "/home/duy.le004/phd/MoE/network/configs/configs.yaml"

if __name__ == "__main__":

    load_dotenv()
    hf_token = os.environ["HF_TOKEN"]
    pretrained_path = os.environ["PRETRAINED_MODEL_PATH"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = OlmoeForCausalLM.from_pretrained(
        pretrained_model_name_or_path = pretrained_path,
        token = hf_token,
        torch_dtype=torch.bfloat16,
    ).to(device)
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(pretrained_path)
    # print(f"Tokenizer: {tokenizer}")

    attg_module = AttnGate(model).eval()

    # each client holds 64/8 = 8 experts
    router = Router(CONFIG_PATH)

    # Inference
    inputs = tokenizer(EXMPL_PROMPT, return_tensors="pt").to(device)
    output_ids = inputs["input_ids"]
    start = time.time()
    for _ in range(MAX_LENGTH):
        hidden_states, causal_mask, position_ids, position_embeddings, past_key_values = attg_module.prepare_inputs(
            input_ids=output_ids,
            attention_mask= torch.ones_like(output_ids, device=output_ids.device),
        )

        for layer_idx in range(model.config.num_hidden_layers):
            ffn_residual, req = attg_module.run_attention_and_gate(
                layer_idx=layer_idx,
                hidden_states=hidden_states,
                causal_mask=causal_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
                past_key_values=past_key_values,
            )

            expert_output = router.forward(
                layer_idx=req["layer_idx"],
                hidden_states=req["hidden_states"],
                top_k_index=req["top_k_index"],
                top_k_weights=req["top_k_weights"],
            )

            hidden_states = attg_module.merge_expert_output(
                ffn_residual=ffn_residual,
                expert_output=expert_output,
                original_shape=req["original_shape"],
            )

        logits = attg_module.final_logits(hidden_states)
        next_token_logits = logits[:, -1, :]
        next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        output_ids = torch.cat([output_ids, next_token_id], dim=-1)

        if next_token_id.item() == tokenizer.eos_token_id:
            break

    outputs = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    print(f"Prompt: {EXMPL_PROMPT}")
    print(f"Decentralized ({MAX_LENGTH} tokens): {outputs}")
    print(f"Took {time.time() - start: .2f} seconds")
    encoding = tokenizer(EXMPL_PROMPT, return_tensors="pt").to(device)
    start = time.time()
    with torch.no_grad():
        out = model.generate(**encoding, max_new_tokens=MAX_LENGTH)
    print(f"Original model ({MAX_LENGTH} tokens): {tokenizer.decode(out[0])}")
    print(f"Took {time.time() - start: .2f} seconds")