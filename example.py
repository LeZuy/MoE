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
CONFIG_PATH = "./network/configs/configs.yaml"

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

    attg_module = AttnGate(model).eval()

    # each client holds 64/8 = 8 experts
    router = Router(CONFIG_PATH)

    # -----------------------------------------------------------------------
    # Pipelined autoregressive decoding
    #
    # For each layer i the timeline is:
    #   send_request(i) → [network in-flight] → recv_response(i)
    #
    # We overlap the network round-trip for layer i with the local attention
    # computation of layer i+1 (a 1-step send-ahead):
    #
    #   Layer i:   attn_i → gate_i → send_i ──────────→ recv_i → merge_i
    #                                        ↘ (network)↗
    #   Layer i+1:                  attn_{i+1} → gate_{i+1} → send_{i+1}
    # -----------------------------------------------------------------------
    inputs = tokenizer(EXMPL_PROMPT, return_tensors="pt").to(device)
    output_ids = inputs["input_ids"]
    num_layers = model.config.num_hidden_layers

    start = time.time()
    for _ in range(MAX_LENGTH):
        preprocessed = attg_module.preprocess_inputs(
            input_ids=output_ids,
            attention_mask=torch.ones_like(output_ids, device=output_ids.device),
        )

        hidden_states = preprocessed["inputs_embeds"]

        # --- Warm-start: layer 0 attention + gate + send (no recv yet) ----
        ffn_residual, req = attg_module.forward(
            layer_idx=0,
            hidden_states=hidden_states,
            causal_mask=preprocessed["causal_mask"],
            position_ids=preprocessed["position_ids"],
            position_embeddings=preprocessed["position_embeddings"],
            past_key_values=preprocessed["past_key_values"],
        )
        pending = router.send_request(
            layer_idx=req["layer_idx"],
            hidden_states=req["hidden_states"],
            top_k_index=req["top_k_index"],
            top_k_weights=req["top_k_weights"],
        )

        # --- Pipeline body: recv layer i-1, then attn+send layer i --------
        for layer_idx in range(1, num_layers):
            # Collect the previous layer's expert output first so we can
            # update hidden_states before running the next attention block.
            expert_output = router.recv_response(pending)
            hidden_states = attg_module.merge_expert_output(
                ffn_residual=ffn_residual,
                expert_output=expert_output,
                original_shape=req["original_shape"],
            )

            # Run this layer's attention + gate with the freshly merged state.
            ffn_residual, req = attg_module.forward(
                layer_idx=layer_idx,
                hidden_states=hidden_states,
                causal_mask=preprocessed["causal_mask"],
                position_ids=preprocessed["position_ids"],
                position_embeddings=preprocessed["position_embeddings"],
                past_key_values=preprocessed["past_key_values"],
            )

            # Dispatch expert request for this layer before blocking on recv.
            pending = router.send_request(
                layer_idx=req["layer_idx"],
                hidden_states=req["hidden_states"],
                top_k_index=req["top_k_index"],
                top_k_weights=req["top_k_weights"],
            )

        # --- Flush: collect the last layer's expert output -----------------
        expert_output = router.recv_response(pending)
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