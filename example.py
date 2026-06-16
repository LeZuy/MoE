import os
import torch
import time
from dotenv import load_dotenv
from transformers import AutoTokenizer
from models.olmoe.modeling_olmoe import OlmoeForCausalLM
from models.olmoe.decentralized.attngate import AttnGate
from network.router_agent import Router

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROMPTS = [
    "What is Bitcoin?",
    "Explain quantum computing in simple terms.",
    "What is the capital of France?",
]
MAX_LENGTH = 8
CONFIG_PATH = "./network/configs/configs.yaml"

if __name__ == "__main__":

    load_dotenv()
    hf_token = os.environ["HF_TOKEN"]
    pretrained_path = os.environ["PRETRAINED_MODEL_PATH"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = OlmoeForCausalLM.from_pretrained(
        pretrained_model_name_or_path=pretrained_path,
        token=hf_token,
        torch_dtype=torch.bfloat16,
    ).to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(pretrained_path)

    # Decoder-only models have no dedicated pad token; use EOS.
    # Left-pad so that real tokens are right-aligned and position IDs
    # remain contiguous across all sequences in the batch.
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    attg_module = AttnGate(model).eval()

    # each client holds 64/8 = 8 experts
    router = Router(CONFIG_PATH)

    # -----------------------------------------------------------------------
    # Batched pipelined autoregressive decoding
    #
    # AttnGate already handles arbitrary batch sizes:
    #   - preprocess_inputs  : embeds [B, seq] → works for any B
    #   - forward            : flattens to [B*seq, hidden] before gate → any B
    #   - merge_expert_output: reshapes back to [B, seq, hidden] → any B
    #
    # Per-sequence EOS tracking:
    #   finished[b] = True once sequence b has emitted EOS.
    #   Finished sequences keep receiving eos_token_id so output_ids stays
    #   well-formed, but they are excluded from the break condition check.
    #
    # Attention mask:
    #   We carry the original padding mask from tokenization and extend it by
    #   one column of 1s at each step (newly appended tokens are always real).
    # -----------------------------------------------------------------------
    encoding = tokenizer(PROMPTS, return_tensors="pt", padding=True).to(device)
    output_ids    = encoding["input_ids"]       # [B, prompt_len]
    attn_mask     = encoding["attention_mask"]  # [B, prompt_len]  (0 = pad, 1 = real)
    batch_size    = output_ids.shape[0]
    num_layers    = model.config.num_hidden_layers

    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

    start = time.time()
    for _ in range(MAX_LENGTH):
        preprocessed = attg_module.preprocess_inputs(
            input_ids=output_ids,
            attention_mask=attn_mask,
        )

        hidden_states = preprocessed["inputs_embeds"]

        # --- Warm-start: layer 0 attention + gate + send (no recv yet) ------
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

        # --- Pipeline body: recv layer i-1, then attn+send layer i ----------
        for layer_idx in range(1, num_layers):
            expert_output = router.recv_response(pending)
            hidden_states = attg_module.merge_expert_output(
                ffn_residual=ffn_residual,
                expert_output=expert_output,
                original_shape=req["original_shape"],
            )

            ffn_residual, req = attg_module.forward(
                layer_idx=layer_idx,
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

        # --- Flush: collect the last layer's expert output ------------------
        expert_output = router.recv_response(pending)
        hidden_states = attg_module.merge_expert_output(
            ffn_residual=ffn_residual,
            expert_output=expert_output,
            original_shape=req["original_shape"],
        )

        # Pick next token for each sequence in the batch
        logits           = attg_module.final_logits(hidden_states)
        next_token_logits = logits[:, -1, :]                         # [B, vocab]
        next_token_id    = torch.argmax(next_token_logits, dim=-1)   # [B]

        # Finished sequences emit EOS so output_ids stays well-formed
        next_token_id = torch.where(
            finished,
            torch.tensor(tokenizer.eos_token_id, device=device),
            next_token_id,
        )

        output_ids = torch.cat(
            [output_ids, next_token_id.unsqueeze(1)], dim=-1
        )                                                             # [B, seq+1]

        # Extend attention mask: new column is always 1 (real token)
        attn_mask = torch.cat(
            [attn_mask, torch.ones(batch_size, 1, dtype=attn_mask.dtype, device=device)],
            dim=-1,
        )

        finished = finished | (next_token_id == tokenizer.eos_token_id)
        if finished.all():
            break

    decentralized_time = time.time() - start

    print(f"\n{'='*60}")
    print(f"Decentralized inference  ({MAX_LENGTH} max tokens, batch={batch_size})")
    print(f"Took {decentralized_time:.2f} seconds")
    print(f"{'='*60}")
    for i, prompt in enumerate(PROMPTS):
        decoded = tokenizer.decode(output_ids[i], skip_special_tokens=True)
        print(f"[{i}] Prompt : {prompt}")
        print(f"[{i}] Output : {decoded}")
        print()

    # -----------------------------------------------------------------------
    # Baseline: HuggingFace generate() for comparison
    # -----------------------------------------------------------------------
    encoding_base = tokenizer(PROMPTS, return_tensors="pt", padding=True).to(device)
    start = time.time()
    with torch.no_grad():
        out = model.generate(**encoding_base, max_new_tokens=MAX_LENGTH)
    baseline_time = time.time() - start

    print(f"{'='*60}")
    print(f"Baseline HF generate()   ({MAX_LENGTH} max tokens, batch={batch_size})")
    print(f"Took {baseline_time:.2f} seconds")
    print(f"{'='*60}")
    for i, prompt in enumerate(PROMPTS):
        print(f"[{i}] Prompt : {prompt}")
        print(f"[{i}] Output : {tokenizer.decode(out[i], skip_special_tokens=True)}")
        print()

