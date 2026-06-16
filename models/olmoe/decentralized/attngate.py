import torch
from torch import nn
from transformers.cache_utils import DynamicCache
from transformers.masking_utils import create_causal_mask

class AttnGate(nn.Module):
    def __init__(self, full_model):
        super().__init__()
        self.config = full_model.config
        self.embed_tokens = full_model.model.embed_tokens
        self.layers = full_model.model.layers
        self.rotary_emb = full_model.model.rotary_emb
        self.norm = full_model.model.norm
        self.lm_head = full_model.lm_head

    @torch.no_grad()
    def preprocess_inputs(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values=None,
        use_cache: bool = True,
    ):
        # KV cache
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        # Prefill: input_ids shape [B, prompt_len]
        # Decode:  input_ids shape [B, 1]
        inputs_embeds = self.embed_tokens(input_ids)

        past_seen_tokens = (
            past_key_values.get_seq_length()
            if past_key_values is not None
            else 0
        )

        position_ids = torch.arange(
            inputs_embeds.shape[1],
            device=inputs_embeds.device,
        ) + past_seen_tokens
        position_ids = position_ids.unsqueeze(0)

        if attention_mask is None:
            attention_mask = torch.ones(
                (input_ids.shape[0], past_seen_tokens + input_ids.shape[1]),
                device=input_ids.device,
                dtype=torch.long,
            )

        causal_mask = create_causal_mask(
            config=self.config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )

        position_embeddings = self.rotary_emb(inputs_embeds, position_ids)

        return {
            "inputs_embeds": inputs_embeds,
            "causal_mask": causal_mask,
            "position_ids": position_ids,
            "position_embeddings": position_embeddings,
            "past_key_values": past_key_values,
            "attention_mask": attention_mask,
        }
    
    @torch.no_grad()
    def forward(
        self,
        layer_idx: int,
        hidden_states: torch.Tensor,
        causal_mask: torch.Tensor,
        position_ids: torch.Tensor,
        position_embeddings,
        past_key_values=None,
    ):
        layer = self.layers[layer_idx]

        residual = hidden_states
        hidden_states = layer.input_layernorm(hidden_states)

        attn_output, _ = layer.self_attn(
            hidden_states=hidden_states,
            attention_mask=causal_mask,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
            past_key_values=past_key_values,
            use_cache=True,
        )

        hidden_states = residual + attn_output

        ffn_residual = hidden_states
        moe_input = layer.post_attention_layernorm(hidden_states)

        batch_size, sequence_length, hidden_dim = moe_input.shape
        flat_moe_input = moe_input.view(-1, hidden_dim)

        router_logits, top_k_weights, top_k_index = layer.mlp.gate(flat_moe_input)

        req = {
            "layer_idx": layer_idx,
            "hidden_states": flat_moe_input,
            "top_k_index": top_k_index,
            "top_k_weights": top_k_weights,
            "original_shape": (batch_size, sequence_length, hidden_dim),
        }

        return ffn_residual, req

    @torch.no_grad()
    def merge_expert_output(self, ffn_residual, expert_output, original_shape):
        batch_size, sequence_length, hidden_dim = original_shape
        expert_output = expert_output.reshape(batch_size, sequence_length, hidden_dim)
        return ffn_residual + expert_output

    @torch.no_grad()
    def final_logits(self, hidden_states):
        hidden_states = self.norm(hidden_states)
        return self.lm_head(hidden_states)