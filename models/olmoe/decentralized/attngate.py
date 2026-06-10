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
    def prepare_inputs(self, input_ids, attention_mask=None, past_key_values=None, use_cache=True):
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        inputs_embeds = self.embed_tokens(input_ids)

        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        position_ids = torch.arange(
            inputs_embeds.shape[1],
            device=inputs_embeds.device
        ) + past_seen_tokens
        position_ids = position_ids.unsqueeze(0)

        causal_mask = create_causal_mask(
            config=self.config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )

        position_embeddings = self.rotary_emb(inputs_embeds, position_ids)

        return inputs_embeds, causal_mask, position_ids, position_embeddings, past_key_values
    
    @torch.no_grad()
    def run_attention_and_gate(
        self,
        layer_idx,
        hidden_states,
        causal_mask,
        position_ids,
        position_embeddings,
        past_key_values=None,
        use_cache=True,
    ):
        layer = self.layers[layer_idx]

        residual = hidden_states
        x = layer.input_layernorm(hidden_states)

        attn_out, _ = layer.self_attn(
            hidden_states=x,
            attention_mask=causal_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            position_embeddings=position_embeddings,
        )

        hidden_after_attn = residual + attn_out

        ffn_residual = hidden_after_attn
        moe_input = layer.post_attention_layernorm(hidden_after_attn)

        batch_size, sequence_length, hidden_dim = moe_input.shape
        flat_moe_input = moe_input.view(-1, hidden_dim)

        router_logits, top_k_weights, top_k_index = layer.mlp.gate(flat_moe_input)

        request = {
            "layer_idx": layer_idx,
            "hidden_states": flat_moe_input,
            "top_k_index": top_k_index,
            "top_k_weights": top_k_weights,
            "original_shape": (batch_size, sequence_length, hidden_dim),
        }

        return ffn_residual, request

    @torch.no_grad()
    def merge_expert_output(self, ffn_residual, expert_output, original_shape):
        batch_size, sequence_length, hidden_dim = original_shape
        expert_output = expert_output.reshape(batch_size, sequence_length, hidden_dim)
        return ffn_residual + expert_output

    @torch.no_grad()
    def final_logits(self, hidden_states):
        hidden_states = self.norm(hidden_states)
        return self.lm_head(hidden_states)