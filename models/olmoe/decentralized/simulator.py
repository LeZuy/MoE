# distributed/simulation.py
import torch
import math

from typing import List, Optional
from transformers import PreTrainedTokenizer

from .utils import sample_next_token
from .dispatch import ExpertDispatcher
from .local import LocalExperts, LocalNode
from ..modeling_olmoe import OlmoeForCausalLM, OlmoeConfig

class DeOlmoeSimulator:
    def __init__(
        self,
        model: OlmoeForCausalLM,
        num_clients: int,
        device: Optional[torch.device] = None,
        parallel: bool = False,
    ):
        self.model = model
        self.num_clients = num_clients
        self.device = device or next(model.parameters()).device
        self.parallel = parallel

        config: OlmoeConfig = model.config
        self.num_layers = config.num_hidden_layers
        self.num_experts = config.num_local_experts       # 64
        self.num_experts_per_tok = config.num_experts_per_tok  # 8
        j = math.ceil(self.num_experts / num_clients)

        # Partition experts: client i holds experts [i*j, (i+1)*j)
        self.e_partitions: List[List[int]] =[
            list(range(i * j, min((i + 1) * j, self.num_experts)))
            for i in range(num_clients)
            if i * j < self.num_experts
        ]

        # Build per-layer: node_clients[layer], dispatchers[layer]
        self.nodes: List[LocalNode] = []
        self.dispatchers: List[ExpertDispatcher] = []

        for layer_idx in range(self.num_layers):
            layer = model.model.layers[layer_idx]
            moe_block = layer.mlp   # OlmoeSparseMoeBlock
            experts = moe_block.experts

            rotary_emb = (
                getattr(layer, "rotary_emb", None)
                or getattr(layer.self_attn, "rotary_emb", None)
                or getattr(model.model, "rotary_emb", None)
            )

            node = LocalNode(
                node_id=layer_idx,
                attention=layer.self_attn,
                input_layernorm=layer.input_layernorm,
                post_attn_layernorm=layer.post_attention_layernorm,
                router=moe_block.gate,
                rotary_emb=rotary_emb,        
                num_experts_per_tok=self.num_experts_per_tok,
                num_experts=self.num_experts,
                device=self.device,
            )
            self.nodes.append(node)

            e_list = [
                LocalExperts.from_olmoe_experts(
                    experts=experts,
                    client_id=cid,
                    expert_ids=self.e_partitions[cid],
                    config=config,
                    device=self.device,
                )
                for cid in range(len(self.e_partitions))
            ]
            self.dispatchers.append(ExpertDispatcher(e_list))

    def forward_layer(
        self,
        layer_idx: int,
        hidden_states: torch.Tensor,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        use_cache=False,
        cache_position=None,
    ):
        node = self.nodes[layer_idx]
        dispatcher = self.dispatchers[layer_idx]

        residual, moe_input = node.run_attention(
            hidden_states, attention_mask, position_ids,
            past_key_value, use_cache, cache_position,
        )

        moe_out = node.run_moe(moe_input, dispatcher, parallel=self.parallel)

        return residual + moe_out

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask=None,
    ) -> torch.Tensor:
        with torch.no_grad(): 
            input_ids = input_ids.to(self.device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)

            hidden_states = self.model.model.embed_tokens(input_ids)
            seq_len = hidden_states.shape[1]
            position_ids = torch.arange(seq_len, device=self.device).unsqueeze(0)
            cache_position = torch.arange(seq_len, device=self.device)

            causal_mask = self._prepare_causal_mask(attention_mask, hidden_states)

            for layer_idx in range(self.num_layers):
                new_hidden = self.forward_layer(
                    layer_idx, hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    cache_position=cache_position,
                )
                del hidden_states 
                hidden_states = new_hidden
                torch.cuda.empty_cache()

            hidden_states = self.model.model.norm(hidden_states)
            return self.model.lm_head(hidden_states)

    def _prepare_causal_mask(
        self, 
        attention_mask: torch.Tensor, 
        hidden_states: torch.Tensor
    )->torch.Tensor:
        if hasattr(self.model.model, "_update_causal_mask"):
            cache_position = torch.arange(
                hidden_states.shape[1], device=self.device
            )
            mask = self.model.model._update_causal_mask(
                attention_mask, hidden_states,
                cache_position=cache_position,
                past_key_values=None,
                output_attentions=False,
            )
            if mask is not None and mask.dtype != hidden_states.dtype:
                mask = mask.to(hidden_states.dtype)
            return mask
    
    def generate(
        self,
        tokenizer: PreTrainedTokenizer,
        prompt: str,
        max_tokens: int = 50
    ) -> str :
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        for _ in range(max_tokens):
            logits = self.forward(input_ids, attention_mask)

            # next_token_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            next_token_id = sample_next_token(logits[:, -1, :])

            if next_token_id.item() == tokenizer.eos_token_id:
                break

            input_ids = torch.cat([input_ids, next_token_id], dim=-1)
            attention_mask = torch.cat(
                [attention_mask, torch.ones(1, 1, dtype=attention_mask.dtype, device=self.device)],
                dim=-1
            )

        prompt_len = inputs["input_ids"].shape[1]
        generated_ids = input_ids[:, prompt_len:]
        return tokenizer.decode(generated_ids[0], skip_special_tokens=True)