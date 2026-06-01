import copy
import torch
import torch.nn as nn
from typing import  List, Dict, Tuple
from models.olmoe.modeling_olmoe import OlmoeExperts, OlmoeConfig

class LocalExperts:
    """A client keeps a subset of experts,
    Unaware of attention and gating layers"""
    def __init__(
        self, 
        client_id: int, 
        g_ids: List[int],   # global expert indices
        local_experts: OlmoeExperts,  # client posses j experts
        device: torch.device = None,
    ):
        self.client_id = client_id
        self.g_ids = g_ids
        # map global indices to local [8, 16, 32, ...] => [0, 1, 2, ...]
        self.e_map = { g_id: i for i, g_id in enumerate(g_ids) }
        self.local_experts = local_experts
        self.device = device or torch.device("cpu")

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor
    ) -> torch.Tensor:
        """Hidden states passed into OlmoeExperts.forward, 
           - indices are mapped
           - masking token """
        hidden_states = hidden_states.to(self.device)
        top_k_index = top_k_index.to(self.device)
        top_k_weights = top_k_weights.to(self.device)

        max_expert_id = int(top_k_index.max().item()) + 1
        lookup = torch.full((max_expert_id,), -1, dtype=torch.long, device=self.device)
        for gid, lid in self.e_map.items():
            if gid < max_expert_id:
                lookup[gid] = lid

        this_client = (lookup[top_k_index] >= 0)
        masked_index = lookup[top_k_index].clone()
        masked_index[~this_client] = len(self.g_ids)
        masked_weights = top_k_weights * this_client.to(top_k_weights.dtype)

        return self.local_experts(hidden_states, masked_index, masked_weights)
    
    @staticmethod
    def from_olmoe_experts(
        experts,
        client_id: int,
        expert_ids: List[int],
        config: OlmoeConfig,
        device: torch.device = None,
    ) -> "LocalExperts":
        import copy
        sub_config = copy.copy(config)
        sub_config.num_local_experts = len(expert_ids)

        sub_experts = OlmoeExperts(sub_config)

        idx = torch.tensor(expert_ids, dtype=torch.long)

        # ✅ Dùng narrow + contiguous thay vì copy toàn bộ
        # Chỉ hiệu quả nếu expert_ids liên tiếp — dùng advanced index nếu không
        with torch.no_grad():
            sub_experts.gate_up_proj = nn.Parameter(
                experts.gate_up_proj[idx],   # view nếu idx liên tiếp
                requires_grad=False
            )
            sub_experts.down_proj = nn.Parameter(
                experts.down_proj[idx],
                requires_grad=False
            )

        # ✅ Không .to(device) — share tensor với model gốc, tránh copy
        sub_experts.act_fn = experts.act_fn

        return LocalExperts(
            client_id=client_id,
            g_ids=expert_ids,
            local_experts=sub_experts,
            device=device,
        )
    
class LocalNode:
    """ - attention layer, 
        - input_layernorm, 
        - post_attention_layernorm 
        - router """
    def __init__(
        self,
        node_id: int,
        attention: nn.Module,          # OlmoeAttention
        input_layernorm: nn.Module,    # OlmoeRMSNorm
        post_attn_layernorm: nn.Module,# OlmoeRMSNorm
        router: nn.Module,             # OlmoeTopKRouter
        rotary_emb: nn.Module,
        num_experts_per_tok: int,      # top-k
        num_experts: int,          
        device: torch.device = None,
    ):
        self.node_id = node_id
        self.attention = attention
        self.input_layernorm = input_layernorm
        self.post_attn_layernorm = post_attn_layernorm
        self.rotary_emb = rotary_emb
        self.router = router
        self.num_experts_per_tok = num_experts_per_tok
        self.num_experts = num_experts
        self.device = device or torch.device("cpu")

    def run_attention(
        self,
        hidden_states: torch.Tensor,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        output_attentions=False,
        use_cache=False,
        cache_position=None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        attn_output, _ = self.attention(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
        )

        hidden_states = residual + attn_output
        residual = hidden_states
        moe_input = self.post_attn_layernorm(hidden_states)
        return residual, moe_input

    def run_moe(
        self,
        moe_input: torch.Tensor,
        dispatcher,
        parallel: bool = False,
    ) -> torch.Tensor:
        B, S, H = moe_input.shape
        flat = moe_input.view(-1, H)

        with torch.no_grad():
            # test = self.router(flat[:1])
            # print(type(test), test[0].shape if isinstance(test, tuple) else test.shape)
            _, top_k_weights, top_k_index = self.router(flat)
            moe_out = dispatcher.dispatch(flat, top_k_index, top_k_weights, parallel=parallel)

            return moe_out.view(B, S, H)