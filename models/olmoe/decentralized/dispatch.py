import torch
from typing import Dict, List

class ExpertDispatcher:
    def __init__(self, expert_clients, device):
        self.expert_clients = expert_clients
        self.device = device

    @torch.no_grad()
    def dispatch(self, layer_idx, hidden_states, top_k_index, top_k_weights):
        print(hidden_states.dtype)
        final_hidden_states = torch.zeros_like(hidden_states)

        num_experts = len(self.expert_clients)

        for expert_id in range(num_experts):
            token_idx, top_k_pos = torch.where(top_k_index == expert_id)

            if token_idx.numel() == 0:
                continue

            current_state = hidden_states[token_idx]
            current_weights = top_k_weights[token_idx, top_k_pos].to(self.device, dtype=torch.float32)

            response = self.expert_clients[expert_id].forward(
                layer_idx=layer_idx,
                token_idx=token_idx,
                hidden_states=current_state,
                weights=current_weights,
            )

            partial_output = response["partial_output"].to(self.device)
            token_idx = response["token_idx"].to(self.device)

            final_hidden_states.index_add_(0, token_idx, partial_output)

        return final_hidden_states