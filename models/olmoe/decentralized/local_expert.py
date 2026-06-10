import torch
import torch.nn as nn
import torch.nn.functional as F

class LocalExpert(nn.Module):

    def __init__(self, model, expert_id: int, device="cpu"):
        super().__init__()
        self.expert_id = expert_id
        self.device = torch.device(device)

        self.num_layers = model.config.num_hidden_layers

        gate_up_list = []
        down_list = []

        for layer_idx in range(self.num_layers):
            experts = model.model.layers[layer_idx].mlp.experts

            gate_up = experts.gate_up_proj[expert_id].detach().clone().to(self.device)
            down = experts.down_proj[expert_id].detach().clone().to(self.device)

            gate_up_list.append(gate_up)
            down_list.append(down)

        # Buffer instead of ParameterList
        for layer_idx, gate_up in enumerate(gate_up_list):
            self.register_buffer(f"gate_up_proj_{layer_idx}", gate_up)

        for layer_idx, down in enumerate(down_list):
            self.register_buffer(f"down_proj_{layer_idx}", down)

        # act_fn
        self.act_fn = model.model.layers[0].mlp.experts.act_fn

    def gate_up_proj(self, layer_idx):
        return getattr(self, f"gate_up_proj_{layer_idx}")

    def down_proj(self, layer_idx):
        return getattr(self, f"down_proj_{layer_idx}")

    @torch.no_grad()
    def forward(self, layer_idx, token_idx, hidden_states, weights):
        """
        hidden_states: [num_selected_tokens, hidden_dim]
        weights:       [num_selected_tokens]
        token_idx:     [num_selected_tokens]
        """

        hidden_states = hidden_states.to(self.device)
        weights = weights.to(self.device)
        token_idx = token_idx.to(self.device)

        gate_up_weight = self.gate_up_proj(layer_idx)
        down_weight = self.down_proj(layer_idx)

        gate, up = F.linear(hidden_states, gate_up_weight).chunk(2, dim=-1)
        hidden = self.act_fn(gate) * up
        out = F.linear(hidden, down_weight)

        out = out * weights[:, None]

        return {
            "token_idx": token_idx,
            "partial_output": out,
        }