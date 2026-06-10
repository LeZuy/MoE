import torch
from torch import nn

class LocalExpert(nn.Module):
    def __init__(self, full_model):
        super().__init__()
        self.experts_by_layer = nn.ModuleList([
            layer.mlp.experts for layer in full_model.model.layers
        ])

    @torch.no_grad()
    def forward(self, layer_idx, hidden_states, top_k_index, top_k_weights):
        experts = self.experts_by_layer[layer_idx]
        return experts(hidden_states, top_k_index, top_k_weights)