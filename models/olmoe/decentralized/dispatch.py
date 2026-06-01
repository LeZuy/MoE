import torch
from typing import Dict, List
from .local import LocalExperts

class ExpertDispatcher:
    def __init__(self, experts: List[LocalExperts]):
        self.experts = experts
        # Build: expert_id -> client
        self.expert_to_client: Dict[int, LocalExperts] = {}
        for e in experts:
            for e_id in e.e_map:
                self.expert_to_client[e_id] = e

    def dispatch(
        self,
        hidden_states: torch.Tensor, 
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
        parallel: bool = True,
    ) -> torch.Tensor:

        def _call(local_e: LocalExperts):
            return local_e.forward(hidden_states, top_k_index, top_k_weights)

        if parallel and len(self.experts) > 1:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=len(self.experts)) as exe:
                results = list(exe.map(_call, self.experts))
        else:
            results = [_call(e) for e in self.experts]

        return sum(results)