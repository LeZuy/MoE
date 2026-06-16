import os
import zmq
import yaml
import uuid
import time
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from network.expert_agent import ExpertClient
from network.utils import load_placement, map_ec, request_to_log_obj

CONFIG_PATH = "/home/duy.le004/phd/MoE/network/configs/configs.yaml"
EXPERT_ADDS_PATH = "/home/duy.le004/phd/MoE/network/configs/expert_addrs.yaml"

def get_expert_address(adds_path: str = CONFIG_PATH) -> dict[int, str]:
    with open(adds_path, "r") as f:
        return yaml.safe_load(f)

def get_map_node_expert(config_path:str = CONFIG_PATH) -> dict[int, int]:
    return map_ec(load_placement(config_path))

@dataclass
class PendingRequest:
    """Carries everything needed to collect responses for one already-sent
    expert round-trip.  Returned by Router.send_request and consumed by
    Router.recv_response."""
    node_requests: dict          # node_id → request dict (contains request_id)
    original_device: torch.device
    original_dtype: torch.dtype
    final: torch.Tensor          # pre-allocated accumulator (CPU)


class Router:
    def __init__(self, config_path: str = CONFIG_PATH):
        """
        expert_nodes:
            node_id -> address
            {
                0: "tcp://127.0.0.1:5000",
                1: "tcp://127.0.0.1:5001",
            }
        """
        self.map_node_expert = get_map_node_expert(config_path)
        self.map_expert_addr = get_expert_address(EXPERT_ADDS_PATH)
        self.expert_clients = {
            expert_id: ExpertClient(address)
            for expert_id, address in self.map_expert_addr.items()
        } 
        print(f"[{datetime.now().strftime('%H:%M:%S')}][router] Got expert node addresses: {self.map_expert_addr}")

    def _build_node_requests(self, layer_idx, hidden_cpu, index_cpu, weights_cpu):
        """Group token-expert pairs by remote expert node."""
        grouped_pairs: dict[int, list[tuple[int, int]]] = {}

        num_tokens, top_k = index_cpu.shape
        for token_idx in range(num_tokens):
            for top_k_pos in range(top_k):
                expert_id = int(index_cpu[token_idx, top_k_pos].item())
                node_id = self.map_node_expert[expert_id]["rank"]
                grouped_pairs.setdefault(node_id, []).append((token_idx, top_k_pos))

        requests = {}
        for node_id, pairs in grouped_pairs.items():
            token_idx = torch.tensor([p[0] for p in pairs], dtype=torch.long)
            top_k_pos = torch.tensor([p[1] for p in pairs], dtype=torch.long)

            requests[node_id] = {
                "type": "forward",
                "request_id": str(uuid.uuid4()),
                "layer_idx": layer_idx,
                "hidden_states": hidden_cpu[token_idx],
                "expert_ids": index_cpu[token_idx, top_k_pos],
                "weights": weights_cpu[token_idx, top_k_pos],
                "token_idx": token_idx,
            }

        return requests
    
    @torch.no_grad()
    def send_request(
        self,
        layer_idx,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> PendingRequest:
        """Dispatch expert requests for one layer and return immediately.

        Fires all ZMQ sends without blocking on any recv.  The caller is free
        to do other work (e.g. run the next layer's attention) before calling
        recv_response() to collect the results.
        """
        original_device = hidden_states.device
        original_dtype = hidden_states.dtype

        hidden_cpu = hidden_states.detach().to("cpu")
        index_cpu = top_k_index.detach().to("cpu")
        weights_cpu = top_k_weights.detach().to("cpu")

        os.makedirs("./logs/packets/router", exist_ok=True)

        node_requests = self._build_node_requests(
            layer_idx=layer_idx,
            hidden_cpu=hidden_cpu,
            index_cpu=index_cpu,
            weights_cpu=weights_cpu,
        )

        for node_id, request in node_requests.items():
            client = self.expert_clients[node_id]
            print(
                f"[{datetime.now().strftime('%H:%M:%S')}][router] async-send"
                f" {len(request['token_idx'])} token-expert pairs"
                f" to expert node {node_id} at {client.address}",
                flush=True,
            )
            client.send(request)

        return PendingRequest(
            node_requests=node_requests,
            original_device=original_device,
            original_dtype=original_dtype,
            final=torch.zeros(hidden_cpu.shape, dtype=hidden_cpu.dtype),
        )

    @torch.no_grad()
    def recv_response(self, pending: PendingRequest) -> torch.Tensor:
        """Collect expert responses for a previously dispatched PendingRequest.

        Blocks until all expert nodes have replied, receiving from whichever
        node responds first to avoid head-of-line blocking.  Returns the
        accumulated expert output cast back to the original device/dtype.
        """
        # Concurrent recv across *different* sockets — each ExpertClient owns
        # its own ZMQ socket so there is no cross-thread socket sharing.
        with ThreadPoolExecutor(max_workers=len(pending.node_requests)) as executor:
            future_to_node = {
                executor.submit(self.expert_clients[node_id].recv): node_id
                for node_id in pending.node_requests
            }

            for future in as_completed(future_to_node):
                node_id = future_to_node[future]
                request = pending.node_requests[node_id]
                response = future.result()  # re-raises any TimeoutError from ExpertClient

                if not response["ok"]:
                    raise RuntimeError(f"Expert node {node_id} failed: {response}")

                if response.get("request_id") != request["request_id"]:
                    raise RuntimeError(
                        f"Mismatched response from expert node {node_id}: "
                        f"expected request_id={request['request_id']}, "
                        f"got request_id={response.get('request_id')}"
                    )

                pending.final.index_add_(0, response["token_idx"], response["partial_output"])

                with open(os.path.join("./logs/packets/router", f"RES_{response['request_id']}.txt"), "w") as f:
                    yaml.safe_dump(request_to_log_obj(response), f, sort_keys=False)

                print(
                    f"[{datetime.now().strftime('%H:%M:%S')}][router] async-recv"
                    f" response from expert node {node_id}"
                    f" latency_ms={response.get('latency_ms')}",
                    flush=True,
                )

        return pending.final.to(device=pending.original_device, dtype=pending.original_dtype)

    @torch.no_grad()
    def forward(self, layer_idx, hidden_states, top_k_index, top_k_weights) -> torch.Tensor:
        """Convenience wrapper: send + recv in one blocking call.

        Use send_request / recv_response directly when you want to overlap
        the network round-trip with local computation.
        """
        pending = self.send_request(
            layer_idx=layer_idx,
            hidden_states=hidden_states,
            top_k_index=top_k_index,
            top_k_weights=top_k_weights,
        )
        return self.recv_response(pending)

    def close(self) -> None:
        for client in self.expert_clients.values():
            client.close()