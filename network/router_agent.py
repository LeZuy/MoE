import os
import yaml
import uuid
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
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
    def forward(self, layer_idx, hidden_states, top_k_index, top_k_weights):
        original_device = hidden_states.device
        original_dtype = hidden_states.dtype

        hidden_cpu = hidden_states.detach().to("cpu")
        index_cpu = top_k_index.detach().to("cpu")
        weights_cpu = top_k_weights.detach().to("cpu")

        final = torch.zeros_like(hidden_cpu)
        os.makedirs("./logs/packets/router", exist_ok=True)

        node_requests = self._build_node_requests(
            layer_idx=layer_idx,
            hidden_cpu=hidden_cpu,
            index_cpu=index_cpu,
            weights_cpu=weights_cpu,
        )

        # send to every expert node.
        for node_id, request in node_requests.items():
            client = self.expert_clients[node_id]

            with open(os.path.join("./logs/packets/router", f"REQ_{request['request_id']}.txt"), "w") as f:
                    yaml.safe_dump(request_to_log_obj(request), f, sort_keys=False)

            print(f"[{datetime.now().strftime('%H:%M:%S')}][router] async-send {len(request['token_idx'])} token-expert pairs "
                f"to expert node {node_id} at {client.address}",
                flush=True,
            )

            client.send(request)

        # Receive from all expert nodes concurrently — whichever replies first
        # is processed first, avoiding head-of-line blocking.
        # Each node has its own ExpertClient/ZMQ socket, so concurrent recv
        # across *different* sockets is safe.
        with ThreadPoolExecutor(max_workers=len(node_requests)) as executor:
            future_to_node = {
                executor.submit(self.expert_clients[node_id].recv): node_id
                for node_id in node_requests
            }

            for future in as_completed(future_to_node):
                node_id = future_to_node[future]
                request = node_requests[node_id]
                response = future.result()  # re-raises any recv exception

                if not response["ok"]:
                    raise RuntimeError(f"Expert node {node_id} failed: {response}")

                if response.get("request_id") != request["request_id"]:
                    raise RuntimeError(
                        f"Mismatched response from expert node {node_id}: "
                        f"expected request_id={request['request_id']}, "
                        f"got request_id={response.get('request_id')}"
                    )

                partial_output = response["partial_output"]
                response_token_idx = response["token_idx"]

                final.index_add_(0, response_token_idx, partial_output)

                with open(os.path.join("./logs/packets/router", f"RES_{response['request_id']}.txt"), "w") as f:
                        yaml.safe_dump(request_to_log_obj(response), f, sort_keys=False)

                print(f"[{datetime.now().strftime('%H:%M:%S')}][router] async-recv response from expert node {node_id} "
                    f"latency_ms={response.get('latency_ms')}",
                    flush=True,
                )

        return final.to(device=original_device, dtype=original_dtype)

    def close(self) -> None:
        for client in self.expert_clients.values():
            client.close()