import os
import zmq
import yaml
import uuid
import time
import torch
from datetime import datetime
from collections import defaultdict
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
            node_id: ExpertClient(address)
            for node_id, address in self.map_expert_addr.items()
        } 
        print(f"[{datetime.now().strftime('%H:%M:%S')}][router] Got expert node addresses: {self.map_expert_addr}")

    def _build_node_requests(self, layer_idx, hidden_cpu, index_cpu, weights_cpu):
        """Group token-expert pairs by remote expert node."""
        grouped_pairs: dict[int, dict[int, list[tuple[int, int]]]] = defaultdict(lambda: defaultdict(list))

        num_tokens, top_k = index_cpu.shape

        for token_idx in range(num_tokens):
            for top_k_pos in range(top_k):
                expert_id = int(index_cpu[token_idx, top_k_pos].item())
                node_id = self.map_node_expert[expert_id]["rank"]

                grouped_pairs[node_id][expert_id].append((token_idx, top_k_pos))

        requests = {}

        for node_id, expert_groups in grouped_pairs.items():
            token_to_local: dict[int, int] = {}
            unique_tokens: list[int] = []
            for pairs in expert_groups.values():
                for token_idx, _ in pairs:
                    if token_idx not in token_to_local:
                        token_to_local[token_idx] = len(unique_tokens)
                        unique_tokens.append(token_idx)

            original_token_idx = torch.tensor(unique_tokens, dtype=torch.long)
            groups = {}

            for expert_id, pairs in expert_groups.items():
                pair_token_idx = torch.tensor([p[0] for p in pairs], dtype=torch.long)
                pair_top_k_pos = torch.tensor([p[1] for p in pairs], dtype=torch.long)

                local_pos = torch.tensor(
                    [token_to_local[int(t)] for t in pair_token_idx],
                    dtype=torch.long,
                )

                groups[int(expert_id)] = {
                    "local_pos": local_pos,
                    "weights": weights_cpu[pair_token_idx, pair_top_k_pos],
                }

            requests[node_id] = {
                "type": "forward_grouped",
                "request_id": str(uuid.uuid4()),
                "layer_idx": layer_idx,
                "hidden_states": hidden_cpu[original_token_idx],
                "original_token_idx": original_token_idx,
                "groups": groups,
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

            # with open(os.path.join("./logs/packets/router", f"REQ_{request['request_id']}.txt"), "w") as f:
            #         yaml.safe_dump(request_to_log_obj(request), f, sort_keys=False)
            num_items = (
                len(request["token_idx"])
                if "token_idx" in request
                else sum(len(group["local_pos"]) for group in request["groups"].values())
            )

            # num_unique_tokens = (
            #     len(request["token_idx"])
            #     if "token_idx" in request
            #     else len(request["original_token_idx"])
            # )

            # print(f"[{datetime.now().strftime('%H:%M:%S')}][router] \
            #       async-send {num_items} token-expert pairs,\
            #         {num_unique_tokens} unique tokens token-expert pairs\
            #         to expert node {node_id} at {client.address}",
            #     flush=True,
            # )

            client.send(request)

        # receive from every expert node and accumulate outputs.
        poller = zmq.Poller()

        pending = set(node_requests.keys())
        socket_to_node = {}

        for node_id in pending:
            client = self.expert_clients[node_id]
            poller.register(client.socket, zmq.POLLIN)
            socket_to_node[client.socket] = node_id

        timeout_ms = 1200_000
        deadline = time.time() + timeout_ms / 1000.0

        while pending:
            remaining_ms = int((deadline - time.time()) * 1000)

            if remaining_ms <= 0:
                raise TimeoutError(f"Timed out waiting for expert nodes: {sorted(pending)}")

            events = dict(poller.poll(remaining_ms))

            if not events:
                raise TimeoutError(f"Timed out waiting for expert nodes: {sorted(pending)}")

            for socket, event in events.items():
                if not (event & zmq.POLLIN):
                    continue

                node_id = socket_to_node[socket]
                client = self.expert_clients[node_id]
                request = node_requests[node_id]

                response = client.recv()

                poller.unregister(client.socket)
                pending.remove(node_id)

                if not response["ok"]:
                    raise RuntimeError(f"[{datetime.now().strftime('%H:%M:%S')}] Expert node {node_id} failed: {response}")

                if response.get("request_id") != request["request_id"]:
                    raise RuntimeError(
                        f"Mismatched response from expert node {node_id}: "
                        f"expected request_id={request['request_id']}, "
                        f"got request_id={response.get('request_id')}"
                    )

                partial_output = response["partial_output"]
                response_token_idx = response["token_idx"]

                final.index_add_(0, response_token_idx, partial_output)

                # with open(os.path.join("./logs/packets/router", 
                #                  f"RES_{response['request_id']}.txt",), "w",) as f:
                #     yaml.safe_dump(request_to_log_obj(response), f, sort_keys=False)

                # print(f"[{datetime.now().strftime('%H:%M:%S')}][router] async-recv response from expert node {node_id} "
                #     f"latency_ms={response.get('latency_ms')}",
                #     flush=True,)
                
        return final.to(device=original_device, dtype=original_dtype)

    def close(self) -> None:
        for client in self.expert_clients.values():
            client.close()