import argparse
import time
import zmq
import torch
import io
import os
import yaml
from dotenv import load_dotenv

from ..local_expert import LocalExpert
from ...modeling_olmoe import OlmoeForCausalLM
from .utils import load_placement, get_client_config

CONFIG_PATH = "./models/olmoe/decentralized/network/config.yaml"

def encode_torch(obj: dict) -> bytes:
    """
    Serialize Python dict with torch.Tensor.
    To be repraced with msgpack/numpy later.
    """
    buffer = io.BytesIO()
    torch.save(obj, buffer)
    return buffer.getvalue()

def decode_torch(payload: bytes) -> dict:
    buffer = io.BytesIO(payload)
    return torch.load(buffer, map_location="cpu")

class ExpertClient:
    def __init__(self, address: str, timeout_ms: int = 30000):
        self.address = address
        self.context = zmq.Context.instance()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
        self.socket.setsockopt(zmq.SNDTIMEO, timeout_ms)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.connect(address)

    def send(self, request: dict) -> None:
        self.socket.send(encode_torch(request))

    def recv(self) -> dict:
        return decode_torch(self.socket.recv())

    def forward(self, request: dict) -> dict:
        """Synchronous compatibility path: send one request and wait for it."""
        self.send(request)
        return self.recv()

    def close(self) -> None:
        self.socket.close()
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, required=False)
    parser.add_argument("--rank", type=int, default=0)
    args = parser.parse_args()

    load_dotenv()
    hf_token = os.environ["HF_TOKEN"]
    pretrained_path = os.environ["PRETRAINED_MODEL_PATH"]

    client_cfg = get_client_config(load_placement(CONFIG_PATH), rank=args.rank)
    expert_ids = client_cfg["expert_ids"]
    device = client_cfg["device"]
    port = client_cfg["port"]
    host = args.host
    rank = args.rank

    print(f"[{host}:{port}] expert node {rank} is loading pretrained ...", flush=True)
    
    base_model = OlmoeForCausalLM.from_pretrained(
        pretrained_model_name_or_path=pretrained_path,
        token=hf_token,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )
    base_model.eval()  

    experts = {
        expert_id: LocalExpert( 
            model=base_model,
            expert_id=expert_id,
            device=device,
        ) for expert_id in expert_ids
    }

    print(f"[{host}:{port}] agent {rank} holds {expert_ids}", flush=True)
   
    try: 
        context = zmq.Context()
        socket = context.socket(zmq.REP)
        socket.bind(f"tcp://{host}:{port}")
        print(f"[expert-agent {rank}] listening on tcp://{host}:{port}", flush=True)
        while True:
            request = None

            try:
                payload = socket.recv()
                request = decode_torch(payload)

                print(
                    f"[expert-agent {rank}] received request_id={request.get('request_id')} "
                    f"keys={list(request.keys())}",
                    flush=True,
                )

                if request.get("type") == "shutdown":
                    socket.send(encode_torch({
                        "ok": True,
                        "request_id": request.get("request_id"),
                        "type": "shutdown_ack",
                    }))
                    break

                layer_idx = request["layer_idx"]
                hidden_states = request["hidden_states"]
                request_expert_ids = request["expert_ids"]
                weights = request["weights"]
                original_token_idx = request["token_idx"]

                os.makedirs(f"./logs/packets/agent_{rank}", exist_ok=True)

                with open(os.path.join(f"./logs/packets/agent_{rank}", 
                                       f"REQ_{request['request_id']}.txt"),"w",) as f:
                    yaml.dump(request, f)

                started = time.time()

                partial_output_by_pair = torch.zeros_like(hidden_states)

                for expert_id in expert_ids:
                    pair_pos = torch.where(request_expert_ids == expert_id)[0]

                    if pair_pos.numel() == 0:
                        continue

                    current_state = hidden_states[pair_pos]
                    current_weights = weights[pair_pos]

                    expert = experts[expert_id]

                    print(
                        f"[{host}:{port}] expert {expert_id} processing "
                        f"{pair_pos.numel()} token-expert pairs",
                        flush=True,
                    )

                    result = expert.forward(
                        layer_idx=layer_idx,
                        token_idx=pair_pos,
                        hidden_states=current_state,
                        weights=current_weights,
                    )

                    returned_pair_pos = result["token_idx"].to(partial_output_by_pair.device)
                    partial_output = result["partial_output"].to(partial_output_by_pair.device)

                    partial_output_by_pair.index_add_(
                        0,
                        returned_pair_pos,
                        partial_output.to(partial_output_by_pair.dtype),
                    )

                latency_ms = (time.time() - started) * 1000

                response = {
                    "ok": True,
                    "request_id": request.get("request_id"),
                    "rank": args.rank,
                    "expert_ids": expert_ids,
                    "token_idx": original_token_idx.cpu(),
                    "partial_output": partial_output_by_pair.cpu(),
                    "latency_ms": latency_ms,
                }

                with open(os.path.join(f"./logs/packets/agent_{rank}", 
                                       f"RES_{request['request_id']}.txt"), "w",) as f:
                    yaml.dump(response, f)

                print(
                    f"[{host}:{port}] expert-agent {rank} sending response: "
                    f"response_id=RES_{response['request_id']} "
                    f"latency_ms={response['latency_ms']:.2f}",
                    flush=True,
                )

                socket.send(encode_torch(response))

            except Exception as exc:
                err_response = {
                    "ok": False,
                    "request_id": request.get("request_id") if request is not None else None,
                    "rank": rank,
                    "error": repr(exc),
                }

                print(
                    f"[expert-agent {rank}] ERROR: {repr(exc)}",
                    flush=True,
                )

                socket.send(encode_torch(err_response))
    
    except KeyboardInterrupt:
        print("Shutting down server...")

    finally:
        # GRACEFULL TERMINATION
        socket.setsockopt(zmq.LINGER, 0)  # to avoid hanging infinitely
        socket.close()                      # .close()  for all sockets & devices
        context.term()        
        context.destroy()