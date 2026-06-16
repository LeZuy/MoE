import os
import io
import zmq
import time
import yaml
import fcntl
import torch
import argparse
from datetime import datetime
from dotenv import load_dotenv
from models.olmoe.modeling_olmoe import OlmoeForCausalLM
from models.olmoe.decentralized.local_expert import LocalExpert
from network.utils import load_placement, get_client_config, request_to_log_obj

CONFIG_PATH = "./network/configs/configs.yaml"
ADDRS_PATH = "./network/configs/expert_addrs.yaml"

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

def register_address_locked(
    rank: int,
    host: str,
    port: int,
    path: str = ADDRS_PATH,
):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    lock_path = str(path) + ".lock"
    tmp_path = str(path) + f".tmp.{os.getpid()}"

    address = f"tcp://{host}:{port}"

    with open(lock_path, "w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)

        try:
            if os.path.exists(path):
                with open(path, "r") as f:
                    data = yaml.safe_load(f) or {}
            else:
                data = {}

            data[int(rank)] = address

            with open(tmp_path, "w") as f:
                yaml.safe_dump(data, f, sort_keys=True)

            os.replace(tmp_path, path)

        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)

class ExpertClient:
    def __init__(self, address: str, timeout_ms: int = 30000):
        self.address = address
        self.timeout_ms = timeout_ms
        self.context = zmq.Context.instance()
        self._make_socket()

    def _make_socket(self) -> None:
        """Create (or re-create) a fresh REQ socket with the configured options."""
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        self.socket.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.connect(self.address)

    def _reconnect(self) -> None:
        """Close the broken socket and open a fresh one.

        zmq.REQ enforces a strict send→recv state machine. After a timeout or
        a remote crash, the socket stays in EXPECTING_REPLY and every subsequent
        send raises EFSM — there is no in-place recovery. Closing and re-creating
        the socket is the only safe reset.
        """
        try:
            self.socket.close()
        except Exception:
            pass
        self._make_socket()

    def send(self, request: dict) -> None:
        try:
            self.socket.send(encode_torch(request))
        except zmq.ZMQError as exc:
            # EAGAIN  → send timed out (SNDTIMEO elapsed).
            # EFSM    → socket is in wrong state (broken after a prior timeout).
            # Either way, reset the socket so the next forward pass can start clean.
            self._reconnect()
            raise TimeoutError(
                f"[ExpertClient] send to {self.address} failed ({exc}); socket has been reset."
            ) from exc

    def recv(self) -> dict:
        try:
            return decode_torch(self.socket.recv())
        except zmq.ZMQError as exc:
            # EAGAIN → recv timed out (RCVTIMEO elapsed, likely expert node crashed).
            # Reset so the next call can attempt a fresh send→recv cycle.
            self._reconnect()
            raise TimeoutError(
                f"[ExpertClient] recv from {self.address} timed out ({exc}); socket has been reset."
            ) from exc

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
    parser.add_argument("--id", type=int, default=0)
    args = parser.parse_args()

    load_dotenv()
    hf_token = os.environ["HF_TOKEN"]
    pretrained_path = os.environ["PRETRAINED_MODEL_PATH"]

    client_cfg = get_client_config(load_placement(CONFIG_PATH), rank=args.id)
    expert_ids = client_cfg["expert_ids"]
    device = client_cfg["device"]
    port = client_cfg["port"]
    host = args.host
    rank = args.id

    # print(f"[{datetime.now().strftime('%H:%M:%S')}][{host}:{port}] expert node {rank} is loading pretrained ...", flush=True)
    
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

    # print(f"[{datetime.now().strftime('%H:%M:%S')}][{host}:{port}] agent {rank} holds {expert_ids}", flush=True)
    try:
        context = zmq.Context()
        socket = context.socket(zmq.REP)
        socket.setsockopt(zmq.LINGER, 0)
        socket.bind(f"tcp://{host}:{port}")
        # print(f"[{datetime.now().strftime('%H:%M:%S')}][{host}:{port}] expert-agent {rank} listening on tcp://{host}:{port}", flush=True)
        register_address_locked(rank=rank, host=host, port=port)
        while True:
            request = None

            try:
                payload = socket.recv()
                request = decode_torch(payload)

                # print(
                #     f"[{datetime.now().strftime('%H:%M:%S')}][{host}:{port}] expert-agent {rank} received request_id={request.get('request_id')} "
                #     f"keys={list(request.keys())}",
                #     flush=True,
                # )

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

                # os.makedirs(f"./logs/packets/agent_{rank}", exist_ok=True)
                # with open(os.path.join(f"./logs/packets/agent_{rank}",
                #                        f"REQ_{request['request_id']}.txt"), "w") as f:
                #     yaml.safe_dump(request_to_log_obj(request), f, sort_keys=False)

                started = time.time()

                partial_output_by_pair = torch.zeros_like(hidden_states)

                for expert_id in expert_ids:
                    pair_pos = torch.where(request_expert_ids == expert_id)[0]

                    if pair_pos.numel() == 0:
                        continue

                    current_state = hidden_states[pair_pos]
                    current_weights = weights[pair_pos]

                    expert = experts[expert_id]

                    # print(
                    #     f"[{datetime.now().strftime('%H:%M:%S')}][{host}:{port}] expert {expert_id} processing "
                    #     f"{pair_pos.numel()} token-expert pairs",
                    #     flush=True,
                    # )

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
                    "rank": rank,
                    "expert_ids": expert_ids,
                    "token_idx": original_token_idx.cpu(),
                    "partial_output": partial_output_by_pair.cpu(),
                    "latency_ms": latency_ms,
                }

                # with open(os.path.join(f"./logs/packets/agent_{rank}",
                #                        f"RES_{request['request_id']}.txt"), "w") as f:
                #     yaml.safe_dump(request_to_log_obj(response), f, sort_keys=False)

                print(
                    f"[{datetime.now().strftime('%H:%M:%S')}][{host}:{port}] expert-agent {rank} sending response: "
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

                # print(f"[{datetime.now().strftime('%H:%M:%S')}][{host}:{port}] expert-agent {rank} ERROR: {repr(exc)}",
                #     flush=True,
                # )

                socket.send(encode_torch(err_response))

    except KeyboardInterrupt:
        print(f"[{datetime.now().strftime('%H:%M:%S')}][{host}:{port}] node {rank} shutting down agent...")

    finally:
        socket.setsockopt(zmq.LINGER, 0)
        socket.close()
        context.term()
        context.destroy()