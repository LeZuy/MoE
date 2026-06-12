import argparse
import time
import zmq
import torch

from .protocol import encode_torch, decode_torch

class DummyLocalExpert:
    """
    Test: multiple 2 tensors.
    """
    def forward(self, hidden_states, top_k_index, top_k_weights):
        return hidden_states * 2.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--rank", type=int, default=0)
    args = parser.parse_args()

    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind(f"tcp://{args.host}:{args.port}")

    expert = DummyLocalExpert()

    print(f"[expert-server {args.rank}] listening on tcp://{args.host}:{args.port}", flush=True)

    while True:
        payload = socket.recv()
        request = decode_torch(payload)

        if request.get("type") == "shutdown":
            socket.send(encode_torch({"ok": True}))
            break

        hidden_states = request["hidden_states"]
        top_k_index = request["top_k_index"]
        top_k_weights = request["top_k_weights"]

        started = time.time()
        output = expert.forward(hidden_states, top_k_index, top_k_weights)
        latency_ms = (time.time() - started) * 1000

        response = {
            "ok": True,
            "request_id": request.get("request_id"),
            "output": output,
            "latency_ms": latency_ms,
            "rank": args.rank,
        }

        socket.send(encode_torch(response))

if __name__ == "__main__":
    main()