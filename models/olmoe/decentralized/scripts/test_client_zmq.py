import uuid
import zmq
import torch

from ..network.protocol import encode_torch, decode_torch


def main():
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect("tcp://127.0.0.1:5000")

    hidden_states = torch.randn(4, 2048)
    top_k_index = torch.zeros(4, 8, dtype=torch.long)
    top_k_weights = torch.ones(4, 8, dtype=torch.float32)

    request = {
        "type": "forward",
        "request_id": str(uuid.uuid4()),
        "layer_idx": 0,
        "hidden_states": hidden_states,
        "top_k_index": top_k_index,
        "top_k_weights": top_k_weights,
    }

    socket.send(encode_torch(request))
    response = decode_torch(socket.recv())

    print("ok:", response["ok"])
    print("rank:", response["rank"])
    print("latency_ms:", response["latency_ms"])
    print("output shape:", response["output"].shape)
    print("max diff:", (response["output"] - hidden_states * 2).abs().max().item())


if __name__ == "__main__":
    main()