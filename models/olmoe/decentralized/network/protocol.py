import io
import torch

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