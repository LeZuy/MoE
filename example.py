import os
import torch
import time
from dotenv import load_dotenv
from transformers import AutoTokenizer
from models.olmoe.modeling_olmoe import OlmoeForCausalLM
from models.olmoe.decentralized.distributed import DistributedOlmoe
from models.olmoe.decentralized.attngate import AttnGate
from network.router_agent import Router

EXMPL_PROMPT = "What is Bitcoin?"
MAX_LENGTH = 8
CONFIG_PATH = "/home/duy.le004/phd/MoE/network/configs/configs.yaml"

if __name__ == "__main__":

    load_dotenv()
    hf_token = os.environ["HF_TOKEN"]
    pretrained_path = os.environ["PRETRAINED_MODEL_PATH"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = OlmoeForCausalLM.from_pretrained(
        pretrained_model_name_or_path = pretrained_path,
        token = hf_token,
        torch_dtype=torch.bfloat16,
    ).to(device)
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(pretrained_path)
    # print(f"Tokenizer: {tokenizer}")

    attg_module = AttnGate(model).eval()

    # each client holds 64/8 = 8 experts
    router = Router(CONFIG_PATH)

    dist_model = DistributedOlmoe(attg_module, router, tokenizer, device)
    # Inference
    encoding = tokenizer(EXMPL_PROMPT, return_tensors="pt").to(device)
    start = time.time()
    out = dist_model.generate(**encoding, max_new_tokens=MAX_LENGTH)
    print(f"Prompt: {EXMPL_PROMPT}")
    print(f"Decentralized ({MAX_LENGTH} tokens): {tokenizer.decode(out[0])}")
    print(f"Took {time.time() - start: .2f} seconds")

    start = time.time()
    with torch.no_grad():
        out = model.generate(**encoding, max_new_tokens=MAX_LENGTH)
    print(f"Original model ({MAX_LENGTH} tokens): {tokenizer.decode(out[0])}")
    print(f"Took {time.time() - start: .2f} seconds")