import os
import torch
import time
from dotenv import load_dotenv
from transformers import AutoTokenizer
from models.olmoe.modeling_olmoe import OlmoeForCausalLM
from models.olmoe.decentralized.distributed import DistributedOlmoe
from models.olmoe.decentralized.attngate import AttnGate
from network.router_agent import Router

CONFIG_PATH = "/home/duy.le004/phd/MoE/network/configs/configs.yaml"
BATCH_SIZE = [1, 4, 8, 16]
MAX_LENGTH = 64
EXMPL_PROMPT = "What is Bitcoin?"

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
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    attg_module = AttnGate(model).eval()

    # each client holds 64/8 = 8 experts
    router = Router(CONFIG_PATH)

    dist_model = DistributedOlmoe(attg_module, router, tokenizer, device)
    # Inference
    prompts = [
        "What is Bitcoin?",
        "Explain machine learning.",
        "What is a GPU?",
        "Define blockchain.",
    ]

    encoding = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
    start = time.time()
    
    out = dist_model.generate(**encoding, max_new_tokens=MAX_LENGTH, do_sample=False)
    
    print(f"Prompt: {prompts}")
    print(f"Decentralized ({MAX_LENGTH} tokens):")
    decoded = tokenizer.batch_decode(out, skip_special_tokens=True)
    for i, text in enumerate(decoded):
        print(f"[{i}] {text}")
    print(f"Took {time.time() - start: .2f} seconds")

    start = time.time()
    with torch.no_grad():
        out = model.generate(**encoding, max_new_tokens=MAX_LENGTH, do_sample=False)
    print(f"Original model ({MAX_LENGTH} tokens):")
    decoded = tokenizer.batch_decode(out, skip_special_tokens=True)
    for i, text in enumerate(decoded):
        print(f"[{i}] {text}")
    print(f"Took {time.time() - start: .2f} seconds")