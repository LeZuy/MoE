import os
import torch

from dotenv import load_dotenv
from transformers import AutoTokenizer
from models.olmoe.modeling_olmoe import OlmoeForCausalLM
from models.olmoe.decentralized.simulator import DeOlmoeSimulator

EXMPL_PROMPT = "What is Bitcoin?"
MAX_LENGTH = 64

if __name__ == "__main__":

    load_dotenv()
    hf_token = os.environ["HF_TOKEN"]
    pretrained_path = os.environ["PRETRAINED_MODEL_PATH"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = OlmoeForCausalLM.from_pretrained(
        pretrained_model_name_or_path = pretrained_path,
        token = hf_token,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(pretrained_path)
    print(f"Tokenizer: {tokenizer}")

    # each client holds 64/8 = 8 experts
    simulator = DeOlmoeSimulator(model, num_clients=8, parallel=False)

    # Inference
    inputs = tokenizer(EXMPL_PROMPT, return_tensors="pt")
    outputs = simulator.generate(tokenizer, EXMPL_PROMPT, max_tokens=MAX_LENGTH)
    print(f"Prompt: {EXMPL_PROMPT}")
    print(f"Answers ({MAX_LENGTH} tokens): {outputs}")

    encoding = tokenizer(EXMPL_PROMPT, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**encoding, max_length=MAX_LENGTH)
    print(f"Original: {tokenizer.decode(out[0])}")

    # input_ids = inputs["input_ids"].to(device)
    # attention_mask = inputs["attention_mask"].to(device)
    # ref_logits = model(input_ids, attention_mask=attention_mask).logits
    # sim_logits = simulator.forward(input_ids, attention_mask)
    # diff = (ref_logits - sim_logits).abs()
    # print(f"Max diff:  {diff.max().item():.6f}")
    # print(f"Mean diff: {diff.mean().item():.6f}")

    # ref_token = ref_logits[:, -1, :].argmax(dim=-1)
    # sim_token = sim_logits[:, -1, :].argmax(dim=-1)
    # print(f"Ref next token: {tokenizer.decode(ref_token)!r}")
    # print(f"Sim next token: {tokenizer.decode(sim_token)!r}")
    # verify_layer_by_layer(model, simulator, tokenizer, "Bitcoin is", device)
    # verify_moe_output(model, simulator, tokenizer, "Bitcoin is", device, layer_idx=0)
    # verify_moe_internals(model, simulator, tokenizer, "Bitcoin is", device, layer_idx=0)
    # verify_attention_output(model, simulator, tokenizer, "Bitcoin is", device, layer_idx=0)
