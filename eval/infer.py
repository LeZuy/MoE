import torch
import argparse

from transformers import AutoTokenizer
from .. models.olmoe.modeling_olmoe import OlmoeForCausalLM

PRETRAINED_PATH = "/home/duy.le004/.cache/huggingface/hub/models--allenai--OLMoE-1B-7B-0924/snapshots/6d84c48581ece794365f2b8e9cfb043c68ade9c5"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--pretrained_path", type=str, default = PRETRAINED_PATH)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Loading model from: {args.pretrained_path}")

    model = OlmoeForCausalLM.from_pretrained(args.pretrained_path).to(device)

    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_path)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"Set {tokenizer.pad_token}")
        
    print(f"Number of layers: {model.config.num_hidden_layers}")
    print(f"Number of experts: {model.config.num_experts}")
    print(f"Experts selected per token: {model.config.num_experts_per_tok}")
