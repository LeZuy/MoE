from models.olmoe.modeling_olmoe import OlmoeForCausalLM
from transformers import AutoTokenizer

PRETRAINED_PATH = "/home/duy.le004/.cache/huggingface/hub/models--allenai--OLMoE-1B-7B-0924/snapshots/6d84c48581ece794365f2b8e9cfb043c68ade9c5"

model = OlmoeForCausalLM.from_pretrained(PRETRAINED_PATH)
print(f"Model: {model}")
tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_PATH)
print(f"Tokenizer: {tokenizer}")
out = model.generate(**tokenizer("Bitcoin is", return_tensors="pt"), max_length=64)
print(f"Output: {out}")
print(f"Detokenized: {tokenizer.decode(out[0])}")