from transformers import OlmoeForCausalLM, AutoTokenizer

model = OlmoeForCausalLM.from_pretrained("allenai/OLMoE-1B-7B-0924")
tokenizer = AutoTokenizer.from_pretrained("allenai/OLMoE-1B-7B-0924")
out = model.generate(**tokenizer("Bitcoin is", return_tensors="pt"), max_length=64)