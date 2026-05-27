import os
import torch

from dotenv import load_dotenv
from trl import SFTConfig, SFTTrainer
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, BitsAndBytesConfig
from models.olmoe.modeling_olmoe import OlmoeForCausalLM

if __name__ == "__main__":

    load_dotenv()
    hf_token = os.environ["HF_TOKEN"]
    pretrained_path = os.environ["PRETRAINED_MODEL_PATH"]

    # Load model in 4-bit and tokenizer
    model = OlmoeForCausalLM.from_pretrained(
        pretrained_path,    
        device_map="auto",
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        ),
        torch_dtype=torch.bfloat16,
    )

    tokenizer = AutoTokenizer.from_pretrained(pretrained_path, use_fast=False)   

    # Add LoRA adapters to model
    model = prepare_model_for_kbit_training(model)
    config = LoraConfig(
        r=64, 
        lora_alpha=16, 
        target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj'],
        lora_dropout=0.1, 
        bias="none", 
        modules_to_save = ["lm_head", "embed_tokens"],		# needed because we added new tokens to tokenizer/model
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, config)
    model.config.use_cache = False
    