import os
import argparse

import torch
import torch.nn.functional as F

from dotenv import load_dotenv

from models.olmoe.modeling_olmoe import OlmoeForCausalLM
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from datasets import get_dataset_config_names, load_dataset, load_dataset_builder

PRETRAINED_PATH = "/home/duy.le004/.cache/huggingface/hub/models--allenai--OLMoE-1B-7B-0924/snapshots/6d84c48581ece794365f2b8e9cfb043c68ade9c5"
DATASET_NAME = "cais/mmlu"
CHOICES = ["A", "B", "C", "D"]

def format_subject(subject: str) -> str:
    l = subject.split("_")
    s = ""
    for entry in l:
        s += " " + entry
    return s

def format_example(ds_inst: dict, include_answer: bool =False) -> str:
    prompt = ds_inst["question"]
    
    for char, choice in zip(CHOICES, ds_inst["choices"]):
        prompt += f"\n{char}. {choice}"

    prompt += "\nAnswer:"
    if include_answer:
        prompt += f" {CHOICES[int(ds_inst['answer'])]}"

    return prompt

def gen_prompt(ds_inst: dict, dev_examples : list, subject: str) -> str:
    
    prompt = f"The following are multiple choice questions (with answers) about {format_subject(subject)}.\n\n"
    for dev_example in dev_examples:
        prompt += format_example(dev_example, include_answer=True)
        prompt += "\n\n"

    prompt += format_example(ds_inst, include_answer=False)

    return prompt

@torch.inference_mode()
def score_choice(
    model: OlmoeForCausalLM,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    answer_letter: str,
    device: torch.device,
) -> float:
    
    prompt_ids = tokenizer(
        prompt,
        add_special_tokens=False,
        return_tensors="pt",
    )["input_ids"]

    answer_ids = tokenizer(
        f" {answer_letter}",
        add_special_tokens=False,
        return_tensors="pt",
    )["input_ids"]

    input_ids = torch.cat([prompt_ids, answer_ids], dim=1).to(device)

    outputs = model(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        use_cache=False,
    )

    prompt_len = prompt_ids.shape[1]
    answer_len = answer_ids.shape[1]

    answer_logits = outputs.logits[:, prompt_len - 1 : prompt_len - 1 + answer_len, :].float()

    answer_targets = input_ids[:, prompt_len : prompt_len + answer_len]

    selected_log_probs = F.log_softmax(answer_logits, dim=-1).gather(
        dim=-1,
        index=answer_targets.unsqueeze(-1),
    ).squeeze(-1)

    return selected_log_probs.sum().item()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--pretrained_path", type=str, default = PRETRAINED_PATH)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Loading model from: {args.pretrained_path}")

    load_dotenv()
    hf_token = os.environ["HF_TOKEN"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = OlmoeForCausalLM.from_pretrained(
        pretrained_model_name_or_path=args.pretrained_path,
        token=hf_token
        ).to(device)

    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_path)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"Set {tokenizer.pad_token}")
        
    print(f"Number of layers: {model.config.num_hidden_layers}")
    print(f"Number of experts: {model.config.num_experts}")
    print(f"Experts selected per token: {model.config.num_experts_per_tok}")
    
    subjects = [
        s for s in get_dataset_config_names(DATASET_NAME)
        if s not in {"all", "auxiliary_train"}
    ]
    
    print(f"Subjects: {subjects}")

    all_correct = 0
    all_total = 0

    for subject in subjects:
        ds_test = load_dataset(DATASET_NAME, subject)["test"]
        ds_dev = load_dataset(DATASET_NAME, subject)["dev"]
        print(f"\nSubject: {subject} \nDataset features: {ds_test}")
        subject_correct = 0
        for example in ds_test:
            question = example["question"]
            choices = example["choices"]
            gold = CHOICES[int(example["answer"])]

            prompt = gen_prompt(ds_inst = example, 
                                dev_examples = ds_dev,
                                subject = example["subject"])
            # print(f"\nPrompt: {prompt}")

            scores = {
                answer: score_choice(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=prompt,
                    answer_letter=answer,
                    device=device,
                )
                for answer in CHOICES
            }

            prediction = max(scores, key=scores.get)
            # print(f"Prediction: {prediction}, Gold: {gold}")
            subject_correct += int(prediction == gold)
        subject_accuracy = subject_correct / len(ds_test)
        print(f"{subject}: {subject_accuracy:.4f}")

        all_correct += subject_correct
        all_total += len(ds_test)

    print(f"Overall MMLU accuracy: {all_correct / all_total:.4f}")
