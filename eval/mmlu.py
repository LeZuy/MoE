import os
import time
import argparse

import torch
import torch.nn.functional as F

from dotenv import load_dotenv

from models.olmoe.modeling_olmoe import OlmoeForCausalLM
from models.olmoe.decentralized.distributed import DistributedOlmoe
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from datasets import get_dataset_config_names, load_dataset, load_dataset_builder
from models.olmoe.decentralized.attngate import AttnGate
from network.router_agent import Router

DATASET_NAME = "cais/mmlu"
CHOICES = ["A", "B", "C", "D"]
CONFIG_PATH = "./network/configs/configs.yaml"

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
def load_distributed_model() -> tuple[DistributedOlmoe, PreTrainedTokenizerBase]:

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Loading model from: {pretrained_path}")

    model_kwargs = {
        "pretrained_model_name_or_path": pretrained_path,
    }
    if hf_token is not None:
        model_kwargs["token"] = hf_token
    if device.type == "cuda":
        model_kwargs["torch_dtype"] = torch.bfloat16

    base_model = OlmoeForCausalLM.from_pretrained(**model_kwargs).to(device)
    base_model.eval()

    attg_module = AttnGate(base_model).eval()
    router = Router(CONFIG_PATH)

    tokenizer_kwargs = {}
    if hf_token is not None:
        tokenizer_kwargs["token"] = hf_token
    tokenizer = AutoTokenizer.from_pretrained(pretrained_path, **tokenizer_kwargs)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"Set pad_token to eos_token: {tokenizer.pad_token}")

    dist_model = DistributedOlmoe(
        attg_module=attg_module,
        router=router,
        tokenizer=tokenizer,
        device=device,
    )
    return dist_model, tokenizer


def evaluate_subject(
    model: DistributedOlmoe | OlmoeForCausalLM,
    tokenizer: PreTrainedTokenizerBase,
    subject: str,
    num_examples: int,
) -> tuple[int, int]:
    ds_test = load_dataset(DATASET_NAME, subject)["test"]
    ds_dev = load_dataset(DATASET_NAME, subject)["dev"]

    limit = len(ds_test) if num_examples <= 0 else min(num_examples, len(ds_test))
    print(f"\nSubject: {subject}")
    print(f"Test examples: {len(ds_test)}; evaluating: {limit}")

    correct = 0
    started = time.time()

    for i in range(limit):
        example = ds_test[i]
        gold = CHOICES[int(example["answer"])]

        prompt = gen_prompt(
            ds_inst=example,
            dev_examples=ds_dev,
            subject=subject,
        )

        scores = {
            answer: score_choice(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                answer_letter=answer,
            )
            for answer in CHOICES
        }

        prediction = max(scores, key=scores.get)
        is_correct = prediction == gold
        correct += int(is_correct)

        print(
            f"[{i + 1}/{limit}] Prediction: {prediction}, Gold: {gold}, "
            f"Correct: {is_correct}, Scores: {scores}"
        )

    accuracy = correct / limit if limit > 0 else 0.0
    print(f"{subject}: accuracy={accuracy:.4f} ({correct}/{limit})")
    print(f"Subject took {time.time() - started:.2f}s")
    return correct, limit

@torch.inference_mode()
def score_choice(
    model: DistributedOlmoe | OlmoeForCausalLM,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    answer_letter: str,
) -> float:
    started = time.time()

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

    input_ids = torch.cat([prompt_ids, answer_ids], dim=1).to(model.device)
    attention_mask = torch.ones_like(input_ids, device=model.device)

    print(f"Prompt ID shape: {prompt_ids.shape}")
    print(f"Answer ID shape: {answer_ids.shape}")
    print(f"Input ID shape: {input_ids.shape}")
    print(f"Tokenization took {time.time() - started:.2f}s")

    tick = time.time()
    if isinstance(model, DistributedOlmoe):
        logits, _ = model.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=None,
            use_cache=False, 
        )
    elif isinstance(model, OlmoeForCausalLM):
        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=None,
            use_cache=False,
        ).logits
    print(f"Took {time.time() - tick:.2f}s")

    prompt_len = prompt_ids.shape[1]
    answer_len = answer_ids.shape[1]

    answer_logits = logits[:, prompt_len - 1 : prompt_len - 1 + answer_len, :].float()
    answer_targets = input_ids[:, prompt_len : prompt_len + answer_len]

    selected_log_probs = F.log_softmax(answer_logits, dim=-1).gather(
        dim=-1,
        index=answer_targets.unsqueeze(-1),
    ).squeeze(-1)

    score = selected_log_probs.sum().item()
    print(f"Finished answer {answer_letter}: score={score:.4f}")
    return score

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_path", type=str, default=None)

    parser.add_argument("--num_examples", type=int, default=1)

    args = parser.parse_args()

    load_dotenv()
    hf_token = os.environ.get("HF_TOKEN")
    pretrained_path = os.environ["PRETRAINED_MODEL_PATH"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_model = OlmoeForCausalLM.from_pretrained(
        pretrained_model_name_or_path=pretrained_path,
        token=hf_token,
        torch_dtype=torch.bfloat16,
    ).to(device)

    base_model.eval()

    dist_model, tokenizer = load_distributed_model()
    
    subjects = [
        s for s in get_dataset_config_names(DATASET_NAME)
        if s not in {"all", "auxiliary_train"}
    ]
    eval_subjects = [subjects[0]]

    print(f"Subjects: {eval_subjects}")

    all_correct = 0
    all_total = 0
    overall_started = time.time()

    for subject in eval_subjects:
        subject_correct, subject_total = evaluate_subject(
            model=base_model,
            tokenizer=tokenizer,
            subject=subject,
            num_examples=args.num_examples,
        )
        all_correct += subject_correct
        all_total += subject_total

    overall_accuracy = all_correct / all_total if all_total > 0 else 0.0
    print(
        f"Overall MMLU accuracy: {overall_accuracy:.4f} "
        # f"({all_correct}/{all_total})"
    )
    print(f"Total took {time.time() - overall_started:.2f}s")
    
    overall_started = time.time()
    for subject in eval_subjects:
        subject_correct, subject_total = evaluate_subject(
            model=dist_model,
            tokenizer=tokenizer,
            subject=subject,
            num_examples=args.num_examples,
        )
        all_correct += subject_correct
        all_total += subject_total

    overall_accuracy = all_correct / all_total if all_total > 0 else 0.0
    print(
        f"Overall MMLU accuracy: {overall_accuracy:.4f} "
        # f"({all_correct}/{all_total})"
    )
    print(f"Total took {time.time() - overall_started:.2f}s")

