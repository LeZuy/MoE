import os
import re
import time
import torch
import argparse

from dotenv import load_dotenv
from datasets import load_dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from models.olmoe.modeling_olmoe import OlmoeForCausalLM
from models.olmoe.decentralized.attngate import AttnGate
from models.olmoe.decentralized.distributed import DistributedOlmoe
from network.router_agent import Router


DATASET_NAME = "openai/gsm8k"
DATASET_CONFIG = "main"
CONFIG_PATH = "./network/configs/configs.yaml"


def format_gsm8k_example(example: dict, include_answer: bool = False) -> str:
    prompt = "Problem: " + example["question"].strip() + "\nSolution:"

    if include_answer:
        prompt += " " + example["answer"].strip()

    return prompt

def gen_prompt(example: dict, fewshot_examples: list[dict] | None = None) -> str:
    prompt = ""

    if fewshot_examples:
        for ex in fewshot_examples:
            prompt += f"Question: {ex['question'].strip()}\n"
            prompt += f"Answer: {ex['answer'].strip()}\n\n"

    prompt += f"Question: {example['question'].strip()}\n"
    prompt += "Answer:"

    return prompt


def extract_number(text: str) -> str | None:
    text = text.replace(",", "")

    match = re.search(r"####\s*(-?\d+(?:\.\d+)?)", text)
    if match:
        return match.group(1)

    nums = re.findall(r"-?\d+(?:\.\d+)?", text)
    if not nums:
        return None

    return nums[-1]


def normalize_number(x: str | None) -> str | None:
    if x is None:
        return None

    try:
        value = float(x)
        if value.is_integer():
            return str(int(value))
        return str(value)
    except ValueError:
        return x.strip()


@torch.inference_mode()
def generate_gsm8k_answer(
    model: DistributedOlmoe | OlmoeForCausalLM,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    max_new_tokens: int = 256,
) -> str:
    is_distributed = isinstance(model, DistributedOlmoe)

    if is_distributed:
        device = model.device
        backend_name = "DistributedOlmoe"
    else:
        device = next(model.parameters()).device
        backend_name = "OlmoeForCausalLM"

    input_ids = tokenizer(
        prompt,
        add_special_tokens=False,
        return_tensors="pt",
    )["input_ids"].to(device)

    eos_token_id = tokenizer.eos_token_id

    tick = time.time()

    if is_distributed:
        logits, state = model.prefill(input_ids)

    else:
        attention_mask = torch.ones_like(input_ids, device=device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
        )

        logits = outputs.logits
        past_key_values = outputs.past_key_values

    print(f"[{backend_name}] Prefill took {time.time() - tick:.2f}s")

    generated_tokens: list[torch.Tensor] = []

    next_token = torch.argmax(
        logits[:, -1, :].float(),
        dim=-1,
        keepdim=True,
    )

    for step in range(max_new_tokens):
        generated_tokens.append(next_token)

        if eos_token_id is not None and int(next_token.item()) == eos_token_id:
            print(f"[{backend_name}] Stop: EOS at step {step}")
            break

        generated_ids = torch.cat(generated_tokens, dim=1)

        generated_text = tokenizer.decode(
            generated_ids[0],
            skip_special_tokens=True,
        )

        if re.search(r"####\s*-?\d+", generated_text):
            print(f"[{backend_name}] Stop: found answer at step {step}")
            break

        # if "\nQuestion:" in generated_text or "\nProblem:" in generated_text:
        #     print(f"[{backend_name}] Stop: model started a new sample at step {step}")
        #     break

        tick = time.time()

        if is_distributed:
            logits, state = model.decode_step(next_token, state)

        else:
            attention_mask = torch.cat(
                [
                    attention_mask,
                    torch.ones(
                        attention_mask.shape[0],
                        1,
                        device=device,
                        dtype=attention_mask.dtype,
                    ),
                ],
                dim=1,
            )

            outputs = model(
                input_ids=next_token.to(device),
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )

            logits = outputs.logits
            past_key_values = outputs.past_key_values

        print(f"[{backend_name}] Decode step {step} took {time.time() - tick:.2f}s")

        next_token = torch.argmax(
            logits[:, -1, :].float(),
            dim=-1,
            keepdim=True,
        )

    if len(generated_tokens) == 0:
        return ""

    generated_ids = torch.cat(generated_tokens, dim=1)

    return tokenizer.decode(
        generated_ids[0],
        skip_special_tokens=True,
    )


def evaluate_one_example(
    model: DistributedOlmoe,
    tokenizer: PreTrainedTokenizerBase,
    example: dict,
    fewshot_examples: list[dict] | None = None,
    max_new_tokens: int = 256,
) -> dict:
    prompt = gen_prompt(example, fewshot_examples=fewshot_examples)

    generated = generate_gsm8k_answer(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
    )

    pred = normalize_number(extract_number(generated))
    gold = normalize_number(extract_number(example["answer"]))

    return {
        "question": example["question"],
        "generated": generated,
        "prediction": pred,
        "gold": gold,
        "correct": int(pred == gold),
    }


if __name__ == "__main__":
    load_dotenv()

    hf_token = os.environ.get("HF_TOKEN")
    pretrained_path = os.environ["PRETRAINED_MODEL_PATH"]

    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_path", type=str, default=pretrained_path)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--num_examples", type=int, default=1)
    parser.add_argument("--num_fewshot", type=int, default=8)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")
    print(f"Loading model from: {args.pretrained_path}")

    base_model = OlmoeForCausalLM.from_pretrained(
        pretrained_model_name_or_path=args.pretrained_path,
        token=hf_token,
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
    ).to(device)

    base_model.eval()
    base_model.config.use_cache = True

    attg_module = AttnGate(base_model).eval()
    router = Router(CONFIG_PATH)

    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_path)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"Set pad_token = {tokenizer.pad_token}")
    print(tokenizer.chat_template) 

    dist_model = DistributedOlmoe(
        attg_module=attg_module,
        router=router,
        tokenizer=tokenizer,
        device=device,
    )

    ds = load_dataset(DATASET_NAME, DATASET_CONFIG)

    ds_train = ds["train"]
    ds_test = ds["test"]

    fewshot_examples = None
    if args.num_fewshot > 0:
        fewshot_examples = [ds_train[i] for i in range(args.num_fewshot)]

    correct = 0
    total = 0

    start_all = time.time()

    for i in range(args.num_examples):
        start = time.time()

        example = ds_test[i]

        result = evaluate_one_example(
            model=dist_model,
            tokenizer=tokenizer,
            example=example,
            fewshot_examples=fewshot_examples,
            max_new_tokens=args.max_new_tokens,
        )

        correct += result["correct"]
        total += 1

        print(f"Question: {result['question']}")
        print(f"Generated:\n{result['generated']}")
        print(f"Prediction: {result['prediction']}")
        print(f"Gold: {result['gold']}")
        print(f"Correct: {result['correct']}")
        print(f"Accuracy: {correct / total:.4f}")
        print(f"Took {time.time() - start:.2f}s")

        result = evaluate_one_example(
            model=base_model,
            tokenizer=tokenizer,
            example=example,
            fewshot_examples=fewshot_examples,
            max_new_tokens=args.max_new_tokens,
        )

        correct += result["correct"]
        total += 1

        print(f"Question: {result['question']}")
        print(f"Generated:\n{result['generated']}")
        print(f"Prediction: {result['prediction']}")
        print(f"Gold: {result['gold']}")
        print(f"Correct: {result['correct']}")
        print(f"Accuracy: {correct / total:.4f}")
        print(f"Took {time.time() - start:.2f}s")

    print(f"\nFinal accuracy: {correct / total:.4f}")
    print(f"Total time: {time.time() - start_all:.2f}s")