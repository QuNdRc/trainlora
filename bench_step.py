"""
bench_step.py — measure single-step LoRA training time on real hardware.

Instantiates the model in bfloat16, attaches LoRA, takes a single sample,
runs one or more optimizer steps, and prints seconds per step.

Usage:
  python bench_step.py --dataset dataset_norm.json --steps 1
"""

from __future__ import annotations

import argparse
import json
import time

import torch
from peft import LoraConfig, get_peft_model
from safetensors.torch import load_file
from transformers import AutoConfig, AutoTokenizer
from transformers.models.gemma4.modeling_gemma4 import Gemma4ForCausalLM

MODEL_ID = "google/gemma-4-E2B-it"
LOCAL_WEIGHTS = "C:/ai/weight/model.safetensors"

SYSTEM_PROMPT = (
    "You are an apartment layout planner. Analyze the layout and propose fix commands. "
    "Answer in exactly two parts: first a short human-readable line starting with "
    "'Решение:' in Russian, then, on the next line(s), a single valid JSON object "
    'like {"commands":[...],"explanation":"..."}. You may only use the commands '
    "'move_target', 'swap' or 'add_door'. Return at most 3 commands. "
    "The JSON block must be valid JSON."
)

TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="dataset_norm.json")
    p.add_argument("--steps", type=int, default=1)
    p.add_argument("--max-length", type=int, default=256)
    return p.parse_args()


def _load_model() -> Gemma4ForCausalLM:
    config = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
    text_config = getattr(config, "text_config", config)
    model = Gemma4ForCausalLM(text_config).to(torch.bfloat16)

    raw = load_file(LOCAL_WEIGHTS, device="cpu")
    prefix = "model.language_model."
    state_dict = {
        "model." + k[len(prefix):]: v for k, v in raw.items() if k.startswith(prefix)
    }
    model.load_state_dict(state_dict, strict=False)
    return model


def main() -> None:
    args = parse_args()
    device = torch.device("cpu")

    print(f"Model:   {MODEL_ID}")
    print(f"Dataset: {args.dataset}")
    print(f"Steps:   {args.steps}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    with open(args.dataset, encoding="utf-8") as f:
        data = json.load(f)
    print(f"Samples: {len(data)}")

    t0 = time.time()
    model = _load_model()
    print(f"Model loaded in {time.time() - t0:.1f}s")

    peft_config = LoraConfig(
        r=16, lora_alpha=32, target_modules=TARGET_MODULES,
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    model.to(device)

    # Build a minimal sample for benchmarking
    prompt = tokenizer.apply_chat_template(
        [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": ""}],
        tokenize=False, add_generation_prompt=True,
    )
    response = 'Решение: test response.\n{"commands":[]}'
    full_text = prompt + response
    encoded = tokenizer(full_text, truncation=True, max_length=args.max_length, return_tensors="pt")
    prompt_enc = tokenizer(prompt, truncation=True, max_length=args.max_length, return_tensors="pt")

    labels = encoded["input_ids"].clone()
    labels[:, :prompt_enc["input_ids"].shape[1]] = -100
    sample = {k: v.to(device) for k, v in encoded.items()}
    sample["labels"] = labels.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)

    times: list[float] = []
    for step in range(args.steps):
        model.train()
        step_start = time.time()
        out = model(**sample)
        loss = out.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        elapsed = time.time() - step_start
        times.append(elapsed)
        print(f"  Step {step + 1}: {elapsed:.2f}s  loss={loss.item():.4f}")

    avg = sum(times) / len(times)
    print(f"Average: {avg:.2f}s/step")
    examples = len(data)
    steps_per_epoch = examples // 1
    est = avg * steps_per_epoch * 5 / 3600
    print(f"Estimated for {examples} examples × 5 epochs: ~{est:.1f}h")


if __name__ == "__main__":
    main()