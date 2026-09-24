"""
LoRA fine-tuning pipeline for Gemma-4 (google/gemma-4-E2B-it) on CPU.

Environment: torch 2.13+cpu, transformers 5.15.1, peft 0.20.0, datasets.

The text-only backbone (Gemma4ForCausalLM) is instantiated from the model's
text_config in bfloat16.  Pretrained weights are loaded from a local safetensors
file using the "model.language_model." prefix, then remapped to the CausalLM
state dict.

Each dataset is a JSON array of examples:

    {
      "instruction":  "...",
      "input_lines":  ["Room r0: ...", ...],
      "output_text":  "Human-readable reasoning in Russian.",
      "output":       {"commands": [...], "explanation": "..."}
    }

The assistant response is assembled in a two-part format:
"Решение: <output_text>\\n<JSON>".  Loss is masked with -100 on all tokens
up to the end of the user prompt, so the model learns only the assistant part.

Usage:
  python train_lora.py --dataset dataset_norm.json  --output ./expert_lora_ready  --adapter-name expert
  python train_lora.py --dataset dataset_planer.json --output ./planner_lora_ready --adapter-name planner
  python train_lora.py --dataset dataset_pluginist.json --output ./pluginist_lora_ready --adapter-name pluginist
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from safetensors.torch import load_file
from transformers import (
    AutoConfig,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from transformers.models.gemma4.modeling_gemma4 import Gemma4ForCausalLM

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

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
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

MAX_LENGTH = 1024
LORA_RANK = 16
LORA_ALPHA = 32
LEARNING_RATE = 2e-4
EPOCHS = 5
BATCH_SIZE = 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LoRA fine-tuning for Gemma-4 on CPU.")
    parser.add_argument("--dataset", default="dataset.json")
    parser.add_argument("--output", default="./lora_output")
    parser.add_argument("--adapter-name", default="planner")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate_dataset(path: str) -> None:
    if os.path.isfile(path):
        return
    print("=" * 70)
    print("  ERROR: dataset file not found.")
    print(f"  Expected: {os.path.abspath(path)}")
    print("=" * 70)
    print("  Usage:")
    print("    python train_lora.py --dataset dataset_norm.json --output ./expert_lora_ready")
    print("    python train_lora.py --dataset dataset_planer.json --output ./planner_lora_ready")
    print("=" * 70)
    sys.exit(1)


# ---------------------------------------------------------------------------
# Model setup
# ---------------------------------------------------------------------------

def _build_model(device: torch.device) -> Gemma4ForCausalLM:
    """Instantiate Gemma4ForCausalLM in bfloat16 and load local weights."""

    config = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
    text_config = getattr(config, "text_config", config)

    model = Gemma4ForCausalLM(text_config).to(torch.bfloat16)

    if not os.path.isfile(LOCAL_WEIGHTS):
        print(f"WARNING: {LOCAL_WEIGHTS} not found — model is randomly initialized.")
        return model.to(device)

    raw = load_file(LOCAL_WEIGHTS, device="cpu")
    prefix = "model.language_model."
    state_dict = {
        "model." + k[len(prefix):]: v
        for k, v in raw.items()
        if k.startswith(prefix)
    }
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"WARNING: {len(missing)} missing keys (e.g. {missing[:3]})")
    if unexpected:
        print(f"WARNING: {len(unexpected)} unexpected keys (e.g. {unexpected[:3]})")
    return model.to(device)


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------

def tokenize_function(examples: dict) -> dict:
    """Tokenize batched examples with a system/user/assistant chat template.

    Labels are masked with -100 for all tokens up to the assistant response,
    so the model only receives loss on the generated part.
    """
    input_lines = examples.get("input_lines") or []
    instructions = examples.get("instruction") or []
    output_texts = examples.get("output_text") or []
    outputs = examples.get("output") or []
    n = len(examples["output"])

    prompts: list[str] = []
    responses: list[str] = []

    for i in range(n):
        instr = (instructions[i] if i < len(instructions) else "") or ""
        lines = input_lines[i] if i < len(input_lines) else ""
        if isinstance(lines, list):
            lines = "\n".join(str(x) for x in lines)
        else:
            lines = str(lines or "")
        user_text = instr + ("\n" + lines if lines else "")

        prompts.append(
            tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_text},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
        )

        parts: list[str] = []
        out_text = (output_texts[i] if i < len(output_texts) else "") or ""
        out_text = out_text.strip() if isinstance(out_text, str) else str(out_text or "").strip()
        if out_text:
            parts.append(f"Решение: {out_text}")
        out = outputs[i] if i < len(outputs) else None
        if out is not None:
            parts.append(json.dumps(out, ensure_ascii=False))
        responses.append("\n".join(parts))

    enc_prompt = tokenizer(prompts, padding=False, truncation=True, max_length=MAX_LENGTH)
    enc_full = tokenizer(
        [p + r for p, r in zip(prompts, responses)],
        padding=False, truncation=True, max_length=MAX_LENGTH,
    )

    labels = []
    for i, ids in enumerate(enc_full["input_ids"]):
        prompt_len = len(enc_prompt["input_ids"][i])
        lbl = ids[:]
        for j in range(min(prompt_len, len(lbl))):
            lbl[j] = -100
        labels.append(lbl)

    return {
        "input_ids": enc_full["input_ids"],
        "attention_mask": enc_full["attention_mask"],
        "labels": labels,
    }


# ---------------------------------------------------------------------------
# Data collator
# ---------------------------------------------------------------------------

def data_collator(batch: list[dict]) -> dict[str, torch.Tensor]:
    """Pad tensors to the maximum length within the batch."""
    max_len = max(len(b["input_ids"]) for b in batch)
    pad_id = tokenizer.pad_token_id or 0

    result: dict[str, list[list[int]]] = {"input_ids": [], "attention_mask": [], "labels": []}
    for b in batch:
        n = max_len - len(b["input_ids"])
        result["input_ids"].append(b["input_ids"] + [pad_id] * n)
        result["attention_mask"].append(b["attention_mask"] + [0] * n)
        result["labels"].append(b["labels"] + [-100] * n)

    return {k: torch.tensor(v) for k, v in result.items()}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    _validate_dataset(args.dataset)

    print(f"Dataset: {args.dataset}")
    print(f"Output:  {args.output}")

    device = torch.device("cpu")
    print(f"Device:  {device}")

    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = _build_model(device)

    peft_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        target_modules=TARGET_MODULES,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    dataset = load_dataset("json", data_files=args.dataset, split="train")
    tokenized = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
    )

    training_args = TrainingArguments(
        output_dir=args.output,
        per_device_train_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        logging_steps=10,
        save_strategy="no",
        use_cpu=True,
        fp16=False,
        bf16=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=data_collator,
    )

    trainer.train()

    model.config._name_or_path = MODEL_ID
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    print(f"Adapter '{args.adapter_name}' saved to {args.output}")


if __name__ == "__main__":
    main()
